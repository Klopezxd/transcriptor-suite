"""Límites de autenticación sin esperas reales ni credenciales del despliegue."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.auth import (
    LoginAttemptLimiter,
    LoginRateLimitError,
    _clean_ip,
    extract_client_identifier,
)


class LoginLimiterTests(unittest.TestCase):
    def test_blocks_after_five_failures_without_checking_password(self):
        limiter = LoginAttemptLimiter()
        verifier = Mock(return_value=False)
        for _ in range(5):
            self.assertFalse(limiter.authenticate("wrong", "client", verifier))
        with self.assertRaises(LoginRateLimitError):
            limiter.authenticate("correct", "client", verifier)
        self.assertEqual(verifier.call_count, 5)

    def test_window_expires(self):
        clock = Mock(return_value=0)
        limiter = LoginAttemptLimiter(max_attempts=1, clock=clock)
        limiter.authenticate("wrong", "client", lambda _: False)
        clock.return_value = 60
        self.assertTrue(limiter.authenticate("correct", "client", lambda _: True))

    def test_success_resets_failures(self):
        limiter = LoginAttemptLimiter(max_attempts=2)
        limiter.authenticate("wrong", "client", lambda _: False)
        self.assertTrue(limiter.authenticate("correct", "client", lambda _: True))
        self.assertFalse(limiter.authenticate("wrong", "client", lambda _: False))
        self.assertTrue(limiter.authenticate("correct", "client", lambda _: True))

    def test_clients_are_independent(self):
        limiter = LoginAttemptLimiter(max_attempts=1)
        limiter.authenticate("wrong", "first", lambda _: False)
        self.assertTrue(limiter.authenticate("correct", "second", lambda _: True))

    def test_memory_is_bounded(self):
        limiter = LoginAttemptLimiter(max_clients=2)
        for client in ["first", "second", "third"]:
            limiter.authenticate("wrong", client, lambda _: False)
        self.assertEqual(len(limiter._failures), 2)

    def test_global_limiter_blocks_client_rotation_and_lru_eviction(self):
        clock = Mock(return_value=0.0)
        limiter = LoginAttemptLimiter(
            max_attempts=5,
            window_seconds=60,
            max_clients=3,
            max_global_attempts=4,
            global_window_seconds=60,
            clock=clock,
        )
        verifier = Mock(return_value=False)
        for idx in range(4):
            self.assertFalse(limiter.authenticate("wrong", f"spoofed_{idx}", verifier))

        # Al alcanzar el límite global, la rotación de clientes queda bloqueada antes de verificar
        # y sin seguir evictando la tabla LRU.
        with self.assertRaises(LoginRateLimitError):
            limiter.authenticate("correct", "spoofed_new", verifier)
        self.assertEqual(verifier.call_count, 4)
        self.assertEqual(len(limiter._failures), 3)

        # Al expirar la ventana global vuelve a permitir autenticación legítima
        clock.return_value = 60.0
        self.assertTrue(limiter.authenticate("correct", "admin", lambda _: True))

    def test_concurrent_global_limit_cannot_be_bypassed_by_client_rotation(self):
        limiter = LoginAttemptLimiter(max_attempts=5, max_clients=100, max_global_attempts=6)
        verifier = Mock(return_value=False)

        def attempt(idx):
            try:
                return limiter.authenticate("wrong", f"client_{idx}", verifier)
            except LoginRateLimitError:
                return "blocked"

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(attempt, range(24)))
        self.assertEqual(results.count("blocked"), 18)
        self.assertEqual(verifier.call_count, 6)

    def test_concurrent_attempts_do_not_bypass_limit(self):
        limiter = LoginAttemptLimiter()
        verifier = Mock(return_value=False)

        def attempt(_):
            try:
                return limiter.authenticate("wrong", "client", verifier)
            except LoginRateLimitError:
                return "blocked"

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(attempt, range(20)))
        self.assertEqual(results.count("blocked"), 15)
        self.assertEqual(verifier.call_count, 5)

    def test_invalid_limits_are_rejected(self):
        for options in [
            {"max_attempts": 0},
            {"window_seconds": 0},
            {"max_clients": 0},
            {"max_global_attempts": 0},
            {"global_window_seconds": 0},
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                LoginAttemptLimiter(**options)


class ClientIdentifierTests(unittest.TestCase):
    def test_none_request(self):
        self.assertEqual(extract_client_identifier(None), "unknown")

    def test_clean_ip_validates_with_ipaddress(self):
        self.assertEqual(_clean_ip("198.51.100.1"), "198.51.100.1")
        self.assertEqual(_clean_ip("198.51.100.55:44321"), "198.51.100.55")
        self.assertEqual(_clean_ip("[2001:0db8::0001]:8080"), "2001:db8::1")
        self.assertEqual(_clean_ip("::ffff:127.0.0.1"), "127.0.0.1")
        for invalid in [
            "",
            None,
            "localhost",
            "not-an-ip",
            "999.999.999.999",
            "198.51.100.1:99999",
            "198.51.100.1:abc",
            "[2001:db8::1",
            "[2001:db8::1]:abc",
        ]:
            with self.subTest(invalid=invalid):
                self.assertEqual(_clean_ip(invalid), "")

    def test_cf_connecting_ip_header_from_trusted_proxy(self):
        request = SimpleNamespace(
            headers={"cf-connecting-ip": "198.51.100.1"},
            client=SimpleNamespace(host="127.0.0.1"),
        )
        self.assertEqual(extract_client_identifier(request), "198.51.100.1")

    def test_x_forwarded_for_rightmost_untrusted_hop(self):
        request = SimpleNamespace(
            headers={"x-forwarded-for": "1.1.1.1, 203.0.113.195, 10.0.0.1"},
            client=SimpleNamespace(host="127.0.0.1"),
        )
        self.assertEqual(extract_client_identifier(request), "203.0.113.195")

    def test_x_forwarded_for_multi_proxy_with_configured_trusted_cidr(self):
        request = SimpleNamespace(
            headers={"x-forwarded-for": "9.9.9.9, 198.51.100.77, 172.64.0.10, 10.0.0.2"},
            client=SimpleNamespace(host="10.0.0.1"),
        )
        with patch.dict("os.environ", {"TRUSTED_PROXY_IPS": "172.64.0.0/16"}):
            self.assertEqual(extract_client_identifier(request), "198.51.100.77")

    def test_spoofed_headers_ignored_from_public_direct_connection(self):
        request = SimpleNamespace(
            headers={
                "cf-connecting-ip": "1.2.3.4",
                "x-forwarded-for": "5.6.7.8",
                "x-real-ip": "9.9.9.9",
            },
            session_hash="attacker_rotated_session",
            client=SimpleNamespace(host="198.51.100.42"),
        )
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(extract_client_identifier(request), "198.51.100.42")

    def test_public_proxy_trusted_only_when_in_trusted_proxy_ips(self):
        request = SimpleNamespace(
            headers={"x-forwarded-for": "1.2.3.4, 203.0.113.88"},
            client=SimpleNamespace(host="198.51.100.42"),
        )
        with patch.dict("os.environ", {"TRUSTED_PROXY_IPS": "198.51.100.42"}):
            self.assertEqual(extract_client_identifier(request), "203.0.113.88")

    def test_absent_client_ignores_headers_without_explicit_proxy_config(self):
        request = SimpleNamespace(headers={"x-forwarded-for": "203.0.113.195"})
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(extract_client_identifier(request), "local_client")

        with patch.dict("os.environ", {"TRUST_PROXY_HEADERS": "true"}, clear=True):
            self.assertEqual(extract_client_identifier(request), "203.0.113.195")

    def test_x_real_ip_header(self):
        request = SimpleNamespace(
            headers={"x-real-ip": "192.0.2.1"},
            client=SimpleNamespace(host="127.0.0.1"),
        )
        self.assertEqual(extract_client_identifier(request), "192.0.2.1")

    def test_session_hash_for_localhost(self):
        request = SimpleNamespace(
            headers={},
            session_hash="tab_unique_456",
            client=SimpleNamespace(host="127.0.0.1"),
        )
        self.assertEqual(extract_client_identifier(request), "session_tab_unique_456")

    def test_direct_client_host(self):
        request = SimpleNamespace(headers={}, client=SimpleNamespace(host="192.168.1.50"))
        self.assertEqual(extract_client_identifier(request), "192.168.1.50")

    def test_local_fallback(self):
        request = SimpleNamespace(headers={}, client=SimpleNamespace(host="127.0.0.1"))
        self.assertEqual(extract_client_identifier(request), "local_client")

    def test_ip_with_port_is_sanitized(self):
        request = SimpleNamespace(
            headers={"x-forwarded-for": "198.51.100.55:44321, 10.0.0.1"},
            client=SimpleNamespace(host="127.0.0.1"),
        )
        self.assertEqual(extract_client_identifier(request), "198.51.100.55")

    def test_ipv6_with_bracket_and_port(self):
        request = SimpleNamespace(
            headers={"cf-connecting-ip": "[2001:db8::1]:8080"},
            client=SimpleNamespace(host="127.0.0.1"),
        )
        self.assertEqual(extract_client_identifier(request), "2001:db8::1")


if __name__ == "__main__":
    unittest.main()
