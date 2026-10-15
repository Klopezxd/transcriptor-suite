"""Límites de autenticación sin esperas reales ni credenciales del despliegue."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from src.auth import (
    LoginAttemptLimiter,
    LoginRateLimitError,
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
        for options in [{"max_attempts": 0}, {"window_seconds": 0}, {"max_clients": 0}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                LoginAttemptLimiter(**options)


class ClientIdentifierTests(unittest.TestCase):
    def test_none_request(self):
        self.assertEqual(extract_client_identifier(None), "unknown")

    def test_cf_connecting_ip_header(self):
        request = SimpleNamespace(headers={"cf-connecting-ip": "198.51.100.1"})
        self.assertEqual(extract_client_identifier(request), "198.51.100.1")

    def test_x_forwarded_for_first_element(self):
        request = SimpleNamespace(headers={"x-forwarded-for": "203.0.113.195, 10.0.0.1"})
        self.assertEqual(extract_client_identifier(request), "203.0.113.195")

    def test_x_real_ip_header(self):
        request = SimpleNamespace(headers={"x-real-ip": "192.0.2.1"})
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


if __name__ == "__main__":
    unittest.main()
