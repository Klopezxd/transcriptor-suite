"""Límites de autenticación sin esperas reales ni credenciales del despliegue."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from src.auth import LoginAttemptLimiter, LoginRateLimitError


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


if __name__ == "__main__":
    unittest.main()
