"""Protección del acceso administrativo sin almacenar contraseñas ni claves API."""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable


class LoginRateLimitError(Exception):
    """El cliente debe esperar antes de volver a intentar autenticarse."""


class LoginAttemptLimiter:
    """Limita fallos por cliente durante una ventana, con memoria acotada por proceso."""

    def __init__(
        self,
        max_attempts: int = 5,
        window_seconds: float = 60,
        max_clients: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_attempts < 1 or window_seconds <= 0 or max_clients < 1:
            raise ValueError("Los límites de autenticación deben ser positivos.")
        self._max_attempts = max_attempts
        self._window_seconds = window_seconds
        self._max_clients = max_clients
        self._clock = clock
        self._failures: OrderedDict[str, tuple[int, float]] = OrderedDict()
        self._lock = threading.Lock()

    def authenticate(self, pin: str, client: str, verifier: Callable[[str], bool]) -> bool:
        """Verifica bajo el mismo bloqueo que cuenta los fallos concurrentes."""
        with self._lock:
            now = self._clock()
            attempts, started = self._failures.get(client, (0, now))
            if now - started >= self._window_seconds:
                attempts, started = 0, now
            if attempts >= self._max_attempts:
                raise LoginRateLimitError(
                    "Demasiados intentos. Espera un minuto antes de volver a ingresar."
                )
            if verifier(pin):
                self._failures.pop(client, None)
                return True
            if client not in self._failures and len(self._failures) >= self._max_clients:
                self._failures.popitem(last=False)
            self._failures[client] = (attempts + 1, started)
            self._failures.move_to_end(client)
            return False
