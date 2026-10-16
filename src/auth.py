"""Protección del acceso administrativo sin almacenar contraseñas ni claves API."""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any


class LoginRateLimitError(Exception):
    """El cliente debe esperar antes de volver a intentar autenticarse."""


def _clean_ip(raw_ip: str | None) -> str:
    if not raw_ip or not str(raw_ip).strip():
        return ""
    cleaned = str(raw_ip).strip()
    # IPv6 entre corchetes con puerto: [2001:db8::1]:8080
    if cleaned.startswith("[") and "]" in cleaned:
        cleaned = cleaned[1 : cleaned.index("]")]
    # IPv4 con puerto: 192.0.2.1:8080
    elif ":" in cleaned and cleaned.count(":") == 1:
        cleaned = cleaned.split(":", 1)[0]
    return cleaned.strip()


def extract_client_identifier(request: Any) -> str:
    """
    Identifica al cliente de forma robusta e independiente en entornos con reverse proxy
    (Hugging Face Spaces, Cloudflare, Nginx) y sesiones locales de Gradio, evitando
    que múltiples usuarios compartan la misma IP local de proxy (DoS colectivo).
    """
    if request is None:
        return "unknown"

    headers = getattr(request, "headers", None)
    if headers and hasattr(headers, "get"):
        cf_ip = _clean_ip(headers.get("cf-connecting-ip"))
        if cf_ip and cf_ip not in {"127.0.0.1", "::1", "localhost"}:
            return cf_ip

        x_forwarded = headers.get("x-forwarded-for")
        if x_forwarded and x_forwarded.strip():
            client_ip = _clean_ip(x_forwarded.split(",")[0])
            if client_ip and client_ip not in {"127.0.0.1", "::1", "localhost"}:
                return client_ip

        x_real = _clean_ip(headers.get("x-real-ip"))
        if x_real and x_real not in {"127.0.0.1", "::1", "localhost"}:
            return x_real

    session_hash = getattr(request, "session_hash", None)
    if session_hash:
        return f"session_{session_hash}"

    client = getattr(request, "client", None)
    host = _clean_ip(getattr(client, "host", None))
    if host and host not in {"127.0.0.1", "::1", "localhost"}:
        return host

    return "local_client"


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
