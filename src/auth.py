"""Protección del acceso administrativo sin almacenar contraseñas ni claves API."""

import ipaddress
import os
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import Any

_IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network

# Redes de reverse proxy locales/privadas confiables cuando hay conexión directa desde proxy
_DEFAULT_TRUSTED_PROXY_NETWORKS: tuple[_IPNetwork, ...] = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


class LoginRateLimitError(Exception):
    """El cliente debe esperar antes de volver a intentar autenticarse."""


def _clean_ip(raw_ip: str | None) -> str:
    """Normaliza y valida una dirección IPv4/IPv6 (opcionalmente con puerto)."""
    if not raw_ip or not str(raw_ip).strip():
        return ""
    cleaned = str(raw_ip).strip()

    # IPv6 entre corchetes con puerto opcional: [2001:db8::1]:8080
    if cleaned.startswith("["):
        closing = cleaned.find("]")
        if closing == -1:
            return ""
        suffix = cleaned[closing + 1 :]
        if suffix:
            if not suffix.startswith(":") or not suffix[1:].isdigit():
                return ""
            if not (0 <= int(suffix[1:]) <= 65535):
                return ""
        cleaned = cleaned[1:closing].strip()
    # IPv4 con puerto: 192.0.2.1:8080
    elif cleaned.count(":") == 1:
        host_part, port_part = cleaned.split(":", 1)
        if not port_part.isdigit() or not (0 <= int(port_part) <= 65535):
            return ""
        cleaned = host_part.strip()

    try:
        ip_obj = ipaddress.ip_address(cleaned)
    except ValueError:
        return ""

    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        ip_obj = ip_obj.ipv4_mapped
    return str(ip_obj)


def _get_configured_trusted_proxies() -> tuple[_IPNetwork, ...]:
    """Parsea redes/IPs de proxies confiables desde la variable TRUSTED_PROXY_IPS."""
    raw = os.getenv("TRUSTED_PROXY_IPS", "")
    if not raw.strip():
        return ()
    networks: list[_IPNetwork] = []
    for item in raw.split(","):
        candidate = item.strip()
        if not candidate:
            continue
        try:
            networks.append(ipaddress.ip_network(candidate, strict=False))
        except ValueError:
            continue
    return tuple(networks)


def _is_behind_explicit_proxy(configured_proxies: tuple[_IPNetwork, ...]) -> bool:
    """Indica si el entorno declaró explícitamente operar detrás de un reverse proxy."""
    if configured_proxies:
        return True
    flag = os.getenv("TRUST_PROXY_HEADERS", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _parse_ip_for_network_check(ip_str: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if not ip_str:
        return None
    base = ip_str.split("%", 1)[0]
    try:
        return ipaddress.ip_address(base)
    except ValueError:
        return None


def _is_trusted_proxy(ip_str: str, configured_proxies: tuple[_IPNetwork, ...]) -> bool:
    ip_obj = _parse_ip_for_network_check(ip_str)
    if ip_obj is None:
        return False
    return any(ip_obj in net for net in configured_proxies) or any(
        ip_obj in net for net in _DEFAULT_TRUSTED_PROXY_NETWORKS
    )


def _is_loopback(ip_str: str) -> bool:
    ip_obj = _parse_ip_for_network_check(ip_str)
    return bool(ip_obj and ip_obj.is_loopback)


def _extract_forwarded_for(
    raw_header: str, configured_proxies: tuple[_IPNetwork, ...]
) -> tuple[str, str]:
    """
    Recorre X-Forwarded-For de derecha a izquierda para extraer el primer salto que no
    pertenece a proxies confiables, evitando spoofing en los valores más a la izquierda.
    Retorna (ip_cliente_no_proxy, fallback_proxy_interno_no_loopback).
    """
    internal_fallback = ""
    for raw_hop in reversed(raw_header.split(",")):
        token = raw_hop.strip()
        if not token:
            continue
        hop_ip = _clean_ip(token)
        if not hop_ip:
            break
        if not _is_trusted_proxy(hop_ip, configured_proxies):
            return hop_ip, ""
        if not _is_loopback(hop_ip):
            internal_fallback = hop_ip
    return "", internal_fallback


def extract_client_identifier(request: Any) -> str:
    """
    Identifica al cliente de forma resistente a spoofing de cabeceras proxy.

    Confía en CF-Connecting-IP, X-Forwarded-For (salto más a la derecha fuera de proxies
    confiables) y X-Real-IP únicamente cuando `request.client.host` pertenece a proxies
    confiables (`TRUSTED_PROXY_IPS` o redes loopback/privadas de reverse proxy), o cuando
    `request.client` está ausente únicamente si hay proxy explícito (`TRUSTED_PROXY_IPS` o
    `TRUST_PROXY_HEADERS=true`). Si `request.client.host` es una IP pública directa, ignora
    cabeceras falsificables y mantiene la identidad vinculada a la IP de conexión.
    """
    if request is None:
        return "unknown"

    client = getattr(request, "client", None)
    peer_ip = _clean_ip(getattr(client, "host", None))
    configured_proxies = _get_configured_trusted_proxies()

    if peer_ip:
        trust_proxy_headers = _is_trusted_proxy(peer_ip, configured_proxies)
    else:
        trust_proxy_headers = _is_behind_explicit_proxy(configured_proxies)

    internal_xff_fallback = ""
    if trust_proxy_headers:
        headers = getattr(request, "headers", None)
        if headers and hasattr(headers, "get"):
            cf_ip = _clean_ip(headers.get("cf-connecting-ip"))
            if cf_ip and not _is_loopback(cf_ip):
                return cf_ip

            x_forwarded = headers.get("x-forwarded-for")
            if x_forwarded and x_forwarded.strip():
                client_ip, internal_xff_fallback = _extract_forwarded_for(
                    x_forwarded, configured_proxies
                )
                if client_ip and not _is_loopback(client_ip):
                    return client_ip

            x_real = _clean_ip(headers.get("x-real-ip"))
            if x_real and not _is_loopback(x_real):
                return x_real

    if peer_ip and not trust_proxy_headers:
        return peer_ip

    session_hash = getattr(request, "session_hash", None)
    if session_hash:
        return f"session_{session_hash}"

    if peer_ip and not _is_loopback(peer_ip):
        return peer_ip

    if internal_xff_fallback:
        return internal_xff_fallback

    return "local_client"


class LoginAttemptLimiter:
    """Limita fallos por cliente y de forma global por ventana deslizante, con memoria acotada."""

    def __init__(
        self,
        max_attempts: int = 5,
        window_seconds: float = 60,
        max_clients: int = 4096,
        clock: Callable[[], float] = time.monotonic,
        max_global_attempts: int | None = None,
        global_window_seconds: float | None = None,
    ) -> None:
        resolved_global_attempts = (
            max(20, max_attempts * 4)
            if max_global_attempts is None
            else max_global_attempts
        )
        resolved_global_window = (
            window_seconds if global_window_seconds is None else global_window_seconds
        )
        if (
            max_attempts < 1
            or window_seconds <= 0
            or max_clients < 1
            or resolved_global_attempts < 1
            or resolved_global_window <= 0
        ):
            raise ValueError("Los límites de autenticación deben ser positivos.")
        self._max_attempts = max_attempts
        self._window_seconds = window_seconds
        self._max_clients = max_clients
        self._clock = clock
        self._max_global_attempts = resolved_global_attempts
        self._global_window_seconds = resolved_global_window
        self._failures: OrderedDict[str, tuple[int, float]] = OrderedDict()
        self._global_failures: deque[float] = deque(maxlen=resolved_global_attempts)
        self._lock = threading.Lock()

    def authenticate(self, pin: str, client: str, verifier: Callable[[str], bool]) -> bool:
        """Verifica bajo el mismo bloqueo que controla los fallos por cliente y globales."""
        with self._lock:
            now = self._clock()
            while (
                self._global_failures
                and (now - self._global_failures[0]) >= self._global_window_seconds
            ):
                self._global_failures.popleft()

            attempts, started = self._failures.get(client, (0, now))
            if now - started >= self._window_seconds:
                attempts, started = 0, now

            if attempts >= self._max_attempts or (
                len(self._global_failures) >= self._max_global_attempts
            ):
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
            self._global_failures.append(now)
            return False
