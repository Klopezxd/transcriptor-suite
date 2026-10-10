"""
Módulo de utilidades: formateo, gestión de memoria, logging y excepciones.
Aplica principios de Clean Code con funciones puras y manejo de recursos.
"""

import gc
import logging
import os
import sys
from pathlib import Path

from src.config import TimestampMode

# --- Excepciones de Dominio ---


class TranscriptionError(Exception):
    """Excepción base para errores en el pipeline de transcripción."""

    pass


class MissingTokenError(TranscriptionError):
    """Lanzada cuando se requiere un token de Hugging Face y no fue provisto."""

    pass


class AudioProcessingError(TranscriptionError):
    """Lanzada cuando un archivo multimedia no puede ser procesado o cargado."""

    pass


# --- Logging Configuración ---


def setup_logger(name: str = "whisperx_transcriptor", level: int = logging.INFO) -> logging.Logger:
    """Configura y retorna un logger estructurado."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# --- Gestión de Memoria ---


def clear_gpu_vram() -> None:
    """Libera la memoria de GPU (VRAM) y ejecuta recolección de basura."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --- Formateo de Tiempos y Salidas ---


def format_timestamp(seconds: float | None) -> str:
    """
    Convierte una marca de tiempo en segundos a formato HH:MM:SS.

    Args:
        seconds: Tiempo en segundos (flotante o None).

    Returns:
        Cadena formateada en 'HH:MM:SS'.
    """
    if seconds is None or seconds < 0:
        return "00:00:00"
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_segment(
    text: str,
    start: float,
    end: float,
    speaker: str | None = None,
    mode: TimestampMode = TimestampMode.SIMPLE,
) -> str:
    """
    Formatea una línea de transcripción según el modo de marcas de tiempo y hablante.

    Args:
        text: Contenido transcrito del segmento.
        start: Segundo inicial.
        end: Segundo final.
        speaker: Identificador del hablante (si la diarización está activa).
        mode: TimestampMode (SIMPLE, RANGE o NONE).

    Returns:
        Línea de texto con el formato especificado.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return ""

    speaker_prefix = f"[{speaker}] " if speaker else ""
    start_str = format_timestamp(start)
    end_str = format_timestamp(end)

    if mode == TimestampMode.SIMPLE:
        return f"[{start_str}] {speaker_prefix}{cleaned_text}\n"
    elif mode == TimestampMode.RANGE:
        return f"[{start_str} - {end_str}] {speaker_prefix}{cleaned_text}\n"
    else:
        return f"{speaker_prefix}{cleaned_text}\n"


# --- Manejo Seguro de Secretos ---


def _resolve_secret(env_var: str, explicit_val: str | None = None) -> str | None:
    """
    Resuelve una credencial de forma segura con orden de prioridad:
    1. Valor explícito pasado como parámetro.
    2. Variable de entorno del sistema (os.getenv).
    3. Archivo .env local si existe.
    """
    if explicit_val and explicit_val.strip():
        return explicit_val.strip()

    env_val = os.getenv(env_var)
    if env_val and env_val.strip():
        return env_val.strip()

    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{env_var}=") and not line.startswith("#"):
                        val = line.split("=", 1)[1].strip().strip("\"'")
                        if val:
                            return val
        except Exception:
            pass

    return None


def resolve_hf_token(explicit_token: str | None = None) -> str | None:
    """Resuelve el token de Hugging Face (HF_TOKEN)."""
    return _resolve_secret("HF_TOKEN", explicit_token)


def resolve_deepgram_api_key(explicit_key: str | None = None) -> str | None:
    """Resuelve la API Key de Deepgram (DEEPGRAM_API_KEY)."""
    return _resolve_secret("DEEPGRAM_API_KEY", explicit_key)


def resolve_assemblyai_api_key(explicit_key: str | None = None) -> str | None:
    """Resuelve la API Key de AssemblyAI (ASSEMBLYAI_API_KEY)."""
    return _resolve_secret("ASSEMBLYAI_API_KEY", explicit_key)


def resolve_admin_pin(explicit_pin: str | None = None) -> str | None:
    """Resuelve el PIN maestro de administración (ADMIN_PIN o ADMIN_PASSWORD)."""
    if explicit_pin and explicit_pin.strip():
        return explicit_pin.strip()
    pin = _resolve_secret("ADMIN_PIN")
    if not pin:
        pin = _resolve_secret("ADMIN_PASSWORD")
    return pin


def verify_admin_pin(entered_pin: str | None) -> bool:
    """Valida si el PIN ingresado coincide con el configurado en el servidor."""
    if not entered_pin or not str(entered_pin).strip():
        return False
    master_pin = resolve_admin_pin()
    if not master_pin:
        return False
    return str(entered_pin).strip() == str(master_pin).strip()
