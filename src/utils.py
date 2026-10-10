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
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
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
    mode: TimestampMode = TimestampMode.SIMPLE
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

def resolve_hf_token(explicit_token: str | None = None) -> str | None:
    """
    Resuelve el token de Hugging Face de forma segura sin exponer credenciales.
    Prioridad: argumento explícito > variable de entorno HF_TOKEN > archivo .env.
    """
    if explicit_token and explicit_token.strip():
        return explicit_token.strip()

    env_token = os.getenv("HF_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()

    # Intentar leer desde archivo .env local si existe
    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                        token_val = line.split("=", 1)[1].strip().strip("\"'")
                        if token_val:
                            return token_val
        except Exception:
            pass
    return None


def resolve_deepgram_api_key(explicit_key: str | None = None) -> str | None:
    """
    Resuelve la API Key de Deepgram.
    Prioridad: argumento explícito > variable de entorno DEEPGRAM_API_KEY > archivo .env.
    """
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    env_key = os.getenv("DEEPGRAM_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DEEPGRAM_API_KEY=") and not line.startswith("#"):
                        val = line.split("=", 1)[1].strip().strip("\"'")
                        if val:
                            return val
        except Exception:
            pass

    return None


def resolve_assemblyai_api_key(explicit_key: str | None = None) -> str | None:
    """
    Resuelve la API Key de AssemblyAI.
    Prioridad: argumento explícito > variable de entorno ASSEMBLYAI_API_KEY > archivo .env.
    """
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    env_key = os.getenv("ASSEMBLYAI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ASSEMBLYAI_API_KEY=") and not line.startswith("#"):
                        val = line.split("=", 1)[1].strip().strip("\"'")
                        if val:
                            return val
        except Exception:
            pass

    return None

