"""
Módulo de utilidades: formateo, gestión de memoria, logging y excepciones.
Aplica principios de Clean Code con funciones puras y manejo de recursos.
"""

import contextlib
import contextvars
import gc
import hmac
import logging
import math
import os
import sys
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

from src.config import TimestampMode

# --- Excepciones de Dominio ---


class TranscriptionError(Exception):
    """Excepción base para errores en el pipeline de transcripción."""


class MissingTokenError(TranscriptionError):
    """Lanzada cuando se requiere un token de Hugging Face y no fue provisto."""


class AudioProcessingError(TranscriptionError):
    """Lanzada cuando un archivo multimedia no puede ser procesado o cargado."""


# --- Configuración de Consola y Logging ---


def configure_utf8_console() -> None:
    """Garantiza codificación UTF-8 en las salidas estándar en entornos Windows."""
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                with contextlib.suppress(Exception):
                    reconfigure(encoding="utf-8")


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


# --- Gestión de Progreso Continuo en Tiempo Real ---


def make_progress_notifier(
    logger: logging.Logger,
    progress_callback: Callable[[float, str], None] | None,
) -> Callable[[float, str], None]:
    """
    Crea un notificador thread-safe y monotónico que actualiza la barra de progreso
    de Gradio en tiempo real y registra en consola únicamente los cambios de etapa.
    """
    last_logged_msg: str | None = None
    last_p: float = 0.0
    lock = threading.Lock()

    def notify(progress_val: float, message: str) -> None:
        nonlocal last_logged_msg, last_p
        should_log = False
        with lock:
            clamped = min(1.0, max(last_p, progress_val))
            last_p = clamped
            notify.current_progress = clamped  # type: ignore[attr-defined]
            if message != last_logged_msg:
                should_log = True
                last_logged_msg = message

        if should_log:
            logger.info("[%d%%] %s", int(clamped * 100), message)

        if progress_callback:
            with contextlib.suppress(Exception):
                progress_callback(clamped, message)

    notify.current_progress = 0.0  # type: ignore[attr-defined]
    return notify


@contextlib.contextmanager
def smooth_progress_stage(
    notify: Callable[[float, str], None],
    start_p: float,
    end_p: float,
    message: str | Callable[[float, float], str],
    tau_seconds: float = 8.0,
    interval_seconds: float = 0.25,
) -> Iterator[None]:
    """
    Avanza suavemente la barra de progreso entre start_p y end_p en un hilo
    secundario (preservando contextvars de Gradio) mientras se ejecuta una
    operación bloqueante de red o GPU.
    """
    stop_event = threading.Event()
    ctx = contextvars.copy_context()
    t0 = time.perf_counter()

    initial_msg = message(start_p, 0.0) if callable(message) else message
    notify(start_p, initial_msg)

    def _ticker() -> None:
        current_p = start_p
        last_tick = t0
        tau = max(0.5, tau_seconds)
        while not stop_event.wait(interval_seconds):
            now = time.perf_counter()
            dt = max(0.01, now - last_tick)
            last_tick = now
            elapsed = now - t0
            base_p = max(current_p, float(getattr(notify, "current_progress", current_p)))
            step_fraction = 1.0 - math.exp(-dt / tau)
            current_p = base_p + max(0.0, end_p - base_p) * step_fraction
            current_msg = message(current_p, elapsed) if callable(message) else message
            notify(current_p, current_msg)

    worker = threading.Thread(target=ctx.run, args=(_ticker,), daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop_event.set()
        worker.join(timeout=1.0)


class ProgressFileReader:
    """
    Envoltorio de archivo binario que reporta el progreso real de lectura/subida
    de bytes sin perder el encabezado Content-Length en peticiones HTTP.
    """

    def __init__(
        self,
        file_obj: BinaryIO,
        total_size: int,
        notify: Callable[[float, str], None],
        start_p: float,
        end_p: float,
        upload_message: str,
    ) -> None:
        self._file = file_obj
        self.total_size = max(1, total_size)
        self._notify = notify
        self._start_p = start_p
        self._end_p = end_p
        self._upload_message = upload_message
        self.bytes_read = 0
        self._last_emit = 0.0

    def __len__(self) -> int:
        return self.total_size

    def read(self, size: int = -1) -> bytes:
        chunk = self._file.read(size)
        if chunk:
            self.bytes_read += len(chunk)
            now = time.perf_counter()
            if (now - self._last_emit) >= 0.12 or self.bytes_read >= self.total_size:
                self._last_emit = now
                ratio = min(1.0, self.bytes_read / self.total_size)
                current_p = self._start_p + (self._end_p - self._start_p) * ratio
                self._notify(current_p, self._upload_message)
        return chunk


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
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "00:00:00"
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration_human(seconds: float | None) -> str:
    """
    Convierte una duración en segundos a una representación humana legible (minutos y segundos).

    Args:
        seconds: Tiempo transcurrido en segundos.

    Returns:
        Cadena con formato 'X min Y s (Z.Zs)' o 'Z.Zs'.
    """
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "0.0s"
    if seconds < 60:
        return f"{seconds:.1f} segundos"
    total_secs = int(seconds)
    mins = total_secs // 60
    rem_secs = total_secs % 60
    return f"{mins} min {rem_secs} s ({seconds:.1f}s)"


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
    if mode == TimestampMode.RANGE:
        return f"[{start_str} - {end_str}] {speaker_prefix}{cleaned_text}\n"
    return f"{speaker_prefix}{cleaned_text}\n"


def render_segments_text(
    segments: Iterable[dict[str, Any]],
    mode: TimestampMode = TimestampMode.SIMPLE,
    include_speaker: bool = True,
) -> str:
    """
    Transforma una colección de segmentos crudos en el texto final formateado.
    """
    formatted_lines: list[str] = []
    for seg in segments:
        speaker = seg.get("speaker") if include_speaker else None
        line = format_segment(
            text=str(seg.get("text", "")),
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
            speaker=speaker,
            mode=mode,
        )
        if line:
            formatted_lines.append(line)
    return "".join(formatted_lines)


def save_transcription_file(
    formatted_text: str,
    input_path: Path,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Persiste el texto transcrito en un archivo '<nombre>_TRANSCRIPCION.txt' en UTF-8.
    """
    target_dir = Path(output_dir) if output_dir else input_path.parent
    temporary_path = None
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file = target_dir / f"{input_path.stem}_TRANSCRIPCION.txt"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target_dir, delete=False, suffix=".tmp"
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(formatted_text)
        temporary_path.replace(output_file)
        return output_file
    except OSError as err:
        raise TranscriptionError("No se pudo guardar el archivo de transcripción.") from err
    finally:
        if temporary_path is not None:
            with contextlib.suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def save_consolidated_batch_file(
    content: str,
    output_dir: str | Path,
    timestamp_str: str | None = None,
) -> Path:
    """
    Persiste de forma atómica el texto unificado de un lote multiarchivo en un archivo .txt único.

    Args:
        content: Texto consolidado con separadores por archivo.
        output_dir: Carpeta destino.
        timestamp_str: Marca de tiempo opcional para el nombre del archivo.

    Returns:
        Ruta Path del archivo consolidado generado.
    """
    ts = timestamp_str or time.strftime("%Y%m%d_%H%M%S")
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_file = target_dir / f"transcripcion_consolidada_lote_{ts}.txt"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target_dir, delete=False, suffix=".tmp"
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
        temporary_path.replace(output_file)
        return output_file
    except OSError as err:
        raise TranscriptionError(
            "No se pudo guardar el archivo de transcripción consolidada."
        ) from err
    finally:
        if temporary_path is not None:
            with contextlib.suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def validate_media_path(media_path: str | Path) -> Path:
    """Valida archivos locales antes de abrir conexiones o cargar modelos."""
    input_path = Path(media_path)
    try:
        if not input_path.is_file():
            raise AudioProcessingError(f"No se encontró un archivo válido: {input_path}")
        if input_path.stat().st_size == 0:
            raise AudioProcessingError("El archivo multimedia está vacío.")
    except OSError as err:
        raise AudioProcessingError(f"No se pudo acceder al archivo: {input_path}") from err
    return input_path


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
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, raw_val = line.partition("=")
                    if key.strip() == env_var:
                        val = raw_val.strip()
                        if val.startswith(('"', "'")):
                            quote_char = val[0]
                            closing_idx = val.find(quote_char, 1)
                            val = val[1:closing_idx] if closing_idx != -1 else val.strip("\"'")
                        else:
                            if " #" in val:
                                val = val.split(" #", 1)[0].rstrip()
                            elif "\t#" in val:
                                val = val.split("\t#", 1)[0].rstrip()
                        if val:
                            return val
        except OSError:
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
    return _resolve_secret("ADMIN_PIN") or _resolve_secret("ADMIN_PASSWORD")


def verify_admin_pin(entered_pin: str | None) -> bool:
    """Valida en tiempo constante si el PIN ingresado coincide con el configurado en el servidor."""
    if not entered_pin or not str(entered_pin).strip():
        return False
    master_pin = resolve_admin_pin()
    if not master_pin:
        return False
    return hmac.compare_digest(
        str(entered_pin).strip().encode("utf-8"),
        str(master_pin).strip().encode("utf-8"),
    )


def create_zip_archive(files: list[Path], output_zip_path: Path) -> Path:
    """
    Empaqueta una lista de archivos existentes en un archivo ZIP de forma atómica.

    Args:
        files: Lista de rutas Path a los archivos a incluir.
        output_zip_path: Ruta destino para el archivo .zip.

    Returns:
        Path del archivo ZIP generado.
    """
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    temp_zip = output_zip_path.with_suffix(".tmp.zip")
    try:
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in files:
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.name)
        temp_zip.replace(output_zip_path)
        return output_zip_path
    finally:
        if temp_zip.exists():
            with contextlib.suppress(OSError):
                temp_zip.unlink(missing_ok=True)

