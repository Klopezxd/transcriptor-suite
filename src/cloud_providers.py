"""
Módulo de proveedores en la nube para transcripción y diarización.
Integra Deepgram (Nova-2/Nova-3) y AssemblyAI (STT-3.7/Universal-2) con llamadas REST optimizadas.
"""

from __future__ import annotations

import contextlib
import datetime
import math
import os
import secrets
import subprocess
import tempfile
import time
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from src.config import AssemblyAIModel, DeepgramModel, TranscriptionConfig
from src.utils import (
    ProgressFileReader,
    TranscriptionError,
    make_progress_notifier,
    render_segments_text,
    resolve_assemblyai_api_key,
    resolve_deepgram_api_key,
    save_transcription_file,
    setup_logger,
    smooth_progress_stage,
    validate_media_path,
)

if TYPE_CHECKING:
    from src.transcriber import TranscriptionResult

logger = setup_logger("cloud_providers")

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".flv",
    ".wmv",
    ".m4v",
    ".webm",
    ".mpeg",
    ".mpg",
    ".ts",
    ".mts",
    ".m2ts",
    ".3gp",
    ".ogv",
}
FFMPEG_EXTRACT_TIMEOUT_SEC = 180
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
MAX_HTTP_RETRIES = 2
RETRY_BASE_DELAY_SEC = 0.5
RETRY_MAX_DELAY_SEC = 30.0
MAX_CONSECUTIVE_POLL_ERRORS = 12


def _truncate_response_text(text: Any, max_len: int = 300) -> str:
    """Resume cuerpos de error HTTP sin exponer páginas HTML extensas."""
    if not isinstance(text, str):
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len]}..."


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convierte valores numéricos arbitrarios a float finito de forma segura."""
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _format_speaker_label(raw_speaker: Any) -> str | None:
    """Formatea etiquetas de hablante tolerando int, float, str o None."""
    if raw_speaker is None or isinstance(raw_speaker, bool):
        return None
    if isinstance(raw_speaker, int):
        return f"SPEAKER_{raw_speaker:02d}"
    if isinstance(raw_speaker, float):
        if math.isfinite(raw_speaker) and raw_speaker.is_integer():
            return f"SPEAKER_{int(raw_speaker):02d}"
        return f"SPEAKER_{raw_speaker}"
    text = str(raw_speaker).strip()
    if not text:
        return None
    if text.isdigit():
        return f"SPEAKER_{int(text):02d}"
    if text.upper().startswith("SPEAKER_"):
        return text
    return f"SPEAKER_{text}"


def _parse_retry_after(headers: Any) -> float | None:
    """Extrae segundos de espera desde el encabezado HTTP Retry-After si es válido."""
    if not headers or not hasattr(headers, "get"):
        return None
    raw_val = headers.get("Retry-After")
    if not isinstance(raw_val, str):
        return None
    val = raw_val.strip()
    if not val:
        return None
    try:
        seconds = float(val)
        if math.isfinite(seconds):
            return min(RETRY_MAX_DELAY_SEC, max(0.0, seconds))
    except ValueError:
        pass
    with contextlib.suppress(TypeError, ValueError, OverflowError):
        retry_dt = parsedate_to_datetime(val)
        if retry_dt.tzinfo is None:
            retry_dt = retry_dt.replace(tzinfo=datetime.timezone.utc)
        delta = (retry_dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        if math.isfinite(delta):
            return min(RETRY_MAX_DELAY_SEC, max(0.0, delta))
    return None


def _compute_backoff_delay(
    attempt: int,
    headers: Any = None,
    base_delay: float = RETRY_BASE_DELAY_SEC,
) -> float:
    """Calcula espera respetando Retry-After o backoff exponencial con jitter."""
    retry_after = _parse_retry_after(headers)
    if retry_after is not None:
        return min(RETRY_MAX_DELAY_SEC, max(0.1, retry_after))
    jitter = secrets.randbelow(250) / 1000.0
    return min(RETRY_MAX_DELAY_SEC, base_delay * (2**attempt) + jitter)


def _request_with_retry(
    method: str,
    url: str,
    *,
    before_retry: Callable[[], None] | None = None,
    max_retries: int = MAX_HTTP_RETRIES,
    **kwargs: Any,
) -> requests.Response:
    """
    Ejecuta peticiones HTTP con reintentos sobre códigos transitorios (429/5xx) y timeouts,
    preservando llamadas directas a requests.post / requests.get para compatibilidad con mocks.
    """
    sender = requests.post if method.upper() == "POST" else requests.get
    last_response: requests.Response | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0 and before_retry is not None:
            before_retry()
        try:
            resp = sender(url, **kwargs)
        except requests.exceptions.RequestException as err:
            if attempt >= max_retries:
                raise
            delay = _compute_backoff_delay(attempt)
            logger.warning(
                "Error de red transitorio (%s); reintentando en %.2fs (%d/%d)...",
                err,
                delay,
                attempt + 1,
                max_retries,
            )
            time.sleep(delay)
            continue

        last_response = resp
        if resp.status_code not in RETRYABLE_STATUS_CODES or attempt >= max_retries:
            return resp

        delay = _compute_backoff_delay(attempt, getattr(resp, "headers", None))
        logger.warning(
            "Respuesta HTTP %s transitoria; reintentando en %.2fs (%d/%d)...",
            resp.status_code,
            delay,
            attempt + 1,
            max_retries,
        )
        time.sleep(delay)

    if last_response is None:
        raise TranscriptionError("No se recibió respuesta HTTP del proveedor.")
    return last_response


def _log_ffmpeg_stderr(stage: str, res: Any) -> None:
    """Registra stderr truncado de ffmpeg para diagnóstico sin contaminar logs."""
    stderr_val = getattr(res, "stderr", None)
    if isinstance(stderr_val, bytes):
        text = stderr_val.decode("utf-8", errors="replace").strip()
    elif isinstance(stderr_val, str):
        text = stderr_val.strip()
    else:
        return
    if text:
        logger.debug("ffmpeg (%s) stderr: %s", stage, _truncate_response_text(text, 300))


def extract_audio_for_cloud(
    media_path: Path, notify: Callable[[float, str], None] | None = None
) -> tuple[Path, bool]:
    """
    Si el archivo es un video, extrae la pista de audio con ffmpeg a un archivo temporal ligero
    para reducir drásticamente el peso de subida por red a las APIs de nube (Deepgram / AssemblyAI).

    Returns:
        (ruta_audio, es_temporal): tuple con el Path a subir y un booleano indicando si debe eliminarse al finalizar.
    """
    if media_path.suffix.lower() not in VIDEO_EXTENSIONS:
        return media_path, False

    fd, temp_file_str = tempfile.mkstemp(suffix=".m4a", prefix="cloud_audio_")
    os.close(fd)
    temp_audio = Path(temp_file_str)

    if notify:
        notify(0.04, f"Extrayendo pista de audio de {media_path.name} para optimizar subida...")

    # Intento 1: Copia de stream directo sin recodificar (-vn -c:a copy) - 100% Lossless
    cmd_copy = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-protocol_whitelist",
        "file",
        "-i",
        str(media_path),
        "-vn",
        "-c:a",
        "copy",
        str(temp_audio),
    ]
    try:
        res = subprocess.run(  # noqa: S603
            cmd_copy,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=FFMPEG_EXTRACT_TIMEOUT_SEC,
        )
        if res.returncode == 0 and temp_audio.is_file() and temp_audio.stat().st_size > 0:
            logger.info(
                "Pista de audio extraída sin pérdida (stream copy) de %s a %s (%d bytes)",
                media_path.name,
                temp_audio.name,
                temp_audio.stat().st_size,
            )
            return temp_audio, True
        _log_ffmpeg_stderr("stream_copy", res)
    except FileNotFoundError:
        logger.warning(
            "ffmpeg no está instalado en el entorno; se subirá %s original.",
            media_path.name,
        )
        temp_audio.unlink(missing_ok=True)
        return media_path, False
    except subprocess.TimeoutExpired:
        logger.warning(
            "Tiempo límite agotado (%ds) extrayendo audio de %s; se subirá archivo original.",
            FFMPEG_EXTRACT_TIMEOUT_SEC,
            media_path.name,
        )
        temp_audio.unlink(missing_ok=True)
        return media_path, False
    except Exception as err:
        logger.debug("Fallo en extracción directa de audio con ffmpeg: %s", err)

    temp_audio.unlink(missing_ok=True)

    # Intento 2: Fallback Lossless garantizado con FLAC (Free Lossless Audio Codec)
    fd_flac, temp_flac_str = tempfile.mkstemp(suffix=".flac", prefix="cloud_audio_")
    os.close(fd_flac)
    temp_flac = Path(temp_flac_str)
    cmd_fallback = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-protocol_whitelist",
        "file",
        "-i",
        str(media_path),
        "-vn",
        "-c:a",
        "flac",
        str(temp_flac),
    ]
    try:
        res = subprocess.run(  # noqa: S603
            cmd_fallback,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=FFMPEG_EXTRACT_TIMEOUT_SEC,
        )
        if res.returncode == 0 and temp_flac.is_file() and temp_flac.stat().st_size > 0:
            logger.info(
                "Pista de audio convertida a FLAC (Lossless) para %s (%d bytes)",
                media_path.name,
                temp_flac.stat().st_size,
            )
            return temp_flac, True
        _log_ffmpeg_stderr("flac_fallback", res)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as err:
        logger.warning(
            "No se pudo extraer audio del video %s: %s. Se subirá el archivo original.",
            media_path.name,
            err,
        )

    temp_flac.unlink(missing_ok=True)
    return media_path, False


def _response_json(response: requests.Response) -> dict[str, Any]:
    """Normaliza respuestas JSON inválidas al contrato de errores del dominio."""
    try:
        data = response.json()
    except ValueError as err:
        raise TranscriptionError("El proveedor devolvió una respuesta JSON inválida.") from err
    if not isinstance(data, dict):
        raise TranscriptionError("El proveedor devolvió una estructura de respuesta inválida.")
    return data


class DeepgramPipeline:
    """
    Cliente optimizado para la API de Deepgram (Nova-2 y Nova-3).
    Soporta audio masivo, diarización nativa y marcas de tiempo por palabra.
    """

    API_URL = "https://api.deepgram.com/v1/listen"
    MAX_SEGMENT_DURATION_SEC = 25.0
    REQUEST_TIMEOUT_SEC = 600

    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config
        self.api_key = resolve_deepgram_api_key(config.api_key)
        if not self.api_key:
            raise TranscriptionError(
                "No se encontró la API Key de Deepgram.\n"
                "Configura 'DEEPGRAM_API_KEY' en tu entorno o ingrésala en la interfaz."
            )

    def _build_query_params(self) -> dict[str, str]:
        params: dict[str, str] = {
            "model": self.config.model_name or DeepgramModel.NOVA_3.value,
            "smart_format": "true",
            "punctuate": "true",
            "diarize": "true" if self.config.enable_diarization else "false",
        }
        if self.config.language:
            params["language"] = self.config.language
        else:
            params["detect_language"] = "true"
        return params

    def _send_audio_request(
        self,
        input_path: Path,
        params: dict[str, str],
        notify: Callable[[float, str], None],
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/octet-stream",
        }
        model_label = params.get("model", DeepgramModel.NOVA_3.value)
        file_size = input_path.stat().st_size
        upload_msg = f"Subiendo audio a Deepgram ({model_label})..."

        try:
            with open(input_path, "rb") as audio_file:
                reader = ProgressFileReader(
                    file_obj=audio_file,
                    total_size=file_size,
                    notify=notify,
                    start_p=0.10,
                    end_p=0.52,
                    upload_message=upload_msg,
                )
                notify(0.10, upload_msg)

                def _stage_message(current_p: float, _elapsed: float) -> str:
                    if reader.bytes_read < reader.total_size:
                        return upload_msg
                    if current_p < 0.78:
                        return f"Transcribiendo audio en Deepgram ({model_label})..."
                    return "Alineando marcas temporales e interlocutores en Deepgram..."

                with smooth_progress_stage(
                    notify=notify,
                    start_p=0.52,
                    end_p=0.92,
                    message=_stage_message,
                    tau_seconds=6.5,
                    interval_seconds=0.25,
                    gate=lambda: reader.bytes_read >= reader.total_size,
                ):
                    response = _request_with_retry(
                        "POST",
                        self.API_URL,
                        before_retry=lambda: reader.seek(0),
                        params=params,
                        headers=headers,
                        data=reader,
                        timeout=self.REQUEST_TIMEOUT_SEC,
                    )

            if response.status_code == 401:
                raise TranscriptionError(
                    "Clave de API de Deepgram no autorizada (401). Verifica que tu DEEPGRAM_API_KEY sea válida."
                )
            if response.status_code == 429:
                raise TranscriptionError(
                    "Límite de peticiones de Deepgram alcanzado (HTTP 429). Intenta de nuevo en unos instantes."
                )
            if response.status_code != 200:
                detail = _truncate_response_text(getattr(response, "text", ""))
                raise TranscriptionError(
                    f"Error de Deepgram (HTTP {response.status_code}): {detail}"
                )

            return _response_json(response)
        except requests.exceptions.RequestException as req_err:
            raise TranscriptionError(f"Fallo de conexión con Deepgram: {req_err}") from req_err

    def _extract_segments(self, channel_alt: dict[str, Any]) -> list[dict[str, Any]]:
        words = channel_alt.get("words") or []
        segments: list[dict[str, Any]] = []

        if not words:
            full_transcript = channel_alt.get("transcript") or ""
            if full_transcript:
                segments.append(
                    {
                        "text": str(full_transcript),
                        "start": 0.0,
                        "end": 0.0,
                        "speaker": None,
                    }
                )
            return segments

        current_speaker: Any = None
        current_words: list[str] = []
        seg_start: float = 0.0
        seg_end: float = 0.0

        def flush_segment() -> None:
            if not current_words:
                return
            segments.append(
                {
                    "text": " ".join(current_words),
                    "start": seg_start,
                    "end": seg_end,
                    "speaker": _format_speaker_label(current_speaker),
                }
            )

        for word_item in words:
            if not isinstance(word_item, dict):
                continue
            spk = word_item.get("speaker") if self.config.enable_diarization else None
            raw_word = word_item.get("punctuated_word")
            if raw_word is None:
                raw_word = word_item.get("word")
            w_text = str(raw_word or "")
            w_start = _safe_float(word_item.get("start"), 0.0)
            w_end = _safe_float(word_item.get("end"), w_start)

            if not current_words:
                current_speaker = spk
                seg_start = w_start
                seg_end = w_end
                current_words = [w_text]
            elif spk == current_speaker and (w_end - seg_start < self.MAX_SEGMENT_DURATION_SEC):
                current_words.append(w_text)
                seg_end = w_end
            else:
                flush_segment()
                current_speaker = spk
                seg_start = w_start
                seg_end = w_end
                current_words = [w_text]

        flush_segment()
        return segments

    def process(
        self,
        media_path: str | Path,
        output_dir: str | Path | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> TranscriptionResult:
        from src.transcriber import TranscriptionResult

        input_path = validate_media_path(media_path)

        notify = make_progress_notifier(logger, progress_callback)

        notify(0.05, f"Preparando archivo multimedia: {input_path.name}")
        params = self._build_query_params()

        payload_path, is_temp = extract_audio_for_cloud(input_path, notify)
        try:
            data = self._send_audio_request(payload_path, params, notify)
        finally:
            if is_temp and payload_path.exists():
                payload_path.unlink(missing_ok=True)

        notify(0.94, "Procesando respuesta y formateando marcas temporales...")
        try:
            channel = data["results"]["channels"][0]
            channel_alt = channel["alternatives"][0]
            detected_language = channel.get("detected_language") or self.config.language or "unknown"
        except (KeyError, IndexError, TypeError) as err:
            raise TranscriptionError(
                "La respuesta de Deepgram no contiene transcripciones válidas."
            ) from err

        segments = self._extract_segments(channel_alt)

        notify(0.98, "Escribiendo archivo de transcripción...")
        formatted_text = render_segments_text(
            segments=segments,
            mode=self.config.timestamp_mode,
            include_speaker=True,
        )
        output_file = save_transcription_file(
            formatted_text=formatted_text,
            input_path=input_path,
            output_dir=output_dir,
        )

        notify(1.0, f"Transcripción completada con Deepgram: {output_file.name}")

        return TranscriptionResult(
            raw_segments=segments,
            detected_language=detected_language,
            formatted_text=formatted_text,
            output_path=output_file,
        )


def resolve_assemblyai_speech_models(model_name: str | None) -> list[str]:
    """
    Resuelve la lista de modelos para el parámetro 'speech_models' de AssemblyAI.
    Para máxima fidelidad y cobertura, 'assemblyai-stt-3-7' incluye 'universal-2' como respaldo.
    Soporta 'universal-3-5-pro' y alias para retrocompatibilidad.
    """
    raw = (model_name or "").strip().lower()
    if raw in {"universal-2", "nano"}:
        return ["universal-2"]
    if raw in {
        "assemblyai-stt-3-7",
        "assemblyai-stt-3.7",
        "stt-3-7",
        "stt-3.7",
        "best",
        "default",
        "",
    }:
        return ["assemblyai-stt-3-7", "universal-2"]
    if raw in {"universal-3-5-pro", "universal-3.5-pro"}:
        return ["universal-3-5-pro", "universal-2"]
    return [raw]


class AssemblyAIPipeline:
    """
    Cliente optimizado para la API de AssemblyAI (STT-3.7 y Universal-2).
    Soporta subida en streaming, sondeo de estado y diarización con etiquetas de interlocutor.
    """

    UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
    TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
    UPLOAD_CHUNK_SIZE = 262_144
    POLL_INTERVAL_SEC = 2
    MAX_POLL_SECONDS = 3600

    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config
        self.api_key = resolve_assemblyai_api_key(config.api_key)
        if not self.api_key:
            raise TranscriptionError(
                "No se encontró la API Key de AssemblyAI.\n"
                "Configura 'ASSEMBLYAI_API_KEY' en tu entorno o ingrésala en la interfaz."
            )

    def _upload_media(
        self,
        input_path: Path,
        headers: dict[str, str],
        notify: Callable[[float, str], None],
    ) -> str:
        msg = f"Subiendo archivo a AssemblyAI: {input_path.name}..."
        file_size = input_path.stat().st_size
        try:
            with open(input_path, "rb") as audio_file:
                reader = ProgressFileReader(
                    file_obj=audio_file,
                    total_size=file_size,
                    notify=notify,
                    start_p=0.08,
                    end_p=0.42,
                    upload_message=msg,
                )
                notify(0.08, msg)
                upload_res = _request_with_retry(
                    "POST",
                    self.UPLOAD_URL,
                    before_retry=lambda: reader.seek(0),
                    headers=headers,
                    data=reader,
                    timeout=600,
                )
            if upload_res.status_code == 401:
                raise TranscriptionError(
                    "Clave de API de AssemblyAI no autorizada (401). Verifica tu ASSEMBLYAI_API_KEY."
                )
            if upload_res.status_code == 429:
                raise TranscriptionError(
                    "Límite de peticiones al subir archivo a AssemblyAI (HTTP 429). Intenta de nuevo más tarde."
                )
            if upload_res.status_code != 200:
                detail = _truncate_response_text(getattr(upload_res, "text", ""))
                raise TranscriptionError(
                    f"Error al subir archivo a AssemblyAI (HTTP {upload_res.status_code}): {detail}"
                )

            audio_url = _response_json(upload_res).get("upload_url")
            if not audio_url:
                raise TranscriptionError("AssemblyAI no retornó una URL de archivo válida.")
            return str(audio_url)
        except requests.exceptions.RequestException as err:
            raise TranscriptionError(f"Fallo durante la subida a AssemblyAI: {err}") from err

    def _submit_transcript_job(self, audio_url: str, headers: dict[str, str]) -> str:
        transcript_payload: dict[str, Any] = {
            "audio_url": audio_url,
            "speech_models": resolve_assemblyai_speech_models(self.config.model_name),
            "speaker_labels": self.config.enable_diarization,
        }
        if self.config.language:
            transcript_payload["language_code"] = self.config.language
        else:
            transcript_payload["language_detection"] = True

        try:
            submit_res = _request_with_retry(
                "POST",
                self.TRANSCRIPT_URL,
                headers=headers,
                json=transcript_payload,
                timeout=60,
            )
            if submit_res.status_code == 401:
                raise TranscriptionError(
                    "Clave de API de AssemblyAI no autorizada (401). Verifica tu ASSEMBLYAI_API_KEY."
                )
            if submit_res.status_code != 200:
                detail = _truncate_response_text(getattr(submit_res, "text", ""))
                raise TranscriptionError(
                    f"Error al solicitar transcripción en AssemblyAI (HTTP {submit_res.status_code}): {detail}"
                )
            transcript_id = _response_json(submit_res).get("id")
            if not transcript_id:
                raise TranscriptionError("AssemblyAI no retornó un ID de transcripción válido.")
            return str(transcript_id)
        except requests.exceptions.RequestException as err:
            raise TranscriptionError(f"Error al contactar con AssemblyAI: {err}") from err

    def _poll_completion(
        self,
        transcript_id: str,
        headers: dict[str, str],
        notify: Callable[[float, str], None],
    ) -> dict[str, Any]:
        poll_url = f"{self.TRANSCRIPT_URL}/{transcript_id}"
        deadline = time.monotonic() + self.MAX_POLL_SECONDS
        model_label = self.config.model_name or AssemblyAIModel.ASSEMBLYAI_STT_3_7.value
        consecutive_errors = 0

        with smooth_progress_stage(
            notify=notify,
            start_p=0.48,
            end_p=0.92,
            message=f"Transcribiendo en la nube con AssemblyAI ({model_label})...",
            tau_seconds=10.0,
            interval_seconds=0.25,
        ):
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                try:
                    poll_res = requests.get(
                        poll_url,
                        headers=headers,
                        timeout=min(30.0, max(1.0, remaining)),
                    )
                    if poll_res.status_code == 200:
                        consecutive_errors = 0
                        status_data = _response_json(poll_res)
                        status = status_data.get("status")

                        if status == "completed":
                            return status_data
                        if status == "error":
                            error_msg = status_data.get("error", "Error desconocido")
                            raise TranscriptionError(
                                f"AssemblyAI falló durante el procesamiento: {error_msg}"
                            )
                        if status not in {"queued", "processing"}:
                            raise TranscriptionError(
                                "AssemblyAI devolvió un estado de transcripción inválido."
                            )
                        sleep_for = min(
                            self.POLL_INTERVAL_SEC, max(0.0, deadline - time.monotonic())
                        )
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                        continue

                    if poll_res.status_code in RETRYABLE_STATUS_CODES:
                        consecutive_errors += 1
                        if consecutive_errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                            raise TranscriptionError(
                                f"AssemblyAI devolvió errores transitorios repetidos (HTTP {poll_res.status_code})."
                            )
                        delay = _compute_backoff_delay(
                            min(consecutive_errors - 1, 5),
                            getattr(poll_res, "headers", None),
                            base_delay=float(self.POLL_INTERVAL_SEC),
                        )
                        logger.warning(
                            "Sondeo AssemblyAI recibió HTTP %s; esperando %.2fs (%d/%d)...",
                            poll_res.status_code,
                            delay,
                            consecutive_errors,
                            MAX_CONSECUTIVE_POLL_ERRORS,
                        )
                        sleep_for = min(delay, max(0.0, deadline - time.monotonic()))
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                        continue

                    if poll_res.status_code < 400:
                        logger.warning(
                            "Respuesta inesperada HTTP %s en sondeo de AssemblyAI.",
                            poll_res.status_code,
                        )
                        sleep_for = min(
                            self.POLL_INTERVAL_SEC, max(0.0, deadline - time.monotonic())
                        )
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                        continue

                    raise TranscriptionError(
                        f"Error de AssemblyAI durante el sondeo (HTTP {poll_res.status_code})."
                    )
                except requests.exceptions.RequestException as err:
                    consecutive_errors += 1
                    logger.warning("Fallo temporal de sondeo: %s", err)
                    if consecutive_errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                        raise TranscriptionError(
                            f"Fallo de conexión repetido sondeando AssemblyAI: {err}"
                        ) from err
                    delay = _compute_backoff_delay(
                        min(consecutive_errors - 1, 5),
                        base_delay=float(self.POLL_INTERVAL_SEC),
                    )
                    sleep_for = min(delay, max(0.0, deadline - time.monotonic()))
                    if sleep_for > 0:
                        time.sleep(sleep_for)

        raise TranscriptionError(
            "Tiempo máximo de espera agotado para la transcripción en AssemblyAI."
        )

    def _extract_segments(self, final_data: dict[str, Any]) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        utterances = final_data.get("utterances")

        if self.config.enable_diarization and utterances:
            for utt in utterances:
                if not isinstance(utt, dict):
                    continue
                speaker_val = utt.get("speaker")
                speaker_label = _format_speaker_label(
                    "A" if speaker_val is None else speaker_val
                )
                segments.append(
                    {
                        "text": str(utt.get("text") or ""),
                        "start": _safe_float(utt.get("start"), 0.0) / 1000.0,
                        "end": _safe_float(utt.get("end"), 0.0) / 1000.0,
                        "speaker": speaker_label,
                    }
                )
            return segments

        words = final_data.get("words") or []
        if words:
            current_words: list[str] = []
            seg_start = (
                _safe_float(words[0].get("start"), 0.0) / 1000.0
                if isinstance(words[0], dict)
                else 0.0
            )
            seg_end = (
                _safe_float(words[0].get("end"), 0.0) / 1000.0
                if isinstance(words[0], dict)
                else 0.0
            )

            for word_item in words:
                if not isinstance(word_item, dict):
                    continue
                if not current_words:
                    seg_start = _safe_float(word_item.get("start"), 0.0) / 1000.0
                w_text = str(word_item.get("text") or "")
                w_end = _safe_float(word_item.get("end"), seg_start * 1000.0) / 1000.0
                current_words.append(w_text)
                seg_end = w_end

                if (seg_end - seg_start) >= 20.0 or w_text.endswith((".", "?", "!")):
                    segments.append(
                        {
                            "text": " ".join(current_words),
                            "start": seg_start,
                            "end": seg_end,
                            "speaker": None,
                        }
                    )
                    current_words = []
                    seg_start = seg_end

            if current_words:
                segments.append(
                    {
                        "text": " ".join(current_words),
                        "start": seg_start,
                        "end": seg_end,
                        "speaker": None,
                    }
                )
            return segments

        full_text = final_data.get("text") or ""
        if full_text:
            segments.append(
                {
                    "text": str(full_text),
                    "start": 0.0,
                    "end": 0.0,
                    "speaker": None,
                }
            )
        return segments

    def process(
        self,
        media_path: str | Path,
        output_dir: str | Path | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> TranscriptionResult:
        from src.transcriber import TranscriptionResult

        input_path = validate_media_path(media_path)

        notify = make_progress_notifier(logger, progress_callback)
        headers = {"authorization": self.api_key or ""}

        # --- Paso 1: Extracción y subida de archivo (limpieza inmediata del temporal) ---
        payload_path, is_temp = extract_audio_for_cloud(input_path, notify)
        try:
            audio_url = self._upload_media(payload_path, headers, notify)
        finally:
            if is_temp and payload_path.exists():
                payload_path.unlink(missing_ok=True)

        # --- Paso 2: Solicitud de transcripción ---
        model_label = self.config.model_name or AssemblyAIModel.ASSEMBLYAI_STT_3_7.value
        notify(0.45, f"Iniciando transcripción con modelo '{model_label}'...")
        transcript_id = self._submit_transcript_job(audio_url, headers)

        # --- Paso 3: Sondeo periódico (Polling) ---
        final_data = self._poll_completion(transcript_id, headers, notify)

        # --- Paso 4: Formateo y Persistencia ---
        notify(0.95, "Formateando segmentos y hablantes...")
        detected_language = final_data.get("language_code") or self.config.language or "unknown"
        segments = self._extract_segments(final_data)

        formatted_text = render_segments_text(
            segments=segments,
            mode=self.config.timestamp_mode,
            include_speaker=True,
        )
        output_file = save_transcription_file(
            formatted_text=formatted_text,
            input_path=input_path,
            output_dir=output_dir,
        )

        notify(1.0, f"Transcripción completada con AssemblyAI: {output_file.name}")

        return TranscriptionResult(
            raw_segments=segments,
            detected_language=detected_language,
            formatted_text=formatted_text,
            output_path=output_file,
        )
