"""
Módulo de proveedores en la nube para transcripción y diarización.
Integra Deepgram (Nova-2/Nova-3) y AssemblyAI (Universal-1/Nano) con llamadas REST optimizadas.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from src.config import TranscriptionConfig
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

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".m4v"}


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
    cmd_copy = ["ffmpeg", "-y", "-i", str(media_path), "-vn", "-c:a", "copy", str(temp_audio)]
    try:
        res = subprocess.run(
            cmd_copy,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if res.returncode == 0 and temp_audio.is_file() and temp_audio.stat().st_size > 0:
            logger.info(
                "Pista de audio extraída sin pérdida (stream copy) de %s a %s (%d bytes)",
                media_path.name,
                temp_audio.name,
                temp_audio.stat().st_size,
            )
            return temp_audio, True
    except Exception as err:
        logger.debug("Fallo en extracción directa de audio con ffmpeg: %s", err)

    if temp_audio.exists():
        temp_audio.unlink(missing_ok=True)

    # Intento 2: Fallback Lossless garantizado con FLAC (Free Lossless Audio Codec)
    temp_flac = temp_audio.with_suffix(".flac")
    cmd_fallback = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-c:a",
        "flac",
        str(temp_flac),
    ]
    try:
        res = subprocess.run(
            cmd_fallback,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if res.returncode == 0 and temp_flac.is_file() and temp_flac.stat().st_size > 0:
            logger.info(
                "Pista de audio convertida a FLAC (Lossless) para %s (%d bytes)",
                media_path.name,
                temp_flac.stat().st_size,
            )
            return temp_flac, True
    except Exception as err:
        logger.warning(
            "No se pudo extraer audio del video %s: %s. Se subirá el archivo original.",
            media_path.name,
            err,
        )

    if temp_flac.exists():
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
            "model": self.config.model_name or "nova-2",
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
        model_label = params.get("model", "nova-2")
        file_size = input_path.stat().st_size

        try:
            with open(input_path, "rb") as audio_file:
                reader = ProgressFileReader(
                    file_obj=audio_file,
                    total_size=file_size,
                    notify=notify,
                    start_p=0.10,
                    end_p=0.52,
                    upload_message=f"Subiendo audio a Deepgram ({model_label})...",
                )

                def _stage_message(current_p: float, _elapsed: float) -> str:
                    if reader.bytes_read < reader.total_size:
                        return f"Subiendo audio a Deepgram ({model_label})..."
                    if current_p < 0.78:
                        return f"Transcribiendo audio en Deepgram ({model_label})..."
                    return "Alineando marcas temporales e interlocutores en Deepgram..."

                with smooth_progress_stage(
                    notify=notify,
                    start_p=0.10,
                    end_p=0.92,
                    message=_stage_message,
                    tau_seconds=6.5,
                    interval_seconds=0.25,
                ):
                    response = requests.post(
                        self.API_URL,
                        params=params,
                        headers=headers,
                        data=reader,
                        timeout=self.REQUEST_TIMEOUT_SEC,
                    )

            if response.status_code == 401:
                raise TranscriptionError(
                    "Clave de API de Deepgram no autorizada (401). Verifica que tu DEEPGRAM_API_KEY sea válida."
                )
            if response.status_code != 200:
                raise TranscriptionError(
                    f"Error de Deepgram (HTTP {response.status_code}): {response.text}"
                )

            return _response_json(response)
        except requests.exceptions.RequestException as req_err:
            raise TranscriptionError(f"Fallo de conexión con Deepgram: {req_err}") from req_err

    def _extract_segments(self, channel_alt: dict[str, Any]) -> list[dict[str, Any]]:
        words = channel_alt.get("words", [])
        segments: list[dict[str, Any]] = []

        if not words:
            full_transcript = channel_alt.get("transcript", "")
            if full_transcript:
                segments.append(
                    {
                        "text": full_transcript,
                        "start": 0.0,
                        "end": 0.0,
                        "speaker": None,
                    }
                )
            return segments

        current_speaker: int | None = None
        current_words: list[str] = []
        seg_start: float = 0.0
        seg_end: float = 0.0

        def flush_segment() -> None:
            if not current_words:
                return
            speaker_label = (
                f"SPEAKER_{current_speaker:02d}" if current_speaker is not None else None
            )
            segments.append(
                {
                    "text": " ".join(current_words),
                    "start": seg_start,
                    "end": seg_end,
                    "speaker": speaker_label,
                }
            )

        for word_item in words:
            spk = word_item.get("speaker") if self.config.enable_diarization else None
            w_text = word_item.get("punctuated_word", word_item.get("word", ""))
            w_start = float(word_item.get("start", 0.0))
            w_end = float(word_item.get("end", 0.0))

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


class AssemblyAIPipeline:
    """
    Cliente optimizado para la API de AssemblyAI (Universal-1 y Nano).
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

    @staticmethod
    def _file_chunk_stream(
        file_path: Path,
        chunk_size: int,
        notify: Callable[[float, str], None],
        start_p: float,
        end_p: float,
    ) -> Iterator[bytes]:
        total_size = max(1, file_path.stat().st_size)
        bytes_sent = 0
        msg = f"Subiendo archivo a AssemblyAI: {file_path.name}..."
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                bytes_sent += len(chunk)
                ratio = min(1.0, bytes_sent / total_size)
                notify(start_p + (end_p - start_p) * ratio, msg)
                yield chunk

    def _upload_media(
        self,
        input_path: Path,
        headers: dict[str, str],
        notify: Callable[[float, str], None],
    ) -> str:
        msg = f"Subiendo archivo a AssemblyAI: {input_path.name}..."
        try:
            with smooth_progress_stage(
                notify=notify,
                start_p=0.08,
                end_p=0.42,
                message=msg,
                tau_seconds=5.0,
                interval_seconds=0.25,
            ):
                upload_res = requests.post(
                    self.UPLOAD_URL,
                    headers=headers,
                    data=self._file_chunk_stream(
                        input_path, self.UPLOAD_CHUNK_SIZE, notify, 0.08, 0.42
                    ),
                    timeout=600,
                )
            if upload_res.status_code == 401:
                raise TranscriptionError(
                    "Clave de API de AssemblyAI no autorizada (401). Verifica tu ASSEMBLYAI_API_KEY."
                )
            if upload_res.status_code != 200:
                raise TranscriptionError(
                    f"Error al subir archivo a AssemblyAI (HTTP {upload_res.status_code}): {upload_res.text}"
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
            "speech_model": self.config.model_name or "best",
            "speaker_labels": self.config.enable_diarization,
        }
        if self.config.language:
            transcript_payload["language_code"] = self.config.language
        else:
            transcript_payload["language_detection"] = True

        try:
            submit_res = requests.post(
                self.TRANSCRIPT_URL,
                headers=headers,
                json=transcript_payload,
                timeout=60,
            )
            if submit_res.status_code != 200:
                raise TranscriptionError(
                    f"Error al solicitar transcripción en AssemblyAI: {submit_res.text}"
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
        model_label = self.config.model_name or "best"

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
                time.sleep(min(self.POLL_INTERVAL_SEC, max(0.0, remaining)))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                try:
                    poll_res = requests.get(poll_url, headers=headers, timeout=min(30, remaining))
                    if poll_res.status_code not in {200, 408, 429} and poll_res.status_code < 500:
                        raise TranscriptionError(
                            f"Error de AssemblyAI durante el sondeo (HTTP {poll_res.status_code})."
                        )
                    if poll_res.status_code == 200:
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
                except requests.exceptions.RequestException as err:
                    logger.warning("Fallo temporal de sondeo: %s", err)

        raise TranscriptionError(
            "Tiempo máximo de espera agotado para la transcripción en AssemblyAI."
        )

    def _extract_segments(self, final_data: dict[str, Any]) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        utterances = final_data.get("utterances")

        if self.config.enable_diarization and utterances:
            for utt in utterances:
                segments.append(
                    {
                        "text": utt.get("text", ""),
                        "start": float(utt.get("start", 0)) / 1000.0,
                        "end": float(utt.get("end", 0)) / 1000.0,
                        "speaker": f"SPEAKER_{utt.get('speaker', 'A')}",
                    }
                )
            return segments

        words = final_data.get("words", [])
        if words:
            current_words: list[str] = []
            seg_start = float(words[0].get("start", 0)) / 1000.0
            seg_end = float(words[0].get("end", 0)) / 1000.0

            for word_item in words:
                if not current_words:
                    seg_start = float(word_item.get("start", 0)) / 1000.0
                w_text = word_item.get("text", "")
                w_end = float(word_item.get("end", 0)) / 1000.0
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

        full_text = final_data.get("text", "")
        if full_text:
            segments.append(
                {
                    "text": full_text,
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

        # --- Paso 1: Subida de archivo ---
        payload_path, is_temp = extract_audio_for_cloud(input_path, notify)
        try:
            audio_url = self._upload_media(payload_path, headers, notify)

            # --- Paso 2: Solicitud de transcripción ---
            notify(0.45, f"Iniciando transcripción con modelo '{self.config.model_name or 'best'}'...")
            transcript_id = self._submit_transcript_job(audio_url, headers)

            # --- Paso 3: Sondeo periódico (Polling) ---
            final_data = self._poll_completion(transcript_id, headers, notify)
        finally:
            if is_temp and payload_path.exists():
                payload_path.unlink(missing_ok=True)

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
