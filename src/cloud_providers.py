"""
Módulo de proveedores en la nube para transcripción y diarización.
Integra Deepgram (Nova-2/Nova-3) y AssemblyAI (Universal-1/Nano) con llamadas REST optimizadas.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from src.config import TranscriptionConfig
from src.utils import (
    AudioProcessingError,
    TranscriptionError,
    format_segment,
    resolve_assemblyai_api_key,
    resolve_deepgram_api_key,
    setup_logger,
)

logger = setup_logger("cloud_providers")


class DeepgramPipeline:
    """
    Cliente optimizado para la API de Deepgram (Nova-2 y Nova-3).
    Soporta audio masivo, diarización nativa y marcas de tiempo por palabra.
    """

    API_URL = "https://api.deepgram.com/v1/listen"

    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config
        self.api_key = resolve_deepgram_api_key(config.api_key)
        if not self.api_key:
            raise TranscriptionError(
                "No se encontró la API Key de Deepgram.\n"
                "Configura 'DEEPGRAM_API_KEY' en tu entorno o ingrésala en la interfaz."
            )

    def process(
        self,
        media_path: str | Path,
        output_dir: str | Path | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> Any:
        from src.transcriber import TranscriptionResult

        input_path = Path(media_path)
        if not input_path.exists():
            raise AudioProcessingError(f"No se encontró el archivo: {input_path}")

        def notify(p: float, msg: str) -> None:
            logger.info("[%d%%] %s", int(p * 100), msg)
            if progress_callback:
                progress_callback(p, msg)

        notify(0.10, f"Preparando archivo multimedia: {input_path.name}")

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

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/octet-stream",
        }

        notify(0.30, f"Enviando audio a Deepgram ({params['model']})...")

        try:
            with open(input_path, "rb") as audio_file:
                response = requests.post(
                    self.API_URL,
                    params=params,
                    headers=headers,
                    data=audio_file,
                    timeout=600,
                )

            if response.status_code == 401:
                raise TranscriptionError(
                    "Clave de API de Deepgram no autorizada (401). Verifica que tu DEEPGRAM_API_KEY sea válida."
                )
            if response.status_code != 200:
                raise TranscriptionError(
                    f"Error de Deepgram (HTTP {response.status_code}): {response.text}"
                )

            data = response.json()
        except requests.exceptions.RequestException as req_err:
            raise TranscriptionError(f"Fallo de conexión con Deepgram: {req_err}") from req_err

        notify(0.85, "Procesando respuesta y formateando marcas temporales...")

        try:
            channel_alt = data["results"]["channels"][0]["alternatives"][0]
            detected_language = data["results"]["channels"][0].get("detected_language", "es")
        except (KeyError, IndexError) as err:
            raise TranscriptionError(
                "La respuesta de Deepgram no contiene transcripciones válidas."
            ) from err

        words = channel_alt.get("words", [])
        segments: list[dict[str, Any]] = []

        if words:
            current_speaker: int | None = None
            current_words: list[str] = []
            seg_start: float = 0.0
            seg_end: float = 0.0

            for w in words:
                spk = w.get("speaker") if self.config.enable_diarization else None
                w_text = w.get("punctuated_word", w.get("word", ""))
                w_start = float(w.get("start", 0.0))
                w_end = float(w.get("end", 0.0))

                if current_speaker is None:
                    current_speaker = spk
                    seg_start = w_start
                    seg_end = w_end
                    current_words = [w_text]
                elif spk == current_speaker and (w_end - seg_start < 25.0):
                    current_words.append(w_text)
                    seg_end = w_end
                else:
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
                    current_speaker = spk
                    seg_start = w_start
                    seg_end = w_end
                    current_words = [w_text]

            if current_words:
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
        else:
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

        notify(0.95, "Escribiendo archivo de transcripción...")

        formatted_lines: list[str] = []
        for seg in segments:
            line = format_segment(
                text=seg.get("text", ""),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                speaker=seg.get("speaker"),
                mode=self.config.timestamp_mode,
            )
            if line:
                formatted_lines.append(line)

        formatted_text = "".join(formatted_lines)

        target_dir = Path(output_dir) if output_dir else input_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file = target_dir / f"{input_path.stem}_TRANSCRIPCION.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted_text)

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

    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config
        self.api_key = resolve_assemblyai_api_key(config.api_key)
        if not self.api_key:
            raise TranscriptionError(
                "No se encontró la API Key de AssemblyAI.\n"
                "Configura 'ASSEMBLYAI_API_KEY' en tu entorno o ingrésala en la interfaz."
            )

    def process(
        self,
        media_path: str | Path,
        output_dir: str | Path | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> Any:
        from src.transcriber import TranscriptionResult

        input_path = Path(media_path)
        if not input_path.exists():
            raise AudioProcessingError(f"No se encontró el archivo: {input_path}")

        def notify(p: float, msg: str) -> None:
            logger.info("[%d%%] %s", int(p * 100), msg)
            if progress_callback:
                progress_callback(p, msg)

        headers = {"authorization": self.api_key}

        # --- Paso 1: Subida de archivo ---
        notify(0.15, f"Subiendo archivo a AssemblyAI: {input_path.name}...")

        def file_generator(file_path: Path, chunk_size: int = 5_242_880):
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    yield chunk

        try:
            upload_res = requests.post(
                self.UPLOAD_URL,
                headers=headers,
                data=file_generator(input_path),
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

            audio_url = upload_res.json().get("upload_url")
            if not audio_url:
                raise TranscriptionError("AssemblyAI no retornó una URL de archivo válida.")
        except requests.exceptions.RequestException as err:
            raise TranscriptionError(f"Fallo durante la subida a AssemblyAI: {err}") from err

        # --- Paso 2: Solicitud de transcripción ---
        notify(0.35, f"Iniciando transcripción con modelo '{self.config.model_name or 'best'}'...")

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
            transcript_id = submit_res.json().get("id")
        except requests.exceptions.RequestException as err:
            raise TranscriptionError(f"Error al contactar con AssemblyAI: {err}") from err

        # --- Paso 3: Sondeo periódico (Polling) ---
        poll_url = f"{self.TRANSCRIPT_URL}/{transcript_id}"
        notify(0.45, "Procesando en los servidores de AssemblyAI...")

        elapsed = 0
        final_data: dict[str, Any] = {}

        while True:
            time.sleep(3)
            elapsed += 3
            progress_val = min(0.45 + (elapsed * 0.015), 0.90)
            notify(progress_val, f"Transcribiendo en la nube con AssemblyAI ({elapsed}s)...")

            try:
                poll_res = requests.get(poll_url, headers=headers, timeout=30)
                if poll_res.status_code == 200:
                    status_data = poll_res.json()
                    status = status_data.get("status")

                    if status == "completed":
                        final_data = status_data
                        break
                    elif status == "error":
                        error_msg = status_data.get("error", "Error desconocido")
                        raise TranscriptionError(
                            f"AssemblyAI falló durante el procesamiento: {error_msg}"
                        )
            except requests.exceptions.RequestException as err:
                logger.warning("Fallo temporal de sondeo: %s", err)

        notify(0.92, "Formateando segmentos y hablantes...")

        segments: list[dict[str, Any]] = []
        utterances = final_data.get("utterances")
        detected_language = final_data.get("language_code", "es")

        if self.config.enable_diarization and utterances:
            for utt in utterances:
                start_s = float(utt.get("start", 0)) / 1000.0
                end_s = float(utt.get("end", 0)) / 1000.0
                spk = utt.get("speaker", "A")
                text = utt.get("text", "")
                segments.append(
                    {
                        "text": text,
                        "start": start_s,
                        "end": end_s,
                        "speaker": f"SPEAKER_{spk}",
                    }
                )
        else:
            words = final_data.get("words", [])
            if words:
                current_words: list[str] = []
                seg_start = float(words[0].get("start", 0)) / 1000.0
                seg_end = float(words[0].get("end", 0)) / 1000.0

                for w in words:
                    w_text = w.get("text", "")
                    w_end = float(w.get("end", 0)) / 1000.0
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
            else:
                full_text = final_data.get("text", "")
                segments.append(
                    {
                        "text": full_text,
                        "start": 0.0,
                        "end": 0.0,
                        "speaker": None,
                    }
                )

        formatted_lines: list[str] = []
        for seg in segments:
            line = format_segment(
                text=seg.get("text", ""),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                speaker=seg.get("speaker"),
                mode=self.config.timestamp_mode,
            )
            if line:
                formatted_lines.append(line)

        formatted_text = "".join(formatted_lines)

        target_dir = Path(output_dir) if output_dir else input_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file = target_dir / f"{input_path.stem}_TRANSCRIPCION.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted_text)

        notify(1.0, f"Transcripción completada con AssemblyAI: {output_file.name}")

        return TranscriptionResult(
            raw_segments=segments,
            detected_language=detected_language,
            formatted_text=formatted_text,
            output_path=output_file,
        )
