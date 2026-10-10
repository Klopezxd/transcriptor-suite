"""
Orquestador del pipeline WhisperX.
Aplica principios de Clean Code, desacoplamiento y gestión óptima de VRAM.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import TranscriptionConfig
from src.utils import (
    AudioProcessingError,
    MissingTokenError,
    TranscriptionError,
    clear_gpu_vram,
    format_segment,
    resolve_hf_token,
    setup_logger,
)

logger = setup_logger("whisperx_pipeline")


@dataclass
class TranscriptionResult:
    """Resultado estructurado de un proceso de transcripción."""
    raw_segments: list[dict[str, Any]]
    detected_language: str
    formatted_text: str
    output_path: Path | None = None


class WhisperXPipeline:
    """
    Pipeline unificado de transcripción, alineación fonética y diarización.

    Garantiza liberación de VRAM en cada fase para operar fluidamente
    en GPUs de consumo (e.g. NVIDIA GTX 1650 con 4GB VRAM) y en ZeroGPU.
    """

    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Valida que la configuración cuente con los requisitos necesarios."""
        if self.config.enable_diarization:
            resolved_token = resolve_hf_token(self.config.hf_token)
            if not resolved_token:
                raise MissingTokenError(
                    "La diarización (identificación de personas) requiere un token de Hugging Face. "
                    "Configura la variable de entorno HF_TOKEN o provéelo en los argumentos."
                )

    def process(
        self,
        media_path: str | Path,
        output_dir: str | Path | None = None,
        progress_callback: Callable[[float, str], None] | None = None
    ) -> TranscriptionResult:
        """
        Ejecuta el pipeline completo sobre el archivo de audio o video provisto.

        Args:
            media_path: Ruta al archivo multimedia (MP4, MP3, WAV, MKV, etc.).
            output_dir: Carpeta destino para el archivo .txt de salida (opcional).
            progress_callback: Función callback (progreso: 0.0 a 1.0, mensaje: str).

        Returns:
            TranscriptionResult con los datos procesados y ruta del archivo generado.
        """
        input_path = Path(media_path)
        if not input_path.exists():
            raise AudioProcessingError(f"No se encontró el archivo: {input_path}")

        def notify(progress: float, message: str) -> None:
            logger.info("[%d%%] %s", int(progress * 100), message)
            if progress_callback:
                progress_callback(progress, message)

        try:
            import whisperx
        except ImportError as err:
            raise TranscriptionError("WhisperX no está instalado en el entorno actual.") from err

        device = self.config.device
        compute_type = self.config.compute_type

        try:
            # --- FASE 1: Carga y Extracción de Audio ---
            notify(0.10, f"Cargando archivo multimedia: {input_path.name}")
            audio = whisperx.load_audio(str(input_path))

            # --- FASE 2: Transcripción ASR con WhisperX ---
            notify(0.25, f"Transcribiendo con modelo '{self.config.model_name}' ({compute_type})...")
            model = whisperx.load_model(
                self.config.model_name,
                device=device,
                compute_type=compute_type,
                language=self.config.language
            )
            raw_result = model.transcribe(audio, batch_size=self.config.batch_size)
            detected_language = raw_result.get("language", "es")
            del model
            clear_gpu_vram()

            # --- FASE 3: Alineación Fonética de Marcas de Tiempo ---
            notify(0.55, "Alineando marcas de tiempo a nivel de palabra...")
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=device
            )
            aligned_result = whisperx.align(
                raw_result["segments"],
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )
            del align_model
            clear_gpu_vram()

            # --- FASE 4: Diarización de Hablantes (Opcional) ---
            segments = aligned_result["segments"]
            if self.config.enable_diarization:
                notify(0.75, "Identificando interlocutores (Diarización con Pyannote)...")
                try:
                    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

                    hf_token = resolve_hf_token(self.config.hf_token)
                    try:
                        diarize_model = DiarizationPipeline(token=hf_token, device=device)
                    except TypeError:
                        diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)

                    diarize_segments = diarize_model(audio)
                    diarized_result = assign_word_speakers(diarize_segments, aligned_result)
                    segments = diarized_result["segments"]
                    del diarize_model
                    clear_gpu_vram()
                except Exception as diarize_err:
                    clear_gpu_vram()
                    err_str = str(diarize_err)
                    if any(term in err_str.lower() for term in ["401", "403", "gated", "restricted", "unauthorized"]):
                        raise TranscriptionError(
                            "Error de acceso a los modelos de Pyannote en Hugging Face.\n\n"
                            "Verifica los siguientes requisitos:\n"
                            "1. Aceptar condiciones en: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                            "2. Aceptar condiciones en: https://huggingface.co/pyannote/segmentation-3.0\n"
                            "3. Asegurarte de que tu token de Hugging Face tenga permisos de lectura (Read)."
                        ) from diarize_err
                    raise TranscriptionError(f"Error durante la diarización: {diarize_err}") from diarize_err

            # --- FASE 5: Generación y Escritura del Archivo Final ---
            notify(0.90, "Generando archivo de transcripción...")
            formatted_lines: list[str] = []
            for seg in segments:
                text = seg.get("text", "")
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                speaker = seg.get("speaker") if self.config.enable_diarization else None

                line = format_segment(
                    text=text,
                    start=start,
                    end=end,
                    speaker=speaker,
                    mode=self.config.timestamp_mode
                )
                if line:
                    formatted_lines.append(line)

            formatted_text = "".join(formatted_lines)

            # Determinar ruta de salida
            target_dir = Path(output_dir) if output_dir else input_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            output_file = target_dir / f"{input_path.stem}_TRANSCRIPCION.txt"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(formatted_text)

            notify(1.0, f"Transcripción completada con éxito en: {output_file.name}")

            return TranscriptionResult(
                raw_segments=segments,
                detected_language=detected_language,
                formatted_text=formatted_text,
                output_path=output_file
            )

        except Exception as e:
            clear_gpu_vram()
            if isinstance(e, TranscriptionError):
                raise
            raise TranscriptionError(f"Error durante el procesamiento: {str(e)}") from e


def create_transcription_pipeline(config: TranscriptionConfig):
    """
    Fábrica unificada de pipelines.
    Retorna la instancia correspondiente al proveedor configurado.
    """
    from src.config import ProviderType

    if config.provider == ProviderType.DEEPGRAM.value:
        from src.cloud_providers import DeepgramPipeline
        return DeepgramPipeline(config)
    elif config.provider == ProviderType.ASSEMBLYAI.value:
        from src.cloud_providers import AssemblyAIPipeline
        return AssemblyAIPipeline(config)
    else:
        return WhisperXPipeline(config)

