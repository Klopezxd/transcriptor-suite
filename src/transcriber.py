"""
Orquestador del pipeline WhisperX y contrato base de transcripción.
Aplica principios de Clean Code, SOLID (DIP/SRP) y gestión óptima de VRAM.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.config import ProviderType, TranscriptionConfig
from src.utils import (
    AudioProcessingError,
    MissingTokenError,
    TranscriptionError,
    clear_gpu_vram,
    render_segments_text,
    resolve_hf_token,
    save_transcription_file,
    setup_logger,
)

logger = setup_logger("whisperx_pipeline")

ProgressCallback = Callable[[float, str], None]


@dataclass
class TranscriptionResult:
    """Resultado estructurado de un proceso de transcripción."""

    raw_segments: list[dict[str, Any]]
    detected_language: str
    formatted_text: str
    output_path: Path | None = None


class TranscriptionPipeline(Protocol):
    """Contrato común para todos los motores de transcripción (Local y Cloud)."""

    def process(
        self,
        media_path: str | Path,
        output_dir: str | Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        """Procesa un archivo multimedia y retorna su resultado estructurado."""
        ...


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

    def _run_diarization(
        self,
        audio: Any,
        aligned_result: dict[str, Any],
        device: str,
    ) -> list[dict[str, Any]]:
        """Ejecuta el modelo de diarización Pyannote y asigna hablantes por palabra."""
        try:
            from whisperx.diarize import DiarizationPipeline, assign_word_speakers

            hf_token = resolve_hf_token(self.config.hf_token)
            try:
                diarize_model = DiarizationPipeline(token=hf_token, device=device)
            except TypeError:
                diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)

            diarize_segments = diarize_model(audio)
            diarized_result = assign_word_speakers(diarize_segments, aligned_result)
            segments: list[dict[str, Any]] = diarized_result["segments"]
            del diarize_model
            clear_gpu_vram()
            return segments
        except Exception as diarize_err:
            clear_gpu_vram()
            err_str = str(diarize_err).lower()
            if any(
                term in err_str
                for term in ("401", "403", "gated", "restricted", "unauthorized")
            ):
                raise TranscriptionError(
                    "Error de acceso a los modelos de Pyannote en Hugging Face.\n\n"
                    "Verifica los siguientes requisitos:\n"
                    "1. Aceptar condiciones en: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                    "2. Aceptar condiciones en: https://huggingface.co/pyannote/segmentation-3.0\n"
                    "3. Asegurarte de que tu token de Hugging Face tenga permisos de lectura (Read)."
                ) from diarize_err
            raise TranscriptionError(
                f"Error durante la diarización: {diarize_err}"
            ) from diarize_err

    def process(
        self,
        media_path: str | Path,
        output_dir: str | Path | None = None,
        progress_callback: ProgressCallback | None = None,
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
            notify(
                0.25, f"Transcribiendo con modelo '{self.config.model_name}' ({compute_type})..."
            )
            model = whisperx.load_model(
                self.config.model_name,
                device=device,
                compute_type=compute_type,
                language=self.config.language,
            )
            raw_result = model.transcribe(audio, batch_size=self.config.batch_size)
            detected_language = raw_result.get("language", "es")
            del model
            clear_gpu_vram()

            # --- FASE 3: Alineación Fonética de Marcas de Tiempo ---
            notify(0.55, "Alineando marcas de tiempo a nivel de palabra...")
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_language, device=device
            )
            aligned_result = whisperx.align(
                raw_result["segments"],
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            del align_model
            clear_gpu_vram()

            # --- FASE 4: Diarización de Hablantes (Opcional) ---
            segments: list[dict[str, Any]] = aligned_result["segments"]
            if self.config.enable_diarization:
                notify(0.75, "Identificando interlocutores (Diarización con Pyannote)...")
                segments = self._run_diarization(audio, aligned_result, device)

            # --- FASE 5: Generación y Escritura del Archivo Final ---
            notify(0.90, "Generando archivo de transcripción...")
            formatted_text = render_segments_text(
                segments=segments,
                mode=self.config.timestamp_mode,
                include_speaker=self.config.enable_diarization,
            )
            output_file = save_transcription_file(
                formatted_text=formatted_text,
                input_path=input_path,
                output_dir=output_dir,
            )

            notify(1.0, f"Transcripción completada con éxito en: {output_file.name}")

            return TranscriptionResult(
                raw_segments=segments,
                detected_language=detected_language,
                formatted_text=formatted_text,
                output_path=output_file,
            )

        except TranscriptionError:
            clear_gpu_vram()
            raise
        except Exception as err:
            clear_gpu_vram()
            raise TranscriptionError(f"Error durante el procesamiento: {err}") from err


def create_transcription_pipeline(config: TranscriptionConfig) -> TranscriptionPipeline:
    """
    Fábrica unificada de pipelines.
    Retorna la instancia correspondiente al proveedor configurado.
    """
    if config.provider == ProviderType.DEEPGRAM.value:
        from src.cloud_providers import DeepgramPipeline

        return DeepgramPipeline(config)
    if config.provider == ProviderType.ASSEMBLYAI.value:
        from src.cloud_providers import AssemblyAIPipeline

        return AssemblyAIPipeline(config)
    return WhisperXPipeline(config)
