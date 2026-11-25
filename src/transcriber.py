"""
Orquestador del pipeline WhisperX y contrato base de transcripción.
Aplica principios de Clean Code, SOLID (DIP/SRP) y gestión óptima de VRAM.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.config import ComputeType, ProviderType, TranscriptionConfig
from src.utils import (
    MissingTokenError,
    TranscriptionError,
    clear_gpu_vram,
    make_progress_notifier,
    render_segments_text,
    resolve_hf_token,
    save_transcription_file,
    setup_logger,
    smooth_progress_stage,
    validate_media_path,
)

logger = setup_logger("whisperx_pipeline")

ProgressCallback = Callable[[float, str], None]


def _is_cuda_oom_or_backend_error(err: Exception) -> bool:
    """Detecta errores de memoria CUDA (OOM) o incompatibilidad de backend/cuantización en GPU."""
    if isinstance(err, MemoryError) or err.__class__.__name__ == "OutOfMemoryError":
        return True
    err_str = str(err).lower()
    return any(
        term in err_str
        for term in (
            "out of memory",
            "cuda",
            "cublas",
            "cudnn",
            "device-side assert",
            "float16",
            "compute type",
        )
    )


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

    def _resolve_execution_device(self) -> tuple[str, str]:
        """Resuelve el dispositivo y compute_type efectivo dentro del contexto actual (incl. ZeroGPU)."""
        device = self.config.device
        if self.config.device_auto:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        compute_type = self.config.compute_type
        if device == "cpu" and compute_type == ComputeType.FLOAT16.value:
            compute_type = ComputeType.INT8.value
        return device, compute_type

    def _transcribe_with_fallback(
        self,
        whisperx_mod: Any,
        audio: Any,
        initial_device: str,
        initial_compute_type: str,
        notify: Callable[[float, str], None],
    ) -> tuple[dict[str, Any], str]:
        """Ejecuta Fase 2 (ASR) reduciendo batch_size ante CUDA OOM y degradando a CPU si es necesario."""
        device = initial_device
        compute_type = (
            ComputeType.INT8.value
            if device == "cpu" and initial_compute_type == ComputeType.FLOAT16.value
            else initial_compute_type
        )

        if device == "cuda":
            batch_candidates = [self.config.batch_size] + [
                b for b in (4, 1) if b < self.config.batch_size
            ]
            compute_candidates = (
                [compute_type, ComputeType.INT8.value]
                if compute_type != ComputeType.INT8.value
                else [compute_type]
            )

            for curr_compute in compute_candidates:
                for batch_idx, curr_batch in enumerate(batch_candidates):
                    model = None
                    in_transcribe = False
                    try:
                        model = whisperx_mod.load_model(
                            self.config.model_name,
                            device="cuda",
                            compute_type=curr_compute,
                            language=self.config.language,
                        )
                        in_transcribe = True
                        raw_result: dict[str, Any] = model.transcribe(audio, batch_size=curr_batch)
                        return raw_result, "cuda"
                    except Exception as err:
                        if not _is_cuda_oom_or_backend_error(err):
                            raise
                        model = None
                        clear_gpu_vram()
                        if not in_transcribe:
                            logger.warning(
                                "Error CUDA cargando modelo con compute_type=%s: %s",
                                curr_compute,
                                err,
                            )
                            break
                        next_batch = (
                            batch_candidates[batch_idx + 1]
                            if batch_idx + 1 < len(batch_candidates)
                            else None
                        )
                        if next_batch is not None:
                            logger.warning(
                                "CUDA OOM con batch_size=%d (%s); reintentando con batch_size=%d...",
                                curr_batch,
                                err,
                                next_batch,
                            )
                            notify(
                                0.22,
                                f"VRAM limitada; reintentando con lote reducido ({next_batch})...",
                            )
                        elif curr_compute != ComputeType.INT8.value:
                            logger.warning(
                                "Fallo CUDA con compute_type=%s (%s); reintentando en GPU con int8...",
                                curr_compute,
                                err,
                            )
                            notify(0.22, "Ajustando precisión GPU a int8 por memoria limitada...")
                    finally:
                        model = None
                        clear_gpu_vram()

            logger.warning(
                "No fue posible completar ASR en CUDA tras reducir batch_size; degradando a CPU (int8)."
            )
            notify(0.24, "GPU sin memoria suficiente; continuando transcripción en CPU (int8)...")
            device = "cpu"
            compute_type = ComputeType.INT8.value

        cpu_batch = (
            min(self.config.batch_size, 4)
            if initial_device == "cuda"
            else self.config.batch_size
        )
        model = None
        try:
            model = whisperx_mod.load_model(
                self.config.model_name,
                device=device,
                compute_type=compute_type,
                language=self.config.language,
            )
            raw_result = model.transcribe(audio, batch_size=cpu_batch)
            return raw_result, device
        finally:
            model = None
            clear_gpu_vram()

    def _run_diarization(
        self,
        audio: Any,
        aligned_result: dict[str, Any],
        device: str,
    ) -> list[dict[str, Any]]:
        """Ejecuta el modelo de diarización Pyannote y asigna hablantes por palabra."""
        diarize_model = None
        diarize_segments = None
        try:
            from whisperx.diarize import DiarizationPipeline, assign_word_speakers

            hf_token = resolve_hf_token(self.config.hf_token)
            devices_to_try = [device, "cpu"] if device == "cuda" else [device]

            for idx, target_device in enumerate(devices_to_try):
                try:
                    try:
                        diarize_model = DiarizationPipeline(token=hf_token, device=target_device)
                    except TypeError:
                        diarize_model = DiarizationPipeline(
                            use_auth_token=hf_token, device=target_device
                        )

                    diarize_segments = diarize_model(audio)
                    diarized_result = assign_word_speakers(diarize_segments, aligned_result)
                    segments: list[dict[str, Any]] = diarized_result["segments"]
                    return segments
                except Exception as diarize_err:
                    diarize_model = None
                    diarize_segments = None
                    if (
                        idx == 0
                        and target_device == "cuda"
                        and _is_cuda_oom_or_backend_error(diarize_err)
                    ):
                        logger.warning(
                            "Fallo CUDA/VRAM durante la diarización (%s); reintentando en CPU...",
                            diarize_err,
                        )
                        clear_gpu_vram()
                        continue
                    raise

            raise TranscriptionError("No se pudo ejecutar la diarización.")
        except TranscriptionError:
            raise
        except Exception as diarize_err:
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
        finally:
            diarize_model = None
            diarize_segments = None
            clear_gpu_vram()

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
        input_path = validate_media_path(media_path)

        notify = make_progress_notifier(logger, progress_callback)

        try:
            import whisperx
        except ImportError as err:
            raise TranscriptionError("WhisperX no está instalado en el entorno actual.") from err

        device, compute_type = self._resolve_execution_device()

        audio = raw_result = aligned_result = align_model = metadata = None
        try:
            # --- FASE 1: Carga y Extracción de Audio ---
            with smooth_progress_stage(
                notify,
                start_p=0.05,
                end_p=0.18,
                message=f"Cargando archivo multimedia: {input_path.name}...",
                tau_seconds=3.0,
            ):
                audio = whisperx.load_audio(str(input_path))

            # --- FASE 2: Transcripción ASR con WhisperX ---
            asr_end_p = 0.52 if self.config.enable_diarization else 0.62
            with smooth_progress_stage(
                notify,
                start_p=0.18,
                end_p=asr_end_p,
                message=(
                    f"Transcribiendo con modelo '{self.config.model_name}' ({compute_type})..."
                ),
                tau_seconds=18.0,
            ):
                raw_result, device = self._transcribe_with_fallback(
                    whisperx, audio, device, compute_type, notify
                )
                detected_language = (
                    raw_result.get("language") or self.config.language or "unknown"
                )

            raw_segments: list[dict[str, Any]] = raw_result.get("segments") or []
            raw_result = None
            aligned_result = {"segments": raw_segments}

            # --- FASE 3: Alineación Fonética de Marcas de Tiempo ---
            align_end_p = 0.72 if self.config.enable_diarization else 0.88
            with smooth_progress_stage(
                notify,
                start_p=asr_end_p,
                end_p=align_end_p,
                message="Alineando marcas de tiempo a nivel de palabra...",
                tau_seconds=8.0,
            ):
                if raw_segments:
                    if str(detected_language).strip().lower() in {"unknown", ""}:
                        logger.warning(
                            "Idioma no detectado ('%s'); omitiendo alineación fonética.",
                            detected_language,
                        )
                        notify(
                            align_end_p,
                            "Idioma no compatible con alineación; conservando marcas base.",
                        )
                    else:
                        align_devices = [device, "cpu"] if device == "cuda" else [device]
                        for idx, align_device in enumerate(align_devices):
                            align_model = None
                            metadata = None
                            try:
                                align_model, metadata = whisperx.load_align_model(
                                    language_code=detected_language, device=align_device
                                )
                                aligned_result = whisperx.align(
                                    raw_segments,
                                    align_model,
                                    metadata,
                                    audio,
                                    align_device,
                                    return_char_alignments=False,
                                )
                                break
                            except Exception as align_err:
                                if (
                                    idx == 0
                                    and align_device == "cuda"
                                    and _is_cuda_oom_or_backend_error(align_err)
                                ):
                                    logger.warning(
                                        "Fallo CUDA en alineación (%s); reintentando en CPU...",
                                        align_err,
                                    )
                                    continue
                                logger.warning(
                                    "Alineación fonética omitida para idioma '%s': %s. "
                                    "Conservando marcas temporales base.",
                                    detected_language,
                                    align_err,
                                )
                                notify(
                                    align_end_p,
                                    "Alineación fonética omitida; conservando marcas de tiempo base.",
                                )
                                aligned_result = {"segments": raw_segments}
                                break
                            finally:
                                align_model = None
                                metadata = None
                                clear_gpu_vram()

            if not self.config.enable_diarization:
                audio = None

            # --- FASE 4: Diarización de Hablantes (Opcional) ---
            segments: list[dict[str, Any]] = aligned_result.get("segments") or []
            if self.config.enable_diarization and segments:
                with smooth_progress_stage(
                    notify,
                    start_p=0.72,
                    end_p=0.90,
                    message="Identificando interlocutores (Diarización con Pyannote)...",
                    tau_seconds=14.0,
                ):
                    segments = self._run_diarization(audio, aligned_result, device)

            audio = None
            aligned_result = None

            # --- FASE 5: Generación y Escritura del Archivo Final ---
            notify(0.94, "Generando archivo de transcripción...")
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
            raise
        except Exception as err:
            raise TranscriptionError(f"Error durante el procesamiento: {err}") from err
        finally:
            align_model = metadata = raw_result = aligned_result = audio = None
            clear_gpu_vram()


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
    if config.provider == ProviderType.WHISPERX.value:
        return WhisperXPipeline(config)
    raise TranscriptionError(f"Proveedor no soportado: {config.provider}")
