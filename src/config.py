"""
Configuración centralizada y modelos de datos para WhisperX Transcriptor.
Aplica principios de Clean Code con inmutabilidad y validación de tipos.
"""

from dataclasses import dataclass
from enum import Enum


class ProviderType(str, Enum):
    """Proveedores de transcripción soportados."""
    WHISPERX = "WhisperX (ZeroGPU / Local)"
    DEEPGRAM = "Deepgram (Cloud API)"
    ASSEMBLYAI = "AssemblyAI (Cloud API)"


class ModelName(str, Enum):
    """Modelos Whisper soportados."""
    LARGE_V3_TURBO = "large-v3-turbo"
    LARGE_V3 = "large-v3"
    MEDIUM = "medium"
    SMALL = "small"
    BASE = "base"


class DeepgramModel(str, Enum):
    """Modelos de voz de Deepgram."""
    NOVA_2 = "nova-2"       # Recomendado: ultrarrápido y coste óptimo
    NOVA_3 = "nova-3"       # Nueva generación: máxima precisión contextual


class AssemblyAIModel(str, Enum):
    """Modelos de voz de AssemblyAI."""
    BEST = "best"           # Universal-1: máxima precisión acústica
    NANO = "nano"           # Ligero: procesamiento rápido para pruebas


class ComputeType(str, Enum):
    """Tipos de computación para cuantización de VRAM."""
    FLOAT16 = "float16"
    INT8 = "int8"
    FLOAT32 = "float32"


class TimestampMode(str, Enum):
    """Formatos de visualización para marcas de tiempo."""
    SIMPLE = "simple"       # [HH:MM:SS]
    RANGE = "range"         # [HH:MM:SS - HH:MM:SS]
    NONE = "none"           # Sin marcas de tiempo


@dataclass(frozen=True)
class TranscriptionConfig:
    """
    Configuración inmutable para la ejecución del pipeline de transcripción.

    Attributes:
        provider: Proveedor del motor ('WhisperX', 'Deepgram' o 'AssemblyAI').
        model_name: Nombre o identificador del modelo del proveedor.
        device: Dispositivo de cómputo para WhisperX ('cuda' o 'cpu').
        compute_type: Precisión o cuantización ('float16', 'int8', 'float32').
        batch_size: Tamaño de lote para la inferencia de WhisperX.
        enable_diarization: Si se debe ejecutar la diarización de hablantes.
        timestamp_mode: Formato de las marcas de tiempo en la salida.
        hf_token: Token de Hugging Face para pyannote (si provider=WhisperX).
        api_key: Clave de API para proveedores en la nube (Deepgram / AssemblyAI).
        language: Código de idioma ISO (opcional, detección automática por defecto).
    """
    provider: str = ProviderType.WHISPERX.value
    model_name: str = ModelName.LARGE_V3_TURBO.value
    device: str = "cuda"
    compute_type: str = ComputeType.INT8.value
    batch_size: int = 16
    enable_diarization: bool = False
    timestamp_mode: TimestampMode = TimestampMode.SIMPLE
    hf_token: str | None = None
    api_key: str | None = None
    language: str | None = None

    @classmethod
    def create_default(
        cls,
        provider: str = ProviderType.WHISPERX.value,
        model_name: str = ModelName.LARGE_V3_TURBO.value,
        enable_diarization: bool = False,
        hf_token: str | None = None,
        api_key: str | None = None,
        device: str | None = None,
        compute_type: str | None = None
    ) -> "TranscriptionConfig":
        """Crea una configuración con selección automática de hardware y proveedor."""
        detected_device = "cpu"
        detected_compute = "int8"

        # Solo verificar PyTorch CUDA si se utiliza WhisperX
        if provider == ProviderType.WHISPERX.value:
            try:
                import torch
                detected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                detected_compute = compute_type or ("int8" if detected_device == "cuda" else "int8")
            except ImportError:
                detected_device = "cpu"
                detected_compute = "int8"

        return cls(
            provider=provider,
            model_name=model_name,
            device=detected_device,
            compute_type=detected_compute,
            enable_diarization=enable_diarization,
            hf_token=hf_token,
            api_key=api_key,
        )


PROVIDER_MODELS: dict[str, list[str]] = {
    ProviderType.WHISPERX.value: [
        ModelName.LARGE_V3_TURBO.value,
        ModelName.LARGE_V3.value,
        ModelName.MEDIUM.value,
        ModelName.SMALL.value,
        ModelName.BASE.value,
    ],
    ProviderType.DEEPGRAM.value: [
        DeepgramModel.NOVA_2.value,
        DeepgramModel.NOVA_3.value,
    ],
    ProviderType.ASSEMBLYAI.value: [
        AssemblyAIModel.BEST.value,
        AssemblyAIModel.NANO.value,
    ],
}

