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

    NOVA_2 = "nova-2"  # Recomendado: ultrarrápido y coste óptimo
    NOVA_3 = "nova-3"  # Nueva generación: máxima precisión contextual


class AssemblyAIModel(str, Enum):
    """Modelos de voz de AssemblyAI."""

    BEST = "best"  # Universal-1: máxima precisión acústica
    NANO = "nano"  # Ligero: procesamiento rápido para pruebas


class ComputeType(str, Enum):
    """Tipos de computación para cuantización de VRAM."""

    FLOAT16 = "float16"
    INT8 = "int8"
    FLOAT32 = "float32"


class TimestampMode(str, Enum):
    """Formatos de visualización para marcas de tiempo."""

    SIMPLE = "simple"  # [HH:MM:SS]
    RANGE = "range"  # [HH:MM:SS - HH:MM:SS]
    NONE = "none"  # Sin marcas de tiempo


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

    provider: str = ProviderType.DEEPGRAM.value
    model_name: str = DeepgramModel.NOVA_2.value
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
        provider: str = ProviderType.DEEPGRAM.value,
        model_name: str | None = None,
        enable_diarization: bool = False,
        timestamp_mode: TimestampMode = TimestampMode.SIMPLE,
        hf_token: str | None = None,
        api_key: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
    ) -> "TranscriptionConfig":
        """Crea una configuración inmutable con selección automática de hardware y proveedor."""
        detected_device = "cpu"
        detected_compute = ComputeType.INT8.value

        # Solo verificar PyTorch CUDA si se utiliza WhisperX
        if provider == ProviderType.WHISPERX.value:
            try:
                import torch

                detected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                detected_compute = compute_type or ComputeType.INT8.value
            except ImportError:
                detected_device = "cpu"
                detected_compute = ComputeType.INT8.value

        resolved_model = model_name or get_default_model_for_provider(provider)

        return cls(
            provider=provider,
            model_name=resolved_model,
            device=detected_device,
            compute_type=detected_compute,
            enable_diarization=enable_diarization,
            timestamp_mode=timestamp_mode,
            hf_token=hf_token,
            api_key=api_key,
            language=language,
        )


DEFAULT_PROVIDER_MODELS: dict[str, str] = {
    ProviderType.WHISPERX.value: ModelName.LARGE_V3_TURBO.value,
    ProviderType.DEEPGRAM.value: DeepgramModel.NOVA_2.value,
    ProviderType.ASSEMBLYAI.value: AssemblyAIModel.BEST.value,
}


def get_default_model_for_provider(provider: str) -> str:
    """Retorna el identificador del modelo predeterminado para el proveedor indicado."""
    return DEFAULT_PROVIDER_MODELS.get(provider, ModelName.LARGE_V3_TURBO.value)


TIMESTAMP_LABEL_MAP: dict[str, TimestampMode] = {
    "Simple ([HH:MM:SS])": TimestampMode.SIMPLE,
    "Rango ([Inicio - Fin])": TimestampMode.RANGE,
    "Texto continuo (Sin marcas)": TimestampMode.NONE,
}

TIMESTAMP_CHOICES: list[str] = list(TIMESTAMP_LABEL_MAP.keys())


PROVIDER_MODEL_CHOICES: dict[str, list[tuple[str, str]]] = {
    ProviderType.WHISPERX.value: [
        (
            "large-v3-turbo  ·  [Recomendado · Rápido & Gratuito en GPU]",
            ModelName.LARGE_V3_TURBO.value,
        ),
        ("large-v3  ·  [Alta Capacidad · Mayor Tiempo GPU]", ModelName.LARGE_V3.value),
        ("medium  ·  [Equilibrado · Menor VRAM]", ModelName.MEDIUM.value),
        ("small  ·  [Ligero · Inferencia Rápida]", ModelName.SMALL.value),
        ("base  ·  [Ultraligero · Básico]", ModelName.BASE.value),
    ],
    ProviderType.DEEPGRAM.value: [
        ("nova-2  ·  [Recomendado · Ultrarrápido & Más Económico]", DeepgramModel.NOVA_2.value),
        ("nova-3  ·  [Última Gen · Robusto en Ruido & Acentos]", DeepgramModel.NOVA_3.value),
    ],
    ProviderType.ASSEMBLYAI.value: [
        ("best  ·  [Universal-1 · Máxima Fidelidad Editorial]", AssemblyAIModel.BEST.value),
        ("nano  ·  [Ultraligero · Mayor Ahorro de Créditos]", AssemblyAIModel.NANO.value),
    ],
}

PROVIDER_MODELS: dict[str, list[str]] = {
    provider: [val for _, val in choices] for provider, choices in PROVIDER_MODEL_CHOICES.items()
}
