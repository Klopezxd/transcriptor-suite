#!/usr/bin/env python3
"""
Aplicación Web Gradio para WhisperX Transcriptor.
Diseñada para despliegue en Hugging Face Spaces (ZeroGPU) y ejecución local.
"""

import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import gradio as gr

from src.config import (
    PROVIDER_MODEL_CHOICES,
    ModelName,
    ProviderType,
    TimestampMode,
    TranscriptionConfig,
)
from src.transcriber import create_transcription_pipeline
from src.utils import (
    TranscriptionError,
    clear_gpu_vram,
    resolve_assemblyai_api_key,
    resolve_deepgram_api_key,
    resolve_hf_token,
    setup_logger,
    verify_admin_pin,
)

logger = setup_logger("whisperx_app")

# Compatibilidad con ZeroGPU en Hugging Face Spaces (duración extendida para audios largos)
try:
    import spaces

    GPU_DECORATOR = spaces.GPU(duration=180)
except (ImportError, TypeError, AttributeError):

    def GPU_DECORATOR(func):
        return func


@GPU_DECORATOR
def execute_whisperx(pipeline, media_path: str, progress_callback):
    """Ejecuta el pipeline local de WhisperX bajo asignación de GPU."""
    return pipeline.process(media_path=media_path, progress_callback=progress_callback)


def process_transcription(
    media_file,
    provider: str,
    model_name: str,
    enable_diarization: bool,
    timestamp_option: str,
    hf_token_input: str = "",
    deepgram_key_input: str = "",
    assemblyai_key_input: str = "",
    is_admin_session: bool = False,
    progress=gr.Progress(),
):
    """Procesa el archivo multimedia según el proveedor seleccionado y retorna texto y descarga."""
    if not media_file:
        return (
            "⚠️ Por favor selecciona o arrastra un archivo de audio o video.",
            gr.DownloadButton(label="Descargar Documento (.txt)", value=None, interactive=False),
        )

    media_path = media_file.name if hasattr(media_file, "name") else str(media_file)

    # Mapeo de formato de marcas de tiempo
    mode_map = {
        "Simple ([HH:MM:SS])": TimestampMode.SIMPLE,
        "Rango ([Inicio - Fin])": TimestampMode.RANGE,
        "Texto continuo (Sin marcas)": TimestampMode.NONE,
    }
    selected_mode = mode_map.get(timestamp_option, TimestampMode.SIMPLE)

    # Validación y resolución de credenciales según el motor elegido
    resolved_hf_token = None
    resolved_api_key = None

    if provider == ProviderType.WHISPERX.value:
        resolved_hf_token = resolve_hf_token(hf_token_input)
        if enable_diarization and not resolved_hf_token:
            return (
                "⚠️ La diarización con WhisperX requiere un token de Hugging Face.\n\n"
                "Por favor ingrésalo en el campo 'Token de Hugging Face' abajo o configúralo como 'HF_TOKEN' en los Secrets del Space.\n\n"
                "Requisitos en Hugging Face:\n"
                "1. Aceptar condiciones en: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "2. Aceptar condiciones en: https://huggingface.co/pyannote/segmentation-3.0\n"
                "3. Crear token con permisos 'Read' en: https://huggingface.co/settings/tokens",
                gr.DownloadButton(
                    label="Descargar Documento (.txt)", value=None, interactive=False
                ),
            )

    elif provider == ProviderType.DEEPGRAM.value:
        # Prioridad: Si ingresó clave en la interfaz, se respeta. Si no, solo se usa la clave del servidor si es administrador.
        if deepgram_key_input and deepgram_key_input.strip():
            resolved_api_key = resolve_deepgram_api_key(deepgram_key_input)
        elif is_admin_session:
            resolved_api_key = resolve_deepgram_api_key()
        else:
            resolved_api_key = None

        if not resolved_api_key:
            return (
                "⚠️ Se requiere una clave de API de Deepgram para utilizar este motor.\n\n"
                "• Si tienes tu propia clave, ingrésala en el acordeón 'Credenciales de Deepgram Cloud API'.\n"
                "• Si eres el propietario, desbloquea el 'Acceso Propietario / Modo Administrador' arriba para autorizar tus claves maestras.\n"
                "• Para transcribir de forma 100% gratuita sin necesidad de claves externas, selecciona 'WhisperX (ZeroGPU / Local)'.",
                gr.DownloadButton(
                    label="Descargar Documento (.txt)", value=None, interactive=False
                ),
            )

    elif provider == ProviderType.ASSEMBLYAI.value:
        # Prioridad: Si ingresó clave en la interfaz, se respeta. Si no, solo se usa la clave del servidor si es administrador.
        if assemblyai_key_input and assemblyai_key_input.strip():
            resolved_api_key = resolve_assemblyai_api_key(assemblyai_key_input)
        elif is_admin_session:
            resolved_api_key = resolve_assemblyai_api_key()
        else:
            resolved_api_key = None

        if not resolved_api_key:
            return (
                "⚠️ Se requiere una clave de API de AssemblyAI para utilizar este motor.\n\n"
                "• Si tienes tu propia clave, ingrésala en el acordeón 'Credenciales de AssemblyAI Cloud API'.\n"
                "• Si eres el propietario, desbloquea el 'Acceso Propietario / Modo Administrador' arriba para autorizar tus claves maestras.\n"
                "• Para transcribir de forma 100% gratuita sin necesidad de claves externas, selecciona 'WhisperX (ZeroGPU / Local)'.",
                gr.DownloadButton(
                    label="Descargar Documento (.txt)", value=None, interactive=False
                ),
            )

    def on_progress(p: float, msg: str) -> None:
        progress(p, desc=msg)

    try:
        config = TranscriptionConfig.create_default(
            provider=provider,
            model_name=model_name,
            enable_diarization=enable_diarization,
            hf_token=resolved_hf_token,
            api_key=resolved_api_key,
        )
        object.__setattr__(config, "timestamp_mode", selected_mode)

        pipeline = create_transcription_pipeline(config)

        # Si es WhisperX, ejecutar en el runner de ZeroGPU; si es API Cloud, ejecutar directamente
        if provider == ProviderType.WHISPERX.value:
            result = execute_whisperx(pipeline, media_path, on_progress)
        else:
            result = pipeline.process(
                media_path=media_path,
                progress_callback=on_progress,
            )

        return (
            result.formatted_text,
            gr.DownloadButton(
                label=f"Descargar {result.output_path.name}",
                value=str(result.output_path),
                interactive=True,
            ),
        )

    except TranscriptionError as e:
        clear_gpu_vram()
        logger.error("Error en procesamiento web: %s", e)
        return (
            f"❌ Error de transcripción: {e}",
            gr.DownloadButton(label="Descargar Documento (.txt)", value=None, interactive=False),
        )
    except Exception as e:
        clear_gpu_vram()
        logger.exception("Error inesperado en app web")
        return (
            f"❌ Ocurrió un error inesperado durante el procesamiento: {e}",
            gr.DownloadButton(label="Descargar Documento (.txt)", value=None, interactive=False),
        )


# --- Sistema de Diseño: Aurora Studio Enterprise ---
# Inspirado en diseño nórdico minimalista, noche ártica profunda (#030712), Aurora Green (#00FF87), Cyan Glacial (#00D4FF) y Amethyst (#A855F7)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --aurora-green: #00FF87;
    --ice-cyan: #00D4FF;
    --sky-cyan: #38BDF8;
    --amethyst: #A855F7;
    --deep-midnight: #030712;
    --surface-slate: rgba(15, 23, 42, 0.75);
    --surface-darker: rgba(7, 12, 24, 0.88);
    --glass-border: rgba(56, 189, 248, 0.16);
    --glass-border-focus: rgba(0, 255, 135, 0.45);
    --frost-white: #F8FAFC;
    --muted-slate: #94A3B8;
    --dark-text: #030712;
}

/* Atmósfera de Noche Ártica con Ondas de Aurora Boreal */
body, gradio-app {
    background-color: var(--deep-midnight) !important;
    background-image: 
        radial-gradient(circle at 15% 12%, rgba(0, 255, 135, 0.07) 0%, transparent 40%),
        radial-gradient(circle at 85% 10%, rgba(168, 85, 247, 0.09) 0%, transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(0, 212, 255, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 80% 85%, rgba(0, 255, 135, 0.04) 0%, transparent 35%) !important;
    background-attachment: fixed !important;
    color: var(--frost-white) !important;
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

.gradio-container {
    max-width: 1400px !important;
    width: 96% !important;
    margin: 0 auto !important;
    padding-top: 1.25rem !important;
    padding-bottom: 2.5rem !important;
}

/* Tarjetas y Paneles Frosted Glass */
.gradio-container .block,
.gradio-container .panel,
.gradio-container .form {
    background: var(--surface-slate) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45) !important;
}

/* Cabecera Estilo Nórdico */
#app-header {
    text-align: center;
    padding: 1.75rem 1rem 1.25rem 1rem;
    position: relative;
    margin-bottom: 0.75rem;
}

.aurora-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.4rem 1.05rem;
    border-radius: 9999px;
    background: rgba(0, 255, 135, 0.09);
    border: 1px solid rgba(0, 255, 135, 0.32);
    color: var(--aurora-green);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.85rem;
    box-shadow: 0 2px 12px rgba(0, 255, 135, 0.15);
}

.aurora-badge-dot {
    width: 7.5px;
    height: 7.5px;
    border-radius: 50%;
    background-color: var(--aurora-green);
    box-shadow: 0 0 10px var(--aurora-green);
    animation: aurora-pulse 2.2s infinite ease-in-out;
}

@keyframes aurora-pulse {
    0%, 100% { opacity: 0.5; transform: scale(0.9); }
    50% { opacity: 1; transform: scale(1.25); }
}

.aurora-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.035em;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(135deg, #FFFFFF 20%, var(--aurora-green) 70%, var(--ice-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.85rem;
}

.header-mic-svg {
    stroke: var(--aurora-green);
    filter: drop-shadow(0 0 14px rgba(0, 255, 135, 0.55));
}

.aurora-subtitle {
    color: var(--muted-slate);
    font-size: 1.02rem;
    max-width: 720px;
    margin: 0 auto 0.9rem auto;
    line-height: 1.55;
    font-weight: 400;
}

.header-engine-chips {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 0.5rem;
}

.engine-chip {
    font-size: 0.74rem;
    font-weight: 600;
    padding: 3.5px 11px;
    border-radius: 9999px;
    letter-spacing: 0.02em;
}

.engine-chip-wx {
    background: rgba(0, 255, 135, 0.1);
    color: var(--aurora-green);
    border: 1px solid rgba(0, 255, 135, 0.28);
}

.engine-chip-dg {
    background: rgba(0, 212, 255, 0.1);
    color: var(--ice-cyan);
    border: 1px solid rgba(0, 212, 255, 0.28);
}

.engine-chip-aai {
    background: rgba(168, 85, 247, 0.12);
    color: #D8B4FE;
    border: 1px solid rgba(168, 85, 247, 0.32);
}

/* Panel Header Card Ribbons */
.panel-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 11px 16px;
    margin-bottom: 14px;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
}

.panel-header-icon {
    font-size: 1.35rem;
    line-height: 1;
}

.panel-header-title {
    font-size: 0.96rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -0.01em;
}

.panel-header-desc {
    font-size: 0.77rem;
    color: var(--muted-slate);
    margin: 2px 0 0 0;
}

/* Segmented Engine Selector Radio */
.engine-tabs-selector fieldset {
    display: flex !important;
    flex-direction: row !important;
    gap: 8px !important;
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
    flex-wrap: wrap !important;
}

.engine-tabs-selector label {
    flex: 1 1 160px !important;
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.22) !important;
    border-radius: 10px !important;
    padding: 11px 12px !important;
    cursor: pointer !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    color: #E2E8F0 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
}

.engine-tabs-selector label:hover {
    border-color: var(--ice-cyan) !important;
    background: rgba(0, 212, 255, 0.12) !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
}

.engine-tabs-selector label.selected,
.engine-tabs-selector label:has(input:checked) {
    border-color: var(--aurora-green) !important;
    background: rgba(0, 255, 135, 0.14) !important;
    box-shadow: 0 0 16px rgba(0, 255, 135, 0.25) !important;
    color: var(--aurora-green) !important;
}

/* Zona de Carga de Archivos */
.gradio-container .file-upload,
.gradio-container [data-testid="file-upload"] {
    border: 1.5px dashed rgba(56, 189, 248, 0.35) !important;
    border-radius: 12px !important;
    background: rgba(8, 14, 26, 0.7) !important;
    transition: all 0.25s ease !important;
    padding: 1.25rem !important;
}

.gradio-container .file-upload:hover,
.gradio-container [data-testid="file-upload"]:hover {
    border-color: var(--aurora-green) !important;
    background: rgba(0, 255, 135, 0.05) !important;
    box-shadow: 0 0 20px rgba(0, 255, 135, 0.18) !important;
}

.format-badges-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    margin-top: 6px;
    margin-bottom: 12px;
}

.format-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2.5px 7px;
    border-radius: 5px;
    background: rgba(56, 189, 248, 0.1);
    color: var(--sky-cyan);
    border: 1px solid rgba(56, 189, 248, 0.25);
}

.format-chip-note {
    font-size: 0.74rem;
    color: var(--muted-slate);
    margin-left: 4px;
}

/* Inputs, Selectores y Foco */
.gradio-container input[type="text"],
.gradio-container input[type="password"],
.gradio-container select,
.gradio-container textarea {
    background-color: var(--surface-darker) !important;
    border: 1px solid rgba(56, 189, 248, 0.22) !important;
    border-radius: 10px !important;
    color: var(--frost-white) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.gradio-container input:focus,
.gradio-container select:focus,
.gradio-container textarea:focus {
    border-color: var(--aurora-green) !important;
    box-shadow: 0 0 0 2px rgba(0, 255, 135, 0.35) !important;
    outline: none !important;
}

/* Área de Salida: Tipografía Monospace JetBrains */
.output-transcript textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.92rem !important;
    line-height: 1.72 !important;
    background: #040914 !important;
    border: 1px solid rgba(56, 189, 248, 0.24) !important;
    border-radius: 10px !important;
    color: #F1F5F9 !important;
    padding: 1.1rem !important;
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.65) !important;
}

/* Botón Principal (Aurora Green + Cyan Gradient) */
.btn-primary, button.btn-primary, button.primary {
    background: linear-gradient(135deg, #00FF87 0%, #00D4FF 100%) !important;
    color: var(--dark-text) !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.24s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(0, 255, 135, 0.35) !important;
    padding: 0.85rem 1.6rem !important;
    cursor: pointer !important;
}

.btn-primary:hover, button.btn-primary:hover, button.primary:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 6px 28px rgba(0, 255, 135, 0.55), 0 0 15px rgba(0, 212, 255, 0.4) !important;
    filter: brightness(1.06) !important;
}

.btn-primary:active, button.btn-primary:active, button.primary:active {
    transform: translateY(1px) scale(0.99) !important;
}

/* Botón Secundario (Limpiar) */
.btn-secondary, button.btn-secondary, button.secondary {
    background: rgba(15, 23, 42, 0.85) !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.btn-secondary:hover, button.btn-secondary:hover, button.secondary:hover {
    background: rgba(56, 189, 248, 0.15) !important;
    border-color: var(--sky-cyan) !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
}

/* Botón de Descarga */
.btn-download, button.btn-download {
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: 1px solid rgba(56, 189, 248, 0.32) !important;
    background: rgba(15, 23, 42, 0.85) !important;
    color: var(--frost-white) !important;
    transition: all 0.22s ease !important;
}

.btn-download:hover:not(:disabled), button.btn-download:hover:not(:disabled) {
    border-color: var(--ice-cyan) !important;
    background: rgba(0, 212, 255, 0.18) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 18px rgba(0, 212, 255, 0.3) !important;
    transform: translateY(-1px) !important;
}

/* Barras de Desplazamiento Personalizadas */
::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}
::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb {
    background: rgba(56, 189, 248, 0.35);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--aurora-green);
}

/* Acordeones Frosted Glass */
.gradio-container .accordion {
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    background: rgba(10, 22, 40, 0.55) !important;
    margin-bottom: 0.75rem !important;
}

/* Pie de Página Nórdico */
#app-footer {
    text-align: center;
    margin-top: 2.5rem;
    margin-bottom: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(0, 212, 255, 0.12);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.85rem;
    color: var(--muted-slate);
}

.footer-tech {
    letter-spacing: 0.02em;
}

.footer-tech strong {
    color: var(--frost-white);
    font-weight: 600;
}

.footer-links a {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--ice-cyan);
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s ease;
}

.footer-links a:hover {
    color: var(--aurora-green);
}

/* --- Barra de Estado Inferior Estilo Nórdico (Inspirada en Preview Suite) --- */
.statusbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 9px 18px;
    background: rgba(3, 7, 18, 0.92);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 10px;
    font-size: 11.5px;
    color: var(--muted-slate);
    margin-top: 2rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
}

.statusbar-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.statusbar-stats {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #CBD5E1;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.statusbar-stats-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--aurora-green);
    box-shadow: 0 0 10px var(--aurora-green);
    display: inline-block;
}

.statusbar-right {
    display: flex;
    gap: 9px;
    align-items: center;
}

.author-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 212, 255, 0.1);
    padding: 3.5px 11px;
    border-radius: 6px;
    color: var(--ice-cyan);
    font-weight: 600;
    font-size: 11.5px;
    border: 1px solid rgba(0, 212, 255, 0.28);
    white-space: nowrap;
    transition: all 0.2s ease;
}

.author-badge:hover {
    background: rgba(0, 212, 255, 0.2);
    border-color: var(--ice-cyan);
    transform: translateY(-1px);
}

.author-badge svg {
    color: var(--ice-cyan);
    width: 13px;
    height: 13px;
}

.antigravity-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(168, 85, 247, 0.15);
    padding: 3.5px 10px;
    border-radius: 6px;
    color: #D8B4FE;
    font-size: 11.5px;
    font-weight: 600;
    border: 1px solid rgba(168, 85, 247, 0.35);
    transition: all 0.2s ease;
    white-space: nowrap;
    cursor: default;
}

.antigravity-badge:hover {
    background: rgba(168, 85, 247, 0.28);
    border-color: #C084FC;
    box-shadow: 0 0 12px rgba(192, 132, 252, 0.4);
    transform: translateY(-1px);
}

.antigravity-badge svg {
    color: #F472B6;
    width: 12px;
    height: 12px;
}

.version-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 3.5px 9px;
    border-radius: 6px;
    background: rgba(0, 255, 135, 0.12);
    color: var(--aurora-green);
    border: 1px solid rgba(0, 255, 135, 0.32);
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    white-space: nowrap;
    cursor: default;
}

.version-badge:hover {
    background: rgba(0, 255, 135, 0.25);
    border-color: var(--aurora-green);
    box-shadow: 0 0 12px rgba(0, 255, 135, 0.4);
    transform: translateY(-1px);
}

/* Modal Overlay para Login de Administrador */
.admin-modal-overlay:not(.hide):not([style*="display: none"]) {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(2, 6, 23, 0.82) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    z-index: 99999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 1.5rem !important;
}

.admin-modal-overlay.hide,
.admin-modal-overlay[style*="display: none"] {
    display: none !important;
}

.admin-modal-card {
    background: rgba(15, 23, 42, 0.96) !important;
    border: 1px solid rgba(0, 255, 135, 0.35) !important;
    border-radius: 16px !important;
    max-width: 440px !important;
    width: 100% !important;
    padding: 24px !important;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.85), 0 0 35px rgba(0, 255, 135, 0.15) !important;
    margin: auto !important;
}

.modal-header-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 14px !important;
}

.modal-title-box {
    display: flex;
    align-items: center;
    gap: 10px;
}

.modal-title-icon {
    font-size: 1.45rem;
}

.modal-title-text {
    font-size: 1.05rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0;
}

.modal-title-desc {
    font-size: 0.78rem;
    color: #94A3B8;
    margin: 2px 0 0 0;
}

.modal-close-btn {
    max-width: 34px !important;
    min-width: 34px !important;
    height: 34px !important;
    padding: 0 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    color: #94A3B8 !important;
    cursor: pointer !important;
}

.modal-close-btn:hover {
    color: #FFFFFF !important;
    background: rgba(244, 63, 94, 0.25) !important;
    border-color: #F43F5E !important;
}

.modal-pin-input input {
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.15em !important;
    font-size: 1.1rem !important;
    text-align: center !important;
    margin-top: 10px !important;
    margin-bottom: 12px !important;
}

.modal-submit-btn {
    width: 100% !important;
    margin-top: 4px !important;
}

/* Pill de Sesión Activa de Propietario */
.admin-active-pill {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 8px 20px;
    margin: 0 auto 1.25rem auto;
    max-width: 640px;
    background: rgba(0, 255, 135, 0.08);
    border: 1px solid rgba(0, 255, 135, 0.35);
    border-radius: 9999px;
    color: var(--aurora-green);
    font-size: 0.84rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    box-shadow: 0 2px 14px rgba(0, 255, 135, 0.15);
    animation: fadeInPill 0.35s ease-out;
}

.admin-active-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--aurora-green);
    box-shadow: 0 0 10px var(--aurora-green);
}

.admin-active-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(0, 255, 135, 0.16);
    border: 1px solid rgba(0, 255, 135, 0.3);
    color: #FFFFFF;
}

@keyframes fadeInPill {
    from { opacity: 0; transform: translateY(-5px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Botones Interactivos de la Barra de Estado */
.statusbar-left-col {
    display: flex !important;
    align-items: center !important;
}

.statusbar-right-row {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    gap: 8px !important;
}

.statusbar-btn {
    padding: 5px 13px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1px solid transparent !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    min-width: unset !important;
    height: auto !important;
    background: transparent !important;
}

.statusbar-github-btn {
    background: rgba(56, 189, 248, 0.1) !important;
    border-color: rgba(56, 189, 248, 0.25) !important;
    color: var(--sky-cyan) !important;
}

.statusbar-github-btn:hover {
    background: rgba(56, 189, 248, 0.22) !important;
    border-color: var(--sky-cyan) !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
}

.statusbar-author-btn {
    background: rgba(0, 212, 255, 0.1) !important;
    border-color: rgba(0, 212, 255, 0.28) !important;
    color: var(--ice-cyan) !important;
}

.statusbar-author-btn:hover {
    background: rgba(0, 255, 135, 0.18) !important;
    border-color: var(--aurora-green) !important;
    color: var(--aurora-green) !important;
    box-shadow: 0 0 12px rgba(0, 255, 135, 0.35) !important;
    transform: translateY(-1px) !important;
}

.statusbar-version-btn {
    font-family: 'JetBrains Mono', monospace !important;
    background: rgba(0, 255, 135, 0.1) !important;
    border-color: rgba(0, 255, 135, 0.28) !important;
    color: var(--aurora-green) !important;
    cursor: default !important;
}

/* Adaptabilidad Móvil */
@media (max-width: 768px) {
    .gradio-container {
        width: 100% !important;
        padding: 0.75rem !important;
    }
    .aurora-title {
        font-size: 1.85rem;
    }
    .aurora-subtitle {
        font-size: 0.92rem;
    }
    .engine-tabs-selector label {
        flex: 1 1 100% !important;
    }
    .statusbar {
        flex-direction: column !important;
        gap: 12px !important;
        padding: 12px 14px !important;
    }
    .statusbar-left-col, .statusbar-right-row {
        justify-content: center !important;
        flex-wrap: wrap !important;
        width: 100% !important;
    }
}
"""

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#030712",
    body_background_fill_dark="#030712",
    body_text_color="#F8FAFC",
    body_text_color_dark="#F8FAFC",
    body_text_color_subdued="#94A3B8",
    body_text_color_subdued_dark="#94A3B8",
    background_fill_primary="#0F172A",
    background_fill_primary_dark="#0F172A",
    background_fill_secondary="#080E1A",
    background_fill_secondary_dark="#080E1A",
    block_background_fill="rgba(15, 23, 42, 0.75)",
    block_background_fill_dark="rgba(15, 23, 42, 0.75)",
    block_border_color="rgba(56, 189, 248, 0.16)",
    block_border_color_dark="rgba(56, 189, 248, 0.16)",
    block_radius="10px",
    block_shadow="0 6px 24px rgba(0, 0, 0, 0.45)",
    block_title_text_color="#F8FAFC",
    block_title_text_color_dark="#F8FAFC",
    block_label_text_color="#94A3B8",
    block_label_text_color_dark="#94A3B8",
    border_color_primary="rgba(56, 189, 248, 0.16)",
    border_color_primary_dark="rgba(56, 189, 248, 0.16)",
    button_primary_background_fill="#00FF87",
    button_primary_background_fill_dark="#00FF87",
    button_primary_background_fill_hover="#00E67A",
    button_primary_background_fill_hover_dark="#00E67A",
    button_primary_text_color="#030712",
    button_primary_text_color_dark="#030712",
    button_secondary_background_fill="rgba(15, 23, 42, 0.85)",
    button_secondary_background_fill_dark="rgba(15, 23, 42, 0.85)",
    button_secondary_border_color="rgba(56, 189, 248, 0.25)",
    button_secondary_border_color_dark="rgba(56, 189, 248, 0.25)",
    button_secondary_text_color="#F8FAFC",
    button_secondary_text_color_dark="#F8FAFC",
    input_background_fill="#080E1A",
    input_background_fill_dark="#080E1A",
    input_border_color="rgba(56, 189, 248, 0.22)",
    input_border_color_dark="rgba(56, 189, 248, 0.22)",
    input_border_color_focus="#00FF87",
    input_border_color_focus_dark="#00FF87",
    checkbox_background_color_selected="#00FF87",
    checkbox_background_color_selected_dark="#00FF87",
    checkbox_border_color_selected="#00FF87",
    checkbox_border_color_selected_dark="#00FF87",
    checkbox_label_text_color="#F8FAFC",
    checkbox_label_text_color_dark="#F8FAFC",
    color_accent="#00D4FF",
    color_accent_soft="rgba(0, 212, 255, 0.15)",
)

# Compatibilidad entre Gradio < 6.0 y Gradio >= 6.0 (theme y css pasaron de Blocks a launch)
blocks_kwargs = {"title": "WhisperX Studio · Enterprise ASR Suite"}
launch_kwargs = {}

try:
    _gradio_major = int(gr.__version__.split(".")[0])
    if _gradio_major >= 6:
        launch_kwargs["theme"] = theme
        launch_kwargs["css"] = custom_css
    else:
        blocks_kwargs["theme"] = theme
        blocks_kwargs["css"] = custom_css
except Exception:
    blocks_kwargs["theme"] = theme
    blocks_kwargs["css"] = custom_css

with gr.Blocks(**blocks_kwargs) as demo:
    gr.HTML(
        """
        <div id="app-header">
            <div class="aurora-badge">
                <span class="aurora-badge-dot"></span>
                <span>WhisperX Studio · ZeroGPU & Cloud Hybrid Engine</span>
            </div>
            <h1 class="aurora-title">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#00FF87" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" class="header-mic-svg">
                    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" x2="12" y1="19" y2="22"/>
                </svg>
                <span>WhisperX Transcriptor</span>
            </h1>
            <p class="aurora-subtitle">
                Transcripción fonética neuronal de alta velocidad, alineación acústica palabra por palabra y diarización de interlocutores
            </p>
            <div class="header-engine-chips">
                <span class="engine-chip engine-chip-wx">⚡ WhisperX (ZeroGPU / Local CUDA)</span>
                <span class="engine-chip engine-chip-dg">⚡ Deepgram (Nova-2 / Nova-3)</span>
                <span class="engine-chip engine-chip-aai">⚡ AssemblyAI (Universal-1 / Conformer)</span>
            </div>
        </div>
        """
    )

    admin_session_state = gr.State(value=False)

    # Banner flotante minimalista de Estado Propietario (visible únicamente cuando se autentica)
    admin_status_banner = gr.HTML("", visible=False)

    # Ventana Modal emergente para Login de Administrador (invisible por defecto)
    with gr.Group(visible=False, elem_classes=["admin-modal-overlay"]) as admin_modal:
        with gr.Group(elem_classes=["admin-modal-card"]):
            with gr.Row(elem_classes=["modal-header-row"]):
                gr.HTML(
                    """
                    <div class="modal-title-box">
                        <span class="modal-title-icon">🔐</span>
                        <div>
                            <h4 class="modal-title-text">Acceso Propietario</h4>
                            <p class="modal-title-desc">Desbloquea credenciales maestras y simplifica la interfaz</p>
                        </div>
                    </div>
                    """
                )
                btn_close_modal = gr.Button(
                    "✕", elem_classes=["modal-close-btn"], variant="secondary"
                )

            admin_pin_input = gr.Textbox(
                label="PIN Maestro",
                placeholder="Ingresa tu PIN de 4 dígitos...",
                type="password",
                show_label=False,
                elem_classes=["modal-pin-input"],
            )

            btn_unlock_admin = gr.Button(
                "Desbloquear Modo Propietario",
                variant="primary",
                elem_classes=["modal-submit-btn"],
            )
            admin_modal_msg = gr.Markdown("", visible=False)

    with gr.Row(equal_height=False):
        # Columna Izquierda: Entrada y Controles
        with gr.Column(scale=1):
            gr.HTML(
                """
                <div class="panel-header">
                    <span class="panel-header-icon">🎙️</span>
                    <div>
                        <h3 class="panel-header-title">Ingesta y Configuración del Motor</h3>
                        <p class="panel-header-desc">Selecciona el motor acústico y carga tu archivo multimedia</p>
                    </div>
                </div>
                """
            )

            provider_selector = gr.Radio(
                choices=[
                    ProviderType.WHISPERX.value,
                    ProviderType.DEEPGRAM.value,
                    ProviderType.ASSEMBLYAI.value,
                ],
                value=ProviderType.WHISPERX.value,
                label="Motor de Transcripción",
                info="Elige entre aceleración ZeroGPU (Hugging Face) o procesamiento ultrarrápido en la nube.",
                elem_classes=["engine-tabs-selector"],
            )

            input_media = gr.File(
                label="Archivo Multimedia",
                file_types=["audio", "video"],
                file_count="single",
            )

            gr.HTML(
                """
                <div class="format-badges-row">
                    <span class="format-chip">MP4</span>
                    <span class="format-chip">MP3</span>
                    <span class="format-chip">WAV</span>
                    <span class="format-chip">M4A</span>
                    <span class="format-chip">FLAC</span>
                    <span class="format-chip">OGG</span>
                    <span class="format-chip">WEBM</span>
                    <span class="format-chip-note">Hasta 4GB · Inferencia Acelerada</span>
                </div>
                """
            )

            diarization_checkbox = gr.Checkbox(
                value=False,
                label="Identificar quién habla (Diarización)",
                info="Separa y etiqueta las intervenciones de cada locutor automáticamente.",
            )

            # Acordeón Credenciales Hugging Face (para WhisperX)
            with gr.Accordion(
                "Credenciales de Hugging Face (Pyannote Diarización)", open=False, visible=True
            ) as hf_acc:
                hf_token_input = gr.Textbox(
                    label="Token de Hugging Face (HF_TOKEN)",
                    placeholder="hf_... (opcional si ya está configurado en Secrets)",
                    type="password",
                    info="Requerido si activas la diarización con WhisperX. Rol 'Read' con términos de Pyannote 3.1 y Segmentation 3.0 aceptados.",
                )

            # Acordeón Credenciales Deepgram
            with gr.Accordion(
                "Credenciales de Deepgram Cloud API", open=True, visible=False
            ) as deepgram_acc:
                deepgram_key_input = gr.Textbox(
                    label="Deepgram API Key",
                    placeholder="Pega tu clave personal de Deepgram aquí (o defínela como DEEPGRAM_API_KEY)",
                    type="password",
                    info="Permite usar Nova-2 / Nova-3 con diarización ultrarrápida. Obtén tu clave en console.deepgram.com.",
                )

            # Acordeón Credenciales AssemblyAI
            with gr.Accordion(
                "Credenciales de AssemblyAI Cloud API", open=True, visible=False
            ) as assemblyai_acc:
                assemblyai_key_input = gr.Textbox(
                    label="AssemblyAI API Key",
                    placeholder="Pega tu clave personal de AssemblyAI aquí (o defínela como ASSEMBLYAI_API_KEY)",
                    type="password",
                    info="Permite usar Universal-1 (máxima precisión) y Nano (ahorro). Obtén tu clave en assemblyai.com.",
                )

            with gr.Accordion("Opciones avanzadas", open=False):
                model_selector = gr.Dropdown(
                    choices=PROVIDER_MODEL_CHOICES[ProviderType.WHISPERX.value],
                    value=ModelName.LARGE_V3_TURBO.value,
                    label="Modelo de Voz",
                    info="Gratuito en ZeroGPU · large-v3-turbo es la opción recomendada (más rápida y misma precisión que large-v3).",
                )

                timestamp_selector = gr.Radio(
                    choices=[
                        "Simple ([HH:MM:SS])",
                        "Rango ([Inicio - Fin])",
                        "Texto continuo (Sin marcas)",
                    ],
                    value="Simple ([HH:MM:SS])",
                    label="Marcas de tiempo",
                )

            with gr.Row():
                btn_clear = gr.ClearButton(
                    value="Limpiar",
                    variant="secondary",
                    elem_classes=["btn-secondary"],
                )
                btn_transcribe = gr.Button(
                    "⚡ Iniciar Transcripción",
                    variant="primary",
                    size="lg",
                    elem_classes=["btn-primary"],
                )

        # Columna Derecha: Salida y Descarga
        with gr.Column(scale=1):
            gr.HTML(
                """
                <div class="panel-header">
                    <span class="panel-header-icon">📄</span>
                    <div>
                        <h3 class="panel-header-title">Consola de Transcripción y Salida</h3>
                        <p class="panel-header-desc">Previsualización de texto normalizado y descarga de artefactos</p>
                    </div>
                </div>
                """
            )

            output_text = gr.Textbox(
                label="Resultado de la Transcripción",
                placeholder="El texto transcrito con marcas temporales y locutores aparecerá aquí una vez finalizado el procesamiento...",
                lines=15,
                elem_classes=["output-transcript"],
            )
            download_file = gr.DownloadButton(
                label="Descargar Documento (.txt)",
                value=None,
                interactive=False,
                size="lg",
                variant="secondary",
                elem_classes=["btn-download"],
            )

    # Conexión de eventos reactivos
    def on_provider_change(selected_provider: str, is_admin: bool):
        choices = PROVIDER_MODEL_CHOICES.get(selected_provider, [])
        default_val = choices[0][1] if choices else ""

        is_wx = selected_provider == ProviderType.WHISPERX.value
        is_dg = selected_provider == ProviderType.DEEPGRAM.value
        is_aai = selected_provider == ProviderType.ASSEMBLYAI.value

        # Si el usuario es administrador, OCULTAR TODOS los acordeones de claves para máxima comodidad
        show_hf = is_wx and not is_admin
        show_dg = is_dg and not is_admin
        show_aai = is_aai and not is_admin

        info_texts = {
            ProviderType.WHISPERX.value: (
                "Inferencia en GPU (Gratuita en ZeroGPU). "
                "large-v3-turbo es la opción más rápida (~90s/2h) y eficiente."
            ),
            ProviderType.DEEPGRAM.value: (
                "Inferencia ultrarrápida (~20s/2h). "
                "nova-2 es el más económico (~$0.26/h) · nova-3 para acústica con ruido o acentos (~$0.31/h)."
            ),
            ProviderType.ASSEMBLYAI.value: (
                "Inferencia acústica de alta fidelidad (~60s/2h). "
                "best para precisión editorial (~$0.39/h) · nano para máximo ahorro (~$0.17/h)."
            ),
        }

        return (
            gr.Dropdown(
                choices=choices, value=default_val, info=info_texts.get(selected_provider, "")
            ),
            gr.Accordion(visible=show_hf),
            gr.Accordion(visible=show_dg),
            gr.Accordion(visible=show_aai),
        )

    provider_selector.change(
        fn=on_provider_change,
        inputs=[provider_selector, admin_session_state],
        outputs=[model_selector, hf_acc, deepgram_acc, assemblyai_acc],
    )

    def handle_admin_unlock(entered_pin: str, current_provider: str):
        if verify_admin_pin(entered_pin):
            success_pill = (
                '<div class="admin-active-pill">'
                '<span class="admin-active-dot"></span>'
                "<span>✓ Modo Propietario Activo · Credenciales maestras vinculadas</span>"
                '<span class="admin-active-tag">Modo Optimizado</span>'
                "</div>"
            )
            return (
                True,
                gr.Group(visible=False),  # Cierra la ventana emergente
                gr.HTML(success_pill, visible=True),
                gr.Accordion(visible=False),  # Oculta cuadro de HF Token
                gr.Accordion(visible=False),  # Oculta cuadro de Deepgram Key
                gr.Accordion(visible=False),  # Oculta cuadro de AssemblyAI Key
                gr.Markdown("", visible=False),
            )
        error_box = (
            '<div style="background: rgba(244,63,94,0.15); border: 1px solid rgba(244,63,94,0.4); padding: 8px 12px; border-radius: 8px; color: #FB7185; font-size: 13px; font-weight: 500; margin-top: 8px;">'
            "✗ PIN maestro incorrecto o no configurado en ADMIN_PIN."
            "</div>"
        )
        is_wx = current_provider == ProviderType.WHISPERX.value
        is_dg = current_provider == ProviderType.DEEPGRAM.value
        is_aai = current_provider == ProviderType.ASSEMBLYAI.value
        return (
            False,
            gr.Group(visible=True),
            gr.HTML("", visible=False),
            gr.Accordion(visible=is_wx),
            gr.Accordion(visible=is_dg),
            gr.Accordion(visible=is_aai),
            gr.Markdown(error_box, visible=True),
        )

    btn_unlock_admin.click(
        fn=handle_admin_unlock,
        inputs=[admin_pin_input, provider_selector],
        outputs=[
            admin_session_state,
            admin_modal,
            admin_status_banner,
            hf_acc,
            deepgram_acc,
            assemblyai_acc,
            admin_modal_msg,
        ],
    )
    admin_pin_input.submit(
        fn=handle_admin_unlock,
        inputs=[admin_pin_input, provider_selector],
        outputs=[
            admin_session_state,
            admin_modal,
            admin_status_banner,
            hf_acc,
            deepgram_acc,
            assemblyai_acc,
            admin_modal_msg,
        ],
    )

    btn_close_modal.click(
        fn=lambda: gr.Group(visible=False),
        inputs=None,
        outputs=[admin_modal],
    )

    def handle_url_auth(request: gr.Request, current_provider: str):
        if request:
            params = getattr(request, "query_params", {})
            pin_val = None
            if isinstance(params, dict):
                pin_val = params.get("pin")
            elif hasattr(params, "get"):
                pin_val = params.get("pin")

            if pin_val and verify_admin_pin(pin_val):
                success_pill = (
                    '<div class="admin-active-pill">'
                    '<span class="admin-active-dot"></span>'
                    "<span>✓ Modo Propietario Autenticado vía URL · Credenciales maestras vinculadas</span>"
                    '<span class="admin-active-tag">Modo Optimizado</span>'
                    "</div>"
                )
                return (
                    True,
                    gr.HTML(success_pill, visible=True),
                    gr.Accordion(visible=False),
                    gr.Accordion(visible=False),
                    gr.Accordion(visible=False),
                )
        is_wx = current_provider == ProviderType.WHISPERX.value
        is_dg = current_provider == ProviderType.DEEPGRAM.value
        is_aai = current_provider == ProviderType.ASSEMBLYAI.value
        return (
            False,
            gr.HTML("", visible=False),
            gr.Accordion(visible=is_wx),
            gr.Accordion(visible=is_dg),
            gr.Accordion(visible=is_aai),
        )

    demo.load(
        fn=handle_url_auth,
        inputs=[provider_selector],
        outputs=[
            admin_session_state,
            admin_status_banner,
            hf_acc,
            deepgram_acc,
            assemblyai_acc,
        ],
    )

    btn_clear.add([input_media, output_text])
    btn_clear.click(
        fn=lambda: gr.DownloadButton(
            label="Descargar Documento (.txt)", value=None, interactive=False
        ),
        outputs=download_file,
    )

    btn_transcribe.click(
        fn=process_transcription,
        inputs=[
            input_media,
            provider_selector,
            model_selector,
            diarization_checkbox,
            timestamp_selector,
            hf_token_input,
            deepgram_key_input,
            assemblyai_key_input,
            admin_session_state,
        ],
        outputs=[output_text, download_file],
    )

    # Barra de Estado Inferior Interactiva
    with gr.Row(elem_classes=["statusbar"]):
        with gr.Column(scale=2, min_width=320, elem_classes=["statusbar-left-col"]):
            gr.HTML(
                """
                <span class="statusbar-stats">
                    <span class="statusbar-stats-dot"></span>
                    WhisperX ZeroGPU · Deepgram Nova-2 · AssemblyAI Universal-1
                </span>
                """
            )
        with gr.Row(scale=3, elem_classes=["statusbar-right-row"]):
            btn_github = gr.Button(
                "Repositorio GitHub",
                elem_classes=["statusbar-btn", "statusbar-github-btn"],
                link="https://github.com/Klopezxd/whisperx-transcriptor",
            )
            btn_author = gr.Button(
                "Autor: Klever López 🔐",
                elem_classes=["statusbar-btn", "statusbar-author-btn"],
            )
            btn_version = gr.Button(
                "v2.2.0 Studio",
                elem_classes=["statusbar-btn", "statusbar-version-btn"],
            )

    btn_author.click(
        fn=lambda: (gr.Group(visible=True), gr.Markdown("", visible=False)),
        inputs=None,
        outputs=[admin_modal, admin_modal_msg],
    )

if __name__ == "__main__":
    demo.launch(**launch_kwargs)
