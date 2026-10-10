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
    --surface-slate: rgba(13, 21, 38, 0.82);
    --surface-darker: rgba(7, 12, 24, 0.9);
    --glass-border: rgba(56, 189, 248, 0.18);
    --frost-white: #F8FAFC;
    --muted-slate: #94A3B8;
    --dark-text: #030712;
}

/* ── BASE RESET & STRICT SINGLE-LANDING VIEWPORT ─── */
*, *::before, *::after {
    box-sizing: border-box !important;
}

html, body {
    height: 100vh !important;
    max-height: 100vh !important;
    width: 100vw !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    background-color: var(--deep-midnight) !important;
    background-image:
        radial-gradient(circle at 12% 10%, rgba(0, 255, 135, 0.07) 0%, transparent 40%),
        radial-gradient(circle at 88% 8%, rgba(168, 85, 247, 0.08) 0%, transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(0, 212, 255, 0.04) 0%, transparent 50%),
        radial-gradient(circle at 80% 90%, rgba(0, 255, 135, 0.04) 0%, transparent 35%) !important;
    background-attachment: fixed !important;
    color: var(--frost-white) !important;
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* Root Gradio Container: Fills 100vh cleanly */
.gradio-container {
    max-width: 1440px !important;
    width: 98% !important;
    height: 100vh !important;
    max-height: 100vh !important;
    margin: 0 auto !important;
    padding: 7px 12px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    overflow: hidden !important;
    gap: 6px !important;
    box-sizing: border-box !important;
}

.gradio-container > .gap,
.gradio-container > div > .gap {
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    gap: 6px !important;
    min-height: 0 !important;
    box-sizing: border-box !important;
}

/* ── HEADER ──────────────────────────────────────── */
#app-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 6px 14px !important;
    height: 44px !important;
    min-height: 44px !important;
    background: rgba(13, 21, 38, 0.75) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35) !important;
    flex-shrink: 0 !important;
    box-sizing: border-box !important;
    margin: 0 !important;
}

.header-left {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}

.header-icon-box {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 28px !important;
    height: 28px !important;
    border-radius: 7px !important;
    background: rgba(0, 255, 135, 0.12) !important;
    border: 1px solid rgba(0, 255, 135, 0.28) !important;
    flex-shrink: 0 !important;
}

.header-mic-svg {
    stroke: var(--aurora-green) !important;
    filter: drop-shadow(0 0 6px rgba(0, 255, 135, 0.6)) !important;
}

.header-title-group {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

.app-title-text {
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(135deg, #FFFFFF 20%, var(--aurora-green) 70%, var(--ice-cyan) 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.app-tag-pill {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    padding: 2px 7px !important;
    border-radius: 9999px !important;
    background: rgba(0, 255, 135, 0.1) !important;
    border: 1px solid rgba(0, 255, 135, 0.3) !important;
    color: var(--aurora-green) !important;
    letter-spacing: 0.03em !important;
}

.header-right {
    display: flex !important;
    align-items: center !important;
}

.app-subtitle-compact {
    font-size: 0.78rem !important;
    color: var(--muted-slate) !important;
    font-weight: 400 !important;
}

/* ── MAIN WORKSPACE ROW ──────────────────────────── */
.main-columns-row {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    display: flex !important;
    gap: 12px !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
}

.main-columns-row > .flex,
.main-columns-row > div > .flex {
    height: 100% !important;
    align-items: stretch !important;
    gap: 12px !important;
    min-height: 0 !important;
    display: flex !important;
    box-sizing: border-box !important;
}

/* ── UNIFIED GLASS CARDS (Left & Right Columns) ──── */
.unified-card {
    background: var(--surface-slate) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45) !important;
    padding: 12px 16px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
    position: relative !important;
}

.unified-card > .block,
.unified-card > div > .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    margin: 0 !important;
    padding: 0 !important;
}

.compact-panel-title {
    display: flex !important;
    align-items: center !important;
    gap: 7px !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.16) !important;
    flex-shrink: 0 !important;
    margin-bottom: 2px !important;
}

.panel-icon { font-size: 0.95rem !important; }
.panel-title-text {
    font-size: 0.86rem !important;
    font-weight: 700 !important;
    color: var(--frost-white) !important;
    letter-spacing: -0.01em !important;
}

/* ── CONTROLS & DROPDOWNS ────────────────────────── */
.control-row {
    display: flex !important;
    gap: 10px !important;
    align-items: flex-end !important;
    margin: 0 !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

.custom-dropdown {
    overflow: visible !important;
    position: relative !important;
}

.custom-dropdown label span,
.gradio-container label span {
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    color: var(--muted-slate) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    margin-bottom: 3px !important;
}

.gradio-container input[type="text"],
.gradio-container input[type="password"],
.gradio-container select,
.custom-dropdown .wrap,
.custom-dropdown input {
    background-color: var(--surface-darker) !important;
    border: 1px solid rgba(56, 189, 248, 0.22) !important;
    border-radius: 8px !important;
    color: var(--frost-white) !important;
    font-size: 0.83rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    height: 38px !important;
    min-height: 38px !important;
}

.gradio-container input:focus,
.gradio-container select:focus,
.custom-dropdown .wrap:focus-within {
    border-color: var(--aurora-green) !important;
    box-shadow: 0 0 0 2px rgba(0, 255, 135, 0.2) !important;
    outline: none !important;
}

/* Dropdown Floating Options Menu */
ul.options,
.options,
.gradio-container ul.options,
.gradio-container .options {
    z-index: 999999 !important;
    position: absolute !important;
    background: #070D1A !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
    border-radius: 8px !important;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.95) !important;
    max-height: 200px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

ul.options li, .options li,
.gradio-container ul.options li, .gradio-container .options li {
    padding: 7px 12px !important;
    font-size: 0.82rem !important;
    color: #F8FAFC !important;
    cursor: pointer !important;
    background: transparent !important;
}

ul.options li:hover, .options li:hover,
ul.options li.selected, .options li.selected {
    background: rgba(0, 255, 135, 0.16) !important;
    color: var(--aurora-green) !important;
}

/* ── FILE UPLOAD (CLEAN, COMPACT, NEVER OVERFLOWS) ─── */
.custom-file-upload {
    overflow: hidden !important;
    box-sizing: border-box !important;
    margin: 2px 0 !important;
}

.custom-file-upload [data-testid="file-upload"],
.custom-file-upload .file-upload,
.gradio-container .file-upload {
    border: 1.5px dashed rgba(56, 189, 248, 0.35) !important;
    border-radius: 9px !important;
    background: rgba(7, 13, 26, 0.65) !important;
    transition: all 0.24s ease !important;
    padding: 6px 10px !important;
    height: 86px !important;
    min-height: 86px !important;
    max-height: 86px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    cursor: pointer !important;
}

.custom-file-upload [data-testid="file-upload"]:hover,
.custom-file-upload .file-upload:hover,
.gradio-container .file-upload:hover {
    border-color: var(--aurora-green) !important;
    background: rgba(0, 255, 135, 0.05) !important;
    box-shadow: 0 0 16px rgba(0, 255, 135, 0.16) !important;
}

.custom-file-upload svg,
.gradio-container .file-upload svg {
    width: 24px !important;
    height: 24px !important;
    stroke: var(--ice-cyan) !important;
    margin: 0 0 2px 0 !important;
    flex-shrink: 0 !important;
}

.custom-file-upload span,
.gradio-container .file-upload span {
    font-size: 0.76rem !important;
    color: var(--muted-slate) !important;
    line-height: 1.2 !important;
    text-align: center !important;
}

/* ── OPTIONS ROW: CHECKBOX & TIMESTAMPS ─────────── */
.options-row {
    display: flex !important;
    gap: 10px !important;
    align-items: flex-end !important;
    margin: 0 !important;
}

.custom-checkbox {
    display: flex !important;
    align-items: center !important;
    height: 38px !important;
    min-height: 38px !important;
    background: rgba(7, 13, 26, 0.6) !important;
    border: 1px solid rgba(56, 189, 248, 0.22) !important;
    border-radius: 8px !important;
    padding: 0 12px !important;
    box-sizing: border-box !important;
    transition: border-color 0.2s, background 0.2s !important;
}

.custom-checkbox:hover {
    border-color: rgba(0, 255, 135, 0.4) !important;
    background: rgba(0, 255, 135, 0.04) !important;
}

.custom-checkbox label {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    cursor: pointer !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--frost-white) !important;
    margin: 0 !important;
    white-space: nowrap !important;
}

.custom-checkbox input[type="checkbox"] {
    width: 16px !important;
    height: 16px !important;
    accent-color: var(--aurora-green) !important;
    cursor: pointer !important;
    margin: 0 !important;
}

/* ── ACCORDIONS (API KEYS) ───────────────────────── */
.credentials-accordion {
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    background: rgba(7, 13, 26, 0.6) !important;
    margin: 2px 0 !important;
    padding: 4px 8px !important;
}

/* ── ACTION BUTTONS ──────────────────────────────── */
.action-buttons-row {
    display: flex !important;
    gap: 8px !important;
    align-items: center !important;
    margin-top: auto !important;
    padding-top: 4px !important;
    flex-shrink: 0 !important;
}

.btn-primary, button.btn-primary, button.primary {
    background: linear-gradient(135deg, #00FF87 0%, #00D4FF 100%) !important;
    color: var(--dark-text) !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    border: none !important;
    border-radius: 9px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.22s cubic-bezier(.4,0,.2,1) !important;
    box-shadow: 0 4px 16px rgba(0, 255, 135, 0.35) !important;
    padding: 0.45rem 1rem !important;
    height: 40px !important;
    cursor: pointer !important;
}

.btn-primary:hover, button.btn-primary:hover, button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(0, 255, 135, 0.5) !important;
    filter: brightness(1.05) !important;
}

.btn-secondary, button.btn-secondary, button.secondary {
    background: rgba(15, 23, 42, 0.85) !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    padding: 0.45rem 0.9rem !important;
    height: 40px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.btn-secondary:hover, button.btn-secondary:hover, button.secondary:hover {
    background: rgba(56, 189, 248, 0.15) !important;
    border-color: var(--sky-cyan) !important;
    color: #FFF !important;
    transform: translateY(-1px) !important;
}

/* ── OUTPUT COLUMN (TRANSCRIPTION & SALIDA) ──────── */
.output-column {
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}

.output-transcript {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    margin: 4px 0 !important;
}

.output-transcript > .block,
.output-transcript > div {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.output-transcript textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.5 !important;
    background: #040914 !important;
    border: 1px solid rgba(56, 189, 248, 0.22) !important;
    border-radius: 9px !important;
    color: #F1F5F9 !important;
    padding: 10px 14px !important;
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.65) !important;
    flex: 1 1 auto !important;
    height: 100% !important;
    min-height: 160px !important;
    resize: none !important;
    overflow-y: auto !important;
    box-sizing: border-box !important;
}

.output-transcript textarea:focus {
    border-color: var(--aurora-green) !important;
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.65), 0 0 0 2px rgba(0, 255, 135, 0.2) !important;
    outline: none !important;
}

.btn-download, button.btn-download {
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    background: rgba(15, 23, 42, 0.85) !important;
    color: var(--frost-white) !important;
    padding: 0.4rem 1rem !important;
    height: 38px !important;
    transition: all 0.22s ease !important;
    flex-shrink: 0 !important;
    cursor: pointer !important;
    width: 100% !important;
}

.btn-download:hover:not(:disabled), button.btn-download:hover:not(:disabled) {
    border-color: var(--ice-cyan) !important;
    background: rgba(0, 212, 255, 0.18) !important;
    color: #FFF !important;
    box-shadow: 0 4px 18px rgba(0, 212, 255, 0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── STATUS BAR ──────────────────────────────────── */
.statusbar {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 0 14px !important;
    height: 36px !important;
    min-height: 36px !important;
    background: rgba(4, 9, 20, 0.92) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(56, 189, 248, 0.18) !important;
    border-radius: 9px !important;
    font-size: 11px !important;
    color: var(--muted-slate) !important;
    flex-shrink: 0 !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.5) !important;
    overflow: visible !important;
    box-sizing: border-box !important;
    margin: 0 !important;
}

.statusbar-left, .statusbar-right {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    overflow: visible !important;
    padding: 0 !important;
    margin: 0 !important;
}

.statusbar-right { justify-content: flex-end !important; }

.statusbar-tech-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    color: var(--muted-slate) !important;
    padding: 2px 8px !important;
    border-radius: 6px !important;
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(56, 189, 248, 0.14) !important;
}

.statusbar-tech-dot {
    width: 6px !important;
    height: 6px !important;
    border-radius: 50% !important;
    background: var(--aurora-green) !important;
    box-shadow: 0 0 6px var(--aurora-green) !important;
}

.author-badge, button.author-badge,
.github-badge, button.github-badge {
    display: inline-flex !important;
    align-items: center !important;
    gap: 5px !important;
    padding: 3px 9px !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    white-space: nowrap !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    min-width: unset !important;
    height: 26px !important;
    line-height: normal !important;
    box-shadow: none !important;
}

.author-badge, button.author-badge {
    background: rgba(0, 212, 255, 0.1) !important;
    color: var(--ice-cyan) !important;
    border: 1px solid rgba(0, 212, 255, 0.28) !important;
}

.author-badge:hover, button.author-badge:hover {
    background: rgba(0, 212, 255, 0.22) !important;
    border-color: var(--ice-cyan) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 0 10px rgba(0, 212, 255, 0.35) !important;
}

.github-badge, button.github-badge {
    background: rgba(56, 189, 248, 0.1) !important;
    border: 1px solid rgba(56, 189, 248, 0.28) !important;
    color: var(--sky-cyan) !important;
}

.github-badge:hover, button.github-badge:hover {
    background: rgba(56, 189, 248, 0.22) !important;
    border-color: var(--sky-cyan) !important;
    color: #FFF !important;
    transform: translateY(-1px) !important;
}

.statusbar-badges {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
}

.antigravity-badge {
    display: inline-flex !important;
    align-items: center !important;
    gap: 5px !important;
    background: rgba(168, 85, 247, 0.15) !important;
    padding: 3px 9px !important;
    border-radius: 6px !important;
    color: #D8B4FE !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    border: 1px solid rgba(168, 85, 247, 0.35) !important;
    white-space: nowrap !important;
    height: 26px !important;
    box-sizing: border-box !important;
}

.version-badge {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10.5px !important;
    font-weight: 700 !important;
    padding: 3px 8px !important;
    border-radius: 6px !important;
    background: rgba(0, 255, 135, 0.12) !important;
    color: var(--aurora-green) !important;
    border: 1px solid rgba(0, 255, 135, 0.32) !important;
    letter-spacing: 0.03em !important;
    white-space: nowrap !important;
    height: 26px !important;
    display: inline-flex !important;
    align-items: center !important;
    box-sizing: border-box !important;
}

/* ── SCROLLBARS ──────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.5); border-radius: 4px; }
::-webkit-scrollbar-thumb { background: rgba(56, 189, 248, 0.3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--aurora-green); }

/* ── ADMIN MODAL & ACTIVE PILL ───────────────────── */
.admin-active-pill {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 3px 14px;
    margin: 0 auto 3px auto;
    max-width: 540px;
    background: rgba(0, 255, 135, 0.08);
    border: 1px solid rgba(0, 255, 135, 0.35);
    border-radius: 9999px;
    color: var(--aurora-green);
    font-size: 0.73rem;
    font-weight: 600;
    box-shadow: 0 2px 10px rgba(0, 255, 135, 0.15);
    animation: fadeInPill 0.35s ease-out;
}

.admin-active-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--aurora-green);
    box-shadow: 0 0 8px var(--aurora-green);
    flex-shrink: 0;
}

.admin-active-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 1px 7px;
    border-radius: 4px;
    background: rgba(0, 255, 135, 0.16);
    border: 1px solid rgba(0, 255, 135, 0.3);
    color: #FFF;
}

@keyframes fadeInPill {
    from { opacity: 0; transform: translateY(-4px); }
    to   { opacity: 1; transform: translateY(0); }
}

.admin-modal-overlay {
    position: fixed !important;
    top: 0 !important; left: 0 !important;
    width: 100vw !important; height: 100vh !important;
    background: rgba(2, 6, 23, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    z-index: 999999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 1.5rem !important;
}

.admin-modal-card {
    background: rgba(13, 21, 38, 0.97) !important;
    border: 1px solid rgba(0, 255, 135, 0.35) !important;
    border-radius: 16px !important;
    max-width: 420px !important;
    width: 100% !important;
    padding: 22px !important;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.85), 0 0 35px rgba(0, 255, 135, 0.15) !important;
}

.admin-modal-card .block,
.admin-modal-card > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.modal-header-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 14px !important;
}

.modal-title-box { display: flex; align-items: center; gap: 10px; }
.modal-title-icon { font-size: 1.4rem; }
.modal-title-text { font-size: 1rem; font-weight: 700; color: #F8FAFC; margin: 0; }
.modal-title-desc { font-size: 0.75rem; color: #94A3B8; margin: 2px 0 0 0; }

.modal-close-btn {
    max-width: 32px !important; min-width: 32px !important;
    height: 32px !important; padding: 0 !important;
    border-radius: 8px !important; font-size: 14px !important;
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    color: #94A3B8 !important; cursor: pointer !important;
}

.modal-close-btn:hover {
    color: #FFF !important;
    background: rgba(244, 63, 94, 0.25) !important;
    border-color: #F43F5E !important;
}

.modal-pin-input input {
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.15em !important;
    font-size: 1.1rem !important;
    text-align: center !important;
    margin-top: 8px !important;
    margin-bottom: 10px !important;
}

.modal-submit-btn { width: 100% !important; margin-top: 4px !important; }

/* ── MOBILE RESPONSIVE ───────────────────────────── */
@media (max-width: 768px) {
    html, body { overflow: auto !important; height: auto !important; max-height: none !important; }
    .gradio-container {
        height: auto !important; max-height: none !important;
        overflow: visible !important; width: 100% !important; padding: 6px !important;
    }
    .gradio-container > .gap,
    .gradio-container > div > .gap {
        height: auto !important; max-height: none !important; overflow: visible !important;
    }
    .main-columns-row { overflow: visible !important; height: auto !important; flex: none !important; }
    .main-columns-row > .flex,
    .main-columns-row > div > .flex {
        flex-direction: column !important; height: auto !important;
    }
    .unified-card { height: auto !important; max-height: none !important; flex: none !important; }
    .output-transcript textarea { min-height: 180px !important; height: 200px !important; }
    #app-header { height: auto !important; flex-direction: column !important; align-items: flex-start !important; gap: 4px !important; padding: 8px 12px !important; }
    .statusbar { height: auto !important; flex-direction: column !important; gap: 6px !important; padding: 8px 10px !important; }
    .statusbar-left, .statusbar-right { justify-content: center !important; flex-wrap: wrap !important; width: 100% !important; }
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
    block_background_fill="transparent",
    block_background_fill_dark="transparent",
    block_border_color="transparent",
    block_border_color_dark="transparent",
    block_radius="10px",
    block_shadow="none",
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

# Compatibilidad entre Gradio < 6.0 y Gradio >= 6.0
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
    # Inyección de estilos CSS garantizada para todas las versiones de Gradio e iframes de Hugging Face
    gr.HTML(f"<style>{custom_css}</style>")

    gr.HTML(
        """
        <header id="app-header">
            <div class="header-left">
                <div class="header-icon-box">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" class="header-mic-svg">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                        <line x1="12" x2="12" y1="19" y2="22"/>
                    </svg>
                </div>
                <div class="header-title-group">
                    <h1 class="app-title-text">WhisperX Transcriptor</h1>
                    <span class="app-tag-pill">ZeroGPU & Cloud</span>
                </div>
            </div>
            <div class="header-right">
                <span class="app-subtitle-compact">Transcripción fonética neuronal, marcas de tiempo y diarización</span>
            </div>
        </header>
        """
    )

    admin_session_state = gr.State(value=False)

    # Banner flotante minimalista de Estado Propietario (visible únicamente cuando se autentica)
    admin_status_banner = gr.HTML("", visible=False)

    with gr.Row(equal_height=True, elem_classes=["main-columns-row"]):
        # Columna Izquierda: Entrada y Controles
        with gr.Column(scale=1, elem_classes=["unified-card", "left-card"]):
            gr.HTML(
                """
                <div class="compact-panel-title">
                    <span class="panel-icon">🎙️</span>
                    <span class="panel-title-text">Configuración del Motor y Entrada</span>
                </div>
                """
            )

            with gr.Row(elem_classes=["control-row"]):
                provider_selector = gr.Dropdown(
                    choices=[
                        ProviderType.WHISPERX.value,
                        ProviderType.DEEPGRAM.value,
                        ProviderType.ASSEMBLYAI.value,
                    ],
                    value=ProviderType.WHISPERX.value,
                    label="Motor de Transcripción",
                    scale=1,
                    elem_classes=["custom-dropdown"],
                )
                model_selector = gr.Dropdown(
                    choices=PROVIDER_MODEL_CHOICES[ProviderType.WHISPERX.value],
                    value=ModelName.LARGE_V3_TURBO.value,
                    label="Modelo",
                    scale=1,
                    elem_classes=["custom-dropdown"],
                )

            input_media = gr.File(
                label="Archivo de Audio o Video",
                file_types=["audio", "video"],
                file_count="single",
                height=90,
                elem_classes=["custom-file-upload"],
            )

            with gr.Row(elem_classes=["control-row", "options-row"]):
                diarization_checkbox = gr.Checkbox(
                    value=False,
                    label="Identificar quién habla (Diarización)",
                    scale=1,
                    elem_classes=["custom-checkbox"],
                )
                timestamp_selector = gr.Dropdown(
                    choices=[
                        "Simple ([HH:MM:SS])",
                        "Rango ([Inicio - Fin])",
                        "Texto continuo (Sin marcas)",
                    ],
                    value="Simple ([HH:MM:SS])",
                    label="Marcas de tiempo",
                    scale=1,
                    elem_classes=["custom-dropdown"],
                )

            # Acordeón Credenciales Hugging Face (para WhisperX con Diarización)
            with gr.Accordion(
                "Credenciales de Hugging Face (Pyannote Diarización)", open=False, visible=False, elem_classes=["credentials-accordion"]
            ) as hf_acc:
                hf_token_input = gr.Textbox(
                    label="Token de Hugging Face (HF_TOKEN)",
                    placeholder="hf_... (opcional si ya está configurado en Secrets)",
                    type="password",
                    info="Requerido si activas la diarización con WhisperX. Rol 'Read' con términos de Pyannote 3.1 y Segmentation 3.0 aceptados.",
                )

            # Acordeón Credenciales Deepgram
            with gr.Accordion(
                "Credenciales de Deepgram Cloud API", open=True, visible=False, elem_classes=["credentials-accordion"]
            ) as deepgram_acc:
                deepgram_key_input = gr.Textbox(
                    label="Deepgram API Key",
                    placeholder="Pega tu clave personal de Deepgram aquí (o defínela como DEEPGRAM_API_KEY)",
                    type="password",
                    info="Permite usar Nova-2 / Nova-3 con diarización ultrarrápida. Obtén tu clave en console.deepgram.com.",
                )

            # Acordeón Credenciales AssemblyAI
            with gr.Accordion(
                "Credenciales de AssemblyAI Cloud API", open=True, visible=False, elem_classes=["credentials-accordion"]
            ) as assemblyai_acc:
                assemblyai_key_input = gr.Textbox(
                    label="AssemblyAI API Key",
                    placeholder="Pega tu clave personal de AssemblyAI aquí (o defínela como ASSEMBLYAI_API_KEY)",
                    type="password",
                    info="Permite usar Universal-1 (máxima precisión) y Nano (ahorro). Obtén tu clave en assemblyai.com.",
                )

            with gr.Row(elem_classes=["action-buttons-row"]):
                btn_clear = gr.ClearButton(
                    value="Limpiar",
                    variant="secondary",
                    elem_classes=["btn-secondary"],
                    scale=1,
                )
                btn_transcribe = gr.Button(
                    "⚡ Iniciar Transcripción",
                    variant="primary",
                    elem_classes=["btn-primary"],
                    scale=2,
                )

        # Columna Derecha: Salida y Descarga
        with gr.Column(scale=1, elem_classes=["unified-card", "output-column"]):
            gr.HTML(
                """
                <div class="compact-panel-title">
                    <span class="panel-icon">📄</span>
                    <span class="panel-title-text">Transcripción y Salida</span>
                </div>
                """
            )

            output_text = gr.Textbox(
                show_label=False,
                placeholder="El texto transcrito con marcas temporales y locutores aparecerá aquí una vez finalizado el procesamiento...",
                lines=12,
                max_lines=35,
                elem_classes=["output-transcript"],
            )
            download_file = gr.DownloadButton(
                label="Descargar Documento (.txt)",
                value=None,
                interactive=False,
                size="sm",
                variant="secondary",
                elem_classes=["btn-download"],
            )

    # Conexión de eventos reactivos
    def on_provider_change(selected_provider: str, is_admin: bool, is_diarize: bool):
        choices = PROVIDER_MODEL_CHOICES.get(selected_provider, [])
        default_val = choices[0][1] if choices else ""

        is_wx = selected_provider == ProviderType.WHISPERX.value
        is_dg = selected_provider == ProviderType.DEEPGRAM.value
        is_aai = selected_provider == ProviderType.ASSEMBLYAI.value

        # Si el usuario es administrador, OCULTAR TODOS los acordeones de claves para máxima comodidad
        show_hf = is_wx and is_diarize and not is_admin
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
        inputs=[provider_selector, admin_session_state, diarization_checkbox],
        outputs=[model_selector, hf_acc, deepgram_acc, assemblyai_acc],
    )

    def on_diarization_change(is_diarize: bool, selected_provider: str, is_admin: bool):
        show_hf = is_diarize and (selected_provider == ProviderType.WHISPERX.value) and not is_admin
        return gr.Accordion(visible=show_hf)

    diarization_checkbox.change(
        fn=on_diarization_change,
        inputs=[diarization_checkbox, provider_selector, admin_session_state],
        outputs=[hf_acc],
    )

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

    def handle_admin_unlock(entered_pin: str, current_provider: str, is_diarize: bool):
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
            gr.Accordion(visible=is_wx and is_diarize),
            gr.Accordion(visible=is_dg),
            gr.Accordion(visible=is_aai),
            gr.Markdown(error_box, visible=True),
        )

    btn_unlock_admin.click(
        fn=handle_admin_unlock,
        inputs=[admin_pin_input, provider_selector, diarization_checkbox],
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
        inputs=[admin_pin_input, provider_selector, diarization_checkbox],
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

    def handle_url_auth(request: gr.Request, current_provider: str, is_diarize: bool):
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
            gr.Accordion(visible=is_wx and is_diarize),
            gr.Accordion(visible=is_dg),
            gr.Accordion(visible=is_aai),
        )

    demo.load(
        fn=handle_url_auth,
        inputs=[provider_selector, diarization_checkbox],
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

    # Barra de Estado Inferior Interactiva (Estilo Preview Suite)
    with gr.Row(elem_classes=["statusbar"]):
        with gr.Row(elem_classes=["statusbar-left"]):
            btn_github = gr.Button(
                "Repositorio GitHub",
                elem_classes=["github-badge"],
                link="https://github.com/Klopezxd/whisperx-transcriptor",
                size="sm",
            )
            gr.HTML(
                """
                <span class="statusbar-tech-pill">
                    <span class="statusbar-tech-dot"></span>
                    WhisperX · Deepgram · AssemblyAI
                </span>
                """
            )
        with gr.Row(elem_classes=["statusbar-right"]):
            btn_author = gr.Button(
                "Autor: Klever López 🔐",
                elem_classes=["author-badge"],
                size="sm",
            )
            gr.HTML(
                """
                <div class="statusbar-badges">
                    <span class="antigravity-badge" title="Made by Antigravity">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12">
                            <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path>
                        </svg>
                        Made by Antigravity
                    </span>
                    <span class="version-badge" title="v2.2.0">v2.2.0</span>
                </div>
                """
            )

    btn_author.click(
        fn=lambda: (gr.Group(visible=True), gr.Markdown("", visible=False)),
        inputs=None,
        outputs=[admin_modal, admin_modal_msg],
    )

if __name__ == "__main__":
    demo.launch(**launch_kwargs)
