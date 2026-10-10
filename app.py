#!/usr/bin/env python3
"""
Transcriptor Suite - Interfaz Gráfica Aurora Boreal Nórdica (Gradio).
Desarrollada desde cero con arquitectura modular, diseño responsivo,
patrón Aurora Boreal Nórdica y gestión limpia de credenciales.
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

# Compatibilidad con ZeroGPU en Hugging Face Spaces
try:
    import spaces

    GPU_DECORATOR = spaces.GPU(duration=180)
except (ImportError, TypeError, AttributeError):

    def GPU_DECORATOR(func):
        return func


@GPU_DECORATOR
def execute_whisperx(pipeline, media_path: str, progress_callback):
    """Ejecuta el pipeline local de WhisperX bajo aceleración de GPU si está disponible."""
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
    """Procesa el archivo multimedia y retorna la transcripción, el botón de descarga y estadísticas."""
    if not media_file:
        return (
            "⚠️ Por favor selecciona o arrastra un archivo de audio o video para comenzar.",
            gr.DownloadButton(label="📥 Descargar Documento (.txt)", value=None, interactive=False),
            '<span class="status-badge status-warning">Sin archivo</span>',
        )

    media_path = media_file.name if hasattr(media_file, "name") else str(media_file)

    mode_map = {
        "Simple ([HH:MM:SS])": TimestampMode.SIMPLE,
        "Rango ([Inicio - Fin])": TimestampMode.RANGE,
        "Texto continuo (Sin marcas)": TimestampMode.NONE,
    }
    selected_mode = mode_map.get(timestamp_option, TimestampMode.SIMPLE)

    resolved_hf_token = None
    resolved_api_key = None

    if provider == ProviderType.WHISPERX.value:
        resolved_hf_token = resolve_hf_token(hf_token_input)
        if enable_diarization and not resolved_hf_token:
            return (
                "⚠️ La diarización con WhisperX requiere un token de Hugging Face.\n\n"
                "Requisitos para activarla:\n"
                "1. Acepta los términos de uso en: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "2. Acepta los términos en: https://huggingface.co/pyannote/segmentation-3.0\n"
                "3. Ingresa tu token con permisos de lectura en el campo de Hugging Face.",
                gr.DownloadButton(
                    label="📥 Descargar Documento (.txt)", value=None, interactive=False
                ),
                '<span class="status-badge status-error">Token HF requerido</span>',
            )

    elif provider == ProviderType.DEEPGRAM.value:
        if deepgram_key_input and deepgram_key_input.strip():
            resolved_api_key = resolve_deepgram_api_key(deepgram_key_input)
        elif is_admin_session:
            resolved_api_key = resolve_deepgram_api_key()
        else:
            resolved_api_key = None

        if not resolved_api_key:
            return (
                "⚠️ Se requiere una clave de API de Deepgram para utilizar este motor.\n\n"
                "• Si tienes tu clave personal, ingrésala en el panel de credenciales.\n"
                "• Si eres administrador, inicia sesión para utilizar las claves del servidor.\n"
                "• O selecciona 'WhisperX (ZeroGPU / Local)' para transcribir de forma 100% gratuita.",
                gr.DownloadButton(
                    label="📥 Descargar Documento (.txt)", value=None, interactive=False
                ),
                '<span class="status-badge status-error">API Key requerida</span>',
            )

    elif provider == ProviderType.ASSEMBLYAI.value:
        if assemblyai_key_input and assemblyai_key_input.strip():
            resolved_api_key = resolve_assemblyai_api_key(assemblyai_key_input)
        elif is_admin_session:
            resolved_api_key = resolve_assemblyai_api_key()
        else:
            resolved_api_key = None

        if not resolved_api_key:
            return (
                "⚠️ Se requiere una clave de API de AssemblyAI para utilizar este motor.\n\n"
                "• Si tienes tu clave personal, ingrésala en el panel de credenciales.\n"
                "• Si eres administrador, inicia sesión para utilizar las claves del servidor.\n"
                "• O selecciona 'WhisperX (ZeroGPU / Local)' para transcribir de forma 100% gratuita.",
                gr.DownloadButton(
                    label="📥 Descargar Documento (.txt)", value=None, interactive=False
                ),
                '<span class="status-badge status-error">API Key requerida</span>',
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

        if provider == ProviderType.WHISPERX.value:
            result = execute_whisperx(pipeline, media_path, on_progress)
        else:
            result = pipeline.process(
                media_path=media_path,
                progress_callback=on_progress,
            )

        lang_info = (
            f"Idioma: {result.detected_language.upper()}"
            if getattr(result, "detected_language", None)
            else "Completado"
        )
        status_html = f'<span class="status-badge status-success">✓ {lang_info}</span>'

        return (
            result.formatted_text,
            gr.DownloadButton(
                label=f"📥 Descargar {result.output_path.name}",
                value=str(result.output_path),
                interactive=True,
            ),
            status_html,
        )

    except TranscriptionError as e:
        clear_gpu_vram()
        logger.error("Error en procesamiento web: %s", e)
        return (
            f"❌ Error de transcripción:\n{e}",
            gr.DownloadButton(label="📥 Descargar Documento (.txt)", value=None, interactive=False),
            '<span class="status-badge status-error">Error</span>',
        )
    except Exception as e:
        clear_gpu_vram()
        logger.exception("Error inesperado en aplicación web")
        return (
            f"❌ Ocurrió un error inesperado durante el procesamiento:\n{e}",
            gr.DownloadButton(label="📥 Descargar Documento (.txt)", value=None, interactive=False),
            '<span class="status-badge status-error">Error inesperado</span>',
        )


# --- Hojas de Estilo CSS: Skin Aurora Boreal Nórdica ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --aurora-green: #00FF87;
    --arctic-blue: #0A1628;
    --deep-midnight: #050B1A;
    --ice-cyan: #00D4FF;
    --violet-sky: #7B2FBE;
    --glacier-pink: #FF6EC7;
    --frost-white: #E8F4F8;
    --snow: #FAFBFC;
    --arctic-input: #060E1D;

    --block-border-width: 0px !important;
    --block-background-fill: transparent !important;
    --block-shadow: none !important;
    --panel-border-width: 0px !important;
    --form-gap-width: 0px !important;
    --checkbox-label-border-width: 0px !important;
    --checkbox-label-background-fill: transparent !important;
}

*, *::before, *::after {
    box-sizing: border-box !important;
}

body, html {
    background-color: var(--deep-midnight) !important;
    background-image:
        radial-gradient(circle at 18% 12%, rgba(0, 255, 135, 0.08) 0%, transparent 45%),
        radial-gradient(circle at 82% 10%, rgba(123, 47, 190, 0.10) 0%, transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(0, 212, 255, 0.04) 0%, transparent 50%),
        radial-gradient(circle at 80% 90%, rgba(255, 110, 199, 0.04) 0%, transparent 40%) !important;
    background-attachment: fixed !important;
    color: var(--frost-white) !important;
    font-family: 'Montserrat', system-ui, -apple-system, sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
    -webkit-font-smoothing: antialiased;
}

.gradio-container {
    max-width: 1440px !important;
    width: 96% !important;
    margin: 0 auto !important;
    padding: 10px 14px 20px 14px !important;
}

/* ── NAVBAR HEADER ────────────────────────────────── */
.app-navbar {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 8px 18px !important;
    background: rgba(10, 22, 40, 0.82) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    border: 1px solid rgba(0, 212, 255, 0.14) !important;
    border-radius: 16px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 4px 20px rgba(5, 11, 26, 0.5), 0 0 1px 1px rgba(0, 212, 255, 0.08) !important;
}

.nav-brand {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
}

.brand-icon {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 32px !important;
    height: 32px !important;
    background: rgba(0, 255, 135, 0.1) !important;
    border: 1px solid rgba(0, 255, 135, 0.35) !important;
    border-radius: 9px !important;
    color: var(--aurora-green) !important;
    box-shadow: 0 0 12px rgba(0, 255, 135, 0.18) !important;
    flex-shrink: 0 !important;
}

.brand-titles h1 {
    font-size: 1.08rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    line-height: 1.2 !important;
    background: linear-gradient(135deg, #FAFBFC 15%, #00FF87 55%, #00D4FF 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.brand-titles p {
    font-size: 0.72rem !important;
    color: #8BA2BF !important;
    margin: 1px 0 0 0 !important;
}

.nav-actions {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    flex: 0 0 auto !important;
    width: auto !important;
    margin: 0 !important;
}

/* ── BOTONCITO DE LOGIN ADMIN (ESQUINA SUPERIOR DERECHA) ── */
.btn-login-compact {
    width: auto !important;
    min-width: unset !important;
    max-width: 110px !important;
    height: 32px !important;
    padding: 4px 12px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
    background: rgba(232, 244, 248, 0.05) !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    color: var(--frost-white) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 5px !important;
    white-space: nowrap !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.btn-login-compact:hover {
    background: rgba(0, 212, 255, 0.1) !important;
    border-color: rgba(0, 255, 135, 0.45) !important;
    color: var(--aurora-green) !important;
    box-shadow: 0 0 14px rgba(0, 255, 135, 0.22) !important;
    transform: translateY(-1px) !important;
}

/* ── MODULOS EN CUADROS REDONDEADOS (AURORA BOREAL) ── */
.module-card {
    background: rgba(10, 22, 40, 0.78) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(0, 212, 255, 0.14) !important;
    border-radius: 16px !important;
    padding: 13px 16px !important;
    margin-bottom: 11px !important;
    box-shadow: 0 4px 20px rgba(5, 11, 26, 0.6), 0 0 1px 1px rgba(0, 212, 255, 0.08) !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}

.module-card:hover {
    border-color: rgba(0, 255, 135, 0.35) !important;
    box-shadow: 0 6px 26px rgba(5, 11, 26, 0.7), 0 0 16px rgba(0, 255, 135, 0.12) !important;
}

/* ── CERO CUADROS ANIDADOS (ELIMINA BORDES Y FONDOS INTERNOS DE GRADIO) ── */
.module-card .block,
.module-card .gr-block,
.module-card .form,
.module-card fieldset,
.module-card .gradio-group,
.module-card .group,
.module-card div[class*="block"],
.module-card div[class*="form"],
.module-card div[class*="group"],
.module-card div[data-testid*="block"],
.module-card div[data-testid*="group"],
.module-card div[data-testid*="form"],
.module-card .wrap,
.module-card .container {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    outline: none !important;
}

.module-card > div,
.module-card .gradio-row,
.module-card .gradio-form {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.module-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 8px !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1) !important;
}

.module-title {
    display: flex !important;
    align-items: center !important;
    gap: 7px !important;
    font-size: 0.84rem !important;
    font-weight: 700 !important;
    color: var(--frost-white) !important;
    letter-spacing: -0.01em !important;
}

.module-title-icon {
    font-size: 0.95rem !important;
}

/* ── FORM ELEMENTS SUAVES ────────────────────────── */
.gradio-container label span {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.71rem !important;
    font-weight: 600 !important;
    color: #8BA2BF !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
    margin-bottom: 3px !important;
}

.gradio-container input[type="text"],
.gradio-container input[type="password"],
.gradio-container select {
    background-color: var(--arctic-input) !important;
    border: 1px solid rgba(0, 212, 255, 0.14) !important;
    border-radius: 10px !important;
    color: var(--frost-white) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.82rem !important;
    transition: all 0.2s ease !important;
}

.gradio-container input[type="text"]:focus,
.gradio-container input[type="password"]:focus,
.gradio-container select:focus {
    border-color: var(--aurora-green) !important;
    box-shadow: 0 0 0 2px rgba(0, 255, 135, 0.22), 0 0 12px rgba(0, 255, 135, 0.15) !important;
    outline: none !important;
}

.module-card label:has(input[type="checkbox"]) {
    display: inline-flex !important;
    align-items: center !important;
    gap: 7px !important;
    padding: 4px 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    cursor: pointer !important;
    margin-top: 1px !important;
}

.module-card label:has(input[type="checkbox"]):hover {
    color: var(--aurora-green) !important;
}

/* ── MODULO 1: ARRASTRAR ARCHIVO ──────────────────── */
.file-upload-zone {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.file-upload-zone .file-upload,
.file-upload-zone [data-testid="file-upload"],
.file-upload-zone .upload-container {
    background: rgba(6, 14, 29, 0.6) !important;
    border: 1.5px dashed rgba(0, 212, 255, 0.26) !important;
    border-radius: 12px !important;
    min-height: 80px !important;
    padding: 6px !important;
    transition: all 0.25s ease !important;
}

.file-upload-zone .file-upload:hover,
.file-upload-zone [data-testid="file-upload"]:hover,
.file-upload-zone .upload-container:hover {
    border-color: var(--aurora-green) !important;
    background: rgba(0, 255, 135, 0.04) !important;
    box-shadow: 0 0 16px rgba(0, 255, 135, 0.16) !important;
}

.file-hint {
    font-size: 0.69rem !important;
    color: #627A97 !important;
    margin-top: 2px !important;
}

/* ── MODULO 4: RESULTADO DE LA TRANSCRIPCIÓN (ALTURA SIMÉTRICA) ── */
.result-card {
    display: flex !important;
    flex-direction: column !important;
}

.output-textbox {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.output-textbox textarea {
    height: 495px !important;
    max-height: 495px !important;
    min-height: 495px !important;
    background-color: #040813 !important;
    border: 1px solid rgba(0, 212, 255, 0.12) !important;
    border-radius: 12px !important;
    color: var(--frost-white) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.65 !important;
    padding: 12px 14px !important;
    resize: none !important;
    overflow-y: auto !important;
    box-sizing: border-box !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.output-textbox textarea:focus {
    border-color: rgba(0, 255, 135, 0.4) !important;
    box-shadow: 0 0 16px rgba(0, 212, 255, 0.12) !important;
}

.output-textbox textarea::-webkit-scrollbar {
    width: 6px;
}
.output-textbox textarea::-webkit-scrollbar-track {
    background: rgba(4, 8, 19, 0.4);
    border-radius: 8px;
}
.output-textbox textarea::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 255, 0.22);
    border-radius: 8px;
}
.output-textbox textarea::-webkit-scrollbar-thumb:hover {
    background: var(--aurora-green);
    box-shadow: 0 0 8px rgba(0, 255, 135, 0.5);
}

/* ── BOTONES DE ACCIÓN ───────────────────────────── */
.actions-row {
    margin-top: 4px !important;
    gap: 8px !important;
}

.btn-primary {
    background: linear-gradient(135deg, #00FF87 0%, #00D4FF 100%) !important;
    color: #050B1A !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.86rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 9px 16px !important;
    box-shadow: 0 4px 18px rgba(0, 255, 135, 0.32) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.btn-primary:hover {
    background: linear-gradient(135deg, #00D4FF 0%, #7B2FBE 100%) !important;
    color: #FAFBFC !important;
    box-shadow: 0 6px 24px rgba(0, 212, 255, 0.48), 0 0 16px rgba(123, 47, 190, 0.3) !important;
    transform: translateY(-1px) !important;
}

.btn-secondary {
    background: rgba(232, 244, 248, 0.04) !important;
    border: 1px solid rgba(0, 212, 255, 0.16) !important;
    color: var(--frost-white) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    border-radius: 10px !important;
    padding: 9px 14px !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.btn-secondary:hover {
    background: rgba(0, 212, 255, 0.08) !important;
    border-color: rgba(0, 255, 135, 0.4) !important;
    color: var(--aurora-green) !important;
    box-shadow: 0 0 12px rgba(0, 255, 135, 0.15) !important;
}

/* ── STATUS BADGES & AVISOS ──────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    border-radius: 7px;
    font-size: 0.70rem;
    font-weight: 600;
}

.status-success {
    background: rgba(0, 255, 135, 0.12);
    color: var(--aurora-green);
    border: 1px solid rgba(0, 255, 135, 0.35);
    box-shadow: 0 0 10px rgba(0, 255, 135, 0.18);
}

.status-error {
    background: rgba(239, 68, 68, 0.14);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

.status-warning {
    background: rgba(255, 110, 199, 0.12);
    color: var(--glacier-pink);
    border: 1px solid rgba(255, 110, 199, 0.35);
}

.info-notice {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 6px 10px;
    border-radius: 9px;
    font-size: 0.72rem;
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.22);
    color: #7DD3FC;
    margin-top: 4px;
}

.info-free {
    background: rgba(0, 255, 135, 0.08) !important;
    border-color: rgba(0, 255, 135, 0.28) !important;
    color: var(--aurora-green) !important;
    box-shadow: 0 0 12px rgba(0, 255, 135, 0.12) !important;
}

/* ── MODAL ADMIN ─────────────────────────────────── */
.modal-overlay {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(3, 7, 18, 0.85) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    z-index: 99999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 16px !important;
}

.modal-dialog {
    background: #0A1628 !important;
    border: 1px solid rgba(0, 212, 255, 0.22) !important;
    border-radius: 18px !important;
    width: 100% !important;
    max-width: 380px !important;
    padding: 22px !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8), 0 0 24px rgba(0, 212, 255, 0.15) !important;
}

.modal-title-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 10px !important;
}

.modal-title {
    font-size: 0.96rem !important;
    font-weight: 700 !important;
    color: var(--frost-white) !important;
    margin: 0 !important;
}

.btn-modal-close {
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    color: #8BA2BF !important;
    background: rgba(232, 244, 248, 0.05) !important;
    border: 1px solid rgba(0, 212, 255, 0.15) !important;
    cursor: pointer !important;
}

.btn-modal-close:hover {
    color: #F87171 !important;
    background: rgba(239, 68, 68, 0.1) !important;
    border-color: rgba(239, 68, 68, 0.3) !important;
}

.modal-subtext {
    font-size: 0.75rem !important;
    color: #8BA2BF !important;
    margin: 0 0 12px 0 !important;
}

.app-footer {
    text-align: center !important;
    font-size: 0.70rem !important;
    color: #627A97 !important;
    padding: 12px 0 4px 0 !important;
    border-top: 1px solid rgba(0, 212, 255, 0.06) !important;
    margin-top: 14px !important;
}
"""

# Configuración de tema personalizado para Gradio (Aurora Boreal Nórdica)
custom_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Montserrat"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#050B1A",
    body_text_color="#E8F4F8",
    background_fill_primary="#0A1628",
    background_fill_secondary="#060E1D",
    border_color_primary="rgba(0, 212, 255, 0.14)",
    border_color_primary_dark="rgba(0, 212, 255, 0.14)",
    block_border_width="0px",
    block_border_width_dark="0px",
    block_background_fill="transparent",
    block_background_fill_dark="transparent",
    block_shadow="none",
    block_shadow_dark="none",
    panel_border_width="0px",
    panel_border_width_dark="0px",
    panel_background_fill="transparent",
    panel_background_fill_dark="transparent",
    form_gap_width="0px",
    checkbox_label_background_fill="transparent",
    checkbox_label_background_fill_dark="transparent",
    checkbox_label_border_width="0px",
    checkbox_label_border_width_dark="0px",
)

blocks_kwargs = {"title": "Transcriptor Suite"}
launch_kwargs = {}

try:
    _gradio_major = int(gr.__version__.split(".")[0])
    if _gradio_major >= 6:
        launch_kwargs["theme"] = custom_theme
        launch_kwargs["css"] = custom_css
    else:
        blocks_kwargs["theme"] = custom_theme
        blocks_kwargs["css"] = custom_css
except Exception:
    blocks_kwargs["theme"] = custom_theme
    blocks_kwargs["css"] = custom_css

with gr.Blocks(**blocks_kwargs) as demo:
    gr.HTML(f"<style>{custom_css}</style>")

    admin_state = gr.State(value=False)

    # ── 1. BARRA SUPERIOR CON LOGIN ADMIN COMPACTO A LA DERECHA ─
    with gr.Row(elem_classes=["app-navbar"]):
        gr.HTML(
            """
            <div class="nav-brand">
                <div class="brand-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                        <line x1="12" x2="12" y1="19" y2="22"/>
                    </svg>
                </div>
                <div class="brand-titles">
                    <h1>Transcriptor Suite</h1>
                    <p>Suite Neuronal de Reconocimiento de Voz · WhisperX, Deepgram & AssemblyAI</p>
                </div>
            </div>
            """
        )

        with gr.Row(elem_classes=["nav-actions"]):
            btn_open_login = gr.Button(
                "🔐 Admin",
                size="sm",
                variant="secondary",
                elem_classes=["btn-secondary", "btn-login-compact"],
            )
            btn_logout = gr.Button(
                "🔓 Salir",
                size="sm",
                variant="secondary",
                elem_classes=["btn-secondary", "btn-login-compact"],
                visible=False,
            )

    # ── 2. MODAL DE LOGIN DE ADMINISTRADOR ──────────────────────
    with gr.Column(visible=False, elem_classes=["modal-overlay"]) as modal_login:
        with gr.Column(elem_classes=["modal-dialog"]):
            with gr.Row(elem_classes=["modal-title-row"]):
                gr.HTML('<h3 class="modal-title">Acceso de Administrador</h3>')
                btn_close_login = gr.Button(
                    "✕", size="sm", variant="secondary", elem_classes=["btn-modal-close"]
                )

            gr.HTML(
                '<p class="modal-subtext">Ingresa tu PIN de acceso para activar las credenciales maestras del servidor.</p>'
            )

            input_pin = gr.Textbox(
                type="password",
                placeholder="Ingresa PIN o contraseña...",
                show_label=False,
                container=False,
            )

            login_msg = gr.HTML("", visible=False)

            with gr.Row():
                btn_cancel_login = gr.Button("Cancelar", variant="secondary", scale=1)
                btn_submit_login = gr.Button("Ingresar", variant="primary", scale=2)

    # ── 3. CUERPO PRINCIPAL (LANDING PAGE ÚNICA DE 2 COLUMNAS) ───
    with gr.Row():
        # Columna Izquierda: Módulos de Entrada y Configuración
        with gr.Column(scale=5):
            # Módulo 1: Archivo Multimedia (Drag & Drop)
            with gr.Column(min_width=0, elem_classes=["module-card"]):
                gr.HTML(
                    """
                    <div class="module-header">
                        <div class="module-title">
                            <span class="module-title-icon">📁</span>
                            <span>Archivo Multimedia</span>
                        </div>
                    </div>
                    """
                )
                media_input = gr.File(
                    label="Arrastra o selecciona tu archivo multimedia",
                    show_label=False,
                    file_types=["audio", "video"],
                    file_count="single",
                    height=85,
                    container=False,
                    elem_classes=["file-upload-zone"],
                )
                gr.HTML(
                    """
                    <div class="file-hint">
                        Formatos soportados: MP3, WAV, M4A, MP4, MKV, FLAC, OGG, WEBM, MOV, AAC.
                    </div>
                    """
                )

            # Módulo 2: Motor y Opciones
            with gr.Column(min_width=0, elem_classes=["module-card"]):
                gr.HTML(
                    """
                    <div class="module-header">
                        <div class="module-title">
                            <span class="module-title-icon">⚙️</span>
                            <span>Motor y Opciones</span>
                        </div>
                    </div>
                    """
                )
                with gr.Row():
                    provider_dropdown = gr.Dropdown(
                        choices=[
                            ProviderType.WHISPERX.value,
                            ProviderType.DEEPGRAM.value,
                            ProviderType.ASSEMBLYAI.value,
                        ],
                        value=ProviderType.WHISPERX.value,
                        label="Proveedor de Transcripción",
                        scale=1,
                    )
                    model_dropdown = gr.Dropdown(
                        choices=PROVIDER_MODEL_CHOICES[ProviderType.WHISPERX.value],
                        value=ModelName.LARGE_V3_TURBO.value,
                        label="Modelo de Transcripción",
                        scale=1,
                    )

                with gr.Row():
                    diarize_checkbox = gr.Checkbox(
                        value=False,
                        label="Identificar quién habla (Diarización)",
                        scale=1,
                    )
                    timestamps_dropdown = gr.Dropdown(
                        choices=[
                            "Simple ([HH:MM:SS])",
                            "Rango ([Inicio - Fin])",
                            "Texto continuo (Sin marcas)",
                        ],
                        value="Simple ([HH:MM:SS])",
                        label="Formato de Marcas de Tiempo",
                        scale=1,
                    )

                provider_info_box = gr.HTML(
                    """
                    <div class="info-notice">
                        <span>💡</span>
                        <span><b>WhisperX:</b> Inferencia GPU local / ZeroGPU. Modelo large-v3-turbo ofrece máxima velocidad y eficiencia.</span>
                    </div>
                    """
                )

            # Módulo 3: Credenciales de API (Modo Invitado / Admin)
            with gr.Column(
                visible=True, min_width=0, elem_classes=["module-card"]
            ) as guest_keys_panel:
                gr.HTML(
                    """
                    <div class="module-header">
                        <div class="module-title">
                            <span class="module-title-icon">🔑</span>
                            <span>Credenciales de API</span>
                        </div>
                    </div>
                    """
                )

                whisperx_notice = gr.HTML(
                    """
                    <div class="info-notice info-free">
                        <span>✨</span>
                        <span>WhisperX en GPU es 100% gratuito y no requiere ninguna clave de API externa.</span>
                    </div>
                    """,
                    visible=True,
                )

                hf_token_box = gr.Textbox(
                    label="Token de Hugging Face (HF_TOKEN)",
                    placeholder="hf_...",
                    type="password",
                    info="Requerido al activar Diarización con WhisperX. Rol 'Read' con términos de pyannote aceptados.",
                    visible=False,
                )

                deepgram_key_box = gr.Textbox(
                    label="Deepgram API Key",
                    placeholder="Pega tu clave de API de Deepgram...",
                    type="password",
                    info="Obtén tu clave con créditos iniciales gratuitos en console.deepgram.com",
                    visible=False,
                )

                assemblyai_key_box = gr.Textbox(
                    label="AssemblyAI API Key",
                    placeholder="Pega tu clave de API de AssemblyAI...",
                    type="password",
                    info="Obtén tu clave gratuita en assemblyai.com",
                    visible=False,
                )

            # Botones de Acción (Configuración)
            with gr.Row(elem_classes=["actions-row"]):
                btn_clear = gr.Button(
                    "🗑️ Limpiar", variant="secondary", elem_classes=["btn-secondary"], scale=1
                )
                btn_transcribe = gr.Button(
                    "⚡ Iniciar Transcripción",
                    variant="primary",
                    elem_classes=["btn-primary"],
                    scale=2,
                )

        # Columna Derecha: Módulo del Resultado de la Transcripción
        with gr.Column(scale=6):
            # Módulo 4: Resultado de la Transcripción
            with gr.Column(min_width=0, elem_classes=["module-card", "result-card"]):
                with gr.Row(elem_classes=["module-header"]):
                    gr.HTML(
                        """
                        <div class="module-title">
                            <span class="module-title-icon">📝</span>
                            <span>Resultado de la Transcripción</span>
                        </div>
                        """
                    )
                    status_badge = gr.HTML(
                        '<span class="status-badge status-warning">En espera</span>'
                    )

                output_textbox = gr.Textbox(
                    show_label=False,
                    placeholder="El texto transcrito con locutores y marcas de tiempo aparecerá aquí...",
                    container=False,
                    elem_classes=["output-textbox"],
                )

                with gr.Row(elem_classes=["actions-row"]):
                    btn_download = gr.DownloadButton(
                        label="📥 Descargar Documento (.txt)",
                        value=None,
                        interactive=False,
                        variant="secondary",
                        scale=2,
                    )
                    btn_copy = gr.Button(
                        "📋 Copiar texto",
                        variant="secondary",
                        scale=1,
                    )

    # ── FOOTER ─────────────────────────────────────────────────
    gr.HTML(
        """
        <footer class="app-footer">
            Transcriptor Suite · Modelos neuronales ASR con alineación fonética fonema a fonema y diarización de hablantes.
        </footer>
        """
    )

    # ── 4. LÓGICA REACTIVA Y EVENTOS ───────────────────────────

    def update_guest_api_panel(selected_provider: str, is_diarize: bool, is_admin: bool):
        """Controla la visibilidad del panel de claves de acuerdo a si es invitado o administrador."""
        if is_admin:
            return (
                gr.Column(visible=False),
                gr.HTML(visible=False),
                gr.Textbox(visible=False),
                gr.Textbox(visible=False),
                gr.Textbox(visible=False),
            )

        is_wx = selected_provider == ProviderType.WHISPERX.value
        is_dg = selected_provider == ProviderType.DEEPGRAM.value
        is_aai = selected_provider == ProviderType.ASSEMBLYAI.value

        show_notice = is_wx and (not is_diarize)
        show_hf = is_wx and is_diarize
        show_dg = is_dg
        show_aai = is_aai

        return (
            gr.Column(visible=True),
            gr.HTML(visible=show_notice),
            gr.Textbox(visible=show_hf),
            gr.Textbox(visible=show_dg),
            gr.Textbox(visible=show_aai),
        )

    def on_provider_selected(selected_provider: str, is_diarize: bool, is_admin: bool):
        """Actualiza modelos disponibles, panel de claves y descripción."""
        choices = PROVIDER_MODEL_CHOICES.get(selected_provider, [])
        default_val = choices[0][1] if choices else ""

        info_dict = {
            ProviderType.WHISPERX.value: (
                '<div class="info-notice"><span>💡</span>'
                "<span><b>WhisperX:</b> Inferencia GPU local / ZeroGPU. Modelo <code>large-v3-turbo</code> ofrece máxima velocidad y eficiencia.</span></div>"
            ),
            ProviderType.DEEPGRAM.value: (
                '<div class="info-notice"><span>💡</span>'
                "<span><b>Deepgram:</b> Inferencia ultrarrápida (~20s por audio largo). Modelo <code>nova-2</code> es el más económico y <code>nova-3</code> óptimo para ruido y acentos.</span></div>"
            ),
            ProviderType.ASSEMBLYAI.value: (
                '<div class="info-notice"><span>💡</span>'
                "<span><b>AssemblyAI:</b> Precisión acústica y editorial de alta fidelidad. <code>best</code> (Universal-1) y <code>nano</code> (ahorro).</span></div>"
            ),
        }

        panel_vis = update_guest_api_panel(selected_provider, is_diarize, is_admin)

        return (
            gr.Dropdown(choices=choices, value=default_val),
            gr.HTML(value=info_dict.get(selected_provider, "")),
            *panel_vis,
        )

    provider_dropdown.change(
        fn=on_provider_selected,
        inputs=[provider_dropdown, diarize_checkbox, admin_state],
        outputs=[
            model_dropdown,
            provider_info_box,
            guest_keys_panel,
            whisperx_notice,
            hf_token_box,
            deepgram_key_box,
            assemblyai_key_box,
        ],
    )

    diarize_checkbox.change(
        fn=update_guest_api_panel,
        inputs=[provider_dropdown, diarize_checkbox, admin_state],
        outputs=[
            guest_keys_panel,
            whisperx_notice,
            hf_token_box,
            deepgram_key_box,
            assemblyai_key_box,
        ],
    )

    # --- Flujo de Autenticación de Administrador ---
    def open_login_dialog():
        return gr.Column(visible=True), "", gr.HTML(visible=False)

    def close_login_dialog():
        return gr.Column(visible=False), "", gr.HTML(visible=False)

    btn_open_login.click(
        fn=open_login_dialog,
        outputs=[modal_login, input_pin, login_msg],
    )
    btn_close_login.click(
        fn=close_login_dialog,
        outputs=[modal_login, input_pin, login_msg],
    )
    btn_cancel_login.click(
        fn=close_login_dialog,
        outputs=[modal_login, input_pin, login_msg],
    )

    def process_admin_login(pin: str, current_provider: str, is_diarize: bool):
        if verify_admin_pin(pin):
            return (
                True,  # admin_state
                gr.Column(visible=False),  # modal_login
                gr.Button(visible=False),  # btn_open_login
                gr.Button(visible=True),  # btn_logout
                gr.Column(visible=False),  # guest_keys_panel
                gr.HTML(visible=False),
                gr.Textbox(visible=False),
                gr.Textbox(visible=False),
                gr.Textbox(visible=False),
                gr.HTML(visible=False),
                "",  # limpia input_pin
            )
        else:
            err_html = '<div style="color: #F87171; font-size: 0.76rem; margin-top: 6px;">✗ PIN incorrecto o no configurado en el servidor.</div>'
            panel_vis = update_guest_api_panel(current_provider, is_diarize, False)
            return (
                False,  # admin_state
                gr.Column(visible=True),  # modal_login sigue abierto
                gr.Button(visible=True),  # btn_open_login
                gr.Button(visible=False),  # btn_logout
                *panel_vis,
                gr.HTML(value=err_html, visible=True),
                pin,
            )

    btn_submit_login.click(
        fn=process_admin_login,
        inputs=[input_pin, provider_dropdown, diarize_checkbox],
        outputs=[
            admin_state,
            modal_login,
            btn_open_login,
            btn_logout,
            guest_keys_panel,
            whisperx_notice,
            hf_token_box,
            deepgram_key_box,
            assemblyai_key_box,
            login_msg,
            input_pin,
        ],
    )

    input_pin.submit(
        fn=process_admin_login,
        inputs=[input_pin, provider_dropdown, diarize_checkbox],
        outputs=[
            admin_state,
            modal_login,
            btn_open_login,
            btn_logout,
            guest_keys_panel,
            whisperx_notice,
            hf_token_box,
            deepgram_key_box,
            assemblyai_key_box,
            login_msg,
            input_pin,
        ],
    )

    def process_admin_logout(current_provider: str, is_diarize: bool):
        """Cierra sesión y restaura el panel de claves para invitados."""
        panel_vis = update_guest_api_panel(current_provider, is_diarize, False)
        return (
            False,  # admin_state
            gr.Button(visible=True),  # btn_open_login
            gr.Button(visible=False),  # btn_logout
            *panel_vis,
        )

    btn_logout.click(
        fn=process_admin_logout,
        inputs=[provider_dropdown, diarize_checkbox],
        outputs=[
            admin_state,
            btn_open_login,
            btn_logout,
            guest_keys_panel,
            whisperx_notice,
            hf_token_box,
            deepgram_key_box,
            assemblyai_key_box,
        ],
    )

    # --- Autenticación silenciosa por URL (?pin=...) ---
    def check_url_auth(request: gr.Request, current_provider: str, is_diarize: bool):
        if request:
            params = getattr(request, "query_params", {})
            pin_val = None
            if isinstance(params, dict) or hasattr(params, "get"):
                pin_val = params.get("pin")

            if pin_val and verify_admin_pin(pin_val):
                return (
                    True,  # admin_state
                    gr.Button(visible=False),
                    gr.Button(visible=True),
                    gr.Column(visible=False),
                    gr.HTML(visible=False),
                    gr.Textbox(visible=False),
                    gr.Textbox(visible=False),
                    gr.Textbox(visible=False),
                )

        panel_vis = update_guest_api_panel(current_provider, is_diarize, False)
        return (
            False,
            gr.Button(visible=True),
            gr.Button(visible=False),
            *panel_vis,
        )

    demo.load(
        fn=check_url_auth,
        inputs=[provider_dropdown, diarize_checkbox],
        outputs=[
            admin_state,
            btn_open_login,
            btn_logout,
            guest_keys_panel,
            whisperx_notice,
            hf_token_box,
            deepgram_key_box,
            assemblyai_key_box,
        ],
    )

    # --- Limpiar campos ---
    def clear_all():
        return (
            None,
            "",
            gr.DownloadButton(label="📥 Descargar Documento (.txt)", value=None, interactive=False),
            '<span class="status-badge status-warning">En espera</span>',
        )

    btn_clear.click(
        fn=clear_all,
        outputs=[media_input, output_textbox, btn_download, status_badge],
    )

    # --- Copiar al portapapeles ---
    btn_copy.click(
        fn=None,
        inputs=[output_textbox],
        js="""(text) => {
            if (!text) return;
            navigator.clipboard.writeText(text);
        }""",
    )

    # --- Iniciar Transcripción ---
    btn_transcribe.click(
        fn=process_transcription,
        inputs=[
            media_input,
            provider_dropdown,
            model_dropdown,
            diarize_checkbox,
            timestamps_dropdown,
            hf_token_box,
            deepgram_key_box,
            assemblyai_key_box,
            admin_state,
        ],
        outputs=[output_textbox, btn_download, status_badge],
    )


if __name__ == "__main__":
    demo.launch(**launch_kwargs)
