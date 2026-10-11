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
    DeepgramModel,
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
    width: 98% !important;
    margin: 0 auto !important;
    padding: 4px 12px 4px 12px !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Ocultar loaders invisibles de Gradio, bloques <style> sin contenido visual y el footer por defecto */
.wrap.default.full.hide,
.module-card > .wrap.hide,
.modal-overlay > .wrap.hide,
.block:has(> .html-container > div > style),
.block:has(> div > style),
footer:not(.app-footer) {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Reducir gaps de filas y columnas de Gradio para que todo quepa en 1 sola pantalla */
.gradio-container .row,
.gradio-container .gradio-row {
    gap: 8px !important;
}

.gradio-container .column,
.gradio-container .gradio-column {
    gap: 6px !important;
}

/* ── ELIMINA LA CAJA EXTERIOR QUE CUBRE A TODOS LOS MÓDULOS Y PADDING EXTRA ── */
.gradio-container,
.gradio-container-6-30-0,
.gradio-container > .main,
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain,
main.contain,
gradio-app {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

.gradio-container > .main,
.gradio-container > .main > .wrap,
main.contain,
main.contain > .wrap {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

/* ── PREVIENE QUE CONTENEDORES PADRE CAPTUREN POSITION: FIXED ── */
html,
body,
gradio-app,
.gradio-container,
.gradio-container > .main,
.main,
.wrap,
.contain,
main.contain {
    transform: none !important;
    filter: none !important;
    perspective: none !important;
    contain: none !important;
}

/* ── NAVBAR HEADER (COMPACTO) ─────────────────────── */
.app-navbar {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 6px 14px !important;
    background: rgba(10, 22, 40, 0.82) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    border: 1px solid rgba(0, 212, 255, 0.14) !important;
    border-radius: 13px !important;
    margin-bottom: 6px !important;
    box-shadow: 0 4px 16px rgba(5, 11, 26, 0.5), 0 0 1px 1px rgba(0, 212, 255, 0.08) !important;
}

.app-navbar .html-container,
.app-navbar .prose {
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
}

.nav-brand {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}

.brand-icon {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 28px !important;
    height: 28px !important;
    background: rgba(0, 255, 135, 0.1) !important;
    border: 1px solid rgba(0, 255, 135, 0.35) !important;
    border-radius: 8px !important;
    color: var(--aurora-green) !important;
    box-shadow: 0 0 10px rgba(0, 255, 135, 0.18) !important;
    flex-shrink: 0 !important;
}

.brand-titles h1 {
    font-size: 0.98rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    line-height: 1.15 !important;
    background: linear-gradient(135deg, #FAFBFC 15%, #00FF87 55%, #00D4FF 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.brand-titles p {
    font-size: 0.68rem !important;
    color: #8BA2BF !important;
    margin: 0 !important;
    line-height: 1.2 !important;
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
    max-width: 105px !important;
    height: 28px !important;
    padding: 3px 10px !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
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

/* ── MODULOS EN CUADROS REDONDEADOS (AURORA BOREAL COMPACTOS) ── */
.module-card {
    background: #0A1628 !important;
    border: 1px solid rgba(0, 212, 255, 0.14) !important;
    border-radius: 13px !important;
    padding: 8px 14px !important;
    margin-bottom: 0 !important;
    gap: 4px !important;
    box-shadow: 0 4px 16px rgba(5, 11, 26, 0.6), 0 0 1px 1px rgba(0, 212, 255, 0.08) !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    position: relative !important;
    overflow: visible !important;
}

.module-card:hover {
    border-color: rgba(0, 255, 135, 0.35) !important;
    box-shadow: 0 6px 22px rgba(5, 11, 26, 0.7), 0 0 14px rgba(0, 255, 135, 0.12) !important;
}

/* ── CERO CUADROS ANIDADOS Y CERO PADDING EXTRA EN BLOQUES INTERNOS ── */
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
.module-card .container,
.module-card .html-container,
.module-card .prose {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
    outline: none !important;
}

.module-card > div,
.module-card .gradio-row,
.module-card .gradio-form {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}

.module-header,
.gradio-container .row.module-header,
.module-card .row.module-header {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    margin-bottom: 3px !important;
    padding-bottom: 4px !important;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1) !important;
}

/* En encabezados con múltiples elementos (como Resultado + "En espera"), alinear extremos en la misma fila */
.module-header > div:first-child,
.row.module-header > .block:first-child {
    flex: 0 1 auto !important;
    width: auto !important;
    min-width: 0 !important;
    margin-right: auto !important;
}

.status-badge-container,
.row.module-header > .status-badge-container,
.module-header .status-badge-container,
.status-badge-container .html-container,
.status-badge-container .prose {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: fit-content !important;
    margin-left: auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    text-align: right !important;
}

.module-title {
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    font-size: 0.79rem !important;
    font-weight: 700 !important;
    color: var(--frost-white) !important;
    letter-spacing: -0.01em !important;
    line-height: 1.2 !important;
}

.module-title-icon {
    font-size: 0.88rem !important;
}

/* ── FORM ELEMENTS SUAVES Y COMPACTOS ────────────── */
.gradio-container label span,
.module-card span[data-testid="block-info"] {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.66rem !important;
    font-weight: 600 !important;
    color: #8BA2BF !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
    margin-bottom: 2px !important;
    line-height: 1.15 !important;
}

/* Ocultar o compactar subtítulo secundario de input para ahorrar espacio vertical */
.module-card .info,
.module-card div[class*="info"] {
    font-size: 0.65rem !important;
    color: #627A97 !important;
    margin-top: 2px !important;
    margin-bottom: 0 !important;
    line-height: 1.15 !important;
}

.gradio-container input[type="text"],
.gradio-container input[type="password"],
.gradio-container select {
    background-color: var(--arctic-input) !important;
    border: 1px solid rgba(0, 212, 255, 0.14) !important;
    border-radius: 8px !important;
    color: var(--frost-white) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.77rem !important;
    height: 31px !important;
    min-height: 31px !important;
    padding: 3px 10px !important;
    transition: all 0.2s ease !important;
}

.gradio-container input[type="text"]:focus,
.gradio-container input[type="password"]:focus,
.gradio-container select:focus {
    border-color: var(--aurora-green) !important;
    box-shadow: 0 0 0 2px rgba(0, 255, 135, 0.22), 0 0 12px rgba(0, 255, 135, 0.15) !important;
    outline: none !important;
}

/* ── DROPDOWNS Y MENÚS DESPLEGABLES (AURORA BOREAL) ── */
.module-card .wrap:has(.secondary-wrap),
.module-card .gradio-dropdown .wrap,
.module-card .wrap-inner,
.module-card .gradio-dropdown .wrap-inner {
    background: var(--arctic-input) !important;
    border: 1px solid rgba(0, 212, 255, 0.16) !important;
    border-radius: 8px !important;
    min-height: 32px !important;
    height: 32px !important;
    transition: all 0.2s ease !important;
    position: relative !important;
}

/* El contenedor envolvente de la página NUNCA debe tener borde ni fondo */
.main > .wrap,
.contain > .wrap,
div.wrap.svelte-zxu34v,
div.wrap:not(.module-card *) {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background: transparent !important;
}

.module-card .secondary-wrap {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    height: 100% !important;
    padding: 1px 8px !important;
    background: transparent !important;
}

.module-card .wrap:has(.secondary-wrap):focus-within,
.module-card .gradio-dropdown .wrap:focus-within,
.module-card .wrap-inner:focus-within,
.module-card .gradio-dropdown .wrap-inner:focus-within {
    border-color: var(--aurora-green) !important;
    box-shadow: 0 0 0 2px rgba(0, 255, 135, 0.22), 0 0 12px rgba(0, 255, 135, 0.15) !important;
}

.module-card input[role="combobox"],
.gradio-container input[role="combobox"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--frost-white) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    outline: none !important;
    width: 100% !important;
    padding: 2px 0 !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    cursor: pointer !important;
    user-select: none !important;
    caret-color: transparent !important;
}

.dropdown-arrow,
.module-card .dropdown-arrow,
.gradio-container .dropdown-arrow {
    fill: #8BA2BF !important;
    color: #8BA2BF !important;
    width: 14px !important;
    height: 14px !important;
    min-width: 14px !important;
    flex-shrink: 0 !important;
    margin-left: 4px !important;
    transition: transform 0.2s ease, fill 0.2s ease !important;
}

.module-card .wrap:hover .dropdown-arrow,
.gradio-container .wrap:hover .dropdown-arrow {
    fill: var(--aurora-green) !important;
    color: var(--aurora-green) !important;
}

/* ── MENÚ DE OPCIONES DESPLEGABLE (CON CUADRITOS ESTILIZADOS) ── */
.options,
ul.options,
.option-list,
div[class*="options"],
ul[role="listbox"],
div[role="listbox"],
.dropdown-menu {
    background: #060E1D !important;
    border: 1px solid rgba(0, 212, 255, 0.28) !important;
    border-radius: 10px !important;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.95), 0 0 18px rgba(0, 212, 255, 0.18) !important;
    z-index: 999999 !important;
    color: var(--frost-white) !important;
    max-height: 210px !important;
    overflow-y: auto !important;
    margin-top: 4px !important;
    padding: 5px !important;
}

li[role="option"],
.item,
div[class*="options"] li,
.option-list li,
ul.options > li,
.options .item,
[role="listbox"] > li,
[role="listbox"] > div,
[role="option"] {
    padding: 6px 10px !important;
    font-size: 0.76rem !important;
    font-family: 'Montserrat', sans-serif !important;
    color: var(--frost-white) !important;
    cursor: pointer !important;
    border: 1px solid rgba(0, 212, 255, 0.14) !important;
    border-radius: 7px !important;
    margin-bottom: 4px !important;
    background: rgba(10, 22, 40, 0.8) !important;
    transition: all 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
}

li[role="option"]:last-child,
.item:last-child,
ul.options > li:last-child,
[role="listbox"] > :last-child {
    margin-bottom: 0 !important;
}

li[role="option"]:hover,
li[role="option"].selected,
li[role="option"][aria-selected="true"],
.item:hover,
.item.selected,
[role="option"]:hover,
[role="option"].selected,
[role="option"][aria-selected="true"] {
    background: rgba(0, 255, 135, 0.18) !important;
    border-color: var(--aurora-green) !important;
    color: var(--aurora-green) !important;
    font-weight: 600 !important;
    box-shadow: 0 0 12px rgba(0, 255, 135, 0.18) !important;
}

.module-card label:has(input[type="checkbox"]) {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    padding: 2px 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    cursor: pointer !important;
    margin-top: 12px !important;
}

.module-card label:has(input[type="checkbox"]):hover {
    color: var(--aurora-green) !important;
}

/* ── MODULO 1: ARRASTRAR ARCHIVO (COMPACTO) ───────── */
.file-upload-zone {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    width: 100% !important;
}

.file-upload-zone .boundedheight,
.file-upload-zone [aria-label*="upload"],
.file-upload-zone button {
    background: rgba(6, 14, 29, 0.6) !important;
    border: 1.5px dashed rgba(0, 212, 255, 0.28) !important;
    border-radius: 10px !important;
    width: 100% !important;
    min-height: 50px !important;
    height: 50px !important;
    padding: 4px 12px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    box-sizing: border-box !important;
    transition: all 0.25s ease !important;
}

.file-upload-zone .boundedheight:hover,
.file-upload-zone [aria-label*="upload"]:hover,
.file-upload-zone button:hover {
    border-color: var(--aurora-green) !important;
    background: rgba(0, 255, 135, 0.05) !important;
    box-shadow: 0 0 14px rgba(0, 255, 135, 0.16) !important;
}

/* Reemplazo limpio del texto nativo sin overflow */
.file-upload-zone [data-testid="upload-text"],
.file-upload-zone button > div {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    font-size: 0 !important;
    color: transparent !important;
    line-height: 0 !important;
    width: 100% !important;
}

.file-upload-zone [data-testid="upload-icon"] svg,
.file-upload-zone svg.feather-upload,
.file-upload-zone button svg {
    display: inline-block !important;
    width: 18px !important;
    height: 18px !important;
    margin: 0 !important;
    color: var(--ice-cyan) !important;
    flex-shrink: 0 !important;
}

.file-upload-zone .or,
.file-upload-zone span.or {
    display: none !important;
}

.file-upload-zone [data-testid="upload-text"]::after,
.file-upload-zone button > div::after {
    content: "Arrastre el archivo aquí o haga clic para cargar" !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.76rem !important;
    line-height: 1.25 !important;
    font-weight: 500 !important;
    color: var(--frost-white) !important;
    text-align: center !important;
    display: inline-block !important;
    margin: 0 !important;
}

.file-upload-zone input[type="file"],
.file-upload-zone [type="file"] {
    display: none !important;
    opacity: 0 !important;
    position: absolute !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
}

.file-hint {
    font-size: 0.65rem !important;
    color: #627A97 !important;
    margin-top: 2px !important;
    line-height: 1.15 !important;
}

/* ── MODULO 4: RESULTADO DE LA TRANSCRIPCIÓN (ALTURA SIMÉTRICA) ── */
.result-column {
    display: flex !important;
    flex-direction: column !important;
}

.result-card {
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    height: 100% !important;
}

.output-textbox {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
}

.output-textbox > div,
.output-textbox .container,
.output-textbox label {
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
}

.output-textbox textarea {
    height: 383px !important;
    min-height: 383px !important;
    flex: 1 1 auto !important;
    background-color: #040813 !important;
    border: 1px solid rgba(0, 212, 255, 0.12) !important;
    border-radius: 10px !important;
    color: var(--frost-white) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.55 !important;
    padding: 10px 12px !important;
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

/* ── BOTONES DE ACCIÓN (COMPACTOS) ───────────────── */
.actions-row {
    margin-top: 2px !important;
    gap: 8px !important;
}

.btn-primary {
    background: linear-gradient(135deg, #00FF87 0%, #00D4FF 100%) !important;
    color: #050B1A !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.80rem !important;
    border: none !important;
    border-radius: 9px !important;
    height: 34px !important;
    min-height: 34px !important;
    padding: 6px 14px !important;
    box-shadow: 0 4px 14px rgba(0, 255, 135, 0.28) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.btn-primary:hover {
    background: linear-gradient(135deg, #00D4FF 0%, #7B2FBE 100%) !important;
    color: #FAFBFC !important;
    box-shadow: 0 6px 20px rgba(0, 212, 255, 0.45), 0 0 14px rgba(123, 47, 190, 0.3) !important;
    transform: translateY(-1px) !important;
}

.btn-secondary {
    background: rgba(232, 244, 248, 0.04) !important;
    border: 1px solid rgba(0, 212, 255, 0.16) !important;
    color: var(--frost-white) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.77rem !important;
    border-radius: 9px !important;
    height: 34px !important;
    min-height: 34px !important;
    padding: 6px 12px !important;
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
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.68rem;
    font-weight: 600;
    white-space: nowrap;
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
    gap: 6px;
    padding: 4px 9px;
    border-radius: 8px;
    font-size: 0.68rem;
    line-height: 1.25;
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.22);
    color: #7DD3FC;
    margin-top: 2px;
}

.info-free {
    background: rgba(0, 255, 135, 0.08) !important;
    border-color: rgba(0, 255, 135, 0.28) !important;
    color: var(--aurora-green) !important;
    box-shadow: 0 0 12px rgba(0, 255, 135, 0.12) !important;
}

/* ── MODAL ADMIN (CENTRADO EN PESTAÑA VISIBLE Y ACOMPAÑA EL SCROLL) ── */
.modal-overlay,
div.modal-overlay,
.gradio-container .modal-overlay,
.gradio-container div.modal-overlay {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(3, 7, 18, 0.82) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    z-index: 2147483640 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    box-sizing: border-box !important;
}

.modal-dialog,
div.modal-dialog,
.gradio-container .modal-dialog,
.gradio-container div.modal-dialog {
    position: fixed !important;
    top: var(--visible-center-y, 50vh) !important;
    left: 50vw !important;
    transform: translate(-50%, -50%) !important;
    background: #0A1628 !important;
    border: 1px solid rgba(0, 212, 255, 0.28) !important;
    border-radius: 16px !important;
    width: 90vw !important;
    max-width: 370px !important;
    height: auto !important;
    min-height: unset !important;
    max-height: fit-content !important;
    flex: 0 0 auto !important;
    margin: 0 !important;
    padding: 20px !important;
    z-index: 2147483647 !important;
    box-shadow: 0 20px 55px rgba(0, 0, 0, 0.92), 0 0 24px rgba(0, 212, 255, 0.22) !important;
    box-sizing: border-box !important;
    transition: top 0.08s ease-out !important;
}

.modal-dialog .html-container,
.modal-dialog .prose {
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
}

.modal-title-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 6px !important;
    flex-wrap: nowrap !important;
    width: 100% !important;
}

.modal-title {
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    color: var(--frost-white) !important;
    margin: 0 !important;
}

.btn-modal-close {
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    max-width: 28px !important;
    flex: 0 0 28px !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 7px !important;
    font-size: 0.80rem !important;
    color: #8BA2BF !important;
    background: rgba(232, 244, 248, 0.05) !important;
    border: 1px solid rgba(0, 212, 255, 0.15) !important;
    cursor: pointer !important;
}

.btn-modal-close:hover {
    color: #F87171 !important;
    background: rgba(239, 68, 68, 0.12) !important;
    border-color: rgba(239, 68, 68, 0.35) !important;
}

.modal-subtext {
    font-size: 0.73rem !important;
    color: #8BA2BF !important;
    margin: 0 0 10px 0 !important;
    line-height: 1.35 !important;
}

.app-footer {
    text-align: center !important;
    font-size: 0.65rem !important;
    color: #627A97 !important;
    padding: 4px 0 0 0 !important;
    border-top: 1px solid rgba(0, 212, 255, 0.06) !important;
    margin-top: 4px !important;
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
                    "✕",
                    size="sm",
                    variant="secondary",
                    min_width=32,
                    scale=0,
                    elem_classes=["btn-modal-close"],
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

            with gr.Row(elem_classes=["actions-row"]):
                btn_cancel_login = gr.Button(
                    "Cancelar",
                    variant="secondary",
                    elem_classes=["btn-secondary"],
                    min_width=80,
                    scale=1,
                )
                btn_submit_login = gr.Button(
                    "Ingresar",
                    variant="primary",
                    elem_classes=["btn-primary"],
                    min_width=120,
                    scale=2,
                )

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
                    height=60,
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
                        value=ProviderType.DEEPGRAM.value,
                        label="Proveedor de Transcripción",
                        filterable=False,
                        scale=1,
                    )
                    model_dropdown = gr.Dropdown(
                        choices=PROVIDER_MODEL_CHOICES[ProviderType.DEEPGRAM.value],
                        value=DeepgramModel.NOVA_2.value,
                        label="Modelo de Transcripción",
                        filterable=False,
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
                        filterable=False,
                        scale=1,
                    )

                provider_info_box = gr.HTML(
                    """
                    <div class="info-notice">
                        <span>💡</span>
                        <span><b>Deepgram:</b> Inferencia ultrarrápida (~20s por audio largo). Modelo <code>nova-2</code> es el más económico y <code>nova-3</code> óptimo para ruido y acentos.</span>
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
                    visible=False,
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
                    visible=True,
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
        with gr.Column(scale=6, elem_classes=["result-column"]):
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
                        '<span class="status-badge status-warning">En espera</span>',
                        elem_classes=["status-badge-container"],
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
                        elem_classes=["btn-secondary"],
                        scale=2,
                    )
                    btn_copy = gr.Button(
                        "📋 Copiar texto",
                        variant="secondary",
                        elem_classes=["btn-secondary"],
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
            gr.Dropdown(choices=choices, value=default_val, filterable=False),
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

    _modal_center_js = """(...args) => {
        const setupModalCenter = () => {
            const overlay = document.querySelector('.modal-overlay');
            const dialog = document.querySelector('.modal-dialog');
            if (!overlay || !dialog) return;
            if (overlay.parentElement !== document.body) {
                document.body.appendChild(overlay);
            }
            const applyCenter = (centerY, centerX) => {
                dialog.style.setProperty('--visible-center-y', `${Math.round(centerY)}px`);
                dialog.style.setProperty('top', `${Math.round(centerY)}px`, 'important');
                dialog.style.setProperty('left', `${Math.round(centerX)}px`, 'important');
                dialog.style.setProperty('transform', 'translate(-50%, -50%)', 'important');
            };
            const vh = window.visualViewport ? window.visualViewport.height : window.innerHeight;
            const vw = window.visualViewport ? window.visualViewport.width : window.innerWidth;
            const offsetTop = window.visualViewport ? window.visualViewport.offsetTop : 0;
            const offsetLeft = window.visualViewport ? window.visualViewport.offsetLeft : 0;
            applyCenter(offsetTop + vh / 2, offsetLeft + vw / 2);

            if ('IntersectionObserver' in window) {
                const io = new IntersectionObserver((entries) => {
                    for (const entry of entries) {
                        if (entry.intersectionRect && entry.intersectionRect.height > 0) {
                            const cy = entry.intersectionRect.top + entry.intersectionRect.height / 2;
                            const cx = entry.intersectionRect.left + entry.intersectionRect.width / 2;
                            applyCenter(cy, cx);
                        }
                    }
                    io.disconnect();
                }, { root: null });
                io.observe(document.documentElement);
            }
        };

        setupModalCenter();
        setTimeout(setupModalCenter, 40);
        setTimeout(setupModalCenter, 150);

        if (!window.__transcriptorModalTracker) {
            window.__transcriptorModalTracker = true;
            window.addEventListener('scroll', setupModalCenter, { passive: true, capture: true });
            window.addEventListener('resize', setupModalCenter, { passive: true });
            if (window.visualViewport) {
                window.visualViewport.addEventListener('scroll', setupModalCenter, { passive: true });
                window.visualViewport.addEventListener('resize', setupModalCenter, { passive: true });
            }
            setInterval(() => {
                const overlay = document.querySelector('.modal-overlay');
                if (overlay && !overlay.classList.contains('hide') && getComputedStyle(overlay).display !== 'none') {
                    setupModalCenter();
                }
            }, 180);
        }
        return args;
    }"""

    btn_open_login.click(
        fn=open_login_dialog,
        outputs=[modal_login, input_pin, login_msg],
        js=_modal_center_js,
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

    demo.load(fn=None, js=_modal_center_js)

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
