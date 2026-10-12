#!/usr/bin/env python3
"""
Transcriptor Suite - Interfaz Gráfica Aurora Boreal Nórdica (Gradio).
Desarrollada desde cero con arquitectura modular, diseño responsivo,
patrón Aurora Boreal Nórdica y gestión limpia de credenciales.
"""

import sys
import time

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


IDLE_TIME_BADGE = '<span class="status-badge status-time status-time-idle">⏱️ Transcrito en -- s</span>'


def _build_status_badges(status_span: str, time_span: str = IDLE_TIME_BADGE) -> str:
    return f'<div class="status-badges-group">{time_span}{status_span}</div>'


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
            _build_status_badges('<span class="status-badge status-warning">Sin archivo</span>'),
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
                _build_status_badges(
                    '<span class="status-badge status-error">Token HF requerido</span>'
                ),
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
                _build_status_badges(
                    '<span class="status-badge status-error">API Key requerida</span>'
                ),
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
                _build_status_badges(
                    '<span class="status-badge status-error">API Key requerida</span>'
                ),
            )

    def on_progress(p: float, msg: str) -> None:
        progress(p, desc=msg)

    start_time = time.perf_counter()

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

        elapsed_seconds = max(0.1, time.perf_counter() - start_time)
        time_badge_html = (
            f'<span class="status-badge status-time">'
            f"⏱️ Transcrito en {elapsed_seconds:.1f} segundos"
            f"</span>"
        )

        lang_info = (
            f"Idioma: {result.detected_language.upper()}"
            if getattr(result, "detected_language", None)
            else "Completado"
        )
        status_html = _build_status_badges(
            f'<span class="status-badge status-success">✓ {lang_info}</span>',
            time_span=time_badge_html,
        )

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
            _build_status_badges('<span class="status-badge status-error">Error</span>'),
        )
    except Exception as e:
        clear_gpu_vram()
        logger.exception("Error inesperado en aplicación web")
        return (
            f"❌ Ocurrió un error inesperado durante el procesamiento:\n{e}",
            gr.DownloadButton(label="📥 Descargar Documento (.txt)", value=None, interactive=False),
            _build_status_badges('<span class="status-badge status-error">Error inesperado</span>'),
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

.gradio-container,
.gradio-container > .main,
.gradio-container .contain,
main.contain {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important;
    box-sizing: border-box !important;
}

.gradio-container {
    padding: clamp(4px, 1vh, 14px) clamp(10px, 1.6vw, 28px) !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

/* ── LAYOUT RESPONSIVO DE PANTALLA COMPLETA (ADAPTACIÓN EN VIVO SIN RECARGAR ENTRE MONITOR 24"-27" Y LAPTOP 14"-16") ── */
@media (min-width: 760px) and (min-height: 460px) {
    html,
    body,
    gradio-app {
        height: var(--app-vh, 100dvh) !important;
        max-height: var(--app-vh, 100dvh) !important;
        overflow: hidden !important;
    }

    .gradio-container,
    .gradio-container > .main,
    .gradio-container > .main > .wrap,
    main.contain,
    main.contain > .wrap,
    .gradio-container div:has(> .main-body-row) {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 0% !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }

    .main-body-row,
    .gradio-container .row.main-body-row {
        flex: 1 1 0% !important;
        height: 100% !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: stretch !important;
        gap: clamp(8px, 1.1vw, 18px) !important;
        overflow: visible !important;
    }

    .left-column,
    .gradio-container .column.left-column {
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        height: 100% !important;
        min-height: 0 !important;
        gap: clamp(4px, 0.9vh, 14px) !important;
        overflow: visible !important;
    }

    .upload-card,
    .gradio-container .column.upload-card {
        flex: 1 1 0% !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        overflow: hidden !important;
        z-index: 5 !important;
    }

    .options-card,
    .gradio-container .column.options-card {
        flex: 0 0 auto !important;
        flex-shrink: 0 !important;
        height: fit-content !important;
        min-height: fit-content !important;
        max-height: none !important;
        overflow: visible !important;
        overflow-x: visible !important;
        overflow-y: visible !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        gap: clamp(6px, 1.15vh, 14px) !important;
        z-index: 40 !important;
    }

    /* En modo Admin (cuando Credenciales de API está oculto), dar más aire natural a Motor y Opciones */
    .left-column:not(:has(.keys-card:not(.hide))) .options-card,
    .gradio-container .column.left-column:not(:has(.keys-card:not(.hide))) .options-card {
        padding: clamp(12px, 2vh, 24px) clamp(16px, 1.7vw, 26px) !important;
        gap: clamp(8px, 1.6vh, 18px) !important;
    }

    .keys-card.hide,
    .gradio-container .column.keys-card.hide {
        display: none !important;
    }

    .keys-card:not(.hide),
    .gradio-container .column.keys-card:not(.hide) {
        flex: 0 0 auto !important;
        flex-shrink: 0 !important;
        height: fit-content !important;
        min-height: fit-content !important;
        max-height: none !important;
        overflow: visible !important;
        overflow-x: visible !important;
        overflow-y: visible !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        gap: clamp(5px, 0.95vh, 12px) !important;
        z-index: 10 !important;
    }
}

/* Ocultar loaders invisibles de Gradio, bloques <style> sin contenido visual y el footer por defecto */
.wrap.default.full.hide,
.module-card > .wrap.hide,
.modal-overlay > .wrap.hide,
.modal-dialog .wrap.default.full,
.block:has(> .html-container > div > style),
.block:has(> div > style),
footer:not(.app-footer) {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Gaps responsivos de filas y columnas de Gradio */
.gradio-container .row,
.gradio-container .gradio-row {
    gap: clamp(8px, 1.1vw, 16px) !important;
}

.gradio-container .column,
.gradio-container .gradio-column {
    gap: clamp(6px, 1vh, 12px) !important;
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

/* ── NAVBAR HEADER (RESPONSIVO Y LEGIBLE) ──────────── */
.app-navbar {
    flex: 0 0 auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: clamp(7px, 1.15vh, 14px) clamp(14px, 1.6vw, 24px) !important;
    background: rgba(10, 22, 40, 0.82) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    border: 1px solid rgba(0, 212, 255, 0.14) !important;
    border-radius: 13px !important;
    margin-bottom: clamp(5px, 0.9vh, 12px) !important;
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
    gap: clamp(10px, 1.1vw, 16px) !important;
}

.brand-icon {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: clamp(32px, 4.5vh, 44px) !important;
    height: clamp(32px, 4.5vh, 44px) !important;
    background: rgba(0, 255, 135, 0.1) !important;
    border: 1px solid rgba(0, 255, 135, 0.35) !important;
    border-radius: 9px !important;
    color: var(--aurora-green) !important;
    box-shadow: 0 0 10px rgba(0, 255, 135, 0.18) !important;
    flex-shrink: 0 !important;
}

.brand-titles h1 {
    font-size: clamp(1.18rem, 2.5vh, 1.55rem) !important;
    font-weight: 800 !important;
    margin: 0 !important;
    line-height: 1.15 !important;
    background: linear-gradient(135deg, #FAFBFC 15%, #00FF87 55%, #00D4FF 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.brand-titles p {
    font-size: clamp(0.82rem, 1.65vh, 1.0rem) !important;
    color: #9BB2CE !important;
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

/* ── BOTÓN DE LOGIN ADMIN / SALIR (ESQUINA SUPERIOR DERECHA) ── */
.btn-login-compact {
    width: auto !important;
    min-width: unset !important;
    max-width: 145px !important;
    height: clamp(34px, 4.5vh, 44px) !important;
    padding: 5px 18px !important;
    font-size: clamp(0.88rem, 1.75vh, 1.04rem) !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
    background: rgba(232, 244, 248, 0.05) !important;
    border: 1px solid rgba(0, 212, 255, 0.22) !important;
    color: var(--frost-white) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
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

/* ── MODULOS EN CUADROS REDONDEADOS (AURORA BOREAL RESPONSIVOS) ── */
.module-card {
    background: #0A1628 !important;
    border: 1px solid rgba(0, 212, 255, 0.14) !important;
    border-radius: 14px !important;
    padding: clamp(9px, 1.45vh, 20px) clamp(14px, 1.6vw, 24px) !important;
    margin-bottom: 0 !important;
    gap: clamp(5px, 0.9vh, 12px) !important;
    box-shadow: 0 4px 16px rgba(5, 11, 26, 0.6), 0 0 1px 1px rgba(0, 212, 255, 0.08) !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    position: relative !important;
    overflow: visible !important;
    box-sizing: border-box !important;
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
    overflow: visible !important;
}

.module-card > div,
.module-card .gradio-row,
.module-card .gradio-form {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}

/* Mantener la cuadrícula 2x2 de Motor y Opciones sin envolver al cambiar de monitor */
.options-card .row,
.options-card .form {
    flex-wrap: nowrap !important;
    overflow: visible !important;
}

.options-card .row > .block,
.options-card .form > .block {
    min-width: 0 !important;
    flex: 1 1 0% !important;
    overflow: visible !important;
}

.options-card .block:focus-within,
.options-card .row:focus-within,
.options-card .block:has(input[role="combobox"][aria-expanded="true"]),
.options-card .row:has(input[role="combobox"][aria-expanded="true"]) {
    position: relative !important;
    z-index: 9999 !important;
}

.module-header,
.gradio-container .row.module-header,
.module-card .row.module-header {
    flex: 0 0 auto !important;
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    margin-bottom: clamp(3px, 0.55vh, 8px) !important;
    padding-bottom: clamp(4px, 0.65vh, 9px) !important;
    border-bottom: 1px solid rgba(0, 212, 255, 0.12) !important;
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
    gap: 8px !important;
    font-size: clamp(1.0rem, 2.1vh, 1.26rem) !important;
    font-weight: 700 !important;
    color: var(--frost-white) !important;
    letter-spacing: -0.01em !important;
    line-height: 1.2 !important;
}

.module-title-icon {
    font-size: clamp(1.08rem, 2.2vh, 1.32rem) !important;
}

/* ── FORM ELEMENTS SUAVES, GRANDES Y LEGIBLES ──────────── */
.gradio-container label span,
.module-card span[data-testid="block-info"] {
    font-family: 'Montserrat', sans-serif !important;
    font-size: clamp(0.80rem, 1.62vh, 0.96rem) !important;
    font-weight: 700 !important;
    color: #A3BBD6 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
    margin-bottom: clamp(3px, 0.5vh, 6px) !important;
    line-height: 1.2 !important;
}

.module-card .info,
.module-card div[class*="info"] {
    font-size: clamp(0.78rem, 1.55vh, 0.94rem) !important;
    color: #88A2C2 !important;
    margin-top: 3px !important;
    margin-bottom: 0 !important;
    line-height: 1.22 !important;
}

.gradio-container input[type="text"]:not([role="combobox"]),
.gradio-container input[type="password"]:not([role="combobox"]),
.gradio-container select {
    background-color: var(--arctic-input) !important;
    border: 1px solid rgba(0, 212, 255, 0.16) !important;
    border-radius: 9px !important;
    color: var(--frost-white) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: clamp(0.90rem, 1.82vh, 1.08rem) !important;
    height: clamp(35px, 4.8vh, 46px) !important;
    min-height: clamp(35px, 4.8vh, 46px) !important;
    padding: 4px 12px !important;
    transition: all 0.2s ease !important;
}

.gradio-container input[type="text"]:not([role="combobox"]):focus,
.gradio-container input[type="password"]:not([role="combobox"]):focus,
.gradio-container select:focus {
    border-color: var(--aurora-green) !important;
    box-shadow: 0 0 0 2px rgba(0, 255, 135, 0.22), 0 0 12px rgba(0, 255, 135, 0.15) !important;
    outline: none !important;
}

/* ── DROPDOWNS Y MENÚS DESPLEGABLES (AURORA BOREAL, BORDE ÚNICO Y DESPLIEGUE INFERIOR) ── */
.module-card .wrap:has(> .wrap-inner > .secondary-wrap),
.module-card .gradio-dropdown .wrap {
    background: var(--arctic-input) !important;
    border: 1px solid rgba(0, 212, 255, 0.18) !important;
    border-radius: 9px !important;
    min-height: clamp(35px, 4.8vh, 46px) !important;
    height: clamp(35px, 4.8vh, 46px) !important;
    transition: all 0.2s ease !important;
    position: relative !important;
    cursor: pointer !important;
    overflow: visible !important;
}

.module-card .wrap-inner,
.module-card .gradio-dropdown .wrap-inner {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    min-height: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    position: relative !important;
    cursor: pointer !important;
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
    padding: 2px 12px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    position: relative !important;
    cursor: pointer !important;
}

.module-card .secondary-wrap .icon-wrap {
    position: absolute !important;
    right: 11px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    pointer-events: none !important;
    margin: 0 !important;
    padding: 0 !important;
}

.module-card .wrap:has(> .wrap-inner > .secondary-wrap):focus-within,
.module-card .gradio-dropdown .wrap:focus-within,
.module-card .wrap:has(> .wrap-inner.show_options) {
    border-color: var(--aurora-green) !important;
    box-shadow: 0 0 0 2px rgba(0, 255, 135, 0.22), 0 0 12px rgba(0, 255, 135, 0.15) !important;
}

.module-card input[role="combobox"],
.gradio-container input[role="combobox"],
.module-card input[role="combobox"]:focus,
.gradio-container input[role="combobox"]:focus {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--frost-white) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: clamp(0.90rem, 1.82vh, 1.08rem) !important;
    font-weight: 500 !important;
    outline: none !important;
    width: 100% !important;
    padding: 2px 26px 2px 0 !important;
    margin: 0 !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    cursor: pointer !important;
    user-select: none !important;
    -webkit-user-select: none !important;
    caret-color: transparent !important;
}

.module-card input[role="combobox"]::selection,
.gradio-container input[role="combobox"]::selection {
    background: transparent !important;
    color: var(--frost-white) !important;
}

.dropdown-arrow,
.module-card .dropdown-arrow,
.gradio-container .dropdown-arrow {
    fill: #9BB2CE !important;
    color: #9BB2CE !important;
    width: 16px !important;
    height: 16px !important;
    min-width: 16px !important;
    flex-shrink: 0 !important;
    margin: 0 !important;
    transition: transform 0.2s ease, fill 0.2s ease !important;
}

.module-card .wrap:has(> .wrap-inner > .secondary-wrap):hover .dropdown-arrow,
.module-card .wrap-inner.show_options .dropdown-arrow,
.module-card .wrap:has(> .wrap-inner input[role="combobox"][aria-expanded="true"]) .dropdown-arrow {
    fill: var(--aurora-green) !important;
    color: var(--aurora-green) !important;
}

.module-card .wrap-inner.show_options .dropdown-arrow,
.module-card .wrap:has(> .wrap-inner input[role="combobox"][aria-expanded="true"]) .dropdown-arrow {
    transform: rotate(180deg) !important;
}

/* ── MENÚ DE OPCIONES DESPLEGABLE (SIEMPRE DEBAJO DEL CUADRO CON CAJITAS ESTILIZADAS) ── */
.module-card .wrap > .options,
.gradio-container .wrap > .options,
div.options.svelte-1ou0lab {
    position: absolute !important;
    top: calc(100% + 5px) !important;
    bottom: auto !important;
    left: 0 !important;
    right: 0 !important;
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    background: #060E1D !important;
    border: 1px solid rgba(0, 212, 255, 0.28) !important;
    border-radius: 10px !important;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.95), 0 0 18px rgba(0, 212, 255, 0.18) !important;
    z-index: 999999 !important;
    color: var(--frost-white) !important;
    max-height: 260px !important;
    overflow-y: auto !important;
    margin: 0 !important;
    padding: 6px !important;
}

.option-list,
ul.option-list,
ul[role="listbox"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    max-height: 248px !important;
    overflow-y: auto !important;
}

li[role="option"],
.item,
.option-list li,
ul.options > li,
.options .item,
[role="listbox"] > li,
[role="option"] {
    width: 100% !important;
    max-width: 100% !important;
    padding: 9px 13px !important;
    font-size: clamp(0.88rem, 1.75vh, 1.04rem) !important;
    font-family: 'Montserrat', sans-serif !important;
    color: var(--frost-white) !important;
    cursor: pointer !important;
    border: 1px solid rgba(0, 212, 255, 0.15) !important;
    border-radius: 8px !important;
    margin-bottom: 5px !important;
    background: rgba(10, 22, 40, 0.85) !important;
    transition: all 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    box-sizing: border-box !important;
}

li[role="option"] > *,
[role="option"] > * {
    pointer-events: none !important;
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
    gap: 8px !important;
    padding: 4px 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    cursor: pointer !important;
    margin-top: clamp(14px, 2.1vh, 23px) !important;
}

.module-card label:has(input[type="checkbox"]) span {
    font-size: clamp(0.76rem, 1.4vh, 0.88rem) !important;
    letter-spacing: 0.01em !important;
    line-height: 1.2 !important;
    margin-bottom: 0 !important;
}

.module-card input[type="checkbox"] {
    width: clamp(17px, 2.2vh, 21px) !important;
    height: clamp(17px, 2.2vh, 21px) !important;
    flex-shrink: 0 !important;
}

.module-card label:has(input[type="checkbox"]):hover {
    color: var(--aurora-green) !important;
}

/* ── MODULO 1: ARRASTRAR ARCHIVO (RESPONSIVO Y ELÁSTICO SIN SOLAPAR TEXTO DE FORMATOS) ── */
.file-upload-zone,
.upload-card .file-upload-zone,
.upload-card .file-upload-zone > div:not([data-testid="status-tracker"]):not(.file-preview-holder):not(.icon-button-wrapper) {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    width: 100% !important;
    flex: 1 1 0% !important;
    min-height: 44px !important;
    height: auto !important;
    display: flex !important;
    flex-direction: column !important;
    position: relative !important;
}

/* Anula el min-height: var(--size-60) (240px) interno de Gradio 6 en .wrap.svelte-1vmd51o */
.file-upload-zone .wrap:not([data-testid="status-tracker"]),
.file-upload-zone [data-upload-density],
.file-upload-zone .icon-wrap {
    min-height: 0 !important;
    height: auto !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Zona de subida vacía (solo aplica al botón dropzone .boundedheight, NUNCA al botón X de borrar) */
.file-upload-zone button.boundedheight,
.file-upload-zone [aria-label*="upload"] {
    background: rgba(6, 14, 29, 0.6) !important;
    border: 1.5px dashed rgba(0, 212, 255, 0.28) !important;
    border-radius: 10px !important;
    width: 100% !important;
    flex: 1 1 0% !important;
    min-height: 44px !important;
    height: 100% !important;
    padding: 6px 14px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
    transition: all 0.25s ease !important;
}

.file-upload-zone button.boundedheight:hover,
.file-upload-zone [aria-label*="upload"]:hover {
    border-color: var(--aurora-green) !important;
    background: rgba(0, 255, 135, 0.05) !important;
    box-shadow: 0 0 14px rgba(0, 255, 135, 0.16) !important;
}

/* Reemplazo limpio del texto nativo sin overflow (solo en el dropzone vacío) */
.file-upload-zone [data-testid="upload-text"],
.file-upload-zone button.boundedheight > div {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 10px !important;
    font-size: 0 !important;
    color: transparent !important;
    line-height: 0 !important;
    width: 100% !important;
    min-height: 0 !important;
    height: auto !important;
    margin: 0 !important;
    padding: 0 !important;
}

.file-upload-zone [data-testid="upload-icon"] svg,
.file-upload-zone svg.feather-upload,
.file-upload-zone button.boundedheight svg {
    display: inline-block !important;
    width: clamp(22px, 3.2vh, 32px) !important;
    height: clamp(22px, 3.2vh, 32px) !important;
    margin: 0 !important;
    color: var(--ice-cyan) !important;
    flex-shrink: 0 !important;
}

.file-upload-zone .or,
.file-upload-zone span.or {
    display: none !important;
}

.file-upload-zone [data-testid="upload-text"]::after,
.file-upload-zone button.boundedheight > div::after {
    content: "Arrastre el archivo aquí o haga clic para cargar" !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: clamp(0.95rem, 2.0vh, 1.20rem) !important;
    line-height: 1.25 !important;
    font-weight: 600 !important;
    color: var(--frost-white) !important;
    text-align: center !important;
    display: inline-block !important;
    margin: 0 !important;
}

/* Estado con archivo cargado: tarjeta limpia de archivo + botón X compacto en la esquina derecha */
.file-upload-zone .file-preview-holder {
    background: rgba(6, 14, 29, 0.82) !important;
    border: 1px solid rgba(0, 255, 135, 0.35) !important;
    border-radius: 10px !important;
    box-shadow: inset 0 0 14px rgba(0, 255, 135, 0.06) !important;
    width: 100% !important;
    flex: 1 1 0% !important;
    min-height: 44px !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 6px 44px 6px 14px !important;
    display: flex !important;
    align-items: center !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
}

.file-upload-zone table.file-preview,
.file-upload-zone table.file-preview tbody {
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}

.file-upload-zone tr.file {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    height: auto !important;
    gap: 12px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
}

.file-upload-zone td.filename {
    display: flex !important;
    align-items: center !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: clamp(0.88rem, 1.75vh, 1.04rem) !important;
    font-weight: 600 !important;
    color: var(--aurora-green) !important;
    padding: 0 !important;
    border: none !important;
    min-width: 0 !important;
    overflow: hidden !important;
}

.file-upload-zone td.download,
.file-upload-zone td.download a {
    display: inline-flex !important;
    align-items: center !important;
    flex-shrink: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: clamp(0.80rem, 1.6vh, 0.94rem) !important;
    font-weight: 600 !important;
    color: #8CE0FF !important;
    text-decoration: none !important;
    padding: 0 !important;
    border: none !important;
}

.file-upload-zone .icon-button-wrapper {
    position: absolute !important;
    top: 50% !important;
    right: 10px !important;
    transform: translateY(-50%) !important;
    z-index: 15 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    width: auto !important;
    height: auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.file-upload-zone button.icon-button,
.file-upload-zone [aria-label="Clear"],
.file-upload-zone [title="Clear"] {
    width: 26px !important;
    height: 26px !important;
    min-width: 26px !important;
    min-height: 26px !important;
    max-width: 26px !important;
    max-height: 26px !important;
    flex: 0 0 26px !important;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 7px !important;
    background: rgba(15, 29, 54, 0.92) !important;
    border: 1px solid rgba(0, 212, 255, 0.28) !important;
    color: #A3BBD6 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}

.file-upload-zone button.icon-button:hover,
.file-upload-zone [aria-label="Clear"]:hover,
.file-upload-zone [title="Clear"]:hover {
    background: rgba(239, 68, 68, 0.22) !important;
    border-color: rgba(239, 68, 68, 0.6) !important;
    color: #F87171 !important;
    box-shadow: 0 0 10px rgba(239, 68, 68, 0.28) !important;
}

.file-upload-zone button.icon-button > div {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

.file-upload-zone button.icon-button > div::after {
    content: none !important;
    display: none !important;
}

.file-upload-zone button.icon-button svg {
    width: 14px !important;
    height: 14px !important;
    color: currentColor !important;
    stroke: currentColor !important;
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
    flex: 0 0 auto !important;
    font-size: clamp(0.78rem, 1.55vh, 0.94rem) !important;
    color: #88A2C2 !important;
    margin-top: 3px !important;
    line-height: 1.2 !important;
}

/* ── MODULO 4: RESULTADO DE LA TRANSCRIPCIÓN (ALTURA SIMÉTRICA Y RESPONSIVA) ── */
.result-column,
.gradio-container .column.result-column {
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
    min-height: 0 !important;
}

.result-card,
.gradio-container .column.result-card {
    flex: 1 1 0% !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    height: 100% !important;
    min-height: 0 !important;
    gap: clamp(6px, 1vh, 14px) !important;
}

.output-textbox,
.output-textbox > div:not([data-testid="status-tracker"]),
.output-textbox .container,
.output-textbox label,
.output-textbox .input-container {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    flex: 1 1 0% !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
    min-height: 0 !important;
    position: relative !important;
}

/* Ocultar trackers duplicados en el badge superior y en la fila de botones */
.status-badge-container [data-testid="status-tracker"],
.module-header [data-testid="status-tracker"],
.actions-row [data-testid="status-tracker"],
[data-testid="status-tracker"].hide {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* Overlay limpio y opaco para el progreso de transcripción dentro del área de resultado */
.output-textbox [data-testid="status-tracker"]:not(.hide),
.file-upload-zone [data-testid="status-tracker"]:not(.hide) {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    background: rgba(4, 8, 19, 0.95) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(0, 212, 255, 0.25) !important;
    border-radius: 10px !important;
    z-index: 40 !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 12px !important;
    padding: 20px !important;
    box-sizing: border-box !important;
}

.output-textbox [data-testid="status-tracker"] .progress-level {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 12px !important;
    width: 100% !important;
    max-width: 440px !important;
    margin: 0 !important;
}

.output-textbox [data-testid="status-tracker"] .progress-level-inner {
    font-family: 'Montserrat', sans-serif !important;
    font-size: clamp(0.95rem, 1.9vh, 1.12rem) !important;
    font-weight: 600 !important;
    color: var(--frost-white) !important;
    text-align: center !important;
    line-height: 1.35 !important;
    margin: 0 !important;
}

.output-textbox [data-testid="status-tracker"] .progress-bar-wrap {
    width: 100% !important;
    max-width: 380px !important;
    height: 10px !important;
    border-radius: 999px !important;
    background: rgba(10, 22, 40, 0.95) !important;
    border: 1px solid rgba(0, 212, 255, 0.32) !important;
    overflow: hidden !important;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.6) !important;
}

.output-textbox [data-testid="status-tracker"] .progress-bar {
    height: 100% !important;
    background: linear-gradient(90deg, #00FF87 0%, #00D4FF 100%) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 12px rgba(0, 255, 135, 0.45) !important;
}

.output-textbox [data-testid="status-tracker"] .meta-text,
.output-textbox [data-testid="status-tracker"] .meta-text-center {
    position: static !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: clamp(0.85rem, 1.7vh, 1.0rem) !important;
    font-weight: 600 !important;
    color: var(--aurora-green) !important;
    background: rgba(0, 255, 135, 0.08) !important;
    border: 1px solid rgba(0, 255, 135, 0.25) !important;
    border-radius: 6px !important;
    padding: 3px 10px !important;
    margin: 2px 0 0 0 !important;
}

.output-textbox textarea {
    flex: 1 1 0% !important;
    height: 0 !important;
    min-height: 110px !important;
    max-height: none !important;
    background-color: #040813 !important;
    border: 1px solid rgba(0, 212, 255, 0.14) !important;
    border-radius: 10px !important;
    color: var(--frost-white) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: clamp(0.92rem, 1.82vh, 1.12rem) !important;
    line-height: 1.6 !important;
    padding: clamp(10px, 1.4vh, 18px) clamp(12px, 1.4vw, 20px) !important;
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

/* ── BOTONES DE ACCIÓN (RESPONSIVOS Y MÁS GRANDES) ─────────────── */
.actions-row {
    flex: 0 0 auto !important;
    margin-top: 0 !important;
    gap: clamp(8px, 0.9vw, 14px) !important;
}

.btn-primary {
    background: linear-gradient(135deg, #00FF87 0%, #00D4FF 100%) !important;
    color: #050B1A !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: clamp(0.95rem, 1.9vh, 1.15rem) !important;
    border: none !important;
    border-radius: 10px !important;
    height: clamp(38px, 5.1vh, 50px) !important;
    min-height: clamp(38px, 5.1vh, 50px) !important;
    padding: 6px 16px !important;
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
    border: 1px solid rgba(0, 212, 255, 0.18) !important;
    color: var(--frost-white) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    font-size: clamp(0.90rem, 1.8vh, 1.08rem) !important;
    border-radius: 10px !important;
    height: clamp(38px, 5.1vh, 50px) !important;
    min-height: clamp(38px, 5.1vh, 50px) !important;
    padding: 6px 14px !important;
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
.status-badges-group {
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    gap: 8px;
    flex-wrap: nowrap;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 7px;
    font-size: clamp(0.82rem, 1.6vh, 0.96rem);
    font-weight: 600;
    white-space: nowrap;
}

.status-time {
    background: rgba(0, 212, 255, 0.12);
    color: var(--ice-cyan);
    border: 1px solid rgba(0, 212, 255, 0.35);
    box-shadow: 0 0 10px rgba(0, 212, 255, 0.16);
    font-variant-numeric: tabular-nums;
}

.status-time-idle {
    background: rgba(155, 178, 206, 0.07);
    color: #88A2C2;
    border: 1px solid rgba(0, 212, 255, 0.18);
    box-shadow: none;
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
    gap: 8px;
    padding: clamp(6px, 0.95vh, 11px) clamp(10px, 1.1vw, 15px);
    border-radius: 9px;
    font-size: clamp(0.82rem, 1.62vh, 0.96rem);
    line-height: 1.34;
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.22);
    color: #8CE0FF;
    margin-top: 2px;
}

.info-free {
    background: rgba(0, 255, 135, 0.08) !important;
    border-color: rgba(0, 255, 135, 0.28) !important;
    color: var(--aurora-green) !important;
    box-shadow: 0 0 12px rgba(0, 255, 135, 0.12) !important;
}

/* Ajuste compacto de espaciados verticales para pantallas bajas (<= 720px de alto) manteniendo letras grandes */
@media (max-height: 720px) {
    .gradio-container {
        padding: 4px clamp(10px, 1.5vw, 22px) !important;
    }
    .app-navbar {
        padding: 5px 14px !important;
        margin-bottom: 4px !important;
    }
    .left-column,
    .gradio-container .column.left-column {
        gap: 5px !important;
    }
    .module-card {
        padding: 6px 14px !important;
        gap: 4px !important;
    }
    .options-card,
    .gradio-container .column.options-card {
        gap: 4px !important;
    }
    .module-header,
    .gradio-container .row.module-header,
    .module-card .row.module-header {
        margin-bottom: 2px !important;
        padding-bottom: 2px !important;
    }
    .gradio-container input[type="text"]:not([role="combobox"]),
    .gradio-container input[type="password"]:not([role="combobox"]),
    .gradio-container select,
    .module-card .wrap:has(> .wrap-inner > .secondary-wrap),
    .module-card .gradio-dropdown .wrap {
        height: 33px !important;
        min-height: 33px !important;
    }
    .btn-primary,
    .btn-secondary {
        height: 35px !important;
        min-height: 35px !important;
    }
    .info-notice {
        padding: 4px 9px !important;
        line-height: 1.2 !important;
    }
    .app-footer {
        padding-top: 2px !important;
        margin-top: 2px !important;
    }
}

/* ── MODAL ADMIN (CENTRADO EN PESTAÑA VISIBLE Y CIERRE INSTANTÁNEO SIN CONGELAMIENTO) ── */
.modal-overlay.hide,
div.modal-overlay.hide,
.gradio-container .modal-overlay.hide,
.gradio-container div.modal-overlay.hide,
.modal-overlay[hidden],
.modal-overlay.hide .modal-dialog {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

.modal-overlay:not(.hide),
div.modal-overlay:not(.hide),
.gradio-container .modal-overlay:not(.hide),
.gradio-container div.modal-overlay:not(.hide) {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    inset: 0 !important;
    width: 100vw !important;
    height: 100dvh !important;
    background: rgba(3, 7, 18, 0.82) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    z-index: 2147483640 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    padding: 20px !important;
    border: none !important;
    box-sizing: border-box !important;
}

.modal-overlay:not(.hide) .modal-dialog,
.gradio-container .modal-overlay:not(.hide) .modal-dialog {
    position: fixed !important;
    top: var(--visible-center-y, 50dvh) !important;
    left: 50vw !important;
    transform: translate(-50%, -50%) !important;
    background: #0A1628 !important;
    border: 1px solid rgba(0, 212, 255, 0.28) !important;
    border-radius: 16px !important;
    width: 90vw !important;
    max-width: 420px !important;
    height: auto !important;
    min-height: unset !important;
    max-height: fit-content !important;
    flex: 0 0 auto !important;
    margin: 0 !important;
    padding: 24px !important;
    z-index: 2147483647 !important;
    box-shadow: 0 20px 55px rgba(0, 0, 0, 0.92), 0 0 24px rgba(0, 212, 255, 0.22) !important;
    box-sizing: border-box !important;
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
    margin-bottom: 8px !important;
    flex-wrap: nowrap !important;
    width: 100% !important;
}

.modal-title {
    font-size: 1.12rem !important;
    font-weight: 700 !important;
    color: var(--frost-white) !important;
    margin: 0 !important;
}

.btn-modal-close {
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    max-width: 32px !important;
    flex: 0 0 32px !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 8px !important;
    font-size: 0.92rem !important;
    color: #9BB2CE !important;
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
    font-size: 0.88rem !important;
    color: #9BB2CE !important;
    margin: 0 0 12px 0 !important;
    line-height: 1.4 !important;
}

.block:has(.app-footer) {
    flex: 0 0 auto !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
}

.app-footer {
    flex: 0 0 auto !important;
    text-align: center !important;
    font-size: clamp(0.78rem, 1.48vh, 0.92rem) !important;
    color: #88A2C2 !important;
    padding: clamp(4px, 0.65vh, 8px) 0 0 0 !important;
    border-top: 1px solid rgba(0, 212, 255, 0.08) !important;
    margin-top: clamp(4px, 0.65vh, 8px) !important;
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
    with gr.Row(elem_classes=["main-body-row"]):
        # Columna Izquierda: Módulos de Entrada y Configuración
        with gr.Column(scale=5, elem_classes=["left-column"]):
            # Módulo 1: Archivo Multimedia (Drag & Drop)
            with gr.Column(min_width=0, elem_classes=["module-card", "upload-card"]):
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
            with gr.Column(min_width=0, elem_classes=["module-card", "options-card"]):
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
                        min_width=100,
                        scale=1,
                    )
                    model_dropdown = gr.Dropdown(
                        choices=PROVIDER_MODEL_CHOICES[ProviderType.DEEPGRAM.value],
                        value=DeepgramModel.NOVA_2.value,
                        label="Modelo de Transcripción",
                        filterable=False,
                        min_width=100,
                        scale=1,
                    )

                with gr.Row():
                    diarize_checkbox = gr.Checkbox(
                        value=False,
                        label="Identificar quién habla (Diarización)",
                        min_width=100,
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
                        min_width=100,
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
                visible=True, min_width=0, elem_classes=["module-card", "keys-card"]
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
                        _build_status_badges(
                            '<span class="status-badge status-warning">En espera</span>'
                        ),
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
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        is_wx = selected_provider == ProviderType.WHISPERX.value
        is_dg = selected_provider == ProviderType.DEEPGRAM.value
        is_aai = selected_provider == ProviderType.ASSEMBLYAI.value

        show_notice = is_wx and (not is_diarize)
        show_hf = is_wx and is_diarize
        show_dg = is_dg
        show_aai = is_aai

        return (
            gr.update(visible=True),
            gr.update(visible=show_notice),
            gr.update(visible=show_hf),
            gr.update(visible=show_dg),
            gr.update(visible=show_aai),
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
            gr.update(choices=choices, value=default_val, filterable=False),
            gr.update(value=info_dict.get(selected_provider, "")),
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
        show_progress="hidden",
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
        show_progress="hidden",
    )

    # --- Flujo de Autenticación de Administrador ---
    def open_login_dialog():
        return gr.update(visible=True), "", gr.update(value="", visible=False)

    def close_login_dialog():
        return gr.update(visible=False), "", gr.update(value="", visible=False)

    _modal_open_js = """(...args) => {
        const updateModalCenter = () => {
            const overlay = document.querySelector('.modal-overlay');
            const dialog = document.querySelector('.modal-dialog');
            if (!overlay || !dialog) return;
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

        const overlay = document.querySelector('.modal-overlay');
        if (overlay) overlay.classList.remove('hide');
        updateModalCenter();
        setTimeout(updateModalCenter, 30);

        if (!window.__transcriptorModalListeners) {
            window.__transcriptorModalListeners = true;
            window.addEventListener('scroll', updateModalCenter, { passive: true, capture: true });
            window.addEventListener('resize', updateModalCenter, { passive: true });
        }
        return args;
    }"""

    _modal_close_js = """(...args) => {
        const overlay = document.querySelector('.modal-overlay');
        if (overlay) overlay.classList.add('hide');
        return args;
    }"""

    btn_open_login.click(
        fn=open_login_dialog,
        outputs=[modal_login, input_pin, login_msg],
        js=_modal_open_js,
        show_progress="hidden",
    )
    btn_close_login.click(
        fn=close_login_dialog,
        outputs=[modal_login, input_pin, login_msg],
        js=_modal_close_js,
        show_progress="hidden",
    )
    btn_cancel_login.click(
        fn=close_login_dialog,
        outputs=[modal_login, input_pin, login_msg],
        js=_modal_close_js,
        show_progress="hidden",
    )

    def process_admin_login(pin: str, current_provider: str, is_diarize: bool):
        if verify_admin_pin(pin):
            return (
                True,  # admin_state
                gr.update(visible=False),  # modal_login
                gr.update(visible=False),  # btn_open_login
                gr.update(visible=True),  # btn_logout
                gr.update(visible=False),  # guest_keys_panel
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="", visible=False),
                "",  # limpia input_pin
            )
        else:
            err_html = '<div style="color: #F87171; font-size: 0.76rem; margin-top: 6px;">✗ PIN incorrecto o no configurado en el servidor.</div>'
            panel_vis = update_guest_api_panel(current_provider, is_diarize, False)
            return (
                False,  # admin_state
                gr.update(visible=True),  # modal_login sigue abierto
                gr.update(visible=True),  # btn_open_login
                gr.update(visible=False),  # btn_logout
                *panel_vis,
                gr.update(value=err_html, visible=True),
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
        show_progress="hidden",
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
        show_progress="hidden",
    )

    def process_admin_logout(current_provider: str, is_diarize: bool):
        """Cierra sesión y restaura el panel de claves para invitados."""
        panel_vis = update_guest_api_panel(current_provider, is_diarize, False)
        return (
            False,  # admin_state
            gr.update(visible=True),  # btn_open_login
            gr.update(visible=False),  # btn_logout
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
        show_progress="hidden",
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
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

        panel_vis = update_guest_api_panel(current_provider, is_diarize, False)
        return (
            False,
            gr.update(visible=True),
            gr.update(visible=False),
            *panel_vis,
        )

    _ui_enhancements_js = """(...args) => {
        if (!window.__transcriptorUiEnhancements) {
            window.__transcriptorUiEnhancements = true;

            // 1. Toggle abrir/cerrar en los menús desplegables al hacer clic (1er clic abre, 2do clic cierra)
            const toggleCombo = (wrap, e) => {
                const input = wrap.querySelector('input[role="combobox"]');
                if (!input) return;
                const wrapInner = wrap.querySelector('.wrap-inner');
                const isOpen =
                    input.getAttribute('aria-expanded') === 'true' ||
                    (wrapInner && wrapInner.classList.contains('show_options')) ||
                    !!wrap.querySelector('.options');

                e.preventDefault();
                e.stopPropagation();
                if (window.getSelection) {
                    const sel = window.getSelection();
                    if (sel && sel.removeAllRanges) sel.removeAllRanges();
                }

                if (isOpen) {
                    input.blur();
                } else {
                    document.querySelectorAll('input[role="combobox"]').forEach((other) => {
                        if (other !== input) other.blur();
                    });
                    if (document.activeElement === input) {
                        input.blur();
                    }
                    input.focus({ preventScroll: true });
                }
            };

            document.addEventListener('mousedown', (e) => {
                if (e.button !== 0) return;

                const optionTarget = e.target.closest('.options, .option-list, [role="listbox"], [role="option"], li.item');
                if (optionTarget) {
                    setTimeout(() => {
                        document.querySelectorAll('input[role="combobox"]').forEach((inp) => inp.blur());
                        if (window.getSelection) {
                            const sel = window.getSelection();
                            if (sel && sel.removeAllRanges) sel.removeAllRanges();
                        }
                    }, 15);
                    return;
                }

                const wrapInner = e.target.closest('.wrap-inner:has(input[role="combobox"])');
                if (wrapInner && wrapInner.parentElement) {
                    window.__lastComboMousedown = Date.now();
                    toggleCombo(wrapInner.parentElement, e);
                    return;
                }

                document.querySelectorAll('input[role="combobox"]').forEach((inp) => {
                    if (
                        document.activeElement === inp ||
                        inp.getAttribute('aria-expanded') === 'true'
                    ) {
                        inp.blur();
                    }
                });
            }, true);

            document.addEventListener('click', (e) => {
                const optionTarget = e.target.closest('.options, .option-list, [role="listbox"], [role="option"], li.item');
                if (optionTarget) return;

                const wrapInner = e.target.closest('.wrap-inner:has(input[role="combobox"])');
                if (!wrapInner || !wrapInner.parentElement) return;

                if (Date.now() - (window.__lastComboMousedown || 0) < 350) {
                    e.preventDefault();
                    e.stopPropagation();
                    return;
                }
                toggleCombo(wrapInner.parentElement, e);
            }, true);

            // 2. Responsividad en tiempo real al cambiar entre monitores (ej. 24" <-> 16" laptop) sin recargar
            let lastParentVh = 0;
            const syncViewport = () => {
                const vh = window.visualViewport ? window.visualViewport.height : window.innerHeight;
                const vw = window.visualViewport ? window.visualViewport.width : window.innerWidth;
                const effectiveVh = lastParentVh > 0 ? lastParentVh : vh;

                if (effectiveVh > 0) {
                    document.documentElement.style.setProperty('--app-vh', `${Math.round(effectiveVh)}px`);
                }
                if (vw > 0) {
                    document.documentElement.style.setProperty('--app-vw', `${Math.round(vw)}px`);
                }

                // Limpiar alturas en línea que Gradio BaseTextbox inyecta al redimensionar
                document.querySelectorAll('.output-textbox textarea').forEach((ta) => {
                    if (ta.style.height) {
                        ta.style.removeProperty('height');
                    }
                });
            };

            const hookParentIframe = () => {
                if (window.parentIFrame && !window.__transcriptorHookedParentIframe) {
                    window.__transcriptorHookedParentIframe = true;
                    const origSize = window.parentIFrame.size ? window.parentIFrame.size.bind(window.parentIFrame) : null;
                    if (origSize) {
                        window.parentIFrame.size = (customH, customW) => {
                            if (lastParentVh > 0) {
                                return origSize(lastParentVh, customW);
                            }
                            return origSize(customH, customW);
                        };
                    }
                    if (typeof window.parentIFrame.getPageInfo === 'function') {
                        window.parentIFrame.getPageInfo((info) => {
                            if (!info || !info.clientHeight) return;
                            const topOffset = Math.max(0, info.offsetTop || 0);
                            const targetH = Math.max(460, info.clientHeight - topOffset);
                            if (Math.abs(targetH - lastParentVh) > 2) {
                                lastParentVh = targetH;
                                document.documentElement.style.setProperty('--app-vh', `${Math.round(targetH)}px`);
                                if (origSize) origSize(targetH);
                            }
                        });
                    }
                }
            };

            syncViewport();
            hookParentIframe();
            setTimeout(() => { syncViewport(); hookParentIframe(); }, 120);
            setTimeout(() => { syncViewport(); hookParentIframe(); }, 500);

            let rafId = 0;
            const onViewportChange = () => {
                if (rafId) cancelAnimationFrame(rafId);
                rafId = requestAnimationFrame(() => {
                    syncViewport();
                    hookParentIframe();
                });
            };

            window.addEventListener('resize', onViewportChange, { passive: true });
            if (window.visualViewport) {
                window.visualViewport.addEventListener('resize', onViewportChange, { passive: true });
            }
            if ('ResizeObserver' in window) {
                const ro = new ResizeObserver(onViewportChange);
                ro.observe(document.documentElement);
            }

            // Detectar cambio de escala DPI al arrastrar la ventana entre monitor externo (100%) y laptop (125%/150%)
            const watchDpiChange = () => {
                const mq = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
                const handler = () => {
                    onViewportChange();
                    setTimeout(onViewportChange, 60);
                    setTimeout(onViewportChange, 180);
                    watchDpiChange();
                };
                if (mq.addEventListener) {
                    mq.addEventListener('change', handler, { once: true });
                }
            };
            watchDpiChange();
        }
        return args;
    }"""

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
        js=_ui_enhancements_js,
        show_progress="hidden",
    )

    # --- Limpiar campos ---
    def clear_all():
        return (
            None,
            "",
            gr.DownloadButton(label="📥 Descargar Documento (.txt)", value=None, interactive=False),
            _build_status_badges('<span class="status-badge status-warning">En espera</span>'),
        )

    btn_clear.click(
        fn=clear_all,
        outputs=[media_input, output_textbox, btn_download, status_badge],
        show_progress="hidden",
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
