#!/usr/bin/env python3
"""
WhisperX Transcriptor - Interfaz Gráfica Moderna (Gradio).
Desarrollada desde cero con arquitectura modular, diseño responsivo y gestión limpia de credenciales.
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
    """
    Procesa el archivo multimedia y retorna la transcripción, el botón de descarga y estadísticas.
    """
    if not media_file:
        return (
            "⚠️ Por favor selecciona o arrastra un archivo de audio o video para comenzar.",
            gr.DownloadButton(label="Descargar transcripción (.txt)", value=None, interactive=False),
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
                gr.DownloadButton(label="Descargar transcripción (.txt)", value=None, interactive=False),
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
                gr.DownloadButton(label="Descargar transcripción (.txt)", value=None, interactive=False),
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
                gr.DownloadButton(label="Descargar transcripción (.txt)", value=None, interactive=False),
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

        lang_info = f"Idioma: {result.detected_language.upper()}" if getattr(result, "detected_language", None) else "Completado"
        status_html = f'<span class="status-badge status-success">✓ {lang_info}</span>'

        return (
            result.formatted_text,
            gr.DownloadButton(
                label=f"Descargar {result.output_path.name}",
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
            gr.DownloadButton(label="Descargar transcripción (.txt)", value=None, interactive=False),
            '<span class="status-badge status-error">Error</span>',
        )
    except Exception as e:
        clear_gpu_vram()
        logger.exception("Error inesperado en aplicación web")
        return (
            f"❌ Ocurrió un error inesperado durante el procesamiento:\n{e}",
            gr.DownloadButton(label="Descargar transcripción (.txt)", value=None, interactive=False),
            '<span class="status-badge status-error">Error inesperado</span>',
        )


# --- Hojas de Estilo CSS Modernas ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-base: #0B0F19;
    --bg-surface: #111827;
    --bg-surface-elevated: #162032;
    --bg-input: #0c1322;
    --border-color: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(56, 189, 248, 0.4);
    --border-focus: #06B6D4;
    --primary-gradient: linear-gradient(135deg, #059669 0%, #0284c7 100%);
    --primary-hover: linear-gradient(135deg, #10B981 0%, #06B6D4 100%);
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --color-emerald: #10B981;
    --color-cyan: #06B6D4;
}

body, html {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container {
    max-width: 1400px !important;
    width: 95% !important;
    margin: 0 auto !important;
    padding: 18px 12px 30px 12px !important;
}

/* ── NAVBAR HEADER ────────────────────────────────── */
.app-navbar {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 14px 22px !important;
    background: rgba(17, 24, 39, 0.8) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 14px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
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
    width: 36px !important;
    height: 36px !important;
    background: rgba(6, 182, 212, 0.12) !important;
    border: 1px solid rgba(6, 182, 212, 0.3) !important;
    border-radius: 9px !important;
    color: var(--color-cyan) !important;
    flex-shrink: 0 !important;
}

.brand-titles h1 {
    font-size: 1.18rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    line-height: 1.2 !important;
    background: linear-gradient(135deg, #FFFFFF 30%, #38BDF8 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.brand-titles p {
    font-size: 0.76rem !important;
    color: var(--text-secondary) !important;
    margin: 2px 0 0 0 !important;
}

.nav-actions {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}

/* ── PANELS & CARDS ──────────────────────────────── */
.panel-card {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 14px !important;
    padding: 16px 18px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.panel-card:hover {
    border-color: rgba(255, 255, 255, 0.12) !important;
}

.panel-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 12px !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.panel-title {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    color: #F1F5F9 !important;
    letter-spacing: -0.01em !important;
}

.panel-title-icon {
    font-size: 1rem !important;
}

/* ── FORM ELEMENTS ───────────────────────────────── */
.gradio-container label span {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    margin-bottom: 4px !important;
}

.gradio-container input[type="text"],
.gradio-container input[type="password"],
.gradio-container select,
.gradio-container textarea {
    background-color: var(--bg-input) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 9px !important;
    color: var(--text-primary) !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}

.gradio-container input[type="text"]:focus,
.gradio-container input[type="password"]:focus,
.gradio-container select:focus,
.gradio-container textarea:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.2) !important;
    outline: none !important;
}

.gradio-container .file-upload,
.gradio-container [data-testid="file-upload"] {
    background: rgba(12, 19, 34, 0.6) !important;
    border: 1.5px dashed rgba(56, 189, 248, 0.28) !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

.gradio-container .file-upload:hover,
.gradio-container [data-testid="file-upload"]:hover {
    border-color: var(--color-cyan) !important;
    background: rgba(6, 182, 212, 0.05) !important;
}

/* ── OUTPUT TEXTBOX ──────────────────────────────── */
.output-textbox textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.84rem !important;
    line-height: 1.6 !important;
    background-color: #070c16 !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    padding: 12px 14px !important;
    resize: vertical !important;
}

/* ── BUTTONS ─────────────────────────────────────── */
.btn-primary {
    background: var(--primary-gradient) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    box-shadow: 0 4px 14px rgba(5, 150, 105, 0.3) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.btn-primary:hover {
    background: var(--primary-hover) !important;
    box-shadow: 0 6px 20px rgba(5, 150, 105, 0.45) !important;
    transform: translateY(-1px) !important;
}

.btn-secondary {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: 9px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.btn-secondary:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}

.btn-nav {
    padding: 6px 14px !important;
    height: 34px !important;
    font-size: 0.8rem !important;
    border-radius: 8px !important;
}

/* ── STATUS BADGES ───────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
}

.status-success {
    background: rgba(16, 185, 129, 0.14);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.status-error {
    background: rgba(239, 68, 68, 0.14);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.status-warning {
    background: rgba(245, 158, 11, 0.14);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.info-notice {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.76rem;
    background: rgba(6, 182, 212, 0.08);
    border: 1px solid rgba(6, 182, 212, 0.2);
    color: #7DD3FC;
    margin-top: 6px;
}

/* ── MODAL ───────────────────────────────────────── */
.modal-overlay {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(4, 8, 18, 0.82) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    z-index: 99999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 16px !important;
}

.modal-dialog {
    background: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 14px !important;
    width: 100% !important;
    max-width: 380px !important;
    padding: 22px !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7) !important;
}

.modal-title-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 12px !important;
}

.modal-title {
    font-size: 0.98rem !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
    margin: 0 !important;
}

.modal-subtext {
    font-size: 0.76rem !important;
    color: var(--text-secondary) !important;
    margin: 0 0 14px 0 !important;
}
"""

# Configuración de tema personalizado para Gradio
custom_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0B0F19",
    body_text_color="#F8FAFC",
    background_fill_primary="#111827",
    background_fill_secondary="#0C1322",
    border_color_primary="rgba(255, 255, 255, 0.08)",
)

# Compatibilidad transparente para Gradio < 6.0 y Gradio >= 6.0
blocks_kwargs = {"title": "WhisperX Transcriptor"}
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
    # Inyección de estilos garantizada para cualquier renderizador o iframe
    gr.HTML(f"<style>{custom_css}</style>")

    # Estado de sesión de administrador (False = Invitado, True = Administrador)
    admin_state = gr.State(value=False)

    # ── 1. BARRA SUPERIOR (HEADER Y ACCESO) ─────────────────────
    with gr.Row(elem_classes=["app-navbar"]):
        gr.HTML(
            """
            <div class="nav-brand">
                <div class="brand-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                        <line x1="12" x2="12" y1="19" y2="22"/>
                    </svg>
                </div>
                <div class="brand-titles">
                    <h1>WhisperX Transcriptor</h1>
                    <p>Suite Neuronal de Reconocimiento de Voz · WhisperX, Deepgram & AssemblyAI</p>
                </div>
            </div>
            """
        )

        with gr.Row(elem_classes=["nav-actions"]):
            btn_open_login = gr.Button("🔐 Iniciar sesión", size="sm", variant="secondary", elem_classes=["btn-secondary", "btn-nav"])
            btn_logout = gr.Button("Cerrar sesión", size="sm", variant="secondary", elem_classes=["btn-secondary", "btn-nav"], visible=False)

    # ── 2. MODAL DE LOGIN DE ADMINISTRADOR ──────────────────────
    with gr.Group(visible=False, elem_classes=["modal-overlay"]) as modal_login:
        with gr.Group(elem_classes=["modal-dialog"]):
            with gr.Row(elem_classes=["modal-title-row"]):
                gr.HTML('<h3 class="modal-title">Acceso de Administrador</h3>')
                btn_close_login = gr.Button("✕", size="sm", variant="secondary", min_width=32)

            gr.HTML('<p class="modal-subtext">Ingresa tu PIN de acceso para activar las credenciales maestras del servidor.</p>')

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

    # ── 3. CUERPO PRINCIPAL (2 COLUMNAS) ────────────────────────
    with gr.Row(equal_height=False):
        # Columna Izquierda: Entrada y Configuración
        with gr.Column(scale=5):
            # Panel 1: Archivo Multimedia
            with gr.Group(elem_classes=["panel-card"]):
                gr.HTML(
                    """
                    <div class="panel-header">
                        <div class="panel-title">
                            <span class="panel-title-icon">📁</span>
                            <span>Archivo de Audio o Video</span>
                        </div>
                    </div>
                    """
                )
                media_input = gr.File(
                    label="Arrastra o selecciona tu archivo multimedia",
                    file_types=["audio", "video"],
                    file_count="single",
                    height=100,
                )
                gr.HTML(
                    """
                    <div style="font-size: 0.72rem; color: #64748B; margin-top: 4px;">
                        Formatos soportados: MP3, WAV, M4A, MP4, MKV, FLAC, OGG, WEBM, MOV, AAC.
                    </div>
                    """
                )

            # Panel 2: Motor y Modelo
            with gr.Group(elem_classes=["panel-card"]):
                gr.HTML(
                    """
                    <div class="panel-header">
                        <div class="panel-title">
                            <span class="panel-title-icon">⚙️</span>
                            <span>Proveedor y Modelo</span>
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

            # Panel 3: Credenciales de API (Solo modo invitado, oculto para administrador)
            with gr.Group(visible=True, elem_classes=["panel-card"]) as guest_keys_panel:
                gr.HTML(
                    """
                    <div class="panel-header">
                        <div class="panel-title">
                            <span class="panel-title-icon">🔑</span>
                            <span>Credenciales de API (Modo Invitado)</span>
                        </div>
                    </div>
                    """
                )

                # Aviso cuando se usa WhisperX sin diarización
                whisperx_notice = gr.HTML(
                    """
                    <div class="info-notice" style="background: rgba(16, 185, 129, 0.08); border-color: rgba(16, 185, 129, 0.25); color: #6EE7B7;">
                        <span>✨</span>
                        <span>WhisperX en GPU es 100% gratuito y no requiere ninguna clave de API externa.</span>
                    </div>
                    """,
                    visible=True,
                )

                # Input para HF Token (WhisperX + Diarización)
                hf_token_box = gr.Textbox(
                    label="Token de Hugging Face (HF_TOKEN)",
                    placeholder="hf_...",
                    type="password",
                    info="Requerido al activar Diarización con WhisperX. Rol 'Read' con términos de pyannote aceptados.",
                    visible=False,
                )

                # Input para Deepgram API Key
                deepgram_key_box = gr.Textbox(
                    label="Deepgram API Key",
                    placeholder="Pega tu clave de API de Deepgram...",
                    type="password",
                    info="Obtén tu clave con créditos iniciales gratuitos en console.deepgram.com",
                    visible=False,
                )

                # Input para AssemblyAI API Key
                assemblyai_key_box = gr.Textbox(
                    label="AssemblyAI API Key",
                    placeholder="Pega tu clave de API de AssemblyAI...",
                    type="password",
                    info="Obtén tu clave gratuita en assemblyai.com",
                    visible=False,
                )

            # Botones de Acción
            with gr.Row():
                btn_clear = gr.Button("🗑️ Limpiar", variant="secondary", elem_classes=["btn-secondary"], scale=1)
                btn_transcribe = gr.Button("⚡ Iniciar Transcripción", variant="primary", elem_classes=["btn-primary"], scale=2)

        # Columna Derecha: Salida de la Transcripción
        with gr.Column(scale=6):
            with gr.Group(elem_classes=["panel-card"]):
                with gr.Row(elem_classes=["panel-header"]):
                    gr.HTML(
                        """
                        <div class="panel-title">
                            <span class="panel-title-icon">📝</span>
                            <span>Resultado de la Transcripción</span>
                        </div>
                        """
                    )
                    status_badge = gr.HTML('<span class="status-badge status-warning">En espera</span>')

                output_textbox = gr.Textbox(
                    show_label=False,
                    placeholder="El texto transcrito con locutores y marcas de tiempo aparecerá aquí...",
                    lines=16,
                    max_lines=35,
                    elem_classes=["output-textbox"],
                )

                with gr.Row():
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
        <footer style="margin-top: 22px; padding: 12px 6px; text-align: center; font-size: 0.74rem; color: #64748B; border-top: 1px solid rgba(255, 255, 255, 0.05);">
            WhisperX Transcriptor · Modelos neuronales ASR con alineación fonética fonema a fonema y diarización de hablantes.
        </footer>
        """
    )

    # ── 4. LÓGICA REACTIVA Y EVENTOS ───────────────────────────

    def update_guest_api_panel(selected_provider: str, is_diarize: bool, is_admin: bool):
        """Controla la visibilidad del panel de claves de acuerdo a si es invitado o administrador."""
        if is_admin:
            # En modo administrador se oculta COMPLETAMENTE el panel de credenciales
            return (
                gr.Group(visible=False),
                gr.HTML(visible=False),
                gr.Textbox(visible=False),
                gr.Textbox(visible=False),
                gr.Textbox(visible=False),
            )

        # Modo invitado: mostrar el panel y adaptar campos según el proveedor
        is_wx = selected_provider == ProviderType.WHISPERX.value
        is_dg = selected_provider == ProviderType.DEEPGRAM.value
        is_aai = selected_provider == ProviderType.ASSEMBLYAI.value

        show_notice = is_wx and (not is_diarize)
        show_hf = is_wx and is_diarize
        show_dg = is_dg
        show_aai = is_aai

        return (
            gr.Group(visible=True),
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
        return gr.Group(visible=True), "", gr.HTML(visible=False)

    def close_login_dialog():
        return gr.Group(visible=False), "", gr.HTML(visible=False)

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
            # Login exitoso: se activa estado admin, se cierra el modal y se oculta el panel de claves
            return (
                True,  # admin_state
                gr.Group(visible=False),  # modal_login
                gr.Button(visible=False),  # btn_open_login
                gr.Button(visible=True),  # btn_logout
                gr.Group(visible=False),  # guest_keys_panel (oculto para administrador)
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
                gr.Group(visible=True),  # modal_login sigue abierto
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
            if isinstance(params, dict):
                pin_val = params.get("pin")
            elif hasattr(params, "get"):
                pin_val = params.get("pin")

            if pin_val and verify_admin_pin(pin_val):
                return (
                    True,  # admin_state
                    gr.Button(visible=False),
                    gr.Button(visible=True),
                    gr.Group(visible=False),  # sin panel para admin
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
            gr.DownloadButton(label="Descargar transcripción (.txt)", value=None, interactive=False),
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
