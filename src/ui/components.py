"""Componentes visuales y módulos de interfaz desacoplados para Gradio."""

from dataclasses import dataclass

import gradio as gr

from src.config import (
    PROVIDER_MODEL_CHOICES,
    TIMESTAMP_CHOICES,
    DeepgramModel,
    ProviderType,
)
from src.ui.helpers import (
    EMPTY_DOWNLOAD_LABEL,
    PROVIDER_INFO_HTML,
    _build_status_badges,
)


@dataclass
class UIComponents:
    """Contenedor tipado con todas las referencias a componentes de la interfaz."""

    # Estados reactivos
    admin_state: gr.State
    batch_cache_state: gr.State

    # Barra de navegación
    btn_open_login: gr.Button
    btn_logout: gr.Button

    # Modal de autenticación
    modal_login: gr.Column
    btn_close_login: gr.Button
    input_pin: gr.Textbox
    login_msg: gr.HTML
    btn_cancel_login: gr.Button
    btn_submit_login: gr.Button

    # Módulo 1: Archivos
    media_input: gr.File

    # Módulo 2: Opciones de motor
    provider_dropdown: gr.Dropdown
    model_dropdown: gr.Dropdown
    diarize_checkbox: gr.Checkbox
    timestamps_dropdown: gr.Dropdown
    provider_info_box: gr.HTML

    # Módulo 3: Credenciales
    guest_keys_panel: gr.Column
    whisperx_notice: gr.HTML
    hf_token_box: gr.Textbox
    deepgram_key_box: gr.Textbox
    assemblyai_key_box: gr.Textbox

    # Botones de acción principales
    btn_clear: gr.Button
    btn_transcribe: gr.Button

    # Módulo 4: Resultados y descargas
    status_badge: gr.HTML
    output_textbox: gr.Textbox
    btn_download: gr.DownloadButton
    btn_download_consolidated: gr.DownloadButton
    btn_copy: gr.Button


def render_navbar() -> tuple[gr.Button, gr.Button]:
    """Renderiza la barra de navegación superior con marca y botón de acceso admin."""
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
    return btn_open_login, btn_logout


def render_login_modal() -> tuple[
    gr.Column, gr.Button, gr.Textbox, gr.HTML, gr.Button, gr.Button
]:
    """Renderiza el diálogo modal flotante de inicio de sesión de administrador."""
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

    return (
        modal_login,
        btn_close_login,
        input_pin,
        login_msg,
        btn_cancel_login,
        btn_submit_login,
    )


def render_media_upload_card() -> gr.File:
    """Renderiza la tarjeta de subida de archivos multimedia con soporte Drag & Drop."""
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
            label="Arrastra o selecciona tus archivos multimedia",
            show_label=False,
            file_types=["audio", "video"],
            file_count="multiple",
            container=False,
            elem_classes=["file-upload-zone"],
        )
        gr.HTML(
            """
            <div class="file-hint">
                <div class="file-hint-formats">
                    Formatos soportados: MP3, WAV, M4A, MP4, MKV, FLAC, OGG, WEBM, MOV, AAC.
                </div>
                <div class="file-hint-optimization">
                    ⚡ Los videos son optimizados localmente en tu navegador sin pérdida de calidad para acelerar la subida.
                </div>
            </div>
            """
        )
    return media_input


def render_options_card() -> tuple[
    gr.Dropdown, gr.Dropdown, gr.Checkbox, gr.Dropdown, gr.HTML
]:
    """Renderiza la tarjeta de selección de motor de transcripción, modelo y opciones."""
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
                allow_custom_value=True,
                min_width=100,
                scale=1,
            )
            model_dropdown = gr.Dropdown(
                choices=PROVIDER_MODEL_CHOICES[ProviderType.DEEPGRAM.value],
                value=DeepgramModel.NOVA_2.value,
                label="Modelo de Transcripción",
                allow_custom_value=True,
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
                choices=TIMESTAMP_CHOICES,
                value=TIMESTAMP_CHOICES[0],
                label="Formato de Marcas de Tiempo",
                allow_custom_value=True,
                min_width=100,
                scale=1,
            )

        provider_info_box = gr.HTML(PROVIDER_INFO_HTML[ProviderType.DEEPGRAM.value])

    return (
        provider_dropdown,
        model_dropdown,
        diarize_checkbox,
        timestamps_dropdown,
        provider_info_box,
    )


def render_credentials_card() -> tuple[
    gr.Column, gr.HTML, gr.Textbox, gr.Textbox, gr.Textbox
]:
    """Renderiza el panel de credenciales de API para modo invitado y avisos informativos."""
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

    return (
        guest_keys_panel,
        whisperx_notice,
        hf_token_box,
        deepgram_key_box,
        assemblyai_key_box,
    )


def render_action_buttons() -> tuple[gr.Button, gr.Button]:
    """Renderiza la fila de botones de acción rápida (Limpiar y Transcribir)."""
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
    return btn_clear, btn_transcribe


def render_results_card() -> tuple[
    gr.HTML, gr.Textbox, gr.DownloadButton, gr.DownloadButton, gr.Button
]:
    """Renderiza la tarjeta del visor de resultados, insignias y botones de descarga."""
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
                label=EMPTY_DOWNLOAD_LABEL,
                value=None,
                interactive=False,
                variant="secondary",
                elem_classes=["btn-secondary"],
                scale=2,
            )
            btn_download_consolidated = gr.DownloadButton(
                label="📄 Descargar Consolidado (.txt)",
                value=None,
                interactive=False,
                visible=False,
                variant="secondary",
                elem_classes=["btn-secondary"],
                scale=2,
            )
            btn_copy = gr.Button(
                "📋 Copiar texto",
                variant="secondary",
                elem_classes=["btn-secondary"],
                interactive=False,
                scale=1,
            )

    return (
        status_badge,
        output_textbox,
        btn_download,
        btn_download_consolidated,
        btn_copy,
    )


def render_footer() -> None:
    """Renderiza el pie de página institucional de la aplicación."""
    gr.HTML(
        """
        <footer class="app-footer">
            Transcriptor Suite · Modelos neuronales ASR con alineación fonética fonema a fonema y diarización de hablantes.
        </footer>
        """
    )


def render_transcriptor_ui(custom_css: str) -> UIComponents:
    """Ensambla el diseño completo de dos columnas de la suite."""
    gr.HTML(f"<style>{custom_css}</style>")

    admin_state = gr.State(value=False)
    batch_cache_state = gr.State(value={"cached_results": {}})

    btn_open_login, btn_logout = render_navbar()
    (
        modal_login,
        btn_close_login,
        input_pin,
        login_msg,
        btn_cancel_login,
        btn_submit_login,
    ) = render_login_modal()

    with gr.Row(elem_classes=["main-body-row"]):
        # Columna Izquierda: Entrada y Opciones
        with gr.Column(scale=5, elem_classes=["left-column"]):
            media_input = render_media_upload_card()
            (
                provider_dropdown,
                model_dropdown,
                diarize_checkbox,
                timestamps_dropdown,
                provider_info_box,
            ) = render_options_card()
            (
                guest_keys_panel,
                whisperx_notice,
                hf_token_box,
                deepgram_key_box,
                assemblyai_key_box,
            ) = render_credentials_card()
            btn_clear, btn_transcribe = render_action_buttons()

        # Columna Derecha: Resultado y Descargas
        with gr.Column(scale=6, elem_classes=["result-column"]):
            (
                status_badge,
                output_textbox,
                btn_download,
                btn_download_consolidated,
                btn_copy,
            ) = render_results_card()

    render_footer()

    return UIComponents(
        admin_state=admin_state,
        batch_cache_state=batch_cache_state,
        btn_open_login=btn_open_login,
        btn_logout=btn_logout,
        modal_login=modal_login,
        btn_close_login=btn_close_login,
        input_pin=input_pin,
        login_msg=login_msg,
        btn_cancel_login=btn_cancel_login,
        btn_submit_login=btn_submit_login,
        media_input=media_input,
        provider_dropdown=provider_dropdown,
        model_dropdown=model_dropdown,
        diarize_checkbox=diarize_checkbox,
        timestamps_dropdown=timestamps_dropdown,
        provider_info_box=provider_info_box,
        guest_keys_panel=guest_keys_panel,
        whisperx_notice=whisperx_notice,
        hf_token_box=hf_token_box,
        deepgram_key_box=deepgram_key_box,
        assemblyai_key_box=assemblyai_key_box,
        btn_clear=btn_clear,
        btn_transcribe=btn_transcribe,
        status_badge=status_badge,
        output_textbox=output_textbox,
        btn_download=btn_download,
        btn_download_consolidated=btn_download_consolidated,
        btn_copy=btn_copy,
    )
