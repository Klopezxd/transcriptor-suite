"""Componentes visuales y módulos de interfaz desacoplados para Gradio."""

from dataclasses import dataclass

import gradio as gr

from src.config import (
    PROVIDER_MODEL_CHOICES,
    TIMESTAMP_CHOICES,
    AssemblyAIModel,
    ProviderType,
)
from src.ui.helpers import (
    EMPTY_DOWNLOAD_LABEL,
    PROVIDER_INFO_HTML,
    _build_status_badges,
)
from src.ui.template import get_transcriptor_html


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


def render_transcriptor_ui(custom_css: str) -> UIComponents:
    """Ensambla el shell visual nativo HTML5 y el contenedor headless bridge con los componentes Gradio."""
    # Inyectar estilos CSS globales e instrucción display: none estricta para el headless bridge
    bridge_style = (
        f"<style>\n"
        f"{custom_css}\n"
        f"#headless-bridge {{ display: none !important; }}\n"
        f"</style>"
    )
    gr.HTML(bridge_style)

    # Renderizar shell HTML5 semántico nativo
    gr.HTML(get_transcriptor_html())

    # Estados reactivos de sesión
    admin_state = gr.State(value=False)
    batch_cache_state = gr.State(value={"cached_results": {}})

    # Contenedor Headless Bridge (oculto visualmente con display: none !important)
    with gr.Column(elem_id="headless-bridge"):
        # Controles de navegación y autenticación
        btn_open_login = gr.Button("Admin", elem_id="bridge_btn_open_login")
        btn_logout = gr.Button("Salir", visible=False, elem_id="bridge_btn_logout")

        with gr.Column(visible=False, elem_id="bridge_modal_login") as modal_login:
            btn_close_login = gr.Button("✕", elem_id="bridge_btn_close_login")
            input_pin = gr.Textbox(
                type="password",
                placeholder="Ingresa PIN...",
                elem_id="bridge_input_pin",
            )
            login_msg = gr.HTML("", visible=False, elem_id="bridge_login_msg")
            btn_cancel_login = gr.Button("Cancelar", elem_id="bridge_btn_cancel_login")
            btn_submit_login = gr.Button("Ingresar", elem_id="bridge_btn_submit_login")

        # Entrada de archivos multimedia
        media_input = gr.File(
            label="Archivos multimedia",
            file_types=["audio", "video"],
            file_count="multiple",
            elem_id="bridge_media_input",
        )

        # Opciones de motor
        provider_dropdown = gr.Dropdown(
            choices=[
                ProviderType.ASSEMBLYAI.value,
                ProviderType.DEEPGRAM.value,
                ProviderType.WHISPERX.value,
            ],
            value=ProviderType.ASSEMBLYAI.value,
            label="Proveedor",
            allow_custom_value=True,
            elem_id="bridge_provider",
        )
        model_dropdown = gr.Dropdown(
            choices=PROVIDER_MODEL_CHOICES[ProviderType.ASSEMBLYAI.value],
            value=AssemblyAIModel.ASSEMBLYAI_STT_3_7.value,
            label="Modelo",
            allow_custom_value=True,
            elem_id="bridge_model",
        )
        diarize_checkbox = gr.Checkbox(
            value=False,
            label="Diarización",
            elem_id="bridge_diarize",
        )
        timestamps_dropdown = gr.Dropdown(
            choices=TIMESTAMP_CHOICES,
            value=TIMESTAMP_CHOICES[0],
            label="Marcas de tiempo",
            allow_custom_value=True,
            elem_id="bridge_timestamps",
        )
        provider_info_box = gr.HTML(
            PROVIDER_INFO_HTML[ProviderType.ASSEMBLYAI.value],
            elem_id="bridge_provider_info",
        )

        # Credenciales de API
        with gr.Column(visible=True, elem_id="bridge_guest_keys_panel") as guest_keys_panel:
            whisperx_notice = gr.HTML(
                "",
                visible=False,
                elem_id="bridge_whisperx_notice",
            )
            hf_token_box = gr.Textbox(
                label="HF Token",
                type="password",
                visible=False,
                elem_id="bridge_hf_token",
            )
            deepgram_key_box = gr.Textbox(
                label="Deepgram API Key",
                type="password",
                visible=False,
                elem_id="bridge_deepgram_key",
            )
            assemblyai_key_box = gr.Textbox(
                label="AssemblyAI API Key",
                type="password",
                visible=True,
                elem_id="bridge_assemblyai_key",
            )

        # Acciones de transcripción
        btn_clear = gr.Button("Limpiar", elem_id="bridge_btn_clear")
        btn_transcribe = gr.Button(
            "⚡ Iniciar Transcripción",
            variant="primary",
            elem_id="btn_transcribe",
            elem_classes=["btn-primary"],
        )

        # Resultados y descargas
        status_badge = gr.HTML(
            _build_status_badges(
                '<span class="status-badge status-warning">En espera</span>'
            ),
            elem_id="bridge_status_badge",
        )
        output_textbox = gr.Textbox(
            show_label=False,
            elem_id="bridge_output_textbox",
        )
        btn_download = gr.DownloadButton(
            label=EMPTY_DOWNLOAD_LABEL,
            value=None,
            interactive=False,
            elem_id="bridge_btn_download",
            elem_classes=["btn-secondary"],
        )
        btn_download_consolidated = gr.DownloadButton(
            label="📄 Descargar Consolidado (.txt)",
            value=None,
            interactive=False,
            visible=False,
            elem_id="bridge_btn_download_consolidated",
            elem_classes=["btn-secondary"],
        )
        btn_copy = gr.Button(
            "Copiar texto",
            interactive=False,
            elem_id="bridge_btn_copy",
        )

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


__all__ = ["UIComponents", "render_transcriptor_ui"]
