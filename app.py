#!/usr/bin/env python3
"""
Transcriptor Suite - Interfaz Gráfica Aurora Boreal Nórdica (Gradio).
Desarrollada con arquitectura modular, diseño responsivo,
patrón Aurora Boreal Nórdica y gestión limpia de credenciales.
"""

import html
import time
from pathlib import Path
from typing import Any

import gradio as gr

from src.auth import (
    LoginAttemptLimiter,
    LoginRateLimitError,
    extract_client_identifier,
)
from src.config import (
    PROVIDER_MODEL_CHOICES,
    TIMESTAMP_LABEL_MAP,
    ProviderType,
    TimestampMode,
    TranscriptionConfig,
    normalize_provider_name,
)
from src.transcriber import create_transcription_pipeline
from src.ui import (
    COPY_CLIPBOARD_JS,
    EMPTY_DOWNLOAD_LABEL,
    MODAL_CLOSE_JS,
    MODAL_OPEN_JS,
    PROVIDER_INFO_HTML,
    _build_status_badges,
    _disabled_consolidated_button,
    _disabled_download_button,
    _disabled_transcribe_button,
    _enabled_transcribe_button,
    _get_file_fingerprint,
    get_blocks_and_launch_kwargs,
    get_custom_css,
    get_custom_theme,
    get_ui_enhancements_js,
    on_media_files_changed,
    render_transcriptor_ui,
)
from src.utils import (
    TranscriptionError,
    clear_gpu_vram,
    configure_utf8_console,
    create_zip_archive,
    format_duration_human,
    resolve_assemblyai_api_key,
    resolve_deepgram_api_key,
    resolve_hf_token,
    save_consolidated_batch_file,
    setup_logger,
    verify_admin_pin,
)

configure_utf8_console()
logger = setup_logger("whisperx_app")
admin_login_limiter = LoginAttemptLimiter()

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


def _resolve_provider_credentials(
    provider: str,
    enable_diarization: bool,
    hf_token_input: str,
    deepgram_key_input: str,
    assemblyai_key_input: str,
    is_admin_session: bool,
) -> tuple[
    str | None,
    str | None,
    tuple[str, gr.DownloadButton, gr.DownloadButton, str, gr.Button] | None,
]:
    """
    Resuelve y valida las credenciales requeridas según el proveedor seleccionado.
    Retorna (hf_token, api_key, error_tuple_or_none).
    """
    try:
        canonical_provider = normalize_provider_name(provider)
    except ValueError as err:
        raise TranscriptionError(f"Proveedor no soportado: {provider}") from err

    if canonical_provider == ProviderType.WHISPERX.value:
        resolved_hf_token = resolve_hf_token(hf_token_input)
        if enable_diarization and not resolved_hf_token:
            return (
                None,
                None,
                (
                    "⚠️ La diarización con WhisperX requiere un token de Hugging Face.\n\n"
                    "Requisitos para activarla:\n"
                    "1. Acepta los términos de uso en: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                    "2. Acepta los términos en: https://huggingface.co/pyannote/segmentation-3.0\n"
                    "3. Ingresa tu token con permisos de lectura en el campo de Hugging Face.",
                    _disabled_download_button(),
                    _disabled_consolidated_button(),
                    _build_status_badges(
                        '<span class="status-badge status-error">Token HF requerido</span>'
                    ),
                    _enabled_transcribe_button(),
                ),
            )
        return resolved_hf_token, None, None

    if canonical_provider == ProviderType.DEEPGRAM.value:
        if deepgram_key_input and deepgram_key_input.strip():
            resolved_api_key = resolve_deepgram_api_key(deepgram_key_input)
        elif is_admin_session:
            resolved_api_key = resolve_deepgram_api_key()
        else:
            resolved_api_key = None

        if not resolved_api_key:
            return (
                None,
                None,
                (
                    "⚠️ Se requiere una clave de API de Deepgram para utilizar este motor.\n\n"
                    "• Si tienes tu clave personal, ingrésala en el panel de credenciales.\n"
                    "• Si eres administrador, inicia sesión para utilizar las claves del servidor.\n"
                    "• O selecciona 'WhisperX (ZeroGPU / Local)' para transcribir de forma 100% gratuita.",
                    _disabled_download_button(),
                    _disabled_consolidated_button(),
                    _build_status_badges(
                        '<span class="status-badge status-error">API Key requerida</span>'
                    ),
                    _enabled_transcribe_button(),
                ),
            )
        return None, resolved_api_key, None

    if canonical_provider == ProviderType.ASSEMBLYAI.value:
        if assemblyai_key_input and assemblyai_key_input.strip():
            resolved_api_key = resolve_assemblyai_api_key(assemblyai_key_input)
        elif is_admin_session:
            resolved_api_key = resolve_assemblyai_api_key()
        else:
            resolved_api_key = None

        if not resolved_api_key:
            return (
                None,
                None,
                (
                    "⚠️ Se requiere una clave de API de AssemblyAI para utilizar este motor.\n\n"
                    "• Si tienes tu clave personal, ingrésala en el panel de credenciales.\n"
                    "• Si eres administrador, inicia sesión para utilizar las claves del servidor.\n"
                    "• O selecciona 'WhisperX (ZeroGPU / Local)' para transcribir de forma 100% gratuita.",
                    _disabled_download_button(),
                    _disabled_consolidated_button(),
                    _build_status_badges(
                        '<span class="status-badge status-error">API Key requerida</span>'
                    ),
                    _enabled_transcribe_button(),
                ),
            )
        return None, resolved_api_key, None

    raise TranscriptionError(f"Proveedor no soportado: {provider}")


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
    batch_cache: dict | None = None,
    progress=gr.Progress(),
):
    """Procesa uno o múltiples archivos multimedia en cola secuencial con caché diferencial por sesión."""
    current_cache = dict(batch_cache) if isinstance(batch_cache, dict) else {}
    cached_results = dict(current_cache.get("cached_results", {}))
    current_cache["cached_results"] = cached_results

    if not media_file:
        return (
            "⚠️ Por favor selecciona o arrastra al menos un archivo multimedia para comenzar.",
            _disabled_download_button(),
            _disabled_consolidated_button(),
            _build_status_badges(
                '<span class="status-badge status-warning">Sin archivo</span>'
            ),
            _enabled_transcribe_button(),
            current_cache,
        )

    file_list = media_file if isinstance(media_file, list) else [media_file]
    file_list = [f for f in file_list if f]
    if not file_list:
        return (
            "⚠️ Por favor selecciona o arrastra al menos un archivo multimedia para comenzar.",
            _disabled_download_button(),
            _disabled_consolidated_button(),
            _build_status_badges(
                '<span class="status-badge status-warning">Sin archivo</span>'
            ),
            _enabled_transcribe_button(),
            current_cache,
        )

    selected_mode = TIMESTAMP_LABEL_MAP.get(timestamp_option, TimestampMode.SIMPLE)
    start_time = time.perf_counter()

    try:
        resolved_hf_token, resolved_api_key, cred_error = (
            _resolve_provider_credentials(
                provider=provider,
                enable_diarization=enable_diarization,
                hf_token_input=hf_token_input,
                deepgram_key_input=deepgram_key_input,
                assemblyai_key_input=assemblyai_key_input,
                is_admin_session=is_admin_session,
            )
        )
        if cred_error is not None:
            return (*cred_error, current_cache)

        config = TranscriptionConfig.create_default(
            provider=provider,
            model_name=model_name,
            enable_diarization=enable_diarization,
            timestamp_mode=selected_mode,
            hf_token=resolved_hf_token,
            api_key=resolved_api_key,
        )

        total_files = len(file_list)
        pending_items = []
        for item in file_list:
            file_name, fp = _get_file_fingerprint(item)
            cached_entry = cached_results.get(fp)
            if (
                cached_entry
                and getattr(cached_entry[1], "output_path", None)
                and Path(cached_entry[1].output_path).is_file()
            ):
                continue
            pending_items.append((item, file_name, fp))

        failed_files: list[tuple[str, str]] = []
        if pending_items:
            pipeline = create_transcription_pipeline(config)
            pending_count = len(pending_items)

            def make_file_progress(current_idx: int, current_name: str):
                def handler(sub_p: float, sub_msg: str) -> None:
                    overall_p = ((current_idx - 1) + sub_p) / pending_count
                    desc = (
                        f"[{current_idx}/{pending_count} nuevos] {current_name}: {sub_msg}"
                        if pending_count > 1
                        else f"{current_name}: {sub_msg}"
                    )
                    progress(overall_p, desc=desc)

                return handler

            for idx, (item, file_name, fp) in enumerate(pending_items, start=1):
                media_path = item.name if hasattr(item, "name") else str(item)
                on_progress = make_file_progress(idx, file_name)

                try:
                    if config.provider == ProviderType.WHISPERX.value:
                        res = execute_whisperx(pipeline, media_path, on_progress)
                    else:
                        res = pipeline.process(
                            media_path=media_path,
                            progress_callback=on_progress,
                        )
                    cached_results[fp] = (file_name, res)
                except (TranscriptionError, Exception) as err:
                    logger.error("Error transcribiendo archivo %s: %s", file_name, err)
                    failed_files.append((file_name, str(err)))
                    clear_gpu_vram()

        # Reconstruir lista ordenada de resultados correspondientes a file_list
        results: list[tuple[str, Any]] = []
        for item in file_list:
            file_name, fp = _get_file_fingerprint(item)
            if fp in cached_results:
                results.append((file_name, cached_results[fp][1]))

        if not results:
            err_details = "\n".join(f"• {fname}: {err}" for fname, err in failed_files)
            clear_gpu_vram()
            return (
                f"❌ Falló el procesamiento del lote:\n\n{err_details}",
                _disabled_download_button(),
                _disabled_consolidated_button(),
                _build_status_badges(
                    '<span class="status-badge status-error">Fallo en lote</span>'
                ),
                _enabled_transcribe_button(),
                current_cache,
            )

        elapsed_seconds = max(0.1, time.perf_counter() - start_time)
        formatted_duration = format_duration_human(elapsed_seconds)
        if total_files > 1:
            time_badge_html = (
                f'<span class="status-badge status-time">'
                f"⏱️ Procesado en {formatted_duration}"
                f"</span>"
            )
        else:
            time_badge_html = (
                f'<span class="status-badge status-time">'
                f"⏱️ Transcrito en {formatted_duration}"
                f"</span>"
            )

        # Flujo 1: Un solo archivo (Comportamiento directo sin banners extra)
        if total_files == 1:
            _, single_res = results[0]
            detected_lang = getattr(single_res, "detected_language", None)
            lang_info = (
                f"Idioma: {detected_lang.upper()}"
                if detected_lang
                and str(detected_lang).strip().lower() not in {"unknown", "none"}
                else "Completado"
            )
            status_html = _build_status_badges(
                f'<span class="status-badge status-success">✓ {html.escape(lang_info)}</span>',
                time_span=time_badge_html,
            )
            output_label = (
                f"📥 Descargar {single_res.output_path.name}"
                if getattr(single_res, "output_path", None)
                else EMPTY_DOWNLOAD_LABEL
            )
            output_val = (
                str(single_res.output_path)
                if getattr(single_res, "output_path", None)
                else None
            )

            return (
                single_res.formatted_text,
                gr.DownloadButton(
                    label=output_label,
                    value=output_val,
                    interactive=bool(output_val),
                    visible=True,
                    variant="primary",
                    elem_classes=["btn-glow-single"],
                ),
                _disabled_consolidated_button(),
                status_html,
                _disabled_transcribe_button(),
                current_cache,
            )

        # Flujo 2: Múltiples archivos en lote (Empaquetado ZIP y texto consolidado)
        blocks: list[str] = []
        for file_name, res in results:
            lang = getattr(res, "detected_language", None)
            lang_tag = (
                f" [Idioma: {lang.upper()}]"
                if lang and str(lang).strip().lower() not in {"unknown", "none"}
                else ""
            )
            header = f"{'=' * 60}\n📁 {file_name}{lang_tag}\n{'=' * 60}"
            blocks.append(f"{header}\n{res.formatted_text}")

        if failed_files:
            err_lines = [
                f"{'=' * 60}\n⚠️ ARCHIVOS NO PROCESADOS (ERRORES)\n{'=' * 60}"
            ]
            for file_name, err_msg in failed_files:
                err_lines.append(f"• {file_name}: {err_msg}")
            blocks.append("\n".join(err_lines))

        combined_text = "\n\n".join(blocks)

        valid_output_paths = [
            res.output_path
            for _, res in results
            if getattr(res, "output_path", None) and Path(res.output_path).is_file()
        ]

        if valid_output_paths:
            batch_dir = valid_output_paths[0].parent
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            consolidated_file = save_consolidated_batch_file(
                content=combined_text,
                output_dir=batch_dir,
                timestamp_str=timestamp_str,
            )
            zip_files = list(valid_output_paths)
            if consolidated_file.is_file():
                zip_files.append(consolidated_file)
            zip_path = batch_dir / f"transcripciones_lote_{timestamp_str}.zip"
            create_zip_archive(zip_files, zip_path)

            output_val = str(zip_path)
            output_label = (
                f"📦 Descargar Lote ({len(valid_output_paths)} archivos) (.zip)"
            )
            cons_val = str(consolidated_file)
            cons_label = "📄 Descargar Consolidado (.txt)"
            cons_interactive = True
            cons_visible = True
            zip_variant = "primary"
            cons_variant = "primary"
            zip_classes = ["btn-glow-zip"]
            cons_classes = ["btn-glow-txt"]
        else:
            output_val = None
            output_label = EMPTY_DOWNLOAD_LABEL
            cons_val = None
            cons_label = "📄 Descargar Consolidado (.txt)"
            cons_interactive = False
            cons_visible = False
            zip_variant = "secondary"
            cons_variant = "secondary"
            zip_classes = ["btn-secondary"]
            cons_classes = ["btn-secondary"]

        cached_count = len(file_list) - len(pending_items)
        if not failed_files:
            if cached_count > 0 and len(pending_items) > 0:
                badge_label = f"✓ {len(results)} listos ({len(pending_items)} nuevos, {cached_count} en caché)"
            else:
                badge_label = f"✓ {len(results)} archivos procesados"
            badge_class = "status-success"
        else:
            badge_label = f"✓ {len(results)}/{total_files} procesados ({len(failed_files)} con error)"
            badge_class = "status-warning"

        status_html = _build_status_badges(
            f'<span class="status-badge {badge_class}">{html.escape(badge_label)}</span>',
            time_span=time_badge_html,
        )

        return (
            combined_text,
            gr.DownloadButton(
                label=output_label,
                value=output_val,
                interactive=bool(output_val),
                visible=True,
                variant=zip_variant,
                elem_classes=zip_classes,
            ),
            gr.DownloadButton(
                label=cons_label,
                value=cons_val,
                interactive=cons_interactive,
                visible=cons_visible,
                variant=cons_variant,
                elem_classes=cons_classes,
            ),
            status_html,
            (
                _disabled_transcribe_button()
                if valid_output_paths
                else _enabled_transcribe_button()
            ),
            current_cache,
        )

    except TranscriptionError as e:
        clear_gpu_vram()
        logger.error("Error en procesamiento web: %s", e)
        return (
            f"❌ Error de transcripción:\n{e}",
            _disabled_download_button(),
            _disabled_consolidated_button(),
            _build_status_badges(
                '<span class="status-badge status-error">Error</span>'
            ),
            _enabled_transcribe_button(),
            current_cache,
        )
    except Exception as e:
        clear_gpu_vram()
        logger.exception("Error inesperado en aplicación web")
        return (
            f"❌ Ocurrió un error inesperado durante el procesamiento:\n{e}",
            _disabled_download_button(),
            _disabled_consolidated_button(),
            _build_status_badges(
                '<span class="status-badge status-error">Error inesperado</span>'
            ),
            _enabled_transcribe_button(),
            current_cache,
        )


def update_guest_api_panel(
    selected_provider: str, is_diarize: bool, is_admin: bool
):
    """Controla la visibilidad del panel de claves de acuerdo a si es invitado o administrador."""
    if is_admin:
        return (
            gr.update(
                visible=False, elem_classes=["module-card", "keys-card", "hide"]
            ),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    if selected_provider not in PROVIDER_MODEL_CHOICES:
        selected_provider = ProviderType.ASSEMBLYAI.value

    is_wx = selected_provider == ProviderType.WHISPERX.value
    is_dg = selected_provider == ProviderType.DEEPGRAM.value
    is_aai = selected_provider == ProviderType.ASSEMBLYAI.value

    show_notice = is_wx and (not is_diarize)
    show_hf = is_wx and is_diarize
    show_dg = is_dg
    show_aai = is_aai

    return (
        gr.update(visible=True, elem_classes=["module-card", "keys-card"]),
        gr.update(visible=show_notice),
        gr.update(visible=show_hf),
        gr.update(visible=show_dg),
        gr.update(visible=show_aai),
    )


def on_provider_selected(
    selected_provider: str, is_diarize: bool, is_admin: bool
):
    """Actualiza modelos disponibles, panel de claves y descripción."""
    if selected_provider not in PROVIDER_MODEL_CHOICES:
        selected_provider = ProviderType.ASSEMBLYAI.value
    choices = PROVIDER_MODEL_CHOICES.get(selected_provider, [])
    default_val = choices[0][1] if choices else ""
    panel_vis = update_guest_api_panel(selected_provider, is_diarize, is_admin)

    return (
        gr.update(choices=choices, value=default_val),
        gr.update(value=PROVIDER_INFO_HTML.get(selected_provider, "")),
        *panel_vis,
    )


def open_login_dialog():
    return gr.update(visible=True), "", gr.update(value="", visible=False)


def close_login_dialog():
    return gr.update(visible=False), "", gr.update(value="", visible=False)


def process_admin_login(
    pin: str,
    current_provider: str,
    is_diarize: bool,
    request: gr.Request = None,
):
    client = extract_client_identifier(request)
    error_message = "✗ PIN incorrecto o no configurado en el servidor."
    try:
        authenticated = admin_login_limiter.authenticate(
            pin, client, verify_admin_pin
        )
    except LoginRateLimitError as err:
        authenticated = False
        error_message = str(err)
    if authenticated:
        return (
            True,  # admin_state
            gr.update(visible=False),  # modal_login
            gr.update(visible=False),  # btn_open_login
            gr.update(visible=True),  # btn_logout
            gr.update(
                visible=False, elem_classes=["module-card", "keys-card", "hide"]
            ),  # guest_keys_panel
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            "",  # limpia input_pin
        )
    else:
        err_html = (
            '<div style="color: #F87171; font-size: 0.76rem; margin-top: 6px;">'
            f"{html.escape(error_message)}</div>"
        )
        panel_vis = update_guest_api_panel(current_provider, is_diarize, False)
        return (
            False,  # admin_state
            gr.update(visible=True),  # modal_login sigue abierto
            gr.update(visible=True),  # btn_open_login
            gr.update(visible=False),  # btn_logout
            *panel_vis,
            gr.update(value=err_html, visible=True),
            "",
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


def initialize_guest_session(
    current_provider: str = ProviderType.ASSEMBLYAI.value,
    is_diarize: bool = False,
):
    if current_provider not in PROVIDER_MODEL_CHOICES:
        current_provider = ProviderType.ASSEMBLYAI.value
    panel_vis = update_guest_api_panel(current_provider, bool(is_diarize), False)
    return (
        False,
        gr.update(visible=True),
        gr.update(visible=False),
        *panel_vis,
    )


def clear_all():
    return (
        None,
        "",
        _disabled_download_button(),
        _disabled_consolidated_button(),
        _build_status_badges(
            '<span class="status-badge status-warning">En espera</span>'
        ),
        _enabled_transcribe_button(),
        {"cached_results": {}},
    )


# Carga de estilos y tema visual modular
custom_css = get_custom_css()
custom_theme = get_custom_theme()
blocks_kwargs, launch_kwargs = get_blocks_and_launch_kwargs(
    custom_css, custom_theme
)
_ui_enhancements_js = get_ui_enhancements_js()

# Construcción de la aplicación Gradio
with gr.Blocks(**blocks_kwargs) as demo:
    ui = render_transcriptor_ui(custom_css)

    # ── Enlace de Eventos y Reactividad ───────────────────────

    # Evento de inicialización de sesión
    demo.load(
        fn=initialize_guest_session,
        inputs=None,
        outputs=[
            ui.admin_state,
            ui.btn_open_login,
            ui.btn_logout,
            ui.guest_keys_panel,
            ui.whisperx_notice,
            ui.hf_token_box,
            ui.deepgram_key_box,
            ui.assemblyai_key_box,
        ],
        js=_ui_enhancements_js,
        show_progress="hidden",
    )

    # Eventos de cambio de proveedor y diarización
    ui.provider_dropdown.change(
        fn=on_provider_selected,
        inputs=[ui.provider_dropdown, ui.diarize_checkbox, ui.admin_state],
        outputs=[
            ui.model_dropdown,
            ui.provider_info_box,
            ui.guest_keys_panel,
            ui.whisperx_notice,
            ui.hf_token_box,
            ui.deepgram_key_box,
            ui.assemblyai_key_box,
        ],
        show_progress="hidden",
    )

    ui.diarize_checkbox.change(
        fn=update_guest_api_panel,
        inputs=[ui.provider_dropdown, ui.diarize_checkbox, ui.admin_state],
        outputs=[
            ui.guest_keys_panel,
            ui.whisperx_notice,
            ui.hf_token_box,
            ui.deepgram_key_box,
            ui.assemblyai_key_box,
        ],
        show_progress="hidden",
    )

    # Eventos del diálogo de login de Administrador
    ui.btn_open_login.click(
        fn=open_login_dialog,
        outputs=[ui.modal_login, ui.input_pin, ui.login_msg],
        js=MODAL_OPEN_JS,
        show_progress="hidden",
    )
    ui.btn_close_login.click(
        fn=close_login_dialog,
        outputs=[ui.modal_login, ui.input_pin, ui.login_msg],
        js=MODAL_CLOSE_JS,
        show_progress="hidden",
    )
    ui.btn_cancel_login.click(
        fn=close_login_dialog,
        outputs=[ui.modal_login, ui.input_pin, ui.login_msg],
        js=MODAL_CLOSE_JS,
        show_progress="hidden",
    )
    ui.btn_submit_login.click(
        fn=process_admin_login,
        inputs=[ui.input_pin, ui.provider_dropdown, ui.diarize_checkbox],
        outputs=[
            ui.admin_state,
            ui.modal_login,
            ui.btn_open_login,
            ui.btn_logout,
            ui.guest_keys_panel,
            ui.whisperx_notice,
            ui.hf_token_box,
            ui.deepgram_key_box,
            ui.assemblyai_key_box,
            ui.login_msg,
            ui.input_pin,
        ],
        show_progress="hidden",
    )
    ui.input_pin.submit(
        fn=process_admin_login,
        inputs=[ui.input_pin, ui.provider_dropdown, ui.diarize_checkbox],
        outputs=[
            ui.admin_state,
            ui.modal_login,
            ui.btn_open_login,
            ui.btn_logout,
            ui.guest_keys_panel,
            ui.whisperx_notice,
            ui.hf_token_box,
            ui.deepgram_key_box,
            ui.assemblyai_key_box,
            ui.login_msg,
            ui.input_pin,
        ],
        show_progress="hidden",
    )

    # Evento de Logout de Administrador
    ui.btn_logout.click(
        fn=process_admin_logout,
        inputs=[ui.provider_dropdown, ui.diarize_checkbox],
        outputs=[
            ui.admin_state,
            ui.btn_open_login,
            ui.btn_logout,
            ui.guest_keys_panel,
            ui.whisperx_notice,
            ui.hf_token_box,
            ui.deepgram_key_box,
            ui.assemblyai_key_box,
        ],
        show_progress="hidden",
    )

    # Evento de limpiar campos
    ui.btn_clear.click(
        fn=clear_all,
        outputs=[
            ui.media_input,
            ui.output_textbox,
            ui.btn_download,
            ui.btn_download_consolidated,
            ui.status_badge,
            ui.btn_transcribe,
            ui.batch_cache_state,
        ],
        show_progress="hidden",
    )
    ui.media_input.clear(
        fn=clear_all,
        outputs=[
            ui.media_input,
            ui.output_textbox,
            ui.btn_download,
            ui.btn_download_consolidated,
            ui.status_badge,
            ui.btn_transcribe,
            ui.batch_cache_state,
        ],
        show_progress="hidden",
    )

    # Detección de archivos multimedia añadidos para reactivar transcripción diferencial
    ui.media_input.change(
        fn=on_media_files_changed,
        inputs=[ui.media_input, ui.batch_cache_state],
        outputs=[ui.btn_transcribe],
        show_progress="hidden",
    )

    # Copiar al portapapeles
    ui.btn_copy.click(
        fn=None,
        inputs=[ui.output_textbox],
        js=COPY_CLIPBOARD_JS,
    )

    # Activar/desactivar botón Copiar texto de forma reactiva
    ui.output_textbox.change(
        fn=lambda t: gr.Button(interactive=bool(t and t.strip())),
        inputs=[ui.output_textbox],
        outputs=[ui.btn_copy],
        show_progress="hidden",
    )

    # Iniciar Transcripción
    ui.btn_transcribe.click(
        fn=process_transcription,
        inputs=[
            ui.media_input,
            ui.provider_dropdown,
            ui.model_dropdown,
            ui.diarize_checkbox,
            ui.timestamps_dropdown,
            ui.hf_token_box,
            ui.deepgram_key_box,
            ui.assemblyai_key_box,
            ui.admin_state,
            ui.batch_cache_state,
        ],
        outputs=[
            ui.output_textbox,
            ui.btn_download,
            ui.btn_download_consolidated,
            ui.status_badge,
            ui.btn_transcribe,
            ui.batch_cache_state,
        ],
    )


if __name__ == "__main__":
    demo.launch(**launch_kwargs)
