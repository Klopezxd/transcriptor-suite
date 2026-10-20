#!/usr/bin/env python3
"""
Transcriptor Suite - Interfaz Gráfica Aurora Boreal Nórdica (Gradio).
Desarrollada desde cero con arquitectura modular, diseño responsivo,
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
    TIMESTAMP_CHOICES,
    TIMESTAMP_LABEL_MAP,
    DeepgramModel,
    ProviderType,
    TimestampMode,
    TranscriptionConfig,
    normalize_provider_name,
)
from src.transcriber import create_transcription_pipeline
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


EMPTY_DOWNLOAD_LABEL = "📥 Descargar Documento (.txt)"

PROVIDER_INFO_HTML: dict[str, str] = {
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


def _build_status_badges(status_span: str, time_span: str = "") -> str:
    return f'<div class="status-badges-group">{time_span}{status_span}</div>'


def _disabled_download_button() -> gr.DownloadButton:
    return gr.DownloadButton(
        label=EMPTY_DOWNLOAD_LABEL,
        value=None,
        interactive=False,
        visible=True,
        variant="secondary",
        elem_classes=["btn-secondary"],
    )


def _disabled_consolidated_button() -> gr.DownloadButton:
    return gr.DownloadButton(
        label="📄 Descargar Consolidado (.txt)",
        value=None,
        interactive=False,
        visible=False,
        variant="secondary",
        elem_classes=["btn-secondary"],
    )


def _enabled_transcribe_button(value: str = "⚡ Iniciar Transcripción") -> gr.Button:
    return gr.Button(
        value=value,
        variant="primary",
        interactive=True,
        elem_classes=["btn-primary"],
    )


def _disabled_transcribe_button(value: str = "⚡ Iniciar Transcripción") -> gr.Button:
    return gr.Button(
        value=value,
        variant="secondary",
        interactive=False,
        elem_classes=["btn-primary", "btn-transcribe-disabled"],
    )


def _get_file_fingerprint(item: Any) -> tuple[str, str]:
    """Retorna (nombre_archivo, huella_unica) basada en el nombre base y tamaño en bytes."""
    media_path = item.name if hasattr(item, "name") else str(item)
    p = Path(media_path)
    file_name = p.name
    try:
        size = p.stat().st_size
    except OSError:
        size = 0
    return file_name, f"{file_name}_{size}"


def on_media_files_changed(media_files, batch_cache: dict | None = None) -> gr.Button:
    """
    Detecta si se agregaron archivos nuevos a la cola y reactiva el botón de transcripción
    mostrando cuántos archivos nuevos están pendientes de transcribir.
    """
    cache = batch_cache if isinstance(batch_cache, dict) else {}
    cached_results = cache.get("cached_results", {})

    file_list = media_files if isinstance(media_files, list) else ([media_files] if media_files else [])
    file_list = [f for f in file_list if f]

    if not file_list:
        if cached_results:
            return _disabled_transcribe_button()
        return _enabled_transcribe_button("⚡ Iniciar Transcripción")

    new_files_count = 0
    for item in file_list:
        _, fp = _get_file_fingerprint(item)
        cached_entry = cached_results.get(fp)
        if not cached_entry or not (
            getattr(cached_entry[1], "output_path", None)
            and Path(cached_entry[1].output_path).is_file()
        ):
            new_files_count += 1

    if new_files_count > 0:
        if cached_results:
            label = (
                f"⚡ Transcribir {new_files_count} archivo nuevo"
                if new_files_count == 1
                else f"⚡ Transcribir {new_files_count} archivos nuevos"
            )
        else:
            label = "⚡ Iniciar Transcripción"
        return _enabled_transcribe_button(label)
    else:
        # Todos los archivos en el componente ya están transcritos
        return _disabled_transcribe_button()


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
            _build_status_badges('<span class="status-badge status-warning">Sin archivo</span>'),
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
            _build_status_badges('<span class="status-badge status-warning">Sin archivo</span>'),
            _enabled_transcribe_button(),
            current_cache,
        )

    selected_mode = TIMESTAMP_LABEL_MAP.get(timestamp_option, TimestampMode.SIMPLE)
    start_time = time.perf_counter()

    try:
        resolved_hf_token, resolved_api_key, cred_error = _resolve_provider_credentials(
            provider=provider,
            enable_diarization=enable_diarization,
            hf_token_input=hf_token_input,
            deepgram_key_input=deepgram_key_input,
            assemblyai_key_input=assemblyai_key_input,
            is_admin_session=is_admin_session,
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
                _build_status_badges('<span class="status-badge status-error">Fallo en lote</span>'),
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
                if detected_lang and str(detected_lang).strip().lower() not in {"unknown", "none"}
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
            output_val = str(single_res.output_path) if getattr(single_res, "output_path", None) else None

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
            err_lines = [f"{'=' * 60}\n⚠️ ARCHIVOS NO PROCESADOS (ERRORES)\n{'=' * 60}"]
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
            output_label = f"📦 Descargar Lote ({len(valid_output_paths)} archivos) (.zip)"
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
            _disabled_transcribe_button() if valid_output_paths else _enabled_transcribe_button(),
            current_cache,
        )

    except TranscriptionError as e:
        clear_gpu_vram()
        logger.error("Error en procesamiento web: %s", e)
        return (
            f"❌ Error de transcripción:\n{e}",
            _disabled_download_button(),
            _disabled_consolidated_button(),
            _build_status_badges('<span class="status-badge status-error">Error</span>'),
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
            _build_status_badges('<span class="status-badge status-error">Error inesperado</span>'),
            _enabled_transcribe_button(),
            current_cache,
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
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        overflow: visible !important;
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

    /* En modo Admin (cuando Credenciales de API está oculto), dar más aire natural a Motor y Opciones y expandir Archivo Multimedia a pantalla completa */
    body.admin-mode .left-column,
    .left-column.admin-mode,
    .left-column:has(.keys-card.hide),
    .left-column:has(.keys-card[style*="display: none"]),
    .left-column:not(:has(.keys-card:not(.hide):not([style*="display: none"]))) {
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        height: 100% !important;
        min-height: 100% !important;
    }

    body.admin-mode .options-card,
    .left-column.admin-mode .options-card,
    .left-column:has(.keys-card.hide) .options-card,
    .left-column:has(.keys-card[style*="display: none"]) .options-card,
    .left-column:not(:has(.keys-card:not(.hide):not([style*="display: none"]))) .options-card,
    .gradio-container .column.left-column:not(:has(.keys-card:not(.hide):not([style*="display: none"]))) .options-card {
        padding: clamp(12px, 2vh, 24px) clamp(16px, 1.7vw, 26px) !important;
        gap: clamp(8px, 1.6vh, 18px) !important;
        flex: 0 0 auto !important;
    }

    body.admin-mode .upload-card,
    .left-column.admin-mode .upload-card,
    .left-column:has(.keys-card.hide) .upload-card,
    .left-column:has(.keys-card[style*="display: none"]) .upload-card,
    .left-column:not(:has(.keys-card:not(.hide):not([style*="display: none"]))) .upload-card,
    .gradio-container .column.left-column:not(:has(.keys-card:not(.hide):not([style*="display: none"]))) .upload-card {
        flex: 1 1 0% !important;
        height: 100% !important;
        min-height: 200px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }

    body.admin-mode .upload-card .file-upload-zone,
    .left-column.admin-mode .upload-card .file-upload-zone,
    body.admin-mode .upload-card .file-preview-holder,
    .left-column.admin-mode .upload-card .file-preview-holder {
        flex: 1 1 0% !important;
        height: 100% !important;
        min-height: 120px !important;
    }

    body.admin-mode .actions-row,
    .left-column.admin-mode .actions-row,
    .left-column:has(.keys-card.hide) .actions-row,
    .left-column:has(.keys-card[style*="display: none"]) .actions-row,
    .left-column:not(:has(.keys-card:not(.hide):not([style*="display: none"]))) .actions-row {
        margin-top: auto !important;
        flex: 0 0 auto !important;
    }

    body.admin-mode .keys-card,
    .keys-card.hide,
    .keys-card[style*="display: none"],
    .gradio-container .column.keys-card.hide,
    .gradio-container .column.keys-card[style*="display: none"] {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
    }

    body:not(.admin-mode) .keys-card:not(.hide):not([style*="display: none"]),
    .keys-card:not(.hide):not([style*="display: none"]),
    .gradio-container .column.keys-card:not(.hide):not([style*="display: none"]) {
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

/* ── MODULO 1: ARRASTRAR ARCHIVO (RESPONSIVO Y ADAPTABLE A TODO EL ALTO DISPONIBLE) ── */
.file-upload-zone,
.upload-card .file-upload-zone,
.upload-card .file-upload-zone > div:not([data-testid="status-tracker"]):not(.file-preview-holder):not(.icon-button-wrapper) {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    width: 100% !important;
    flex: 1 1 0% !important;
    min-height: 48px !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    position: relative !important;
}

/* Anula el min-height: var(--size-60) (240px) interno de Gradio 6 en .wrap.svelte-1vmd51o y permite estiramiento */
.file-upload-zone .wrap:not([data-testid="status-tracker"]),
.file-upload-zone [data-upload-density],
.file-upload-zone .icon-wrap {
    min-height: 0 !important;
    height: 100% !important;
    flex: 1 1 0% !important;
    display: flex !important;
    flex-direction: column !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
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
.file-upload-zone [aria-label*="upload"]:hover,
.file-upload-zone.drag-over-active,
.upload-card.drag-over-active,
.upload-card:has(.drag-over-active) {
    border-color: var(--aurora-green) !important;
    background: rgba(0, 255, 135, 0.08) !important;
    box-shadow: 0 0 18px rgba(0, 255, 135, 0.35) !important;
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

/* Estado con archivo cargado: tarjeta adaptable a toda la altura disponible, con barrita Aurora y sin scroll nativo duplicado */
.file-upload-zone,
.file-upload-zone > div:not(.icon-button-wrapper),
.file-upload-zone div:has(> .file-preview-holder),
.file-upload-zone div:has(> table.file-preview),
.file-upload-zone .file-preview-holder {
    flex: 1 1 0% !important;
    height: 100% !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    box-sizing: border-box !important;
}

/* ── LISTA DE ARCHIVOS CON BARRA LATERAL NATIVA Y FLUIDA ── */
.file-upload-zone .file-preview-holder {
    position: relative !important;
    background: rgba(6, 14, 29, 0.82) !important;
    border: 1px solid rgba(0, 255, 135, 0.35) !important;
    border-radius: 10px !important;
    box-shadow: inset 0 0 14px rgba(0, 255, 135, 0.06) !important;
    width: 100% !important;
    flex: 1 1 0% !important;
    min-height: 48px !important;
    max-height: none !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 6px 6px 6px 12px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    box-sizing: border-box !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    scrollbar-width: thin !important;
    scrollbar-color: rgba(0, 212, 255, 0.35) rgba(4, 10, 24, 0.5) !important;
    -webkit-overflow-scrolling: touch !important;
    overscroll-behavior: contain !important;
}

/* Barra lateral nativa del navegador para la lista de archivos (suave, funcional y sin neón invasivo) */
.file-upload-zone .file-preview-holder::-webkit-scrollbar {
    width: 6px !important;
    display: block !important;
}

.file-upload-zone .file-preview-holder::-webkit-scrollbar-track {
    background: rgba(4, 10, 24, 0.5) !important;
    border-radius: 4px !important;
}

.file-upload-zone .file-preview-holder::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 255, 0.35) !important;
    border-radius: 4px !important;
    min-height: 24px !important;
}

.file-upload-zone .file-preview-holder::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 212, 255, 0.7) !important;
}

/* 1. Global: Toda barra lateral de navegación con scrolleo vertical en la interfaz */
*::-webkit-scrollbar {
    width: 8px !important;
    height: 8px !important;
    display: block !important;
}

*::-webkit-scrollbar-track {
    background: rgba(4, 10, 24, 0.6) !important;
    border-radius: 6px !important;
    border: 1px solid rgba(0, 212, 255, 0.15) !important;
}

*::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 255, 0.35) !important;
    border-radius: 6px !important;
    border: 1px solid rgba(0, 212, 255, 0.4) !important;
    min-height: 28px !important;
}

*::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 212, 255, 0.65) !important;
    border-color: rgba(0, 212, 255, 0.7) !important;
}

*::-webkit-scrollbar-corner {
    background: transparent !important;
}

*::-webkit-scrollbar-button {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

/* Fallback Firefox */
@supports (-moz-appearance: none) {
    * {
        scrollbar-width: thin !important;
        scrollbar-color: rgba(0, 212, 255, 0.35) rgba(4, 10, 24, 0.6) !important;
    }
}

/* Anular scroll redundante en la tabla interna para que el scroll nativo ocurra en .file-preview-holder */
.file-upload-zone table.file-preview,
.file-upload-zone .file-preview,
.file-upload-zone table.file-preview tbody {
    width: 100% !important;
    max-height: none !important;
    height: auto !important;
    display: table !important;
    table-layout: fixed !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    overflow: visible !important;
    scrollbar-width: none !important;
}

.file-upload-zone table.file-preview tbody {
    display: table-row-group !important;
}

.file-upload-zone table.file-preview::-webkit-scrollbar,
.file-upload-zone .file-preview::-webkit-scrollbar,
.file-upload-zone table.file-preview tbody::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

.file-upload-zone tr.file {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    height: auto !important;
    min-height: 28px !important;
    gap: 10px !important;
    padding: 4px 6px 4px 0 !important;
    margin: 0 !important;
    border: none !important;
    border-bottom: 1px solid rgba(0, 212, 255, 0.08) !important;
    background: transparent !important;
    box-sizing: border-box !important;
}

.file-upload-zone tr.file:last-child {
    border-bottom: none !important;
}

.file-upload-zone button.label-clear-button {
    background: transparent !important;
    border: none !important;
    color: #F87171 !important;
    opacity: 0.7 !important;
    cursor: pointer !important;
    font-size: 1.1rem !important;
    line-height: 1 !important;
    padding: 2px 6px !important;
    transition: opacity 0.2s ease, transform 0.2s ease !important;
}

.file-upload-zone button.label-clear-button:hover {
    opacity: 1 !important;
    transform: scale(1.15) !important;
    color: #FF5252 !important;
}

.file-upload-zone td.filename {
    display: flex !important;
    align-items: center !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: clamp(0.84rem, 1.6vh, 0.96rem) !important;
    font-weight: 600 !important;
    color: var(--aurora-green) !important;
    padding: 0 !important;
    border: none !important;
    flex: 1 1 auto !important;
    min-width: 0 !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}

.file-upload-zone td.filename .stem,
.file-upload-zone td.filename span {
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}

.file-upload-zone td.download,
.file-upload-zone td.download a {
    display: inline-flex !important;
    align-items: center !important;
    flex-shrink: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: clamp(0.76rem, 1.4vh, 0.88rem) !important;
    font-weight: 600 !important;
    color: #8CE0FF !important;
    text-decoration: none !important;
    padding: 0 !important;
    border: none !important;
}

/* Reubicación limpia de los botones Gradio ([↑] y [X]) en la cabecera superior derecha del módulo */
.upload-card .icon-button-wrapper,
.file-upload-zone .icon-button-wrapper {
    position: absolute !important;
    top: -36px !important;
    right: 0 !important;
    transform: none !important;
    z-index: 35 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    width: auto !important;
    height: 28px !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: flex-end !important;
    gap: 8px !important;
    pointer-events: auto !important;
}

/* Solo los botones directos del wrapper son botones de acción */
.upload-card .icon-button-wrapper > button,
.file-upload-zone .icon-button-wrapper > button {
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    min-height: 28px !important;
    max-width: 28px !important;
    max-height: 28px !important;
    flex: 0 0 28px !important;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 8px !important;
    opacity: 1 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
    position: relative !important;
    overflow: visible !important;
}

.upload-card .icon-button-wrapper button::after,
.file-upload-zone .icon-button-wrapper button::after {
    display: none !important;
    content: none !important;
}

/* El contenedor interno de subida (.upload-container) no debe tener borde ni fondo para no duplicar la caja */
.upload-card .icon-button-wrapper .upload-container,
.file-upload-zone .icon-button-wrapper .upload-container {
    all: unset !important;
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    z-index: 2 !important;
}

.upload-card .icon-button-wrapper > button > div,
.file-upload-zone .icon-button-wrapper > button > div {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    position: relative !important;
}

.upload-card .icon-button-wrapper > button svg,
.file-upload-zone .icon-button-wrapper > button svg {
    width: 15px !important;
    height: 15px !important;
    display: block !important;
    flex-shrink: 0 !important;
    pointer-events: none !important;
}

/* Botón 1: Subida Manual [↑] - Estilo cyan limpio y nítido */
.upload-card .icon-button-wrapper > button:first-child,
.file-upload-zone .icon-button-wrapper > button:first-child {
    background: rgba(0, 212, 255, 0.16) !important;
    border: 1.5px solid rgba(0, 212, 255, 0.8) !important;
    color: #00D4FF !important;
    box-shadow: 0 0 8px rgba(0, 212, 255, 0.35) !important;
}

.upload-card .icon-button-wrapper > button:first-child:hover,
.file-upload-zone .icon-button-wrapper > button:first-child:hover {
    background: rgba(0, 212, 255, 0.32) !important;
    border-color: #00F0FF !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 16px rgba(0, 212, 255, 0.75) !important;
    transform: scale(1.06) !important;
}

.upload-card .icon-button-wrapper > button:first-child svg,
.file-upload-zone .icon-button-wrapper > button:first-child svg,
.upload-card .icon-button-wrapper > button:first-child path,
.file-upload-zone .icon-button-wrapper > button:first-child path,
.upload-card .icon-button-wrapper > button:first-child polyline,
.file-upload-zone .icon-button-wrapper > button:first-child polyline,
.upload-card .icon-button-wrapper > button:first-child line,
.file-upload-zone .icon-button-wrapper > button:first-child line {
    stroke: #00D4FF !important;
    color: #00D4FF !important;
    filter: drop-shadow(0 0 4px rgba(0, 212, 255, 0.85)) !important;
}

.upload-card .icon-button-wrapper > button:first-child:hover svg,
.file-upload-zone .icon-button-wrapper > button:first-child:hover svg,
.upload-card .icon-button-wrapper > button:first-child:hover path,
.file-upload-zone .icon-button-wrapper > button:first-child:hover path {
    stroke: #FFFFFF !important;
    color: #FFFFFF !important;
    filter: drop-shadow(0 0 8px rgba(0, 212, 255, 1)) !important;
}

/* Botón 2: Borrar / Limpiar [X] - En rojo coral limpio y brillante */
.upload-card .icon-button-wrapper > button:last-child,
.file-upload-zone .icon-button-wrapper > button:last-child {
    background: rgba(239, 68, 68, 0.14) !important;
    border: 1.5px solid rgba(239, 68, 68, 0.7) !important;
    color: #F87171 !important;
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.28) !important;
}

.upload-card .icon-button-wrapper > button:last-child:hover,
.file-upload-zone .icon-button-wrapper > button:last-child:hover {
    background: rgba(239, 68, 68, 0.28) !important;
    border-color: #EF4444 !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 16px rgba(239, 68, 68, 0.7) !important;
    transform: scale(1.06) !important;
}

.upload-card .icon-button-wrapper > button:last-child svg,
.file-upload-zone .icon-button-wrapper > button:last-child svg,
.upload-card .icon-button-wrapper > button:last-child path,
.file-upload-zone .icon-button-wrapper > button:last-child path,
.upload-card .icon-button-wrapper > button:last-child line,
.file-upload-zone .icon-button-wrapper > button:last-child line {
    stroke: #F87171 !important;
    color: #F87171 !important;
    filter: drop-shadow(0 0 4px rgba(239, 68, 68, 0.8)) !important;
}

.upload-card .icon-button-wrapper > button:last-child:hover svg,
.file-upload-zone .icon-button-wrapper > button:last-child:hover svg,
.upload-card .icon-button-wrapper > button:last-child:hover path,
.file-upload-zone .icon-button-wrapper > button:last-child:hover path {
    stroke: #FFFFFF !important;
    color: #FFFFFF !important;
    filter: drop-shadow(0 0 8px rgba(239, 68, 68, 1)) !important;
}

/* Overlay de subida de archivos multimedia con indicador visual en tiempo real */
.upload-loading-overlay {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 44px !important;
    background: rgba(5, 11, 26, 0.94) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border: 1.5px solid var(--aurora-green) !important;
    border-radius: 10px !important;
    z-index: 50 !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    padding: 8px 14px !important;
    box-sizing: border-box !important;
    box-shadow: 0 0 20px rgba(0, 255, 135, 0.3) !important;
    animation: fadeInOverlay 0.2s ease-out !important;
}

.upload-spinner {
    width: 22px !important;
    height: 22px !important;
    border: 2.5px solid rgba(0, 212, 255, 0.25) !important;
    border-top-color: var(--aurora-green) !important;
    border-right-color: var(--ice-cyan) !important;
    border-radius: 50% !important;
    animation: spin-pulse 0.8s linear infinite !important;
}

.upload-loading-text {
    font-family: 'Montserrat', sans-serif !important;
    font-size: clamp(0.85rem, 1.7vh, 1.0rem) !important;
    font-weight: 700 !important;
    color: var(--frost-white) !important;
    text-align: center !important;
    max-width: 90% !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}

.upload-loading-subtext {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: clamp(0.72rem, 1.35vh, 0.84rem) !important;
    color: #00D4FF !important;
    text-align: center !important;
}

@keyframes spin-pulse {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@keyframes fadeInOverlay {
    from { opacity: 0; transform: scale(0.98); }
    to { opacity: 1; transform: scale(1); }
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
    order: 1 !important;
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
    transition: width 0.22s ease-out !important;
}

.output-textbox [data-testid="status-tracker"] .meta-text,
.output-textbox [data-testid="status-tracker"] .meta-text-center {
    order: 2 !important;
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

/* ── BOTONES DE DESCARGA ILUMINADOS (GLOW DINÁMICO) ──────────────── */
.btn-glow-single,
.btn-glow-single button,
.btn-glow-single a {
    background: linear-gradient(135deg, #00FF87 0%, #00D4FF 100%) !important;
    color: #050B1A !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: clamp(0.90rem, 1.8vh, 1.08rem) !important;
    border: 1px solid rgba(0, 255, 135, 0.6) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 18px rgba(0, 255, 135, 0.45), 0 0 16px rgba(0, 212, 255, 0.35) !important;
    animation: pulse-glow-single 2.5s infinite ease-in-out !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.btn-glow-single:hover,
.btn-glow-single button:hover,
.btn-glow-single a:hover {
    background: linear-gradient(135deg, #00D4FF 0%, #00FF87 100%) !important;
    color: #050B1A !important;
    box-shadow: 0 6px 26px rgba(0, 255, 135, 0.7), 0 0 22px rgba(0, 212, 255, 0.55) !important;
    transform: translateY(-2px) !important;
}

/* Lote Multiarchivo - Descargar Lote (.zip): Color Aurora Emerald / Verde Neón */
.btn-glow-zip,
.btn-glow-zip button,
.btn-glow-zip a {
    background: linear-gradient(135deg, #00FF87 0%, #00E5FF 100%) !important;
    color: #050B1A !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: clamp(0.90rem, 1.8vh, 1.08rem) !important;
    border: 1px solid rgba(0, 255, 135, 0.7) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(0, 255, 135, 0.5), 0 0 16px rgba(0, 229, 255, 0.35) !important;
    animation: pulse-glow-zip 2.5s infinite ease-in-out !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.btn-glow-zip:hover,
.btn-glow-zip button:hover,
.btn-glow-zip a:hover {
    background: linear-gradient(135deg, #00E5FF 0%, #00FF87 100%) !important;
    color: #050B1A !important;
    box-shadow: 0 6px 28px rgba(0, 255, 135, 0.75), 0 0 24px rgba(0, 229, 255, 0.55) !important;
    transform: translateY(-2px) !important;
}

/* Lote Multiarchivo - Descargar Consolidado (.txt): Color Violet Sky / Glacier Pink Neón */
.btn-glow-txt,
.btn-glow-txt button,
.btn-glow-txt a {
    background: linear-gradient(135deg, #7B2FBE 0%, #FF1493 100%) !important;
    color: #FAFBFC !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: clamp(0.90rem, 1.8vh, 1.08rem) !important;
    border: 1px solid rgba(255, 20, 147, 0.7) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(255, 20, 147, 0.5), 0 0 18px rgba(123, 47, 190, 0.45) !important;
    animation: pulse-glow-txt 2.5s infinite ease-in-out !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.btn-glow-txt:hover,
.btn-glow-txt button:hover,
.btn-glow-txt a:hover {
    background: linear-gradient(135deg, #FF1493 0%, #7B2FBE 100%) !important;
    color: #FAFBFC !important;
    box-shadow: 0 6px 28px rgba(255, 20, 147, 0.75), 0 0 24px rgba(123, 47, 190, 0.6) !important;
    transform: translateY(-2px) !important;
}

@keyframes pulse-glow-single {
    0%, 100% {
        box-shadow: 0 4px 18px rgba(0, 255, 135, 0.45), 0 0 14px rgba(0, 212, 255, 0.3);
    }
    50% {
        box-shadow: 0 6px 26px rgba(0, 255, 135, 0.7), 0 0 22px rgba(0, 212, 255, 0.55);
    }
}

@keyframes pulse-glow-zip {
    0%, 100% {
        box-shadow: 0 4px 18px rgba(0, 255, 135, 0.45), 0 0 14px rgba(0, 229, 255, 0.3);
    }
    50% {
        box-shadow: 0 6px 28px rgba(0, 255, 135, 0.75), 0 0 22px rgba(0, 229, 255, 0.55);
    }
}

@keyframes pulse-glow-txt {
    0%, 100% {
        box-shadow: 0 4px 18px rgba(255, 20, 147, 0.45), 0 0 14px rgba(123, 47, 190, 0.35);
    }
    50% {
        box-shadow: 0 6px 28px rgba(255, 20, 147, 0.7), 0 0 24px rgba(123, 47, 190, 0.55);
    }
}

/* Deshabilitado para botón Iniciar Transcripción */
.btn-primary:disabled,
.btn-primary[disabled],
button:disabled.btn-primary,
button[disabled].btn-primary,
.btn-transcribe-disabled {
    background: rgba(255, 255, 255, 0.04) !important;
    color: rgba(250, 251, 252, 0.25) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
    opacity: 0.45 !important;
    transform: none !important;
    filter: grayscale(1) !important;
}

.btn-secondary:disabled,
.btn-secondary[disabled],
button:disabled.btn-secondary,
button[disabled].btn-secondary {
    opacity: 0.35 !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
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
    batch_cache_state = gr.State(value={"cached_results": {}})

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
                        Formatos soportados: MP3, WAV, M4A, MP4, MKV, FLAC, OGG, WEBM, MOV, AAC (Soporta múltiples archivos en cola).
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
                gr.update(visible=False, elem_classes=["module-card", "keys-card", "hide"]),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        if selected_provider not in PROVIDER_MODEL_CHOICES:
            selected_provider = ProviderType.DEEPGRAM.value

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

    def on_provider_selected(selected_provider: str, is_diarize: bool, is_admin: bool):
        """Actualiza modelos disponibles, panel de claves y descripción."""
        if selected_provider not in PROVIDER_MODEL_CHOICES:
            selected_provider = ProviderType.DEEPGRAM.value
        choices = PROVIDER_MODEL_CHOICES.get(selected_provider, [])
        default_val = choices[0][1] if choices else ""
        panel_vis = update_guest_api_panel(selected_provider, is_diarize, is_admin)

        return (
            gr.update(choices=choices, value=default_val),
            gr.update(value=PROVIDER_INFO_HTML.get(selected_provider, "")),
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

    def process_admin_login(
        pin: str, current_provider: str, is_diarize: bool, request: gr.Request = None
    ):
        client = extract_client_identifier(request)
        error_message = "✗ PIN incorrecto o no configurado en el servidor."
        try:
            authenticated = admin_login_limiter.authenticate(pin, client, verify_admin_pin)
        except LoginRateLimitError as err:
            authenticated = False
            error_message = str(err)
        if authenticated:
            return (
                True,  # admin_state
                gr.update(visible=False),  # modal_login
                gr.update(visible=False),  # btn_open_login
                gr.update(visible=True),  # btn_logout
                gr.update(visible=False, elem_classes=["module-card", "keys-card", "hide"]),  # guest_keys_panel
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
                f'{html.escape(error_message)}</div>'
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

    # La carga inicial siempre requiere autenticarse mediante el botón Admin.
    def initialize_guest_session(
        current_provider: str = ProviderType.DEEPGRAM.value, is_diarize: bool = False
    ):
        if current_provider not in PROVIDER_MODEL_CHOICES:
            current_provider = ProviderType.DEEPGRAM.value
        panel_vis = update_guest_api_panel(current_provider, bool(is_diarize), False)
        return (
            False,
            gr.update(visible=True),
            gr.update(visible=False),
            *panel_vis,
        )

    _ui_enhancements_js = r"""() => {
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

            // ── MODO ADMIN SEGURO (ZERO RECURSIÓN) ──
            let currentIsAdmin = null;
            const updateAdminMode = () => {
                const keysCard = document.querySelector('.keys-card');
                const btnLogout = document.querySelector('.btn-login-compact:not([style*="display: none"])');
                const isAdmin = Boolean(
                    (keysCard && (keysCard.classList.contains('hide') || keysCard.style.display === 'none')) ||
                    (btnLogout && btnLogout.textContent && btnLogout.textContent.includes('Salir'))
                );
                if (isAdmin !== currentIsAdmin) {
                    currentIsAdmin = isAdmin;
                    document.body.classList.toggle('admin-mode', isAdmin);
                    document.documentElement.classList.toggle('admin-mode', isAdmin);
                    const leftCol = document.querySelector('.left-column');
                    if (leftCol) leftCol.classList.toggle('admin-mode', isAdmin);
                    syncViewport();
                }
            };

            // Limpieza de cualquier rastro previo de barra neón sintética
            document.querySelectorAll('.neon-scrollbar-track').forEach((el) => el.remove());

            // Asegurar que cualquier clic en el botón de subida manual active el selector si no fue interceptado
            document.addEventListener('click', (e) => {
                const uploadBtn = e.target.closest('.icon-button-wrapper > button:first-child, .file-upload-zone [aria-label*="Upload"]');
                if (uploadBtn && uploadBtn.closest('.upload-card, .file-upload-zone')) {
                    if (!e.target.matches('input[type="file"]') && !e.target.closest('.upload-container')) {
                        const fileInput = uploadBtn.querySelector('input[type="file"]') || document.querySelector('.file-upload-zone input[type="file"]');
                        if (fileInput) {
                            fileInput.click();
                        }
                    }
                }
            });

            syncViewport();
            hookParentIframe();
            updateAdminMode();
            setTimeout(() => { syncViewport(); hookParentIframe(); updateAdminMode(); }, 120);
            setTimeout(() => { syncViewport(); hookParentIframe(); updateAdminMode(); }, 500);

            let rafId = 0;
            const onViewportChange = () => {
                if (rafId) cancelAnimationFrame(rafId);
                rafId = requestAnimationFrame(() => {
                    syncViewport();
                    hookParentIframe();
                    updateAdminMode();
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

            // Observer reactivo para la columna izquierda (modo admin)
            let colMoRaf = 0;
            const leftCol = document.querySelector('.left-column');
            if (leftCol && 'MutationObserver' in window) {
                const colMo = new MutationObserver(() => {
                    if (colMoRaf) return;
                    colMoRaf = requestAnimationFrame(() => {
                        colMoRaf = 0;
                        updateAdminMode();
                    });
                });
                colMo.observe(leftCol, {
                    childList: true,
                    subtree: true,
                    attributes: true,
                    attributeFilter: ['style', 'class', 'hidden']
                });
            }

            // Escuchar clics en botones de login/logout/modal para actualizar modo admin inmediatamente
            document.addEventListener('click', (e) => {
                const btn = e.target && e.target.closest ? e.target.closest('.btn-login-compact, #btn_submit_login, .btn-login-submit, .btn-modal-close, .btn-secondary') : null;
                if (btn) {
                    setTimeout(updateAdminMode, 150);
                    setTimeout(updateAdminMode, 500);
                    setTimeout(updateAdminMode, 1000);
                }
            }, true);

            // Escuchar tecla Enter en el campo de PIN del modal de login
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && e.target && e.target.matches('input[type="password"]')) {
                    setTimeout(updateAdminMode, 200);
                    setTimeout(updateAdminMode, 600);
                    setTimeout(updateAdminMode, 1200);
                }
            }, true);

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

            // 3. Indicador de Carga y Soporte de Arrastrar y Soltar (Drag & Drop) continuo
            const showUploadOverlay = (zone, count, mbText) => {
                const targetZone = zone || document.querySelector('.file-upload-zone');
                if (!targetZone) return;
                let overlay = targetZone.querySelector('.upload-loading-overlay');
                if (!overlay) {
                    overlay = document.createElement('div');
                    overlay.className = 'upload-loading-overlay';
                    const fileText = count === 1 ? 'archivo multimedia' : `${count} archivos multimedia`;
                    overlay.innerHTML = `
                        <div class="upload-spinner"></div>
                        <div class="upload-loading-text">Subiendo ${fileText}...</div>
                        <div class="upload-loading-subtext">${mbText ? mbText + ' · ' : ''}Transfiriendo al servidor</div>
                    `;
                    targetZone.appendChild(overlay);
                }
            };

            const hideUploadOverlay = () => {
                document.querySelectorAll('.upload-loading-overlay').forEach((el) => el.remove());
            };

            const handleFilesUploadStart = (zone, files) => {
                if (!files || files.length === 0) return;
                let totalBytes = 0;
                for (let i = 0; i < files.length; i++) {
                    totalBytes += files[i].size || 0;
                }
                const mb = totalBytes > 0 ? `${(totalBytes / (1024 * 1024)).toFixed(1)} MB` : '';
                showUploadOverlay(zone, files.length, mb);

                // Auto-cleanup de seguridad tras 2 minutos si la red tarda o hay error
                setTimeout(() => hideUploadOverlay(), 120000);
            };

            // Hook a fetch para ocultar el overlay inmediatamente al finalizar el upload en Gradio
            if (!window.__transcriptorHookedFetch && window.fetch) {
                window.__transcriptorHookedFetch = true;
                const origFetch = window.fetch;
                window.fetch = async function(...args) {
                    const url = typeof args[0] === 'string' ? args[0] : (args[0] && args[0].url ? args[0].url : '');
                    if (url.includes('/upload')) {
                        try {
                            const res = await origFetch.apply(this, args);
                            setTimeout(() => hideUploadOverlay(), 400);
                            return res;
                        } catch (err) {
                            hideUploadOverlay();
                            throw err;
                        }
                    }
                    return origFetch.apply(this, args);
                };
            }

            // Detectar selección de archivos vía botón [ ↑ ]
            document.addEventListener('change', (e) => {
                const target = e.target;
                if (target && target.type === 'file' && target.closest('.file-upload-zone, .upload-card')) {
                    const zone = target.closest('.file-upload-zone, .upload-card');
                    if (target.files && target.files.length > 0) {
                        handleFilesUploadStart(zone, target.files);
                    }
                }
            }, true);

            const getUploadInput = (zone) => {
                if (!zone) return null;
                return zone.querySelector('input[type="file"]') || document.querySelector('.file-upload-zone input[type="file"]');
            };

            const isFileInputTarget = (e) => {
                return e.target && e.target.closest ? e.target.closest('.file-upload-zone, .upload-card') : null;
            };

            document.addEventListener('dragover', (e) => {
                const zone = isFileInputTarget(e);
                if (zone && e.dataTransfer && e.dataTransfer.types && Array.from(e.dataTransfer.types).includes('Files')) {
                    e.preventDefault();
                    e.stopPropagation();
                    e.dataTransfer.dropEffect = 'copy';
                    zone.classList.add('drag-over-active');
                }
            }, true);

            document.addEventListener('dragleave', (e) => {
                const zone = isFileInputTarget(e);
                if (zone && (!e.relatedTarget || !zone.contains(e.relatedTarget))) {
                    zone.classList.remove('drag-over-active');
                }
            }, true);

            document.addEventListener('drop', (e) => {
                const zone = isFileInputTarget(e);
                if (zone) {
                    zone.classList.remove('drag-over-active');
                    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                        e.preventDefault();
                        e.stopPropagation();
                        const fileInput = getUploadInput(zone);
                        if (fileInput) {
                            handleFilesUploadStart(zone, e.dataTransfer.files);
                            const dt = new DataTransfer();
                            for (let i = 0; i < e.dataTransfer.files.length; i++) {
                                dt.items.add(e.dataTransfer.files[i]);
                            }
                            fileInput.files = dt.files;
                            fileInput.dispatchEvent(new Event('change', { bubbles: true }));
                        }
                    }
                }
            }, true);
        }
    }"""

    demo.load(
        fn=initialize_guest_session,
        inputs=None,
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
            _disabled_download_button(),
            _disabled_consolidated_button(),
            _build_status_badges('<span class="status-badge status-warning">En espera</span>'),
            _enabled_transcribe_button(),
            {"cached_results": {}},
        )

    btn_clear.click(
        fn=clear_all,
        outputs=[
            media_input,
            output_textbox,
            btn_download,
            btn_download_consolidated,
            status_badge,
            btn_transcribe,
            batch_cache_state,
        ],
        show_progress="hidden",
    )

    media_input.clear(
        fn=clear_all,
        outputs=[
            media_input,
            output_textbox,
            btn_download,
            btn_download_consolidated,
            status_badge,
            btn_transcribe,
            batch_cache_state,
        ],
        show_progress="hidden",
    )

    # --- Detección de archivos multimedia añadidos para reactivar transcripción diferencial ---
    media_input.change(
        fn=on_media_files_changed,
        inputs=[media_input, batch_cache_state],
        outputs=[btn_transcribe],
        show_progress="hidden",
    )

    # --- Copiar al portapapeles ---
    btn_copy.click(
        fn=None,
        inputs=[output_textbox],
        js="""(text) => {
            if (!text) return;
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text);
            } else {
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.focus();
                ta.select();
                try {
                    document.execCommand('copy');
                } catch (e) {}
                document.body.removeChild(ta);
            }
        }""",
    )

    # --- Activar/desactivar botón Copiar texto de forma reactiva ---
    output_textbox.change(
        fn=lambda t: gr.Button(interactive=bool(t and t.strip())),
        inputs=[output_textbox],
        outputs=[btn_copy],
        show_progress="hidden",
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
            batch_cache_state,
        ],
        outputs=[
            output_textbox,
            btn_download,
            btn_download_consolidated,
            status_badge,
            btn_transcribe,
            batch_cache_state,
        ],
    )


if __name__ == "__main__":
    demo.launch(**launch_kwargs)
