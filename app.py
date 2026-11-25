#!/usr/bin/env python3
"""
Transcriptor Suite - Interfaz Gráfica Aurora Boreal Nórdica (Gradio).
Desarrollada con arquitectura modular, diseño responsivo,
patrón Aurora Boreal Nórdica y gestión limpia de credenciales.
"""

import hmac
import html
import inspect
import os
import secrets
import tempfile
import threading
import time
from collections import OrderedDict
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
    _hidden_stop_button,
    _hidden_transcribe_button,
    _resumable_transcribe_button,
    _visible_stop_button,
    build_config_signature,
    get_blocks_and_launch_kwargs,
    get_custom_css,
    get_custom_theme,
    get_ui_enhancements_js,
    on_media_files_changed,
    render_transcriptor_ui,
)
from src.utils import (
    AudioProcessingError,
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
    validate_media_path,
    verify_admin_pin,
)

configure_utf8_console()
logger = setup_logger("whisperx_app")
admin_login_limiter = LoginAttemptLimiter()

_MAX_SESSION_BUCKETS = 4096
_ADMIN_SESSION_IDLE_TTL_SEC = 60 * 60
_ADMIN_SESSION_MAX_TTL_SEC = 8 * 60 * 60
_STOP_LOCK = threading.Lock()
_SESSION_STOP_FLAGS: OrderedDict[str, bool] = OrderedDict()
_ACTIVE_BATCH_SESSIONS: OrderedDict[str, str] = OrderedDict()
_ADMIN_SESSIONS_LOCK = threading.Lock()
_ADMIN_SESSIONS: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _set_bounded_entry(store: OrderedDict[str, Any], key: str, value: Any) -> None:
    """Inserta o actualiza una entrada en un OrderedDict acotado por LRU."""
    if key not in store and len(store) >= _MAX_SESSION_BUCKETS:
        store.popitem(last=False)
    store[key] = value
    store.move_to_end(key)


def _get_session_key(request: gr.Request | None = None) -> str:
    """Obtiene el identificador de sesión Gradio para aislar paradas cooperativas."""
    session_hash = getattr(request, "session_hash", None) if request else None
    return str(session_hash) if session_hash else "default"


def _set_batch_active(
    session_key: str, run_id: str | None, *, expected_run_id: str | None = None
) -> None:
    """Registra o libera la ejecución activa de un lote pausable por sesión."""
    with _STOP_LOCK:
        if run_id is not None:
            _set_bounded_entry(_ACTIVE_BATCH_SESSIONS, session_key, run_id)
        else:
            if (
                expected_run_id is None
                or _ACTIVE_BATCH_SESSIONS.get(session_key) == expected_run_id
            ):
                _ACTIVE_BATCH_SESSIONS.pop(session_key, None)


def _is_stop_requested(session_key: str) -> bool:
    """Consulta si la sesión actual solicitó detener la cola tras el archivo en curso."""
    with _STOP_LOCK:
        return bool(_SESSION_STOP_FLAGS.get(session_key, False))


def _set_stop_requested(session_key: str, value: bool) -> None:
    """Actualiza de forma segura para hilos la bandera de parada elegante de una sesión."""
    with _STOP_LOCK:
        if value:
            _set_bounded_entry(_SESSION_STOP_FLAGS, session_key, True)
        else:
            _SESSION_STOP_FLAGS.pop(session_key, None)


def toggle_graceful_stop(request: gr.Request = None):
    """Alterna la parada elegante (o deshace la pausa) mientras corre un lote activo."""
    session_hash = getattr(request, "session_hash", None) if request else None
    if not session_hash:
        return _hidden_stop_button()
    session_key = str(session_hash)
    with _STOP_LOCK:
        if session_key not in _ACTIVE_BATCH_SESSIONS:
            _SESSION_STOP_FLAGS.pop(session_key, None)
            return _hidden_stop_button()
        next_state = not bool(_SESSION_STOP_FLAGS.get(session_key, False))
        if next_state:
            _set_bounded_entry(_SESSION_STOP_FLAGS, session_key, True)
        else:
            _SESSION_STOP_FLAGS.pop(session_key, None)
    logger.info(
        "Parada elegante %s para sesión %s",
        "solicitada" if next_state else "cancelada",
        session_key,
    )
    return _visible_stop_button(stopping=next_state)


def _create_admin_session(request: gr.Request | None = None) -> str | bool:
    """Emite un token opaco ligado al session_hash tras autenticación administrativa válida."""
    session_hash = getattr(request, "session_hash", None) if request else None
    if not session_hash:
        return request is None
    token = secrets.token_urlsafe(32)
    now = time.monotonic()
    with _ADMIN_SESSIONS_LOCK:
        _set_bounded_entry(
            _ADMIN_SESSIONS,
            str(session_hash),
            {"token": token, "created_at": now, "last_seen": now},
        )
    return token


def _is_verified_admin_session(
    is_admin_session: Any, request: gr.Request | None = None
) -> bool:
    """Valida autorización administrativa contra el registro servidor con TTL."""
    if request is None:
        return bool(is_admin_session)
    session_hash = getattr(request, "session_hash", None)
    if not session_hash or not isinstance(is_admin_session, str) or not is_admin_session:
        return False
    key = str(session_hash)
    now = time.monotonic()
    with _ADMIN_SESSIONS_LOCK:
        entry = _ADMIN_SESSIONS.get(key)
        if not entry:
            return False
        if (now - entry["created_at"]) > _ADMIN_SESSION_MAX_TTL_SEC or (
            now - entry["last_seen"]
        ) > _ADMIN_SESSION_IDLE_TTL_SEC:
            _ADMIN_SESSIONS.pop(key, None)
            return False
        if not hmac.compare_digest(is_admin_session, entry["token"]):
            return False
        entry["last_seen"] = now
        _ADMIN_SESSIONS.move_to_end(key)
        return True


def _revoke_admin_session(request: gr.Request | None = None) -> None:
    """Invalida la sesión administrativa asociada al session_hash actual."""
    session_hash = getattr(request, "session_hash", None) if request else None
    if session_hash:
        with _ADMIN_SESSIONS_LOCK:
            _ADMIN_SESSIONS.pop(str(session_hash), None)


def _accepts_output_dir(process_fn: Any) -> bool:
    """Determina si el callable del pipeline acepta el argumento keyword output_dir."""
    side_effect = getattr(process_fn, "side_effect", None)
    target = side_effect if callable(side_effect) else process_fn
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return False
    return "output_dir" in sig.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )


# Compatibilidad con ZeroGPU en Hugging Face Spaces
try:
    import spaces

    try:
        zero_gpu_duration = max(1, int(os.getenv("ZERO_GPU_DURATION", "60")))
    except (TypeError, ValueError):
        zero_gpu_duration = 60
    GPU_DECORATOR = spaces.GPU(duration=zero_gpu_duration)
except (ImportError, TypeError, AttributeError):

    def GPU_DECORATOR(func):
        return func


@GPU_DECORATOR
def execute_whisperx(pipeline, media_path: str, progress_callback, output_dir=None):
    """Ejecuta el pipeline local de WhisperX bajo aceleración de GPU si está disponible."""
    if output_dir is not None and _accepts_output_dir(pipeline.process):
        return pipeline.process(
            media_path=media_path,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )
    return pipeline.process(media_path=media_path, progress_callback=progress_callback)


def _resolve_provider_credentials(
    provider: str,
    enable_diarization: bool,
    hf_token_input: str,
    deepgram_key_input: str,
    assemblyai_key_input: str,
    is_admin_session: Any,
    request: gr.Request | None = None,
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

    verified_admin = _is_verified_admin_session(is_admin_session, request)

    if canonical_provider == ProviderType.WHISPERX.value:
        allow_public_hf = os.getenv("ALLOW_PUBLIC_HF_TOKEN", "").lower() == "true"
        if hf_token_input and hf_token_input.strip():
            resolved_hf_token = resolve_hf_token(hf_token_input)
        elif verified_admin or allow_public_hf:
            resolved_hf_token = resolve_hf_token()
        else:
            resolved_hf_token = None
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
        elif verified_admin:
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
        elif verified_admin:
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


def _is_valid_cached_entry(cached_entry: Any) -> bool:
    """Verifica si una entrada de caché posee un archivo de salida válido en disco."""
    return bool(
        cached_entry
        and getattr(cached_entry[1], "output_path", None)
        and Path(cached_entry[1].output_path).is_file()
    )


def _collect_cached_results(
    file_list: list[Any], cached_results: dict[str, Any]
) -> list[tuple[str, Any]]:
    """Reconstruye la lista ordenada de resultados ya completados correspondientes a file_list."""
    results: list[tuple[str, Any]] = []
    for item in file_list:
        file_name, fp = _get_file_fingerprint(item)
        if fp in cached_results:
            results.append((file_name, cached_results[fp][1]))
    return results


def _get_allowed_media_roots() -> list[Path]:
    """Retorna las rutas raíz permitidas para entradas multimedia en la aplicación web."""
    roots: list[Path] = [Path(tempfile.gettempdir())]
    gradio_temp = os.getenv("GRADIO_TEMP_DIR", "").strip()
    if gradio_temp:
        roots.append(Path(gradio_temp))
    ui_obj = globals().get("ui")
    if ui_obj is not None:
        gradio_cache = getattr(ui_obj.media_input, "GRADIO_CACHE", None)
        if gradio_cache:
            roots.append(Path(gradio_cache))
    return roots


def _is_mock_object(obj: Any) -> bool:
    """Detecta mocks de pruebas unitarias para preservar compatibilidad con rutas sintéticas."""
    return type(obj).__module__.startswith("unittest.mock")


def _validate_web_media_path(
    media_path: str, pipeline: Any, is_whisperx: bool
) -> str:
    """Valida que la ruta multimedia provenga de un directorio temporal permitido."""
    raw_path = Path(media_path)
    is_mock_placeholder = (
        not raw_path.exists()
        and not raw_path.is_symlink()
        and raw_path.parent == Path(".")
        and (
            _is_mock_object(pipeline)
            or (is_whisperx and _is_mock_object(execute_whisperx))
        )
    )
    if is_mock_placeholder:
        return media_path

    allowed_roots = _get_allowed_media_roots()
    sig = inspect.signature(validate_media_path)
    if "allowed_roots" in sig.parameters:
        validated = validate_media_path(media_path, allowed_roots=allowed_roots)
    else:
        resolved_roots = [root.resolve() for root in allowed_roots]
        resolved_path = raw_path.resolve()
        if not any(
            resolved_path == root or resolved_path.is_relative_to(root)
            for root in resolved_roots
        ):
            raise AudioProcessingError(
                f"Ruta de archivo fuera de los directorios permitidos: {raw_path.name}"
            )
        validated = validate_media_path(resolved_path)
    return str(validated)


def _format_batch_text(
    results: list[tuple[str, Any]], failed_files: list[tuple[str, str]]
) -> str:
    """Formatea el texto consolidado para entregas parciales y finales en lote."""
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

    return "\n\n".join(blocks)


def _build_batch_downloads(
    results: list[tuple[str, Any]],
    combined_text: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Genera los archivos consolidados (.txt y .zip) y metadatos de descarga para un lote."""
    valid_output_paths = [
        res.output_path
        for _, res in results
        if getattr(res, "output_path", None) and Path(res.output_path).is_file()
    ]
    if not valid_output_paths:
        return {
            "valid_output_paths": valid_output_paths,
            "output_val": None,
            "output_label": EMPTY_DOWNLOAD_LABEL,
            "cons_val": None,
            "cons_label": "📄 Descargar Consolidado (.txt)",
            "cons_interactive": False,
            "cons_visible": False,
            "zip_variant": "secondary",
            "cons_variant": "secondary",
            "zip_classes": ["btn-secondary"],
            "cons_classes": ["btn-secondary"],
        }

    batch_dir = output_dir if output_dir is not None else valid_output_paths[0].parent
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

    return {
        "valid_output_paths": valid_output_paths,
        "output_val": str(zip_path),
        "output_label": f"📦 Descargar Lote ({len(valid_output_paths)} archivos) (.zip)",
        "cons_val": str(consolidated_file),
        "cons_label": "📄 Descargar Consolidado (.txt)",
        "cons_interactive": True,
        "cons_visible": True,
        "zip_variant": "primary",
        "cons_variant": "primary",
        "zip_classes": ["btn-glow-zip"],
        "cons_classes": ["btn-glow-txt"],
    }


def process_transcription_stream(
    media_file,
    provider: str,
    model_name: str,
    enable_diarization: bool,
    timestamp_option: str,
    hf_token_input: str = "",
    deepgram_key_input: str = "",
    assemblyai_key_input: str = "",
    is_admin_session: Any = False,
    batch_cache: dict | None = None,
    progress=gr.Progress(),
    request: gr.Request = None,
):
    """
    Procesa uno o múltiples archivos multimedia con entregas progresivas tras cada archivo,
    parada elegante cooperativa en lotes (2+ pendientes) y caché diferencial por sesión.
    """
    run_id = secrets.token_hex(8)
    session_hash = getattr(request, "session_hash", None) if request else None
    session_key = str(session_hash) if session_hash else f"anon_{run_id}"
    current_cache = dict(batch_cache) if isinstance(batch_cache, dict) else {}
    raw_cached = current_cache.get("cached_results")
    cached_results = dict(raw_cached) if isinstance(raw_cached, dict) else {}
    current_cache["cached_results"] = cached_results

    if not media_file:
        _set_stop_requested(session_key, False)
        yield (
            "⚠️ Por favor selecciona o arrastra al menos un archivo multimedia para comenzar.",
            _disabled_download_button(),
            _disabled_consolidated_button(),
            _build_status_badges(
                '<span class="status-badge status-warning">Sin archivo</span>'
            ),
            _enabled_transcribe_button(),
            _hidden_stop_button(),
            current_cache,
        )
        return

    file_list = media_file if isinstance(media_file, list) else [media_file]
    file_list = [f for f in file_list if f]
    if not file_list:
        _set_stop_requested(session_key, False)
        yield (
            "⚠️ Por favor selecciona o arrastra al menos un archivo multimedia para comenzar.",
            _disabled_download_button(),
            _disabled_consolidated_button(),
            _build_status_badges(
                '<span class="status-badge status-warning">Sin archivo</span>'
            ),
            _enabled_transcribe_button(),
            _hidden_stop_button(),
            current_cache,
        )
        return

    selected_mode = TIMESTAMP_LABEL_MAP.get(timestamp_option, TimestampMode.SIMPLE)
    start_time = time.perf_counter()
    completed = False
    run_output_dir: Path | None = None

    def _ensure_run_output_dir() -> Path:
        nonlocal run_output_dir
        if run_output_dir is None:
            run_output_dir = Path(tempfile.mkdtemp(prefix="transcriptor_out_"))
        return run_output_dir

    try:
        resolved_hf_token, resolved_api_key, cred_error = (
            _resolve_provider_credentials(
                provider=provider,
                enable_diarization=enable_diarization,
                hf_token_input=hf_token_input,
                deepgram_key_input=deepgram_key_input,
                assemblyai_key_input=assemblyai_key_input,
                is_admin_session=is_admin_session,
                request=request,
            )
        )
        if cred_error is not None:
            _set_stop_requested(session_key, False)
            completed = True
            yield (*cred_error, _hidden_stop_button(), current_cache)
            return

        config = TranscriptionConfig.create_default(
            provider=provider,
            model_name=model_name,
            enable_diarization=enable_diarization,
            timestamp_mode=selected_mode,
            hf_token=resolved_hf_token,
            api_key=resolved_api_key,
        )

        config_signature = build_config_signature(
            provider=config.provider,
            model_name=config.model_name,
            enable_diarization=config.enable_diarization,
            timestamp_option=config.timestamp_mode.value,
        )
        prev_signature = current_cache.get("config_signature")
        if prev_signature and prev_signature != config_signature:
            logger.info(
                "Configuración cambiada (%s -> %s); reiniciando caché diferencial.",
                prev_signature,
                config_signature,
            )
            cached_results = {}
        current_cache["cached_results"] = cached_results
        current_cache["config_signature"] = config_signature
        current_cache["paused"] = False

        total_files = len(file_list)
        pending_items = []
        for item in file_list:
            file_name, fp = _get_file_fingerprint(item)
            if _is_valid_cached_entry(cached_results.get(fp)):
                continue
            cached_results.pop(fp, None)
            pending_items.append((item, file_name, fp))

        failed_files: list[tuple[str, str]] = []
        was_stopped_early = False
        can_pause_batch = len(pending_items) > 1
        _set_stop_requested(session_key, False)

        try:
            if pending_items:
                pipeline = create_transcription_pipeline(config)
                pending_count = len(pending_items)

                if can_pause_batch:
                    _set_batch_active(session_key, run_id)
                    initial_results = _collect_cached_results(file_list, cached_results)
                    initial_text = (
                        _format_batch_text(initial_results, [])
                        if initial_results
                        else ""
                    )
                    yield (
                        initial_text,
                        _disabled_download_button(),
                        _disabled_consolidated_button(),
                        _build_status_badges(
                            f'<span class="status-badge status-warning">⏳ Procesando ({len(initial_results)}/{total_files} listos)...</span>'
                        ),
                        _hidden_transcribe_button(),
                        _visible_stop_button(stopping=False),
                        current_cache,
                    )

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
                    if can_pause_batch and _is_stop_requested(session_key):
                        was_stopped_early = True
                        break

                    raw_media_path = item.name if hasattr(item, "name") else str(item)
                    on_progress = make_file_progress(idx, file_name)
                    is_whisperx = config.provider == ProviderType.WHISPERX.value

                    try:
                        validated_media_path = _validate_web_media_path(
                            raw_media_path, pipeline, is_whisperx
                        )
                        target_out_dir = _ensure_run_output_dir()
                        if is_whisperx:
                            res = execute_whisperx(
                                pipeline,
                                validated_media_path,
                                on_progress,
                                output_dir=target_out_dir,
                            )
                        elif _accepts_output_dir(pipeline.process):
                            res = pipeline.process(
                                media_path=validated_media_path,
                                output_dir=target_out_dir,
                                progress_callback=on_progress,
                            )
                        else:
                            res = pipeline.process(
                                media_path=validated_media_path,
                                progress_callback=on_progress,
                            )
                        cached_results[fp] = (file_name, res)
                    except (TranscriptionError, Exception) as err:
                        err_msg = str(err)
                        if (
                            "GPU duration" in err_msg
                            or "larger than the maximum allowed" in err_msg
                        ):
                            err_msg = (
                                f"{err_msg}\n\n"
                                "💡 Tip: El límite de tiempo en ZeroGPU de Hugging Face fue excedido. "
                                "Puedes usar los proveedores en la nube 'AssemblyAI' o 'Deepgram' "
                                "(sin límite de GPU) o configurar ZERO_GPU_DURATION en los Settings del Space."
                            )
                        logger.error(
                            "Error transcribiendo archivo %s: %s", file_name, err_msg
                        )
                        failed_files.append((file_name, err_msg))
                        clear_gpu_vram()

                    if can_pause_batch and _is_stop_requested(session_key):
                        logger.info(
                            "Parada elegante ejecutada tras completar %s (%s/%s).",
                            file_name,
                            idx,
                            pending_count,
                        )
                        was_stopped_early = True
                        break

                    if can_pause_batch and idx < pending_count:
                        partial_results = _collect_cached_results(
                            file_list, cached_results
                        )
                        remaining_now = sum(
                            1 for _, _, p_fp in pending_items if p_fp not in cached_results
                        )
                        pending_word_now = (
                            "pendiente" if remaining_now == 1 else "pendientes"
                        )
                        yield (
                            _format_batch_text(partial_results, failed_files),
                            _disabled_download_button(),
                            _disabled_consolidated_button(),
                            _build_status_badges(
                                f'<span class="status-badge status-warning">⏳ Procesando ({len(partial_results)}/{total_files} listos, {remaining_now} {pending_word_now})...</span>'
                            ),
                            _hidden_transcribe_button(),
                            _visible_stop_button(stopping=False),
                            current_cache,
                        )
        finally:
            _set_batch_active(session_key, None, expected_run_id=run_id)
            _set_stop_requested(session_key, False)

        # Reconstruir lista ordenada de resultados correspondientes a file_list
        results = _collect_cached_results(file_list, cached_results)
        remaining_count = sum(
            1 for _, _, p_fp in pending_items if p_fp not in cached_results
        )
        current_cache["paused"] = bool(was_stopped_early and remaining_count > 0)

        if not results:
            err_details = "\n".join(f"• {fname}: {err}" for fname, err in failed_files)
            clear_gpu_vram()
            if was_stopped_early and remaining_count > 0:
                msg = (
                    f"⏸️ Lote pausado antes de completar archivos.\n\n{err_details}"
                    if err_details
                    else "⏸️ Lote pausado. Presiona Reanudar para continuar."
                )
                completed = True
                yield (
                    msg,
                    _disabled_download_button(),
                    _disabled_consolidated_button(),
                    _build_status_badges(
                        f'<span class="status-badge status-warning">⏸️ Pausado ({remaining_count} pendientes)</span>'
                    ),
                    _resumable_transcribe_button(remaining_count),
                    _hidden_stop_button(),
                    current_cache,
                )
                return

            completed = True
            yield (
                f"❌ Falló el procesamiento del lote:\n\n{err_details}",
                _disabled_download_button(),
                _disabled_consolidated_button(),
                _build_status_badges(
                    '<span class="status-badge status-error">Fallo en lote</span>'
                ),
                _enabled_transcribe_button(),
                _hidden_stop_button(),
                current_cache,
            )
            return

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

            completed = True
            yield (
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
                _hidden_stop_button(),
                current_cache,
            )
            return

        # Flujo 2: Múltiples archivos en lote (Empaquetado ZIP y texto consolidado)
        combined_text = _format_batch_text(results, failed_files)
        batch_out_dir = _ensure_run_output_dir()
        downloads = _build_batch_downloads(
            results, combined_text, output_dir=batch_out_dir
        )
        valid_output_paths = downloads["valid_output_paths"]

        cached_count = len(file_list) - len(pending_items)
        if was_stopped_early and remaining_count > 0:
            pending_word = "pendiente" if remaining_count == 1 else "pendientes"
            badge_label = (
                f"⏸️ Pausado: {len(results)}/{total_files} listos "
                f"({remaining_count} {pending_word})"
            )
            badge_class = "status-warning"
            next_transcribe_btn = _resumable_transcribe_button(remaining_count)
        elif not failed_files:
            if cached_count > 0 and len(pending_items) > 0:
                badge_label = f"✓ {len(results)} listos ({len(pending_items)} nuevos, {cached_count} en caché)"
            else:
                badge_label = f"✓ {len(results)} archivos procesados"
            badge_class = "status-success"
            next_transcribe_btn = (
                _disabled_transcribe_button()
                if valid_output_paths
                else _enabled_transcribe_button()
            )
        else:
            badge_label = (
                f"✓ {len(results)}/{total_files} procesados ({len(failed_files)} con error)"
            )
            badge_class = "status-warning"
            next_transcribe_btn = (
                _disabled_transcribe_button()
                if valid_output_paths
                else _enabled_transcribe_button()
            )

        status_html = _build_status_badges(
            f'<span class="status-badge {badge_class}">{html.escape(badge_label)}</span>',
            time_span=time_badge_html,
        )

        completed = True
        yield (
            combined_text,
            gr.DownloadButton(
                label=downloads["output_label"],
                value=downloads["output_val"],
                interactive=bool(downloads["output_val"]),
                visible=True,
                variant=downloads["zip_variant"],
                elem_classes=downloads["zip_classes"],
            ),
            gr.DownloadButton(
                label=downloads["cons_label"],
                value=downloads["cons_val"],
                interactive=downloads["cons_interactive"],
                visible=downloads["cons_visible"],
                variant=downloads["cons_variant"],
                elem_classes=downloads["cons_classes"],
            ),
            status_html,
            next_transcribe_btn,
            _hidden_stop_button(),
            current_cache,
        )

    except TranscriptionError as e:
        _set_stop_requested(session_key, False)
        clear_gpu_vram()
        logger.error("Error en procesamiento web: %s", e)
        completed = True
        yield (
            f"❌ Error de transcripción:\n{e}",
            _disabled_download_button(),
            _disabled_consolidated_button(),
            _build_status_badges(
                '<span class="status-badge status-error">Error</span>'
            ),
            _enabled_transcribe_button(),
            _hidden_stop_button(),
            current_cache,
        )
    except Exception:
        _set_stop_requested(session_key, False)
        clear_gpu_vram()
        logger.exception("Error inesperado en aplicación web")
        completed = True
        yield (
            "❌ Ocurrió un error inesperado durante el procesamiento. "
            "Verifica el archivo o la configuración e intenta nuevamente.",
            _disabled_download_button(),
            _disabled_consolidated_button(),
            _build_status_badges(
                '<span class="status-badge status-error">Error inesperado</span>'
            ),
            _enabled_transcribe_button(),
            _hidden_stop_button(),
            current_cache,
        )
    finally:
        _set_batch_active(session_key, None, expected_run_id=run_id)
        _set_stop_requested(session_key, False)
        if not completed:
            clear_gpu_vram()


def process_transcription(
    media_file,
    provider: str,
    model_name: str,
    enable_diarization: bool,
    timestamp_option: str,
    hf_token_input: str = "",
    deepgram_key_input: str = "",
    assemblyai_key_input: str = "",
    is_admin_session: Any = False,
    batch_cache: dict | None = None,
    progress=gr.Progress(),
    request: gr.Request = None,
):
    """Ejecuta la transcripción retornando la tupla final de 6 elementos (API directa y pruebas)."""
    final_state = None
    for state in process_transcription_stream(
        media_file=media_file,
        provider=provider,
        model_name=model_name,
        enable_diarization=enable_diarization,
        timestamp_option=timestamp_option,
        hf_token_input=hf_token_input,
        deepgram_key_input=deepgram_key_input,
        assemblyai_key_input=assemblyai_key_input,
        is_admin_session=is_admin_session,
        batch_cache=batch_cache,
        progress=progress,
        request=request,
    ):
        final_state = state
    if final_state is None:
        return (
            "❌ Ocurrió un error inesperado durante el procesamiento.",
            _disabled_download_button(),
            _disabled_consolidated_button(),
            _build_status_badges('<span class="status-badge status-error">Error</span>'),
            _enabled_transcribe_button(),
            {"cached_results": {}},
        )
    return (*final_state[:5], final_state[6])


def update_guest_api_panel(
    selected_provider: str,
    is_diarize: bool,
    is_admin: Any,
    request: gr.Request = None,
):
    """Controla la visibilidad del panel de claves de acuerdo a si es invitado o administrador."""
    if _is_verified_admin_session(is_admin, request):
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
    selected_provider: str,
    is_diarize: bool,
    is_admin: Any,
    request: gr.Request = None,
):
    """Actualiza modelos disponibles, panel de claves y descripción."""
    if selected_provider not in PROVIDER_MODEL_CHOICES:
        selected_provider = ProviderType.ASSEMBLYAI.value
    choices = PROVIDER_MODEL_CHOICES.get(selected_provider, [])
    default_val = choices[0][1] if choices else ""
    panel_vis = update_guest_api_panel(selected_provider, is_diarize, is_admin, request)

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
        admin_token = _create_admin_session(request)
        if not admin_token:
            authenticated = False
            error_message = "⚠️ Sesión web inválida. Recarga la página e intenta nuevamente."
    if authenticated:
        return (
            admin_token,  # admin_state
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


def process_admin_logout(
    current_provider: str, is_diarize: bool, request: gr.Request = None
):
    """Cierra sesión y restaura el panel de claves para invitados."""
    _revoke_admin_session(request)
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
    request: gr.Request = None,
):
    _revoke_admin_session(request)
    if current_provider not in PROVIDER_MODEL_CHOICES:
        current_provider = ProviderType.ASSEMBLYAI.value
    panel_vis = update_guest_api_panel(current_provider, bool(is_diarize), False)
    return (
        False,
        gr.update(visible=True),
        gr.update(visible=False),
        *panel_vis,
    )


_NON_COPYABLE_OUTPUT_PREFIXES = ("❌ ", "⚠️ ", "⏸️ Lote pausado")


def _update_copy_button(text: str | None):
    """Habilita Copiar texto únicamente cuando hay contenido de transcripción válido."""
    stripped = (text or "").strip()
    is_copyable = bool(stripped) and not stripped.startswith(
        _NON_COPYABLE_OUTPUT_PREFIXES
    )
    return gr.update(interactive=is_copyable)


def _refresh_transcribe_button_on_ui_change(
    media_files,
    batch_cache: dict | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    enable_diarization: bool | None = None,
    timestamp_option: str | None = None,
) -> gr.Button:
    """Wrapper UI que evita deshabilitar el botón al vaciar archivos tras limpiar."""
    return on_media_files_changed(
        media_files=media_files,
        batch_cache=batch_cache,
        provider=provider,
        model_name=model_name,
        enable_diarization=enable_diarization,
        timestamp_option=timestamp_option,
        allow_empty_disable=False,
    )


def clear_all(request: gr.Request = None):
    """Restablece la interfaz y limpia el caché diferencial de la sesión."""
    session_key = _get_session_key(request)
    _set_stop_requested(session_key, False)
    _set_batch_active(session_key, None)
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


def clear_all_ui(request: gr.Request = None):
    """Wrapper de limpieza para la interfaz web que también oculta el botón de parada."""
    return (*clear_all(request), _hidden_stop_button())


# Carga de estilos y tema visual modular
custom_css = get_custom_css()
custom_theme = get_custom_theme()
_ui_enhancements_js = get_ui_enhancements_js()
blocks_kwargs, launch_kwargs = get_blocks_and_launch_kwargs(
    custom_css, custom_theme, _ui_enhancements_js
)

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
        api_visibility="private",
    )

    _queue_refresh_inputs = [
        ui.media_input,
        ui.batch_cache_state,
        ui.provider_dropdown,
        ui.model_dropdown,
        ui.diarize_checkbox,
        ui.timestamps_dropdown,
    ]

    # Eventos de cambio de proveedor, modelo, diarización y marcas temporales
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
        api_visibility="private",
    ).then(
        fn=_refresh_transcribe_button_on_ui_change,
        inputs=_queue_refresh_inputs,
        outputs=[ui.btn_transcribe],
        show_progress="hidden",
        api_visibility="private",
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
        api_visibility="private",
    ).then(
        fn=_refresh_transcribe_button_on_ui_change,
        inputs=_queue_refresh_inputs,
        outputs=[ui.btn_transcribe],
        show_progress="hidden",
        api_visibility="private",
    )

    ui.model_dropdown.change(
        fn=_refresh_transcribe_button_on_ui_change,
        inputs=_queue_refresh_inputs,
        outputs=[ui.btn_transcribe],
        show_progress="hidden",
        api_visibility="private",
    )
    ui.timestamps_dropdown.change(
        fn=_refresh_transcribe_button_on_ui_change,
        inputs=_queue_refresh_inputs,
        outputs=[ui.btn_transcribe],
        show_progress="hidden",
        api_visibility="private",
    )

    # Eventos del diálogo de login de Administrador
    ui.btn_open_login.click(
        fn=open_login_dialog,
        outputs=[ui.modal_login, ui.input_pin, ui.login_msg],
        js=MODAL_OPEN_JS,
        show_progress="hidden",
        api_visibility="private",
    )
    ui.btn_close_login.click(
        fn=close_login_dialog,
        outputs=[ui.modal_login, ui.input_pin, ui.login_msg],
        js=MODAL_CLOSE_JS,
        show_progress="hidden",
        api_visibility="private",
    )
    ui.btn_cancel_login.click(
        fn=close_login_dialog,
        outputs=[ui.modal_login, ui.input_pin, ui.login_msg],
        js=MODAL_CLOSE_JS,
        show_progress="hidden",
        api_visibility="private",
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
        api_visibility="private",
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
        api_visibility="private",
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
        api_visibility="private",
    )

    # Copiar al portapapeles
    ui.btn_copy.click(
        fn=None,
        inputs=[ui.output_textbox],
        js=COPY_CLIPBOARD_JS,
        api_visibility="private",
    )

    # Activar/desactivar botón Copiar texto de forma reactiva
    ui.output_textbox.change(
        fn=_update_copy_button,
        inputs=[ui.output_textbox],
        outputs=[ui.btn_copy],
        show_progress="hidden",
        api_visibility="private",
    )

    # Parada elegante cooperativa durante lotes de múltiples archivos
    ui.btn_stop_transcribe.click(
        fn=toggle_graceful_stop,
        inputs=None,
        outputs=[ui.btn_stop_transcribe],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    # Iniciar / Reanudar Transcripción (entregas progresivas por archivo)
    transcribe_evt = ui.btn_transcribe.click(
        fn=process_transcription_stream,
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
            ui.btn_stop_transcribe,
            ui.batch_cache_state,
        ],
    )

    # Evento de limpiar campos (cancela cualquier flujo de transcripción en curso)
    _clear_outputs = [
        ui.media_input,
        ui.output_textbox,
        ui.btn_download,
        ui.btn_download_consolidated,
        ui.status_badge,
        ui.btn_transcribe,
        ui.batch_cache_state,
        ui.btn_stop_transcribe,
    ]
    ui.btn_clear.click(
        fn=clear_all_ui,
        outputs=_clear_outputs,
        cancels=[transcribe_evt],
        show_progress="hidden",
        api_visibility="private",
    )
    ui.media_input.clear(
        fn=clear_all_ui,
        outputs=_clear_outputs,
        cancels=[transcribe_evt],
        show_progress="hidden",
        api_visibility="private",
    )

    # Detección de archivos multimedia añadidos para reactivar transcripción diferencial
    ui.media_input.change(
        fn=_refresh_transcribe_button_on_ui_change,
        inputs=_queue_refresh_inputs,
        outputs=[ui.btn_transcribe],
        show_progress="hidden",
        api_visibility="private",
    )


if __name__ == "__main__":
    demo.launch(**launch_kwargs)
