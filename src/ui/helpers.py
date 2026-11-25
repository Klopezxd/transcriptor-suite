"""Funciones auxiliares, constructores de botones y generación de estados de UI."""

from pathlib import Path
from typing import Any

import gradio as gr

from src.config import TIMESTAMP_LABEL_MAP, ProviderType, TimestampMode, normalize_provider_name

EMPTY_DOWNLOAD_LABEL = "📥 Descargar Documento (.txt)"

PROVIDER_INFO_HTML: dict[str, str] = {
    ProviderType.WHISPERX.value: (
        '<div class="info-notice"><span>💡</span>'
        "<span><b>WhisperX:</b> Inferencia GPU local / ZeroGPU. Modelo <code>large-v3-turbo</code> ofrece máxima velocidad y eficiencia.</span></div>"
    ),
    ProviderType.DEEPGRAM.value: (
        '<div class="info-notice"><span>💡</span>'
        "<span><b>Deepgram:</b> Inferencia ultrarrápida (~20s por audio largo). Modelo <code>nova-3</code> óptimo para ruido y acentos (por defecto) y <code>nova-2</code> para máxima economía.</span></div>"
    ),
    ProviderType.ASSEMBLYAI.value: (
        '<div class="info-notice"><span>💡</span>'
        "<span><b>AssemblyAI:</b> Precisión acústica y comprensión contextual de alta fidelidad. Modelo <code>assemblyai-stt-3-7</code> por defecto con respaldo multilingüe <code>universal-2</code>.</span></div>"
    ),
}


def _build_status_badges(status_span: str, time_span: str = "") -> str:
    """Construye un contenedor flex para agrupar insignias de tiempo y estado."""
    return f'<div class="status-badges-group">{time_span}{status_span}</div>'


def _disabled_download_button() -> gr.DownloadButton:
    """Genera el botón de descarga individual en estado inactivo."""
    return gr.DownloadButton(
        label=EMPTY_DOWNLOAD_LABEL,
        value=None,
        interactive=False,
        visible=True,
        variant="secondary",
        elem_classes=["btn-secondary"],
    )


def _disabled_consolidated_button() -> gr.DownloadButton:
    """Genera el botón de descarga consolidada en estado inactivo y oculto."""
    return gr.DownloadButton(
        label="📄 Descargar Consolidado (.txt)",
        value=None,
        interactive=False,
        visible=False,
        variant="secondary",
        elem_classes=["btn-secondary"],
    )


def build_config_signature(
    provider: str,
    model_name: str,
    enable_diarization: bool,
    timestamp_option: str,
) -> str:
    """Genera una firma estable de configuración para evitar mezclar resultados de motores distintos."""
    try:
        canonical_provider = normalize_provider_name(provider)
    except ValueError:
        canonical_provider = str(provider or "").strip()

    ts_mode = TIMESTAMP_LABEL_MAP.get(timestamp_option)
    if ts_mode is not None:
        ts_value = ts_mode.value
    elif isinstance(timestamp_option, TimestampMode):
        ts_value = timestamp_option.value
    else:
        ts_value = str(timestamp_option or TimestampMode.SIMPLE.value)

    return f"{canonical_provider}|{str(model_name or '').strip()}|{int(bool(enable_diarization))}|{ts_value}"


def _enabled_transcribe_button(value: str = "⚡ Iniciar Transcripción") -> gr.Button:
    """Genera el botón de transcripción en estado activo."""
    return gr.Button(
        value=value,
        variant="primary",
        elem_id="btn_transcribe",
        interactive=True,
        visible=True,
        elem_classes=["btn-primary"],
    )


def _disabled_transcribe_button(value: str = "⚡ Iniciar Transcripción") -> gr.Button:
    """Genera el botón de transcripción en estado deshabilitado."""
    return gr.Button(
        value=value,
        variant="secondary",
        elem_id="btn_transcribe",
        interactive=False,
        visible=True,
        elem_classes=["btn-primary", "btn-transcribe-disabled"],
    )


def _hidden_transcribe_button() -> gr.Button:
    """Oculta temporalmente el botón principal mientras se muestra el botón de parada."""
    return gr.Button(
        value="⚡ Iniciar Transcripción",
        variant="primary",
        elem_id="btn_transcribe",
        interactive=False,
        visible=False,
        elem_classes=["btn-primary"],
    )


def _resumable_transcribe_button(remaining_count: int) -> gr.Button:
    """Genera el botón para reanudar un lote pausado indicando cuántos archivos faltan."""
    label = (
        "▶️ Reanudar (1 pendiente)"
        if remaining_count == 1
        else f"▶️ Reanudar ({remaining_count} pendientes)"
    )
    return _enabled_transcribe_button(label)


def _visible_stop_button(stopping: bool = False) -> gr.Button:
    """Muestra el botón de parada elegante durante un lote de múltiples archivos."""
    if stopping:
        return gr.Button(
            value="↩️ Cancelar pausa (seguir con la cola)",
            variant="secondary",
            elem_id="btn_stop_transcribe",
            interactive=True,
            visible=True,
            elem_classes=["btn-stop-transcribe", "is-stopping"],
        )
    return gr.Button(
        value="🛑 Detener tras este archivo",
        variant="stop",
        elem_id="btn_stop_transcribe",
        interactive=True,
        visible=True,
        elem_classes=["btn-stop-transcribe"],
    )


def _hidden_stop_button() -> gr.Button:
    """Oculta el botón de parada elegante cuando no hay un lote pausable en curso."""
    return gr.Button(
        value="🛑 Detener tras este archivo",
        variant="stop",
        elem_id="btn_stop_transcribe",
        interactive=False,
        visible=False,
        elem_classes=["btn-stop-transcribe"],
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


def on_media_files_changed(
    media_files,
    batch_cache: dict | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    enable_diarization: bool | None = None,
    timestamp_option: str | None = None,
    *,
    allow_empty_disable: bool = True,
) -> gr.Button:
    """
    Detecta si se agregaron archivos nuevos o pendientes en la cola y reactiva el botón
    de transcripción mostrando cuántos archivos están pendientes de transcribir.
    """
    cache = batch_cache if isinstance(batch_cache, dict) else {}
    cached_results = cache.get("cached_results") or {}

    if provider is not None and cache.get("config_signature"):
        current_sig = build_config_signature(
            provider=provider,
            model_name=model_name or "",
            enable_diarization=bool(enable_diarization),
            timestamp_option=timestamp_option or "",
        )
        if current_sig != cache.get("config_signature"):
            cached_results = {}

    file_list = (
        media_files
        if isinstance(media_files, list)
        else ([media_files] if media_files else [])
    )
    file_list = [f for f in file_list if f]

    if not file_list:
        if cached_results and allow_empty_disable and not cache.get("cleared"):
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
        if cached_results and cache.get("paused"):
            return _resumable_transcribe_button(new_files_count)
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
