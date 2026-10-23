"""Módulo de Interfaz de Usuario (UI) para Transcriptor Suite."""

from src.ui.assets import (
    COPY_CLIPBOARD_JS,
    MODAL_CLOSE_JS,
    MODAL_OPEN_JS,
    get_custom_css,
    get_ui_enhancements_js,
)
from src.ui.components import UIComponents, render_transcriptor_ui
from src.ui.helpers import (
    EMPTY_DOWNLOAD_LABEL,
    PROVIDER_INFO_HTML,
    _build_status_badges,
    _disabled_consolidated_button,
    _disabled_download_button,
    _disabled_transcribe_button,
    _enabled_transcribe_button,
    _get_file_fingerprint,
    on_media_files_changed,
)
from src.ui.theme import get_blocks_and_launch_kwargs, get_custom_theme

__all__ = [
    "get_custom_css",
    "get_ui_enhancements_js",
    "MODAL_OPEN_JS",
    "MODAL_CLOSE_JS",
    "COPY_CLIPBOARD_JS",
    "get_custom_theme",
    "get_blocks_and_launch_kwargs",
    "EMPTY_DOWNLOAD_LABEL",
    "PROVIDER_INFO_HTML",
    "_build_status_badges",
    "_disabled_download_button",
    "_disabled_consolidated_button",
    "_enabled_transcribe_button",
    "_disabled_transcribe_button",
    "_get_file_fingerprint",
    "on_media_files_changed",
    "UIComponents",
    "render_transcriptor_ui",
]
