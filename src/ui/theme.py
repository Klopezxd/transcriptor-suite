"""Definición del tema Gradio (Aurora Boreal Nórdica) y argumentos de inicialización."""

import os

import gradio as gr


def _resolve_max_file_size() -> str:
    """Resuelve el tamaño máximo de archivo permitido desde MAX_UPLOAD_SIZE_MB (por defecto 2gb)."""
    raw_mb = os.getenv("MAX_UPLOAD_SIZE_MB", "").strip()
    if raw_mb:
        try:
            mb_value = int(raw_mb)
            if mb_value > 0:
                return f"{mb_value}mb"
        except ValueError:
            pass
    return "2gb"


def get_custom_theme() -> gr.themes.Base:
    """Construye y retorna el tema visual base Aurora Boreal Nórdica."""
    return gr.themes.Base(
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


def get_blocks_and_launch_kwargs(
    custom_css: str, custom_theme: gr.themes.Base, custom_js: str = ""
) -> tuple[dict, dict]:
    """Determina los argumentos compatibles según la versión mayor de Gradio instalada."""
    blocks_kwargs = {"title": "Transcriptor Suite"}
    launch_kwargs: dict = {
        "max_file_size": _resolve_max_file_size(),
        "blocked_paths": [".env", ".git", "src", "app.py", "tests"],
    }

    try:
        gradio_major = int(gr.__version__.split(".")[0])
        if gradio_major >= 6:
            launch_kwargs["theme"] = custom_theme
            launch_kwargs["css"] = custom_css
            if custom_js:
                launch_kwargs["js"] = custom_js
        else:
            blocks_kwargs["theme"] = custom_theme
            blocks_kwargs["css"] = custom_css
            if custom_js:
                blocks_kwargs["js"] = custom_js
    except Exception:
        blocks_kwargs["theme"] = custom_theme
        blocks_kwargs["css"] = custom_css
        if custom_js:
            blocks_kwargs["js"] = custom_js

    return blocks_kwargs, launch_kwargs
