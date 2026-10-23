"""Definición del tema Gradio (Aurora Boreal Nórdica) y argumentos de inicialización."""

import gradio as gr


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
    custom_css: str, custom_theme: gr.themes.Base
) -> tuple[dict, dict]:
    """Determina los argumentos compatibles según la versión mayor de Gradio instalada."""
    blocks_kwargs = {"title": "Transcriptor Suite"}
    launch_kwargs = {}

    try:
        gradio_major = int(gr.__version__.split(".")[0])
        if gradio_major >= 6:
            launch_kwargs["theme"] = custom_theme
            launch_kwargs["css"] = custom_css
        else:
            blocks_kwargs["theme"] = custom_theme
            blocks_kwargs["css"] = custom_css
    except Exception:
        blocks_kwargs["theme"] = custom_theme
        blocks_kwargs["css"] = custom_css

    return blocks_kwargs, launch_kwargs
