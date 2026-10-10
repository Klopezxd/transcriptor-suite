#!/usr/bin/env python3
"""
Aplicación Web Gradio para WhisperX Transcriptor.
Diseñada para despliegue en Hugging Face Spaces (ZeroGPU) y ejecución local.
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
)

logger = setup_logger("whisperx_app")

# Compatibilidad con ZeroGPU en Hugging Face Spaces (duración extendida para audios largos)
try:
    import spaces

    GPU_DECORATOR = spaces.GPU(duration=180)
except (ImportError, TypeError, AttributeError):

    def GPU_DECORATOR(func):
        return func


@GPU_DECORATOR
def execute_whisperx(pipeline, media_path: str, progress_callback):
    """Ejecuta el pipeline local de WhisperX bajo asignación de GPU."""
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
    progress=gr.Progress(),
):
    """Procesa el archivo multimedia según el proveedor seleccionado y retorna texto y descarga."""
    if not media_file:
        return (
            "⚠️ Por favor selecciona o arrastra un archivo de audio o video.",
            gr.DownloadButton(label="Descargar Documento (.txt)", value=None, interactive=False),
        )

    media_path = media_file.name if hasattr(media_file, "name") else str(media_file)

    # Mapeo de formato de marcas de tiempo
    mode_map = {
        "Simple ([HH:MM:SS])": TimestampMode.SIMPLE,
        "Rango ([Inicio - Fin])": TimestampMode.RANGE,
        "Texto continuo (Sin marcas)": TimestampMode.NONE,
    }
    selected_mode = mode_map.get(timestamp_option, TimestampMode.SIMPLE)

    # Validación y resolución de credenciales según el motor elegido
    resolved_hf_token = None
    resolved_api_key = None

    if provider == ProviderType.WHISPERX.value:
        resolved_hf_token = resolve_hf_token(hf_token_input)
        if enable_diarization and not resolved_hf_token:
            return (
                "⚠️ La diarización con WhisperX requiere un token de Hugging Face.\n\n"
                "Por favor ingrésalo en el campo 'Token de Hugging Face' abajo o configúralo como 'HF_TOKEN' en los Secrets del Space.\n\n"
                "Requisitos en Hugging Face:\n"
                "1. Aceptar condiciones en: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "2. Aceptar condiciones en: https://huggingface.co/pyannote/segmentation-3.0\n"
                "3. Crear token con permisos 'Read' en: https://huggingface.co/settings/tokens",
                gr.DownloadButton(
                    label="Descargar Documento (.txt)", value=None, interactive=False
                ),
            )

    elif provider == ProviderType.DEEPGRAM.value:
        resolved_api_key = resolve_deepgram_api_key(deepgram_key_input)
        if not resolved_api_key:
            return (
                "⚠️ Se requiere una clave de API de Deepgram para utilizar este motor.\n\n"
                "• Si tienes tu propia clave, ingrésala en el acordeón 'Credenciales de Deepgram Cloud API'.\n"
                "• Si aún no dispones de una, puedes crear una cuenta gratuita en: https://console.deepgram.com/signup\n"
                "• Para transcribir de forma 100% gratuita sin necesidad de claves externas, selecciona 'WhisperX (ZeroGPU / Local)'.",
                gr.DownloadButton(
                    label="Descargar Documento (.txt)", value=None, interactive=False
                ),
            )

    elif provider == ProviderType.ASSEMBLYAI.value:
        resolved_api_key = resolve_assemblyai_api_key(assemblyai_key_input)
        if not resolved_api_key:
            return (
                "⚠️ Se requiere una clave de API de AssemblyAI para utilizar este motor.\n\n"
                "• Si tienes tu propia clave, ingrésala en el acordeón 'Credenciales de AssemblyAI Cloud API'.\n"
                "• Si aún no dispones de una, puedes crear una cuenta gratuita en: https://www.assemblyai.com/dashboard/signup\n"
                "• Para transcribir de forma 100% gratuita sin necesidad de claves externas, selecciona 'WhisperX (ZeroGPU / Local)'.",
                gr.DownloadButton(
                    label="Descargar Documento (.txt)", value=None, interactive=False
                ),
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

        # Si es WhisperX, ejecutar en el runner de ZeroGPU; si es API Cloud, ejecutar directamente
        if provider == ProviderType.WHISPERX.value:
            result = execute_whisperx(pipeline, media_path, on_progress)
        else:
            result = pipeline.process(
                media_path=media_path,
                progress_callback=on_progress,
            )

        return (
            result.formatted_text,
            gr.DownloadButton(
                label=f"Descargar {result.output_path.name}",
                value=str(result.output_path),
                interactive=True,
            ),
        )

    except TranscriptionError as e:
        clear_gpu_vram()
        logger.error("Error en procesamiento web: %s", e)
        return (
            f"❌ Error de transcripción: {e}",
            gr.DownloadButton(label="Descargar Documento (.txt)", value=None, interactive=False),
        )
    except Exception as e:
        clear_gpu_vram()
        logger.exception("Error inesperado en app web")
        return (
            f"❌ Ocurrió un error inesperado durante el procesamiento: {e}",
            gr.DownloadButton(label="Descargar Documento (.txt)", value=None, interactive=False),
        )


# --- Sistema de Diseño: Aurora Boreal Nórdica ---
# Inspirado en minimalismo escandinavo, noche ártica (#050B1A), Aurora Green (#00FF87) y Cyan Glacial (#00D4FF)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Montserrat:wght@400;500;600;700;800&display=swap');

:root {
    --aurora-green: #00FF87;
    --arctic-blue: #0A1628;
    --deep-midnight: #050B1A;
    --frost-white: #E8F4F8;
    --snow: #FAFBFC;
    --ice-cyan: #00D4FF;
    --violet-sky: #7B2FBE;
    --glacier-pink: #FF6EC7;
    --muted-slate: #8CA0BA;
    --glass-border: rgba(0, 212, 255, 0.16);
    --glass-surface: rgba(10, 22, 40, 0.72);
}

/* Atmósfera de Noche Ártica con Ondas de Aurora Boreal */
body, gradio-app {
    background-color: var(--deep-midnight) !important;
    background-image: 
        radial-gradient(circle at 18% 10%, rgba(0, 255, 135, 0.08) 0%, transparent 45%),
        radial-gradient(circle at 82% 14%, rgba(123, 47, 190, 0.12) 0%, transparent 50%),
        radial-gradient(circle at 50% 60%, rgba(0, 212, 255, 0.06) 0%, transparent 55%),
        radial-gradient(circle at 85% 85%, rgba(255, 110, 199, 0.04) 0%, transparent 40%) !important;
    background-attachment: fixed !important;
    color: var(--frost-white) !important;
    font-family: 'Montserrat', system-ui, -apple-system, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

.gradio-container {
    max-width: 1360px !important;
    width: 95% !important;
    margin: 0 auto !important;
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
}

/* Tarjetas y Paneles Frosted Glass */
.gradio-container .block,
.gradio-container .panel,
.gradio-container .form {
    background: var(--glass-surface) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35) !important;
}

/* Cabecera Estilo Nórdico */
#app-header {
    text-align: center;
    padding: 1.5rem 1rem 1.25rem 1rem;
    position: relative;
    margin-bottom: 1rem;
}

.aurora-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.9rem;
    border-radius: 9999px;
    background: rgba(0, 255, 135, 0.08);
    border: 1px solid rgba(0, 255, 135, 0.28);
    color: var(--aurora-green);
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}

.aurora-badge-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: var(--aurora-green);
    box-shadow: 0 0 8px var(--aurora-green);
    animation: aurora-pulse 2.2s infinite ease-in-out;
}

@keyframes aurora-pulse {
    0%, 100% { opacity: 0.5; transform: scale(0.9); }
    50% { opacity: 1; transform: scale(1.2); }
}

.aurora-title {
    font-size: 2.35rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(135deg, #FFFFFF 25%, var(--aurora-green) 75%, var(--ice-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
}

.aurora-subtitle {
    color: var(--muted-slate);
    font-size: 1.02rem;
    max-width: 680px;
    margin: 0 auto;
    line-height: 1.55;
    font-weight: 400;
}

/* Botón Principal (Aurora Green #00FF87) */
.btn-primary, button.btn-primary, button.primary {
    background: var(--aurora-green) !important;
    color: var(--deep-midnight) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    letter-spacing: 0.01em !important;
    transition: all 0.22s ease-in-out !important;
    box-shadow: 0 2px 14px rgba(0, 255, 135, 0.25) !important;
    padding: 0.75rem 1.5rem !important;
}

.btn-primary:hover, button.btn-primary:hover, button.primary:hover {
    background: #00E67A !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 22px rgba(0, 255, 135, 0.45) !important;
}

.btn-primary:active, button.btn-primary:active, button.primary:active {
    transform: translateY(1px) !important;
}

/* Botón Secundario (Limpiar) */
.btn-secondary, button.btn-secondary, button.secondary {
    background: rgba(15, 29, 52, 0.75) !important;
    color: var(--frost-white) !important;
    border: 1px solid rgba(0, 212, 255, 0.25) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.btn-secondary:hover, button.btn-secondary:hover, button.secondary:hover {
    background: rgba(0, 212, 255, 0.12) !important;
    border-color: rgba(0, 212, 255, 0.5) !important;
    color: #FFFFFF !important;
}

/* Botón de Descarga */
.btn-download {
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: 1px solid rgba(0, 212, 255, 0.35) !important;
    transition: all 0.2s ease !important;
}

/* Área de Salida: Tipografía Monospace JetBrains */
.output-transcript textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    background: #060e1d !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    color: var(--frost-white) !important;
}

/* Acordeones Frosted Glass */
.gradio-container .accordion {
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    background: rgba(10, 22, 40, 0.5) !important;
    margin-bottom: 0.75rem !important;
}

/* Inputs, Selectores y Foco */
.gradio-container input[type="text"],
.gradio-container input[type="password"],
.gradio-container select,
.gradio-container textarea {
    background-color: #07101e !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    border-radius: 8px !important;
    color: var(--frost-white) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.gradio-container input:focus,
.gradio-container select:focus,
.gradio-container textarea:focus {
    border-color: var(--aurora-green) !important;
    box-shadow: 0 0 0 2px rgba(0, 255, 135, 0.3) !important;
    outline: none !important;
}

/* Zona de Carga de Archivos */
.gradio-container .file-upload,
.gradio-container [data-testid="file-upload"] {
    border: 1.5px dashed rgba(0, 212, 255, 0.3) !important;
    border-radius: 8px !important;
    background: rgba(8, 18, 34, 0.5) !important;
    transition: all 0.25s ease !important;
}

.gradio-container .file-upload:hover,
.gradio-container [data-testid="file-upload"]:hover {
    border-color: var(--aurora-green) !important;
    background: rgba(0, 255, 135, 0.04) !important;
}

/* Pie de Página Nórdico */
#app-footer {
    text-align: center;
    margin-top: 2.5rem;
    margin-bottom: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(0, 212, 255, 0.12);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.85rem;
    color: var(--muted-slate);
}

.footer-tech {
    letter-spacing: 0.02em;
}

.footer-tech strong {
    color: var(--frost-white);
    font-weight: 600;
}

.footer-links a {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--ice-cyan);
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s ease;
}

.footer-links a:hover {
    color: var(--aurora-green);
}

/* --- Barra de Estado Inferior Estilo Nórdico (Inspirada en Preview Suite) --- */
.statusbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 16px;
    background: rgba(5, 11, 26, 0.88);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 212, 255, 0.18);
    border-radius: 8px;
    font-size: 11.5px;
    color: var(--muted-slate);
    margin-top: 2rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.statusbar-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.statusbar-stats {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #cbd5e1;
    display: inline-flex;
    align-items: center;
    gap: 7px;
}

.statusbar-stats-dot {
    width: 6.5px;
    height: 6.5px;
    border-radius: 50%;
    background: var(--aurora-green);
    box-shadow: 0 0 8px var(--aurora-green);
    display: inline-block;
}

.statusbar-right {
    display: flex;
    gap: 8px;
    align-items: center;
}

.author-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(0, 212, 255, 0.1);
    padding: 3px 10px;
    border-radius: 6px;
    color: var(--ice-cyan);
    font-weight: 500;
    font-size: 11.5px;
    border: 1px solid rgba(0, 212, 255, 0.25);
    white-space: nowrap;
}

.author-badge svg {
    color: var(--ice-cyan);
    width: 13px;
    height: 13px;
}

.antigravity-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(123, 47, 190, 0.15);
    padding: 3px 9px;
    border-radius: 6px;
    color: #d8b4fe;
    font-size: 11.5px;
    font-weight: 500;
    border: 1px solid rgba(123, 47, 190, 0.35);
    transition: all 0.2s ease;
    white-space: nowrap;
    cursor: default;
}

.antigravity-badge:hover {
    background: rgba(123, 47, 190, 0.28);
    border-color: #c084fc;
    box-shadow: 0 0 10px rgba(192, 132, 252, 0.35);
    transform: translateY(-1px);
}

.antigravity-badge svg {
    color: var(--glacier-pink);
    width: 12px;
    height: 12px;
}

.version-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 6px;
    background: rgba(0, 255, 135, 0.12);
    color: var(--aurora-green);
    border: 1px solid rgba(0, 255, 135, 0.3);
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    white-space: nowrap;
    cursor: default;
}

.version-badge:hover {
    background: rgba(0, 255, 135, 0.24);
    border-color: var(--aurora-green);
    box-shadow: 0 0 10px rgba(0, 255, 135, 0.35);
    transform: translateY(-1px);
}

/* Adaptabilidad Móvil */
@media (max-width: 768px) {
    .gradio-container {
        width: 100% !important;
        padding: 0.75rem !important;
    }
    .aurora-title {
        font-size: 1.65rem;
    }
    .aurora-subtitle {
        font-size: 0.9rem;
    }
    .statusbar {
        flex-direction: column;
        gap: 8px;
        padding: 10px 12px;
    }
    .statusbar-left {
        justify-content: center;
    }
    .statusbar-right {
        justify-content: center;
        flex-wrap: wrap;
        gap: 6px;
    }
}
"""

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Montserrat"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#050B1A",
    body_background_fill_dark="#050B1A",
    body_text_color="#E8F4F8",
    body_text_color_dark="#E8F4F8",
    body_text_color_subdued="#8CA0BA",
    body_text_color_subdued_dark="#8CA0BA",
    background_fill_primary="#0A1628",
    background_fill_primary_dark="#0A1628",
    background_fill_secondary="#081220",
    background_fill_secondary_dark="#081220",
    block_background_fill="rgba(10, 22, 40, 0.72)",
    block_background_fill_dark="rgba(10, 22, 40, 0.72)",
    block_border_color="rgba(0, 212, 255, 0.16)",
    block_border_color_dark="rgba(0, 212, 255, 0.16)",
    block_radius="8px",
    block_shadow="0 4px 20px rgba(0, 0, 0, 0.4)",
    block_title_text_color="#E8F4F8",
    block_title_text_color_dark="#E8F4F8",
    block_label_text_color="#8CA0BA",
    block_label_text_color_dark="#8CA0BA",
    border_color_primary="rgba(0, 212, 255, 0.16)",
    border_color_primary_dark="rgba(0, 212, 255, 0.16)",
    button_primary_background_fill="#00FF87",
    button_primary_background_fill_dark="#00FF87",
    button_primary_background_fill_hover="#00E67A",
    button_primary_background_fill_hover_dark="#00E67A",
    button_primary_text_color="#050B1A",
    button_primary_text_color_dark="#050B1A",
    button_secondary_background_fill="rgba(15, 29, 52, 0.75)",
    button_secondary_background_fill_dark="rgba(15, 29, 52, 0.75)",
    button_secondary_border_color="rgba(0, 212, 255, 0.25)",
    button_secondary_border_color_dark="rgba(0, 212, 255, 0.25)",
    button_secondary_text_color="#E8F4F8",
    button_secondary_text_color_dark="#E8F4F8",
    input_background_fill="#07101E",
    input_background_fill_dark="#07101E",
    input_border_color="rgba(0, 212, 255, 0.2)",
    input_border_color_dark="rgba(0, 212, 255, 0.2)",
    input_border_color_focus="#00FF87",
    input_border_color_focus_dark="#00FF87",
    checkbox_background_color_selected="#00FF87",
    checkbox_background_color_selected_dark="#00FF87",
    checkbox_border_color_selected="#00FF87",
    checkbox_border_color_selected_dark="#00FF87",
    checkbox_label_text_color="#E8F4F8",
    checkbox_label_text_color_dark="#E8F4F8",
    color_accent="#00D4FF",
    color_accent_soft="rgba(0, 212, 255, 0.15)",
)

# Compatibilidad entre Gradio < 6.0 y Gradio >= 6.0 (theme y css pasaron de Blocks a launch)
blocks_kwargs = {"title": "WhisperX Transcriptor · Aurora Boreal"}
launch_kwargs = {}

try:
    _gradio_major = int(gr.__version__.split(".")[0])
    if _gradio_major >= 6:
        launch_kwargs["theme"] = theme
        launch_kwargs["css"] = custom_css
    else:
        blocks_kwargs["theme"] = theme
        blocks_kwargs["css"] = custom_css
except Exception:
    blocks_kwargs["theme"] = theme
    blocks_kwargs["css"] = custom_css

with gr.Blocks(**blocks_kwargs) as demo:
    gr.HTML(
        """
        <div id="app-header">
            <div class="aurora-badge">
                <span class="aurora-badge-dot"></span>
                <span>Arquitectura Nórdica · Acelerada con ZeroGPU</span>
            </div>
            <h1 class="aurora-title">
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#00FF87" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; filter: drop-shadow(0 0 10px rgba(0,255,135,0.45));">
                    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" x2="12" y1="19" y2="22"/>
                </svg>
                <span>WhisperX Transcriptor</span>
            </h1>
            <p class="aurora-subtitle">
                Transcripción fonética de alta velocidad, alineación temporal de palabras y diarización de interlocutores
            </p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        # Columna Izquierda: Entrada y Controles
        with gr.Column(scale=1):
            provider_selector = gr.Radio(
                choices=[
                    ProviderType.WHISPERX.value,
                    ProviderType.DEEPGRAM.value,
                    ProviderType.ASSEMBLYAI.value,
                ],
                value=ProviderType.WHISPERX.value,
                label="Motor de Transcripción",
                info="Elige entre aceleración ZeroGPU (Hugging Face) o procesamiento ultrarrápido en la nube.",
            )

            input_media = gr.File(
                label="Archivo Multimedia",
                file_types=["audio", "video"],
                file_count="single",
            )

            diarization_checkbox = gr.Checkbox(
                value=False,
                label="Identificar quién habla (Diarización)",
                info="Separa y etiqueta las intervenciones de cada locutor automáticamente.",
            )

            # Acordeón Credenciales Hugging Face (para WhisperX)
            with gr.Accordion(
                "Credenciales de Hugging Face (Pyannote Diarización)", open=False, visible=True
            ) as hf_acc:
                hf_token_input = gr.Textbox(
                    label="Token de Hugging Face (HF_TOKEN)",
                    placeholder="hf_... (opcional si ya está configurado en Secrets)",
                    type="password",
                    info="Requerido si activas la diarización con WhisperX. Rol 'Read' con términos de Pyannote 3.1 y Segmentation 3.0 aceptados.",
                )

            # Acordeón Credenciales Deepgram
            with gr.Accordion(
                "Credenciales de Deepgram Cloud API", open=True, visible=False
            ) as deepgram_acc:
                deepgram_key_input = gr.Textbox(
                    label="Deepgram API Key",
                    placeholder="Pega tu clave personal de Deepgram aquí (o defínela como DEEPGRAM_API_KEY)",
                    type="password",
                    info="Permite usar Nova-2 / Nova-3 con diarización ultrarrápida. Obtén tu clave en console.deepgram.com.",
                )

            # Acordeón Credenciales AssemblyAI
            with gr.Accordion(
                "Credenciales de AssemblyAI Cloud API", open=True, visible=False
            ) as assemblyai_acc:
                assemblyai_key_input = gr.Textbox(
                    label="AssemblyAI API Key",
                    placeholder="Pega tu clave personal de AssemblyAI aquí (o defínela como ASSEMBLYAI_API_KEY)",
                    type="password",
                    info="Permite usar Universal-1 (máxima precisión) y Nano (ahorro). Obtén tu clave en assemblyai.com.",
                )

            with gr.Accordion("Opciones avanzadas", open=False):
                model_selector = gr.Dropdown(
                    choices=PROVIDER_MODEL_CHOICES[ProviderType.WHISPERX.value],
                    value=ModelName.LARGE_V3_TURBO.value,
                    label="Modelo de Voz",
                    info="Gratuito en ZeroGPU · large-v3-turbo es la opción recomendada (más rápida y misma precisión que large-v3).",
                )

                timestamp_selector = gr.Radio(
                    choices=[
                        "Simple ([HH:MM:SS])",
                        "Rango ([Inicio - Fin])",
                        "Texto continuo (Sin marcas)",
                    ],
                    value="Simple ([HH:MM:SS])",
                    label="Marcas de tiempo",
                )

            with gr.Row():
                btn_clear = gr.ClearButton(
                    value="Limpiar",
                    variant="secondary",
                    elem_classes=["btn-secondary"],
                )
                btn_transcribe = gr.Button(
                    "Iniciar Transcripción",
                    variant="primary",
                    size="lg",
                    elem_classes=["btn-primary"],
                )

        # Columna Derecha: Salida y Descarga
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Resultado de la Transcripción",
                placeholder="El texto transcrito aparecerá aquí...",
                lines=14,
                elem_classes=["output-transcript"],
            )
            download_file = gr.DownloadButton(
                label="Descargar Documento (.txt)",
                value=None,
                interactive=False,
                size="lg",
                variant="secondary",
                elem_classes=["btn-download"],
            )

    # Conexión de eventos reactivos
    def on_provider_change(selected_provider: str):
        choices = PROVIDER_MODEL_CHOICES.get(selected_provider, [])
        default_val = choices[0][1] if choices else ""

        is_wx = selected_provider == ProviderType.WHISPERX.value
        is_dg = selected_provider == ProviderType.DEEPGRAM.value
        is_aai = selected_provider == ProviderType.ASSEMBLYAI.value

        info_texts = {
            ProviderType.WHISPERX.value: (
                "Inferencia en GPU (Gratuita en ZeroGPU). "
                "large-v3-turbo es la opción más rápida (~90s/2h) y eficiente."
            ),
            ProviderType.DEEPGRAM.value: (
                "Inferencia ultrarrápida (~20s/2h). "
                "nova-2 es el más económico (~$0.26/h) · nova-3 para acústica con ruido o acentos (~$0.31/h)."
            ),
            ProviderType.ASSEMBLYAI.value: (
                "Inferencia acústica de alta fidelidad (~60s/2h). "
                "best para precisión editorial (~$0.39/h) · nano para máximo ahorro (~$0.17/h)."
            ),
        }

        return (
            gr.Dropdown(
                choices=choices, value=default_val, info=info_texts.get(selected_provider, "")
            ),
            gr.Accordion(visible=is_wx),
            gr.Accordion(visible=is_dg),
            gr.Accordion(visible=is_aai),
        )

    provider_selector.change(
        fn=on_provider_change,
        inputs=[provider_selector],
        outputs=[model_selector, hf_acc, deepgram_acc, assemblyai_acc],
    )

    btn_clear.add([input_media, output_text])
    btn_clear.click(
        fn=lambda: gr.DownloadButton(
            label="Descargar Documento (.txt)", value=None, interactive=False
        ),
        outputs=download_file,
    )

    btn_transcribe.click(
        fn=process_transcription,
        inputs=[
            input_media,
            provider_selector,
            model_selector,
            diarization_checkbox,
            timestamp_selector,
            hf_token_input,
            deepgram_key_input,
            assemblyai_key_input,
        ],
        outputs=[output_text, download_file],
    )

    gr.HTML(
        """
        <footer class="statusbar">
            <div class="statusbar-left">
                <span class="statusbar-stats">
                    <span class="statusbar-stats-dot"></span>
                    WhisperX ZeroGPU · Deepgram Nova-2 · AssemblyAI Universal-1
                </span>
            </div>
            <div class="statusbar-right">
                <a href="https://github.com/Klopezxd/whisperx-transcriptor" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">
                    <span class="author-badge">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="4 17 10 11 4 5"></polyline>
                            <line x1="12" y1="19" x2="20" y2="19"></line>
                        </svg>
                        Autor: Klever López
                    </span>
                </a>
                <span class="antigravity-badge" id="antigravity-badge" title="Made by Antigravity">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path>
                    </svg>
                    Made by Antigravity
                </span>
                <span class="version-badge" id="version-badge" title="v2.1.0">v2.1.0</span>
            </div>
        </footer>
        """
    )

if __name__ == "__main__":
    demo.launch(**launch_kwargs)
