#!/usr/bin/env python3
"""
Interfaz de Línea de Comandos (CLI) para Transcriptor Suite.
Permite procesamiento por lotes o interactivo con selección gráfica de respaldo.
"""

import argparse
import sys

from src.config import (
    ComputeType,
    ProviderType,
    TimestampMode,
    TranscriptionConfig,
)
from src.transcriber import create_transcription_pipeline
from src.utils import (
    AudioProcessingError,
    TranscriptionError,
    configure_utf8_console,
    setup_logger,
    validate_media_path,
)

configure_utf8_console()
logger = setup_logger("whisperx_cli")


def select_file_gui() -> str | None:
    """Abre un diálogo de Tkinter para seleccionar el archivo multimedia."""
    root = None
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_file = filedialog.askopenfilename(
            title="Selecciona el archivo de audio o video",
            filetypes=[
                ("Archivos Multimedia", "*.mp4 *.mkv *.avi *.mov *.mp3 *.wav *.m4a *.flac"),
                ("Todos los archivos", "*.*"),
            ],
        )
        return selected_file if selected_file else None
    except Exception as e:
        logger.warning("No se pudo abrir el selector gráfico: %s", e)
        return None
    finally:
        if root is not None:
            root.destroy()


def parse_arguments() -> argparse.Namespace:
    """Configura y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Transcriptor Suite - Pipeline de transcripción, alineación fonética y diarización.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Ruta al archivo multimedia. Si se omite, se abrirá un diálogo de selección.",
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        default=ProviderType.WHISPERX.value,
        choices=[p.value for p in ProviderType],
        help="Proveedor de inferencia: WhisperX (local/ZeroGPU), Deepgram o AssemblyAI.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Modelo a utilizar (ej. 'large-v3-turbo' para WhisperX, 'nova-2' para Deepgram, 'best' para AssemblyAI).",
    )
    parser.add_argument(
        "-d",
        "--diarize",
        action="store_true",
        help="Activa la identificación y segmentación de personas/hablantes.",
    )
    parser.add_argument(
        "--timestamps",
        type=str,
        default=TimestampMode.SIMPLE.value,
        choices=[t.value for t in TimestampMode],
        help="Modo de marcas de tiempo: 'simple' [HH:MM:SS], 'range' [Inicio - Fin], o 'none'.",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        default=None,
        help="Token de Hugging Face para Pyannote (o definir variable HF_TOKEN).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Clave de API para proveedores en la nube (Deepgram / AssemblyAI).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Dispositivo de ejecución local (autodetectado si no se especifica).",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default=None,
        choices=[c.value for c in ComputeType],
        help="Tipo de cuantización ('int8' recomendado para 4GB VRAM).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Carpeta donde se guardará la transcripción.",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default=None,
        help="Código de idioma (ej: 'es', 'en'). Autodetectado si no se provee.",
    )

    return parser.parse_args()


def main() -> None:
    """Función principal del CLI."""
    args = parse_arguments()

    # 1. Determinar archivo de entrada
    input_file = args.input
    if not input_file:
        print("\n[*] No se indicó un archivo por línea de comandos. Abriendo selector gráfico...")
        input_file = select_file_gui()
        if not input_file:
            print("[!] Operación cancelada: No se seleccionó ningún archivo.")
            sys.exit(0)

    try:
        media_path = validate_media_path(input_file)
    except AudioProcessingError as err:
        print(f"[X] Error: {err}")
        sys.exit(1)

    # 2. Construir configuración inmutable
    try:
        config = TranscriptionConfig.create_default(
            provider=args.provider,
            model_name=args.model,
            enable_diarization=args.diarize,
            timestamp_mode=TimestampMode(args.timestamps),
            hf_token=args.token,
            api_key=args.api_key,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
        )
    except ValueError as err:
        logger.error("Configuración inválida: %s", err)
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  TRANSCRIPTOR SUITE (CLI)")
    print("=" * 55)
    print(f" * Archivo:       {media_path.name}")
    print(f" * Proveedor:     {config.provider}")
    print(f" * Modelo:        {config.model_name}")
    if config.provider == ProviderType.WHISPERX.value:
        print(f" * Dispositivo:   {config.device} ({config.compute_type})")
    print(f" * Diarización:   {'Activada' if config.enable_diarization else 'Desactivada'}")
    print(f" * Timestamps:    {config.timestamp_mode.value}")
    print("=" * 55 + "\n")

    try:
        pipeline = create_transcription_pipeline(config)
        result = pipeline.process(
            media_path=media_path,
            output_dir=args.output_dir,
        )
        print("\n" + "=" * 55)
        print("  Transcripción completada con éxito.")
        print(f"  Guardado en: {result.output_path}")
        print("=" * 55 + "\n")
    except TranscriptionError as err:
        logger.error("Fallo en la transcripción: %s", err)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[!] Proceso interrumpido por el usuario.")
        sys.exit(130)


if __name__ == "__main__":
    main()
