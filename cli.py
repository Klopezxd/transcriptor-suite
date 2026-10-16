#!/usr/bin/env python3
"""
Interfaz de Línea de Comandos (CLI) para Transcriptor Suite.
Permite procesamiento por lotes o interactivo con selección gráfica de respaldo.
"""

import argparse
import sys
from pathlib import Path

from src.config import (
    ComputeType,
    ProviderType,
    TimestampMode,
    TranscriptionConfig,
    normalize_provider_name,
)
from src.transcriber import create_transcription_pipeline
from src.utils import (
    AudioProcessingError,
    TranscriptionError,
    configure_utf8_console,
    save_consolidated_batch_file,
    setup_logger,
    validate_media_path,
)

configure_utf8_console()
logger = setup_logger("whisperx_cli")


def select_file_gui() -> list[str]:
    """Abre un diálogo de Tkinter para seleccionar uno o varios archivos multimedia."""
    root = None
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_files = filedialog.askopenfilenames(
            title="Selecciona uno o más archivos de audio o video",
            filetypes=[
                ("Archivos Multimedia", "*.mp4 *.mkv *.avi *.mov *.mp3 *.wav *.m4a *.flac"),
                ("Todos los archivos", "*.*"),
            ],
        )
        return list(selected_files) if selected_files else []
    except Exception as e:
        logger.warning("No se pudo abrir el selector gráfico: %s", e)
        return []
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
        nargs="*",
        type=str,
        default=None,
        help="Ruta(s) a los archivos multimedia. Si se omite, se abrirá un diálogo de selección.",
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=normalize_provider_name,
        default=ProviderType.WHISPERX.value,
        choices=[p.value for p in ProviderType],
        help="Proveedor de inferencia: 'whisperx', 'deepgram' o 'assemblyai'.",
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
    """Función principal del CLI con soporte para procesamiento individual o por lotes."""
    args = parse_arguments()

    # 1. Determinar archivos de entrada
    raw_inputs = args.input
    if not raw_inputs:
        print("\n[*] No se indicó ningún archivo por línea de comandos. Abriendo selector gráfico...")
        raw_inputs = select_file_gui()
        if not raw_inputs:
            print("[!] Operación cancelada: No se seleccionó ningún archivo.")
            sys.exit(0)

    # Validar archivos multimedia
    valid_paths = []
    for item in raw_inputs:
        try:
            valid_paths.append(validate_media_path(item))
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

    total_files = len(valid_paths)
    print("\n" + "=" * 55)
    print("  TRANSCRIPTOR SUITE (CLI)")
    print("=" * 55)
    if total_files == 1:
        print(f" * Archivo:       {valid_paths[0].name}")
    else:
        print(f" * Archivos:      {total_files} multimedia en cola")
    print(f" * Proveedor:     {config.provider}")
    print(f" * Modelo:        {config.model_name}")
    if config.provider == ProviderType.WHISPERX.value:
        print(f" * Dispositivo:   {config.device} ({config.compute_type})")
    print(f" * Diarización:   {'Activada' if config.enable_diarization else 'Desactivada'}")
    print(f" * Timestamps:    {config.timestamp_mode.value}")
    print("=" * 55 + "\n")

    try:
        pipeline = create_transcription_pipeline(config)
        successful_results = []
        failed_count = 0

        for idx, media_path in enumerate(valid_paths, start=1):
            if total_files > 1:
                print(f"[{idx}/{total_files}] Procesando: {media_path.name}...")
            try:
                result = pipeline.process(
                    media_path=media_path,
                    output_dir=args.output_dir,
                )
                successful_results.append((result, media_path))
                if total_files == 1:
                    print("\n" + "=" * 55)
                    print("  Transcripción completada con éxito.")
                    print(f"  Guardado en: {result.output_path}")
                    print("=" * 55 + "\n")
                else:
                    out_name = result.output_path.name if result.output_path else "OK"
                    print(f"  ✓ Completado: {out_name}\n")
            except TranscriptionError as err:
                logger.error("Fallo en la transcripción de %s: %s", media_path.name, err)
                failed_count += 1
                if total_files == 1:
                    sys.exit(1)

        if total_files > 1:
            if successful_results:
                first_res, _ = successful_results[0]
                batch_dir = (
                    first_res.output_path.parent
                    if first_res.output_path
                    else Path(args.output_dir or "transcripciones")
                )
                combined_chunks = []
                for res, path_item in successful_results:
                    separator = f"{'=' * 60}\nARCHIVO: {path_item.name}\n{'=' * 60}\n\n"
                    combined_chunks.append(f"{separator}{res.formatted_text.strip()}\n\n")

                combined_text = "".join(combined_chunks).strip() + "\n"
                consolidated_path = save_consolidated_batch_file(
                    content=combined_text,
                    output_dir=batch_dir,
                )
                print(f"  Archivo consolidado: {consolidated_path.name}")

            print("=" * 55)
            print(
                f"  Resumen: {len(successful_results)} exitosos, {failed_count} fallidos de {total_files} archivos."
            )
            print("=" * 55 + "\n")
            if failed_count > 0 and not successful_results:
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n[!] Proceso interrumpido por el usuario.")
        sys.exit(130)


if __name__ == "__main__":
    main()
