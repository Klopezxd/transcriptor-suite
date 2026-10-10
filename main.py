#!/usr/bin/env python3
"""
Punto de entrada principal para WhisperX Transcriptor CLI.
Provee compatibilidad directa con `python main.py` delegando a `cli.main()`.
"""

from cli import main

if __name__ == "__main__":
    main()
