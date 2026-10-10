# Registro de Cambios (Changelog)

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
y este proyecto se adhiere a [Semantic Versioning](https://semver.org/lang/es/).

---

## [2.1.0] - 2026-10-09

### Agregado
* **Sistema de Diseño "Aurora Boreal Nórdica":**
  * Paleta de colores inspirada en la atmósfera ártica: Lienzo de Noche Ártica (`#050B1A`), Aurora Green (`#00FF87`), Ice Cyan (`#00D4FF`), Violet Sky (`#7B2FBE`) y Glacier Pink (`#FF6EC7`).
  * Paneles con acabado *Frosted Glass* (`backdrop-filter: blur(14px)` y bordes sutiles en azul glacial).
  * Tipografías integradas desde Google Fonts: **Montserrat** (títulos y controles UI) y **JetBrains Mono** (marcas de tiempo y transcripciones).
  * Eliminación de emojis en la interfaz gráfica en favor de iconografía vectorial SVG limpia.
* **Barra de Estado Inferior (`statusbar`):**
  * Barra inferior minimalista inspirada en [Preview Suite](https://github.com/Klopezxd/preview-suite).
  * Indicador de estado y motor activo con pulso verde aurora.
  * Insignia de autor: **Klever López**.
  * Insignia de autoría asistida: **Made by Antigravity**.
  * Píldora de versión del sistema: **v2.1.0**.
* **Entrada Directa de Credenciales en la Web:**
  * Nuevo campo `Token de Hugging Face (HF_TOKEN)` en el acordeón de configuración de diarización para permitir transcribir sin depender exclusivamente de los Secrets del Space.
* **Guía Maestra de Secretos y API Keys:**
  * Documentación destacada en la cabecera de `README.md` y `README.es.md` con instrucciones paso a paso para tokens de GitHub Actions, Hugging Face Spaces y proveedores de nube (Deepgram y AssemblyAI).

### Cambiado / Optimizado
* **Duración Extendida en Hugging Face ZeroGPU:**
  * Actualizado el decorador a `@spaces.GPU(duration=180)` para reservar hasta 3 minutos de GPU continua, eliminando la advertencia *"GPU task is exceeding its requested duration and might be aborted"* en grabaciones largas de ~2 horas.
* **Resiliencia en Diarización con Pyannote Audio:**
  * Soporte retrocompatible dual para `DiarizationPipeline(token=...)` y `DiarizationPipeline(use_auth_token=...)`.
  * Captura descriptiva de errores de autenticación con enlaces directos para aceptar las licencias de `pyannote/speaker-diarization-3.1` y `pyannote/segmentation-3.0`.

---

## [2.0.0] - 2026-10-09

### Agregado
* Arquitectura modular desacoplada en `src/` (`config.py`, `transcriber.py`, `utils.py`).
* Soporte para modelos Whisper `large-v3-turbo`, `large-v3`, `medium`, `small` y `base`.
* Alineación fonética precisa a nivel de palabra utilizando modelos acústicos Wav2Vec2.
* Diarización de interlocutores automática basada en Pyannote Audio.
* Cuantización en `INT8` y recolección forzada de VRAM entre fases para operar fluidamente en GPUs con 4GB de VRAM.
* Doble interfaz de ejecución: Web UI en Gradio y CLI con selector gráfico Tkinter.
* Workflow de GitHub Actions para sincronización automática con Hugging Face Spaces.
