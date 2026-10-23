# Registro de Cambios (Changelog)

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
y este proyecto se adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [2.4.1] - 2026-10-23

### Agregado
* **Codificación Ultra Ligera en Cliente (MP3 64 kbps a 16 kHz) con `lamejs`:**
  * Compresión acústica local que reduce hasta 4 veces más el tamaño frente a PCM WAV (ej. 10 minutos pasan de 19.2 MB a sólo 4.58 MB; 30 minutos a ~13.7 MB), acelerando la subida a Hugging Face Space hasta en un 97%.
  * Fallback automático a 16 kHz Mono WAV ante cualquier eventualidad.
* **Insignia Permanente y Feedback en Tiempo Real de Ahorro en la Interfaz:**
  * Visualización visible e inconfundible del ahorro exacto (`124.5 MB ➔ 4.6 MB (-96% de ancho de banda)` y megabytes ahorrados) tanto en el overlay en vivo durante la subida como en una insignia permanente bajo la zona de carga de archivos.
  * Notificación informativa clara en caso de que un video requiera procesamiento directo en servidor.

### Corregido
* **Prevención de Carga Innecesaria de Audio:**
  * Si el archivo de video original ya es más liviano que el audio generado (ej. videos ultra comprimidos), se conserva el video original para no ralentizar la transferencia.
* **Cierre Prematuro del Overlay de Carga:**
  * Corregido el hook de red para evitar que solicitudes internas `/api/` en segundo plano ocultaran el overlay antes de que la subida a Hugging Face Space finalizara.

---

## [2.4.0] - 2026-10-22

### Agregado
* **Optimización y Extracción de Audio en el Cliente (Navegador) con Web Audio API:**
  * Decodificación y remuestreo nativo en el navegador a 16 kHz Mono WAV PCM de 16 bits para archivos de video (`.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, `.m4v`, `.flv`, `.wmv`).
  * Reducción del peso de subida entre un 90% y 98% (ej. de 1.5 GB a ~30-50 MB) con 0% de pérdida de calidad acústica para los modelos WhisperX, Deepgram y AssemblyAI.
  * Compatibilidad uniforme e interceptación transparente tanto en el selector manual de archivos como en arrastrar y soltar (Drag & Drop).
  * Overlay visual dinámico con progreso en tiempo real (`⚡ Optimizando video (X/N)...`) y fallback automático al archivo original ante fallos de decodificación.

### Cambiado / Refactorizado
* **Arquitectura Modular de Presentación y Frontend (`assets/` y `src/ui/`):**
  * Desacoplamiento de más de 2,500 líneas de CSS y JavaScript que residían como cadenas multilínea en `app.py`.
  * Creación del directorio de activos estáticos: `assets/css/theme.css` (estilos puros Aurora Boreal Nórdica) y `assets/js/ui_enhancements.js`.
  * Creación del paquete `src/ui/`:
    * `components.py`: Componentes modulares de Gradio (Navbar, Modal de Login, Tarjetas de Entrada, Motor, Credenciales y Resultados).
    * `helpers.py`: Generación de estados de botones, insignias de estado y huellas digitales de archivos.
    * `theme.py`: Configuración de temas visuales y compatibilidad con versiones de Gradio.
    * `assets.py`: Gestor y lector eficiente con caché para archivos estáticos.
  * Reducción drástica de `app.py` de 3,860 líneas a ~790 líneas (reducción del 80%), manteniéndolo como un orquestador legible y mantenible.

### Corregido
* **Alineación y Centrado de la Zona de Carga (Dropzone):**
  * Desacoplamiento de dimensiones de `.icon-wrap` y configuración en columna centrada para evitar que la flecha y el texto se separen a los extremos en modo Admin o pantallas altas.
* **Jerarquía Visual de Formatos y Aviso de Optimización:**
  * Separación en renglones independientes para los formatos soportados y el aviso de optimización de video en cliente, mejorando la legibilidad.

---

## [2.3.0] - 2026-10-16

### Agregado
* **Procesamiento Diferencial Inteligente con Caché de Sesión (`gr.State`):**
  * Soporte para añadir archivos multimedia adicionales a la cola tras una transcripción completada sin re-procesar los archivos previos.
  * Huella digital $O(1)$ por archivo (`nombre_tamaño`) para verificar la existencia del resultado en caché y en disco, omitiendo llamadas a la API o al modelo local (ahorro del 100% de créditos de API y ciclos de GPU).
  * Reactivación dinámica del botón de transcripción vía `media_input.change`: muestra en tiempo real `⚡ Transcribir N archivo(s) nuevo(s)` únicamente cuando hay archivos nuevos pendientes.
  * Reensamblaje transparente de los resultados en el orden original de la cola, actualizando la caja de texto, regenerando el archivo consolidado `.txt` y el archivo comprimido `.zip` con todos los elementos de la sesión.
  * Limpieza total y reseteo del estado de caché al hacer clic en **"🗑️ Limpiar"**.
* **Suite de Pruebas Automatizadas:**
  * Ampliación a 90 pruebas unitarias con cobertura específica para caché diferencial de sesión y reactivación dinámica de UI.

---

## [2.2.0] - 2026-10-16

### Agregado
* **Transcripción por Lotes Multiarchivo (FIFO Queue):**
  * Soporte para seleccionar o arrastrar múltiples archivos de audio y video en la interfaz Web y CLI.
  * Procesamiento secuencial ordenado para preservar la VRAM y prevenir desbordamientos de memoria en GPUs de consumo y ZeroGPU.
* **Descarga Consolidada en Archivo Único (`.txt`):**
  * Nueva opción para descargar todas las transcripciones de un lote unificadas en un único archivo de texto plano con encabezados y separadores legibles por archivo.
  * Inclusión del archivo consolidado dentro del archivo comprimido `.zip` del lote.
* **Iluminación Dinámica y Protección de Ejecución Accidental:**
  * Al finalizar la transcripción, el botón **"⚡ Iniciar Transcripción"** se deshabilita automáticamente (`interactive=False`) con estilo atenuado y cursor bloqueado para evitar clics accidentales que consuman saldo de API, reactivándose únicamente tras hacer clic en **"🗑️ Limpiar"**.
  * **Iluminación individual:** El botón de descarga para un solo archivo se ilumina en degradado Verde Aurora / Ice Cyan con resplandor pulsante.
  * **Diferenciación cromática en lotes grupales:** En transcripciones multiarchivo, ambos botones se iluminan en colores contrastantes del sistema nórdico:
    * `📦 Descargar Lote (.zip)` resalta en **Aurora Emerald / Verde Neón**.
    * `📄 Descargar Consolidado (.txt)` resalta en **Violet Sky / Rosa Glaciar Neón**.
* **Suite de Pruebas Automatizadas:**
  * Ampliación de la cobertura a 88 pruebas unitarias de regresión abarcando dominio, CLI, UI y manejo de errores.

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
* **Especificación Formal de Credenciales y Secretos:**
  * Documentación técnica estandarizada para la inyección de secretos en GitHub Actions, Hugging Face Spaces y entornos locales.
* **Diagrama de Arquitectura de Sistemas y Taxonomía de Modelos:**
  * Modelado formal de arquitectura de software en Mermaid describiendo los subsistemas de integración continua (CI/CD), presentación, enrutamiento desacoplado, motores de inferencia heterogéneos y exportación de artefactos.
  * Evaluación comparativa y taxonomía de rendimiento para modelos ASR en Deepgram (Nova), AssemblyAI (Conformer/Nano) y WhisperX (Open-Weights).
  * Mapa detallado de módulos y responsabilidades para auditoría técnica de repositorio.

### Cambiado / Optimizado
* **Duración Extendida en Hugging Face ZeroGPU:**
  * Actualizado el decorador a `@spaces.GPU(duration=180)` para reservar hasta 3 minutos de GPU continua, eliminando la advertencia *"GPU task is exceeding its requested duration and might be aborted"* en grabaciones largas de ~2 horas.
* **Resiliencia en Diarización con Pyannote Audio:**
  * Soporte retrocompatible dual para `DiarizationPipeline(token=...)` y `DiarizationPipeline(use_auth_token=...)`.
  * Captura descriptiva de errores de autenticación con enlaces directos para aceptar las licencias de `pyannote/speaker-diarization-3.1` y `pyannote/segmentation-3.0`.
* **Auditoría Integral de Código (Clean Code & Refactorización):**
  * Eliminación de importaciones y modelos huérfanos en `app.py`.
  * Refactorización DRY (Don't Repeat Yourself) de resolución de secretos (`_resolve_secret`) en `src/utils.py`.
  * Creación del wrapper de compatibilidad `main.py` delegando a `cli.main()`.
  * Actualización de `cli.py` con soporte multiplataforma para `--provider` y `--api-key`.
  * Sincronización de dependencias en `environment.yml` y actualización de versión a `2.1.0` en `pyproject.toml`.
  * Formateo y validación de código con `ruff` (`ruff format` y `ruff check`) con 0 advertencias.

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
