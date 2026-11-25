# Registro de Cambios (Changelog)

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
y este proyecto se adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [2.4.5] - 2026-11-24

### Agregado
* **Parada Elegante, Entrega Parcial Progresiva y Reanudación de Lotes:**
  * Conversión del procesador web en flujo progresivo (`process_transcription_stream`), entregando resultados en caché tras cada archivo completado.
  * Nuevo botón cooperativo **`🛑 Detener tras este archivo`** (`btn_stop_transcribe`) que alterna el mismo espacio visual del botón principal únicamente cuando existen 2 o más archivos pendientes, permitiendo finalizar limpiamente el archivo en curso, generar el `.txt` consolidado y `.zip` parcial con lo ya transcrito, y deshacer la pausa con un segundo clic (**`↩️ Cancelar pausa (seguir con la cola)`**) antes de que termine el archivo actual.
  * Al pausar, el botón principal pasa a **`▶️ Reanudar (N pendientes)`** retomando el lote exactamente desde el siguiente archivo sin repetir los ya transcritos.
  * Invalidación automática del caché diferencial mediante firma de configuración (`build_config_signature`) si se cambia proveedor, modelo, diarización o marcas de tiempo antes de reanudar.
* **Normalización de finales de línea y pruebas de regresión:**
  * Añadido [`.gitattributes`](.gitattributes) para garantizar finales de línea `LF` consistentes en archivos de código, documentación y dependencias (`packages.txt`).
  * Ampliación de la suite automatizada a **108 pruebas unitarias y de regresión** (`tests/test_app.py`, `tests/test_auth.py`, `tests/test_core.py`).

### Cambiado
* **Consolidación 100% Web-First y separación de `audio-extractor`:**
  * Eliminados los componentes legados exclusivos de escritorio/CLI (`main.py`, `cli.py`, `run_transcriptor.bat`, `environment.yml`, `bin/ffmpeg.exe` y `tests/test_cli.py`) para mantener el repositorio enfocado en Hugging Face Spaces y despliegue web Gradio.
  * El extractor opcional por lotes en Windows fue separado en su propio repositorio dedicado: [`Klopezxd/audio-extractor`](https://github.com/Klopezxd/audio-extractor), enlazado desde la documentación a su release compilado.
* **Fijación de dependencias y metadatos:**
  * Versión del paquete sincronizada en `2.4.5` (`pyproject.toml`, `src/__init__.py`) y dependencia `whisperX` fijada a commit SHA en `requirements.txt`.
  * Workflows de GitHub Actions (`.github/workflows/lint.yml`, `sync-to-hf.yml`) fijados a commit SHAs con `persist-credentials: false` y verificación estática de seguridad con Ruff (`--select S`).

### Seguridad
* **Sesiones Admin verificadas en servidor y aislamiento estricto de secretos (`app.py`, `src/auth.py`, `src/utils.py`):**
  * Autenticación de administrador mediante tokens aleatorios en servidor (`_ADMIN_SESSIONS` con `secrets.token_urlsafe(32)`, validación en tiempo constante vía `hmac.compare_digest`, TTL de inactividad de 60 min y TTL absoluto de 8 h).
  * Protección privada integral de los secretos del servidor (`HF_TOKEN`, `DEEPGRAM_API_KEY` y `ASSEMBLYAI_API_KEY` permanecen bloqueados y reservados exclusivamente para sesiones Admin; `ALLOW_PUBLIC_HF_TOKEN` permanece en `false` por defecto).
  * Soporte para hash de PIN `ADMIN_PIN_HASH` (`scrypt$salt_hex$digest_hex`, con retrocompatibilidad para `ADMIN_PIN`), resolución segura de `.env` independiente del directorio de trabajo y 21 de 22 eventos Gradio marcados con `api_visibility="private"`.
  * Endurecimiento de `src/auth.py` contra IP spoofing mediante validación estricta con `ipaddress`, confianza exclusiva en `X-Forwarded-For` / `CF-Connecting-IP` / `X-Real-IP` desde `TRUSTED_PROXY_IPS` o `TRUST_PROXY_HEADERS=true`, y limitador global secundario contra rotación o evicción LRU.
  * Aislamiento de descargas por ejecución mediante `output_dir` temporal (`tempfile.mkdtemp`), rechazo de symlinks y verificación de contención canónica en `validate_media_path`, deduplicación de rutas dentro de `create_zip_archive`, `max_file_size` (`MAX_UPLOAD_SIZE_MB`, defecto `2gb`) y `blocked_paths`.
  * Eliminación de superficies DOM XSS en `assets/js/ui_enhancements.js` sustituyendo `innerHTML` y manejadores inline por `createElement`, `textContent` y `addEventListener`.

### Corregido
* **Motor WhisperX y conectores Cloud (`src/transcriber.py`, `src/cloud_providers.py`, `src/utils.py`):**
  * Fallback automático en Fase 3 si falla la carga/ejecución de alineación fonética Wav2Vec2 o si el idioma detectado es desconocido, conservando los segmentos transcritos sin abortar el lote.
  * Reevaluación dinámica de disponibilidad CUDA dentro de `@spaces.GPU` (`device_auto`), reducción progresiva de `batch_size` (`16 -> 4 -> 1`) y degradación segura a `cpu` (`int8`) ante CUDA OOM.
  * Reintentos con backoff exponencial y respeto de `Retry-After` ante códigos `408`/`429`/`5xx`, progreso monotónico desacoplado de carga (`ProgressFileReader` con `seek`/`tell`/`readable`), eliminación inmediata de temporales tras subir audio a AssemblyAI, parseo seguro de `speaker`/`null` y endurecimiento de `ffmpeg` (`-nostdin`, `-protocol_whitelist file`, `timeout=180`, soporte `.webm`).
* **Interfaz Gradio 6.30 / Svelte 5 (`assets/css/theme.css`, `assets/js/ui_enhancements.js`, `src/ui/components.py`, `src/ui/helpers.py`):**
  * Corrección de reglas `.hidden` frente a `.hide` para evitar la visibilidad simultánea de `🔐 Admin`/`🔓 Salir` y botones ocultos en móvil/pausa.
  * Sustitución de mutación directa de `textContent` en `#btn_transcribe` durante la cola por `data-queue-label` + CSS `::after` compatible con Svelte 5, y cancelación limpia de `transcribe_evt` y decodificaciones Web Audio al pulsar Limpiar.
  * Alineación simétrica de columnas en escritorio (`1280x800` y `1920x1080`) bajo `main.fillable > .wrap > .contain` (`equal_height=True`, `align-self: stretch; height: auto`), ajuste sin recorte de la etiqueta `Lossless` y fallback responsivo para viewports bajos (`height < 560px`).

## [2.4.4] - 2026-11-10

### Corregido
* **Desplazamiento Vertical en Pantalla Dividida (Split Screen / Snap):**
  * Resuelto el bloqueo de desplazamiento (scroll) en Hugging Face Spaces (`<= 1024px`) y ventanas divididas en PC al sincronizar dinámicamente la altura del iframe (`window.parentIFrame.size`) con la altura acumulada del contenido (`getContentHeight`).
  * Ajuste de la media query de escritorio a `min-width: 1025px` con `overflow-y: auto`, eliminando la colisión que bloqueaba el scroll en resoluciones intermedias.
* **Restauración a Pantalla Completa sin Colapso Visual:**
  * Asignadas alturas mínimas garantizadas en escritorio (`min-height: clamp(160px, 22vh, 240px)` en `.upload-card`, `120px` en `.file-upload-zone` y `80px` en `.file-preview-holder`), impidiendo que flexbox aplaste la zona de subida al maximizar la ventana con credenciales de API visibles.
  * Implementada la consulta reactiva `requestParentPageInfo()` en cada evento de cambio de viewport para recalcular inmediatamente las dimensiones al alternar entre media pantalla y pantalla completa.
* **Compatibilidad con Cuotas de Hugging Face ZeroGPU:**
  * Ajustada la duración por defecto del decorador `@spaces.GPU` a `60` segundos (configurable vía variable de entorno `ZERO_GPU_DURATION`), eliminando el error `'The requested GPU duration (270s) is larger than the maximum allowed'` causado por el multiplicador de reserva 1.5x de Hugging Face sobre el límite de 120s del tier gratuito.
  * Añadida notificación de ayuda contextual en caso de agotamiento temporal de cuotas GPU.
* **Actualización del Conector AssemblyAI:**
  * Migración del parámetro deprecado `speech_model` a la lista prioritaria `speech_models`, estableciendo `universal-3-5-pro` como modelo predeterminado de máxima precisión con respaldo en `universal-2`.

## [2.4.3] - 2026-11-05

### Agregado
* **Cola Inteligente de Transcripción Asíncrona («Dejarlo Trabajando»):**
  * Posibilidad de hacer clic en **"⚡ Iniciar Transcripción"** en cualquier momento mientras se analizan, optimizan o suben los archivos (tanto vía selector como mediante arrastrar y soltar).
  * La interfaz intercepta la acción en fase de captura, activa el modo en cola (`⏳ En cola (iniciará al terminar subida)`) con iluminación pulsante ámbar y lanza la transcripción automáticamente en cuanto los archivos terminan de subirse al servidor, permitiendo desentenderse del proceso.
  * Permite cancelar la programación con un segundo clic sobre el botón antes de la subida.

### Optimizado
* **Detección Instantánea de Archivos No Optimizables (<30 ms por lote):**
  * Sondeo previo ultrarrápido con metadatos HTML5 que evita leer archivos completos en memoria (`ArrayBuffer`) cuando no es viable optimizarlos.
  * Detección y desvío inmediato a subida directa para contenedores no decodificables por la Web Audio API del navegador (`.mkv`, `.avi`, `.flv`, `.wmv`) y para videos con bitrate ultra liviano cuyo peso original ya es menor o igual que el WAV PCM de 16 kHz esperado.
  * Eliminación del cuello de botella en lotes extensos (ej. 21 videos): los archivos no optimizables pasan de inmediato a subida directa sin re-decodificaciones innecesarias, aprovechando la extracción acelerada con FFmpeg nativo en servidor.

## [2.4.2] - 2026-10-23

### Agregado
* **Utilidad Ligera de Extracción de Audio (`extraer_audio.bat` - Cero Python):**
  * Script directo para Windows diseñado para resolver el cuello de botella de grabaciones universitarias de OBS Studio y videos pesados (>1.5 GB).
  * Reduce archivos de 1.5 GB a ~50 MB (-96% de peso) en segundos extrayendo audio en formato AAC Mono a 64 kbps (`.m4a`).
  * 100% nativo en Windows: funciona con simple arrastrar y soltar (Drag & Drop) sobre el icono, sin requerir Python, Conda ni librerías adicionales.
* **Extracción Web en Cliente (WAV PCM 16 kHz Mono):**
  * En navegador: generación directa de **WAV PCM lineal de 16 bits a 16 kHz Mono** para videos <= 450 MB.
  * Eliminación de la biblioteca `lame.min.js`, reduciendo el bundle frontend web en ~160 KB.
* **Barra de Estado Ultra Compacta (Zero-Scroll / Ahorro de Espacio Vertical):**
  * Reemplazo de los 3 bloques apilados por una barra de estado dinámica y fluida de 1 sola línea (22px).
  * Umbral de 450 MB en cliente para delegar videos extensos directamente a FFmpeg en servidor sin congelamientos de memoria.

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
