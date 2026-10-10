# Pipeline de Transcripción y Diarización con WhisperX

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live%20Demo-blue)](https://huggingface.co/spaces/Klopezxd/transcriptor-whisperx)
![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg?logo=nvidia&logoColor=white)
![Linter](https://img.shields.io/badge/Linter-Ruff-000000.svg?logo=ruff&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> Pipeline de transcripción de audio y video con alineación fonética a nivel de palabra y diarización de locutores basado en WhisperX y Pyannote Audio.

*Read this document in English: [README.md](README.md)*

---

> [!IMPORTANT]
> # 🔑 GUÍA MAESTRA DE API KEYS Y SECRETOS (¡CONFIGURACIÓN OBLIGATORIA!)
> Este proyecto conecta **GitHub** con **Hugging Face Spaces**. Para que todo funcione al 100% (sincronización automática del repositorio y diarización de interlocutores), debes configurar los siguientes tokens:
>
> ### 1️⃣ TOKEN EN GITHUB: Para sincronizar el código con Hugging Face (`HF_TOKEN`)
> * **Para qué sirve:** Permite que cada vez que hagas `git push` a GitHub, el workflow de GitHub Actions actualice automáticamente tu aplicación en Hugging Face Spaces ([Klopezxd/transcriptor-whisperx](https://huggingface.co/spaces/Klopezxd/transcriptor-whisperx)).
> * **Cómo obtenerlo:**
>   1. Entra a [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens).
>   2. Crea un nuevo token con permisos de **ESCRITURA (`Write`)** *(si creas uno de lectura `Read`, GitHub no podrá subir los cambios al Space y fallará)*.
> * **Dónde configurarlo en GitHub:**
>   1. Ve a tu repositorio en GitHub: [whisperx-transcriptor](https://github.com/Klopezxd/whisperx-transcriptor).
>   2. Entra a la pestaña **Settings** ➔ Menú lateral izquierdo: **Secrets and variables** ➔ **Actions**.
>   3. Clic en el botón verde **New repository secret**.
>   4. **Name:** `HF_TOKEN` | **Secret:** Tu token de Hugging Face (`hf_...`).
>
> ---
>
> ### 2️⃣ TOKEN EN HUGGING FACE SPACES: Para Diarización de Hablantes (`HF_TOKEN`)
> * **Para qué sirve:** Permite que WhisperX descargue los modelos de Pyannote en el Space para separar e identificar quién habla.
> * **PASO OBLIGATORIO PREVIO (Aceptar licencias):** Con tu cuenta de Hugging Face, debes entrar y hacer clic en **"Aceptar condiciones"** en **AMBOS** modelos:
>   * 👉 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
>   * 👉 [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) *(¡si falta aceptar este segundo modelo, la diarización fallará con error 401/403!)*.
> * **Dónde configurarlo en Hugging Face Spaces:**
>   1. Ve a tu Space: [huggingface.co/spaces/Klopezxd/transcriptor-whisperx](https://huggingface.co/spaces/Klopezxd/transcriptor-whisperx).
>   2. Entra a la pestaña **Settings** ➔ Baja hasta la sección **Variables and secrets**.
>   3. En la subsección **Secrets**, haz clic en **New secret**.
>   4. **Name:** `HF_TOKEN` | **Value:** Tu token de Hugging Face (`hf_...`).
>   *(Nota: También puedes ingresar tu token directamente en el campo "Token de Hugging Face" dentro de la interfaz web si no deseas guardarlo en Secrets).*
>
> ---
>
> ### 3️⃣ SALDO GRATUITO DE $200 USD: Deepgram Cloud API (Opcional pero recomendado)
> * **Para qué sirve:** Transcribir audios pesados (de 2 a 5+ horas) en menos de 20 segundos con diarización incluida, sin consumir cuota de ZeroGPU ni interferir con la cuota de Hermes Agent en Groq.
> * **Requisitos:** Cero. No requiere tarjeta de crédito ni correo institucional. Válido para cualquier usuario nuevo.
> * **Cómo obtenerlo:**
>   1. Entra a [console.deepgram.com/signup](https://console.deepgram.com/signup) y regístrate con tu correo (o con Google/GitHub).
>   2. Al ingresar al panel, verás tus **$200 USD de saldo de bienvenida** activos automáticamente (rinden para +600 horas de audio).
>   3. En el menú lateral izquierdo, ve a **API Keys** ➔ **Create a New API Key** para generar tu clave.
>
> ---


## Características Principales

* **Inferencia de Alto Rendimiento:** Basado en **WhisperX** (faster-whisper / CTranslate2), hasta $5\times$ más rápido que implementaciones tradicionales de Whisper.
* **Alineación Fonética Precisa:** Integra modelos **Wav2Vec2** para alinear fonéticamente marcas de tiempo exactas por palabra.
* **Diarización de Hablantes:** Identifica y segmenta interlocutores automáticamente mediante **Pyannote Audio 3.1**.
* **Optimización de Memoria (4GB VRAM & ZeroGPU):** Diseñado con cuantización `INT8` y recolección forzada de memoria entre fases, permitiendo ejecución fluida tanto en GPUs de consumo (ej. NVIDIA GTX 1650 con 4GB VRAM) como en **Hugging Face ZeroGPU**.
* **Doble Interfaz (Web + CLI):**
  * **Web UI (Gradio):** Interfaz para navegador con controles de parámetros en tiempo real.
  * **CLI (Terminal):** Soporte completo de argumentos por consola (`argparse`) con selector gráfico interactivo de respaldo.
* **Ingeniería de Software:** Tipado estricto (Type Hints), logging estructurado, pipeline desacoplado y gestión segura de credenciales vía variables de entorno.

---

## Demo en Vivo

Puedes probar el pipeline directamente en la nube sin configuración local:

👉 **[Abrir en Hugging Face Spaces: Klopezxd/transcriptor-whisperx](https://huggingface.co/spaces/Klopezxd/transcriptor-whisperx)**

---

## Arquitectura del Pipeline

El pipeline procesa el archivo multimedia en 5 etapas secuenciales con liberación forzada de VRAM entre cada una:

```mermaid
flowchart TD
    A["Archivo Multimedia<br>(MP4 / WAV / MP3)"] --> B["Extracción y Carga<br>de Audio"]
    B --> C["1. Transcripción ASR<br>(WhisperX)"]
    C -->|Libera VRAM| D["2. Alineación Fonética<br>(Wav2Vec2)"]
    D -->|Libera VRAM| E{"¿Diarización<br>Activa?"}
    E -->|Sí| F["3. Segmentación de Hablantes<br>(Pyannote Audio)"]
    E -->|No| G["4. Formateador de Segmentos"]
    F -->|Libera VRAM| G
    G --> H["Documento Final (.txt)<br>+ Vista Previa"]
```

---

## Requisitos del Sistema

* **Sistema Operativo:** Windows 10/11 o Linux (Ubuntu 20.04+).
* **Python:** 3.10 (recomendado).
* **FFmpeg:** Instalado en el sistema y disponible en el `PATH`.
* **Hardware GPU (Opcional pero recomendado):** NVIDIA GeForce GTX 1650 o superior con drivers actualizados y soporte CUDA 12.x. (También funciona en CPU).

---

## Instalación Local

### Opción 1: Con Conda / Miniconda (Recomendado)

```bash
# 1. Clonar el repositorio
git clone https://github.com/Klopezxd/whisperx-transcriptor.git
cd whisperx-transcriptor

# 2. Crear el entorno aislado con Python 3.10
conda create -n transcriptor python=3.10 -y
conda activate transcriptor

# 3. Instalar PyTorch con aceleración CUDA 12.x
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# 4. Instalar WhisperX y dependencias del proyecto
pip install -r requirements.txt

# 5. Verificar aceleración GPU
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
```

### Opción 2: Con entorno virtual venv estándar

```bash
python -m venv venv
# En Windows:
.\venv\Scripts\activate
# En Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## Configuración del Token de Hugging Face (Diarización)

La identificación de interlocutores requiere el modelo `pyannote/speaker-diarization-3.1`. Para utilizarlo:

1. Crea una cuenta gratuita en [Hugging Face](https://huggingface.co/).
2. Acepta los términos de uso en:
   * [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   * [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Genera un Token de lectura en [Hugging Face Settings](https://huggingface.co/settings/tokens).
4. Configúralo en tu entorno:
   ```bash
   # Copiar la plantilla de ejemplo
   cp .env.example .env
   # Editar .env y colocar:
   HF_TOKEN=hf_tu_token_aqui
   ```
   *También puedes pasarlo como argumento `--token` en la CLI.*

---

## Modos de Ejecución: Nube vs. Local

| Modo | Ideal para | Ventajas Principales | Limitaciones |
|---|---|---|---|
| **Hugging Face Spaces (Nube)** | Audios cortos y medianos (< 20-30 min), acceso inmediato | Cero instalación; corre en GPU NVIDIA A10G (24GB VRAM) en la nube | Sujeto a cuotas diarias gratuitas de ZeroGPU |
| **Ejecución Local (PC)** | Grabaciones largas (1 a 5+ horas), lotes masivos, audios confidenciales | Sin límites de tiempo, 100% privado y offline | Utiliza hardware local (GPU/CPU) |

---

## Guía de Uso

### 1. Lanzador Rápido en Windows (1 Clic)

Haz doble clic en `run_transcriptor.bat` en la raíz del proyecto (o ejecútalo desde terminal). Detecta y activa el entorno Conda automáticamente y abre la ventana gráfica para elegir tu archivo.

---

### 2. Interfaz Web Local (Gradio)

Lanza la misma interfaz web directamente en tu máquina:

```bash
python app.py
```
Abre tu navegador en `http://127.0.0.1:7860` para usar la interfaz visual.

---

### 3. Interfaz de Línea de Comandos (CLI)

```bash
# Modo interactivo (abre ventana gráfica para elegir el archivo si no pasas -i):
python cli.py

# Transcripción directa con modelo turbo:
python cli.py -i "video_clase.mp4" -m large-v3-turbo

# Transcripción con Diarización (identificación de personas):
python cli.py -i "reunion.mp4" -d

# Guardar en carpeta específica con formato de marcas de tiempo en rango:
python cli.py -i "conferencia.wav" --timestamps range -o "./transcripciones"
```

#### Opciones CLI disponibles:

| Flag | Tipo | Descripción | Por defecto |
|---|---|---|---|
| `-i`, `--input` | String | Ruta al archivo multimedia. Si se omite, abre selector gráfico. | `None` |
| `-m`, `--model` | Choice | `large-v3-turbo`, `large-v3`, `medium`, `small`, `base` | `large-v3-turbo` |
| `-d`, `--diarize` | Flag | Activa la identificación de hablantes (requiere token). | `False` |
| `--timestamps` | Choice | Formato de tiempo: `simple`, `range`, `none` | `simple` |
| `-t`, `--token` | String | Token explícito de Hugging Face. | Variable `HF_TOKEN` |
| `--compute-type` | Choice | Cuantización: `int8` (recomendado 4GB), `float16`, `float32` | `int8` |
| `-o`, `--output-dir` | String | Directorio destino para el `.txt`. | Mismo del archivo |

---

## Estructura del Proyecto

```text
whisperx-transcriptor/
├── .github/
│   └── workflows/
│       ├── sync-to-hf.yml       # CI/CD: Sincronización automática con Hugging Face Spaces
│       └── lint.yml             # Análisis estático de código con Ruff
├── src/
│   ├── __init__.py
│   ├── config.py                # Dataclasses inmutables y enums
│   ├── transcriber.py           # Pipeline orquestador WhisperX + Pyannote
│   └── utils.py                 # Funciones puras de VRAM, formateo y seguridad
├── app.py                       # Interfaz Web Gradio (Local + ZeroGPU)
├── cli.py                       # Interfaz CLI avanzada
├── requirements.txt             # Dependencias Python
├── packages.txt                 # Dependencias del SO (FFmpeg en HF Spaces)
├── environment.yml              # Definición de entorno Conda
├── .env.example                 # Plantilla de credenciales
├── .gitignore                   # Exclusión estricta de medios y temporales
├── pyproject.toml               # Configuración de empaquetado y linters
├── LICENSE                      # Licencia MIT
├── README.md                    # Documentación principal en Inglés
└── README.es.md                 # Documentación en Español
```

---

## Licencia y Autoría

* **Autor:** [Klever López](https://github.com/Klopezxd)
* **Licencia:** MIT License — libre para uso personal, académico y comercial.

