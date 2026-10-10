# WhisperX Audio Transcriptor & Diarization Pipeline

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live%20Demo-blue)](https://huggingface.co/spaces/Klopezxd/transcriptor-whisperx)
![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg?logo=nvidia&logoColor=white)
![Linter](https://img.shields.io/badge/Linter-Ruff-000000.svg?logo=ruff&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> High-throughput speech transcription pipeline featuring word-level phoneme alignment and speaker diarization using WhisperX and Pyannote Audio.

*Leer este documento en Español: [README.es.md](README.es.md)*

---

> [!IMPORTANT]
> # 🔑 MASTER GUIDE TO API KEYS & SECRETS (MANDATORY CONFIGURATION!)
> This repository bridges **GitHub** and **Hugging Face Spaces**. For automated deployment and speaker diarization to work properly, you must configure the following tokens:
>
> ### 1️⃣ TOKEN IN GITHUB: For Syncing Code to Hugging Face (`HF_TOKEN`)
> * **Purpose:** Enables GitHub Actions to push repository changes directly to your Hugging Face Space ([Klopezxd/transcriptor-whisperx](https://huggingface.co/spaces/Klopezxd/transcriptor-whisperx)) whenever you run `git push`.
> * **How to obtain it:**
>   1. Go to [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens).
>   2. Generate a token with **WRITE (`Write`)** permissions *(a `Read` token will fail during git push)*.
> * **Where to configure it in GitHub:**
>   1. Navigate to your GitHub repository: [whisperx-transcriptor](https://github.com/Klopezxd/whisperx-transcriptor).
>   2. Click on **Settings** ➔ Left sidebar: **Secrets and variables** ➔ **Actions**.
>   3. Click the green button **New repository secret**.
>   4. **Name:** `HF_TOKEN` | **Secret:** Your Hugging Face write token (`hf_...`).
>
> ---
>
> ### 2️⃣ TOKEN IN HUGGING FACE SPACES: For Speaker Diarization (`HF_TOKEN`)
> * **Purpose:** Allows WhisperX to pull Pyannote audio models in the Space to identify individual speakers.
> * **MANDATORY PREREQUISITE (Accept Licenses):** Using your Hugging Face account, you MUST visit and click **"Accept conditions"** on **BOTH** model repositories:
>   * 👉 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
>   * 👉 [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) *(missing this will trigger a 401/403 authorization error!)*.
> * **Where to configure it in Hugging Face Spaces:**
>   1. Open your Space: [huggingface.co/spaces/Klopezxd/transcriptor-whisperx](https://huggingface.co/spaces/Klopezxd/transcriptor-whisperx).
>   2. Navigate to **Settings** ➔ Scroll down to **Variables and secrets**.
>   3. Under **Secrets**, click **New secret**.
>   4. **Name:** `HF_TOKEN` | **Value:** Your Hugging Face token (`hf_...`).
>   *(Note: You can also enter the token directly in the "Token de Hugging Face" field inside the web UI).*
>
> ---
>
> ### 3️⃣ FREE $200 USD PROMOTIONAL CREDIT: Deepgram Cloud API (Recommended for large audio)
> * **Purpose:** Transcribe massive audio files (2 to 5+ hours) in under 20 seconds with built-in diarization without burning ZeroGPU quotas.
> * **Requirements:** None. No credit card required, open to all new users.
> * **How to get it:**
>   1. Sign up at [console.deepgram.com/signup](https://console.deepgram.com/signup) with any email or GitHub.
>   2. Your account is automatically credited with **$200 in free credits** (+700 hours of transcription).
>   3. Navigate to **API Keys** ➔ **Create a New API Key**.
> * **Where to configure it:** Hugging Face Space Secrets (`DEEPGRAM_API_KEY`), local `.env` file, or directly in the web UI.
>
> ---
>
> ### 4️⃣ FREE $50 USD PROMOTIONAL CREDIT: AssemblyAI Cloud API (High-precision alternative)
> * **Purpose:** Studio-grade transcription fidelity (Universal-1) and utterance-level diarization without GPU consumption.
> * **Requirements:** None. No credit card required.
> * **How to get it:**
>   1. Sign up at [assemblyai.com/dashboard/signup](https://www.assemblyai.com/dashboard/signup).
>   2. Your dashboard is automatically credited with **$50 in free credits** (~150 hours of transcription).
>   3. Copy your key under **API Keys**.
> * **Where to configure it:** Hugging Face Space Secrets (`ASSEMBLYAI_API_KEY`), local `.env` file, or directly in the web UI.
>
> ---
>
> ### 📋 SECRETS & CONFIGURATION MATRIX
>
> | Key / Environment Variable | Provider | Where to Obtain | Where to Configure | Required? | Primary Purpose |
> | :--- | :--- | :--- | :--- | :--- | :--- |
> | **`HF_TOKEN`** *(Write)* | Hugging Face | [HF Tokens](https://huggingface.co/settings/tokens) | **GitHub Repository Secrets** (`Settings > Secrets > Actions`) | **Yes** | Syncs code from GitHub to Hugging Face Space on every `git push`. |
> | **`HF_TOKEN`** *(Read)* | Hugging Face | [HF Tokens](https://huggingface.co/settings/tokens) | **HF Space Secrets** or Web UI | **Yes** *(for WhisperX)* | Downloads Pyannote speaker diarization models (requires model license consent). |
> | **`DEEPGRAM_API_KEY`** | Deepgram | [Deepgram Console](https://console.deepgram.com) | **HF Space Secrets**, Web UI or `.env` | Optional | Fast cloud transcription (20s) and diarization without GPU quotas. |
> | **`ASSEMBLYAI_API_KEY`** | AssemblyAI | [AssemblyAI Dashboard](https://www.assemblyai.com) | **HF Space Secrets**, Web UI or `.env` | Optional | High-precision cloud transcription without GPU quotas. |


## Key Features

* **High-Throughput ASR:** Powered by **WhisperX** (faster-whisper / CTranslate2), delivering up to $5\times$ speedups over vanilla Whisper implementations.
* **Word-Level Phoneme Alignment:** Incorporates **Wav2Vec2** acoustic models to achieve exact word-boundary timestamp alignment.
* **Speaker Diarization:** Identifies and segments discrete speakers seamlessly using **Pyannote Audio 3.1**.
* **Consumer GPU & ZeroGPU Ready:** Architected with `INT8` quantization and explicit VRAM garbage collection between pipeline stages, running efficiently on 4GB VRAM cards (e.g., NVIDIA GeForce GTX 1650) as well as **Hugging Face ZeroGPU** runners.
* **Dual Interface (Web + CLI):**
  * **Web UI (Gradio):** Interactive browser application.
  * **CLI (Terminal):** Fully featured command-line interface with interactive graphical file picker fallback.
* **Production Engineering:** Explicit type hints, structured logging, decoupled domain logic, and secure environment-based credential handling.

---

## Live Interactive Demo

Test the pipeline directly on Hugging Face Spaces:

👉 **[Launch on Hugging Face Spaces: Klopezxd/transcriptor-whisperx](https://huggingface.co/spaces/Klopezxd/transcriptor-whisperx)**

---

## 🏛️ System Architecture

The pipeline implements a decoupled, multi-engine architecture continuously synced between GitHub and Hugging Face Spaces. Users can switch between local/ZeroGPU execution and high-speed cloud APIs with a single click:

```mermaid
flowchart TD
    subgraph CI["🔄 Continuous Deployment (CI/CD)"]
        GH["GitHub Repository<br><b>whisperx-transcriptor</b>"] -->|git push| GHA["GitHub Actions<br><code>sync-to-hf.yml</code>"]
        GHA -->|Uses HF_TOKEN Write| HFS["Hugging Face Space<br><code>Klopezxd/transcriptor-whisperx</code>"]
    end

    subgraph INGRESS["🖥️ Ingress & User Interface"]
        USER(["User / Web Browser"]) --> UI["Web UI (Gradio)<br><code>app.py</code><br><i>Nordic Aurora Theme + Statusbar</i>"]
        CLI["Terminal CLI<br><code>main.py</code>"] --> ROUTER
        UI --> ROUTER{"Engine Router<br><code>src/transcriber.py</code>"}
    end

    subgraph ENGINES["⚙️ ASR & Diarization Engines"]
        ROUTER -->|Provider: WhisperX| WX_BOX["WhisperX Engine (ZeroGPU / Local)<br><code>src/transcriber.py</code>"]
        ROUTER -->|Provider: Deepgram| DG_BOX["Deepgram REST API<br><code>src/cloud_providers.py</code>"]
        ROUTER -->|Provider: AssemblyAI| AAI_BOX["AssemblyAI REST API<br><code>src/cloud_providers.py</code>"]

        WX_BOX -->|@spaces.GPU| WX_PIPE["1. Faster-Whisper ASR<br>2. Wav2Vec2 Alignment<br>3. Pyannote 3.1 Diarization"]
        WX_PIPE -.->|Requires HF_TOKEN Read| HF_MODELS[("Hugging Face Hub<br><i>Pyannote Models</i>")]

        DG_BOX -->|Direct HTTP POST| DG_API[("Deepgram Cloud<br>Models: nova-2 / nova-3<br><i>$200 USD Credit</i>")]
        AAI_BOX -->|HTTP Chunked Upload| AAI_API[("AssemblyAI Cloud<br>Models: best / nano<br><i>$50 USD Credit</i>")]
    end

    subgraph EGRESS["📄 Unified Normalization & Multi-Format Export"]
        WX_PIPE --> HARMONIZER["Result Normalizer<br><code>TranscriptionResult</code>"]
        DG_API --> HARMONIZER
        AAI_API --> HARMONIZER

        HARMONIZER --> EXPORTS["Output Artifacts:<br>• Plain Text (.txt)<br>• Subtitles (.srt)<br>• Web Subtitles (.vtt)<br>• Structured Data (.json)"]
        EXPORTS --> DOWNLOADS["Direct Download Buttons<br>& Interactive Speaker Preview"]
    end

    HFS -.->|Deploys & Serves| UI
```

---

### 📂 Project Structure & Module Responsibilities

If you need to audit or manage this repository in the future without a local clone, here is the exact architectural breakdown of each component:

```text
whisperx-transcriptor/
│
├── app.py                  # Gradio Web Interface: Nordic Aurora theme, academic statusbar v2.1.0,
│                           # reactive model dropdowns, and isolated @spaces.GPU decorator execution.
│
├── main.py                 # Terminal CLI entry point: direct parameter execution (--media, --model,
│                           # --diarize) with interactive graphical file picker fallback.
│
├── requirements.txt        # Runtime dependencies (torch, whisperx, gradio, requests, pyannote).
├── CHANGELOG.md            # Formal versioning and release notes conforming to Keep a Changelog.
├── .env.example            # Environment variable template for local development and secrets testing.
│
├── .github/workflows/
│   └── sync-to-hf.yml      # CI/CD Workflow: Automatically syncs git commits on main to Hugging Face
│                           # Space using secret HF_TOKEN (with Write permissions).
│
└── src/
    ├── config.py           # Strongly typed data models and enums: ProviderType, WhisperModel,
    │                       # DeepgramModel, AssemblyAIModel, PROVIDER_MODELS, and TranscriptionConfig.
    │
    ├── transcriber.py      # Pipeline factory: create_transcription_pipeline, VRAM memory managers,
    │                       # Pyannote Audio integration, and local WhisperXPipeline orchestrator.
    │
    ├── cloud_providers.py  # Lightweight cloud REST connectors: DeepgramPipeline & AssemblyAIPipeline
    │                       # featuring word-level diarization, async polling, and ZERO GPU usage.
    │
    └── utils.py            # Utility helpers: hierarchical API key resolution (arg > os.getenv > .env),
                            # subtitle formatters (.srt, .vtt, .json), and VRAM garbage collectors.
```

---

### ⚡ Engine Execution Lifecycle

1. **WhisperX Pipeline (Local GPU or Hugging Face ZeroGPU):**
   * **Compute Cost:** Local GPU VRAM or Hugging Face ZeroGPU minutes (timeout set to 180s for files up to ~2 hours).
   * **Sequential Stages with VRAM Eviction:**
     * `Audio Loading` ➔ `WhisperX ASR` (CTranslate2) ➔ `VRAM Flush`
     * ➔ `Wav2Vec2 Alignment` ➔ `VRAM Flush`
     * ➔ `Pyannote 3.1 Diarization` (if enabled, authenticated via `HF_TOKEN`) ➔ `VRAM Flush`.
2. **Cloud Pipelines (Deepgram & AssemblyAI):**
   * **Compute Cost:** Zero GPU compute (0 seconds consumed from ZeroGPU quotas).
   * **Stages:** Raw media is streamed or uploaded over HTTPS authenticated with `DEEPGRAM_API_KEY` or `ASSEMBLYAI_API_KEY`. Word/utterance timestamps and speaker labels are normalized into the standard `TranscriptionResult` object.

---

## System Requirements

* **OS:** Windows 10/11 or Linux (Ubuntu 20.04+).
* **Python:** 3.10 (recommended).
* **FFmpeg:** Installed and exported to system `PATH`.
* **Hardware GPU (Optional):** NVIDIA GeForce GTX 1650 or greater with CUDA 12.x support (CPU fallback fully supported).

---

## Installation & Setup

### Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/Klopezxd/whisperx-transcriptor.git
cd whisperx-transcriptor

# Create and activate isolated Python 3.10 environment
conda create -n transcriptor python=3.10 -y
conda activate transcriptor

# Install PyTorch with CUDA 12.x wheel
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install dependencies
pip install -r requirements.txt

# Verify CUDA acceleration
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available())"
```

---

## Hugging Face Authentication (Speaker Diarization)

Speaker diarization leverages gated models from Pyannote. To enable it:

1. Create a free account at [Hugging Face](https://huggingface.co/).
2. Accept model usage terms:
   * [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   * [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Generate a read-access token under [Hugging Face Settings](https://huggingface.co/settings/tokens).
4. Export the token:
   ```bash
   cp .env.example .env
   # Add your token to .env:
   HF_TOKEN=hf_your_token_here
   ```

---

## Execution Modes: Cloud vs. Local

| Execution Mode | Best For | Key Advantage | Constraints |
|---|---|---|---|
| **Hugging Face Spaces (Cloud)** | Short-to-medium files (< 20-30 mins), instant access | Zero install; runs on cloud NVIDIA A10G (24GB VRAM) | Subject to daily ZeroGPU community quotas |
| **Local Execution (PC)** | Long recordings (1 to 5+ hours), batch jobs, private/sensitive data | Unlimited processing time, 100% offline privacy | Uses local machine hardware (GPU/CPU) |

---

## Usage

### 1. Windows One-Click Launcher (Desktop)

Double-click `run_transcriptor.bat` in the project root (or run it via terminal). It automatically activates the `transcriptor` environment and opens the graphical file picker dialog.

---

### 2. Local Web Application (Gradio)

Run the full interactive web application locally on your machine:

```bash
python app.py
```
Navigate to `http://127.0.0.1:7860` in any modern web browser.

---

### 3. Command Line Interface (CLI)

```bash
# Interactive mode (opens file selection dialog if no -i argument is provided):
python cli.py

# Direct processing with turbo model:
python cli.py -i "lecture.mp4" -m large-v3-turbo

# Enable multi-speaker diarization:
python cli.py -i "interview.mp4" -d

# Custom output directory with range timestamps:
python cli.py -i "meeting.wav" --timestamps range -o "./transcripts"
```

#### Available CLI Arguments:

| Flag | Type | Description | Default |
|---|---|---|---|
| `-i`, `--input` | String | Path to media file (launches GUI picker if omitted). | `None` |
| `-m`, `--model` | Choice | `large-v3-turbo`, `large-v3`, `medium`, `small`, `base` | `large-v3-turbo` |
| `-d`, `--diarize` | Flag | Enables speaker diarization (requires token). | `False` |
| `--timestamps` | Choice | Timestamp formatting: `simple`, `range`, `none` | `simple` |
| `-t`, `--token` | String | Hugging Face token override. | `HF_TOKEN` env |
| `--compute-type` | Choice | Quantization: `int8` (recommended for 4GB), `float16`, `float32` | `int8` |
| `-o`, `--output-dir` | String | Output folder for `.txt` transcripts. | Source directory |

---

## Repository Structure

```text
whisperx-transcriptor/
├── .github/
│   └── workflows/
│       ├── sync-to-hf.yml       # Continuous Deployment to Hugging Face Spaces
│       └── lint.yml             # Code quality and linting via Ruff
├── src/
│   ├── __init__.py
│   ├── config.py                # Strongly typed dataclasses & enums
│   ├── transcriber.py           # Core decoupled WhisperX orchestration pipeline
│   └── utils.py                 # Pure helper functions, memory management, and security
├── app.py                       # Gradio Web UI entrypoint (Local + ZeroGPU)
├── cli.py                       # Advanced CLI entrypoint
├── requirements.txt             # Python dependencies
├── packages.txt                 # OS packages (FFmpeg for Hugging Face Spaces)
├── environment.yml              # Conda environment definition
├── .env.example                 # Credential template
├── .gitignore                   # Strict exclusions for artifacts and large media
├── pyproject.toml               # Modern packaging metadata & Ruff configuration
├── LICENSE                      # MIT License
├── README.md                    # English documentation (Primary)
└── README.es.md                 # Spanish documentation
```

---

## License & Authorship

* **Author:** [Klever López](https://github.com/Klopezxd)
* **License:** MIT License — Open source for academic, professional, and personal use.

