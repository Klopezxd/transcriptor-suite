"""Plantilla HTML5 nativa semántica para Transcriptor Suite (Aurora Boreal Nórdica)."""


def get_transcriptor_html() -> str:
    """
    Retorna la estructura HTML5 completa y semántica de la interfaz de Transcriptor Suite.
    Diseñada para desacoplar el DOM nativo de los componentes internos de Gradio,
    operando en conjunto con el controlador vanilla JS y el puente headless.
    """
    return """
<div class="transcriptor-app" id="transcriptor_app">
    <!-- Barra de Navegación Superior -->
    <header class="app-navbar" role="banner">
        <div class="nav-brand">
            <div class="brand-icon" aria-hidden="true">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" x2="12" y1="19" y2="22"/>
                </svg>
            </div>
            <div class="brand-titles">
                <h1>Transcriptor Suite</h1>
                <p>Suite Neuronal de Reconocimiento de Voz · WhisperX, Deepgram &amp; AssemblyAI</p>
            </div>
        </div>
        <div class="nav-actions">
            <button id="btn_open_login" type="button" class="btn-secondary btn-login-compact" aria-label="Abrir diálogo de Administrador">
                <span>🔐</span> Admin
            </button>
            <button id="btn_logout" type="button" class="btn-secondary btn-login-compact" style="display: none;" aria-label="Cerrar sesión de Administrador">
                <span>🔓</span> Salir
            </button>
        </div>
    </header>

    <!-- Cuerpo Principal de 2 Columnas -->
    <main class="main-body-row" role="main">
        <!-- Columna Izquierda: Entrada de Archivos, Opciones y Credenciales -->
        <section class="left-column" aria-label="Panel de configuración y carga">
            <!-- Módulo 1: Archivo Multimedia -->
            <article class="module-card upload-card" id="card_upload">
                <div class="module-header">
                    <div class="module-title">
                        <span class="module-title-icon" aria-hidden="true">📁</span>
                        <span>Archivo Multimedia</span>
                    </div>
                </div>
                <div id="drop_zone" class="file-upload-zone" tabindex="0" role="button" aria-label="Zona de arrastre y subida de archivos">
                    <input type="file" id="media_file_input" multiple accept="audio/*,video/*,.wav,.flac,.m4a,.mp3,.mp4,.mkv,.webm" class="visually-hidden" aria-hidden="true" />
                    <div class="upload-prompt" id="upload_prompt">
                        <div class="upload-icon" aria-hidden="true">
                            <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                <polyline points="17 8 12 3 7 8"/>
                                <line x1="12" x2="12" y1="3" y2="15"/>
                            </svg>
                        </div>
                        <p class="upload-text-main">Arrastra tus archivos aquí o haz clic para explorar</p>
                        <p class="upload-text-sub">Soporta múltiples pistas de audio o video simultáneas</p>
                    </div>
                    <div id="file_list_container" class="file-preview-holder" style="display: none;">
                        <table class="file-preview-table" id="file_preview_table">
                            <tbody id="file_preview_tbody"></tbody>
                        </table>
                    </div>
                </div>
                <div class="file-status-strip">
                    <div class="status-strip-content">
                        <span class="status-formats">Formatos: WAV, FLAC, M4A, MP3, MP4, MKV, WEBM</span>
                        <span class="status-dot">·</span>
                        <span class="status-hint">⚡ Audio sin pérdida (Lossless)</span>
                    </div>
                </div>
            </article>

            <!-- Módulo 2: Motor y Opciones -->
            <article class="module-card options-card" id="card_options">
                <div class="module-header">
                    <div class="module-title">
                        <span class="module-title-icon" aria-hidden="true">⚙️</span>
                        <span>Motor y Opciones</span>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group flex-1">
                        <label for="provider_select">Proveedor de Transcripción</label>
                        <select id="provider_select" class="form-select">
                            <option value="AssemblyAI (Cloud API)" selected>AssemblyAI (Cloud API)</option>
                            <option value="Deepgram (Cloud API)">Deepgram (Cloud API)</option>
                            <option value="WhisperX (ZeroGPU / Local)">WhisperX (ZeroGPU / Local)</option>
                        </select>
                    </div>
                    <div class="form-group flex-1">
                        <label for="model_select">Modelo de Transcripción</label>
                        <select id="model_select" class="form-select">
                            <option value="assemblyai-stt-3-7" selected>assemblyai-stt-3-7 · Nueva Generación (Recomendado)</option>
                            <option value="universal-2">universal-2 · Cobertura Extendida (99 idiomas)</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group flex-1 checkbox-group">
                        <label class="checkbox-label" for="diarize_checkbox">
                            <input type="checkbox" id="diarize_checkbox" />
                            <span class="checkbox-custom"></span>
                            <span class="checkbox-text">Identificar quién habla (Diarización)</span>
                        </label>
                    </div>
                    <div class="form-group flex-1">
                        <label for="timestamps_select">Formato de Marcas de Tiempo</label>
                        <select id="timestamps_select" class="form-select">
                            <option value="Simple ([HH:MM:SS])" selected>Simple ([HH:MM:SS])</option>
                            <option value="Rango ([Inicio - Fin])">Rango ([Inicio - Fin])</option>
                            <option value="Texto continuo (Sin marcas)">Texto continuo (Sin marcas)</option>
                        </select>
                    </div>
                </div>
                <div id="provider_info_box" class="provider-info-container">
                    <div class="info-notice">
                        <span>💡</span>
                        <span><b>AssemblyAI:</b> Precisión acústica y comprensión contextual de alta fidelidad. Modelo <code>assemblyai-stt-3-7</code> por defecto con respaldo multilingüe <code>universal-2</code>.</span>
                    </div>
                </div>
            </article>

            <!-- Módulo 3: Credenciales de API (Modo Invitado) -->
            <article id="guest_keys_panel" class="module-card keys-card">
                <div class="module-header">
                    <div class="module-title">
                        <span class="module-title-icon" aria-hidden="true">🔑</span>
                        <span>Credenciales de API</span>
                    </div>
                </div>
                <div id="whisperx_notice" class="info-notice info-free" style="display: none;">
                    <span>✨</span>
                    <span>WhisperX en GPU es 100% gratuito y no requiere ninguna clave de API externa.</span>
                </div>
                <div id="hf_token_group" class="form-group" style="display: none;">
                    <label for="hf_token_input">Token de Hugging Face (HF_TOKEN)</label>
                    <input type="password" id="hf_token_input" class="form-input" placeholder="hf_..." autocomplete="off" />
                    <span class="form-help">Requerido al activar Diarización con WhisperX. Rol 'Read' con términos de pyannote aceptados.</span>
                </div>
                <div id="deepgram_key_group" class="form-group" style="display: none;">
                    <label for="deepgram_key_input">Deepgram API Key</label>
                    <input type="password" id="deepgram_key_input" class="form-input" placeholder="Pega tu clave de API de Deepgram..." autocomplete="off" />
                    <span class="form-help">Obtén tu clave con créditos iniciales gratuitos en console.deepgram.com</span>
                </div>
                <div id="assemblyai_key_group" class="form-group">
                    <label for="assemblyai_key_input">AssemblyAI API Key</label>
                    <input type="password" id="assemblyai_key_input" class="form-input" placeholder="Pega tu clave de API de AssemblyAI..." autocomplete="off" />
                    <span class="form-help">Obtén tu clave gratuita en assemblyai.com</span>
                </div>
            </article>

            <!-- Botones de Acción Principales -->
            <div class="actions-row main-actions">
                <button id="btn_clear" type="button" class="btn-secondary btn-action-clear">
                    <span>🗑️</span> Limpiar
                </button>
                <button id="btn_transcribe" type="button" class="btn-primary btn-action-transcribe">
                    <span>⚡</span> Iniciar Transcripción
                </button>
            </div>
        </section>

        <!-- Columna Derecha: Visor de Resultados y Descargas -->
        <section class="result-column" aria-label="Panel de resultados y exportación">
            <article class="module-card result-card">
                <div class="module-header result-header">
                    <div class="module-title">
                        <span class="module-title-icon" aria-hidden="true">📝</span>
                        <span>Resultado de la Transcripción</span>
                    </div>
                    <div id="status_badge_container" class="status-badge-container">
                        <div class="status-badges-group">
                            <span class="status-badge status-warning">En espera</span>
                        </div>
                    </div>
                </div>
                <div class="textarea-container">
                    <textarea id="output_transcript" class="output-textarea" placeholder="El texto transcrito con locutores y marcas de tiempo aparecerá aquí..." readonly spellcheck="false"></textarea>
                </div>
                <div class="actions-row result-actions">
                    <button id="btn_download" type="button" class="btn-secondary" disabled>
                        <span>📥</span> Descargar Documento (.txt)
                    </button>
                    <button id="btn_download_consolidated" type="button" class="btn-secondary" style="display: none;" disabled>
                        <span>📄</span> Descargar Consolidado (.txt)
                    </button>
                    <button id="btn_copy" type="button" class="btn-secondary" disabled>
                        <span>📋</span> Copiar texto
                    </button>
                </div>
            </article>
        </section>
    </main>

    <!-- Pie de Página Institucional -->
    <footer class="app-footer" role="contentinfo">
        <p>Transcriptor Suite · Modelos neuronales ASR con alineación fonética fonema a fonema y diarización de hablantes.</p>
    </footer>

    <!-- Diálogo Modal Flotante de Administrador -->
    <div id="modal_login" class="modal-overlay hide" role="dialog" aria-modal="true" aria-labelledby="modal_login_title">
        <div class="modal-dialog">
            <div class="modal-title-row">
                <h3 id="modal_login_title" class="modal-title">Acceso de Administrador</h3>
                <button id="btn_close_login" type="button" class="btn-modal-close" aria-label="Cerrar diálogo">✕</button>
            </div>
            <p class="modal-subtext">Ingresa tu PIN de acceso para activar las credenciales maestras del servidor.</p>
            <div class="form-group modal-input-group">
                <input type="password" id="input_pin" class="form-input modal-pin-input" placeholder="Ingresa PIN o contraseña..." autocomplete="current-password" />
            </div>
            <div id="login_msg" class="login-msg" style="display: none;"></div>
            <div class="actions-row modal-actions">
                <button id="btn_cancel_login" type="button" class="btn-secondary">Cancelar</button>
                <button id="btn_submit_login" type="button" class="btn-primary">Ingresar</button>
            </div>
        </div>
    </div>
</div>
""".strip()
