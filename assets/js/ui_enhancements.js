() => {
        if (!window.__transcriptorUiEnhancements) {
            window.__transcriptorUiEnhancements = true;

            // 1. Toggle abrir/cerrar en los menús desplegables al hacer clic (1er clic abre, 2do clic cierra)
            const toggleCombo = (wrap, e) => {
                const input = wrap.querySelector('input[role="combobox"]');
                if (!input) return;
                const wrapInner = wrap.querySelector('.wrap-inner');
                const isOpen =
                    input.getAttribute('aria-expanded') === 'true' ||
                    (wrapInner && wrapInner.classList.contains('show_options')) ||
                    !!wrap.querySelector('.options');

                e.preventDefault();
                e.stopPropagation();
                if (window.getSelection) {
                    const sel = window.getSelection();
                    if (sel && sel.removeAllRanges) sel.removeAllRanges();
                }

                if (isOpen) {
                    input.blur();
                } else {
                    document.querySelectorAll('input[role="combobox"]').forEach((other) => {
                        if (other !== input) other.blur();
                    });
                    if (document.activeElement === input) {
                        input.blur();
                    }
                    input.focus({ preventScroll: true });
                }
            };

            document.addEventListener('mousedown', (e) => {
                if (e.button !== 0) return;

                const optionTarget = e.target.closest('.options, .option-list, [role="listbox"], [role="option"], li.item');
                if (optionTarget) {
                    setTimeout(() => {
                        document.querySelectorAll('input[role="combobox"]').forEach((inp) => inp.blur());
                        if (window.getSelection) {
                            const sel = window.getSelection();
                            if (sel && sel.removeAllRanges) sel.removeAllRanges();
                        }
                    }, 15);
                    return;
                }

                const wrapInner = e.target.closest('.wrap-inner:has(input[role="combobox"])');
                if (wrapInner && wrapInner.parentElement) {
                    window.__lastComboMousedown = Date.now();
                    toggleCombo(wrapInner.parentElement, e);
                    return;
                }

                document.querySelectorAll('input[role="combobox"]').forEach((inp) => {
                    if (
                        document.activeElement === inp ||
                        inp.getAttribute('aria-expanded') === 'true'
                    ) {
                        inp.blur();
                    }
                });
            }, true);

            document.addEventListener('click', (e) => {
                const optionTarget = e.target.closest('.options, .option-list, [role="listbox"], [role="option"], li.item');
                if (optionTarget) return;

                const wrapInner = e.target.closest('.wrap-inner:has(input[role="combobox"])');
                if (!wrapInner || !wrapInner.parentElement) return;

                if (Date.now() - (window.__lastComboMousedown || 0) < 350) {
                    e.preventDefault();
                    e.stopPropagation();
                    return;
                }
                toggleCombo(wrapInner.parentElement, e);
            }, true);

            // 2. Responsividad en tiempo real al cambiar entre monitores (ej. 24" <-> 16" laptop) sin recargar
            let lastParentVh = 0;
            const syncViewport = () => {
                const vh = window.visualViewport ? window.visualViewport.height : window.innerHeight;
                const vw = window.visualViewport ? window.visualViewport.width : window.innerWidth;
                const effectiveVh = lastParentVh > 0 ? lastParentVh : vh;

                if (effectiveVh > 0) {
                    document.documentElement.style.setProperty('--app-vh', `${Math.round(effectiveVh)}px`);
                }
                if (vw > 0) {
                    document.documentElement.style.setProperty('--app-vw', `${Math.round(vw)}px`);
                }

                // Limpiar alturas en línea que Gradio BaseTextbox inyecta al redimensionar
                document.querySelectorAll('.output-textbox textarea').forEach((ta) => {
                    if (ta.style.height) {
                        ta.style.removeProperty('height');
                    }
                });
            };

            const hookParentIframe = () => {
                if (window.parentIFrame && !window.__transcriptorHookedParentIframe) {
                    window.__transcriptorHookedParentIframe = true;
                    const origSize = window.parentIFrame.size ? window.parentIFrame.size.bind(window.parentIFrame) : null;
                    if (origSize) {
                        window.parentIFrame.size = (customH, customW) => {
                            if (lastParentVh > 0) {
                                return origSize(lastParentVh, customW);
                            }
                            return origSize(customH, customW);
                        };
                    }
                    if (typeof window.parentIFrame.getPageInfo === 'function') {
                        window.parentIFrame.getPageInfo((info) => {
                            if (!info || !info.clientHeight) return;
                            const topOffset = Math.max(0, info.offsetTop || 0);
                            const targetH = Math.max(460, info.clientHeight - topOffset);
                            if (Math.abs(targetH - lastParentVh) > 2) {
                                lastParentVh = targetH;
                                document.documentElement.style.setProperty('--app-vh', `${Math.round(targetH)}px`);
                                if (origSize) origSize(targetH);
                            }
                        });
                    }
                }
            };

            // ── MODO ADMIN SEGURO (ZERO RECURSIÓN) ──
            let currentIsAdmin = null;
            const updateAdminMode = () => {
                const keysCard = document.querySelector('.keys-card');
                const btnLogout = document.querySelector('.btn-login-compact:not([style*="display: none"])');
                const isAdmin = Boolean(
                    (keysCard && (keysCard.classList.contains('hide') || keysCard.style.display === 'none')) ||
                    (btnLogout && btnLogout.textContent && btnLogout.textContent.includes('Salir'))
                );
                if (isAdmin !== currentIsAdmin) {
                    currentIsAdmin = isAdmin;
                    document.body.classList.toggle('admin-mode', isAdmin);
                    document.documentElement.classList.toggle('admin-mode', isAdmin);
                    const leftCol = document.querySelector('.left-column');
                    if (leftCol) leftCol.classList.toggle('admin-mode', isAdmin);
                    syncViewport();
                }
            };

            // Limpieza de cualquier rastro previo de barra neón sintética
            document.querySelectorAll('.neon-scrollbar-track').forEach((el) => el.remove());

            // Asegurar que cualquier clic en el botón de subida manual active el selector si no fue interceptado
            document.addEventListener('click', (e) => {
                const uploadBtn = e.target.closest('.icon-button-wrapper > button:first-child, .file-upload-zone [aria-label*="Upload"]');
                if (uploadBtn && uploadBtn.closest('.upload-card, .file-upload-zone')) {
                    if (!e.target.matches('input[type="file"]') && !e.target.closest('.upload-container')) {
                        const fileInput = uploadBtn.querySelector('input[type="file"]') || document.querySelector('.file-upload-zone input[type="file"]');
                        if (fileInput) {
                            fileInput.click();
                        }
                    }
                }
            });

            syncViewport();
            hookParentIframe();
            updateAdminMode();
            setTimeout(() => { syncViewport(); hookParentIframe(); updateAdminMode(); }, 120);
            setTimeout(() => { syncViewport(); hookParentIframe(); updateAdminMode(); }, 500);

            let rafId = 0;
            const onViewportChange = () => {
                if (rafId) cancelAnimationFrame(rafId);
                rafId = requestAnimationFrame(() => {
                    syncViewport();
                    hookParentIframe();
                    updateAdminMode();
                });
            };

            window.addEventListener('resize', onViewportChange, { passive: true });
            if (window.visualViewport) {
                window.visualViewport.addEventListener('resize', onViewportChange, { passive: true });
            }
            if ('ResizeObserver' in window) {
                const ro = new ResizeObserver(onViewportChange);
                ro.observe(document.documentElement);
            }

            // Observer reactivo para la columna izquierda (modo admin)
            let colMoRaf = 0;
            const leftCol = document.querySelector('.left-column');
            if (leftCol && 'MutationObserver' in window) {
                const colMo = new MutationObserver(() => {
                    if (colMoRaf) return;
                    colMoRaf = requestAnimationFrame(() => {
                        colMoRaf = 0;
                        updateAdminMode();
                    });
                });
                colMo.observe(leftCol, {
                    childList: true,
                    subtree: true,
                    attributes: true,
                    attributeFilter: ['style', 'class', 'hidden']
                });
            }

            // Escuchar clics en botones de login/logout/modal para actualizar modo admin inmediatamente
            document.addEventListener('click', (e) => {
                const btn = e.target && e.target.closest ? e.target.closest('.btn-login-compact, #btn_submit_login, .btn-login-submit, .btn-modal-close, .btn-secondary') : null;
                if (btn) {
                    setTimeout(updateAdminMode, 150);
                    setTimeout(updateAdminMode, 500);
                    setTimeout(updateAdminMode, 1000);
                }
            }, true);

            // Escuchar tecla Enter en el campo de PIN del modal de login
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && e.target && e.target.matches('input[type="password"]')) {
                    setTimeout(updateAdminMode, 200);
                    setTimeout(updateAdminMode, 600);
                    setTimeout(updateAdminMode, 1200);
                }
            }, true);

            // Detectar cambio de escala DPI al arrastrar la ventana entre monitor externo (100%) y laptop (125%/150%)
            const watchDpiChange = () => {
                const mq = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
                const handler = () => {
                    onViewportChange();
                    setTimeout(onViewportChange, 60);
                    setTimeout(onViewportChange, 180);
                    watchDpiChange();
                };
                if (mq.addEventListener) {
                    mq.addEventListener('change', handler, { once: true });
                }
            };
            watchDpiChange();

            // 3. Extracción y Compresión de Audio en el Navegador y Soporte Drag & Drop Continuo
            window.__downloadAudioExtractorBat = () => {
                const batContent = `@echo off\r\nsetlocal enabledelayedexpansion\r\ntitle Extractor de Audio - Transcriptor Suite\r\ncolor 0B\r\nchcp 65001 >nul\r\n\r\necho ======================================================\r\necho       EXTRACTOR DE AUDIO - TRANSCRIPTOR SUITE\r\necho ======================================================\r\necho.\r\n\r\n:: 1. Buscar FFmpeg en el sistema\r\nset "FFMPEG="\r\nwhere ffmpeg >nul 2>&1 && set "FFMPEG=ffmpeg"\r\n\r\nif "!FFMPEG!"=="" (\r\n    if exist "%LOCALAPPDATA%\\Microsoft\\WinGet\\Links\\ffmpeg.exe" (\r\n        set "FFMPEG=%LOCALAPPDATA%\\Microsoft\\WinGet\\Links\\ffmpeg.exe"\r\n    ) else if exist "%~dp0ffmpeg.exe" (\r\n        set "FFMPEG=%~dp0ffmpeg.exe"\r\n    ) else if exist "C:\\ffmpeg\\bin\\ffmpeg.exe" (\r\n        set "FFMPEG=C:\\ffmpeg\\bin\\ffmpeg.exe"\r\n    )\r\n)\r\n\r\nif "!FFMPEG!"=="" (\r\n    color 0C\r\n    echo [ERROR] No se encontro FFmpeg en tu computadora.\r\n    echo.\r\n    echo Para instalarlo en 1 minuto, abre PowerShell y escribe:\r\n    echo     winget install Gyan.FFmpeg\r\n    echo.\r\n    echo O descarga ffmpeg.exe y colocalo en esta misma carpeta.\r\n    echo ======================================================\r\n    pause\r\n    exit /b 1\r\n)\r\n\r\n:: 2. Si no arrastro archivos, pedir ruta por teclado\r\nif "%~1"=="" (\r\n    echo Arrastra un archivo de video a esta ventana o escribe su ruta:\r\n    set /p "INPUT_FILE=> "\r\n    set "INPUT_FILE=!INPUT_FILE:"=!"\r\n    if "!INPUT_FILE!"=="" exit /b 0\r\n    call :PROCESAR "!INPUT_FILE!"\r\n    goto :FIN\r\n)\r\n\r\n:: 3. Procesar todos los archivos arrastrados (uno o varios a la vez)\r\n:LOOP\r\nif "%~1"=="" goto :FIN\r\ncall :PROCESAR "%~1"\r\nshift\r\ngoto :LOOP\r\n\r\n:PROCESAR\r\nset "ORIGINAL=%~1"\r\nif not exist "!ORIGINAL!" (\r\n    echo [ERROR] No se encuentra el archivo: "!ORIGINAL!"\r\n    echo.\r\n    exit /b 1\r\n)\r\n\r\nset "DESTINO=%~dpn1_audio.m4a"\r\n\r\necho ------------------------------------------------------\r\necho Procesando: "%~nx1"\r\necho Extrayendo audio en calidad original (Stream Copy)...\r\n\r\n"!FFMPEG!" -y -hide_banner -loglevel warning -stats -i "!ORIGINAL!" -vn -c:a copy "!DESTINO!"\r\n\r\nif !ERRORLEVEL! neq 0 (\r\n    echo [AVISO] Formato no compatible con copia directa. Codificando a AAC...\r\n    "!FFMPEG!" -y -hide_banner -loglevel warning -stats -i "!ORIGINAL!" -vn -c:a aac -b:a 64k -ac 1 "!DESTINO!"\r\n)\r\n\r\nif !ERRORLEVEL! equ 0 (\r\n    echo.\r\n    echo [EXITO] Audio listo para subir:\r\n    echo         "!DESTINO!"\r\n) else (\r\n    echo.\r\n    echo [ERROR] Ocurrio un fallo al procesar "%~nx1".\r\n)\r\necho ------------------------------------------------------\r\necho.\r\nexit /b 0\r\n\r\n:FIN\r\necho ======================================================\r\necho Proceso finalizado.\r\necho Puedes subir los audios generados a Transcriptor Suite.\r\necho ======================================================\r\necho.\r\npause\r\n`;
                const blob = new Blob([batContent], { type: 'application/x-bat' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'extraer_audio.bat';
                document.body.appendChild(a);
                a.click();
                setTimeout(() => {
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }, 100);
            };

            // ── ESTADO GLOBAL DE EXTRACCIÓN, SUBIDA Y COLA DE TRANSCRIPCIÓN ──
            let isExtractingAudio = false;
            let isUploadingFiles = false;
            let activeUploadCount = 0;
            let isTranscriptionQueued = false;
            let isAutoTriggeringTranscription = false;
            let queueTriggerTimer = null;
            let abortCurrentOptimization = false;

            window.__abortOptimization = () => {
                abortCurrentOptimization = true;
                console.log('[Transcriptor] Omitiendo optimización a petición del usuario. Pasando a subida directa...');
                const overlay = document.querySelector('.upload-loading-overlay');
                if (overlay) {
                    const txt = overlay.querySelector('.upload-loading-subtext');
                    if (txt) txt.textContent = '⚡ Omitiendo optimización... Subiendo original';
                }
            };

            const getTranscribeButton = () => {
                let btn = document.querySelector('#btn_transcribe') ||
                          document.querySelector('.actions-row .btn-primary') ||
                          document.querySelector('.btn-primary');
                if (btn) return btn;
                const buttons = document.querySelectorAll('button');
                for (let i = 0; i < buttons.length; i++) {
                    if (/transcri/i.test(buttons[i].textContent || '')) {
                        return buttons[i];
                    }
                }
                return null;
            };

            const updateQueueButtonVisual = () => {
                const btn = getTranscribeButton();
                if (!btn) return;
                if (isTranscriptionQueued) {
                    btn.classList.add('btn-transcribe-queued');
                    if (!btn.dataset.originalText) {
                        btn.dataset.originalText = btn.textContent || '⚡ Iniciar Transcripción';
                    }
                    btn.textContent = '⏳ En cola (iniciará al terminar subida)';
                    btn.title = 'Haz clic para cancelar la ejecución en cola';

                    // Actualizar badges en overlays abiertos
                    document.querySelectorAll('.upload-loading-overlay .upload-loading-info').forEach((info) => {
                        if (!info.querySelector('.upload-loading-queue-badge')) {
                            const badge = document.createElement('div');
                            badge.className = 'upload-loading-queue-badge';
                            badge.textContent = '⏳ Transcripción en cola: iniciará automáticamente al completar';
                            info.appendChild(badge);
                        }
                    });
                } else {
                    btn.classList.remove('btn-transcribe-queued');
                    if (btn.dataset.originalText) {
                        btn.textContent = btn.dataset.originalText;
                        delete btn.dataset.originalText;
                    }
                    btn.title = '';
                    document.querySelectorAll('.upload-loading-queue-badge').forEach((el) => el.remove());
                }
            };

            const checkAndExecuteQueue = () => {
                if (!isTranscriptionQueued) return;
                if (isExtractingAudio || isUploadingFiles || activeUploadCount > 0) return;

                if (queueTriggerTimer) clearTimeout(queueTriggerTimer);
                queueTriggerTimer = setTimeout(() => {
                    if (!isTranscriptionQueued) return;
                    if (isExtractingAudio || isUploadingFiles || activeUploadCount > 0) return;

                    const btn = getTranscribeButton();
                    if (!btn) return;

                    console.log('[Transcriptor] Ejecutando automáticamente transcripción en cola.');
                    isTranscriptionQueued = false;
                    updateQueueButtonVisual();

                    const strip = document.querySelector('.file-status-strip');
                    if (strip) {
                        strip.innerHTML = `
                            <div class="status-strip-content is-optimized">
                                <span class="status-pill success">🚀 Cola ejecutada</span>
                                <span class="status-text">Iniciando transcripción de archivos...</span>
                            </div>
                        `;
                    }

                    isAutoTriggeringTranscription = true;
                    btn.click();
                    setTimeout(() => {
                        isAutoTriggeringTranscription = false;
                    }, 800);
                }, 350);
            };

            const isAudioFile = (file) => {
                if (!file) return false;
                const name = (file.name || '').toLowerCase();
                const audioExts = ['.wav', '.flac', '.mp3', '.m4a', '.ogg', '.opus', '.aac', '.wma', '.aiff', '.alac'];
                return audioExts.some(ext => name.endsWith(ext)) || (file.type && file.type.startsWith('audio/'));
            };

            const isVideoFile = (file) => {
                if (!file) return false;
                if (isAudioFile(file)) return false;
                const name = (file.name || '').toLowerCase();
                const videoExts = ['.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.flv', '.wmv'];
                return videoExts.some(ext => name.endsWith(ext)) || (file.type && file.type.startsWith('video/'));
            };

            // Contenedores decodificables nativamente por la Web Audio API de los navegadores (Chromium/Firefox/Safari).
            // MKV, AVI, FLV, WMV NO son decodificables en Web Audio decodeAudioData y deben subirse directamente.
            const isBrowserDecodableVideo = (file) => {
                if (!file) return false;
                const name = (file.name || '').toLowerCase();
                const decodableExts = ['.mp4', '.mov', '.m4v', '.webm'];
                return decodableExts.some(ext => name.endsWith(ext));
            };

            const formatBytes = (bytes) => {
                if (!bytes || bytes <= 0) return '0 MB';
                const mb = bytes / (1024 * 1024);
                if (mb < 0.1) return `${Math.round(bytes / 1024)} KB`;
                return `${mb.toFixed(1)} MB`;
            };

            // Sondeo ultrarrápido (<30ms) de viabilidad de optimización mediante metadatos HTML5
            const probeVideoOptimizationFeasibility = (file) => new Promise((resolve) => {
                if (!isBrowserDecodableVideo(file)) {
                    resolve({
                        canOptimize: false,
                        reason: 'Contenedor MKV/AVI/FLV no decodificable en navegador; subida directa para FFmpeg nativo'
                    });
                    return;
                }

                // Límite seguro para decodificación Web Audio en el navegador (450 MB)
                const MAX_BROWSER_DECODE_SIZE = 450 * 1024 * 1024;
                if (file.size > MAX_BROWSER_DECODE_SIZE) {
                    resolve({
                        canOptimize: false,
                        largeVideo: true,
                        reason: `Video extenso (${formatBytes(file.size)} > 450MB); subida directa para FFmpeg nativo`
                    });
                    return;
                }

                const v = document.createElement('video');
                v.preload = 'metadata';
                let isDone = false;

                const timer = setTimeout(() => {
                    if (!isDone) {
                        isDone = true;
                        cleanup();
                        resolve({ canOptimize: false, reason: 'Tiempo de sondeo de metadatos superado (subida directa)' });
                    }
                }, 600);

                const cleanup = () => {
                    clearTimeout(timer);
                    v.onloadedmetadata = null;
                    v.onerror = null;
                    if (v.src) URL.revokeObjectURL(v.src);
                    v.remove();
                };

                v.onloadedmetadata = () => {
                    if (isDone) return;
                    isDone = true;
                    const duration = v.duration;
                    cleanup();

                    if (!duration || duration <= 0 || !isFinite(duration)) {
                        resolve({ canOptimize: false, reason: 'Duración no detectable en metadatos' });
                        return;
                    }

                    // Tamaño exacto de audio PCM lineal WAV 16-bit 16kHz Mono:
                    // 16,000 muestras/seg * 2 bytes/muestra = 32,000 bytes/seg + 44 bytes de cabecera RIFF
                    const expectedWavBytes = Math.ceil(duration * 32000) + 44;

                    // Si el archivo original de video ya es menor o similar al WAV esperado (<= expectedWavBytes * 1.1):
                    // Significa que el video ya tiene bitrate ultra liviano o es más pequeño que el audio WAV sin pérdida.
                    // Convertirlo a WAV aumentaría el tamaño o no ahorraría nada.
                    if (file.size <= expectedWavBytes * 1.1) {
                        resolve({
                            canOptimize: false,
                            duration,
                            expectedWavBytes,
                            reason: `Video ya optimizado (${formatBytes(file.size)} <= WAV esperado ${formatBytes(expectedWavBytes)})`
                        });
                        return;
                    }

                    resolve({
                        canOptimize: true,
                        duration,
                        expectedWavBytes
                    });
                };

                v.onerror = () => {
                    if (isDone) return;
                    isDone = true;
                    cleanup();
                    resolve({ canOptimize: false, reason: 'Codec no reproducible en navegador' });
                };

                try {
                    v.src = URL.createObjectURL(file);
                } catch (e) {
                    cleanup();
                    resolve({ canOptimize: false, reason: 'No se pudo crear URL de objeto' });
                }
            });

            // Generador de audio sin pérdida WAV PCM 16-bit lineal (Máxima Fidelidad Acústica)
            const audioBufferToWav = (buffer) => {
                const numChannels = 1;
                const sampleRate = buffer.sampleRate;
                const bitDepth = 16;
                const bytesPerSample = bitDepth / 8;
                const blockAlign = numChannels * bytesPerSample;
                const dataSize = buffer.length * blockAlign;
                const headerSize = 44;
                const totalSize = headerSize + dataSize;
                const arrayBuffer = new ArrayBuffer(totalSize);
                const view = new DataView(arrayBuffer);

                const writeString = (offset, str) => {
                    for (let i = 0; i < str.length; i++) {
                        view.setUint8(offset + i, str.charCodeAt(i));
                    }
                };

                writeString(0, 'RIFF');
                view.setUint32(4, 36 + dataSize, true);
                writeString(8, 'WAVE');
                writeString(12, 'fmt ');
                view.setUint32(16, 16, true);
                view.setUint16(20, 1, true); // PCM format
                view.setUint16(22, numChannels, true);
                view.setUint32(24, sampleRate, true);
                view.setUint32(28, sampleRate * blockAlign, true);
                view.setUint16(32, blockAlign, true);
                view.setUint16(34, bitDepth, true);
                writeString(36, 'data');
                view.setUint32(40, dataSize, true);

                const samples = new Int16Array(arrayBuffer, 44, buffer.length);
                const ch0 = buffer.getChannelData(0);
                for (let i = 0; i < buffer.length; i++) {
                    const s = Math.max(-1, Math.min(1, ch0[i]));
                    samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                return new Blob([arrayBuffer], { type: 'audio/wav' });
            };

            const extractAudioFromVideo = async (file, onProgress) => {
                if (onProgress) onProgress('Leyendo video en memoria...');
                const ab = await file.arrayBuffer();
                const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
                if (!AudioCtxClass) {
                    throw new Error('Web Audio no soportado');
                }
                const ctx = new AudioCtxClass();
                let decoded;
                try {
                    if (onProgress) onProgress('Decodificando pista de audio sin pérdida...');
                    decoded = await ctx.decodeAudioData(ab);
                } finally {
                    if (ctx.close) ctx.close().catch(() => {});
                }

                // Resamplear a 16kHz mono (estándar de oro universal sin pérdida acústica para WhisperX, Deepgram y AssemblyAI)
                if (onProgress) onProgress('Generando audio lineal WAV PCM 16 kHz (Máxima Fidelidad)...');
                const targetRate = 16000;
                const targetLength = Math.max(1, Math.ceil(decoded.duration * targetRate));
                const offlineCtx = new OfflineAudioContext(1, targetLength, targetRate);
                const source = offlineCtx.createBufferSource();
                source.buffer = decoded;
                source.connect(offlineCtx.destination);
                source.start(0);
                const resampled = await offlineCtx.startRendering();

                // Generar archivo WAV PCM de 16 bits sin pérdida adicional
                const audioBlob = audioBufferToWav(resampled);
                const baseName = file.name.substring(0, file.name.lastIndexOf('.')) || file.name;
                return new File([audioBlob], `${baseName}.wav`, {
                    type: 'audio/wav',
                    lastModified: Date.now()
                });
            };

            const defaultStripContent = `
                <div class="status-strip-content">
                    <span class="status-formats">Formatos: WAV, FLAC, M4A, MP3, MP4, MKV, WEBM</span>
                    <span class="status-dot">·</span>
                    <span class="status-hint">⚡ Audio sin pérdida (Lossless)</span>
                </div>
            `;

            const updateFileStatusStrip = (zone, stats) => {
                const strip = document.querySelector('.file-status-strip');
                if (!strip) return;

                if (!stats || !stats.processed) {
                    strip.innerHTML = defaultStripContent;
                    return;
                }

                if (stats.success) {
                    strip.innerHTML = `
                        <div class="status-strip-content is-optimized">
                            <span class="status-pill success">⚡ WAV 16 kHz (Sin pérdida)</span>
                            <span class="status-text">${stats.origMb} ➔ <b>${stats.optMb}</b> (${stats.savingsPct}% menos subida)</span>
                        </div>
                    `;
                } else if (stats.largeVideo) {
                    strip.innerHTML = `
                        <div class="status-strip-content is-server">
                            <span class="status-pill server">⚡ Video extenso (${stats.origMb})</span>
                            <span class="status-text">La IA solo necesita el audio · Sube más rápido con <a href="javascript:void(0)" onclick="window.__downloadAudioExtractorBat()" class="status-strip-link" title="Haz clic para descargar extraer_audio.bat directamente">extraer_audio.bat ⬇</a></span>
                        </div>
                    `;
                } else if (stats.directUpload) {
                    strip.innerHTML = `
                        <div class="status-strip-content is-server">
                            <span class="status-pill server">⚡ Subida directa (${stats.origMb})</span>
                            <span class="status-text">${stats.count} archivo(s) listos · Extracción nativa en servidor</span>
                        </div>
                    `;
                } else {
                    strip.innerHTML = `
                        <div class="status-strip-content is-server">
                            <span class="status-pill server">⚡ Subida directa (${stats.origMb})</span>
                            <span class="status-text">Extracción de audio en servidor</span>
                        </div>
                    `;
                }
            };

            const renderOptimizationBadge = (zone, stats) => {
                updateFileStatusStrip(zone, stats);
            };

            const getUploadZone = (zone) => {
                if (zone && zone.classList && zone.classList.contains('file-upload-zone')) return zone;
                return document.querySelector('.file-upload-zone') || zone;
            };

            const updateExtractingOverlay = (zone, current, total, fileName, stageText) => {
                const targetZone = getUploadZone(zone);
                if (!targetZone) return;
                let overlay = targetZone.querySelector('.upload-loading-overlay');
                if (!overlay) {
                    overlay = document.createElement('div');
                    overlay.className = 'upload-loading-overlay';
                    overlay.addEventListener('click', (e) => e.stopPropagation());
                    targetZone.appendChild(overlay);
                }
                const queueBadgeHtml = isTranscriptionQueued ?
                    '<div class="upload-loading-queue-badge">⏳ Transcripción en cola: iniciará automáticamente al completar</div>' : '';
                overlay.innerHTML = `
                    <button class="overlay-dismiss-btn" title="Omitir optimización y subir archivo original" onclick="window.__abortOptimization && window.__abortOptimization()">✕</button>
                    <div class="upload-overlay-body">
                        <div class="upload-spinner"></div>
                        <div class="upload-loading-info">
                            <div class="upload-loading-text">⚡ Optimizando video (${current}/${total})...</div>
                            <div class="upload-loading-subtext">${stageText || 'Extrayendo audio localmente'} · ${fileName}</div>
                            ${queueBadgeHtml}
                            <button type="button" class="btn-skip-optimization" onclick="window.__abortOptimization && window.__abortOptimization()">⚡ Omitir y subir directo</button>
                        </div>
                    </div>
                `;
            };

            const showUploadOverlay = (zone, stats, title, sub) => {
                const targetZone = getUploadZone(zone);
                if (!targetZone) return;
                let overlay = targetZone.querySelector('.upload-loading-overlay');
                if (!overlay) {
                    overlay = document.createElement('div');
                    overlay.className = 'upload-loading-overlay';
                    overlay.addEventListener('click', (e) => e.stopPropagation());
                    targetZone.appendChild(overlay);
                }

                let defaultTitle = '⬆ Subiendo archivos al servidor...';
                let defaultSub = 'Transfiriendo a Hugging Face Space';

                if (stats && stats.success) {
                    defaultTitle = `⚡ ¡Audio sin pérdida extraído! ${stats.origMb} ➔ ${stats.optMb} (-${stats.savingsPct}%)`;
                    defaultSub = `⬆ Subiendo WAV PCM 16 kHz · Máxima fidelidad acústica`;
                } else if (stats && stats.largeVideo) {
                    defaultTitle = `⬆ Subiendo video (${stats.origMb})...`;
                    defaultSub = `La IA solo necesita el audio · Sube más rápido con extraer_audio.bat`;
                } else if (stats && stats.directUpload) {
                    defaultTitle = `⬆ Subiendo ${stats.count} archivo(s) (${stats.origMb})...`;
                    defaultSub = `Subida directa · Extracción nativa con FFmpeg en servidor`;
                } else if (stats && stats.mb) {
                    defaultTitle = stats.anyAudio ? '⬆ Subiendo audio al servidor...' : `Subiendo ${stats.count} archivo(s)...`;
                    defaultSub = `${stats.mb} · Transfiriendo a Hugging Face Space`;
                }

                const queueBadgeHtml = isTranscriptionQueued ?
                    '<div class="upload-loading-queue-badge">⏳ Transcripción en cola: iniciará automáticamente al completar</div>' : '';

                overlay.innerHTML = `
                    <button class="overlay-dismiss-btn" title="Cerrar overlay" onclick="this.closest('.upload-loading-overlay').remove()">✕</button>
                    <div class="upload-overlay-body">
                        <div class="upload-spinner"></div>
                        <div class="upload-loading-info">
                            <div class="upload-loading-text">${title || defaultTitle}</div>
                            <div class="upload-loading-subtext">${sub || defaultSub}</div>
                            ${queueBadgeHtml}
                        </div>
                    </div>
                `;
            };

            const hideUploadOverlay = () => {
                document.querySelectorAll('.upload-loading-overlay').forEach((el) => el.remove());
            };

            // Observador DOM en tiempo real: cierra el overlay en cuanto Gradio monte los archivos en pantalla
            const observeUploadZone = () => {
                const zone = document.querySelector('.file-upload-zone, .upload-card');
                if (!zone || zone.__transcriptorObserverAttached) return;
                zone.__transcriptorObserverAttached = true;
                const mo = new MutationObserver(() => {
                    const hasPreview = zone.querySelector('.file-preview-holder, table.file-preview, [data-testid="file-preview"], .icon-button-wrapper, .filename, table');
                    if (hasPreview) {
                        hideUploadOverlay();
                        if (!isExtractingAudio && !isUploadingFiles) {
                            checkAndExecuteQueue();
                        }
                    }
                });
                mo.observe(zone, { childList: true, subtree: true });
            };
            observeUploadZone();
            setInterval(observeUploadZone, 1000);

            const handleFilesUploadStart = (zone, files, optStats) => {
                if (!files || files.length === 0) return;
                isUploadingFiles = true;
                let totalBytes = 0;
                let anyAudio = false;
                for (let i = 0; i < files.length; i++) {
                    totalBytes += files[i].size || 0;
                    if (isAudioFile(files[i])) {
                        anyAudio = true;
                    }
                }
                const mb = formatBytes(totalBytes);
                if (optStats && optStats.processed) {
                    showUploadOverlay(zone, optStats);
                } else {
                    const stats = { count: files.length, mb, anyAudio };
                    showUploadOverlay(zone, stats);
                }
                setTimeout(() => {
                    hideUploadOverlay();
                    isUploadingFiles = false;
                    checkAndExecuteQueue();
                }, 45000);
            };

            const processFilesWithAudioExtraction = async (zone, files) => {
                isExtractingAudio = true;
                abortCurrentOptimization = false;
                try {
                    const fileArray = Array.from(files || []);
                    if (fileArray.length === 0) {
                        return { files: fileArray, stats: null };
                    }

                    // 1. Sondeo rápido de viabilidad (milisegundos): detecta de inmediato archivos no optimizables
                    updateExtractingOverlay(zone, 1, fileArray.length, 'Analizando archivos...', 'Detectando formato y tamaño...');

                    const feasibilityList = [];
                    for (let i = 0; i < fileArray.length; i++) {
                        const f = fileArray[i];
                        if (isAudioFile(f)) {
                            feasibilityList.push({ file: f, canOptimize: false, isAudio: true });
                        } else if (!isVideoFile(f)) {
                            feasibilityList.push({ file: f, canOptimize: false });
                        } else {
                            const feas = await probeVideoOptimizationFeasibility(f);
                            feasibilityList.push({ file: f, ...feas });
                        }
                    }

                    const optimizableItems = feasibilityList.filter(item => item.canOptimize);

                    // Si ningún archivo se puede optimizar en el navegador (por ser MKV, audio directo o video ya liviano),
                    // pasar de inmediato a la subida directa sin demoras artificiales
                    if (optimizableItems.length === 0) {
                        hideUploadOverlay();
                        const totalBytes = fileArray.reduce((acc, f) => acc + f.size, 0);
                        const anyLarge = feasibilityList.some(item => item.largeVideo);
                        const stats = {
                            processed: true,
                            success: false,
                            largeVideo: anyLarge,
                            directUpload: true,
                            count: fileArray.length,
                            origMb: formatBytes(totalBytes),
                            reason: 'Archivos aptos para subida directa (extracción optimizada en servidor)'
                        };
                        return { files: fileArray, stats };
                    }

                    // 2. Extracción de audio únicamente para los archivos que realmente se benefician
                    const processed = [];
                    let currentOptimizedIdx = 0;
                    let totalOrigBytes = 0;
                    let totalOptBytes = 0;
                    let anyOptimized = false;
                    let anyLargeVideo = false;

                    for (let i = 0; i < feasibilityList.length; i++) {
                        const item = feasibilityList[i];
                        const f = item.file;
                        totalOrigBytes += f.size;
                        const origMb = formatBytes(f.size);

                        if (abortCurrentOptimization) {
                            console.log(`[Transcriptor] Optimización omitida para ${f.name}. Subiendo original.`);
                            totalOptBytes += f.size;
                            processed.push(f);
                            continue;
                        }

                        if (!item.canOptimize) {
                            if (item.largeVideo) anyLargeVideo = true;
                            totalOptBytes += f.size;
                            console.log(`[Transcriptor] Subida directa sin extracción local: ${f.name} (${origMb}). Motivo: ${item.reason || 'No optimizable en cliente'}`);
                            processed.push(f);
                            continue;
                        }

                        currentOptimizedIdx++;
                        updateExtractingOverlay(
                            zone,
                            currentOptimizedIdx,
                            optimizableItems.length,
                            `${f.name} (${origMb})`,
                            'Decodificando pista de audio sin pérdida...'
                        );

                        try {
                            const audioFile = await extractAudioFromVideo(f, (stageText) => {
                                updateExtractingOverlay(zone, currentOptimizedIdx, optimizableItems.length, `${f.name} (${origMb})`, stageText);
                            });

                            const audioMb = formatBytes(audioFile.size);
                            if (audioFile.size < f.size) {
                                const savedBytes = f.size - audioFile.size;
                                const savings = Math.max(0, Math.round((savedBytes / f.size) * 100));
                                totalOptBytes += audioFile.size;
                                anyOptimized = true;
                                console.log(`[Transcriptor] Optimización exitosa: ${f.name} (${origMb}) -> ${audioFile.name} (${audioMb}, -${savings}%)`);
                                processed.push(audioFile);
                            } else {
                                totalOptBytes += f.size;
                                console.log(`[Transcriptor] El archivo original (${origMb}) ya era más ligero que el audio. Conservando original.`);
                                processed.push(f);
                            }
                        } catch (err) {
                            console.warn(`[Transcriptor] No se pudo extraer audio local de ${f.name}, subiendo original:`, err);
                            totalOptBytes += f.size;
                            processed.push(f);
                        }
                    }

                    const savedBytes = Math.max(0, totalOrigBytes - totalOptBytes);
                    const savingsPct = totalOrigBytes > 0 ? Math.round((savedBytes / totalOrigBytes) * 100) : 0;

                    const stats = {
                        processed: true,
                        success: anyOptimized && savingsPct > 0,
                        largeVideo: anyLargeVideo && !anyOptimized,
                        origTotal: totalOrigBytes,
                        optTotal: totalOptBytes,
                        savedBytes: savedBytes,
                        savingsPct: savingsPct,
                        origMb: formatBytes(totalOrigBytes),
                        optMb: formatBytes(totalOptBytes),
                        savedMb: formatBytes(savedBytes),
                    };

                    return { files: processed, stats };
                } finally {
                    isExtractingAudio = false;
                }
            };

            // Multi-Network Hook (Fetch y XHR) para ocultar el overlay inmediatamente al finalizar el upload
            if (!window.__transcriptorHookedFetch && window.fetch) {
                window.__transcriptorHookedFetch = true;
                const origFetch = window.fetch;
                window.fetch = async function(...args) {
                    const url = typeof args[0] === 'string' ? args[0] : (args[0] && args[0].url ? args[0].url : '');
                    const isUpload = url.includes('/upload') || (args[1] && (args[1].body instanceof FormData || args[1].body instanceof Blob || args[1].body instanceof File));
                    if (isUpload) {
                        activeUploadCount++;
                        isUploadingFiles = true;
                    }
                    try {
                        const res = await origFetch.apply(this, args);
                        if (isUpload) {
                            setTimeout(() => {
                                activeUploadCount = Math.max(0, activeUploadCount - 1);
                                if (activeUploadCount === 0) {
                                    isUploadingFiles = false;
                                    hideUploadOverlay();
                                    checkAndExecuteQueue();
                                }
                            }, 300);
                        }
                        return res;
                    } catch (err) {
                        if (isUpload) {
                            activeUploadCount = Math.max(0, activeUploadCount - 1);
                            if (activeUploadCount === 0) isUploadingFiles = false;
                            hideUploadOverlay();
                        }
                        throw err;
                    }
                };
            }

            if (!window.__transcriptorHookedXHR && window.XMLHttpRequest) {
                window.__transcriptorHookedXHR = true;
                const origOpen = XMLHttpRequest.prototype.open;
                const origSend = XMLHttpRequest.prototype.send;
                XMLHttpRequest.prototype.open = function(method, url, ...rest) {
                    this.__transcriptorUrl = String(url || '');
                    return origOpen.call(this, method, url, ...rest);
                };
                XMLHttpRequest.prototype.send = function(...args) {
                    const url = this.__transcriptorUrl || '';
                    const isUpload = url.includes('/upload') || (args[0] && (args[0] instanceof FormData || args[0] instanceof Blob || args[0] instanceof File));
                    if (isUpload) {
                        activeUploadCount++;
                        isUploadingFiles = true;
                        this.addEventListener('loadend', () => {
                            setTimeout(() => {
                                activeUploadCount = Math.max(0, activeUploadCount - 1);
                                if (activeUploadCount === 0) {
                                    isUploadingFiles = false;
                                    hideUploadOverlay();
                                    checkAndExecuteQueue();
                                }
                            }, 300);
                        });
                    }
                    return origSend.apply(this, args);
                };
            }

            // Interceptar clic en "Iniciar Transcripción" mientras se procesan o suben archivos para poner en cola
            document.addEventListener('click', (e) => {
                let btn = e.target.closest('#btn_transcribe, .actions-row .btn-primary, .btn-primary');
                if (!btn) {
                    const candidate = e.target.closest('button');
                    if (candidate && /transcri/i.test(candidate.textContent || '')) {
                        btn = candidate;
                    }
                }
                if (!btn) return;

                if (isAutoTriggeringTranscription) {
                    // Clic automático generado por la cola al estar todo listo: permitir que fluya a Gradio
                    return;
                }

                if (isExtractingAudio || isUploadingFiles || activeUploadCount > 0) {
                    e.preventDefault();
                    e.stopImmediatePropagation();

                    if (isTranscriptionQueued) {
                        isTranscriptionQueued = false;
                        updateQueueButtonVisual();
                        console.log('[Transcriptor] Cola de transcripción cancelada por el usuario.');
                    } else {
                        isTranscriptionQueued = true;
                        updateQueueButtonVisual();
                        console.log('[Transcriptor] Transcripción agregada a la cola. Iniciará al completar la subida.');
                    }
                    return;
                }
            }, true);

            // Detectar selección de archivos vía botón [ ↑ ] o clic en selector
            document.addEventListener('change', async (e) => {
                const target = e.target;
                if (target && target.type === 'file' && target.closest('.file-upload-zone, .upload-card')) {
                    if (target.__audioExtracted) {
                        delete target.__audioExtracted;
                        return;
                    }

                    const originalFiles = Array.from(target.files || []);
                    if (!originalFiles.length) return;

                    const zone = target.closest('.file-upload-zone, .upload-card');
                    const hasVideo = originalFiles.some(isVideoFile);

                    if (!hasVideo) {
                        renderOptimizationBadge(zone, null);
                        handleFilesUploadStart(zone, target.files, null);
                        return;
                    }

                    // Interceptar antes de que Gradio comience la subida del video pesado
                    e.stopImmediatePropagation();
                    e.preventDefault();

                    try {
                        const { files: processedFiles, stats } = await processFilesWithAudioExtraction(zone, originalFiles);
                        renderOptimizationBadge(zone, stats);
                        const dt = new DataTransfer();
                        processedFiles.forEach((f) => dt.items.add(f));
                        target.files = dt.files;
                        target.__audioExtracted = true;
                        handleFilesUploadStart(zone, dt.files, stats);
                        target.dispatchEvent(new Event('change', { bubbles: true }));
                    } catch (err) {
                        console.warn('[Transcriptor] Fallback a subida original:', err);
                        renderOptimizationBadge(zone, {
                            processed: true,
                            success: false,
                            origMb: formatBytes(originalFiles.reduce((acc, f) => acc + f.size, 0)),
                            reason: err.message
                        });
                        target.__audioExtracted = true;
                        handleFilesUploadStart(zone, target.files, null);
                        target.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            }, true);

            const getUploadInput = (zone) => {
                if (!zone) return null;
                return zone.querySelector('input[type="file"]') || document.querySelector('.file-upload-zone input[type="file"]');
            };

            const isFileInputTarget = (e) => {
                return e.target && e.target.closest ? e.target.closest('.file-upload-zone, .upload-card') : null;
            };

            document.addEventListener('dragover', (e) => {
                const zone = isFileInputTarget(e);
                if (zone && e.dataTransfer && e.dataTransfer.types && Array.from(e.dataTransfer.types).includes('Files')) {
                    e.preventDefault();
                    e.stopPropagation();
                    e.dataTransfer.dropEffect = 'copy';
                    zone.classList.add('drag-over-active');
                }
            }, true);

            document.addEventListener('dragleave', (e) => {
                const zone = isFileInputTarget(e);
                if (zone && (!e.relatedTarget || !zone.contains(e.relatedTarget))) {
                    zone.classList.remove('drag-over-active');
                }
            }, true);

            // Soporte Drag & Drop continuo con extracción de audio automática en cliente
            document.addEventListener('drop', async (e) => {
                const zone = isFileInputTarget(e);
                if (zone) {
                    zone.classList.remove('drag-over-active');
                    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                        e.preventDefault();
                        e.stopPropagation();
                        const fileInput = getUploadInput(zone);
                        if (fileInput) {
                            const rawFiles = Array.from(e.dataTransfer.files);
                            try {
                                const { files: processedFiles, stats } = await processFilesWithAudioExtraction(zone, rawFiles);
                                renderOptimizationBadge(zone, stats);
                                const dt = new DataTransfer();
                                processedFiles.forEach((f) => dt.items.add(f));
                                fileInput.files = dt.files;
                                fileInput.__audioExtracted = true;
                                handleFilesUploadStart(zone, dt.files, stats);
                                fileInput.dispatchEvent(new Event('change', { bubbles: true }));
                            } catch (err) {
                                console.warn('[Transcriptor] Fallback drop a original:', err);
                                renderOptimizationBadge(zone, {
                                    processed: true,
                                    success: false,
                                    origMb: formatBytes(rawFiles.reduce((acc, f) => acc + f.size, 0)),
                                    reason: err.message
                                });
                                const dt = new DataTransfer();
                                rawFiles.forEach((f) => dt.items.add(f));
                                fileInput.files = dt.files;
                                fileInput.__audioExtracted = true;
                                handleFilesUploadStart(zone, dt.files, null);
                                fileInput.dispatchEvent(new Event('change', { bubbles: true }));
                            }
                        }
                    }
                }
            }, true);

            // Limpieza del badge de optimización y reset de cola al vaciar la lista
            document.addEventListener('click', (e) => {
                const clearBtn = e.target && e.target.closest ? e.target.closest('#btn_clear, .file-upload-zone [aria-label*="Clear"], .file-upload-zone .icon-button-wrapper > button:last-child') : null;
                if (clearBtn) {
                    isTranscriptionQueued = false;
                    isExtractingAudio = false;
                    isUploadingFiles = false;
                    activeUploadCount = 0;
                    updateQueueButtonVisual();
                    renderOptimizationBadge(null, null);
                }
            }, true);
        }
    }
