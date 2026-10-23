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
            const isVideoFile = (file) => {
                if (!file) return false;
                const name = (file.name || '').toLowerCase();
                const videoExts = ['.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.flv', '.wmv'];
                return videoExts.some(ext => name.endsWith(ext)) || (file.type && file.type.startsWith('video/'));
            };

            const formatBytes = (bytes) => {
                if (!bytes || bytes <= 0) return '0 MB';
                const mb = bytes / (1024 * 1024);
                if (mb < 0.1) return `${Math.round(bytes / 1024)} KB`;
                return `${mb.toFixed(1)} MB`;
            };

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
                            <span class="status-text">La IA solo necesita el audio · Sube más rápido con <a href="https://github.com/Klopezxd/transcriptor-suite/raw/main/extraer_audio.bat" target="_blank" class="status-strip-link" title="Descargar herramienta ligera para Windows">extraer_audio.bat ↗</a></span>
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

            const updateExtractingOverlay = (zone, current, total, fileName, stageText) => {
                const targetZone = zone || document.querySelector('.file-upload-zone');
                if (!targetZone) return;
                let overlay = targetZone.querySelector('.upload-loading-overlay');
                if (!overlay) {
                    overlay = document.createElement('div');
                    overlay.className = 'upload-loading-overlay';
                    targetZone.appendChild(overlay);
                }
                overlay.innerHTML = `
                    <div class="upload-spinner"></div>
                    <div class="upload-loading-text">⚡ Optimizando video (${current}/${total})...</div>
                    <div class="upload-loading-subtext">${stageText || 'Extrayendo audio localmente'} · ${fileName}</div>
                `;
            };

            const showUploadOverlay = (zone, stats, title, sub) => {
                const targetZone = zone || document.querySelector('.file-upload-zone');
                if (!targetZone) return;
                let overlay = targetZone.querySelector('.upload-loading-overlay');
                if (!overlay) {
                    overlay = document.createElement('div');
                    overlay.className = 'upload-loading-overlay';
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
                } else if (stats && stats.mb) {
                    defaultTitle = stats.anyAudio ? '⬆ Subiendo audio al servidor...' : `Subiendo ${stats.count} archivo(s)...`;
                    defaultSub = `${stats.mb} · Transfiriendo a Hugging Face Space`;
                }

                overlay.innerHTML = `
                    <button class="overlay-dismiss-btn" title="Cerrar overlay" onclick="this.closest('.upload-loading-overlay').remove()">✕</button>
                    <div class="upload-spinner"></div>
                    <div class="upload-loading-text">${title || defaultTitle}</div>
                    <div class="upload-loading-subtext">${sub || defaultSub}</div>
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
                    }
                });
                mo.observe(zone, { childList: true, subtree: true });
            };
            observeUploadZone();
            setInterval(observeUploadZone, 1000);

            const handleFilesUploadStart = (zone, files, optStats) => {
                if (!files || files.length === 0) return;
                let totalBytes = 0;
                let anyAudio = false;
                for (let i = 0; i < files.length; i++) {
                    totalBytes += files[i].size || 0;
                    const nm = (files[i].name || '').toLowerCase();
                    if (nm.endsWith('.mp3') || nm.endsWith('.wav') || nm.endsWith('.m4a') || nm.endsWith('.flac')) {
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
                setTimeout(() => hideUploadOverlay(), 45000);
            };

            const processFilesWithAudioExtraction = async (zone, files) => {
                const fileArray = Array.from(files || []);
                const videoIndices = [];
                fileArray.forEach((f, idx) => {
                    if (isVideoFile(f)) videoIndices.push(idx);
                });

                if (videoIndices.length === 0) {
                    return { files: fileArray, stats: null };
                }

                // Límite seguro para decodificación Web Audio en el navegador.
                // Archivos mayores a 450 MB generan más de 2 GB de muestras Float32 PCM desnudas en memoria,
                // excediendo el límite de búfer de Chromium y congelando el navegador.
                // Para videos mayores a 450 MB, el servidor los procesa con FFmpeg nativo en 1 segundo.
                const MAX_BROWSER_DECODE_SIZE = 450 * 1024 * 1024; // 450 MB

                const processed = [];
                let videoProcessed = 0;
                let totalOrigBytes = 0;
                let totalOptBytes = 0;
                let anyOptimized = false;
                let anyLargeVideo = false;
                let anyFallback = false;
                let failureReason = '';

                for (let i = 0; i < fileArray.length; i++) {
                    const f = fileArray[i];
                    if (isVideoFile(f)) {
                        videoProcessed++;
                        totalOrigBytes += f.size;
                        const origMb = formatBytes(f.size);

                        // Si el archivo supera el umbral seguro de decodificación en memoria del navegador:
                        if (f.size > MAX_BROWSER_DECODE_SIZE) {
                            console.log(`[Transcriptor] Video extenso (${origMb} > 450MB). Omitiendo decodificación Web Audio para evitar sobrecarga de RAM; el servidor procesará el audio con FFmpeg.`);
                            totalOptBytes += f.size;
                            anyLargeVideo = true;
                            processed.push(f);
                            continue;
                        }

                        updateExtractingOverlay(zone, videoProcessed, videoIndices.length, `${f.name} (${origMb})`, 'Iniciando optimización...');
                        try {
                            const audioFile = await extractAudioFromVideo(f, (stageText) => {
                                updateExtractingOverlay(zone, videoProcessed, videoIndices.length, `${f.name} (${origMb})`, stageText);
                            });

                            const audioMb = formatBytes(audioFile.size);

                            // Si el audio generado es más pequeño que el video, lo usamos
                            if (audioFile.size < f.size) {
                                const savedBytes = f.size - audioFile.size;
                                const savings = Math.max(0, Math.round((savedBytes / f.size) * 100));
                                totalOptBytes += audioFile.size;
                                anyOptimized = true;
                                console.log(`[Transcriptor] Optimización exitosa: ${f.name} (${origMb}) -> ${audioFile.name} (${audioMb}, -${savings}%)`);
                                processed.push(audioFile);
                            } else {
                                // El video ya era más pequeño que el audio (conservamos el original)
                                totalOptBytes += f.size;
                                console.log(`[Transcriptor] El archivo original (${origMb}) ya era más ligero que el audio. Conservando original.`);
                                processed.push(f);
                            }
                        } catch (err) {
                            console.warn(`[Transcriptor] No se pudo extraer audio local de ${f.name}, subiendo original:`, err);
                            failureReason = err.message || 'Codec no decodificable en navegador';
                            totalOptBytes += f.size;
                            anyFallback = true;
                            processed.push(f);
                        }
                    } else {
                        totalOrigBytes += f.size;
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
                    fallback: anyFallback && !anyOptimized,
                    origTotal: totalOrigBytes,
                    optTotal: totalOptBytes,
                    savedBytes: savedBytes,
                    savingsPct: savingsPct,
                    origMb: formatBytes(totalOrigBytes),
                    optMb: formatBytes(totalOptBytes),
                    savedMb: formatBytes(savedBytes),
                    reason: failureReason
                };

                return { files: processed, stats };
            };

            // Multi-Network Hook (Fetch y XHR) para ocultar el overlay inmediatamente al finalizar el upload
            if (!window.__transcriptorHookedFetch && window.fetch) {
                window.__transcriptorHookedFetch = true;
                const origFetch = window.fetch;
                window.fetch = async function(...args) {
                    const url = typeof args[0] === 'string' ? args[0] : (args[0] && args[0].url ? args[0].url : '');
                    const isUpload = url.includes('/upload') || (args[1] && (args[1].body instanceof FormData || args[1].body instanceof Blob || args[1].body instanceof File));
                    try {
                        const res = await origFetch.apply(this, args);
                        if (isUpload) {
                            setTimeout(() => hideUploadOverlay(), 300);
                        }
                        return res;
                    } catch (err) {
                        if (isUpload) hideUploadOverlay();
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
                    if (url.includes('/upload') || (args[0] && (args[0] instanceof FormData || args[0] instanceof Blob || args[0] instanceof File))) {
                        this.addEventListener('loadend', () => {
                            setTimeout(() => hideUploadOverlay(), 300);
                        });
                    }
                    return origSend.apply(this, args);
                };
            }

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

            // Limpieza del badge de optimización al vaciar la lista
            document.addEventListener('click', (e) => {
                const clearBtn = e.target && e.target.closest ? e.target.closest('#btn_clear, .file-upload-zone [aria-label*="Clear"], .file-upload-zone .icon-button-wrapper > button:last-child') : null;
                if (clearBtn) {
                    renderOptimizationBadge(null, null);
                }
            }, true);
        }
    }
