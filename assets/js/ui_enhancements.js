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

            // 3. Extracción de Audio en el Navegador y Soporte Drag & Drop Continuo
            const isVideoFile = (file) => {
                if (!file) return false;
                const name = (file.name || '').toLowerCase();
                const videoExts = ['.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.flv', '.wmv'];
                return videoExts.some(ext => name.endsWith(ext)) || (file.type && file.type.startsWith('video/'));
            };

            const audioBufferToWav = (buffer) => {
                const numChannels = buffer.numberOfChannels;
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

                const samples = new Int16Array(arrayBuffer, 44, buffer.length * numChannels);
                if (numChannels === 1) {
                    const ch0 = buffer.getChannelData(0);
                    for (let i = 0; i < buffer.length; i++) {
                        const s = Math.max(-1, Math.min(1, ch0[i]));
                        samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }
                } else {
                    const ch0 = buffer.getChannelData(0);
                    const ch1 = buffer.getChannelData(1);
                    let ptr = 0;
                    for (let i = 0; i < buffer.length; i++) {
                        const s0 = Math.max(-1, Math.min(1, ch0[i]));
                        const s1 = Math.max(-1, Math.min(1, ch1[i]));
                        samples[ptr++] = s0 < 0 ? s0 * 0x8000 : s0 * 0x7FFF;
                        samples[ptr++] = s1 < 0 ? s1 * 0x8000 : s1 * 0x7FFF;
                    }
                }
                return new Blob([arrayBuffer], { type: 'audio/wav' });
            };

            const extractAudioFromVideo = async (file) => {
                const ab = await file.arrayBuffer();
                const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
                if (!AudioCtxClass) {
                    throw new Error('Web Audio no soportado');
                }
                const ctx = new AudioCtxClass();
                let decoded;
                try {
                    decoded = await ctx.decodeAudioData(ab);
                } finally {
                    if (ctx.close) ctx.close().catch(() => {});
                }

                // Resamplear a 16kHz mono (estándar de oro universal para WhisperX, Deepgram y AssemblyAI)
                const targetRate = 16000;
                const targetLength = Math.max(1, Math.ceil(decoded.duration * targetRate));
                const offlineCtx = new OfflineAudioContext(1, targetLength, targetRate);
                const source = offlineCtx.createBufferSource();
                source.buffer = decoded;
                source.connect(offlineCtx.destination);
                source.start(0);
                const resampled = await offlineCtx.startRendering();

                const wavBlob = audioBufferToWav(resampled);
                const baseName = file.name.substring(0, file.name.lastIndexOf('.')) || file.name;
                return new File([wavBlob], `${baseName}.wav`, {
                    type: 'audio/wav',
                    lastModified: Date.now()
                });
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
                    <div class="upload-loading-subtext">${stageText || 'Extrayendo audio localmente (100% Lossless)'} · ${fileName}</div>
                `;
            };

            const showUploadOverlay = (zone, count, mbText, title, sub) => {
                const targetZone = zone || document.querySelector('.file-upload-zone');
                if (!targetZone) return;
                let overlay = targetZone.querySelector('.upload-loading-overlay');
                if (!overlay) {
                    overlay = document.createElement('div');
                    overlay.className = 'upload-loading-overlay';
                    targetZone.appendChild(overlay);
                }
                const fileText = count === 1 ? 'audio' : `${count} archivos`;
                const defaultTitle = `⬆ Subiendo ${fileText} al servidor...`;
                const defaultSub = `${mbText ? mbText + ' · ' : ''}Transfiriendo a Hugging Face Space`;
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

            const handleFilesUploadStart = (zone, files) => {
                if (!files || files.length === 0) return;
                let totalBytes = 0;
                let anyWav = false;
                for (let i = 0; i < files.length; i++) {
                    totalBytes += files[i].size || 0;
                    if (files[i].name && files[i].name.endsWith('.wav')) {
                        anyWav = true;
                    }
                }
                const mb = totalBytes > 0 ? `${(totalBytes / (1024 * 1024)).toFixed(1)} MB` : '';
                const title = anyWav ? '⬆ Subiendo audio optimizado...' : `Subiendo ${files.length} archivo(s)...`;
                const sub = `${mb ? mb + ' · ' : ''}Transfiriendo a Hugging Face Space`;
                showUploadOverlay(zone, files.length, mb, title, sub);
                setTimeout(() => hideUploadOverlay(), 45000);
            };

            const processFilesWithAudioExtraction = async (zone, files) => {
                const fileArray = Array.from(files || []);
                const videoIndices = [];
                fileArray.forEach((f, idx) => {
                    if (isVideoFile(f)) videoIndices.push(idx);
                });

                if (videoIndices.length === 0) {
                    return fileArray;
                }

                const processed = [];
                let videoProcessed = 0;

                for (let i = 0; i < fileArray.length; i++) {
                    const f = fileArray[i];
                    if (isVideoFile(f)) {
                        videoProcessed++;
                        const origMb = (f.size / (1024 * 1024)).toFixed(1);
                        updateExtractingOverlay(zone, videoProcessed, videoIndices.length, `${f.name} (${origMb} MB)`, 'Decodificando pista de audio...');
                        try {
                            const audioFile = await extractAudioFromVideo(f);
                            const audioMb = (audioFile.size / (1024 * 1024)).toFixed(1);
                            const savings = Math.max(0, Math.round((1 - audioFile.size / f.size) * 100));
                            console.log(`[Transcriptor] Optimización exitosa: ${f.name} (${origMb} MB) -> ${audioFile.name} (${audioMb} MB, -${savings}%)`);
                            processed.push(audioFile);
                        } catch (err) {
                            console.warn(`[Transcriptor] No se pudo extraer audio local de ${f.name}, subiendo original:`, err);
                            updateExtractingOverlay(zone, videoProcessed, videoIndices.length, f.name, `⚠️ No se pudo extraer audio local (${err.message || 'error'}). Subiendo original...`);
                            processed.push(f);
                        }
                    } else {
                        processed.push(f);
                    }
                }
                return processed;
            };

            // Multi-Network Hook (Fetch y XHR) para ocultar el overlay inmediatamente al finalizar el upload
            if (!window.__transcriptorHookedFetch && window.fetch) {
                window.__transcriptorHookedFetch = true;
                const origFetch = window.fetch;
                window.fetch = async function(...args) {
                    const url = typeof args[0] === 'string' ? args[0] : (args[0] && args[0].url ? args[0].url : '');
                    const isUpload = url.includes('upload') || url.includes('/api/') || (args[1] && (args[1].body instanceof FormData || args[1].body instanceof Blob || args[1].body instanceof File));
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
                    if (url.includes('upload') || (args[0] && (args[0] instanceof FormData || args[0] instanceof Blob || args[0] instanceof File))) {
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
                        handleFilesUploadStart(zone, target.files);
                        return;
                    }

                    // Interceptar antes de que Gradio comience la subida del video pesado
                    e.stopImmediatePropagation();
                    e.preventDefault();

                    try {
                        const processedFiles = await processFilesWithAudioExtraction(zone, originalFiles);
                        const dt = new DataTransfer();
                        processedFiles.forEach((f) => dt.items.add(f));
                        target.files = dt.files;
                        target.__audioExtracted = true;
                        handleFilesUploadStart(zone, dt.files);
                        target.dispatchEvent(new Event('change', { bubbles: true }));
                    } catch (err) {
                        console.warn('[Transcriptor] Fallback a subida original:', err);
                        target.__audioExtracted = true;
                        handleFilesUploadStart(zone, target.files);
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
                                const processedFiles = await processFilesWithAudioExtraction(zone, rawFiles);
                                const dt = new DataTransfer();
                                processedFiles.forEach((f) => dt.items.add(f));
                                fileInput.files = dt.files;
                                fileInput.__audioExtracted = true;
                                handleFilesUploadStart(zone, dt.files);
                                fileInput.dispatchEvent(new Event('change', { bubbles: true }));
                            } catch (err) {
                                console.warn('[Transcriptor] Fallback drop a original:', err);
                                const dt = new DataTransfer();
                                rawFiles.forEach((f) => dt.items.add(f));
                                fileInput.files = dt.files;
                                fileInput.__audioExtracted = true;
                                handleFilesUploadStart(zone, dt.files);
                                fileInput.dispatchEvent(new Event('change', { bubbles: true }));
                            }
                        }
                    }
                }
            }, true);
        }
    }
