/**
 * Transcriptor Suite - Vanilla JS Client Controller (~12 KB)
 * 
 * Controlador modular y desacoplado para la interfaz semántica Aurora Boreal Nórdica.
 * Coordina Drag & Drop, verificación acústica Lossless (Web Audio API / PCM Header),
 * reactividad dinámica de proveedores y sincronización bidireccional con el Headless Bridge.
 */

(() => {
    'use strict';

    if (window.__transcriptorSuiteInitialized) return;
    window.__transcriptorSuiteInitialized = true;

    // ── 1. CONFIGURACIÓN DE PROVEEDORES Y MODELOS ───────────────────────────
    const PROVIDERS = {
        'AssemblyAI (Cloud API)': {
            models: [
                ['assemblyai-stt-3-7', 'assemblyai-stt-3-7 · Nueva Generación (Recomendado)'],
                ['universal-2', 'universal-2 · Cobertura Extendida (99 idiomas)']
            ],
            info: '<div class="info-notice"><span>💡</span><span><b>AssemblyAI:</b> Precisión acústica y comprensión contextual de alta fidelidad. Modelo <code>assemblyai-stt-3-7</code> por defecto con respaldo multilingüe <code>universal-2</code>.</span></div>',
            key: 'assemblyai'
        },
        'Deepgram (Cloud API)': {
            models: [
                ['nova-3', 'nova-3 · Robusto & Acentos (Por Defecto)'],
                ['nova-2', 'nova-2 · Rápido & Económico']
            ],
            info: '<div class="info-notice"><span>💡</span><span><b>Deepgram Nova-3:</b> Velocidad ultrarrápida y alta resistencia a ruido ambiental y acentos complejos. Modelo <code>nova-3</code> por defecto con opción económica <code>nova-2</code>.</span></div>',
            key: 'deepgram'
        },
        'WhisperX (ZeroGPU / Local)': {
            models: [
                ['large-v3-turbo', 'large-v3-turbo · Recomendado'],
                ['large-v3', 'large-v3 · Máxima Precisión'],
                ['medium', 'medium · Equilibrado'],
                ['small', 'small · Ligero & Rápido'],
                ['base', 'base · Ultraligero']
            ],
            info: '<div class="info-notice"><span>💡</span><span><b>WhisperX:</b> Transcripción de código abierto con alineación forzada a nivel de fonema (Wav2Vec2) y diarización opcional de interlocutores vía PyAnnote.</span></div>',
            key: 'whisperx'
        }
    };

    // ── 2. UTILIDADES DOM Y FORMATO ─────────────────────────────────────────
    const $ = (sel, parent = document) => parent.querySelector(sel);

    function formatBytes(bytes) {
        if (!bytes) return '0 B';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return (bytes / Math.pow(1024, i)).toFixed(i ? 1 : 0) + ' ' + ['B', 'KB', 'MB', 'GB'][i];
    }

    async function inspectAudioFormat(file) {
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        if (ext === 'flac') return { isLossless: true, badge: '⚡ FLAC Lossless' };
        if (ext === 'wav') {
            try {
                const buf = await file.slice(0, 44).arrayBuffer();
                const v = new DataView(buf);
                if (v.getUint32(0) === 0x52494646 && v.getUint32(8) === 0x57415645) { // RIFF & WAVE
                    const fmt = v.getUint16(20, true);
                    const rate = v.getUint32(24, true);
                    const bits = v.getUint16(34, true);
                    if (fmt === 1 || fmt === 3) {
                        return { isLossless: true, badge: `⚡ WAV ${bits}b ${Math.round(rate / 1000)}kHz` };
                    }
                }
            } catch (_) {}
            return { isLossless: true, badge: '⚡ WAV Lossless' };
        }
        return { isLossless: false, badge: ext.toUpperCase() || 'MEDIA' };
    }

    function copyToClipboard(text, onDone) {
        const fallback = () => {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.cssText = 'position:fixed;opacity:0;left:-9999px';
            document.body.appendChild(ta);
            ta.select();
            try { document.execCommand('copy'); if (onDone) onDone(); } catch (_) {}
            document.body.removeChild(ta);
        };
        if (navigator.clipboard?.writeText) {
            navigator.clipboard.writeText(text).then(onDone).catch(fallback);
        } else {
            fallback();
        }
    }

    function triggerDownload(filename, content) {
        const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // ── 3. GRADIO HEADLESS BRIDGE HELPERS ───────────────────────────────────
    const bridge = {
        set(id, val) {
            const el = $(`#${id} input, #${id} textarea, #${id} select, #${id}`);
            if (!el) return;
            if (el.type === 'checkbox') el.checked = Boolean(val);
            else el.value = val;
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        },
        click(id) {
            const el = $(`#${id} button, #${id} a, #${id}`);
            if (el?.click) el.click();
        }
    };

    // ── 4. CONTROLADOR NATIVO VANILLA JS ────────────────────────────────────
    class TranscriptorController {
        constructor() {
            this.files = [];
            this.isAdmin = false;
            this.isBusy = false;
            this.root = null;
            this.init();
        }

        init() {
            this.root = $('#transcriptor_app');
            if (!this.root) return setTimeout(() => this.init(), 50);

            this.setupDropZone();
            this.setupOptions();
            this.setupAdminModal();
            this.setupExport();
            this.setupBridgeObserver();

            this.onProviderChanged();
            this.updateTranscribeBtn();
        }

        // ── Drag & Drop y Selección de Archivos ──────────────────────────────
        setupDropZone() {
            const zone = $('#drop_zone', this.root);
            const input = $('#media_file_input', this.root);
            if (!zone || !input) return;

            zone.addEventListener('click', (e) => {
                if (!e.target.closest('.file-remove-btn')) input.click();
            });

            ['dragenter', 'dragover'].forEach((ev) => {
                zone.addEventListener(ev, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
                    zone.classList.add('drag-over');
                });
            });

            ['dragleave', 'dragend'].forEach((ev) => {
                zone.addEventListener(ev, (e) => {
                    e.preventDefault();
                    zone.classList.remove('drag-over');
                });
            });

            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                e.stopPropagation();
                zone.classList.remove('drag-over');
                if (e.dataTransfer?.files?.length) {
                    this.addFiles(Array.from(e.dataTransfer.files));
                }
            });

            input.addEventListener('change', () => {
                if (input.files?.length) {
                    this.addFiles(Array.from(input.files));
                }
            });
        }

        async addFiles(incoming) {
            for (const f of incoming) {
                if (!this.files.some((x) => x.name === f.name && x.size === f.size)) {
                    this.files.push(f);
                }
            }
            await this.renderFiles();
            this.syncFilesToBridge();
            this.updateTranscribeBtn();
        }

        async renderFiles() {
            const empty = this.files.length === 0;
            const prompt = $('#upload_prompt', this.root);
            const container = $('#file_list_container', this.root);
            const tbody = $('#file_preview_tbody', this.root);

            if (prompt) prompt.style.display = empty ? 'flex' : 'none';
            if (container) container.style.display = empty ? 'none' : 'block';
            if (!tbody || empty) { if (tbody) tbody.innerHTML = ''; return; }

            tbody.innerHTML = '';
            for (let i = 0; i < this.files.length; i++) {
                const f = this.files[i];
                const info = await inspectAudioFormat(f);
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="file-name-cell" title="${f.name}">${f.name}</td>
                    <td class="file-size-cell">${formatBytes(f.size)}</td>
                    <td class="file-format-cell">${info.isLossless
                        ? `<span class="file-badge-lossless">${info.badge}</span>`
                        : `<span class="file-badge-format">${info.badge}</span>`}</td>
                    <td class="file-action-cell" style="text-align:right;">
                        <button type="button" class="file-remove-btn" aria-label="Eliminar archivo">✕</button>
                    </td>`;

                tr.querySelector('.file-remove-btn').onclick = (e) => {
                    e.stopPropagation();
                    this.files.splice(i, 1);
                    this.renderFiles();
                    this.syncFilesToBridge();
                    this.updateTranscribeBtn();
                };
                tbody.appendChild(tr);
            }
        }

        syncFilesToBridge() {
            const input = $('#bridge_media_input input[type="file"]') || $('#headless-bridge input[type="file"]');
            if (!input) return;
            const dt = new DataTransfer();
            this.files.forEach((f) => dt.items.add(f));
            input.files = dt.files;
            input.dispatchEvent(new Event('change', { bubbles: true }));
        }

        // ── Reactividad de Menús Desplegables y Credenciales ────────────────
        setupOptions() {
            $('#provider_select', this.root)?.addEventListener('change', () => this.onProviderChanged());
            $('#diarize_checkbox', this.root)?.addEventListener('change', () => this.onDiarizeChanged());
            $('#btn_clear', this.root)?.addEventListener('click', () => this.clearAll());
            $('.main-actions #btn_transcribe', this.root)?.addEventListener('click', () => this.startTranscription());
        }

        onProviderChanged() {
            const provSelect = $('#provider_select', this.root);
            const modSelect = $('#model_select', this.root);
            const infoBox = $('#provider_info_box', this.root);
            if (!provSelect || !modSelect) return;

            const prov = provSelect.value;
            const data = PROVIDERS[prov] || PROVIDERS['AssemblyAI (Cloud API)'];

            modSelect.innerHTML = '';
            data.models.forEach(([val, label], idx) => {
                const opt = document.createElement('option');
                opt.value = val;
                opt.textContent = label;
                if (idx === 0) opt.selected = true;
                modSelect.appendChild(opt);
            });

            if (infoBox && data.info) infoBox.innerHTML = data.info;
            this.updateCredentials();

            bridge.set('bridge_provider', prov);
            bridge.set('bridge_model', modSelect.value);
        }

        onDiarizeChanged() {
            this.updateCredentials();
            bridge.set('bridge_diarize', $('#diarize_checkbox', this.root)?.checked);
        }

        updateCredentials() {
            const panel = $('#guest_keys_panel', this.root);
            if (!panel) return;

            if (this.isAdmin) {
                panel.style.display = 'none';
                return;
            }
            panel.style.display = 'block';

            const prov = $('#provider_select', this.root)?.value || '';
            const diarize = $('#diarize_checkbox', this.root)?.checked;

            $('#whisperx_notice', panel).style.display = (prov.includes('WhisperX') && !diarize) ? 'flex' : 'none';
            $('#hf_token_group', panel).style.display = (prov.includes('WhisperX') && diarize) ? 'block' : 'none';
            $('#deepgram_key_group', panel).style.display = prov.includes('Deepgram') ? 'block' : 'none';
            $('#assemblyai_key_group', panel).style.display = prov.includes('AssemblyAI') ? 'block' : 'none';
        }

        updateTranscribeBtn() {
            const btn = $('.main-actions #btn_transcribe', this.root);
            if (!btn) return;
            if (this.isBusy) {
                btn.disabled = true;
                btn.innerHTML = '<span>⏳</span> Transcribiendo...';
            } else {
                btn.disabled = this.files.length === 0;
                btn.innerHTML = '<span>⚡</span> Iniciar Transcripción';
            }
        }

        // ── Modal de Administrador (Overlay / Dialog) ────────────────────────
        setupAdminModal() {
            const modal = $('#modal_login', this.root);
            const pin = $('#input_pin', this.root);
            const msg = $('#login_msg', this.root);
            if (!modal) return;

            const open = () => {
                modal.classList.remove('hide');
                if (pin) { pin.value = ''; setTimeout(() => pin.focus(), 60); }
                if (msg) msg.style.display = 'none';
                bridge.click('bridge_btn_open_login');
            };

            const close = () => {
                modal.classList.add('hide');
                if (pin) pin.value = '';
                if (msg) msg.style.display = 'none';
                bridge.click('bridge_btn_close_login');
            };

            $('#btn_open_login', this.root)?.addEventListener('click', open);
            $('#btn_close_login', this.root)?.addEventListener('click', close);
            $('#btn_cancel_login', this.root)?.addEventListener('click', close);

            modal.addEventListener('click', (e) => { if (e.target === modal) close(); });
            document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !modal.classList.contains('hide')) close(); });

            const submit = () => {
                const val = pin?.value.trim();
                if (!val) {
                    if (msg) {
                        msg.innerHTML = '<span style="color:var(--warning-amber)">Ingresa un PIN válido</span>';
                        msg.style.display = 'block';
                    }
                    return;
                }
                bridge.set('bridge_input_pin', val);
                bridge.click('bridge_btn_submit_login');
            };

            pin?.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); submit(); } });
            $('#btn_submit_login', this.root)?.addEventListener('click', submit);
            $('#btn_logout', this.root)?.addEventListener('click', () => bridge.click('bridge_btn_logout'));
        }

        // ── Exportación: Copiar y Descargas ──────────────────────────────────
        setupExport() {
            const copyBtn = $('#btn_copy', this.root);
            const dlBtn = $('#btn_download', this.root);
            const consBtn = $('#btn_download_consolidated', this.root);
            const out = $('#output_transcript', this.root);

            copyBtn?.addEventListener('click', () => {
                const text = out?.value || '';
                if (!text) return;
                const orig = copyBtn.innerHTML;
                copyToClipboard(text, () => {
                    copyBtn.innerHTML = '<span>✓</span> ¡Copiado!';
                    copyBtn.classList.add('copied');
                    setTimeout(() => {
                        copyBtn.innerHTML = orig;
                        copyBtn.classList.remove('copied');
                    }, 1800);
                });
            });

            dlBtn?.addEventListener('click', () => {
                const a = $('#bridge_btn_download a');
                if (a?.href) return a.click();
                const b = $('#bridge_btn_download button, #bridge_btn_download');
                if (b && !b.disabled) return b.click();
                if (out?.value) triggerDownload('transcripcion.txt', out.value);
            });

            consBtn?.addEventListener('click', () => {
                bridge.click('bridge_btn_download_consolidated');
            });
        }

        // ── Sincronización con Headless Bridge ───────────────────────────────
        startTranscription() {
            if (!this.files.length) return;
            this.isBusy = true;
            this.updateTranscribeBtn();

            bridge.set('bridge_provider', $('#provider_select', this.root)?.value || '');
            bridge.set('bridge_model', $('#model_select', this.root)?.value || '');
            bridge.set('bridge_diarize', $('#diarize_checkbox', this.root)?.checked || false);
            bridge.set('bridge_timestamps', $('#timestamps_select', this.root)?.value || '');
            bridge.set('bridge_hf_token', $('#hf_token_input', this.root)?.value || '');
            bridge.set('bridge_deepgram_key', $('#deepgram_key_input', this.root)?.value || '');
            bridge.set('bridge_assemblyai_key', $('#assemblyai_key_input', this.root)?.value || '');

            const badge = $('#status_badge_container', this.root);
            if (badge) badge.innerHTML = '<div class="status-badges-group"><span class="status-badge status-warning">Procesando...</span></div>';

            const btn = $('#headless-bridge #btn_transcribe button') || $('#headless-bridge #btn_transcribe');
            if (btn?.click) btn.click();
        }

        clearAll() {
            this.files = [];
            this.renderFiles();
            this.isBusy = false;

            const out = $('#output_transcript', this.root);
            if (out) out.value = '';
            const badge = $('#status_badge_container', this.root);
            if (badge) badge.innerHTML = '<div class="status-badges-group"><span class="status-badge status-warning">En espera</span></div>';
            const dl = $('#btn_download', this.root);
            if (dl) dl.disabled = true;
            const cons = $('#btn_download_consolidated', this.root);
            if (cons) cons.style.display = 'none';
            const copy = $('#btn_copy', this.root);
            if (copy) copy.disabled = true;

            this.updateTranscribeBtn();
            bridge.click('bridge_btn_clear');
        }

        setupBridgeObserver() {
            const bridgeEl = $('#headless-bridge');
            if (!bridgeEl) return;

            let scheduled = false;
            const observer = new MutationObserver(() => {
                if (scheduled) return;
                scheduled = true;
                requestAnimationFrame(() => {
                    scheduled = false;
                    this.syncBridgeOutputs();
                });
            });

            observer.observe(bridgeEl, {
                childList: true,
                subtree: true,
                characterData: true,
                attributes: true,
                attributeFilter: ['class', 'style', 'disabled', 'href', 'value']
            });

            this.syncBridgeOutputs();
        }

        syncBridgeOutputs() {
            // 1. Texto transcrito
            const bridgeOut = $('#bridge_output_textbox textarea') || $('#bridge_output_textbox input');
            const out = $('#output_transcript', this.root);
            if (bridgeOut && out) {
                const text = bridgeOut.value || '';
                if (out.value !== text) out.value = text;
                const copyBtn = $('#btn_copy', this.root);
                if (copyBtn) copyBtn.disabled = !text.trim();
            }

            // 2. Badges de estado
            const bridgeBadge = $('#bridge_status_badge');
            const badgeContainer = $('#status_badge_container', this.root);
            if (bridgeBadge && badgeContainer) {
                const html = bridgeBadge.innerHTML.trim();
                if (html && !html.includes('display: none')) badgeContainer.innerHTML = html;
            }

            // 3. Estado del botón Transcribir
            const bridgeBtn = $('#headless-bridge #btn_transcribe button') || $('#headless-bridge #btn_transcribe');
            if (bridgeBtn) {
                const disabled = bridgeBtn.disabled || bridgeBtn.classList.contains('disabled');
                if (!disabled && this.isBusy) {
                    this.isBusy = false;
                    this.updateTranscribeBtn();
                }
            }

            // 4. Botón de descarga
            const dlContainer = $('#bridge_btn_download');
            const dlBtn = $('#btn_download', this.root);
            if (dlContainer && dlBtn) {
                const a = $('a', dlContainer);
                const b = $('button', dlContainer);
                const ready = (a && a.href) || (b && !b.disabled);
                const hasText = Boolean(out?.value.trim());
                dlBtn.disabled = !(ready || hasText);
            }

            // 5. Botón de descarga consolidada
            const consContainer = $('#bridge_btn_download_consolidated');
            const consBtn = $('#btn_download_consolidated', this.root);
            if (consContainer && consBtn) {
                const visible = consContainer.style.display !== 'none' && !consContainer.classList.contains('hide');
                consBtn.style.display = visible ? 'inline-flex' : 'none';
                if (visible) {
                    const b = $('button, a', consContainer);
                    consBtn.disabled = b ? b.disabled : false;
                }
            }

            // 6. Estado de Administrador
            const logoutEl = $('#bridge_btn_logout');
            if (logoutEl) {
                const adminActive = logoutEl.style.display !== 'none' && !logoutEl.classList.contains('hide');
                if (adminActive !== this.isAdmin) {
                    this.isAdmin = adminActive;
                    const openBtn = $('#btn_open_login', this.root);
                    const closeBtn = $('#btn_logout', this.root);
                    if (openBtn) openBtn.style.display = adminActive ? 'none' : 'inline-flex';
                    if (closeBtn) closeBtn.style.display = adminActive ? 'inline-flex' : 'none';
                    if (adminActive) $('#modal_login', this.root)?.classList.add('hide');
                    this.updateCredentials();
                }
            }

            // 7. Mensaje de login
            const bridgeMsg = $('#bridge_login_msg');
            const localMsg = $('#login_msg', this.root);
            if (bridgeMsg && localMsg && bridgeMsg.textContent.trim()) {
                localMsg.innerHTML = bridgeMsg.innerHTML;
                localMsg.style.display = 'block';
            }
        }
    }

    // ── 5. ARRANQUE DEL CLIENTE ─────────────────────────────────────────────
    function start() {
        if (!window.__transcriptorUiInstance) {
            window.__transcriptorUiInstance = new TranscriptorController();
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start);
    } else {
        start();
    }

    return function transcriptorUiEnhancements(...args) {
        start();
        return args;
    };
})();
