"""Gestión y carga de recursos estáticos (CSS y JavaScript) para la interfaz web."""

from functools import lru_cache
from pathlib import Path

# Directorio raíz del repositorio y carpeta assets
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
CSS_THEME_FILE = ASSETS_DIR / "css" / "theme.css"
JS_ENHANCEMENTS_FILE = ASSETS_DIR / "js" / "ui_enhancements.js"


@lru_cache(maxsize=1)
def get_custom_css() -> str:
    """Retorna los estilos CSS globales desde el archivo de tema Aurora Boreal Nórdica."""
    if CSS_THEME_FILE.is_file():
        return CSS_THEME_FILE.read_text(encoding="utf-8")
    return ""


@lru_cache(maxsize=1)
def get_ui_enhancements_js() -> str:
    """Retorna los scripts JavaScript de cliente (Web Audio, Drag & Drop, DPI, etc.)."""
    if JS_ENHANCEMENTS_FILE.is_file():
        return JS_ENHANCEMENTS_FILE.read_text(encoding="utf-8").strip()
    return "() => {}"


MODAL_OPEN_JS = """(...args) => {
    const updateModalCenter = () => {
        const overlay = document.querySelector('.modal-overlay');
        const dialog = document.querySelector('.modal-dialog');
        if (!overlay || !dialog) return;
        const applyCenter = (centerY, centerX) => {
            dialog.style.setProperty('--visible-center-y', `${Math.round(centerY)}px`);
            dialog.style.setProperty('top', `${Math.round(centerY)}px`, 'important');
            dialog.style.setProperty('left', `${Math.round(centerX)}px`, 'important');
            dialog.style.setProperty('transform', 'translate(-50%, -50%)', 'important');
        };
        const vh = window.visualViewport ? window.visualViewport.height : window.innerHeight;
        const vw = window.visualViewport ? window.visualViewport.width : window.innerWidth;
        const offsetTop = window.visualViewport ? window.visualViewport.offsetTop : 0;
        const offsetLeft = window.visualViewport ? window.visualViewport.offsetLeft : 0;
        applyCenter(offsetTop + vh / 2, offsetLeft + vw / 2);

        if ('IntersectionObserver' in window) {
            const io = new IntersectionObserver((entries) => {
                for (const entry of entries) {
                    if (entry.intersectionRect && entry.intersectionRect.height > 0) {
                        const cy = entry.intersectionRect.top + entry.intersectionRect.height / 2;
                        const cx = entry.intersectionRect.left + entry.intersectionRect.width / 2;
                        applyCenter(cy, cx);
                    }
                }
                io.disconnect();
            }, { root: null });
            io.observe(document.documentElement);
        }
    };

    const overlay = document.querySelector('.modal-overlay');
    if (overlay) overlay.classList.remove('hide');
    updateModalCenter();
    setTimeout(updateModalCenter, 30);

    if (!window.__transcriptorModalListeners) {
        window.__transcriptorModalListeners = true;
        window.addEventListener('scroll', updateModalCenter, { passive: true, capture: true });
        window.addEventListener('resize', updateModalCenter, { passive: true });
    }
    return args;
}"""

MODAL_CLOSE_JS = """(...args) => {
    const overlay = document.querySelector('.modal-overlay');
    if (overlay) overlay.classList.add('hide');
    return args;
}"""

COPY_CLIPBOARD_JS = """(text) => {
    if (!text) return;
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text);
    } else {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        try {
            document.execCommand('copy');
        } catch (e) {}
        document.body.removeChild(ta);
    }
}"""
