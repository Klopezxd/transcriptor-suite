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
    const overlay = document.querySelector('.modal-overlay');
    if (overlay) {
        overlay.classList.remove('hide');
    }
    const input = document.querySelector('.modal-dialog input, input[type="password"]');
    if (input) setTimeout(() => input.focus(), 50);
    return args;
}"""

MODAL_CLOSE_JS = """(...args) => {
    const overlay = document.querySelector('.modal-overlay');
    if (overlay) {
        overlay.classList.add('hide');
    }
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

__all__ = [
    "ROOT_DIR",
    "ASSETS_DIR",
    "CSS_THEME_FILE",
    "JS_ENHANCEMENTS_FILE",
    "get_custom_css",
    "get_ui_enhancements_js",
    "MODAL_OPEN_JS",
    "MODAL_CLOSE_JS",
    "COPY_CLIPBOARD_JS",
]
