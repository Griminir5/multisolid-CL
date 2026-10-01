"""Lightweight font setup shared by the splash and main interface."""
from pathlib import Path

from PyQt6.QtGui import QFontDatabase

_loaded = False


def brand_family(fallback='Impact'):
    global _loaded
    if not _loaded:
        for path in sorted((Path(__file__).parent / 'assets/fonts').glob('*')):
            if path.suffix.lower() in {'.ttf', '.otf'}:
                QFontDatabase.addApplicationFont(str(path))
        _loaded = True
    return next((name for name in QFontDatabase.families() if name.casefold() == 'maratype'), fallback)
