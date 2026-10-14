"""Collect desktop icons, fonts, and pre-rendered offline help."""
from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("packed_bed_ui", includes=["assets/*.svg", "assets/fonts/*", "assets/help/*"])
# Export backends are selected by filename at runtime, beyond Qt canvas imports.
hiddenimports = ["matplotlib.backends.backend_svg", "matplotlib.backends.backend_pdf",
                 "matplotlib.backends.backend_ps"]
