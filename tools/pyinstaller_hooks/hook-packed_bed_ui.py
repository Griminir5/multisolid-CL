"""Collect the desktop's small vector icons, without source copies or fixtures."""
from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("packed_bed_ui", includes=["assets/*.svg"])
