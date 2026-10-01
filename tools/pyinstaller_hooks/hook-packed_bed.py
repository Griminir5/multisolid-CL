"""Only runtime assets; never collect example projects, plugins or outputs."""
from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("packed_bed", include_py_files=True, includes=[
    "compiled/*.cpp", "compiled/*.hpp", "compiled/licenses/*",
    # Scientific provenance fingerprints read these sources at runtime.
    "properties.py", "reactions.py", "parameters.py", "kinetics/*.py",
])
excludedimports = ["packed_bed.examples"]
