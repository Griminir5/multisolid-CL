"""Collect only the DAE Tools adapters used by MultiSolid, with their DLL closure."""
from pathlib import Path
import sys

from PyInstaller.utils.hooks import get_package_paths

_, location = get_package_paths("daetools")
root = Path(location)
datas = [(str(path), "daetools") for pattern in ("*.cfg", "*.txt") for path in root.glob(pattern)]
hiddenimports = ["daetools.pyDAE", "daetools.solvers.superlu", "daetools.solvers.trilinos"]
excludedimports = ["daetools.examples", "daetools.dae_plotter", "daetools.code_generators",
                   "daetools.dae_simulator",
                   "daetools.solvers.superlu_mt", "daetools.solvers.intel_pardiso"]
binaries = []
if sys.platform == "win32":
    import pefile
    native = root / "solibs/Windows_win64"
    extension_dir = native / f"py{sys.version_info.major}{sys.version_info.minor}"
    modules = ("pyCore", "pyActivity", "pyDataReporting", "pyIDAS", "pyUnits", "pySuperLU", "pyTrilinos")
    todo = [extension_dir / (name + ".pyd") for name in modules]
    seen = set()
    while todo:
        path = todo.pop()
        if path.name.lower() in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Required DAE Tools extension is missing: {path}")
        seen.add(path.name.lower())
        binaries.append((str(path), "."))
        with pefile.PE(str(path)) as pe:
            for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", ()):
                name = entry.dll.decode()
                dependency = next((directory / name for directory in (extension_dir, native / "lib")
                                   if (directory / name).is_file()), None)
                if dependency is not None:
                    todo.append(dependency)
else:
    from PyInstaller.utils.hooks import collect_dynamic_libs
    binaries = collect_dynamic_libs("daetools")
