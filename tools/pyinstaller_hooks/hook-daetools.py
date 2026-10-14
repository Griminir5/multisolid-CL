"""Collect the supported DAE Tools adapters and their native dependencies."""
from pathlib import Path
import sys

from PyInstaller.utils.hooks import get_package_paths

_, location = get_package_paths("daetools")
root = Path(location)
datas = [(str(path), "daetools") for pattern in ("*.cfg", "*.txt") for path in root.glob(pattern)]
# DAETools opens pyDAE/../daetools.cfg. POSIX requires pyDAE to physically
# exist even though its Python module is also present in PyInstaller's archive.
datas.append((str(root / "pyDAE/__init__.py"), "daetools/pyDAE"))
hiddenimports = ["daetools.pyDAE", "daetools.solvers.superlu", "daetools.solvers.trilinos"]
excludedimports = ["daetools.examples", "daetools.dae_plotter", "daetools.code_generators",
                   "daetools.dae_simulator",
                   "daetools.solvers.superlu_mt", "daetools.solvers.intel_pardiso"]
binaries = []
modules = ("pyCore", "pyActivity", "pyDataReporting", "pyIDAS", "pyUnits", "pySuperLU", "pyTrilinos")
if sys.platform == "win32":
    import pefile
    native = root / "solibs/Windows_win64"
    extension_dir = native / f"py{sys.version_info.major}{sys.version_info.minor}"
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
elif sys.platform.startswith("linux"):
    from PyInstaller.depend.bindepend import get_imports
    native = root / "solibs/Linux_x86_64"
    extension_dir = native / f"py{sys.version_info.major}{sys.version_info.minor}"
    todo = [extension_dir / (name + ".so") for name in modules]
    seen = set()
    while todo:
        path = todo.pop()
        if path.name in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Required DAE Tools extension is missing: {path}")
        seen.add(path.name)
        binaries.append((str(path), "."))
        for name, dependency in get_imports(str(path), [str(native / "lib"), str(extension_dir)]):
            if dependency is None:
                raise RuntimeError(f"Missing DAE Tools dependency {name} needed by {path}")
            dependency = Path(dependency)
            if dependency.resolve().is_relative_to(root.resolve()):
                todo.append(dependency)
else:
    raise RuntimeError("Desktop bundles support Windows x64 and Linux x86_64.")
