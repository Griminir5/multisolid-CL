"""Build a folder-based desktop bundle from a prepared release environment."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import daetools
    from packed_bed.compiled.bundle import engine_fingerprint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/desktop"))
    parser.add_argument("--work", type=Path, default=Path("build/desktop"))
    parser.add_argument("--compiled-bundle", type=Path, default=Path("desktop/vendor/compiled"))
    parser.add_argument("--graphviz-bundle", type=Path, default=Path("desktop/vendor/graphviz"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # Reject an installed UI wheel with stale or incomplete help before freezing it.
    from build_help import build_pages
    from packed_bed_ui.help import HELP_ROOT
    for filename, content in build_pages().items():
        path = HELP_ROOT / filename
        if not path.is_file() or path.read_bytes() != content.encode('utf-8'):
            raise ValueError('Rebuild and reinstall the UI wheel: bundled help is stale: ' + filename)
    for package in ("packed_bed", "packed_bed_ui", "daetools"):
        from importlib.util import find_spec
        location = Path(find_spec(package).origin).resolve()
        if not location.is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError(f"Install {package} as a wheel in an isolated release environment: {location}")
    bundles = {"compiled": args.compiled_bundle.resolve(), "graphviz": args.graphviz_bundle.resolve()}
    for name, folder in bundles.items():
        if not folder.is_dir():
            raise ValueError(f"Stage the {name} bundle before freezing: {folder}")
    if sys.platform.startswith("linux"):
        from linux_bundle import check_qt, check_runtime
        check_qt()
        check_runtime(bundles["compiled"] / "lib")
    work = args.work.resolve()
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--windowed",
               "--name", "MultiSolid", "--distpath", str(args.output.resolve()),
               "--workpath", str(work), "--specpath", str(work),
               "--additional-hooks-dir", str(root / "tools/pyinstaller_hooks"),
               "--exclude-module", "sksundae", "--exclude-module", "tkinter",
               "--copy-metadata", "multisolid-cl", "--copy-metadata", "multisolid-cl-ui",
               "--copy-metadata", "xarray",
               "--paths", daetools.py_sodir,
               "--exclude-module", "PyQt6.QtWebEngineCore", "--exclude-module", "PyQt6.QtWebEngineWidgets"]
    if sys.platform == "win32":
        from windows_resources import generate
        icon, version_file = generate(root)
        command += ["--icon", str(icon), "--version-file", str(version_file)]
    for module in ("pytest", "_pytest", "setuptools", "wheel", "pip",
                   "SALib", "multiprocess", "lxml", "h5py", "daetools.dae_simulator",
                   "daetools.examples", "packed_bed.examples", "packed_bed.cli", "packed_bed_ui.release_check"):
        command += ["--exclude-module", module]
    for module in ("daetools.solvers.superlu_mt", "pySuperLU_MT",
                   "daetools.solvers.intel_pardiso", "pyIntelPardiso"):
        command += ["--exclude-module", module]
    # Libraries are loaded dynamically by the DAE Tools solver registry.
    for solver in ("superlu", "trilinos"):
        command += ["--hidden-import", "daetools.solvers." + solver]
    # DAE Tools adds these platform-specific extension modules to sys.path at
    # runtime. They are not subpackages and collect-all cannot discover them.
    for module in ("pyCore", "pyActivity", "pyDataReporting", "pyIDAS", "pyUnits",
                   "pySuperLU", "pyTrilinos"):
        command += ["--hidden-import", module]
    command += [str(root / "tools/desktop_launcher.py")]
    # PyInstaller's isolated hook subprocesses include their working directory
    # in module discovery. Never run them from the checkout: example outputs
    # and developer caches must not be collected as package data.
    isolated = work / "freezer-cwd"
    isolated.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    subprocess.run(command, check=True, cwd=isolated, env=environment)
    destination = args.output.resolve() / "MultiSolid"
    for name, folder in bundles.items():
        shutil.copytree(folder, destination / name)
    path = destination / "compiled/manifest.json"
    data = json.loads(path.read_text())
    data["engine_sha256"] = engine_fingerprint()
    path.write_text(json.dumps(data, indent=2) + "\n")
    readme = "PORTABLE-README-LINUX.txt" if sys.platform.startswith("linux") else "PORTABLE-README.txt"
    shutil.copy2(root / "desktop" / readme, destination / "README.txt")
    if sys.platform.startswith("linux"):
        # TIFF is not an application input/output format. Qt's optional plugin
        # may depend on a libtiff ABI that is absent from the build distribution.
        for path in (destination / "_internal").rglob("libqtiff.so"):
            path.unlink()
        from linux_bundle import dependencies
        for path in (destination / "_internal").rglob("libqxcb.so"):
            dependencies(path, library_path=destination / "_internal")
    from release_metadata import remove_duplicate_dae_libraries
    remove_duplicate_dae_libraries(destination)
    from audit_windows_bundle import audit
    audit(destination, work / "MultiSolid/Analysis-00.toc")
    from release_metadata import write_metadata
    write_metadata(destination, root)
    print(destination)


if __name__ == "__main__":
    main()
