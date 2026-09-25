"""Build a folder-based desktop bundle from a prepared release environment."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    import daetools
    from packed_bed.compiled.bundle import engine_fingerprint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/desktop"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for name in ("compiled", "graphviz"):
        if not (root / "desktop/vendor" / name).is_dir():
            raise ValueError(f"Stage desktop/vendor/{name} before freezing the application.")
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--windowed",
               "--name", "MultiSolid", "--distpath", str(args.output.resolve()),
               "--workpath", str(root / "build/desktop"), "--specpath", str(root / "build"),
               "--paths", str(root), "--paths", str(root / "desktop"), "--exclude-module", "sksundae",
               "--paths", daetools.py_sodir,
               "--exclude-module", "PyQt6.QtWebEngineCore", "--exclude-module", "PyQt6.QtWebEngineWidgets"]
    for package in ("packed_bed", "packed_bed_ui", "daetools"):
        command += ["--collect-all", package]
    # Libraries are loaded dynamically by the DAE Tools solver registry.
    for solver in ("superlu", "superlu_mt", "trilinos"):
        command += ["--hidden-import", "daetools.solvers." + solver]
    # DAE Tools adds these platform-specific extension modules to sys.path at
    # runtime. They are not subpackages and collect-all cannot discover them.
    for module in ("pyCore", "pyActivity", "pyDataReporting", "pyIDAS", "pyUnits",
                   "pySuperLU", "pySuperLU_MT", "pyTrilinos"):
        command += ["--hidden-import", module]
    command += [str(root / "tools/desktop_launcher.py")]
    subprocess.run(command, check=True)
    destination = args.output / "MultiSolid"
    for name in ("compiled", "graphviz"):
        shutil.copytree(root / "desktop/vendor" / name, destination / name)
    path = destination / "compiled/manifest.json"
    data = json.loads(path.read_text())
    data["engine_sha256"] = engine_fingerprint()
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
