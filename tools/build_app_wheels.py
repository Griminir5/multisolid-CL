"""Build runtime-only application wheels in clean temporary source trees."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile


def build(wheels=None):
    root = Path(__file__).resolve().parents[1]
    # Read the checkout's build helper, even when an older wheel is installed.
    sys.path.insert(0, str(root))
    try:
        from packed_bed.release import BUILD_IDENTITY_FILE, source_identity
        identity = source_identity(root)
    finally:
        sys.path.pop(0)
    from build_help import build as build_help
    build_help(check=True)
    work = root / "build/release"
    work.mkdir(parents=True, exist_ok=True)
    wheels = Path(wheels).resolve() if wheels is not None else work / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="application-wheels-", dir=work) as temporary:
        sources = []
        for name, base, package in (("engine", root, "packed_bed"), ("ui", root / "desktop", "packed_bed_ui")):
            target = Path(temporary) / name
            target.mkdir()
            for filename in ("pyproject.toml", "README.md", "LICENSE"):
                source = base / filename
                shutil.copy2(source if source.exists() else root / filename, target / filename)
            shutil.copytree(base / package, target / package,
                            ignore=shutil.ignore_patterns("__pycache__", "examples", "tests"))
            (target / package / BUILD_IDENTITY_FILE).write_text(
                json.dumps(identity, indent=2) + "\n", encoding="utf-8")
            sources.append(str(target))
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-cache-dir", "--no-deps",
                        "--no-build-isolation", "--wheel-dir", str(wheels), *sources], check=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    build(parser.parse_args().output)
