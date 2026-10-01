"""Build runtime-only application wheels in clean temporary source trees."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def build():
    root = Path(__file__).resolve().parents[1]
    work = root / "build/release"
    work.mkdir(parents=True, exist_ok=True)
    wheels = work / "wheels"
    wheels.mkdir(exist_ok=True)
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
            sources.append(str(target))
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-cache-dir", "--no-deps",
                        "--no-build-isolation", "--wheel-dir", str(wheels), *sources], check=True)


if __name__ == "__main__":
    build()
