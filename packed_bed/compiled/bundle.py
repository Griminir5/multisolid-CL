"""Discovery of the relocatable, versioned compiler and SUNDIALS distribution."""

import hashlib
import json
import os
import platform
import sys
from functools import lru_cache
from pathlib import Path


def bundle_root():
    if getattr(sys, "frozen", False):
        roots = [Path(sys.executable).parent / "compiled",
                 Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "compiled"]
    elif os.environ.get("MULTISOLID_COMPILED_BUNDLE"):
        roots = [Path(os.environ["MULTISOLID_COMPILED_BUNDLE"]).resolve()]
    else:
        roots = [Path(__file__).resolve().parents[2] / "desktop/vendor/compiled"]
    for root in roots:
        if (root / "manifest.json").is_file():
            return root
    if getattr(sys, "frozen", False) or os.environ.get("MULTISOLID_COMPILED_BUNDLE"):
        raise RuntimeError("The compiled runtime is missing. Restore the application's compiled bundle.")
    return None


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=8)
def _manifest(path, modified):
    data = json.loads(path.read_text(encoding="utf-8"))
    architecture = platform.machine().lower().replace("amd64", "x86_64")
    if (data.get("format_version") != 1 or data.get("platform") != platform.system()
            or data.get("architecture") != architecture):
        raise RuntimeError("The compiled bundle is incompatible with this platform. Restore the matching application bundle.")
    runtime = data["runtime"]
    if (runtime["version"], runtime["precision"], runtime["index_bits"]) != ("7.5.0", "double", 32):
        raise RuntimeError("Unsupported compiled runtime ABI: expected SUNDIALS 7.5.0, double, 32-bit indices.")
    if data["compiler"]["version"] != "0.16.0":
        raise RuntimeError("The managed compiler must be Zig 0.16.0.")
    return data


def manifest(root):
    path = root / "manifest.json"
    try:
        return _manifest(path, path.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Invalid compiled bundle manifest: {exc}") from exc


def asset(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise RuntimeError(f"Missing or invalid compiled bundle asset: {name}")
    expected = manifest(root).get("files", {}).get(name)
    stat = path.stat()
    if expected is None or _asset_hash(path, stat.st_mtime_ns, stat.st_size) != expected:
        raise RuntimeError(f"Damaged compiled bundle asset: {name}. Restore the application bundle.")
    return path


@lru_cache(maxsize=64)
def _asset_hash(path, modified, size):
    return file_hash(path)


def engine_fingerprint():
    root = bundle_root()
    if getattr(sys, "frozen", False):
        return manifest(root)["engine_sha256"]
    engine = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(engine.rglob("*")):
        if path.suffix in {".py", ".hpp", ".cpp"}:
            digest.update(path.relative_to(engine).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()
