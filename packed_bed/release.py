"""Application identity shared by source runs, installed wheels and the desktop."""

from importlib import metadata, util
import json
from pathlib import Path
import subprocess
import sys
import tomllib


MANIFEST_SCHEMA_VERSION = 1
IDENTITY_SCHEMA_VERSION = 1
BUILD_IDENTITY_FILE = "_build_identity.json"


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _source_version(base, name):
    try:
        project = tomllib.loads((base / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        return project["version"] if project["name"] == name else None
    except (OSError, ValueError, KeyError):
        return None


def git_state(root):
    """Inspect only a known source tree, never a results folder or an installation."""
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], check=True,
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        return {"source_commit": git("rev-parse", "HEAD"),
                "source_dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"source_commit": None, "source_dirty": None}


def source_identity(root):
    """Stamp the release inputs before they are copied out of the checkout."""
    root = Path(root)
    engine_version = _source_version(root, "multisolid-cl")
    ui_version = _source_version(root / "desktop", "multisolid-cl-ui")
    if not engine_version or engine_version != ui_version:
        raise ValueError("Engine and desktop versions must match in both pyproject.toml files.")
    desktop = tomllib.loads((root / "desktop/pyproject.toml").read_text(encoding="utf-8"))
    if f"multisolid-cl[compiled]=={engine_version}" not in desktop["project"]["dependencies"]:
        raise ValueError("The desktop engine dependency must match the release version.")
    return {"schema_version": IDENTITY_SCHEMA_VERSION,
            "engine_version": engine_version, "ui_version": ui_version,
            **git_state(root), "build_id": None}


def _package_identity(name, folder):
    # Prefer the loaded source's metadata to a possibly older installed wheel.
    source_version = _source_version(folder.parent, name)
    if source_version:
        return source_version, git_state(folder.parent)
    stamp = _read_json(folder / BUILD_IDENTITY_FILE)
    try:
        package_version = metadata.version(name)
    except metadata.PackageNotFoundError:
        package_version = stamp.get("engine_version" if name == "multisolid-cl" else "ui_version")
    return package_version, stamp


def application_identity():
    """Return JSON-safe provenance for the implementation currently executing.

    Packaged metadata is authoritative. Missing fields remain unknown; a frozen
    executable must never acquire the identity of an unrelated nearby checkout.
    """
    identity = {"schema_version": IDENTITY_SCHEMA_VERSION, "engine_version": None,
                "ui_version": None, "source_commit": None, "source_dirty": None,
                "build_id": None}
    if getattr(sys, "frozen", False):
        release = _read_json(Path(sys.executable).resolve().parent / "release.json")
        for key in identity:
            if key in release:
                identity[key] = release[key]
        # Retain useful versions even in older bundles without the new fields.
        for key, name in (("engine_version", "multisolid-cl"), ("ui_version", "multisolid-cl-ui")):
            if identity[key] is None:
                try:
                    identity[key] = metadata.version(name)
                except metadata.PackageNotFoundError:
                    pass
        return identity
    engine_version, stamp = _package_identity("multisolid-cl", Path(__file__).resolve().parent)
    identity.update(engine_version=engine_version, source_commit=stamp.get("source_commit"),
                    source_dirty=stamp.get("source_dirty"))
    ui = util.find_spec("packed_bed_ui")
    if ui is not None and ui.origin:
        identity["ui_version"], _ = _package_identity("multisolid-cl-ui", Path(ui.origin).resolve().parent)
    return identity
