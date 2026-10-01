"""Record the release environment and carry its dependency license files."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import sys


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def remove_duplicate_dae_libraries(destination):
    """PyInstaller's dependency walk can also copy our root DLLs into solibs."""
    internal = Path(destination).resolve() / "_internal"
    for path in (internal / "daetools/solibs").rglob("*.dll"):
        canonical = internal / path.name
        if canonical.is_file() and sha256(path) == sha256(canonical):
            path.unlink()


def write_metadata(destination, root):
    license_dir = destination / "licenses"
    license_dir.mkdir(exist_ok=True)
    shutil.copy2(root / "LICENSE", destination / "LICENSE.txt")
    shutil.copy2(root / "THIRD_PARTY_NOTICES.md", license_dir / "PROJECT-NOTICES.md")
    shutil.copy2(Path(sys.base_prefix) / "LICENSE.txt", license_dir / "Python-LICENSE.txt")
    packages = []
    for distribution in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = distribution.metadata["Name"]
        notices = []
        for relative in distribution.files or ():
            if not any(token in relative.name.lower() for token in ("license", "licence", "copying", "copyright", "notice")):
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file() or source.suffix.lower() in {".py", ".pyc", ".pyd", ".dll"}:
                continue
            if ".." in relative.parts:
                continue
            target = license_dir / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            notices.append(target.relative_to(destination).as_posix())
        packages.append({"name": name, "version": distribution.version,
                         "license": distribution.metadata.get("License-Expression") or distribution.metadata.get("License"),
                         "notices": notices})
    (license_dir / "README.txt").write_text(
        "MultiSolid is GPL-3.0-only. See ../LICENSE.txt.\n\n"
        "Python package license texts are grouped in this directory. This inventory\n"
        "also records build-only packages; it is not a claim that every listed package\n"
        "is included in the executable. Exact shipped files are in bundle-files.json.\n"
        "Graphviz notices are under ../graphviz; compiler/solver notices are under\n"
        "../compiled/licenses and ../compiled/compiler/LICENSE.\n\n"
        "Keep matching application source and dependency source/build assets with\n"
        "distributions of this build. See ../BUILD-INFO.json for versions and\n"
        "release validation details.\n", encoding="utf-8")
    (destination / "BUILD-INFO.json").write_text(json.dumps({
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "signing": "unsigned", "qualification": "See the accompanying validation report; clean VM qualification is separate."
    }, indent=2) + "\n", encoding="utf-8")
    version = metadata.version("multisolid-cl-ui")
    identity = hashlib.sha256()
    for native_manifest in ("compiled/manifest.json", "graphviz/bundle.json"):
        identity.update(bytes.fromhex(sha256(destination / native_manifest)))
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.suffix in {".exe", ".py", ".pyc", ".pyd", ".dll", ".svg"}:
            identity.update(path.relative_to(destination).as_posix().encode())
            identity.update(bytes.fromhex(sha256(path)))
    release = {"version": version, "build_id": identity.hexdigest()[:12], "architecture": "x64"}
    (destination / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    files = {p.relative_to(destination).as_posix(): sha256(p) for p in sorted(destination.rglob("*"))
             if p.is_file() and p.name != "bundle-files.json"}
    (destination / "bundle-files.json").write_text(json.dumps(files, indent=2) + "\n")
    return release
