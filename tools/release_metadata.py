"""Record the release environment and carry its dependency license files."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import sysconfig


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def remove_duplicate_dae_libraries(destination):
    """PyInstaller can also copy our root native libraries into solibs."""
    internal = Path(destination).resolve() / "_internal"
    for path in (internal / "daetools/solibs").rglob("*"):
        if not path.is_file() or not (path.suffix.lower() == ".dll" or ".so" in path.suffixes):
            continue
        canonical = internal / path.name
        # On Linux PyInstaller often puts a symlink at the root pointing to
        # this exact nested file. That is already deduplicated; keep its target.
        if canonical.is_symlink():
            continue
        if canonical.is_file() and sha256(path) == sha256(canonical):
            path.unlink()


def python_license():
    candidates = [Path(sys.base_prefix) / "LICENSE.txt",
                  Path(sysconfig.get_path("stdlib")) / "LICENSE.txt"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Python licence not found; checked: " + ", ".join(map(str, candidates)))


def content_identity(destination):
    """Hash every shipped file, including ELF executables and versioned .so files."""
    identity = hashlib.sha256()
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name not in {"release.json", "bundle-files.json"}:
            identity.update(path.relative_to(destination).as_posix().encode())
            identity.update(b"\0")
            identity.update(bytes.fromhex(sha256(path)))
    return identity.hexdigest()[:12]


def write_metadata(destination, root):
    from packed_bed.release import BUILD_IDENTITY_FILE, application_identity, source_identity
    identity = application_identity()
    expected = source_identity(root)
    for name, package in (("multisolid-cl", "packed_bed"), ("multisolid-cl-ui", "packed_bed_ui")):
        stamp = Path(metadata.distribution(name).locate_file(f"{package}/{BUILD_IDENTITY_FILE}"))
        if not stamp.is_file() or json.loads(stamp.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Rebuild and reinstall {name}: its source identity does not match this checkout.")
    if identity != expected:
        raise ValueError("The installed application versions do not match the release inputs.")
    license_dir = destination / "licenses"
    license_dir.mkdir(exist_ok=True)
    shutil.copy2(root / "LICENSE", destination / "LICENSE.txt")
    shutil.copy2(root / "THIRD_PARTY_NOTICES.md", license_dir / "PROJECT-NOTICES.md")
    shutil.copy2(python_license(), license_dir / "Python-LICENSE.txt")
    packages = []
    for distribution in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = distribution.metadata["Name"]
        notices = []
        for relative in distribution.files or ():
            if not any(token in relative.name.lower() for token in ("license", "licence", "copying", "copyright", "notice")):
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file() or source.suffix.lower() in {".py", ".pyc", ".pyd", ".dll"} or ".so" in source.suffixes:
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
        "application": identity,
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "signing": "unsigned", "qualification": "See the accompanying validation report; clean VM qualification is separate."
    }, indent=2) + "\n", encoding="utf-8")
    release = {**identity, "version": identity["ui_version"],
               "build_id": content_identity(destination), "architecture": "x64",
               "platform": platform.system().lower()}
    if sys.platform.startswith("linux"):
        from linux_bundle import glibc_requirement
        release["minimum_glibc"] = glibc_requirement(destination)
    (destination / "release.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    files = {p.relative_to(destination).as_posix(): sha256(p) for p in sorted(destination.rglob("*"))
             if p.is_file() and p.name != "bundle-files.json"}
    (destination / "bundle-files.json").write_text(json.dumps(files, indent=2) + "\n")
    return release
