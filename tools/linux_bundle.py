"""Inspect Linux dependencies without relying on the developer's library path."""
import os
from pathlib import Path
import re
import subprocess


BASELINE = re.compile(r"(?:ld-linux.*|lib(?:c|m|mvec|dl|pthread|rt|resolv|util)\.so\..*)$")


def dependencies(path, *, library_path=None):
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("LD_PRELOAD", None)
    environment["LC_ALL"] = "C"
    if library_path is not None:
        environment["LD_LIBRARY_PATH"] = str(library_path)
    result = subprocess.run(["ldd", str(path)], env=environment, capture_output=True, text=True)
    if result.returncode or "not found" in result.stdout:
        raise RuntimeError(f"Unresolved Linux dependencies for {path}:\n{result.stdout}{result.stderr}")
    return [Path(name) for name in re.findall(r"=>\s+(/.*?)\s+\(0x[0-9a-fA-F]+\)", result.stdout)]


def check_runtime(folder):
    """Reject libraries that resolve against the original build/install prefix."""
    folder = Path(folder).resolve()
    libraries = sorted(folder.glob("*.so*"))
    if not libraries:
        raise RuntimeError(f"No Linux runtime libraries found in {folder}")
    for path in libraries:
        dynamic = subprocess.check_output(["readelf", "-d", str(path)], text=True)
        for search in re.findall(r"(?:RPATH|RUNPATH).*?\[(.*?)\]", dynamic):
            if any(entry not in {"$ORIGIN", "${ORIGIN}"} for entry in search.split(":")):
                raise RuntimeError(f"Non-relocatable library search path in {path}: {search}. Rebuild the native runtime.")
        for dependency in dependencies(path):
            if not BASELINE.fullmatch(dependency.name) and not dependency.resolve().is_relative_to(folder):
                raise RuntimeError(f"{path.name} uses {dependency} outside its bundle. Rebuild and restage the runtime.")


def check_qt():
    from PyQt6.QtCore import QLibraryInfo
    plugins = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
    try:
        dependencies(plugins / "platforms/libqxcb.so", library_path=os.environ.get("LD_LIBRARY_PATH"))
    except RuntimeError as exc:
        raise RuntimeError(f"{exc}\nInstall the Linux GUI prerequisites in desktop/BUILD_LINUX.md "
                           "(including libxcb-cursor0 on Ubuntu).") from exc


def glibc_requirement(bundle):
    """Record the actual ELF requirements, including host libraries copied by Qt."""
    versions = set()
    for path in sorted(Path(bundle).rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        # Zig's compiler inputs are not dynamically loaded by the application.
        relative = path.relative_to(bundle)
        if relative.parts[:2] == ("compiled", "compiler") and relative.parts[2:] != ("zig",):
            continue
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
        result = subprocess.run(["readelf", "--version-info", str(path)], check=True, capture_output=True, text=True,
                                env={**os.environ, "LC_ALL": "C"})
        # Only required versions, not the definitions exported by a library.
        needs = result.stdout.partition("Version needs section")[2]
        versions.update(re.findall(r"Name: GLIBC_([0-9.]+)", needs))
    return max(versions, key=lambda value: tuple(map(int, value.split("."))), default=None)
