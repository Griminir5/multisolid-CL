"""Stage the managed compiler/runtime beside a folder-based application bundle."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packed_bed.compiled.bundle import engine_fingerprint, file_hash

MODULES = ("core", "ida", "nvecserial", "sunmatrixsparse", "sunmatrixband",
           "sunlinsolband", "sunlinsolklu", "sunlinsolsuperlumt")


def stage(compiler, runtime, destination):
    compiler, runtime, destination = (Path(p).resolve() for p in (compiler, runtime, destination))
    if destination.exists():
        raise FileExistsError(destination)
    system = platform.system()
    if system not in {"Linux", "Windows"}:
        raise ValueError("Managed bundles support Linux and Windows x64.")
    executable = "zig.exe" if system == "Windows" else "zig"
    version = subprocess.check_output([str(compiler / executable), "version"], text=True).strip()
    if version != "0.16.0":
        raise ValueError("Expected the pinned Zig 0.16.0 distribution.")
    if not (runtime / "sources.json").is_file() or not (runtime / "licenses").is_dir():
        raise ValueError("Build the runtime with build_compiled_runtime first; sources and notices are required.")
    config = (runtime / "include/sundials/sundials_config.h").read_text()
    for declaration in ('SUNDIALS_VERSION "7.5.0"', 'SUNDIALS_DOUBLE_PRECISION 1', 'SUNDIALS_INT32_T 1'):
        if not re.search(r"^#define " + re.escape(declaration) + r"$", config, re.MULTILINE):
            raise ValueError("The native runtime has an incompatible SUNDIALS ABI: " + declaration)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compiled-stage-", dir=destination.parent) as temporary:
        root = Path(temporary) / "compiled"
        shutil.copytree(compiler, root / "compiler")
        shutil.copytree(runtime / "licenses", root / "licenses")
        shutil.copy2(runtime / "sources.json", root / "sources.json")
        lib = root / "lib"
        lib.mkdir()
        for directory in (runtime / "lib", runtime / "bin"):
            for path in directory.glob("*.dll" if system == "Windows" else "*.so*"):
                shutil.copy2(path, lib / path.name)
        if system == "Linux":
            baseline = re.compile(r"(?:ld-linux.*|lib(?:c|m|mvec|dl|pthread|rt|resolv|util)\.so\..*)$")
            for path in list(lib.iterdir()):
                result = subprocess.run(["ldd", str(path)], capture_output=True, text=True,
                                        env={**os.environ, "LD_LIBRARY_PATH": str(lib)}, check=True)
                if "not found" in result.stdout:
                    raise RuntimeError(result.stdout)
                for dependency in re.findall(r"=>\s+(/\S+)", result.stdout):
                    dependency = Path(dependency)
                    if not baseline.fullmatch(dependency.name) and not (lib / dependency.name).exists():
                        shutil.copy2(dependency, lib / dependency.name)
            # Include the build host's runtime notices and source-package identities.
            dependencies = {}
            for path in lib.iterdir():
                if not any((runtime / folder / path.name).exists() for folder in ("bin", "lib")):
                    dependencies[path.name] = file_hash(path)
            (root / "native-dependencies.json").write_text(json.dumps(dependencies, indent=2))
            for folder in Path("/usr/share/doc").glob("gcc-*-base"):
                notice = folder / "copyright"
                if notice.is_file():
                    shutil.copy2(notice, root / "licenses" / (folder.name + ".copyright"))
        else:
            # Runtime DLLs supplied by the release toolchain travel with the bundle.
            import pefile
            todo, seen = list(lib.glob("*.dll")), set()
            dependencies, notice_roots = {}, set()
            system_dir = Path(os.environ["SystemRoot"]) / "System32"
            while todo:
                path = todo.pop()
                if path.name.lower() in seen:
                    continue
                seen.add(path.name.lower())
                with pefile.PE(str(path)) as pe:
                    imports = [entry.dll.decode() for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", ())]
                for name in imports:
                    if (lib / name).exists() or name.lower().startswith(("api-ms-", "ext-ms-")):
                        continue
                    candidates = [runtime / "bin" / name, *[Path(p) / name for p in os.environ.get("PATH", "").split(os.pathsep)]]
                    found = next((p for p in candidates if p.is_file() and not p.is_relative_to(system_dir)), None)
                    if found:
                        shutil.copy2(found, lib / name)
                        todo.append(lib / name)
                        dependencies[name] = file_hash(found)
                        notices = found.parent.parent / "share/licenses"
                        if notices.is_dir():
                            notice_roots.add(notices)
                    elif not (system_dir / name).is_file():
                        raise RuntimeError(f"Missing runtime DLL: {name}")
            if dependencies and not notice_roots:
                raise RuntimeError("Release toolchain notices are missing (expected MSYS2 share/licenses).")
            for index, notices in enumerate(sorted(notice_roots)):
                shutil.copytree(notices, root / "licenses" / f"toolchain-{index}")
            (root / "native-dependencies.json").write_text(json.dumps(dependencies, indent=2))
        libraries = {}
        for module in MODULES:
            # MSVC and MinGW use different prefixes (and sometimes ABI suffixes).
            pattern = (rf"(?:lib)?sundials_{module}(?:[-.]\d+)*\.dll" if system == "Windows"
                       else rf"libsundials_{module}\.so")
            matches = sorted(path for path in lib.iterdir() if re.fullmatch(pattern, path.name))
            if len(matches) != 1:
                raise ValueError(f"Expected one {module} library; found {matches}")
            libraries[module] = "lib/" + matches[0].name
        files = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}
        data = {"format_version": 1, "platform": system, "architecture": "x86_64",
                "compiler": {"version": version, "executable": "compiler/" + executable,
                             "target": "x86_64-windows-gnu" if system == "Windows" else "x86_64-linux-gnu.2.35"},
                "runtime": {"version": "7.5.0", "precision": "double", "index_bits": 32, "libraries": libraries},
                "engine_sha256": engine_fingerprint(), "files": files,
                "bundle_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
        (root / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
        environment = {**os.environ, "MULTISOLID_COMPILED_BUNDLE": str(root), "PATH": str(root / "compiler")}
        subprocess.run([sys.executable, "-m", "packed_bed.compiled.smoke"], env=environment, check=True)
        root.rename(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(stage(args.compiler, args.runtime, args.destination))
