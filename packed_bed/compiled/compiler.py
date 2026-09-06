"""Build and cache model-specific C++ kernels with the Windows MSVC toolchain."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

FLAGS = ("/nologo", "/O2", "/fp:precise", "/EHsc", "/LD")


def find_toolchain() -> Path:
    if platform.system() != "Windows" or platform.machine().lower() not in {
        "amd64",
        "x86_64",
    }:
        raise RuntimeError("The compiled backend currently supports Windows x64 only.")
    installer = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
    vswhere = installer / "Microsoft Visual Studio/Installer/vswhere.exe"
    if vswhere.is_file():
        result = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            candidate = Path(result.stdout.strip()) / "VC/Auxiliary/Build/vcvars64.bat"
            if candidate.is_file():
                return candidate
    raise RuntimeError(
        "The compiled backend needs Visual Studio Build Tools with the C++ x64 toolchain. "
        "Install it, or select solver.backend: daetools."
    )


def compiler_identity() -> tuple[Path, str]:
    """Identify the installed toolchain for both model and binary cache keys."""
    toolchain = find_toolchain()
    version_file = (
        toolchain.parents[2] / "Auxiliary/Build/Microsoft.VCToolsVersion.default.txt"
    )
    version = (
        version_file.read_text().strip()
        if version_file.is_file()
        else str(toolchain.stat().st_mtime_ns)
    )
    return toolchain, version


def compile_kernel(
    source: str, cache_directory: Path, *, extra_flags=()
) -> tuple[Path, dict]:
    toolchain, version = compiler_identity()
    flags = (*FLAGS, *extra_flags)
    digest = hashlib.sha256(
        (str(toolchain) + version + repr(flags) + source).encode()
    ).hexdigest()
    cache_directory = cache_directory.resolve()
    cache_directory.mkdir(parents=True, exist_ok=True)
    library = cache_directory / f"{digest}.dll"
    if library.is_file():
        return library, {"cache_hit": True, "compile_s": 0.0, "kernel_sha256": digest}
    started = perf_counter()
    # Independent temporary directories let concurrent batch workers compile safely.
    with tempfile.TemporaryDirectory(
        prefix="compile-", dir=cache_directory
    ) as temporary:
        directory = Path(temporary)
        (directory / "kernel.cpp").write_text(source, encoding="utf-8")
        (directory / "compile.cmd").write_text(
            '@echo off\ncall "%PACKED_BED_VCVARS%" >nul\n'
            "if errorlevel 1 exit /b 1\n"
            f"cl {' '.join(flags)} kernel.cpp /link /OUT:kernel.dll\n",
            encoding="utf-8",
        )
        environment = {**os.environ, "PACKED_BED_VCVARS": str(toolchain)}
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "compile.cmd"],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"Model kernel compilation failed:\n{result.stdout}{result.stderr}"
            )
        try:
            (directory / "kernel.dll").replace(library)
        except OSError:
            # Another worker may already have published and loaded the same DLL.
            if not library.is_file():
                raise
    return library, {
        "cache_hit": False,
        "compile_s": perf_counter() - started,
        "kernel_sha256": digest,
    }
