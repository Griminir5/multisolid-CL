"""Build and cache native C++ kernels with MSVC, GCC or Clang."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter


def library_suffix() -> str:
    system = platform.system()
    try:
        return {"Windows": ".dll", "Linux": ".so", "Darwin": ".dylib"}[system]
    except KeyError:
        raise RuntimeError(f"Unsupported compiled-backend platform: {system}.") from None


def platform_identity() -> str:
    """Prevent shared caches from mixing operating systems or CPU architectures."""
    return f"{platform.system()}-{platform.machine().lower()}"


def simd_flags(*, avx2=False, fma=False) -> tuple[str, ...]:
    if platform.system() == "Windows":
        return ("/arch:AVX2",) if avx2 or fma else ()
    return (("-mavx2",) if avx2 else ()) + (("-mfma",) if fma else ())


def compiler_flags() -> tuple[str, ...]:
    if platform.system() == "Windows":
        return ("/nologo", "/std:c++17", "/O2", "/fp:precise", "/EHsc", "/LD")
    # Explicit FMA intrinsics remain available, but contracting ordinary
    # expressions changes the scalar/reference arithmetic and Newton trajectory.
    link = "-dynamiclib" if platform.system() == "Darwin" else "-shared"
    # Keep libm's argument order for equal signed zeros in scalar and SIMD lanes.
    return (
        "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
        "-fno-builtin-fmin", "-fno-builtin-fmax", "-fPIC", link,
    )


def find_toolchain() -> Path:
    library_suffix()  # Validate the platform before looking for a compiler.
    if platform.system() != "Windows":
        requested = os.environ.get("CXX")
        for candidate in ((requested,) if requested else ("c++", "g++", "clang++")):
            executable = shutil.which(candidate)
            if executable:
                return Path(executable).absolute()
        raise RuntimeError(
            "The compiled backend needs a C++17 compiler. Install GCC or Clang "
            "(Xcode Command Line Tools on macOS), or set CXX to the compiler executable."
        )
    if platform.machine().lower() not in {
        "amd64",
        "x86_64",
    }:
        raise RuntimeError("The compiled backend needs Windows x64 for the MSVC toolchain.")
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
    if platform.system() != "Windows":
        result = subprocess.run(
            [str(toolchain), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return toolchain, result.stdout.strip()
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
    flags = (*compiler_flags(), *extra_flags)
    # Embed this header so generated sources are self-contained and its changes
    # invalidate binary caches as well as model caches.
    header = Path(__file__).with_name("portable.hpp").read_text(encoding="utf-8")
    source = header + "\n" + source
    digest = hashlib.sha256(
        (platform_identity() + str(toolchain) + version + repr(flags) + source).encode()
    ).hexdigest()
    cache_directory = cache_directory.resolve()
    cache_directory.mkdir(parents=True, exist_ok=True)
    suffix = library_suffix()
    library = cache_directory / f"{digest}{suffix}"
    if library.is_file():
        return library, {"cache_hit": True, "compile_s": 0.0, "kernel_sha256": digest}
    started = perf_counter()
    # Independent temporary directories let concurrent batch workers compile safely.
    with tempfile.TemporaryDirectory(
        prefix="compile-", dir=cache_directory
    ) as temporary:
        directory = Path(temporary)
        (directory / "kernel.cpp").write_text(source, encoding="utf-8")
        output = "kernel" + suffix
        if platform.system() == "Windows":
            (directory / "compile.cmd").write_text(
                '@echo off\ncall "%PACKED_BED_VCVARS%" >nul\n'
                "if errorlevel 1 exit /b 1\n"
                f"cl {' '.join(flags)} kernel.cpp /link /OUT:{output}\n",
                encoding="utf-8",
            )
            command = ["cmd.exe", "/d", "/c", "compile.cmd"]
            environment = {**os.environ, "PACKED_BED_VCVARS": str(toolchain)}
        else:
            command = [str(toolchain), *flags, "kernel.cpp", "-o", output]
            environment = None
        result = subprocess.run(
            command,
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
            (directory / output).replace(library)
        except OSError:
            # Another worker may already have published and loaded this library.
            if not library.is_file():
                raise
    return library, {
        "cache_hit": False,
        "compile_s": perf_counter() - started,
        "kernel_sha256": digest,
    }
