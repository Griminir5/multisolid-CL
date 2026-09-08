"""Exercise native artifacts and wheel discovery across supported platforms."""

import ctypes as C
from pathlib import Path
from types import SimpleNamespace

import pytest

from packed_bed.compiled import compiler, runtime


@pytest.fixture
def cpp_toolchain():
    try:
        compiler.find_toolchain()
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_native_exports_cache_invalidation_and_failure(cpp_toolchain, tmp_path, monkeypatch):
    # Spaces exercise argument handling; no solver runtime is needed for kernels.
    cache = tmp_path / "cache with spaces"
    source = "PB_EXPORT double square(double x) { return x*x; }"
    path, first = compiler.compile_kernel(source, cache)
    assert path.suffix == compiler.library_suffix()
    library = C.CDLL(str(path))
    library.square.argtypes, library.square.restype = [C.c_double], C.c_double
    assert library.square(3.5) == 12.25
    again, second = compiler.compile_kernel(source, cache)
    assert again == path
    assert not first["cache_hit"] and second["cache_hit"]

    changed, _ = compiler.compile_kernel(source.replace("x*x", "x+x"), cache)
    assert changed != path
    toolchain, version = compiler.compiler_identity()
    monkeypatch.setattr(compiler, "compiler_identity", lambda: (toolchain, version + "-new"))
    updated, _ = compiler.compile_kernel(source, cache)
    assert updated != path
    monkeypatch.setattr(compiler, "platform_identity", lambda: "another-platform")
    other_platform, _ = compiler.compile_kernel(source, cache)
    assert other_platform not in {path, updated}
    with pytest.raises(RuntimeError, match="compilation failed"):
        compiler.compile_kernel("this is not C++", cache)
    assert not list(cache.glob("compile-*"))


@pytest.mark.parametrize("system,suffix,link", (
    ("Windows", ".dll", "/LD"),
    ("Linux", ".so", "-shared"),
    ("Darwin", ".dylib", "-dynamiclib"),
))
def test_platform_build_commands(monkeypatch, tmp_path, system, suffix, link):
    monkeypatch.setattr(compiler.platform, "system", lambda: system)
    monkeypatch.setattr(compiler, "compiler_identity", lambda: (Path("compiler path"), "v1"))
    commands = []

    def run(command, *, cwd, **kwargs):
        commands.append(command)
        source = (cwd / "kernel.cpp").read_text()
        assert "#define PB_EXPORT" in source
        if system == "Windows":
            script = (cwd / "compile.cmd").read_text()
            assert link in script and "/arch:AVX2" in script
            assert kwargs["env"]["PACKED_BED_VCVARS"] == "compiler path"
        else:
            assert command[0] == "compiler path"
            assert link in command and "-mavx2" in command and "-mfma" in command
            assert "-ffp-contract=off" in command
            assert not any(arg.startswith("/arch") for arg in command)
        (cwd / ("kernel" + suffix)).write_bytes(b"native library placeholder")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(compiler.subprocess, "run", run)
    library, metadata = compiler.compile_kernel(
        "PB_EXPORT void entry() {}", tmp_path,
        extra_flags=compiler.simd_flags(avx2=True, fma=True),
    )
    assert library.suffix == suffix
    assert not metadata["cache_hit"] and len(commands) == 1


def test_cxx_override_and_missing_compiler(monkeypatch, tmp_path):
    monkeypatch.setattr(compiler.platform, "system", lambda: "Linux")
    # The .exe suffix also lets shutil.which exercise this on Windows hosts.
    executable = tmp_path / "my compiler.exe"
    executable.write_text("")
    executable.chmod(0o755)
    monkeypatch.setenv("CXX", str(executable))
    assert compiler.find_toolchain() == executable
    monkeypatch.setenv("CXX", str(tmp_path / "missing compiler"))
    with pytest.raises(RuntimeError, match="set CXX"):
        compiler.find_toolchain()


@pytest.mark.parametrize("system,filename", (
    ("Windows", "sundials_core-123abc.dll"),
    ("Linux", "libsundials_core-123abc.so.7.5.0"),
    ("Darwin", "libsundials_core.7.5.0.dylib"),
))
def test_wheel_library_discovery(monkeypatch, tmp_path, system, filename):
    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    expected = tmp_path / filename
    expected.touch()
    # Other components must not be mistaken for this module.
    (tmp_path / filename.replace("core", "ida")).touch()
    monkeypatch.setattr(runtime.C, "CDLL", lambda path: path)
    assert runtime.load_runtime_library(tmp_path, "core") == str(expected)
    with pytest.raises(RuntimeError, match="found 0"):
        runtime.load_runtime_library(tmp_path, "sunmatrixband")


@pytest.mark.parametrize("system", ("Windows", "Linux", "Darwin"))
@pytest.mark.parametrize("bad_config", (None, "index", "precision", "missing", "version"))
def test_runtime_checks_abi_and_wheel_layout(monkeypatch, tmp_path, system, bad_config):
    import sys

    package = tmp_path / "sksundae"
    package.mkdir()
    folder = package / ".dylibs" if system == "Darwin" else tmp_path / "scikit_sundae.libs"
    folder.mkdir()
    module = SimpleNamespace(
        __file__=str(package / "__init__.py"), __version__="1.1.3", SUNDIALS_VERSION="7.5.0",
    )
    config = 'SUNDIALS_FLOAT_TYPE = "double"\nSUNDIALS_INT_TYPE = "int"\n'
    if bad_config == "index":
        config = config.replace('"int"', '"long int"')
    if bad_config == "precision":
        config = config.replace('"double"', '"float"')
    if bad_config != "missing":
        (package / "py_config.pxi").write_text(config)
    if bad_config == "version":
        module.SUNDIALS_VERSION = "7.4.0"
    monkeypatch.setitem(sys.modules, "sksundae", module)
    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    if bad_config:
        with pytest.raises(RuntimeError, match="ABI|requires"):
            runtime.check_runtime()
    else:
        assert runtime.check_runtime() == folder
        folder.rmdir()
        with pytest.raises(RuntimeError, match="Bundled SUNDIALS libraries not found"):
            runtime.check_runtime()


def test_non_x86_uses_scalar_without_compiling_a_probe(monkeypatch):
    from packed_bed.compiled import band

    monkeypatch.setattr(band.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(band, "compile_kernel", lambda *a: pytest.fail("Unexpected CPU probe"))
    assert not band.supports_avx2.__wrapped__()
