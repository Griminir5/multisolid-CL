"""Cross-process cache publication, recovery, and cancellation contracts."""

import ctypes
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

import pytest

from packed_bed.compiled.cache import build_events, cache_lock, valid_library
from packed_bed.compiled.compiler import compile_kernel

SOURCE = 'PB_EXPORT double square(double x) { return x*x; }'


def build_shared(cache, queue):
    _, metadata = compile_kernel(SOURCE, Path(cache))
    queue.put(metadata)


def test_identical_workers_share_one_binary(native_tools, tmp_path):
    context = mp.get_context("spawn")
    queue = context.Queue()
    workers = [context.Process(target=build_shared, args=(str(tmp_path), queue)) for _ in range(3)]
    for worker in workers:
        worker.start()
    records = [queue.get(timeout=360) for _ in workers]
    for worker in workers:
        worker.join(10)
        assert worker.exitcode == 0
        worker.close()
    queue.close()
    assert sum(not record["cache_hit"] for record in records) == 1
    assert len({record["kernel_sha256"] for record in records}) == 1
    assert not list(tmp_path.glob("*-tmp-*"))


def test_binary_integrity_rebuilds_and_cache_can_move(native_tools, tmp_path):
    original = tmp_path / "original"
    path, metadata = compile_kernel(SOURCE, original)
    path.write_bytes(b"damaged library")
    assert not valid_library(path)
    repaired, rebuilt = compile_kernel(SOURCE, original)
    assert not rebuilt["cache_hit"] and valid_library(repaired)
    moved = tmp_path / "moved with spaces"
    original.rename(moved)
    path, hit = compile_kernel(SOURCE, moved)
    assert hit["cache_hit"]
    library = ctypes.CDLL(str(path))
    library.square.argtypes = [ctypes.c_double]
    library.square.restype = ctypes.c_double
    assert library.square(4) == 16


def test_waiting_for_lock_is_cancellable(tmp_path):
    events = []
    with cache_lock(tmp_path, "model-test"):
        start = time.monotonic()
        with build_events(events.append, lambda: time.monotonic() - start > .2):
            with pytest.raises(InterruptedError):
                with cache_lock(tmp_path, "model-test"):
                    pytest.fail("Second independent lock acquired")
    assert events[0]["state"] == "waiting_for_compilation"
    with cache_lock(tmp_path, "model-test"):
        pass


def test_cancelled_build_does_not_publish(native_tools, tmp_path):
    with build_events(cancel_requested=lambda: True):
        with pytest.raises(InterruptedError):
            compile_kernel(SOURCE, tmp_path)
    assert not list(tmp_path.glob("*.so")) and not list(tmp_path.glob("*.dll"))
    assert not list(tmp_path.glob("*-tmp-*"))


def test_klu_is_accepted_by_both_backends():
    from packed_bed.config import load_case
    from packed_bed.config.models import RunConfig
    config = load_case(Path(__file__).resolve().parents[2] / "packed_bed/examples/default_case/run.yaml").run.model_dump()
    for backend in ("daetools", "compiled"):
        config["solver"].update(backend=backend, name="klu")
        assert RunConfig.model_validate(config).solver.name == "klu"


def test_bundle_asset_integrity_and_containment(tmp_path, monkeypatch):
    from packed_bed.compiled import bundle
    path = tmp_path / "compiler"
    path.write_bytes(b"verified distribution")
    monkeypatch.setattr(bundle, "manifest", lambda root: {"files": {"compiler": bundle.file_hash(path)}})
    assert bundle.asset(tmp_path, "compiler") == path
    expected = bundle.file_hash(path)
    monkeypatch.setattr(bundle, "manifest", lambda root: {"files": {"compiler": expected}})
    path.write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="Damaged"):
        bundle.asset(tmp_path, "compiler")
    with pytest.raises(RuntimeError, match="invalid"):
        bundle.asset(tmp_path, "../outside")


def test_frozen_application_never_falls_back_to_host(tmp_path, monkeypatch):
    from packed_bed.compiled import bundle
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "MultiSolid"))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "_internal"), raising=False)
    with pytest.raises(RuntimeError, match="runtime is missing"):
        bundle.bundle_root()


def test_compile_in_deep_case_cache(native_tools, tmp_path):
    # round_027's cache is 179 characters; the old build name took it to 264.
    root = tmp_path.resolve()
    cache = root / ("x" * max(1, 180 - len(str(root)) - 1))
    cache.mkdir(parents=True, exist_ok=True)
    assert len(str(cache)) >= 180
    source = 'PB_EXPORT double deep_cache_square(double x) { return x*x; }'
    path, metadata = compile_kernel(source, cache)
    assert not metadata["cache_hit"]
    library = ctypes.CDLL(str(path))
    library.deep_cache_square.argtypes = [ctypes.c_double]
    library.deep_cache_square.restype = ctypes.c_double
    assert library.deep_cache_square(4) == 16
    cached_path, cached = compile_kernel(source, cache)
    assert cached_path == path and cached["cache_hit"]
    assert not list(cache.glob("*-tmp-*"))


def test_short_build_directory_recovers_leftovers_without_removing_other_builds(tmp_path):
    from packed_bed.compiled.cache import build_directory

    key = "binary-" + "a" * 64
    legacy = tmp_path / (key + "-tmp-abandoned")
    legacy.mkdir()
    with cache_lock(tmp_path, key), build_directory(tmp_path, key) as directory:
        assert not legacy.exists()
        abandoned = directory
    # Simulate a killed builder leaving files behind.
    abandoned.mkdir()
    (abandoned / "kernel.cpp").write_text("partial build")
    other_key = "binary-" + "b" * 64
    with cache_lock(tmp_path, other_key), build_directory(tmp_path, other_key) as other:
        with cache_lock(tmp_path, key), build_directory(tmp_path, key) as directory:
            assert not abandoned.exists()
            assert other.is_dir() and directory.is_dir()
        assert other.is_dir()
    assert not list(tmp_path.glob("*-tmp-*"))


def test_short_build_name_collision_waits_without_removing_live_build(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from packed_bed.compiled import cache

    # Force different full keys to choose the same short scratch name.
    monkeypatch.setattr(cache, "hashlib", SimpleNamespace(
        sha256=lambda value: SimpleNamespace(hexdigest=lambda: "0" * 64)))
    with cache_lock(tmp_path, "first"), cache.build_directory(tmp_path, "first") as directory:
        marker = directory / "kernel.cpp"
        marker.write_text("live build")
        started = time.monotonic()
        with cache_lock(tmp_path, "second"):
            with build_events(cancel_requested=lambda: time.monotonic() - started > .2):
                with pytest.raises(InterruptedError):
                    with cache.build_directory(tmp_path, "second"):
                        pytest.fail("Colliding build entered another key's scratch space")
        assert marker.read_text() == "live build"
    assert not list(tmp_path.glob("*-tmp-*"))
