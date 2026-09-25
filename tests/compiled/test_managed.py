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


def test_legacy_klu_and_new_name_are_accepted():
    from packed_bed.config import load_case
    from packed_bed.config.models import RunConfig
    config = load_case(Path(__file__).resolve().parents[2] / "packed_bed/examples/default_case/run.yaml").run.model_dump()
    for backend in ("daetools", "compiled"):
        for name in ("klu", "trilinos_klu"):
            config["solver"].update(backend=backend, name=name)
            assert RunConfig.model_validate(config).solver.name == name


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
