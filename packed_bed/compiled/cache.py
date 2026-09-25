"""Atomic cache publication and cancellable native locks, independent of Qt."""

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

from filelock import FileLock, Timeout

from .bundle import file_hash

_callbacks = ContextVar("compiled_callbacks", default=(None, None))


@contextmanager
def build_events(on_progress=None, cancel_requested=None):
    token = _callbacks.set((on_progress, cancel_requested))
    try:
        yield
    finally:
        _callbacks.reset(token)


def check_cancelled():
    check = _callbacks.get()[1]
    if check is not None and check():
        raise InterruptedError("Execution cancelled.")


def progress(state, **details):
    check_cancelled()
    callback = _callbacks.get()[0]
    if callback is not None:
        callback({"state": state, **details})


@contextmanager
def cache_lock(directory, key):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(directory / (key + ".lock")))
    started = perf_counter()
    notified = False
    while True:
        check_cancelled()
        try:
            lock.acquire(timeout=0.1)
            break
        except Timeout:
            if not notified:
                progress("waiting_for_compilation", message="Waiting for another worker's compilation.")
                notified = True
    try:
        yield perf_counter() - started if notified else 0.0
    finally:
        lock.release()


def write_record(path, record):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix="record-",
                                     encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(record, stream, allow_nan=False)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def valid_library(path):
    try:
        record = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        return record["sha256"] == file_hash(path)
    except (OSError, ValueError, KeyError, TypeError):
        return False


@contextmanager
def build_directory(cache, key):
    """Caller holds key's lock; leftovers therefore have no active builder."""
    for path in cache.glob(key + "-tmp-*"):
        if path.is_dir():
            shutil.rmtree(path)
    with tempfile.TemporaryDirectory(prefix=key + "-tmp-", dir=cache) as directory:
        yield Path(directory)
