"""Publication is atomic across contention, interrupted writers and cleanup errors."""

from pathlib import Path
import os

import pytest

from packed_bed import file_io


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_windows_errors_retry_without_repeating_writer(tmp_path, monkeypatch, code):
    destination = tmp_path / "output"
    destination.write_text("old")
    replace = Path.replace
    attempts, writes, delays = [], [], []

    def locked_once(source, target):
        attempts.append(source)
        if len(attempts) == 1:
            error = PermissionError("reader still open")
            error.winerror = code
            raise error
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", locked_once)
    monkeypatch.setattr(file_io, "sleep", delays.append)
    with file_io.atomic_output(destination) as temporary:
        writes.append(temporary)
        temporary.write_text("new")
        assert destination.read_text() == "old"
    assert destination.read_text() == "new"
    assert len(writes) == len(delays) == 1
    assert attempts == writes * 2
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("error", [OSError("disk full"), KeyboardInterrupt()])
def test_interrupted_writer_keeps_previous_output_and_cleans_partial(tmp_path, error):
    destination = tmp_path / "output"
    destination.write_text("old")
    with pytest.raises(type(error)):
        with file_io.atomic_output(destination) as temporary:
            temporary.write_text("partial")
            raise error
    assert destination.read_text() == "old"
    assert list(tmp_path.iterdir()) == [destination]


def test_overlapping_writers_own_distinct_temporary_files(tmp_path):
    destination = tmp_path / "output"
    with file_io.atomic_output(destination) as first:
        first.write_text("first")
        with file_io.atomic_output(destination) as second:
            assert second != first
            second.write_text("second")
        assert destination.read_text() == "second"
        assert first.read_text() == "first"
    assert destination.read_text() == "first"
    assert list(tmp_path.iterdir()) == [destination]


def test_cancellation_while_waiting_does_not_publish(tmp_path, monkeypatch):
    destination = tmp_path / "output"
    destination.write_text("old")
    cancelled = False

    def locked(source, target):
        error = PermissionError("locked")
        error.winerror = 32
        raise error

    def cancel(_delay):
        nonlocal cancelled
        cancelled = True

    def check():
        if cancelled:
            raise InterruptedError("cancelled")

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(file_io, "sleep", cancel)
    with pytest.raises(InterruptedError, match="cancelled"):
        with file_io.atomic_output(destination, before_replace=check) as temporary:
            temporary.write_text("new")
    assert destination.read_text() == "old"
    assert list(tmp_path.iterdir()) == [destination]


def test_cleanup_error_does_not_mask_original_error(tmp_path, monkeypatch, caplog):
    destination = tmp_path / "output"

    def deny_delete(*args, **kwargs):
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(Path, "unlink", deny_delete)
    with pytest.raises(ValueError, match="original writer error"):
        with file_io.atomic_output(destination) as temporary:
            temporary.write_text("partial")
            raise ValueError("original writer error")
    assert not destination.exists()
    assert "cleanup denied" in caplog.text


@pytest.mark.skipif(os.name != "nt", reason="Windows readers block scratch cleanup")
@pytest.mark.parametrize("interrupted", [False, True])
def test_temporary_directory_waits_for_reader_and_preserves_interruption(tmp_path, monkeypatch, interrupted):
    delays = []
    reader = None

    def release(delay):
        delays.append(delay)
        reader.close()

    monkeypatch.setattr(file_io, "sleep", release)

    def work():
        nonlocal reader
        with file_io.TemporaryDirectory(prefix="compile-", dir=tmp_path) as directory:
            log = Path(directory) / "compiler.log"
            log.write_text("compiler output")
            reader = log.open()
            if interrupted:
                raise InterruptedError("compilation cancelled")

    try:
        if interrupted:
            with pytest.raises(InterruptedError, match="compilation cancelled"):
                work()
        else:
            work()
    finally:
        if reader:
            reader.close()
    assert len(delays) == 1
    assert not list(tmp_path.iterdir())
