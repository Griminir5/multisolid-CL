"""Atomic file publication and bounded retries for Windows reader locks.

Callers still own serialization, transaction ordering, and writer coordination.
Only retry operations that are safe to repeat after a failed filesystem call;
never retry a whole simulation, export, or multi-file transaction here.
"""

from contextlib import contextmanager
import logging
from pathlib import Path
from tempfile import TemporaryDirectory as _TemporaryDirectory
from time import sleep
from uuid import uuid4


_RETRY_DELAYS_S = (0.05, 0.1, 0.2, 0.4)
_LOGGER = logging.getLogger(__name__)


def is_windows_file_lock(error):
    return isinstance(error, PermissionError) and getattr(error, "winerror", None) in (5, 32, 33)


def retry_file_operation(operation, *args, **kwargs):
    """Wait at most 0.75 s for a reader/scanner; propagate persistent errors."""
    for attempt in range(len(_RETRY_DELAYS_S) + 1):
        try:
            return operation(*args, **kwargs)
        except OSError as exc:
            if not is_windows_file_lock(exc) or attempt == len(_RETRY_DELAYS_S):
                raise
            sleep(_RETRY_DELAYS_S[attempt])


@contextmanager
def atomic_output(path, *, before_replace=None):
    """Yield a unique sibling path; publish only after the writer closes it.

The optional check runs before every publication attempt, allowing exports to
honour cancellation even while waiting for a reader to release the destination.
"""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        yield temporary

        def publish():
            if before_replace is not None:
                before_replace()
            temporary.replace(path)

        retry_file_operation(publish)
    except BaseException:
        try:
            retry_file_operation(temporary.unlink, missing_ok=True)
        except OSError as exc:
            # Cleanup must not hide the original write error or cancellation.
            _LOGGER.warning("Could not remove temporary file %s: %s", temporary, exc)
        raise


def write_text(path, text, *, encoding="utf-8"):
    with atomic_output(path) as temporary:
        temporary.write_text(text, encoding=encoding)


class TemporaryDirectory(_TemporaryDirectory):
    """Scratch space with the same cleanup policy as published files."""

    def cleanup(self):
        retry_file_operation(super().cleanup)

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.cleanup()
            return
        try:
            self.cleanup()
        except OSError as error:
            _LOGGER.warning("Could not remove temporary directory %s: %s", self.name, error)
