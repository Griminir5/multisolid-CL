"""Qt owns one project worker; the engine owns the batch and its child processes."""

from pathlib import Path
import shutil
import sys
from time import perf_counter

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from .project import TERMINAL_STATES, read_json, write_json


class RunController(QObject):
    changed = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.poll = QTimer(self)
        self.poll.setInterval(250)
        self.poll.timeout.connect(self._read_status)
        self.path: Path | None = None
        self.active = False
        self.cancelling = False
        self.job = {}
        self.started = 0.0

    def start(self, path: Path) -> None:
        if self.active:
            raise ValueError("Cases are already running.")
        self.path = path
        self.job = read_json(path)
        self.active = True
        self.cancelling = False
        self.started = perf_counter()
        self.process.setStandardOutputFile(str(path.parent / "execution.log"))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(environment)
        args = ["--project-worker", str(path)]
        if not getattr(sys, "frozen", False):
            args = ["-m", "packed_bed_ui", *args]
        self.poll.start()
        self.process.start(sys.executable, args)
        self._read_status()

    def cancel(self) -> None:
        if self.active:
            # The coordinator stops and reaps all children before it exits.
            # Killing it directly could leave native solver workers alive.
            try:
                (self.path.parent / f".cancel-{self.job['attempt_id']}").touch()
                self.cancelling = True
            except OSError as exc:
                self.job["message"] = f"Could not request cancellation: {exc}"
            self.changed.emit(self.job)

    def _read_status(self) -> dict:
        try:
            self.job = read_json(self.path)
        except (OSError, ValueError):
            pass
        if self.active:
            now = perf_counter()
            self.job["elapsed_s"] = now - self.started
            for value in self.job.get("cases", {}).values():
                if value.get("state") not in TERMINAL_STATES and value.get("started_at") is not None:
                    value["elapsed_s"] = max(0.0, now - value["started_at"])
        self.changed.emit(self.job)
        return self.job

    def _error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._finished(-1, QProcess.ExitStatus.CrashExit)

    def _finished(self, exit_code, exit_status) -> None:
        if not self.active:
            return
        self.poll.stop()
        job = self._read_status()
        self.active = False
        for case_id, value in job["cases"].items():
            if value.get("state") not in TERMINAL_STATES:
                value.update(state="cancelled" if self.cancelling else "failed",
                             message="Stopped before completion." if self.cancelling else f"Worker exited ({exit_code}). See execution.log.")
            root = self.path.parent / "cases" / case_id
            # Do not replace the previous outcome of a case that never started.
            try:
                snapshot = read_json(root / "run/snapshot.json")
                if snapshot.get("attempt_id") == job["attempt_id"]:
                    status = read_json(root / "run/status.json")
                    if status.get("state") not in TERMINAL_STATES:
                        write_json(root / "run/status.json", value)
                pending = root / f".pending-{job['attempt_id']}"
                if pending.exists():
                    shutil.rmtree(pending)
            except FileNotFoundError:
                pending = root / f".pending-{job['attempt_id']}"
                if pending.exists():
                    shutil.rmtree(pending)
            except (OSError, ValueError) as exc:
                value["message"] = f"Could not finish saving execution status: {exc}"
        job["state"] = "cancelled" if self.cancelling or job.get("state") == "cancelled" else ("completed" if exit_code == 0 else "failed")
        try:
            write_json(self.path, job)
        except OSError as exc:
            job["message"] = f"Could not save execution status: {exc}"
        (self.path.parent / f".cancel-{job['attempt_id']}").unlink(missing_ok=True)
        self.changed.emit(job)
        self.finished.emit()
