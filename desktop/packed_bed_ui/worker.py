"""Run prepared cases in the engine's shared process scheduler, outside the GUI process."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import signal
import sys
from time import perf_counter
import traceback

from packed_bed.batch import BatchCaseRecord, run_cases_in_processes
from packed_bed.config import load_case

from .project import SNAPSHOT_VERSION, REFERENCES, input_hashes, read_json, require_desktop_solver, write_json


def verify_snapshot(folder: Path):
    metadata = read_json(folder / "snapshot.json")
    if metadata.get("format_version") != SNAPSHOT_VERSION:
        raise ValueError("Unsupported snapshot format.")
    if input_hashes(folder / "inputs") != metadata["input_hashes"]:
        raise ValueError("Snapshot inputs have changed. Run the case again from the project.")
    case = load_case(folder / "inputs" / "run.yaml")
    if case.run.references.model_dump() != REFERENCES:
        raise ValueError("Snapshot references must stay within its inputs folder.")
    for name in ("run", "chemistry", "program", "solids"):
        if not getattr(case, f"{name}_path").is_relative_to(folder / "inputs"):
            raise ValueError("Snapshot inputs must stay within the run folder.")
    if case.output_directory != folder / "output" or case.artifacts_directory != folder / "output" / "artifacts":
        raise ValueError("Snapshot outputs must stay within the run folder.")
    if metadata.get("extensions"):
        raise ValueError("Extension loading is not supported in this starter.")
    require_desktop_solver(case)
    return case


def activate_snapshot(pending: Path) -> Path:
    """Replace the one run slot after validation; roll back a failed directory swap."""
    verify_snapshot(pending)
    folder = pending.parent / "run"
    previous = pending.parent / ".previous-run"
    if previous.exists():
        raise ValueError("An interrupted replacement needs recovery. Reopen the project first.")
    if folder.exists():
        folder.rename(previous)
    try:
        pending.rename(folder)
    except Exception:
        if previous.exists():
            previous.rename(folder)
        raise
    if previous.exists():
        shutil.rmtree(previous)
    return folder


@contextmanager
def diagnostic_log(path: Path):
    """Capture both Python and native solver output in this case's latest log."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    try:
        with path.open("wb") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            if stream is not None:
                stream.flush()
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)


def run_snapshot(folder: str | Path, on_status=None) -> int:
    """Execute a snapshot once; preparation creates a fresh snapshot on each rerun."""
    folder = Path(folder).resolve()
    try:
        with (folder / ".started").open("x", encoding="utf-8"):
            pass
    except OSError:
        traceback.print_exc()
        return 1
    started = perf_counter()

    def status(state, **details):
        value = {"state": state, "elapsed_s": perf_counter() - started, "started_at": started, **details}
        write_json(folder / "status.json", value)
        if on_status is not None:
            on_status(value)

    with diagnostic_log(folder / "worker.log"):
        try:
            status("preparing")
            case = verify_snapshot(folder)
            from packed_bed.simulation import run_case

            result = run_case(case, on_status=status)
            status("completed", plot_errors=result.plot_errors)
            return 0
        except Exception as exc:
            traceback.print_exc()
            status("failed", message=str(exc))
            return 1


def run_prepared_case(run_yaml_path, generate_artifacts_fn, run_case_fn, result_queue):
    """Spawn target: replace and execute only this case's latest-run slot."""
    try:
        folder = activate_snapshot(Path(run_yaml_path).parent.parent)
        if run_snapshot(folder):
            result_queue.put(read_json(folder / "status.json").get("message", "Simulation failed."))
        else:
            result_queue.put((folder / "output", {}, "not_requested", {}))
    except Exception as exc:
        result_queue.put(str(exc))


def run_project_job(path: str | Path, *, case_worker=run_prepared_case) -> int:
    from PyQt6.QtCore import QLockFile

    path = Path(path).resolve()
    lock = QLockFile(str(path.parent / ".solver.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        raise ValueError("Another simulation worker is already using this project.")
    try:
        return _execute_project_job(path, case_worker)
    finally:
        lock.unlock()


def _execute_project_job(path, case_worker):
    job = read_json(path)
    attempt_id = job["attempt_id"]
    if not re.fullmatch(r"[a-f0-9]{32}", attempt_id) or job.get("state") != "queued":
        raise ValueError("This execution has already started or has an invalid identity.")
    started = perf_counter()
    records, cases = [], []
    for case_id in job["cases"]:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", case_id):
            raise ValueError("Invalid case identity.")
        root = (path.parent / "cases" / case_id).resolve()
        if not root.is_relative_to(path.parent):
            raise ValueError("Case folders must stay inside this project.")
        pending = root / f".pending-{attempt_id}"
        snapshot = read_json(pending / "snapshot.json")
        if snapshot.get("case_id") != case_id or snapshot.get("attempt_id") != attempt_id:
            raise ValueError("Prepared inputs do not match this execution.")
        cases.append(verify_snapshot(pending))
        records.append(BatchCaseRecord(case_id, {}, root, pending / "inputs/run.yaml"))
    records = tuple(records)
    job["state"] = "running"
    cancelled = False

    def cancel_signal(*_):
        nonlocal cancelled
        cancelled = True

    def cancel_requested():
        return cancelled or (path.parent / f".cancel-{attempt_id}").exists()

    def checkpoint():
        for record in records:
            value = job["cases"][record.case_id]
            folder = record.case_directory / "run"
            current = False
            try:
                current = read_json(folder / "snapshot.json").get("attempt_id") == attempt_id
                if current:
                    value.update(read_json(folder / "status.json"))
            except (OSError, ValueError):
                pass
            if record.status == "running" and value["state"] == "queued":
                value.update(state="preparing", elapsed_s=record.runtime_s)
            elif record.status in ("success", "simulation_failed", "cancelled"):
                value.update(state={"success": "completed", "simulation_failed": "failed",
                                    "cancelled": "cancelled"}[record.status],
                             elapsed_s=value.get("elapsed_s", record.runtime_s) if record.status == "success" else (record.runtime_s or 0.0),
                             message=record.error)
                if current:
                    write_json(folder / "status.json", value)
        job["elapsed_s"] = perf_counter() - started
        write_json(path, job)

    handlers = {sig: signal.signal(sig, cancel_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        checkpoint()
        run_cases_in_processes(tuple(cases), records, workers=job.get("max_workers", 1),
                               timeout_s=None, generate_artifacts_fn=None, run_case_fn=None,
                               checkpoint=checkpoint, case_worker=case_worker,
                               cancel_requested=cancel_requested)
        job["state"] = ("cancelled" if any(r.status == "cancelled" for r in records) else
                        "completed" if all(r.status == "success" for r in records) else "failed")
        checkpoint()
        return 0 if job["state"] in ("completed", "cancelled") else 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        for record in records:
            previous = record.case_directory / ".previous-run"
            folder = record.case_directory / "run"
            if previous.exists():
                if folder.exists():
                    shutil.rmtree(previous)
                else:
                    previous.rename(folder)
            pending = record.run_yaml_path.parent.parent
            if pending.exists():
                shutil.rmtree(pending)
        (path.parent / f".cancel-{attempt_id}").unlink(missing_ok=True)
