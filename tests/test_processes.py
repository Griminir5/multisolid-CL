"""A stopped worker/coordinator must not leave its compiler children running."""

import ctypes
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from packed_bed.processes import worker_entry, run_compiler
from packed_bed.batch import _terminate_process
from packed_bed.compiled.cache import build_events


def child_command(pid_path):
    return [sys.executable, "-c", "import os,sys,time; from pathlib import Path; "
            "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)", str(pid_path)]


def fake_worker(pid_path):
    subprocess.Popen(child_command(pid_path)).wait()


def coordinator(pid_path):
    process = mp.get_context("spawn").Process(target=worker_entry, args=(fake_worker, (pid_path,)))
    process.start()
    process.join()


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    pytest.fail("Process did not reach the expected state")


def alive(pid):
    if os.name != "nt":
        try:
            return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
        except FileNotFoundError:
            return False
    from ctypes import wintypes as W
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
    kernel.OpenProcess.restype = W.HANDLE
    kernel.GetExitCodeProcess.argtypes = [W.HANDLE, ctypes.POINTER(W.DWORD)]
    kernel.CloseHandle.argtypes = [W.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = W.DWORD()
        assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


@pytest.mark.skipif(sys.platform not in {"linux", "win32"}, reason="Desktop process containment platforms")
@pytest.mark.parametrize("parent_dies", [False, True])
def test_worker_tree_stops(tmp_path, parent_dies):
    pid_path = tmp_path / "child.pid"
    target, args = ((coordinator, (pid_path,)) if parent_dies else (worker_entry, (fake_worker, (pid_path,))))
    process = mp.get_context("spawn").Process(target=target, args=args)
    process.start()
    try:
        wait_for(lambda: pid_path.is_file() and pid_path.read_text().isdigit())
        pid = int(pid_path.read_text())
        assert alive(pid)
        if parent_dies:
            process.kill()
            process.join(10)
        else:
            _terminate_process(process)
        wait_for(lambda: not alive(pid))
    finally:
        if process.is_alive():
            process.kill()
        process.join(10)
        process.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Direct CLI compiler session; Windows uses worker Jobs")
def test_compiler_cancellation_reaps_child(tmp_path):
    pid_path = tmp_path / "compiler.pid"
    with build_events(cancel_requested=pid_path.is_file):
        with pytest.raises(InterruptedError):
            run_compiler(child_command(pid_path), cwd=tmp_path, env=os.environ.copy())
    assert not alive(int(pid_path.read_text()))
