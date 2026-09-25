"""Contain solver workers and their compiler children on Linux and Windows."""

import ctypes as C
import os
import signal
import subprocess
from time import perf_counter

_job = None
_contained = False
worker_started_at = None


def contain_worker():
    global _job, _contained
    if _contained:
        return
    if os.name != "nt":
        os.setsid()
    else:
        from ctypes import wintypes as W

        class Basic(C.Structure):
            _fields_ = [("process_time", C.c_int64), ("job_time", C.c_int64),
                        ("flags", W.DWORD), ("min_working", C.c_size_t),
                        ("max_working", C.c_size_t), ("active", W.DWORD),
                        ("affinity", C.c_size_t), ("priority", W.DWORD), ("scheduling", W.DWORD)]

        class Extended(C.Structure):
            _fields_ = [("basic", Basic), ("io", C.c_uint64 * 6),
                        ("process_memory", C.c_size_t), ("job_memory", C.c_size_t),
                        ("peak_process", C.c_size_t), ("peak_job", C.c_size_t)]

        kernel = C.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]
        kernel.CreateJobObjectW.restype = W.HANDLE
        kernel.SetInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
        kernel.GetCurrentProcess.restype = W.HANDLE
        kernel.CloseHandle.argtypes = [W.HANDLE]
        handle = kernel.CreateJobObjectW(None, None)
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not handle:
            raise C.WinError(C.get_last_error())
        if (not kernel.SetInformationJobObject(handle, 9, C.byref(info), C.sizeof(info))
                or not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess())):
            error = C.get_last_error()
            kernel.CloseHandle(handle)
            raise C.WinError(error)
        # Non-inherited handle: process exit closes it and kills remaining children.
        _job = handle
    _contained = True


def worker_entry(target, args, started_at=None):
    global worker_started_at
    worker_started_at = perf_counter() if started_at is None else started_at
    contain_worker()
    # A killed coordinator cannot run its finally block. Its multiprocessing
    # sentinel still closes, so the child can stop its own process group/job.
    import multiprocessing as mp
    from multiprocessing.connection import wait
    from threading import Thread
    parent = mp.parent_process()
    if parent is not None:
        def parent_closed():
            wait([parent.sentinel])
            if os.name == "nt":
                os._exit(1)  # Closing the worker-owned Job handle stops descendants.
            else:
                kill_group(os.getpid(), signal.SIGKILL)
        Thread(target=parent_closed, daemon=True).start()
    target(*args)


def kill_group(pid, sig=signal.SIGTERM):
    if os.name == "nt":
        return
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def run_compiler(command, *, cwd, env, timeout=300):
    from .compiled.cache import check_cancelled

    if os.name == "nt" and not _contained:
        contain_worker()

    # File-backed output cannot fill a pipe while the caller polls cancellation.
    output = cwd / "compiler.log"
    with output.open("w+b") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=log,
                                   start_new_session=os.name != "nt" and not _contained,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        started = perf_counter()
        try:
            while process.poll() is None:
                check_cancelled()
                if perf_counter() - started > timeout:
                    raise TimeoutError(f"Compilation exceeded {timeout} seconds.")
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            if os.name != "nt" and not _contained:
                kill_group(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            raise
        finally:
            if os.name != "nt" and not _contained:
                kill_group(process.pid, signal.SIGKILL)
        log.seek(0)
        diagnostics = log.read().decode("utf-8", errors="replace")
    if process.returncode:
        raise RuntimeError(f"Model kernel compilation failed:\n{diagnostics}")
    return diagnostics
