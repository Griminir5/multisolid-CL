"""Compare specialized and shared programs on complete, isolated case runs.

Example: python tools/benchmark_shared_programs.py --cases PATH/TO/cases
  --output build/program-benchmark --workers 1 8 32 64

Each directory argument supplies its immediate */run.yaml children. Inputs are
read-only; datasets and manifests go under --output. Windows measurements use a
Job Object to include workers AND compiler descendants, with a memory guard.
Plots are disabled in both variants; numerical settings and reporting stay intact.
"""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_one(task):
    path, output, shared, rtol = task
    from functools import partial
    import packed_bed.compiled as compiled
    from packed_bed.config import load_case
    from packed_bed.simulation import run_case

    original = compiled.prepare_model
    compiled.prepare_model = partial(original, shared_programs=shared)
    try:
        case = load_case(path)
        solver = {"backend": "compiled", "threads": 1}
        if rtol is not None:
            solver["relative_tolerance"] = rtol
        case = replace(case, run=case.run.model_copy(update={
            "solver": case.run.solver.model_copy(update=solver),
            "outputs": case.run.outputs.model_copy(update={
                "directory": str(output), "artifacts_directory": str(Path(output) / "artifacts"),
                "requested_plots": ()})}))
        started = perf_counter()
        result = run_case(case)
        return {"case": str(path), "seconds": perf_counter()-started,
                "results": str(result.results_path), "stats": result.solver_stats}
    except Exception as exc:
        return {"case": str(path), "error": str(exc)}
    finally:
        compiled.prepare_model = original


def run_job(path):
    job = json.loads(path.read_text())
    from packed_bed.batch import BatchCaseRecord, run_cases_in_processes
    from packed_bed.config import load_case

    os.environ["PACKED_BED_COMPILED_CACHE"] = job["cache"]
    job["output"] = str(path.parent)
    started = perf_counter()
    cases, records = [], []
    for i, case_path in enumerate(job["cases"]):
        case = load_case(case_path)
        case = replace(case, run=case.run.model_copy(update={
            "solver": case.run.solver.model_copy(update={"threads": 1})}))
        cases.append(case)
        folder = path.parent / f"case-{i:03d}"
        records.append(BatchCaseRecord(f"case-{i:03d}", {}, folder, case.run_path))
    run_cases_in_processes(tuple(cases), tuple(records), workers=job["workers"], timeout_s=720,
                           generate_artifacts_fn=job, run_case_fn=None, checkpoint=lambda: None,
                           case_worker=benchmark_worker)
    rows = []
    for record in records:
        result_file = record.case_directory / "benchmark.json"
        rows.append(json.loads(result_file.read_text()) if result_file.exists() else
                    {"case": str(record.run_yaml_path), "error": record.error})
    result = {"seconds": perf_counter()-started, "workers": job["workers"], "shared": job["shared"], "cases": rows}
    result["distinct_kernels"] = len({r["stats"]["kernel_sha256"] for r in rows if "stats" in r})
    result["successful"] = sum("stats" in r for r in rows)
    (path.parent / "results.json").write_text(json.dumps(result, indent=2))
    return 0 if result["successful"] == len(rows) else 1


def benchmark_worker(run_path, job, unused, queue):
    index = job["cases"].index(str(run_path))
    folder = Path(job["output"]) / f"case-{index:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    row = run_one((run_path, folder, job["shared"], job.get("rtol")))
    (folder / "benchmark.json").write_text(json.dumps(row, indent=2))
    queue.put(row["error"] if "error" in row else (folder, {}, "not_requested", {}))


def measured_process(command, log_path):
    if os.name != "nt":
        with log_path.open("wb") as log:
            result = subprocess.run(command, stdout=log, stderr=log)
        return {"returncode": result.returncode, "peak_commit_bytes": None}
    import ctypes as C
    from ctypes import wintypes as W

    class Basic(C.Structure):
        _fields_ = [("process_time", C.c_int64), ("job_time", C.c_int64), ("flags", W.DWORD),
                    ("min_working", C.c_size_t), ("max_working", C.c_size_t), ("active", W.DWORD),
                    ("affinity", C.c_size_t), ("priority", W.DWORD), ("scheduling", W.DWORD)]

    class Extended(C.Structure):
        _fields_ = [("basic", Basic), ("io", C.c_uint64 * 6), ("process_memory", C.c_size_t),
                    ("job_memory", C.c_size_t), ("peak_process", C.c_size_t), ("peak_job", C.c_size_t)]

    class Memory(C.Structure):
        _fields_ = [("length", W.DWORD), ("load", W.DWORD)] + [(n, C.c_uint64) for n in
            ("total_physical", "available_physical", "total_commit", "available_commit", "total_virtual", "available_virtual", "extended")]

    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]
    kernel.CreateJobObjectW.restype = W.HANDLE
    kernel.SetInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]
    kernel.QueryInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p]
    kernel.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
    kernel.CloseHandle.argtypes = [W.HANDLE]
    nt = C.WinDLL("ntdll")
    nt.NtResumeProcess.argtypes = [W.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise C.WinError(C.get_last_error())
    info = Extended()
    info.basic.flags = 0x2000  # Kill descendants when the measurement job closes.
    reason, peak = None, 0
    process = None
    try:
        if not kernel.SetInformationJobObject(job, 9, C.byref(info), C.sizeof(info)):
            raise C.WinError(C.get_last_error())
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW | 0x4)
            if not kernel.AssignProcessToJobObject(job, int(process._handle)):
                raise C.WinError(C.get_last_error())
            if nt.NtResumeProcess(int(process._handle)) != 0:
                raise RuntimeError("Cannot resume benchmark process")
            memory = Memory()
            memory.length = C.sizeof(memory)
            if not kernel.GlobalMemoryStatusEx(C.byref(memory)):
                raise C.WinError(C.get_last_error())
            while process.poll() is None:
                if not kernel.QueryInformationJobObject(job, 9, C.byref(info), C.sizeof(info), None):
                    raise C.WinError(C.get_last_error())
                peak = max(peak, info.peak_job)
                if not kernel.GlobalMemoryStatusEx(C.byref(memory)):
                    raise C.WinError(C.get_last_error())
                if memory.available_physical < 6*2**30 or memory.available_commit < 16*2**30:
                    reason = "Memory guard: less than 6 GiB physical or 16 GiB commit available"
                    break
                sleep(.2)
            kernel.QueryInformationJobObject(job, 9, C.byref(info), C.sizeof(info), None)
            peak = max(peak, info.peak_job)
    finally:
        kernel.CloseHandle(job)
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
    return {"returncode": process.returncode, "guard": reason, "peak_commit_bytes": peak,
            "physical_memory_bytes": memory.total_physical, "commit_limit_bytes": memory.total_commit,
            "logical_processors": os.cpu_count()}


def compare_variants(output, workers):
    import numpy as np
    from packed_bed.reports import load_dataset

    specialized = json.loads((output / f"specialized-{workers}/cold/results.json").read_text())["cases"]
    shared = json.loads((output / f"shared-{workers}/cold/results.json").read_text())["cases"]
    comparisons = []
    for a, b in zip(specialized, shared, strict=True):
        assert a["case"] == b["case"]
        expected, actual = load_dataset(a["results"]), load_dataset(b["results"])
        assert set(expected.data_vars) == set(actual.data_vars)
        for name in expected.coords:
            if np.issubdtype(expected[name].dtype, np.number):
                np.testing.assert_allclose(expected[name], actual[name], rtol=1e-12, atol=1e-12)
            else:
                np.testing.assert_array_equal(expected[name], actual[name])
        differences = {}
        for name in expected.data_vars:
            if not np.isfinite(actual[name]).all():
                raise AssertionError(f"Nonfinite {name} in {b['case']}")
            differences[name] = float(abs(actual[name]-expected[name]).max())
        comparisons.append({"case": a["case"], "max_absolute_difference": differences})
    (output / f"differences-{workers}.json").write_text(json.dumps(comparisons, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 8, 32, 64])
    parser.add_argument("--variants", nargs="+", choices=["specialized", "shared"], default=["specialized", "shared"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rtol", type=float, help="Optional relative tolerance override for convergence checks")
    parser.add_argument("--job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.job:
        return run_job(args.job)
    if not args.cases or not args.output or any(w < 1 for w in args.workers) or (args.limit is not None and args.limit < 1):
        parser.error("Supply --cases, --output, and positive --workers")
    if args.rtol is not None and (not math.isfinite(args.rtol) or args.rtol <= 0):
        parser.error("--rtol must be finite and positive")
    cases = [p.resolve() for arg in args.cases for p in (sorted(arg.glob("*/run.yaml")) if arg.is_dir() else [arg])]
    cases = cases[:args.limit]
    if not cases:
        parser.error("No cases found")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for workers in args.workers:
        for variant in args.variants:
            cache = args.output / f"{variant}-{workers}" / "cache"
            for phase in ("cold", "warm"):
                folder = cache.parent / phase
                folder.mkdir(parents=True)
                path = folder / "job.json"
                path.write_text(json.dumps({"cases": list(map(str, cases)), "workers": workers,
                                            "shared": variant == "shared", "cache": str(cache), "rtol": args.rtol}, indent=2))
                measurement = measured_process([sys.executable, str(Path(__file__).resolve()), "--job", str(path)], folder / "worker.log")
                record = {"variant": variant, "workers": workers, "phase": phase, **measurement}
                if (folder / "results.json").exists():
                    result = json.loads((folder / "results.json").read_text())
                    record.update({key: value for key, value in result.items() if key != "cases"})
                    record["results"] = str(folder / "results.json")
                records.append(record)
                (args.output / "summary.json").write_text(json.dumps(records, indent=2))
                print(json.dumps(record), flush=True)
                if measurement["returncode"] or measurement.get("guard"):
                    return 1
        if set(args.variants) == {"shared", "specialized"}:
            compare_variants(args.output, workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
