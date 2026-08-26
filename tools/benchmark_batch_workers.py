"""Benchmark packed-bed batch throughput across worker counts.

The benchmark expands the bundled four-case default batch with an eight-value
replica axis, producing 32 simulations with the same program/geometry mix.
Each case is explicitly single-threaded so worker counts are comparable.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_PATH = (
    REPOSITORY_ROOT / "packed_bed" / "examples" / "default_batch_case" / "batch.yaml"
)
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "benchmark_results"
RESULT_COLUMNS = (
    "requested_workers",
    "effective_workers",
    "run_order",
    "total_cases",
    "succeeded",
    "failed",
    "wall_time_s",
    "speedup_vs_1_worker",
    "parallel_efficiency",
    "throughput_cases_per_s",
    "throughput_cases_per_min",
    "simulated_seconds_per_wall_second",
    "mean_case_runtime_s",
    "median_case_runtime_s",
    "p95_case_runtime_s",
    "max_case_runtime_s",
    "exit_code",
    "started_utc",
    "finished_utc",
)


def _parse_worker_range(raw_value: str) -> tuple[int, ...]:
    try:
        if "-" in raw_value:
            lower_raw, upper_raw = raw_value.split("-", maxsplit=1)
            lower, upper = int(lower_raw), int(upper_raw)
            values = tuple(range(lower, upper + 1))
        else:
            values = tuple(int(item) for item in raw_value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an inclusive range such as 1-32 or a list such as 1,2,4,8"
        ) from exc
    if not values or min(values) < 1 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("worker counts must be unique positive integers")
    return values


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _absolute_reference(batch_dir: Path, value: str) -> str:
    path = Path(value)
    return str((batch_dir / path).resolve() if not path.is_absolute() else path.resolve())


def _replicated_batch_document(source_path: Path, output_directory: Path) -> dict[str, Any]:
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    source_dir = source_path.parent
    source["base_case"] = _absolute_reference(source_dir, source["base_case"])
    source["output_directory"] = str(output_directory.resolve())
    source["programs"] = {
        name: _absolute_reference(source_dir, path)
        for name, path in source.get("programs", {}).items()
    }
    for geometry in source.get("geometries", {}).values():
        if "solids_file" in geometry:
            geometry["solids_file"] = _absolute_reference(
                source_dir, geometry["solids_file"]
            )

    # The repeated patch is intentionally identical. Its useful effect is to
    # force one numerical thread even for the one-worker baseline.
    source["axes"].append(
        {
            "id": "replica",
            "values": [
                {
                    "id": f"r{index:02d}",
                    "patch": {"run": {"solver": {"threads": 1}}},
                }
                for index in range(1, 9)
            ],
        }
    )
    return source


def _percentile_95(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def _read_case_summary(summary_path: Path) -> tuple[list[dict[str, str]], list[float]]:
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    runtimes = [float(row["runtime_s"]) for row in rows if row["runtime_s"]]
    return rows, runtimes


def _write_results(path: Path, results: list[dict[str, Any]]) -> None:
    baseline = next(
        (float(row["wall_time_s"]) for row in results if row["requested_workers"] == 1),
        None,
    )
    normalized: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda row: int(row["requested_workers"])):
        row = dict(result)
        if baseline is not None:
            speedup = baseline / float(row["wall_time_s"])
            row["speedup_vs_1_worker"] = speedup
            row["parallel_efficiency"] = speedup / int(row["effective_workers"])
        normalized.append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized)


def _write_chart(path: Path, results: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(results, key=lambda row: int(row["requested_workers"]))
    workers = [int(row["requested_workers"]) for row in rows]
    elapsed = [float(row["wall_time_s"]) for row in rows]
    throughput = [float(row["throughput_cases_per_min"]) for row in rows]
    case_runtime = [float(row["mean_case_runtime_s"]) for row in rows]

    figure, (elapsed_axis, throughput_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )
    elapsed_axis.plot(workers, elapsed, marker="o", markersize=4, color="#1f77b4")
    elapsed_axis.set_ylabel("Batch wall time (s)")
    elapsed_axis.set_title("Default packed-bed batch: worker scaling over 32 cases")
    elapsed_axis.grid(alpha=0.25)

    throughput_axis.plot(
        workers,
        throughput,
        marker="o",
        markersize=4,
        color="#2ca02c",
        label="Throughput",
    )
    throughput_axis.set_xlabel("Worker processes")
    throughput_axis.set_ylabel("Throughput (cases/min)", color="#2ca02c")
    throughput_axis.tick_params(axis="y", labelcolor="#2ca02c")
    throughput_axis.set_xticks(range(1, 33))
    throughput_axis.grid(alpha=0.25)

    case_axis = throughput_axis.twinx()
    case_axis.plot(
        workers,
        case_runtime,
        marker=".",
        color="#d62728",
        label="Mean case runtime",
    )
    case_axis.set_ylabel("Mean case runtime (s)", color="#d62728")
    case_axis.tick_params(axis="y", labelcolor="#d62728")

    for axis in (elapsed_axis, throughput_axis):
        axis.axvline(16, color="#666666", linestyle="--", linewidth=1, alpha=0.7)
        axis.axvline(32, color="#666666", linestyle=":", linewidth=1, alpha=0.7)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _safe_remove_output(path: Path, benchmark_root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = benchmark_root.resolve()
    if resolved_path.parent != resolved_root or not resolved_path.name.startswith("output_w"):
        raise RuntimeError(f"refusing to remove unexpected benchmark output: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def _run_one(
    *,
    workers: int,
    run_order: int,
    source_path: Path,
    benchmark_root: Path,
    logs_directory: Path,
) -> dict[str, Any]:
    output_directory = benchmark_root / f"output_w{workers:02d}"
    batch_path = benchmark_root / f"batch_w{workers:02d}.yaml"
    batch_document = _replicated_batch_document(source_path, output_directory)
    batch_path.write_text(yaml.safe_dump(batch_document, sort_keys=False), encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "packed_bed",
        "batch",
        str(batch_path),
        "--workers",
        str(workers),
    ]
    started_utc = _utc_now()
    started_at = perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = perf_counter() - started_at
    finished_utc = _utc_now()
    log_path = logs_directory / f"workers_{workers:02d}.log"
    log_path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\n"
        f"EXIT CODE: {completed.returncode}\n\n"
        f"STDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}",
        encoding="utf-8",
    )

    summary_path = output_directory / "summary.csv"
    if not summary_path.exists():
        raise RuntimeError(
            f"workers={workers} exited {completed.returncode} without a summary; see {log_path}"
        )
    rows, runtimes = _read_case_summary(summary_path)
    succeeded = sum(row["status"] == "success" for row in rows)
    failed = len(rows) - succeeded
    result: dict[str, Any] = {
        "requested_workers": workers,
        "effective_workers": min(workers, len(rows)),
        "run_order": run_order,
        "total_cases": len(rows),
        "succeeded": succeeded,
        "failed": failed,
        "wall_time_s": elapsed,
        "speedup_vs_1_worker": "",
        "parallel_efficiency": "",
        "throughput_cases_per_s": len(rows) / elapsed,
        "throughput_cases_per_min": 60.0 * len(rows) / elapsed,
        "simulated_seconds_per_wall_second": 7200.0 * len(rows) / elapsed,
        "mean_case_runtime_s": statistics.fmean(runtimes),
        "median_case_runtime_s": statistics.median(runtimes),
        "p95_case_runtime_s": _percentile_95(runtimes),
        "max_case_runtime_s": max(runtimes),
        "exit_code": completed.returncode,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }
    if completed.returncode != 0 or failed:
        raise RuntimeError(
            f"workers={workers} had {failed} failed cases; output retained at {output_directory}"
        )
    _safe_remove_output(output_directory, benchmark_root)
    return result


def _metadata(source_path: Path, worker_counts: tuple[int, ...], seed: int) -> dict[str, Any]:
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as cpu_key:
                cpu_name = str(winreg.QueryValueEx(cpu_key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return {
        "created_utc": _utc_now(),
        "source_batch": str(source_path),
        "source_batch_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "workload": {
            "cases": 32,
            "base_cases": 4,
            "replicas_per_base_case": 8,
            "simulation_horizon_s_per_case": 7200.0,
            "solver_threads_per_case": 1,
        },
        "worker_counts": list(worker_counts),
        "run_order_policy": "seeded shuffle, with 1 worker forced first for early baseline",
        "shuffle_seed": seed,
        "machine": {
            "cpu": cpu_name,
            "logical_cpus": os.cpu_count(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "measurement": {
            "repetitions_per_worker_count": 1,
            "clock": "time.perf_counter",
            "includes": "batch expansion, validation, process startup, simulation, reports, and summary",
            "excludes": "post-run deletion of case outputs",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_BATCH_PATH)
    parser.add_argument("--workers", type=_parse_worker_range, default=_parse_worker_range("1-32"))
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--ordered",
        action="store_true",
        help="Run worker counts in ascending order instead of a seeded shuffled order.",
    )
    args = parser.parse_args()

    source_path = args.source.resolve()
    worker_counts: tuple[int, ...] = args.workers
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    benchmark_root = (args.results_root / f"default_batch_workers_{timestamp}").resolve()
    logs_directory = benchmark_root / "logs"
    logs_directory.mkdir(parents=True)
    results_path = benchmark_root / "worker_scaling.csv"
    metadata_path = benchmark_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(_metadata(source_path, worker_counts, args.seed), indent=2),
        encoding="utf-8",
    )

    run_sequence = list(worker_counts)
    if not args.ordered:
        random.Random(args.seed).shuffle(run_sequence)
        if 1 in run_sequence:
            run_sequence.remove(1)
            run_sequence.insert(0, 1)

    print(f"Benchmark directory: {benchmark_root}", flush=True)
    print(f"Run sequence: {run_sequence}", flush=True)
    results: list[dict[str, Any]] = []
    for run_order, workers in enumerate(run_sequence, start=1):
        print(
            f"[{run_order:02d}/{len(run_sequence):02d}] workers={workers}: starting",
            flush=True,
        )
        result = _run_one(
            workers=workers,
            run_order=run_order,
            source_path=source_path,
            benchmark_root=benchmark_root,
            logs_directory=logs_directory,
        )
        results.append(result)
        _write_results(results_path, results)
        print(
            f"[{run_order:02d}/{len(run_sequence):02d}] workers={workers}: "
            f"{result['wall_time_s']:.3f} s, "
            f"{result['throughput_cases_per_min']:.2f} cases/min",
            flush=True,
        )

    chart_path = benchmark_root / "worker_scaling.png"
    _write_chart(chart_path, results)
    print(f"Results: {results_path}", flush=True)
    print(f"Chart: {chart_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
