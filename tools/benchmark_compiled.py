"""Measure full desktop worker cost for Standard, first and cached Compiled runs.

Use --cases for one or more existing scientific cases; each case is copied into
an isolated benchmark project. Results and logs remain available for comparison.
"""

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
import yaml


def prepare_case(documents, folder, profile, rtol=None, atol=None):
    documents = deepcopy(documents)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    run = documents["run"]
    run["references"] = {name + "_file": name + ".yaml" for name in ("chemistry", "program", "solids")}
    run["outputs"].update(directory="output", artifacts_directory="output/artifacts", requested_plots=[])
    solver = run["solver"]
    solver.update(backend="daetools" if profile in {"reference", "superlu", "klu"} else "compiled",
                  name="klu" if "klu" in profile else "band" if profile in {"compiled", "compiled_band"} else "superlu",
                  threads=1)
    if rtol is not None:
        solver["relative_tolerance"] = rtol
    if atol is not None:
        solver["concentration_absolute_tolerance"] = atol
    for name, document in documents.items():
        (folder / (name + ".yaml")).write_text(yaml.safe_dump(document, sort_keys=False))
    return folder / "run.yaml", documents


def compare(reference, actual):
    from packed_bed.reports import load_dataset
    expected, received = load_dataset(reference), load_dataset(actual)
    if set(expected.variables) != set(received.variables):
        raise AssertionError("Result quantities differ")
    errors = {}
    for name in expected.coords:
        if np.issubdtype(expected[name].dtype, np.number):
            # Standard advances reporting times by addition; Compiled uses a
            # schedule. Permit floating-point roundoff, not a shifted grid.
            np.testing.assert_allclose(expected[name], received[name], rtol=1e-12, atol=1e-12)
        else:
            np.testing.assert_array_equal(expected[name], received[name])
    for name in expected.data_vars:
        a, b = expected[name].values, received[name].values
        if not np.isfinite(b).all():
            raise AssertionError(f"Nonfinite {name}")
        errors[name] = float(np.max(abs(a - b)))
        if name == "temperature":
            np.testing.assert_allclose(a, b, rtol=0, atol=.02)
        elif name in {"gas_mole_fraction", "solid_mole_fraction"}:
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-4)
    return errors


def main():
    from packed_bed_ui.project import Project, read_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solvers", nargs="+", default=["superlu", "klu", "band"], choices=["superlu", "superlu_mt", "klu", "band"])
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    measurements, references = [], {}
    for workers in args.workers:
        for solver in args.solvers:
            project = Project.create(args.output / f"{solver}-{workers}")
            for case_file in args.cases:
                case = project.add_case_from_files(case_file)
                case.documents["run"]["solver"]["threads"] = 1
                case.save()
            for iteration in range(-1, args.repeats + 1):
                backend = "daetools" if iteration == -1 else "compiled"
                for case in project.cases:
                    settings = case.documents["run"]["solver"]
                    settings.update(backend=backend, name=("klu" if solver == "klu" else "superlu") if iteration == -1 else solver)
                    # Use the same compatible controls in every measured configuration.
                    settings.update(scale_residuals=False, step_growth_threshold=2., nonlinear_refresh_interval=0,
                                    vector_exponentials=False, band_reciprocals=False)
                    case.save()
                job_path = project.prepare_execution(project.cases, max_workers=workers)
                start = perf_counter()
                with (project.root / f"benchmark-worker-{iteration}.log").open("w") as log:
                    subprocess.run([sys.executable, "-m", "packed_bed_ui", "--project-worker", str(job_path)],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                elapsed = perf_counter() - start
                kernels = set()
                for index, case in enumerate(project.cases):
                    manifest = read_json(case.run_folder / "output/manifest.json")
                    (project.root / f"manifest-{index}-{iteration}.json").write_text(json.dumps(manifest, indent=2))
                    shutil.copy2(case.run_folder / "worker.log", project.root / f"case-{index}-{iteration}.log")
                    result = next((case.run_folder / "output").glob("*.nc"))
                    stats = manifest["solver_stats"]
                    if iteration == -1:
                        saved = project.root / f"reference-{index}.nc"
                        shutil.copy2(result, saved)
                        references[index] = saved
                    else:
                        errors = compare(references[index], result)
                        (project.root / f"difference-{index}-{iteration}.json").write_text(json.dumps(errors, indent=2))
                        if iteration > 0 and not stats["cache_hit"]:
                            raise AssertionError("Repeated compiled run missed its kernel cache")
                        kernels.add(stats["kernel_sha256"])
                    measurements.append({"solver": solver, "workers": workers, "backend": backend,
                                         "iteration": iteration, "case": index, "batch_elapsed_s": elapsed,
                                         "worker_elapsed_s": read_json(case.run_folder / "status.json")["end_to_end_s"],
                                         "stats": stats})
                measurements[-1]["distinct_kernels"] = len(kernels)
                (args.output / "measurements.json").write_text(json.dumps(measurements, indent=2))
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["solver", "workers", "backend", "iteration", "case", "batch_elapsed_s", "worker_elapsed_s", "distinct_kernels"])
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != "stats"} for row in measurements)


if __name__ == "__main__":
    main()
