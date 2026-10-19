"""Run the complete reacting example in a fresh folder and check its outputs."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
# Acceptance limits for this example, in kg and J. The desktop preset currently
# gives about 0.0031 kg and 0.00012 J; allow numerical differences across platforms.
BALANCE_LIMITS = {"mass": 0.01, "heat": 1.0}


def verify_results(case):
    import numpy as np
    from packed_bed.plotting import PLOT_REGISTRY
    from packed_bed.reports import REPORT_REGISTRY, load_dataset

    manifest = json.loads((case.output_directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "success" or manifest.get("plots", {}).get("status") != "success":
        raise ValueError("The example did not complete with all requested plots.")
    with load_dataset(case.output_directory / "results.nc") as data:
        horizon = case.run.simulation.time_horizon_s
        interval = case.run.simulation.reporting_interval_s
        expected_times = np.append(np.arange(0, horizon, interval), horizon)
        if data.time.shape != expected_times.shape or not np.allclose(data.time, expected_times, rtol=0, atol=1e-8):
            raise ValueError("The example did not record every reporting time through its full duration.")
        if data.sizes.get("x_cell") != case.run.model.axial_cells:
            raise ValueError("The results do not contain every axial cell.")
        for report in case.run.outputs.requested_reports:
            for field in REPORT_REGISTRY[report].outputs:
                if field.name not in data:
                    raise ValueError(f"Missing requested result: {field.name}")
        for name, values in data.data_vars.items():
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite result: {name}")
        for name in ("temperature", "pressure"):
            if not (data[name] > 0).all():
                raise ValueError(f"Non-positive {name} in the example results.")
        if not np.allclose(data.gas_mole_fraction.sum("gas_species"), 1, rtol=0, atol=1e-6):
            raise ValueError("Gas mole fractions do not sum to one.")
        # This cyclic example can return near its starting composition. Check
        # conversion throughout the run, not only at the final reporting time.
        conversion = float(abs(data.solid_mole_fraction - data.solid_mole_fraction.isel(time=0)).max())
        if conversion < 0.01:
            raise ValueError("The reacting example produced no appreciable solid conversion.")
        balances = {}
        for name, limit in BALANCE_LIMITS.items():
            error = float(abs(data[f"{name}_balance_error"]).max())
            if error > limit:
                raise ValueError(f"{name.capitalize()} balance error {error:g} exceeds the example limit {limit:g}.")
            balances[name] = {"max_abs_error": error, "limit": limit}
        for plot in case.run.outputs.requested_plots:
            path = case.artifacts_directory / PLOT_REGISTRY[plot].filename
            if ElementTree.parse(path).getroot().tag != "{http://www.w3.org/2000/svg}svg":
                raise ValueError(f"Invalid SVG plot: {path.name}")
        return {"reporting_times": len(data.time), "final_time_s": float(data.time[-1]),
                "axial_cells": data.sizes["x_cell"], "balances": balances,
                "plots": list(case.run.outputs.requested_plots), "max_solid_fraction_change": conversion}


def check(output, compiled_bundle=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    record = {"passed": False, "example": "examples/simulations/default_case/run.yaml"}
    print(f"Running the full reacting example. Results and log: {work}", flush=True)
    try:
        from packed_bed.config import load_case

        inputs = work / "case"
        inputs.mkdir()
        for source in (ROOT / "examples/simulations/default_case").glob("*.yaml"):
            shutil.copy2(source, inputs / source.name)
        case = load_case(inputs / "run.yaml")
        environment = os.environ.copy()
        environment.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                           MPLCONFIGDIR=str(work / "matplotlib"),
                           PACKED_BED_COMPILED_CACHE=str(work / "cache"))
        if compiled_bundle is not None:
            environment["MULTISOLID_COMPILED_BUNDLE"] = str(Path(compiled_bundle).resolve())
        with (work / "simulation.log").open("wb") as log:
            subprocess.run([sys.executable, "-m", "packed_bed", str(case.run_path)],
                           cwd=work, env=environment, stdout=log, stderr=log, timeout=1200, check=True)
        record.update(verify_results(case), passed=True)
        print(f"Example passed: {record['final_time_s']:g} s, {record['reporting_times']} reporting times, "
              f"{len(record['plots'])} plots, mass and heat balances within limits.", flush=True)
        return 0
    except Exception:
        record["error"] = traceback.format_exc()
        print(record["error"], file=sys.stderr)
        log = work / "simulation.log"
        if log.exists():
            print(log.read_text(encoding="utf-8", errors="replace")[-4000:], file=sys.stderr)
        return 1
    finally:
        (work / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/example-check",
                        help="Parent folder for fresh test runs and their logs")
    parser.add_argument("--compiled-bundle", type=Path, help="Use this staged native runtime")
    args = parser.parse_args()
    raise SystemExit(check(args.output, args.compiled_bundle))
