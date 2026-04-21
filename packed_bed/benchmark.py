from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from daetools.pyDAE import daeNoOpDataReporter, daePythonStdOutLog

from .config import RunBundle, RunResult, load_run_bundle
from .properties import PROPERTY_REGISTRY
from .reactions import REACTION_CATALOG
from .result_plots import render_run_result_plots
from .solver import (
    _program_breakpoint_times,
    _run_with_breakpoint_nudges,
    _set_reporting_on,
    _warm_start_first_reporting_interval,
    assemble_simulation,
    build_idas_solver,
    configure_evaluation_mode,
)
from .solver_ignore_discontinuities import (
    _run_with_reporting_times as _run_with_manual_reporting_times,
    _warm_start_first_reporting_interval as _manual_warm_start_first_reporting_interval,
)


MEDRANO_RUN_YAML = Path(__file__).parent / "examples" / "medrano_case" / "run.yaml"


@dataclass(frozen=True)
class MedranoBenchmarkSpec:
    name: str
    time_horizon_s: float | None = None
    reporting_interval_s: float | None = None
    axial_cells: int | None = None
    relative_tolerance: float | None = None
    report_mode: str = "benchmark"
    runner: str = "breakpoint"
    warm_start_max_step_s: float | None = 0.1
    render_plots: bool = False
    plot_output_dir: Path | None = None


def _apply_overrides(run_bundle: RunBundle, spec: MedranoBenchmarkSpec) -> RunBundle:
    simulation_updates: dict[str, Any] = {}
    if spec.time_horizon_s is not None:
        simulation_updates["time_horizon_s"] = float(spec.time_horizon_s)
    if spec.reporting_interval_s is not None:
        simulation_updates["reporting_interval_s"] = float(spec.reporting_interval_s)

    model_updates: dict[str, Any] = {}
    if spec.axial_cells is not None:
        model_updates["axial_cells"] = int(spec.axial_cells)

    solver_updates: dict[str, Any] = {}
    if spec.relative_tolerance is not None:
        solver_updates["relative_tolerance"] = float(spec.relative_tolerance)

    run_config = run_bundle.run
    if simulation_updates:
        run_config = run_config.model_copy(
            update={"simulation": run_config.simulation.model_copy(update=simulation_updates)}
        )
    if model_updates:
        run_config = run_config.model_copy(update={"model": run_config.model.model_copy(update=model_updates)})
    if solver_updates:
        run_config = run_config.model_copy(update={"solver": run_config.solver.model_copy(update=solver_updates)})
    return run_bundle.model_copy(update={"run": run_config})


def _reporting_options(run_bundle: RunBundle, spec: MedranoBenchmarkSpec) -> dict[str, Any]:
    if spec.report_mode == "all":
        return {"report_ids": None}
    if spec.report_mode == "requested":
        return {"report_ids": run_bundle.run.outputs.requested_reports}
    if spec.report_mode == "benchmark":
        return {
            "report_ids": run_bundle.run.outputs.requested_reports,
            "include_benchmark_snapshot": True,
        }
    if spec.report_mode == "plot":
        return {
            "report_ids": run_bundle.run.outputs.requested_reports,
            "include_plot_variables": True,
            "include_benchmark_snapshot": True,
        }
    if spec.report_mode == "none":
        return {"report_ids": ()}
    raise ValueError(f"Unsupported report mode '{spec.report_mode}'.")


def _equation_group_name(name: str) -> str:
    short_name = name.split(".", 1)[1] if "." in name else name
    short_name = short_name.split("(", 1)[0]
    grouped_prefixes = (
        "face_flux_",
        "face_enthalpy_flux_",
        "species_balance_cell_",
        "solid_species_balance_cell_",
        "ergun_face_",
        "energy_balance_cell_",
    )
    for prefix in grouped_prefixes:
        if short_name.startswith(prefix):
            return prefix.rstrip("_")
    return short_name


def _structure_metrics(simulation) -> dict[str, Any]:
    group_metrics: dict[str, dict[str, int]] = {}
    for info in simulation.EquationExecutionInfos:
        operation_count, result_count = info.GetComputeStackInfo()
        group_name = _equation_group_name(info.Name)
        metrics = group_metrics.setdefault(
            group_name,
            {"equations": 0, "operations": 0, "results": 0, "max_operations": 0},
        )
        metrics["equations"] += 1
        metrics["operations"] += int(operation_count)
        metrics["results"] += int(result_count)
        metrics["max_operations"] = max(metrics["max_operations"], int(operation_count))

    top_equation_groups = sorted(
        (
            {"name": name, **metrics}
            for name, metrics in group_metrics.items()
        ),
        key=lambda item: item["operations"],
        reverse=True,
    )
    active_memory = getattr(simulation, "ActiveEquationSetMemory", ())
    return {
        "equations": int(simulation.NumberOfEquations),
        "variables": int(simulation.TotalNumberOfVariables),
        "node_counts": {
            str(name): int(value)
            for name, value in simulation.ActiveEquationSetNodeCount.items()
        },
        "active_equation_memory_total": int(sum(active_memory)) if active_memory else 0,
        "top_equation_groups": top_equation_groups[:20],
    }


def _find_reporter_variable(process, variable_name: str):
    matches = sorted(
        key
        for key in process.dictVariables
        if key == variable_name or key.endswith(f".{variable_name}")
    )
    if len(matches) != 1:
        return None
    return process.dictVariables[matches[0]]


def _time_values(variable) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(variable.TimeValues, dtype=float).reshape(-1),
        np.asarray(variable.Values, dtype=float),
    )


def _final_scalar(process, variable_name: str) -> float | None:
    variable = _find_reporter_variable(process, variable_name)
    if variable is None:
        return None
    _, values = _time_values(variable)
    if values.size == 0:
        return None
    return float(np.asarray(values[-1]).reshape(-1)[0])


def _extract_correctness_snapshot(run_bundle: RunBundle, reporter) -> dict[str, Any]:
    process = getattr(reporter, "Process", None)
    if process is None or not hasattr(process, "dictVariables"):
        return {"available": False}

    temperature_variable = _find_reporter_variable(process, "temp_bed")
    pressure_variable = _find_reporter_variable(process, "pres_bed")
    gas_fraction_variable = _find_reporter_variable(process, "y_gas")
    solid_concentration_variable = _find_reporter_variable(process, "c_sol")

    snapshot: dict[str, Any] = {"available": True}
    if temperature_variable is not None:
        time_s, temperature = _time_values(temperature_variable)
        snapshot["final_time_s"] = float(time_s[-1]) if time_s.size else None
        snapshot["outlet_temperature_k"] = float(temperature[-1, -1])
    if pressure_variable is not None:
        _, pressure = _time_values(pressure_variable)
        snapshot["outlet_pressure_pa"] = float(pressure[-1, -1])
    if gas_fraction_variable is not None:
        _, gas_fraction = _time_values(gas_fraction_variable)
        outlet_composition = gas_fraction[-1, :, -1]
        snapshot["outlet_composition"] = {
            species_id: float(value)
            for species_id, value in zip(run_bundle.chemistry.gas_species, outlet_composition)
        }
    if solid_concentration_variable is not None:
        _, solid_concentration = _time_values(solid_concentration_variable)
        area = np.pi * run_bundle.run.model.bed_radius_m**2
        cell_width = run_bundle.run.model.bed_length_m / run_bundle.run.model.axial_cells
        final_inventory = area * cell_width * solid_concentration[-1].sum(axis=1)
        snapshot["solid_inventory_mol"] = {
            species_id: float(value)
            for species_id, value in zip(run_bundle.solids.solid_species, final_inventory)
        }

    for variable_name in (
        "heat_in_total",
        "heat_out_total",
        "heat_bed_total",
        "mass_in_total",
        "mass_out_total",
        "mass_bed_total",
        "material_in_total",
        "material_out_total",
        "material_bed_total",
    ):
        value = _final_scalar(process, variable_name)
        if value is not None:
            snapshot[variable_name] = value

    return snapshot


def _reporter_variable_count(reporter) -> int:
    process = getattr(reporter, "Process", None)
    if process is None or not hasattr(process, "dictVariables"):
        return 0
    return len(process.dictVariables)


def _run_main_integration(simulation, runner: str, breakpoint_times) -> None:
    if runner == "breakpoint":
        _run_with_breakpoint_nudges(simulation, breakpoint_times)
        return
    if runner == "manual":
        _run_with_manual_reporting_times(simulation)
        return
    raise ValueError(f"Unsupported runner '{runner}'.")


def _run_warm_start(simulation, spec: MedranoBenchmarkSpec, breakpoint_times) -> None:
    if spec.warm_start_max_step_s is None:
        return
    if spec.runner == "breakpoint":
        _warm_start_first_reporting_interval(
            simulation,
            breakpoint_times,
            max_step_s=spec.warm_start_max_step_s,
        )
        return
    if spec.runner == "manual":
        _manual_warm_start_first_reporting_interval(
            simulation,
            max_step_s=spec.warm_start_max_step_s,
        )
        return
    raise ValueError(f"Unsupported runner '{spec.runner}'.")


def run_benchmark_once(spec: MedranoBenchmarkSpec, *, run_yaml_path: str | Path = MEDRANO_RUN_YAML) -> dict[str, Any]:
    timings: dict[str, float] = {}

    phase_start = perf_counter()
    run_bundle = load_run_bundle(run_yaml_path)
    timings["load_config_s"] = perf_counter() - phase_start

    phase_start = perf_counter()
    run_bundle = _apply_overrides(run_bundle, spec)
    timings["apply_overrides_s"] = perf_counter() - phase_start

    configure_evaluation_mode()
    phase_start = perf_counter()
    assembly = assemble_simulation(
        run_bundle,
        property_registry=PROPERTY_REGISTRY,
        reaction_catalog=REACTION_CATALOG,
    )
    timings["assemble_s"] = perf_counter() - phase_start

    simulation = assembly.simulation
    reporting_options = _reporting_options(run_bundle, spec)
    if spec.render_plots:
        reporting_options["include_plot_variables"] = True
    _set_reporting_on(simulation, **reporting_options)
    simulation.ReportTimeDerivatives = run_bundle.run.report_time_derivatives
    simulation.ReportingInterval = run_bundle.run.reporting_interval_s
    simulation.TimeHorizon = run_bundle.run.time_horizon_s

    solver = build_idas_solver(relative_tolerance=run_bundle.run.solver.relative_tolerance)
    reporter = daeNoOpDataReporter()
    log = daePythonStdOutLog()
    log.PrintProgress = False

    phase_start = perf_counter()
    simulation.Initialize(solver, reporter, log)
    timings["initialize_s"] = perf_counter() - phase_start
    structure = _structure_metrics(simulation)

    phase_start = perf_counter()
    simulation.SolveInitial()
    timings["solve_initial_s"] = perf_counter() - phase_start

    breakpoint_times = _program_breakpoint_times(simulation)
    phase_start = perf_counter()
    _run_warm_start(simulation, spec, breakpoint_times)
    timings["warm_start_s"] = perf_counter() - phase_start

    phase_start = perf_counter()
    _run_main_integration(simulation, spec.runner, breakpoint_times)
    timings["main_integration_s"] = perf_counter() - phase_start

    phase_start = perf_counter()
    correctness_snapshot = _extract_correctness_snapshot(run_bundle, reporter)
    timings["reporter_extract_s"] = perf_counter() - phase_start

    plot_paths: dict[str, str] = {}
    phase_start = perf_counter()
    if spec.render_plots:
        run_result = RunResult(
            run_bundle=run_bundle,
            output_directory=run_bundle.output_directory,
            success=True,
            reporter=reporter,
            simulation=simulation,
        )
        if spec.plot_output_dir is not None:
            plot_paths = {
                key: str(path)
                for key, path in render_run_result_plots(run_result, output_dir=spec.plot_output_dir).items()
            }
        else:
            with tempfile.TemporaryDirectory(prefix="packed_bed_benchmark_") as temp_dir:
                plot_paths = {
                    key: Path(path).name
                    for key, path in render_run_result_plots(run_result, output_dir=temp_dir).items()
                }
    timings["plotting_s"] = perf_counter() - phase_start

    total_s = sum(timings.values())
    return {
        "name": spec.name,
        "spec": {
            "time_horizon_s": run_bundle.run.time_horizon_s,
            "reporting_interval_s": run_bundle.run.reporting_interval_s,
            "axial_cells": run_bundle.run.model.axial_cells,
            "relative_tolerance": run_bundle.run.solver.relative_tolerance,
            "report_mode": spec.report_mode,
            "runner": spec.runner,
            "warm_start_max_step_s": spec.warm_start_max_step_s,
            "render_plots": spec.render_plots,
            "plot_output_dir": str(spec.plot_output_dir) if spec.plot_output_dir is not None else None,
        },
        "timings_s": timings,
        "total_s": total_s,
        "structure": structure,
        "reporter_variable_count": _reporter_variable_count(reporter),
        "correctness_snapshot": correctness_snapshot,
        "plot_paths": plot_paths,
    }


def summarize_repeats(name: str, measured_results: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [result["total_s"] for result in measured_results]
    timing_keys = sorted(measured_results[0]["timings_s"])
    phase_summary = {}
    for key in timing_keys:
        values = [result["timings_s"][key] for result in measured_results]
        phase_summary[key] = {
            "median_s": statistics.median(values),
            "min_s": min(values),
            "max_s": max(values),
        }

    return {
        "name": name,
        "runs": len(measured_results),
        "total_s": {
            "median_s": statistics.median(totals),
            "min_s": min(totals),
            "max_s": max(totals),
        },
        "phase_timings_s": phase_summary,
        "latest_result": measured_results[-1],
    }


def run_benchmark_repeats(
    spec: MedranoBenchmarkSpec,
    *,
    warmups: int,
    runs: int,
    run_yaml_path: str | Path = MEDRANO_RUN_YAML,
) -> dict[str, Any]:
    for _ in range(warmups):
        with contextlib.redirect_stdout(io.StringIO()):
            run_benchmark_once(spec, run_yaml_path=run_yaml_path)

    measured_results = [
        _run_benchmark_once_quiet(spec, run_yaml_path=run_yaml_path)
        for _ in range(runs)
    ]
    return summarize_repeats(spec.name, measured_results)


def _run_benchmark_once_quiet(spec: MedranoBenchmarkSpec, *, run_yaml_path: str | Path) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        return run_benchmark_once(spec, run_yaml_path=run_yaml_path)


def build_benchmark_specs(tier: str) -> list[MedranoBenchmarkSpec]:
    if tier == "medrano-short":
        return [
            MedranoBenchmarkSpec(
                name="medrano-short",
                time_horizon_s=100.0,
                reporting_interval_s=1.0,
                axial_cells=10,
            )
        ]
    if tier == "medrano-scale":
        return [
            MedranoBenchmarkSpec(
                name=f"medrano-scale-{cells}-cells",
                time_horizon_s=100.0,
                reporting_interval_s=100.0,
                axial_cells=cells,
            )
            for cells in (5, 10, 20, 40)
        ]
    if tier == "medrano-reporting":
        return [
            MedranoBenchmarkSpec(
                name=f"medrano-reporting-{report_mode}-ri{interval:g}",
                time_horizon_s=100.0,
                reporting_interval_s=interval,
                axial_cells=10,
                report_mode=report_mode,
            )
            for interval in (1.0, 10.0, 100.0)
            for report_mode in ("all", "requested")
        ]
    if tier == "medrano-runner":
        specs: list[MedranoBenchmarkSpec] = []
        for runner in ("breakpoint", "manual"):
            for max_step in (0.1, 0.5, 1.0, None):
                label = "disabled" if max_step is None else f"{max_step:g}"
                specs.append(
                    MedranoBenchmarkSpec(
                        name=f"medrano-runner-{runner}-warm{label}",
                        time_horizon_s=100.0,
                        reporting_interval_s=1.0,
                        axial_cells=10,
                        runner=runner,
                        warm_start_max_step_s=max_step,
                    )
                )
        return specs
    if tier == "medrano-full":
        return [MedranoBenchmarkSpec(name="medrano-full")]
    if tier == "all":
        specs = []
        for sub_tier in (
            "medrano-short",
            "medrano-scale",
            "medrano-reporting",
            "medrano-runner",
            "medrano-full",
        ):
            specs.extend(build_benchmark_specs(sub_tier))
        return specs
    raise ValueError(f"Unsupported benchmark tier '{tier}'.")


def _apply_cli_spec_overrides(spec: MedranoBenchmarkSpec, args) -> MedranoBenchmarkSpec:
    updates: dict[str, Any] = {}
    for arg_name, field_name in (
        ("time_horizon", "time_horizon_s"),
        ("reporting_interval", "reporting_interval_s"),
        ("axial_cells", "axial_cells"),
        ("relative_tolerance", "relative_tolerance"),
        ("report_mode", "report_mode"),
        ("runner", "runner"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value

    if args.no_warm_start:
        updates["warm_start_max_step_s"] = None
    elif args.warm_start_max_step is not None:
        updates["warm_start_max_step_s"] = args.warm_start_max_step

    if args.render_plots:
        updates["render_plots"] = True
    if args.plot_output_dir is not None:
        updates["render_plots"] = True
        updates["plot_output_dir"] = args.plot_output_dir

    if not updates:
        return spec
    return replace(spec, **updates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark packed_bed using medrano_case.")
    parser.add_argument(
        "--tier",
        choices=("medrano-short", "medrano-scale", "medrano-reporting", "medrano-runner", "medrano-full", "all"),
        default="medrano-short",
    )
    parser.add_argument("--run-yaml", default=str(MEDRANO_RUN_YAML))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--time-horizon", type=float)
    parser.add_argument("--reporting-interval", type=float)
    parser.add_argument("--axial-cells", type=int)
    parser.add_argument("--relative-tolerance", type=float)
    parser.add_argument("--report-mode", choices=("all", "requested", "benchmark", "plot", "none"))
    parser.add_argument("--runner", choices=("breakpoint", "manual"))
    parser.add_argument("--warm-start-max-step", type=float)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--render-plots", action="store_true")
    parser.add_argument("--plot-output-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative.")

    specs = [
        _apply_cli_spec_overrides(spec, args)
        for spec in build_benchmark_specs(args.tier)
    ]
    summaries = [
        run_benchmark_repeats(
            spec,
            warmups=args.warmups,
            runs=args.runs,
            run_yaml_path=args.run_yaml,
        )
        for spec in specs
    ]
    payload = {
        "benchmark": "packed_bed.medrano_case",
        "tier": args.tier,
        "warmups": args.warmups,
        "runs": args.runs,
        "summaries": summaries,
    }
    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
