from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunResult
from .reporting import DERIVED_REPORT_VARIABLE_NAMES, REPORT_VARIABLE_REGISTRY


@dataclass(frozen=True)
class ReportAxisSpec:
    kind: str
    column_name: str


@dataclass(frozen=True)
class VariableReportSpec:
    variable_name: str
    axes: tuple[ReportAxisSpec, ...]
    value_column_name: str


@dataclass(frozen=True)
class BalanceError:
    max_abs_error: float
    time_s: float
    unit: str


VARIABLE_REPORT_SPECS = {
    "temperature": VariableReportSpec(
        variable_name="temp_bed",
        axes=(ReportAxisSpec("x_cell", "x_cell_m"),),
        value_column_name="temperature_k",
    ),
    "pressure": VariableReportSpec(
        variable_name="pres_bed",
        axes=(ReportAxisSpec("x_cell", "x_cell_m"),),
        value_column_name="pressure_pa",
    ),
    "gas_concentration": VariableReportSpec(
        variable_name="c_gas",
        axes=(
            ReportAxisSpec("gas_species", "gas_species"),
            ReportAxisSpec("x_cell", "x_cell_m"),
        ),
        value_column_name="gas_concentration",
    ),
    "gas_mole_fraction": VariableReportSpec(
        variable_name="y_gas",
        axes=(
            ReportAxisSpec("gas_species", "gas_species"),
            ReportAxisSpec("x_cell", "x_cell_m"),
        ),
        value_column_name="gas_mole_fraction",
    ),
    "solid_concentration": VariableReportSpec(
        variable_name="c_sol",
        axes=(
            ReportAxisSpec("solid_species", "solid_species"),
            ReportAxisSpec("x_cell", "x_cell_m"),
        ),
        value_column_name="solid_concentration",
    ),
    "solid_mole_fraction": VariableReportSpec(
        variable_name="y_sol",
        axes=(
            ReportAxisSpec("solid_species", "solid_species"),
            ReportAxisSpec("x_cell", "x_cell_m"),
        ),
        value_column_name="solid_mole_fraction",
    ),
    "gas_flux": VariableReportSpec(
        variable_name="N_gas_face",
        axes=(
            ReportAxisSpec("gas_species", "gas_species"),
            ReportAxisSpec("x_face", "x_face_m"),
        ),
        value_column_name="gas_flux",
    ),
    "gas_source": VariableReportSpec(
        variable_name="S_gas",
        axes=(
            ReportAxisSpec("gas_species", "gas_species"),
            ReportAxisSpec("x_cell", "x_cell_m"),
        ),
        value_column_name="gas_source",
    ),
    "solid_source": VariableReportSpec(
        variable_name="S_sol",
        axes=(
            ReportAxisSpec("solid_species", "solid_species"),
            ReportAxisSpec("x_cell", "x_cell_m"),
        ),
        value_column_name="solid_source",
    ),
    "reaction_rate": VariableReportSpec(
        variable_name="R_rxn",
        axes=(
            ReportAxisSpec("reaction", "reaction_id"),
            ReportAxisSpec("x_cell", "x_cell_m"),
        ),
        value_column_name="reaction_rate",
    ),
    "gas_enthalpy_flux": VariableReportSpec(
        variable_name="J_gas_face",
        axes=(
            ReportAxisSpec("gas_species", "gas_species"),
            ReportAxisSpec("x_face", "x_face_m"),
        ),
        value_column_name="gas_enthalpy_flux",
    ),
}


BALANCE_REPORT_SPECS = {
    "heat_balance": {
        "variables": DERIVED_REPORT_VARIABLE_NAMES["heat_balance"],
        "columns": (
            "heat_in_total_J",
            "heat_out_total_J",
            "heat_bed_total_J",
            "heat_balance_error_J",
        ),
        "error_key": "heat",
        "unit": "J",
    },
    "mass_balance": {
        "variables": DERIVED_REPORT_VARIABLE_NAMES["mass_balance"],
        "columns": (
            "mass_in_total_kg",
            "mass_out_total_kg",
            "mass_bed_total_kg",
            "mass_balance_error_kg",
        ),
        "error_key": "mass",
        "unit": "kg",
    },
}


def _require_process(run_result: RunResult):
    reporter = run_result.reporter
    if reporter is None or not hasattr(reporter, "Process"):
        raise ValueError("RunResult does not contain a DAETools reporter with a Process payload.")

    process = reporter.Process
    if process is None or not hasattr(process, "dictVariables"):
        raise ValueError("RunResult reporter does not expose Process.dictVariables.")
    return process


def _find_variable(process: Any, variable_name: str):
    matches = sorted(
        key
        for key in process.dictVariables
        if key == variable_name or key.endswith(f".{variable_name}")
    )
    if not matches:
        raise ValueError(f"Reporter does not contain a variable named '{variable_name}'.")
    if len(matches) > 1:
        joined = ", ".join(matches)
        raise ValueError(f"Reporter contains multiple matches for '{variable_name}': {joined}.")
    return process.dictVariables[matches[0]]


def _time_and_values(variable: Any, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(variable.TimeValues, dtype=float).reshape(-1)
    values = np.asarray(variable.Values, dtype=float)
    if values.shape[0] != times.shape[0]:
        raise ValueError(
            f"{label} values have leading dimension {values.shape[0]} but time axis has length {times.shape[0]}."
        )
    return times, values


def _scalar_series(variable: Any, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    times, values = _time_and_values(variable, label=label)
    if values.ndim == 1:
        return times, values
    if values.ndim == 2 and values.shape[1] == 1:
        return times, values[:, 0]
    raise ValueError(f"{label} is not a scalar time series; got array with shape {values.shape}.")


def _require_same_time_axis(reference: np.ndarray, candidate: np.ndarray, *, label: str) -> None:
    if reference.shape != candidate.shape or not np.allclose(reference, candidate, rtol=0.0, atol=1e-12):
        raise ValueError(f"{label} does not share the same time axis as the reference series.")


def _domain_points(variable: Any, domain_index: int, axis_size: int) -> tuple[float | int, ...]:
    domains = getattr(variable, "Domains", ())
    if domain_index >= len(domains):
        return tuple(range(axis_size))

    points = np.asarray(getattr(domains[domain_index], "Points", ()), dtype=float).reshape(-1)
    if points.size != axis_size:
        return tuple(range(axis_size))
    return tuple(float(point) for point in points)


def _axis_values(
    run_result: RunResult,
    variable: Any,
    spec: ReportAxisSpec,
    domain_index: int,
    axis_size: int,
) -> tuple[Any, ...]:
    if spec.kind == "gas_species":
        values = run_result.run_bundle.chemistry.gas_species
    elif spec.kind == "solid_species":
        values = run_result.run_bundle.solids.solid_species
    elif spec.kind == "reaction":
        values = run_result.run_bundle.chemistry.reaction_ids
    else:
        return _domain_points(variable, domain_index, axis_size)

    if len(values) != axis_size:
        raise ValueError(
            f"Report axis '{spec.column_name}' has length {axis_size}, but the run configuration provides {len(values)} labels."
        )
    return tuple(values)


def _write_variable_report_csv(run_result: RunResult, report_id: str, output_path: Path) -> None:
    spec = VARIABLE_REPORT_SPECS[report_id]
    variable = _find_variable(_require_process(run_result), spec.variable_name)
    times, values = _time_and_values(variable, label=f"{report_id} report")
    expected_ndim = 1 + len(spec.axes)
    if values.ndim != expected_ndim:
        raise ValueError(
            f"{report_id} report expected {expected_ndim} dimensions including time; got shape {values.shape}."
        )

    axis_values = tuple(
        _axis_values(run_result, variable, axis_spec, axis_index, axis_size)
        for axis_index, (axis_spec, axis_size) in enumerate(zip(spec.axes, values.shape[1:]))
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", *(axis.column_name for axis in spec.axes), spec.value_column_name))

        if axis_values:
            for time_index, time_s in enumerate(times):
                for value_index in np.ndindex(*(len(values_for_axis) for values_for_axis in axis_values)):
                    writer.writerow(
                        (
                            float(time_s),
                            *(axis_values[axis_index][coordinate] for axis_index, coordinate in enumerate(value_index)),
                            float(values[(time_index, *value_index)]),
                        )
                    )
        else:
            for time_s, value in zip(times, values):
                writer.writerow((float(time_s), float(value)))


def _balance_series(run_result: RunResult, report_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = BALANCE_REPORT_SPECS[report_id]
    process = _require_process(run_result)
    variable_names = spec["variables"]

    first_time, first_values = _scalar_series(
        _find_variable(process, variable_names[0]),
        label=variable_names[0],
    )
    series = [first_values]
    for variable_name in variable_names[1:]:
        times, values = _scalar_series(
            _find_variable(process, variable_name),
            label=variable_name,
        )
        _require_same_time_axis(first_time, times, label=variable_name)
        series.append(values)

    in_total, out_total, bed_total = series
    balance_error = (bed_total - bed_total[0]) + (out_total - in_total)
    return first_time, in_total, out_total, bed_total, balance_error


def _write_balance_report_csv(run_result: RunResult, report_id: str, output_path: Path) -> None:
    spec = BALANCE_REPORT_SPECS[report_id]
    time_s, in_total, out_total, bed_total, balance_error = _balance_series(run_result, report_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", *spec["columns"]))
        for row in zip(time_s, in_total, out_total, bed_total, balance_error):
            writer.writerow(tuple(float(value) for value in row))


def _report_csv_path(output_dir: Path, report_id: str) -> Path:
    safe_report_id = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in report_id)
    return output_dir / f"{safe_report_id}.csv"


def export_requested_report_csvs(run_result: RunResult, output_dir: str | Path | None = None) -> dict[str, Path]:
    resolved_output_dir = Path(output_dir) if output_dir is not None else run_result.output_directory / "reports"
    report_paths: dict[str, Path] = {}

    for report_id in run_result.run_bundle.run.outputs.requested_reports:
        if report_id not in REPORT_VARIABLE_REGISTRY:
            raise ValueError(f"Unknown report id '{report_id}'.")

        output_path = _report_csv_path(resolved_output_dir, report_id)
        if report_id in VARIABLE_REPORT_SPECS:
            _write_variable_report_csv(run_result, report_id, output_path)
        elif report_id in BALANCE_REPORT_SPECS:
            _write_balance_report_csv(run_result, report_id, output_path)
        else:
            raise ValueError(f"Report '{report_id}' does not have a CSV exporter.")
        report_paths[report_id] = output_path

    return report_paths


def compute_balance_errors(run_result: RunResult) -> dict[str, BalanceError]:
    errors: dict[str, BalanceError] = {}
    for report_id, spec in BALANCE_REPORT_SPECS.items():
        time_s, _, _, _, balance_error = _balance_series(run_result, report_id)
        if balance_error.size == 0:
            continue
        max_index = int(np.nanargmax(np.abs(balance_error)))
        errors[spec["error_key"]] = BalanceError(
            max_abs_error=float(abs(balance_error[max_index])),
            time_s=float(time_s[max_index]),
            unit=spec["unit"],
        )
    return errors


def format_balance_error_lines(balance_errors: dict[str, Any]) -> tuple[str, ...]:
    lines: list[str] = []
    for key, label in (("heat", "heat"), ("mass", "mass")):
        error = balance_errors.get(key)
        if error is None:
            lines.append(f"largest {label} balance error: unavailable")
            continue
        lines.append(
            f"largest {label} balance error: {error.max_abs_error:.6g} {error.unit} at t={error.time_s:.6g} s"
        )
    return tuple(lines)


__all__ = [
    "BalanceError",
    "ReportAxisSpec",
    "VariableReportSpec",
    "compute_balance_errors",
    "export_requested_report_csvs",
    "format_balance_error_lines",
]
