#!/usr/bin/env python3
"""Audit packed-bed ML batch inputs and NetCDF results for dataset outliers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr
import yaml


CORE_FLOW_BOUNDS_GHSV_H_1 = (300.0, 1500.0)
CORE_TEMPERATURE_BOUNDS_K = (500.0, 900.0)
CORE_PRESSURE_BOUNDS_PA = (1.0e5, 3.5e6)
NORMAL_MOLAR_DENSITY_MOL_PER_M3 = 100000.0 / (8.31446 * 273.15)
# IDAS trajectories commonly contain mole-fraction undershoots around 1e-7.
# Treat those as solver roundoff; values beyond 1 ppm remain audit failures.
PHYSICAL_TOLERANCE = 1.0e-6
COMPOSITION_SUM_TOLERANCE = 2.0e-4

NUMERIC_RESULT_VARIABLES = (
    "temperature",
    "pressure",
    "gas_flux",
    "gas_mole_fraction",
    "inlet_composition",
    "outlet_composition",
    "outlet_species_flow",
    "solid_mole_fraction",
    "inlet_temperature",
    "outlet_temperature",
    "inlet_pressure",
    "outlet_pressure",
    "pressure_drop",
    "inlet_flow",
    "outlet_flow",
)

STATISTICAL_METRICS = (
    "result_temperature_min_k",
    "result_temperature_max_k",
    "result_pressure_min_pa",
    "result_pressure_max_pa",
    "result_pressure_drop_min_pa",
    "result_pressure_drop_max_pa",
    "result_inlet_flow_max_mol_s",
    "result_outlet_flow_max_mol_s",
    "result_outlet_to_inlet_flow_ratio_max",
    "result_gas_flux_abs_max_mol_m2_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "batch_directory",
        type=Path,
        help="Directory containing batch.yaml, programs/, and output/.",
    )
    parser.add_argument(
        "--report-directory",
        type=Path,
        help="Destination (default: <batch_directory>/output/outlier_audit).",
    )
    parser.add_argument(
        "--result-stride",
        type=int,
        default=1,
        help="Read every Nth time point from spatial result variables (default: 1).",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def finite_floats(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def channel_target_values(channel: dict[str, Any]) -> list[float]:
    values = finite_floats((channel.get("initial"),))
    for step in channel.get("steps", []):
        if isinstance(step, dict) and step.get("kind") == "ramp":
            values.extend(finite_floats((step.get("target"),)))
    return values


def program_cohort(axis_program: str) -> str:
    if axis_program.startswith("test_prog_"):
        return "test"
    if axis_program.startswith("css_prog_"):
        return "css"
    if re.fullmatch(r"prog_\d+", axis_program):
        return "core"
    return "other"


def add_flag(
    flags: list[dict[str, Any]],
    *,
    case_id: str,
    cohort: str,
    category: str,
    severity: str,
    metric: str,
    value: Any,
    threshold: Any,
    detail: str,
) -> None:
    flags.append(
        {
            "case_id": case_id,
            "cohort": cohort,
            "category": category,
            "severity": severity,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "detail": detail,
        }
    )


def scalar_extrema(array: np.ndarray) -> tuple[float, float, int]:
    finite = np.isfinite(array)
    nonfinite = int(array.size - np.count_nonzero(finite))
    if not np.any(finite):
        return math.nan, math.nan, nonfinite
    values = array[finite]
    return float(np.min(values)), float(np.max(values)), nonfinite


def maximum_sum_error(array: np.ndarray, axis: int) -> float:
    sums = np.sum(array, axis=axis)
    finite = np.isfinite(sums)
    if not np.any(finite):
        return math.nan
    return float(np.max(np.abs(sums[finite] - 1.0)))


def array_bound_violation(array: np.ndarray, lower: float, upper: float) -> float:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return math.nan
    return float(max(0.0, lower - float(np.min(finite)), float(np.max(finite)) - upper))


def inspect_program(
    case_id: str,
    cohort: str,
    program: dict[str, Any],
    run: dict[str, Any],
    flags: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    channels = {
        name: program.get(name, {})
        for name in ("inlet_flow", "inlet_temperature", "outlet_pressure")
    }
    extrema: dict[str, tuple[float, float]] = {}
    for name, channel in channels.items():
        values = channel_target_values(channel) if isinstance(channel, dict) else []
        if not values:
            add_flag(
                flags,
                case_id=case_id,
                cohort=cohort,
                category="invalid_input",
                severity="error",
                metric=name,
                value="",
                threshold="finite initial/targets",
                detail="No finite scalar channel values were found.",
            )
            extrema[name] = (math.nan, math.nan)
        else:
            extrema[name] = (min(values), max(values))

    flow_channel = channels["inlet_flow"]
    flow_basis = flow_channel.get("basis", "mol_per_s") if isinstance(flow_channel, dict) else "unknown"
    metrics.update(
        {
            "input_flow_basis": flow_basis,
            "input_flow_min": extrema["inlet_flow"][0],
            "input_flow_max": extrema["inlet_flow"][1],
            "input_temperature_min_k": extrema["inlet_temperature"][0],
            "input_temperature_max_k": extrema["inlet_temperature"][1],
            "input_outlet_pressure_min_pa": extrema["outlet_pressure"][0],
            "input_outlet_pressure_max_pa": extrema["outlet_pressure"][1],
        }
    )

    if cohort == "core":
        bounds = {
            "inlet_flow": CORE_FLOW_BOUNDS_GHSV_H_1,
            "inlet_temperature": CORE_TEMPERATURE_BOUNDS_K,
            "outlet_pressure": CORE_PRESSURE_BOUNDS_PA,
        }
        if flow_basis != "ghsv_per_h":
            add_flag(
                flags,
                case_id=case_id,
                cohort=cohort,
                category="design_bound",
                severity="error",
                metric="input_flow_basis",
                value=flow_basis,
                threshold="ghsv_per_h",
                detail="Core generated program has the wrong inlet-flow basis.",
            )
        for name, (lower, upper) in bounds.items():
            observed_min, observed_max = extrema[name]
            if observed_min < lower - PHYSICAL_TOLERANCE or observed_max > upper + PHYSICAL_TOLERANCE:
                add_flag(
                    flags,
                    case_id=case_id,
                    cohort=cohort,
                    category="design_bound",
                    severity="error",
                    metric=name,
                    value=f"[{observed_min:.12g}, {observed_max:.12g}]",
                    threshold=f"[{lower:.12g}, {upper:.12g}]",
                    detail="Core generated program is outside the generator's declared range.",
                )

    composition = program.get("inlet_composition", {})
    compositions: list[tuple[str, dict[str, Any]]] = []
    if isinstance(composition, dict) and isinstance(composition.get("initial"), dict):
        compositions.append(("initial", composition["initial"]))
    if isinstance(composition, dict):
        for index, step in enumerate(composition.get("steps", [])):
            if isinstance(step, dict) and step.get("kind") == "ramp" and isinstance(step.get("target"), dict):
                compositions.append((f"step[{index}]", step["target"]))
    max_sum_error = 0.0
    max_bound_violation = 0.0
    for label, values_by_species in compositions:
        values = np.asarray(finite_floats(values_by_species.values()), dtype=float)
        if values.size != len(values_by_species):
            add_flag(
                flags,
                case_id=case_id,
                cohort=cohort,
                category="invalid_input",
                severity="error",
                metric="inlet_composition",
                value=label,
                threshold="all finite",
                detail="Composition contains a non-finite/non-numeric value.",
            )
            continue
        max_sum_error = max(max_sum_error, abs(float(values.sum()) - 1.0))
        max_bound_violation = max(max_bound_violation, array_bound_violation(values, 0.0, 1.0))
    metrics["input_composition_sum_error_max"] = max_sum_error
    metrics["input_composition_bound_violation_max"] = max_bound_violation
    if max_sum_error > COMPOSITION_SUM_TOLERANCE:
        add_flag(
            flags,
            case_id=case_id,
            cohort=cohort,
            category="invalid_input",
            severity="error",
            metric="input_composition_sum_error_max",
            value=max_sum_error,
            threshold=COMPOSITION_SUM_TOLERANCE,
            detail="An input mole-fraction vector does not sum to one.",
        )
    if max_bound_violation > PHYSICAL_TOLERANCE:
        add_flag(
            flags,
            case_id=case_id,
            cohort=cohort,
            category="invalid_input",
            severity="error",
            metric="input_composition_bound_violation_max",
            value=max_bound_violation,
            threshold=PHYSICAL_TOLERANCE,
            detail="An input mole fraction is outside [0, 1].",
        )

    durations: list[float] = []
    for channel in program.values():
        if not isinstance(channel, dict):
            continue
        for step in channel.get("steps", []):
            if isinstance(step, dict):
                durations.extend(finite_floats((step.get("duration_s"),)))
    metrics["input_step_duration_min_s"] = min(durations) if durations else math.nan
    if durations and min(durations) <= 0.0:
        add_flag(
            flags,
            case_id=case_id,
            cohort=cohort,
            category="invalid_input",
            severity="error",
            metric="input_step_duration_min_s",
            value=min(durations),
            threshold="> 0",
            detail="Program contains a non-positive step duration.",
        )

    model = run.get("model", {})
    radius = float(model.get("bed_radius_m", math.nan))
    length = float(model.get("bed_length_m", math.nan))
    if flow_basis == "ghsv_per_h" and math.isfinite(radius) and math.isfinite(length):
        scale = math.pi * radius**2 * length * NORMAL_MOLAR_DENSITY_MOL_PER_M3 / 3600.0
        metrics["expected_inlet_flow_min_mol_s"] = extrema["inlet_flow"][0] * scale
        metrics["expected_inlet_flow_max_mol_s"] = extrema["inlet_flow"][1] * scale
    elif flow_basis == "mol_per_s":
        metrics["expected_inlet_flow_min_mol_s"] = extrema["inlet_flow"][0]
        metrics["expected_inlet_flow_max_mol_s"] = extrema["inlet_flow"][1]
    else:
        metrics["expected_inlet_flow_min_mol_s"] = math.nan
        metrics["expected_inlet_flow_max_mol_s"] = math.nan
    return metrics


def sampled_values(dataset: xr.Dataset, variable: str, stride: int) -> np.ndarray:
    data_array = dataset[variable]
    if stride > 1 and "time" in data_array.dims and data_array.ndim > 1:
        data_array = data_array.isel(time=slice(None, None, stride))
    return np.asarray(data_array.values, dtype=float)


def inspect_result(
    case_id: str,
    cohort: str,
    result_path: Path,
    status: str,
    input_metrics: dict[str, Any],
    stride: int,
    flags: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"result_path": str(result_path), "result_present": result_path.is_file()}
    if not result_path.is_file():
        if status == "success":
            add_flag(
                flags,
                case_id=case_id,
                cohort=cohort,
                category="missing_result",
                severity="error",
                metric="results.nc",
                value="missing",
                threshold="present for successful case",
                detail="Successful summary row has no NetCDF result.",
            )
        return metrics

    with xr.open_dataset(result_path, cache=False, decode_cf=False) as dataset:
        missing = [name for name in NUMERIC_RESULT_VARIABLES if name not in dataset]
        if missing:
            add_flag(
                flags,
                case_id=case_id,
                cohort=cohort,
                category="invalid_result",
                severity="error",
                metric="dataset_variables",
                value=";".join(missing),
                threshold="all expected variables present",
                detail="NetCDF result is missing variables.",
            )
        time = np.asarray(dataset["time"].values, dtype=float) if "time" in dataset else np.asarray([])
        metrics["result_time_points"] = int(time.size)
        metrics["result_time_start_s"] = float(time[0]) if time.size else math.nan
        metrics["result_time_end_s"] = float(time[-1]) if time.size else math.nan
        metrics["result_time_strictly_increasing"] = bool(time.size < 2 or np.all(np.diff(time) > 0.0))
        if status == "success" and (time.size != 12001 or not np.isclose(time[-1], 12000.0)):
            add_flag(
                flags,
                case_id=case_id,
                cohort=cohort,
                category="incomplete_result",
                severity="error",
                metric="time",
                value=f"n={time.size}, end={time[-1] if time.size else math.nan}",
                threshold="n=12001, end=12000",
                detail="Successful trajectory is not complete.",
            )
        if not metrics["result_time_strictly_increasing"]:
            add_flag(
                flags,
                case_id=case_id,
                cohort=cohort,
                category="invalid_result",
                severity="error",
                metric="time",
                value="not strictly increasing",
                threshold="strictly increasing",
                detail="Result time coordinate is not monotonic.",
            )

        arrays: dict[str, np.ndarray] = {}
        for name in NUMERIC_RESULT_VARIABLES:
            if name not in dataset:
                continue
            array = sampled_values(dataset, name, stride)
            arrays[name] = array
            minimum, maximum, nonfinite = scalar_extrema(array)
            metrics[f"result_{name}_min"] = minimum
            metrics[f"result_{name}_max"] = maximum
            metrics[f"result_{name}_nonfinite_count"] = nonfinite
            if nonfinite:
                add_flag(
                    flags,
                    case_id=case_id,
                    cohort=cohort,
                    category="nonfinite_result",
                    severity="error",
                    metric=name,
                    value=nonfinite,
                    threshold=0,
                    detail="Result variable contains NaN or infinity.",
                )

        rename = {
            "result_temperature_min": "result_temperature_min_k",
            "result_temperature_max": "result_temperature_max_k",
            "result_pressure_min": "result_pressure_min_pa",
            "result_pressure_max": "result_pressure_max_pa",
            "result_pressure_drop_min": "result_pressure_drop_min_pa",
            "result_pressure_drop_max": "result_pressure_drop_max_pa",
            "result_inlet_flow_min": "result_inlet_flow_min_mol_s",
            "result_inlet_flow_max": "result_inlet_flow_max_mol_s",
            "result_outlet_flow_min": "result_outlet_flow_min_mol_s",
            "result_outlet_flow_max": "result_outlet_flow_max_mol_s",
        }
        for source, destination in rename.items():
            if source in metrics:
                metrics[destination] = metrics[source]

        if "gas_flux" in arrays:
            metrics["result_gas_flux_abs_max_mol_m2_s"] = float(np.nanmax(np.abs(arrays["gas_flux"])))
        if "inlet_flow" in arrays and "outlet_flow" in arrays:
            inlet = arrays["inlet_flow"]
            outlet = arrays["outlet_flow"]
            valid = np.isfinite(inlet) & np.isfinite(outlet) & (inlet > 1.0e-15)
            metrics["result_outlet_to_inlet_flow_ratio_max"] = (
                float(np.max(outlet[valid] / inlet[valid])) if np.any(valid) else math.nan
            )
        if "outlet_species_flow" in arrays and "outlet_flow" in arrays:
            species_axis = dataset["outlet_species_flow"].dims.index("gas_species")
            mismatch = np.abs(np.sum(arrays["outlet_species_flow"], axis=species_axis) - arrays["outlet_flow"])
            metrics["result_outlet_flow_sum_error_max_mol_s"] = float(np.nanmax(mismatch))

        for name, species_dimension in (
            ("gas_mole_fraction", "gas_species"),
            ("inlet_composition", "gas_species"),
            ("outlet_composition", "gas_species"),
            ("solid_mole_fraction", "solid_species"),
        ):
            if name not in arrays:
                continue
            axis = dataset[name].dims.index(species_dimension)
            sum_error = maximum_sum_error(arrays[name], axis)
            violation = array_bound_violation(arrays[name], 0.0, 1.0)
            metrics[f"result_{name}_sum_error_max"] = sum_error
            metrics[f"result_{name}_bound_violation_max"] = violation
            if violation > PHYSICAL_TOLERANCE:
                add_flag(
                    flags,
                    case_id=case_id,
                    cohort=cohort,
                    category="physical_bound",
                    severity="error",
                    metric=name,
                    value=violation,
                    threshold=PHYSICAL_TOLERANCE,
                    detail="Reported mole fraction is outside [0, 1].",
                )
            if sum_error > COMPOSITION_SUM_TOLERANCE:
                add_flag(
                    flags,
                    case_id=case_id,
                    cohort=cohort,
                    category="composition_closure",
                    severity="error",
                    metric=name,
                    value=sum_error,
                    threshold=COMPOSITION_SUM_TOLERANCE,
                    detail="Reported mole fractions do not sum to one.",
                )

        physical_lower_bounds = {
            "temperature": 0.0,
            "pressure": 0.0,
            "inlet_pressure": 0.0,
            "outlet_pressure": 0.0,
            "inlet_flow": 0.0,
            "outlet_flow": 0.0,
        }
        for name, lower_bound in physical_lower_bounds.items():
            if name in arrays:
                minimum = scalar_extrema(arrays[name])[0]
                if minimum < lower_bound - PHYSICAL_TOLERANCE:
                    add_flag(
                        flags,
                        case_id=case_id,
                        cohort=cohort,
                        category="physical_bound",
                        severity="error",
                        metric=name,
                        value=minimum,
                        threshold=f">= {lower_bound}",
                        detail="Result violates a basic physical lower bound.",
                    )

        expected_flow_max = float(input_metrics.get("expected_inlet_flow_max_mol_s", math.nan))
        observed_flow_max = float(metrics.get("result_inlet_flow_max_mol_s", math.nan))
        if math.isfinite(expected_flow_max) and math.isfinite(observed_flow_max):
            allowance = max(1.0e-10, 1.0e-4 * abs(expected_flow_max))
            if observed_flow_max > expected_flow_max + allowance:
                add_flag(
                    flags,
                    case_id=case_id,
                    cohort=cohort,
                    category="input_result_mismatch",
                    severity="error",
                    metric="inlet_flow",
                    value=observed_flow_max,
                    threshold=f"<= configured maximum {expected_flow_max:.12g}",
                    detail="Compiled simulation inlet flow exceeds the program maximum.",
                )
        expected_flow_min = float(input_metrics.get("expected_inlet_flow_min_mol_s", math.nan))
        observed_flow_min = float(metrics.get("result_inlet_flow_min_mol_s", math.nan))
        if math.isfinite(expected_flow_min) and math.isfinite(observed_flow_min):
            allowance = max(1.0e-10, 1.0e-4 * abs(expected_flow_min))
            if observed_flow_min < expected_flow_min - allowance:
                add_flag(
                    flags,
                    case_id=case_id,
                    cohort=cohort,
                    category="input_result_mismatch",
                    severity="error",
                    metric="inlet_flow",
                    value=observed_flow_min,
                    threshold=f">= configured minimum {expected_flow_min:.12g}",
                    detail="Compiled simulation inlet flow is below the program minimum.",
                )
    return metrics


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile))


def add_statistical_flags(
    rows: list[dict[str, Any]],
    flags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    eligible = [row for row in rows if row["cohort"] == "core" and row["status"] == "success"]
    for metric in STATISTICAL_METRICS:
        pairs: list[tuple[dict[str, Any], float]] = []
        for row in eligible:
            values = finite_floats((row.get(metric),))
            if values:
                pairs.append((row, values[0]))
        if not pairs:
            continue
        array = np.asarray([value for _, value in pairs], dtype=float)
        minimum_index = int(np.argmin(array))
        maximum_index = int(np.argmax(array))
        q1, median, q3 = (percentile(array, q) for q in (0.25, 0.5, 0.75))
        iqr = q3 - q1
        lower_fence = q1 - 3.0 * iqr
        upper_fence = q3 + 3.0 * iqr
        p01, p99 = (percentile(array, q) for q in (0.01, 0.99))
        mad = float(np.median(np.abs(array - median)))
        outliers: list[tuple[dict[str, Any], float, str, float]] = []
        for row, value in pairs:
            reasons: list[tuple[str, float]] = []
            if iqr > 0.0 and value < lower_fence:
                reasons.append(("below 3-IQR outer fence", lower_fence))
            if iqr > 0.0 and value > upper_fence:
                reasons.append(("above 3-IQR outer fence", upper_fence))
            if p99 > 0.0 and value > 2.0 * p99:
                reasons.append(("more than 2x the 99th percentile", 2.0 * p99))
            if p01 < 0.0 and value < 2.0 * p01:
                reasons.append(("more negative than 2x the 1st percentile", 2.0 * p01))
            if reasons:
                reason, threshold = reasons[0]
                outliers.append((row, value, reason, threshold))
        for row, value, reason, threshold in outliers:
            add_flag(
                flags,
                case_id=str(row["case_id"]),
                cohort="core",
                category="statistical_outlier",
                severity="review",
                metric=metric,
                value=value,
                threshold=threshold,
                detail=reason,
            )
        summaries.append(
            {
                "metric": metric,
                "count": int(array.size),
                "min": float(np.min(array)),
                "min_case_id": str(pairs[minimum_index][0]["case_id"]),
                "p01": p01,
                "q1": q1,
                "median": median,
                "q3": q3,
                "p99": p99,
                "max": float(np.max(array)),
                "max_case_id": str(pairs[maximum_index][0]["case_id"]),
                "lower_outer_fence": lower_fence,
                "upper_outer_fence": upper_fence,
                "median_absolute_deviation": mad,
                "flag_count": len(outliers),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def status_reason(status: str, error: str) -> str:
    if status == "success":
        return "success"
    if status == "timeout_failed":
        return "timeout"
    match = re.search(r"IDA_[A-Z_]+", error)
    return match.group(0) if match else status


def markdown_report(
    rows: list[dict[str, Any]],
    flags: list[dict[str, Any]],
    metric_summaries: list[dict[str, Any]],
    duplicate_groups: list[list[str]],
    stride: int,
) -> str:
    status_counts = Counter(str(row["status"]) for row in rows)
    cohort_status = Counter((str(row["cohort"]), str(row["status"])) for row in rows)
    flag_categories = Counter(str(flag["category"]) for flag in flags)
    status_by_case = {str(row["case_id"]): str(row["status"]) for row in rows}
    successful_result_errors = sum(
        1
        for flag in flags
        if flag["severity"] == "error"
        and flag["category"] not in {"design_bound", "invalid_input"}
        and status_by_case.get(str(flag["case_id"])) == "success"
    )
    success_rows = [row for row in rows if row["status"] == "success"]
    core_success = [row for row in success_rows if row["cohort"] == "core"]
    lines = [
        "# ML batch outlier audit",
        "",
        f"Audited {len(rows)} configured cases; spatial result stride: {stride}.",
        "",
        "## Completion",
        "",
        "| Status | Cases |",
        "|---|---:|",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "| Cohort | Success | Timeout | Simulation failed | Total |", "|---|---:|---:|---:|---:|"])
    for cohort in sorted({str(row["cohort"]) for row in rows}):
        success = cohort_status[(cohort, "success")]
        timeout = cohort_status[(cohort, "timeout_failed")]
        failed = cohort_status[(cohort, "simulation_failed")]
        lines.append(f"| {cohort} | {success} | {timeout} | {failed} | {success + timeout + failed} |")
    lines.extend(
        [
            "",
            "## Data-quality findings",
            "",
            f"- Successful trajectories: {len(success_rows)} ({len(core_success)} numbered core).",
            f"- Exact duplicate input groups: {len(duplicate_groups)}.",
            f"- Total audit flags: {len(flags)}.",
            f"- Hard result-validity flags on successful trajectories: {successful_result_errors}.",
        ]
    )
    for category, count in sorted(flag_categories.items()):
        lines.append(f"- `{category}` flags: {count}.")
    lines.extend(
        [
            "",
            "The 20 `test`/`css` programs are reported separately from the 1,248 numbered core programs. "
            "Those numbered programs include curated early edge cases as well as generated programs. "
            "Generator-range checks and statistical fences are applied to the numbered cohort so those edge cases remain visible.",
            "",
            "## Core successful-case ranges",
            "",
            "| Metric | Min | P01 | Median | P99 | Max | Max case | Review flags |",
            "|---|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for item in metric_summaries:
        lines.append(
            "| {metric} | {min:.7g} | {p01:.7g} | {median:.7g} | {p99:.7g} | {max:.7g} | {max_case_id} | {flag_count} |".format(
                **item
            )
        )
    review_flags = [flag for flag in flags if flag["category"] == "statistical_outlier"]
    lines.extend(["", "## Statistical review candidates", ""])
    if review_flags:
        lines.extend(["| Case | Metric | Value | Detail |", "|---|---|---:|---|"])
        for flag in review_flags:
            lines.append(f"| {flag['case_id']} | {flag['metric']} | {flag['value']} | {flag['detail']} |")
    else:
        lines.append("No core successful trajectory crossed a conservative 3-IQR or 2x-P99 fence.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Cases that did not finish successfully should be excluded or rerun before dataset construction. "
            "Rows in `outlier_flags.csv` with severity `error` indicate hard input/result validity failures; "
            "severity `review` indicates a statistically unusual but not necessarily invalid trajectory.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.result_stride <= 0:
        raise SystemExit("--result-stride must be positive")
    batch_directory = args.batch_directory.resolve()
    report_directory = (
        args.report_directory.resolve()
        if args.report_directory
        else batch_directory / "output" / "outlier_audit"
    )
    summary_path = batch_directory / "output" / "summary.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    flags: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    semantic_hashes: dict[str, list[str]] = defaultdict(list)
    total = len(summary_rows)
    for index, summary in enumerate(summary_rows, start=1):
        case_id = str(summary["case_id"])
        cohort = program_cohort(str(summary.get("axis_program", "")))
        case_directory = Path(summary["case_directory"])
        program_path = case_directory / "program.yaml"
        run_path = Path(summary["run_yaml"])
        result_path = Path(summary.get("output_directory", case_directory / "output")) / "results.nc"
        program = read_yaml(program_path)
        run = read_yaml(run_path)
        normalized = json.dumps(program, sort_keys=True, separators=(",", ":"), allow_nan=False)
        semantic_hashes[hashlib.sha256(normalized.encode("utf-8")).hexdigest()].append(case_id)

        row: dict[str, Any] = {
            "case_id": case_id,
            "axis_program": summary.get("axis_program", ""),
            "cohort": cohort,
            "status": summary.get("status", ""),
            "failure_reason": status_reason(str(summary.get("status", "")), str(summary.get("error", ""))),
            "runtime_s": summary.get("runtime_s", ""),
            "program_path": str(program_path),
        }
        input_metrics = inspect_program(case_id, cohort, program, run, flags)
        row.update(input_metrics)
        try:
            row.update(
                inspect_result(
                    case_id,
                    cohort,
                    result_path,
                    str(summary.get("status", "")),
                    input_metrics,
                    args.result_stride,
                    flags,
                )
            )
        except Exception as exc:  # preserve the rest of a large audit
            row["result_read_error"] = f"{type(exc).__name__}: {exc}"
            add_flag(
                flags,
                case_id=case_id,
                cohort=cohort,
                category="result_read_error",
                severity="error",
                metric="results.nc",
                value=type(exc).__name__,
                threshold="readable NetCDF",
                detail=str(exc),
            )
        rows.append(row)
        if index == 1 or index % 25 == 0 or index == total:
            print(f"Audited {index}/{total} cases", file=sys.stderr, flush=True)

    duplicate_groups = [case_ids for case_ids in semantic_hashes.values() if len(case_ids) > 1]
    for case_ids in duplicate_groups:
        for case_id in case_ids:
            row = next(row for row in rows if row["case_id"] == case_id)
            add_flag(
                flags,
                case_id=case_id,
                cohort=str(row["cohort"]),
                category="duplicate_input",
                severity="review",
                metric="program",
                value=";".join(case_ids),
                threshold="unique program",
                detail="Program is semantically identical to another batch case.",
            )

    metric_summaries = add_statistical_flags(rows, flags)
    flags.sort(key=lambda item: (item["severity"], item["category"], item["case_id"], item["metric"]))
    report_directory.mkdir(parents=True, exist_ok=True)
    write_csv(report_directory / "per_case_metrics.csv", rows)
    write_csv(
        report_directory / "failed_cases.csv",
        [row for row in rows if row["status"] != "success"],
    )
    write_csv(report_directory / "outlier_flags.csv", flags)
    write_csv(report_directory / "metric_summary.csv", metric_summaries)
    write_csv(
        report_directory / "duplicate_groups.csv",
        [
            {"group": index, "case_count": len(group), "case_ids": ";".join(group)}
            for index, group in enumerate(duplicate_groups, start=1)
        ],
    )
    (report_directory / "report.md").write_text(
        markdown_report(rows, flags, metric_summaries, duplicate_groups, args.result_stride),
        encoding="utf-8",
    )
    print(f"Wrote audit reports to {report_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
