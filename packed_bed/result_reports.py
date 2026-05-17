from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from daetools.pyDAE import daeDataReporterLocal

from .config import DEFAULT_SMOOTH_RAMP_WIDTH_S, RunResult
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
    "velocity": VariableReportSpec(
        variable_name="u_s",
        axes=(ReportAxisSpec("x_face", "x_face_m"),),
        value_column_name="velocity_m_per_s",
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
        "terms": (
            "in_total_J",
            "out_total_J",
            "loss_total_J",
            "bed_total_J",
            "balance_error_J",
        ),
        "error_key": "heat",
        "unit": "J",
    },
    "mass_balance": {
        "variables": DERIVED_REPORT_VARIABLE_NAMES["mass_balance"],
        "terms": (
            "in_total_kg",
            "out_total_kg",
            "bed_total_kg",
            "balance_error_kg",
        ),
        "error_key": "mass",
        "unit": "kg",
    },
}

SPATIAL_AXIS_KINDS = {"x_cell", "x_face"}
REPORTS_FILENAME = "reports.pkl"
BALANCES_FILENAME = "balances.pkl"
FLOW_ATOL = 1e-12


def _require_process(run_result: RunResult):
    reporter = run_result.reporter
    if reporter is None or not hasattr(reporter, "Process"):
        raise ValueError("RunResult does not contain a DAETools reporter with a Process payload.")

    process = reporter.Process
    if process is None or not hasattr(process, "dictVariables"):
        raise ValueError("RunResult reporter does not expose Process.dictVariables.")
    return process


def _find_variable(process: Any, variable_name: str, *, required: bool = True):
    matches = sorted(
        key
        for key in process.dictVariables
        if key == variable_name or key.endswith(f".{variable_name}")
    )
    if not matches:
        if not required:
            return None
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


def _axis_column_multi_index(
    axes: tuple[ReportAxisSpec, ...],
    axis_values: tuple[tuple[Any, ...], ...],
) -> pd.MultiIndex:
    if not axes:
        return pd.MultiIndex.from_tuples([("value",)], names=("value",))

    tuples = [
        tuple(axis_values[axis_index][coordinate] for axis_index, coordinate in enumerate(value_index))
        for value_index in np.ndindex(*(len(values_for_axis) for values_for_axis in axis_values))
    ]
    return pd.MultiIndex.from_tuples(
        tuples,
        names=tuple(axis.column_name for axis in axes),
    )


def _with_feature_level(dataframe: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    tuples = [(feature_name, *tuple(column)) for column in dataframe.columns]
    return pd.DataFrame(
        dataframe.to_numpy(copy=False),
        index=dataframe.index,
        columns=pd.MultiIndex.from_tuples(
            tuples,
            names=("feature", *tuple(dataframe.columns.names)),
        ),
    )


def _smooth_ramp_width_s_for_result(run_result: RunResult) -> float:
    simulation = getattr(run_result, "simulation", None)
    return float(getattr(simulation, "smooth_ramp_width_s", DEFAULT_SMOOTH_RAMP_WIDTH_S))


def _sample_scalar_program(run_result: RunResult, channel_config: Any, time_s: np.ndarray) -> np.ndarray:
    program = channel_config.compile_program(
        repeat=run_result.run_bundle.run.repeat_program,
        time_horizon=run_result.run_bundle.run.time_horizon_s,
    )
    smooth_ramp_width_s = _smooth_ramp_width_s_for_result(run_result)
    return np.asarray(
        [
            program.smoothed_value_at(float(time_value), smooth_ramp_width_s=smooth_ramp_width_s)
            for time_value in time_s
        ],
        dtype=float,
    )


def _sample_inlet_composition(run_result: RunResult, time_s: np.ndarray) -> np.ndarray:
    gas_species = tuple(run_result.run_bundle.chemistry.gas_species)
    program = run_result.run_bundle.program.inlet_composition.compile_program(
        gas_species,
        repeat=run_result.run_bundle.run.repeat_program,
        time_horizon=run_result.run_bundle.run.time_horizon_s,
    )
    smooth_ramp_width_s = _smooth_ramp_width_s_for_result(run_result)
    sampled = np.empty((time_s.size, len(gas_species)), dtype=float)
    for time_index, time_value in enumerate(time_s):
        sampled[time_index, :] = np.asarray(
            program.smoothed_value_at(float(time_value), smooth_ramp_width_s=smooth_ramp_width_s),
            dtype=float,
        )
    return sampled


def _optional_values(
    process: Any,
    variable_name: str,
    *,
    label: str,
    reference_time_s: np.ndarray,
    expected_ndim: int,
) -> np.ndarray | None:
    variable = _find_variable(process, variable_name, required=False)
    if variable is None:
        return None

    time_s, values = _time_and_values(variable, label=label)
    _require_same_time_axis(reference_time_s, time_s, label=label)
    if values.ndim != expected_ndim:
        raise ValueError(f"{label} expected {expected_ndim} dimensions including time; got shape {values.shape}.")
    return values


def _optional_scalar_values(
    process: Any,
    variable_name: str,
    *,
    label: str,
    reference_time_s: np.ndarray,
) -> np.ndarray | None:
    variable = _find_variable(process, variable_name, required=False)
    if variable is None:
        return None

    time_s, values = _scalar_series(variable, label=label)
    _require_same_time_axis(reference_time_s, time_s, label=label)
    return values


def _outlet_species_flow_mol_s(
    run_result: RunResult,
    process: Any,
    reference_time_s: np.ndarray,
) -> np.ndarray | None:
    gas_flux = _optional_values(
        process,
        "N_gas_face",
        label="gas flux report",
        reference_time_s=reference_time_s,
        expected_ndim=3,
    )
    if gas_flux is None:
        return None

    cross_section_area_m2 = math.pi * run_result.run_bundle.run.model.bed_radius_m**2
    return cross_section_area_m2 * gas_flux[:, :, -1]


def _outlet_composition(
    run_result: RunResult,
    process: Any,
    reference_time_s: np.ndarray,
    gas_mole_fraction: np.ndarray,
) -> np.ndarray:
    outlet_composition = gas_mole_fraction[:, :, -1].copy()
    outlet_species_flow = _outlet_species_flow_mol_s(run_result, process, reference_time_s)
    if outlet_species_flow is None:
        return outlet_composition

    total_outlet_flow = outlet_species_flow.sum(axis=1, keepdims=True)
    has_outlet_flow = np.abs(total_outlet_flow[:, 0]) > FLOW_ATOL
    outlet_composition[has_outlet_flow] = np.divide(
        outlet_species_flow[has_outlet_flow],
        total_outlet_flow[has_outlet_flow],
    )
    return outlet_composition


def _build_gas_mole_fraction_dataframe(
    run_result: RunResult,
    process: Any,
    variable: Any,
    time_s: np.ndarray,
    values: np.ndarray,
) -> pd.DataFrame:
    if values.ndim != 3:
        raise ValueError(f"gas_mole_fraction report expected 3 dimensions including time; got shape {values.shape}.")

    gas_species = tuple(run_result.run_bundle.chemistry.gas_species)
    if values.shape[1] != len(gas_species):
        raise ValueError(
            f"gas_mole_fraction report has {values.shape[1]} gas species, but the run configuration provides {len(gas_species)} labels."
        )

    cell_positions_m = _domain_points(variable, 1, values.shape[2])
    bed_length_m = float(run_result.run_bundle.run.model.bed_length_m)
    x_positions_m = (0.0, *cell_positions_m, bed_length_m)

    augmented_values = np.empty((time_s.size, len(gas_species), len(x_positions_m)), dtype=float)
    augmented_values[:, :, 0] = _sample_inlet_composition(run_result, time_s)
    augmented_values[:, :, 1:-1] = values
    augmented_values[:, :, -1] = _outlet_composition(run_result, process, time_s, values)

    return pd.DataFrame(
        augmented_values.reshape(time_s.size, -1),
        index=pd.Index(time_s, name="time_s"),
        columns=pd.MultiIndex.from_product(
            (gas_species, x_positions_m),
            names=("gas_species", "x_cell_m"),
        ),
    )


def _build_boundary_report_dataframe(run_result: RunResult, reference_time_s: np.ndarray) -> pd.DataFrame:
    process = _require_process(run_result)
    bed_length_m = float(run_result.run_bundle.run.model.bed_length_m)

    series: list[np.ndarray] = [
        _sample_scalar_program(run_result, run_result.run_bundle.program.inlet_temperature, reference_time_s),
        _sample_scalar_program(run_result, run_result.run_bundle.program.inlet_flow, reference_time_s),
    ]
    columns: list[tuple[str, float]] = [
        ("inlet_temperature_k", 0.0),
        ("inlet_flowrate_mol_s", 0.0),
    ]

    outlet_species_flow = _outlet_species_flow_mol_s(run_result, process, reference_time_s)
    if outlet_species_flow is not None:
        series.append(outlet_species_flow.sum(axis=1))
        columns.append(("outlet_flowrate_mol_s", bed_length_m))

    series.append(_sample_scalar_program(run_result, run_result.run_bundle.program.outlet_pressure, reference_time_s))
    columns.append(("outlet_pressure_pa", bed_length_m))

    inlet_pressure = _optional_scalar_values(
        process,
        "P_in",
        label="inlet pressure variable",
        reference_time_s=reference_time_s,
    )
    if inlet_pressure is not None:
        series.append(inlet_pressure)
        columns.append(("inlet_pressure_pa", 0.0))

    return pd.DataFrame(
        np.column_stack(series),
        index=pd.Index(reference_time_s, name="time_s"),
        columns=pd.MultiIndex.from_tuples(columns, names=("feature", "x_cell_m")),
    )


def _is_spatial_only_report(report_id: str) -> bool:
    spec = VARIABLE_REPORT_SPECS[report_id]
    return len(spec.axes) == 1 and spec.axes[0].kind in SPATIAL_AXIS_KINDS


def _build_variable_report_dataframe(run_result: RunResult, report_id: str) -> pd.DataFrame:
    spec = VARIABLE_REPORT_SPECS[report_id]
    process = _require_process(run_result)
    variable = _find_variable(process, spec.variable_name)
    times, values = _time_and_values(variable, label=f"{report_id} report")
    expected_ndim = 1 + len(spec.axes)
    if values.ndim != expected_ndim:
        raise ValueError(
            f"{report_id} report expected {expected_ndim} dimensions including time; got shape {values.shape}."
        )
    if report_id == "gas_mole_fraction":
        return _build_gas_mole_fraction_dataframe(run_result, process, variable, times, values)

    axis_values = tuple(
        _axis_values(run_result, variable, axis_spec, axis_index, axis_size)
        for axis_index, (axis_spec, axis_size) in enumerate(zip(spec.axes, values.shape[1:]))
    )

    columns = _axis_column_multi_index(spec.axes, axis_values)
    return pd.DataFrame(
        values.reshape(times.size, -1),
        index=pd.Index(times, name="time_s"),
        columns=columns,
    )


def _balance_series(run_result: RunResult, report_id: str) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
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

    if len(series) < 3:
        raise ValueError(f"Balance report '{report_id}' must contain at least inlet, outlet, and bed totals.")

    in_total = series[0]
    bed_total = series[-1]
    balance_error = (bed_total - bed_total[0]) - in_total
    for exported_total in series[1:-1]:
        balance_error = balance_error + exported_total

    balance_terms = (*series, balance_error)
    if len(spec["terms"]) != len(balance_terms):
        raise ValueError(
            f"Balance report '{report_id}' declares {len(spec['terms'])} terms but produced {len(balance_terms)} series."
        )
    return first_time, balance_terms


def _build_balance_dataframe(run_result: RunResult, report_id: str) -> pd.DataFrame:
    spec = BALANCE_REPORT_SPECS[report_id]
    time_s, balance_terms = _balance_series(run_result, report_id)

    return pd.DataFrame(
        np.column_stack(balance_terms),
        index=pd.Index(time_s, name="time_s"),
        columns=pd.MultiIndex.from_product(
            ((report_id,), spec["terms"]),
            names=("balance", "term"),
        ),
    )


def _concat_same_time_axis(frames: list[pd.DataFrame], *, label: str) -> pd.DataFrame:
    if not frames:
        raise ValueError(f"Cannot build {label} output without any dataframes.")

    reference = frames[0].index.to_numpy(dtype=float)
    for index, frame in enumerate(frames[1:], start=1):
        candidate = frame.index.to_numpy(dtype=float)
        _require_same_time_axis(reference, candidate, label=f"{label} dataframe {index}")
    return pd.concat(frames, axis=1)


def _write_dataframe_pickle(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_pickle(path)


def _safe_pickle_path(output_dir: Path, report_id: str) -> Path:
    safe_report_id = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in report_id)
    return output_dir / f"{safe_report_id}.pkl"


def build_requested_report_dataframes(run_result: RunResult) -> dict[str, pd.DataFrame]:
    requested_report_ids = tuple(run_result.run_bundle.run.outputs.requested_reports)
    dataframes: dict[str, pd.DataFrame] = {}

    unknown_reports = [report_id for report_id in requested_report_ids if report_id not in REPORT_VARIABLE_REGISTRY]
    if unknown_reports:
        raise ValueError(f"Unknown report ids: {', '.join(unknown_reports)}.")

    spatial_frames: list[pd.DataFrame] = []
    balance_frames: list[pd.DataFrame] = []
    variable_report_time_s: np.ndarray | None = None

    for report_id in requested_report_ids:
        if report_id in BALANCE_REPORT_SPECS:
            balance_frames.append(_build_balance_dataframe(run_result, report_id))
            continue

        if report_id not in VARIABLE_REPORT_SPECS:
            raise ValueError(f"Report '{report_id}' does not have a dataframe exporter.")

        dataframe = _build_variable_report_dataframe(run_result, report_id)
        if variable_report_time_s is None:
            variable_report_time_s = dataframe.index.to_numpy(dtype=float)
        if _is_spatial_only_report(report_id):
            spatial_frames.append(_with_feature_level(dataframe, VARIABLE_REPORT_SPECS[report_id].value_column_name))
            continue

        dataframes[report_id] = dataframe

    if variable_report_time_s is not None:
        spatial_frames.append(_build_boundary_report_dataframe(run_result, variable_report_time_s))

    if spatial_frames:
        dataframes["reports"] = _concat_same_time_axis(spatial_frames, label="reports")

    if balance_frames:
        dataframes["balances"] = _concat_same_time_axis(balance_frames, label="balances")

    return dataframes


def _write_report_dataframes(
    dataframes: dict[str, pd.DataFrame],
    requested_report_ids: tuple[str, ...],
    output_dir: Path,
) -> dict[str, Path]:
    report_paths: dict[str, Path] = {}

    for dataframe_id, dataframe in dataframes.items():
        if dataframe_id == "reports":
            output_path = output_dir / REPORTS_FILENAME
        elif dataframe_id == "balances":
            output_path = output_dir / BALANCES_FILENAME
        else:
            output_path = _safe_pickle_path(output_dir, dataframe_id)
        _write_dataframe_pickle(dataframe, output_path)
        report_paths[dataframe_id] = output_path

    if "reports" in report_paths:
        for report_id in requested_report_ids:
            if report_id in VARIABLE_REPORT_SPECS and _is_spatial_only_report(report_id):
                report_paths[report_id] = report_paths["reports"]

    if "balances" in report_paths:
        for report_id in requested_report_ids:
            if report_id in BALANCE_REPORT_SPECS:
                report_paths[report_id] = report_paths["balances"]

    return report_paths


def export_requested_report_pickles(run_result: RunResult, output_dir: str | Path | None = None) -> dict[str, Path]:
    resolved_output_dir = Path(output_dir) if output_dir is not None else run_result.output_directory
    requested_report_ids = tuple(run_result.run_bundle.run.outputs.requested_reports)
    dataframes = build_requested_report_dataframes(run_result)
    return _write_report_dataframes(dataframes, requested_report_ids, resolved_output_dir)


class PackedBedDataFrameReporter(daeDataReporterLocal):
    """DAE Tools reporter that writes requested packed-bed reports as pickled dataframes."""

    def __init__(self, run_bundle, requested_report_ids: tuple[str, ...] | None = None):
        daeDataReporterLocal.__init__(self)
        self.run_bundle = run_bundle
        self.requested_report_ids = (
            tuple(requested_report_ids)
            if requested_report_ids is not None
            else tuple(run_bundle.run.outputs.requested_reports)
        )
        self.output_directory = Path(run_bundle.output_directory)
        self.ProcessName = ""
        self.ConnectString = ""
        self.report_paths: dict[str, Path] = {}
        self.dataframes: dict[str, pd.DataFrame] = {}
        self.write_error: Exception | None = None
        self._connected = False
        self._written = False

    def Connect(self, ConnectString, ProcessName):
        try:
            self.ProcessName = ProcessName
            self.ConnectString = ConnectString
            if ConnectString:
                self.output_directory = Path(ConnectString)
            self.output_directory.mkdir(parents=True, exist_ok=True)
            self._connected = True
            return True
        except Exception:
            traceback.print_exc()
            self._connected = False
            return False

    def Disconnect(self):
        try:
            if not self._written:
                self.write_outputs()
            self._connected = False
            return True
        except Exception as exc:
            self.write_error = exc
            traceback.print_exc()
            self._connected = False
            return False

    def IsConnected(self):
        return self._connected

    def write_outputs(self) -> dict[str, Path]:
        run_result = RunResult(
            run_bundle=self.run_bundle.model_copy(
                update={
                    "run": self.run_bundle.run.model_copy(
                        update={
                            "outputs": self.run_bundle.run.outputs.model_copy(
                                update={"requested_reports": self.requested_report_ids}
                            )
                        }
                    )
                }
            ),
            output_directory=self.output_directory,
            success=True,
            reporter=self,
        )
        self.dataframes = build_requested_report_dataframes(run_result)
        self.report_paths = _write_report_dataframes(
            self.dataframes,
            self.requested_report_ids,
            self.output_directory,
        )
        self._written = True
        return self.report_paths


def compute_balance_errors(run_result: RunResult) -> dict[str, BalanceError]:
    errors: dict[str, BalanceError] = {}
    for report_id, spec in BALANCE_REPORT_SPECS.items():
        time_s, balance_terms = _balance_series(run_result, report_id)
        balance_error = balance_terms[-1]
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
    "BALANCES_FILENAME",
    "PackedBedDataFrameReporter",
    "REPORTS_FILENAME",
    "ReportAxisSpec",
    "VariableReportSpec",
    "build_requested_report_dataframes",
    "compute_balance_errors",
    "export_requested_report_pickles",
    "format_balance_error_lines",
]
