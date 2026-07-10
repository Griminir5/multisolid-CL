"""Result selection, labelled extraction, durable output, balances, and provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np

from .config import Case
from .programs import DEFAULT_SMOOTH_RAMP_WIDTH_S


RESULTS_FILENAME = "results.nc"
MANIFEST_FILENAME = "manifest.json"
TIME_ATOL = 1.0e-12
FLOW_ATOL = 1.0e-12


@dataclass(frozen=True)
class ReportSpec:
    description: str
    model_variables: tuple[str, ...]
    requires_reactions: bool = False


REPORT_SPECS = {
    "temperature": ReportSpec("Bed temperature by cell.", ("temp_bed",)),
    "pressure": ReportSpec("Bed pressure by cell.", ("pres_bed",)),
    "velocity": ReportSpec("Superficial velocity by face.", ("u_s",)),
    "gas_concentration": ReportSpec("Gas concentration by species and cell.", ("c_gas",)),
    "gas_mole_fraction": ReportSpec("Gas mole fraction by species and cell.", ("y_gas",)),
    "solid_concentration": ReportSpec("Solid concentration by species and cell.", ("c_sol",)),
    "solid_mole_fraction": ReportSpec("Derived solid mole fraction.", ("c_sol",)),
    "gas_flux": ReportSpec("Gas molar flux by species and face.", ("N_gas_face",)),
    "reaction_rate": ReportSpec(
        "Reaction rate by reaction and cell.",
        ("R_rxn",),
        requires_reactions=True,
    ),
    "gas_enthalpy_flux": ReportSpec("Gas enthalpy flux by species and face.", ("J_gas_face",)),
    "heat_balance": ReportSpec(
        "Integral heat balance.",
        ("heat_in_total", "heat_out_total", "heat_loss_total", "heat_bed_total"),
    ),
    "mass_balance": ReportSpec(
        "Integral mass balance.",
        ("mass_in_total", "mass_out_total", "mass_bed_total"),
    ),
}

PLOT_REPORT_IDS = ("temperature", "pressure", "gas_mole_fraction", "gas_flux")
PLOT_EXTRA_VARIABLES = ("P_in", "P_out")

# model variable: (dataset variable, non-time dimensions)
MODEL_VARIABLES = {
    "temp_bed": ("temperature", ("x_cell",)),
    "pres_bed": ("pressure", ("x_cell",)),
    "u_s": ("velocity", ("x_face",)),
    "c_gas": ("gas_concentration", ("gas_species", "x_cell")),
    "y_gas": ("gas_mole_fraction", ("gas_species", "x_cell")),
    "c_sol": ("solid_concentration", ("solid_species", "x_cell")),
    "N_gas_face": ("gas_flux", ("gas_species", "x_face")),
    "R_rxn": ("reaction_rate", ("reaction", "x_cell")),
    "J_gas_face": ("gas_enthalpy_flux", ("gas_species", "x_face")),
    "heat_in_total": ("heat_in_total", ()),
    "heat_out_total": ("heat_out_total", ()),
    "heat_loss_total": ("heat_loss_total", ()),
    "heat_bed_total": ("heat_bed_total", ()),
    "mass_in_total": ("mass_in_total", ()),
    "mass_out_total": ("mass_out_total", ()),
    "mass_bed_total": ("mass_bed_total", ()),
    "P_in": ("inlet_pressure", ()),
    "P_out": ("outlet_pressure", ()),
}


@dataclass(frozen=True)
class BalanceError:
    max_abs_error: float
    time_s: float
    unit: str


@dataclass(frozen=True)
class RunResult:
    case: Case
    output_directory: Path
    status: str = "success"
    results_path: Path | None = None
    manifest_path: Path | None = None
    runtime_s: float | None = None
    dataset: Any | None = None
    balance_errors: dict[str, BalanceError] = field(default_factory=dict)
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    reporter: Any | None = None


def reporting_targets(
    report_ids,
    *,
    include_plot_variables: bool = False,
) -> tuple[str, ...]:
    selected = [*report_ids, *(PLOT_REPORT_IDS if include_plot_variables else ())]
    variables = [
        variable
        for report_id in selected
        for variable in REPORT_SPECS[report_id].model_variables
    ]
    if include_plot_variables:
        variables.extend(PLOT_EXTRA_VARIABLES)
    return tuple(dict.fromkeys(variables))


def _find_variable(process, variable_name: str):
    matches = [
        variable
        for name, variable in process.dictVariables.items()
        if name == variable_name or name.endswith(f".{variable_name}")
    ]
    if len(matches) > 1:
        raise ValueError(f"Reporter contains multiple variables named '{variable_name}'.")
    return matches[0] if matches else None


def _time_and_values(variable, label: str) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(variable.TimeValues, dtype=float).reshape(-1)
    values = np.asarray(variable.Values, dtype=float)
    if values.shape[0] != time.size:
        raise ValueError(f"{label} values do not align with their time coordinate.")
    if np.any(np.diff(time) < -TIME_ATOL):
        raise ValueError(f"{label} time coordinates must be non-decreasing.")
    keep = np.ones(time.size, dtype=bool)
    if time.size > 1:
        keep[:-1] = ~np.isclose(time[:-1], time[1:], rtol=0.0, atol=TIME_ATOL)
    return time[keep], values[keep]


def _dimension_coordinate(case: Case, variable, dimension: str, index: int, size: int):
    if dimension == "gas_species":
        values = case.chemistry.gas_species
    elif dimension == "solid_species":
        values = case.solids.solid_species
    elif dimension == "reaction":
        values = case.chemistry.reaction_ids
    else:
        domains = getattr(variable, "Domains", ())
        values = () if index >= len(domains) else getattr(domains[index], "Points", ())
    if len(values) != size:
        raise ValueError(
            f"Reporter dimension '{dimension}' has size {size}, but {len(values)} labels were resolved."
        )
    return np.asarray(values)


def _sample_program(program, time: np.ndarray, smooth_ramp_width_s: float) -> np.ndarray:
    return np.asarray([
        program.value_at(float(value), smooth_ramp_width_s=smooth_ramp_width_s)
        for value in time
    ])


def extract_dataset(
    process,
    case: Case,
    *,
    smooth_ramp_width_s: float = DEFAULT_SMOOTH_RAMP_WIDTH_S,
):
    """Extract every known reported variable into one labelled xarray Dataset."""

    import xarray as xr

    data_vars: dict[str, Any] = {}
    coordinates: dict[str, Any] = {}
    reference_time: np.ndarray | None = None
    for model_name, (dataset_name, dimensions) in MODEL_VARIABLES.items():
        variable = _find_variable(process, model_name)
        if variable is None:
            continue
        time, values = _time_and_values(variable, model_name)
        if reference_time is None:
            reference_time = time
            coordinates["time"] = ("time", time, {"units": "s"})
        elif time.shape != reference_time.shape or not np.allclose(
            time, reference_time, rtol=0.0, atol=TIME_ATOL
        ):
            raise ValueError(f"Reporter variable '{model_name}' has a different time coordinate.")
        if values.ndim != len(dimensions) + 1:
            raise ValueError(
                f"Reporter variable '{model_name}' has shape {values.shape}; expected time plus {dimensions}."
            )
        for index, (dimension, size) in enumerate(zip(dimensions, values.shape[1:])):
            resolved = _dimension_coordinate(case, variable, dimension, index, size)
            if dimension in coordinates and not np.array_equal(coordinates[dimension][1], resolved):
                raise ValueError(f"Reporter variables disagree on coordinate '{dimension}'.")
            attributes = {"units": "m"} if dimension in {"x_cell", "x_face"} else {}
            coordinates[dimension] = (dimension, resolved, attributes)
        units = str(getattr(variable, "Units", ""))
        data_vars[dataset_name] = (
            ("time", *dimensions),
            values,
            {"units": units, "source_variable": model_name},
        )

    if reference_time is None:
        interval = case.run.simulation.reporting_interval_s
        horizon = case.run.simulation.time_horizon_s
        reference_time = interval * np.arange(int(np.floor(horizon / interval)) + 1)
        if np.isclose(reference_time[-1], horizon, rtol=0.0, atol=TIME_ATOL):
            reference_time[-1] = horizon
        else:
            reference_time = np.append(reference_time, horizon)
        coordinates["time"] = ("time", reference_time, {"units": "s"})
    coordinates.setdefault(
        "gas_species",
        ("gas_species", np.asarray(case.chemistry.gas_species)),
    )
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coordinates,
        attrs={
            "system_name": case.run.simulation.system_name,
            "selected_reports": ",".join(case.run.outputs.requested_reports),
        },
    )

    requested_reports = set(case.run.outputs.requested_reports)
    if "solid_concentration" in dataset and "solid_mole_fraction" in requested_reports:
        total = dataset.solid_concentration.sum("solid_species")
        dataset["solid_mole_fraction"] = xr.where(
            total > 0.0,
            dataset.solid_concentration / total,
            0.0,
        )
        dataset.solid_mole_fraction.attrs = {"units": "1", "derived": "true"}
    if "solid_concentration" in dataset and "solid_concentration" not in requested_reports:
        dataset = dataset.drop_vars("solid_concentration")

    time = dataset.time.values
    dataset["inlet_flow"] = (
        "time",
        _sample_program(case.inlet_flow_program, time, smooth_ramp_width_s),
        {"units": "mol/s", "derived": "program"},
    )
    dataset["inlet_temperature"] = (
        "time",
        _sample_program(case.inlet_temperature_program, time, smooth_ramp_width_s),
        {"units": "K", "derived": "program"},
    )
    dataset["programmed_outlet_pressure"] = (
        "time",
        _sample_program(case.outlet_pressure_program, time, smooth_ramp_width_s),
        {"units": "Pa", "derived": "program"},
    )
    dataset["inlet_composition"] = (
        ("time", "gas_species"),
        _sample_program(case.inlet_composition_program, time, smooth_ramp_width_s),
        {"units": "1", "derived": "program"},
    )

    if "temperature" in dataset:
        dataset["outlet_temperature"] = dataset.temperature.isel(x_cell=-1)
    if "pressure" in dataset:
        if "inlet_pressure" not in dataset:
            dataset["inlet_pressure"] = dataset.pressure.isel(x_cell=0)
        if "outlet_pressure" not in dataset:
            dataset["outlet_pressure"] = dataset.programmed_outlet_pressure
        dataset["pressure_drop"] = dataset.inlet_pressure - dataset.outlet_pressure
        for name in ("inlet_pressure", "outlet_pressure", "pressure_drop"):
            dataset[name].attrs = {"units": "Pa", "derived": "true"}
    if "gas_flux" in dataset:
        area = np.pi * case.run.model.bed_radius_m**2
        dataset["outlet_species_flow"] = area * dataset.gas_flux.isel(x_face=-1)
        dataset.outlet_species_flow.attrs = {"units": "mol/s", "derived": "true"}
        dataset["outlet_flow"] = dataset.outlet_species_flow.sum("gas_species")
        dataset.outlet_flow.attrs = {"units": "mol/s", "derived": "true"}
        if "gas_mole_fraction" in dataset:
            fallback = dataset.gas_mole_fraction.isel(x_cell=-1)
            dataset["outlet_composition"] = xr.where(
                abs(dataset.outlet_flow) > FLOW_ATOL,
                dataset.outlet_species_flow / dataset.outlet_flow,
                fallback,
            )
            dataset.outlet_composition.attrs = {"units": "1", "derived": "true"}

    _add_balance_variables(dataset)
    return dataset


def _add_balance_variables(dataset) -> None:
    if all(name in dataset for name in (
        "heat_in_total", "heat_out_total", "heat_loss_total", "heat_bed_total"
    )):
        dataset["heat_balance_error"] = (
            dataset.heat_bed_total - dataset.heat_bed_total.isel(time=0)
            - dataset.heat_in_total + dataset.heat_out_total + dataset.heat_loss_total
        )
        dataset.heat_balance_error.attrs = {"units": "J", "derived": "true"}
    if all(name in dataset for name in (
        "mass_in_total", "mass_out_total", "mass_bed_total"
    )):
        dataset["mass_balance_error"] = (
            dataset.mass_bed_total - dataset.mass_bed_total.isel(time=0)
            - dataset.mass_in_total + dataset.mass_out_total
        )
        dataset.mass_balance_error.attrs = {"units": "kg", "derived": "true"}


def write_dataset(dataset, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    dataset.to_netcdf(temporary, engine="scipy")
    temporary.replace(path)
    return path


def load_dataset(path: str | Path):
    import xarray as xr

    return xr.load_dataset(path, engine="scipy")


def create_dataset_reporter(
    case: Case,
    *,
    smooth_ramp_width_s: float = DEFAULT_SMOOTH_RAMP_WIDTH_S,
):
    """Create the DAETools adapter lazily so pure report metadata stays solver-free."""

    from daetools.pyDAE import daeDataReporterLocal

    class PackedBedDatasetReporter(daeDataReporterLocal):
        def __init__(self):
            daeDataReporterLocal.__init__(self)
            self.ProcessName = ""
            self.ConnectString = ""
            self.output_directory = Path(case.output_directory)
            self.dataset = None
            self.results_path = None
            self.write_error = None
            self._connected = False
            self._written = False

        def Connect(self, connect_string, process_name):
            try:
                self.ProcessName = process_name
                self.ConnectString = connect_string
                self.output_directory = Path(connect_string or case.output_directory)
                self.output_directory.mkdir(parents=True, exist_ok=True)
                self._connected = True
                return True
            except Exception as exc:
                self.write_error = exc
                return False

        def Disconnect(self):
            try:
                if not self._written:
                    self.write_outputs()
                self._connected = False
                return True
            except Exception as exc:
                self.write_error = exc
                self._connected = False
                return False

        def IsConnected(self):
            return self._connected

        def write_outputs(self):
            self.dataset = extract_dataset(
                self.Process,
                case,
                smooth_ramp_width_s=smooth_ramp_width_s,
            )
            self.results_path = write_dataset(
                self.dataset,
                self.output_directory / RESULTS_FILENAME,
            )
            self._written = True
            return self.results_path

    return PackedBedDatasetReporter()


def compute_balance_errors(dataset) -> dict[str, BalanceError]:
    errors = {}
    for key, variable_name, unit in (
        ("heat", "heat_balance_error", "J"),
        ("mass", "mass_balance_error", "kg"),
    ):
        if variable_name not in dataset:
            continue
        values = np.asarray(dataset[variable_name].values, dtype=float)
        if values.size == 0 or np.all(np.isnan(values)):
            continue
        index = int(np.nanargmax(np.abs(values)))
        errors[key] = BalanceError(
            max_abs_error=float(abs(values[index])),
            time_s=float(dataset.time.values[index]),
            unit=unit,
        )
    return errors


def format_balance_error_lines(balance_errors) -> tuple[str, ...]:
    lines = []
    for key in ("heat", "mass"):
        error = balance_errors.get(key)
        if error is not None:
            lines.append(
                f"largest {key} balance error: {error.max_abs_error:.6g} "
                f"{error.unit} at t={error.time_s:.6g} s"
            )
    return tuple(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(path: Path) -> dict[str, Any]:
    try:
        root = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name, distribution in (
        ("xarray", "xarray"),
        ("scipy", "scipy"),
        ("daetools", "daetools"),
    ):
        try:
            versions[name] = version(distribution)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _dataset_inventory(dataset) -> dict[str, Any]:
    if dataset is None:
        return {}
    return {
        "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
        "variables": {
            name: {
                "dimensions": list(variable.dims),
                "units": variable.attrs.get("units", ""),
            }
            for name, variable in dataset.data_vars.items()
        },
    }


def write_run_manifest(
    result: RunResult,
    *,
    failure_stage: str | None = None,
    traceback_text: str | None = None,
) -> Path:
    """Write compact infrastructure provenance and an output inventory."""

    case = result.case
    input_paths = (case.run_path, case.chemistry_path, case.program_path, case.solids_path)
    outputs = {
        "results": result.results_path,
        **result.artifact_paths,
    }
    manifest = {
        "status": result.status,
        "runtime_s": result.runtime_s,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "git": _git_state(case.run_path.parent),
        },
        "configuration": {
            "system_name": case.run.simulation.system_name,
            "reaction_families": [family.name for family in case.reaction_families],
            "reactions": list(case.chemistry.reaction_ids),
            "gas_species": list(case.chemistry.gas_species),
            "solid_species": list(case.solids.solid_species),
            "mass_scheme": case.run.simulation.mass_scheme,
            "heat_scheme": case.run.simulation.heat_scheme,
            "axial_cells": case.run.model.axial_cells,
            "solver": case.run.solver.name,
            "threads": case.run.solver.threads,
            "relative_tolerance": case.run.solver.relative_tolerance,
            "requested_reports": list(case.run.outputs.requested_reports),
        },
        "inputs": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in input_paths if path.is_file()
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in outputs.items() if path is not None and path.is_file()
        },
        "dataset": _dataset_inventory(result.dataset),
        "balances": {name: asdict(error) for name, error in result.balance_errors.items()},
    }
    if failure_stage is not None or traceback_text is not None:
        manifest["failure"] = {
            "stage": failure_stage,
            "traceback": traceback_text,
        }
    path = result.output_directory / MANIFEST_FILENAME
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


__all__ = (
    "BalanceError",
    "MANIFEST_FILENAME",
    "REPORT_SPECS",
    "RESULTS_FILENAME",
    "RunResult",
    "compute_balance_errors",
    "create_dataset_reporter",
    "extract_dataset",
    "format_balance_error_lines",
    "load_dataset",
    "reporting_targets",
    "write_dataset",
    "write_run_manifest",
)
