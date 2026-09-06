"""Selected result extraction, NetCDF output, summaries, and provenance."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from .config import Case


RESULTS_FILENAME = "results.nc"
MANIFEST_FILENAME = "manifest.json"
TIME_ATOL = 1.0e-12
FLOW_ATOL = 1.0e-12


@dataclass(frozen=True)
class ModelField:
    source: str
    output: str
    dimensions: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutputField:
    name: str
    dimensions: tuple[str, ...]


@dataclass(frozen=True)
class ReportSpec:
    description: str
    fields: tuple[ModelField, ...] = ()
    support: tuple[ModelField, ...] = ()
    derived: tuple[OutputField, ...] = ()
    derive: Callable[[Any, Mapping[str, Any], Case], None] | None = None
    requires_reactions: bool = False

    @property
    def model_variables(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(field.source for field in (*self.fields, *self.support)))

    @property
    def outputs(self) -> tuple[OutputField, ...]:
        direct = tuple(
            OutputField(field.output, ("time", *field.dimensions))
            for field in self.fields
        )
        return (*direct, *self.derived)


def _derive_temperature(dataset, _raw, _case) -> None:
    dataset["outlet_temperature"] = dataset.temperature.isel(x_cell=-1, drop=True)
    dataset.outlet_temperature.attrs = {
        "units": dataset.temperature.attrs.get("units", "K"),
        "derived_from": "temperature[x_cell=-1]",
    }


def _derive_pressure(dataset, _raw, _case) -> None:
    dataset["pressure_drop"] = dataset.inlet_pressure - dataset.outlet_pressure
    dataset.pressure_drop.attrs = {
        "units": "Pa",
        "derived_from": "inlet_pressure - outlet_pressure",
    }


def _derive_outlet_composition(dataset, raw, _case) -> None:
    outlet_flux = raw["N_gas_face"].isel(x_face=-1, drop=True)
    total_flux = outlet_flux.sum("gas_species")
    fallback = dataset.gas_mole_fraction.isel(x_cell=-1, drop=True)
    flowing = abs(total_flux) > FLOW_ATOL
    fraction = outlet_flux / total_flux.where(flowing, 1.0)
    dataset["outlet_composition"] = fraction.where(flowing, fallback)
    dataset.outlet_composition.attrs = {
        "units": "1",
        "derived_from": "outlet gas flux; final-cell mole fraction at zero flow",
    }


def _derive_solid_mole_fraction(dataset, raw, _case) -> None:
    concentration = raw["c_sol"]
    total = concentration.sum("solid_species")
    present = total > 0.0
    fraction = concentration / total.where(present, 1.0)
    dataset["solid_mole_fraction"] = fraction.where(present, 0.0)
    dataset.solid_mole_fraction.attrs = {
        "units": "1",
        "derived_from": "c_sol",
    }


def _derive_gas_flux(dataset, _raw, case) -> None:
    area = np.pi * case.run.model.bed_radius_m**2
    dataset["outlet_species_flow"] = area * dataset.gas_flux.isel(x_face=-1, drop=True)
    dataset.outlet_species_flow.attrs = {
        "units": "mol/s",
        "derived_from": "gas_flux[x_face=-1]",
    }
    dataset["outlet_flow"] = dataset.outlet_species_flow.sum("gas_species")
    dataset.outlet_flow.attrs = {
        "units": "mol/s",
        "derived_from": "outlet_species_flow",
    }


def _derive_heat_balance(dataset, _raw, _case) -> None:
    dataset["heat_balance_error"] = (
        dataset.heat_bed_total
        - dataset.heat_bed_total.isel(time=0)
        - dataset.heat_in_total
        + dataset.heat_out_total
        + dataset.heat_loss_total
    )
    dataset.heat_balance_error.attrs = {"units": "J", "derived_from": "heat totals"}


def _derive_mass_balance(dataset, _raw, _case) -> None:
    dataset["mass_balance_error"] = (
        dataset.mass_bed_total
        - dataset.mass_bed_total.isel(time=0)
        - dataset.mass_in_total
        + dataset.mass_out_total
    )
    dataset.mass_balance_error.attrs = {"units": "kg", "derived_from": "mass totals"}


def _field(source: str, output: str, *dimensions: str) -> ModelField:
    return ModelField(source, output, dimensions)


def _output(name: str, *dimensions: str) -> OutputField:
    return OutputField(name, ("time", *dimensions))


REPORT_REGISTRY: Mapping[str, ReportSpec] = MappingProxyType({
    "temperature": ReportSpec(
        "Inlet, cell-centre, and outlet temperature.",
        (_field("T_in", "inlet_temperature"), _field("temp_bed", "temperature", "x_cell")),
        derived=(_output("outlet_temperature"),),
        derive=_derive_temperature,
    ),
    "pressure": ReportSpec(
        "Inlet, cell-centre, outlet, and drop pressure.",
        (
            _field("P_in", "inlet_pressure"),
            _field("pres_bed", "pressure", "x_cell"),
            _field("P_out", "outlet_pressure"),
        ),
        derived=(_output("pressure_drop"),),
        derive=_derive_pressure,
    ),
    "velocity": ReportSpec(
        "Face superficial velocity.",
        (_field("u_s", "velocity", "x_face"),),
    ),
    "gas_concentration": ReportSpec(
        "Gas concentration by species and cell.",
        (_field("c_gas", "gas_concentration", "gas_species", "x_cell"),),
    ),
    "gas_mole_fraction": ReportSpec(
        "Inlet, cell-centre, and outlet gas mole fraction.",
        (
            _field("y_in", "inlet_composition", "gas_species"),
            _field("y_gas", "gas_mole_fraction", "gas_species", "x_cell"),
        ),
        support=(_field("N_gas_face", "", "gas_species", "x_face"),),
        derived=(_output("outlet_composition", "gas_species"),),
        derive=_derive_outlet_composition,
    ),
    "solid_concentration": ReportSpec(
        "Solid concentration by species and cell.",
        (_field("c_sol", "solid_concentration", "solid_species", "x_cell"),),
    ),
    "solid_mole_fraction": ReportSpec(
        "Derived solid mole fraction by species and cell.",
        support=(_field("c_sol", "", "solid_species", "x_cell"),),
        derived=(_output("solid_mole_fraction", "solid_species", "x_cell"),),
        derive=_derive_solid_mole_fraction,
    ),
    "gas_flux": ReportSpec(
        "Inlet flow and face gas flux with outlet flows.",
        (
            _field("F_in", "inlet_flow"),
            _field("N_gas_face", "gas_flux", "gas_species", "x_face"),
        ),
        derived=(
            _output("outlet_species_flow", "gas_species"),
            _output("outlet_flow"),
        ),
        derive=_derive_gas_flux,
    ),
    "reaction_rate": ReportSpec(
        "Reaction rate by reaction and cell.",
        (_field("R_rxn", "reaction_rate", "reaction", "x_cell"),),
        requires_reactions=True,
    ),
    "gas_enthalpy_flux": ReportSpec(
        "Gas enthalpy flux by species and face.",
        (_field("J_gas_face", "gas_enthalpy_flux", "gas_species", "x_face"),),
    ),
    "heat_balance": ReportSpec(
        "Integral heat totals and balance error.",
        tuple(_field(name, name) for name in (
            "heat_in_total", "heat_out_total", "heat_loss_total", "heat_bed_total"
        )),
        derived=(_output("heat_balance_error"),),
        derive=_derive_heat_balance,
    ),
    "mass_balance": ReportSpec(
        "Integral mass totals and balance error.",
        tuple(_field(name, name) for name in (
            "mass_in_total", "mass_out_total", "mass_bed_total"
        )),
        derived=(_output("mass_balance_error"),),
        derive=_derive_mass_balance,
    ),
})


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
    balance_errors: dict[str, BalanceError] = field(default_factory=dict)
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    plot_errors: dict[str, str] = field(default_factory=dict)
    reporter: Any | None = None
    solver_stats: dict[str, Any] = field(default_factory=dict)
    code_git: dict[str, Any] = field(default_factory=lambda: _git_state())

    @property
    def plot_status(self) -> str:
        requested = self.case.run.outputs.requested_plots
        if not requested:
            return "not_requested"
        succeeded = sum(plot_id in self.artifact_paths for plot_id in requested)
        if not self.plot_errors:
            return "success"
        return "partial_failure" if succeeded else "failed"


def _selected_specs(case: Case) -> tuple[ReportSpec, ...]:
    return tuple(REPORT_REGISTRY[report_id] for report_id in case.run.outputs.requested_reports)


def reporting_targets(report_ids) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        variable
        for report_id in report_ids
        for variable in REPORT_REGISTRY[report_id].model_variables
    ))


def _find_variable(process, name: str):
    matches = [
        variable
        for qualified_name, variable in process.dictVariables.items()
        if qualified_name == name or qualified_name.endswith(f".{name}")
    ]
    if len(matches) != 1:
        qualifier = "no" if not matches else "multiple"
        raise ValueError(f"Reporter contains {qualifier} variables named '{name}'.")
    return matches[0]


def _time_and_values(variable, label: str) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(variable.TimeValues, dtype=float).reshape(-1)
    values = np.asarray(variable.Values, dtype=float)
    if values.ndim == 0 or values.shape[0] != time.size:
        raise ValueError(f"{label} values do not align with their time coordinate.")
    if np.any(np.diff(time) < -TIME_ATOL):
        raise ValueError(f"{label} time coordinates must be non-decreasing.")
    keep = np.ones(time.size, dtype=bool)
    if time.size > 1:
        keep[:-1] = ~np.isclose(time[:-1], time[1:], rtol=0.0, atol=TIME_ATOL)
    return time[keep], values[keep]


def _coordinate(case: Case, variable, dimension: str, index: int, size: int):
    configured = {
        "gas_species": case.chemistry.gas_species,
        "solid_species": case.solids.solid_species,
        "reaction": case.chemistry.reaction_ids,
    }
    if dimension in configured:
        values = configured[dimension]
    else:
        domains = getattr(variable, "Domains", ())
        values = () if index >= len(domains) else getattr(domains[index], "Points", ())
    if len(values) != size:
        raise ValueError(
            f"Reporter dimension '{dimension}' has size {size}, "
            f"but {len(values)} labels were resolved."
        )
    return np.asarray(values)


def _scheduled_time(case: Case) -> np.ndarray:
    interval = case.run.simulation.reporting_interval_s
    horizon = case.run.simulation.time_horizon_s
    time = interval * np.arange(int(np.floor(horizon / interval)) + 1)
    if np.isclose(time[-1], horizon, rtol=0.0, atol=TIME_ATOL):
        time[-1] = horizon
    else:
        time = np.append(time, horizon)
    return time


def extract_dataset(process, case: Case):
    """Build exactly the requested labelled reports from a DAETools process."""

    import xarray as xr

    specs = _selected_specs(case)
    fields: dict[str, ModelField] = {}
    for field_spec in (field for spec in specs for field in (*spec.fields, *spec.support)):
        previous = fields.setdefault(field_spec.source, field_spec)
        if previous.dimensions != field_spec.dimensions:
            raise RuntimeError(f"Conflicting dimensions declared for '{field_spec.source}'.")

    raw: dict[str, Any] = {}
    reference_time = None
    coordinates: dict[str, Any] = {}
    for source, field_spec in fields.items():
        variable = _find_variable(process, source)
        time, values = _time_and_values(variable, source)
        if reference_time is None:
            reference_time = time
            coordinates["time"] = ("time", time, {"units": "s"})
        elif time.shape != reference_time.shape or not np.allclose(
            time, reference_time, rtol=0.0, atol=TIME_ATOL
        ):
            raise ValueError(f"Reporter variable '{source}' has a different time coordinate.")
        if values.ndim != len(field_spec.dimensions) + 1:
            raise ValueError(
                f"Reporter variable '{source}' has shape {values.shape}; "
                f"expected time plus {field_spec.dimensions}."
            )
        variable_coords = {"time": coordinates["time"]}
        for index, (dimension, size) in enumerate(zip(field_spec.dimensions, values.shape[1:])):
            resolved = _coordinate(case, variable, dimension, index, size)
            if dimension in coordinates and not np.array_equal(coordinates[dimension][1], resolved):
                raise ValueError(f"Reporter variables disagree on coordinate '{dimension}'.")
            attributes = {"units": "m"} if dimension in {"x_cell", "x_face"} else {}
            coordinates[dimension] = (dimension, resolved, attributes)
            variable_coords[dimension] = coordinates[dimension]
        raw[source] = xr.DataArray(
            values,
            dims=("time", *field_spec.dimensions),
            coords=variable_coords,
            attrs={"units": str(getattr(variable, "Units", "")), "source_variable": source},
        )

    if reference_time is None:
        reference_time = _scheduled_time(case)
        coordinates["time"] = ("time", reference_time, {"units": "s"})
    dataset = xr.Dataset(
        coords={"time": coordinates["time"]},
        attrs={
            "system_name": case.run.simulation.system_name,
            "selected_reports": ",".join(case.run.outputs.requested_reports),
        },
    )
    for spec in specs:
        for field_spec in spec.fields:
            dataset[field_spec.output] = raw[field_spec.source]
        if spec.derive is not None:
            spec.derive(dataset, raw, case)
    return dataset


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


def create_dataset_reporter(case: Case):
    """Create the lazy DAETools-to-NetCDF adapter."""

    from daetools.pyDAE import daeDataReporterLocal

    class PackedBedDatasetReporter(daeDataReporterLocal):
        def __init__(self):
            daeDataReporterLocal.__init__(self)
            self.ProcessName = ""
            self.ConnectString = ""
            self.output_directory = Path(case.output_directory)
            self.results_path = None
            self.write_error = None
            self._connected = False
            self._compiled_process = None

        @property
        def Process(self):
            return self._compiled_process if self._compiled_process is not None else super().Process

        def accept_process(self, process):
            """Accept array results while preserving the usual reporter/plotter interface."""
            self._compiled_process = process

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
            self._connected = False
            try:
                self.finish()
                return True
            except Exception:
                return False

        def IsConnected(self):
            return self._connected

        def finish(self):
            if self.write_error is None and self.results_path is None:
                try:
                    self.results_path = write_dataset(
                        extract_dataset(self.Process, case), self.output_directory / RESULTS_FILENAME,
                    )
                except Exception as exc:
                    self.write_error = exc
            if self.write_error is not None:
                raise RuntimeError("Data reporter failed while writing simulation reports.") from self.write_error
            return self.results_path

    return PackedBedDatasetReporter()


def compute_balance_errors(dataset_or_path) -> dict[str, BalanceError]:
    import xarray as xr

    errors = {}
    source = (
        xr.open_dataset(dataset_or_path, engine="scipy")
        if isinstance(dataset_or_path, (str, Path)) else nullcontext(dataset_or_path)
    )
    with source as dataset:
        for key, unit in (("heat", "J"), ("mass", "kg")):
            variable_name = f"{key}_balance_error"
            if variable_name not in dataset:
                continue
            values = np.asarray(dataset[variable_name].values, dtype=float)
            if values.size == 0 or np.all(np.isnan(values)):
                continue
            index = int(np.nanargmax(np.abs(values)))
            errors[key] = BalanceError(
                max_abs_error=float(abs(values[index])), time_s=float(dataset.time.values[index]), unit=unit,
            )
    return errors


def format_balance_error_lines(balance_errors) -> tuple[str, ...]:
    return tuple(
        f"largest {key} balance error: {error.max_abs_error:.6g} "
        f"{error.unit} at t={error.time_s:.6g} s"
        for key in ("heat", "mass")
        if (error := balance_errors.get(key)) is not None
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    source = str(Path(__file__).resolve().parent)
    try:
        commit = subprocess.run(
            ["git", "-C", source, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", source, "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("numpy", "xarray", "scipy", "matplotlib", "pydantic", "daetools", "scikit-sundae"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _dataset_inventory(path: Path | None) -> dict[str, Any]:
    import xarray as xr

    if path is None or not path.is_file():
        return {}
    with xr.open_dataset(path, engine="scipy") as dataset:
        return {
            "dimensions": dict(dataset.sizes),
            "variables": {
                name: {"dimensions": list(variable.dims), "units": variable.attrs.get("units", "")}
                for name, variable in dataset.data_vars.items()
            },
        }


def _compiled_programs(case: Case) -> dict[str, Any]:
    return {
        name: asdict(program)
        for name, program in (
            ("inlet_flow", case.inlet_flow_program),
            ("inlet_composition", case.inlet_composition_program),
            ("inlet_temperature", case.inlet_temperature_program),
            ("outlet_pressure", case.outlet_pressure_program),
        )
    }


def write_run_manifest(
    result: RunResult,
    *,
    failure_stage: str | None = None,
    traceback_text: str | None = None,
) -> Path:
    """Write resolved provenance, output inventory, and post-processing status."""

    case = result.case
    inputs = {
        "run": case.run_path,
        "chemistry": case.chemistry_path,
        "program": case.program_path,
        "solids": case.solids_path,
    }
    outputs = {"results": result.results_path, **result.artifact_paths}
    output_records = {
        name: {"status": "success", "path": str(path), "sha256": _sha256(path)}
        for name, path in outputs.items() if path is not None and path.is_file()
    }
    output_records.update(
        {name: {"status": "failed", "error": error} for name, error in result.plot_errors.items()}
    )
    manifest = {
        "status": result.status,
        "runtime_s": result.runtime_s,
        "solver_stats": result.solver_stats,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "git": result.code_git,
        },
        "configuration": {
            "run": case.run.model_dump(mode="json"),
            "chemistry": case.chemistry.model_dump(mode="json"),
            "program": case.program.model_dump(mode="json"),
            "solids": case.solids.model_dump(mode="json"),
            "compiled_programs": _compiled_programs(case),
        },
        "inputs": {
            name: {"path": str(inputs[name]), "sha256": digest}
            for name, digest in case.input_hashes.items()
        },
        "outputs": output_records,
        "dataset": _dataset_inventory(result.results_path),
        "balances": {name: asdict(error) for name, error in result.balance_errors.items()},
        "plots": {
            "requested": list(case.run.outputs.requested_plots),
            "status": result.plot_status,
            "errors": result.plot_errors,
        },
    }
    if failure_stage is not None or traceback_text is not None:
        manifest["failure"] = {"stage": failure_stage, "traceback": traceback_text}
    path = result.output_directory / MANIFEST_FILENAME
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


__all__ = (
    "BalanceError", "ReportSpec", "RunResult",
    "MANIFEST_FILENAME", "REPORT_REGISTRY", "RESULTS_FILENAME",
    "compute_balance_errors", "format_balance_error_lines",
    "create_dataset_reporter", "extract_dataset", "load_dataset",
    "reporting_targets", "write_dataset", "write_run_manifest",
)
