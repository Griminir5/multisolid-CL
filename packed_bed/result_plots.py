from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import RunResult
from .reporting import REPORT_VARIABLE_REGISTRY


@dataclass(frozen=True)
class RunResultPlotData:
    time_s: np.ndarray
    axial_positions_m: np.ndarray
    gas_species: tuple[str, ...]
    outlet_composition: np.ndarray
    outlet_temperature_k: np.ndarray
    outlet_pressure_pa: np.ndarray
    outlet_flowrate_mol_s: np.ndarray
    temperature_profile_k: np.ndarray


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
        if required:
            raise ValueError(f"Reporter does not contain a variable named '{variable_name}'.")
        return None
    if len(matches) > 1:
        joined = ", ".join(matches)
        raise ValueError(f"Reporter contains multiple matches for '{variable_name}': {joined}.")
    return process.dictVariables[matches[0]]


def _find_report_variable(run_result: RunResult, report_id: str, *, required: bool = True):
    definition = REPORT_VARIABLE_REGISTRY.get(report_id)
    if definition is None:
        raise ValueError(f"Unknown report id '{report_id}'.")
    if definition.variable_name is None:
        if required:
            raise ValueError(f"Report '{report_id}' does not map to a concrete model variable.")
        return None
    return _find_variable(_require_process(run_result), definition.variable_name, required=required)


def _extract_time_and_values(variable: Any, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(variable.TimeValues, dtype=float).reshape(-1)
    values = np.asarray(variable.Values, dtype=float)
    if values.shape[0] != times.shape[0]:
        raise ValueError(
            f"{label} values have leading dimension {values.shape[0]} but time axis has length {times.shape[0]}."
        )
    return times, values


def _extract_scalar_series(variable: Any, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    times, values = _extract_time_and_values(variable, label=label)
    if values.ndim == 1:
        return times, values
    if values.ndim == 2 and values.shape[1] == 1:
        return times, values[:, 0]
    raise ValueError(f"{label} is not a scalar time series; got array with shape {values.shape}.")


def _extract_matrix_series(variable: Any, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    times, values = _extract_time_and_values(variable, label=label)
    if values.ndim != 2:
        raise ValueError(f"{label} is not a 2D time series; got array with shape {values.shape}.")
    return times, values


def _extract_tensor_series(variable: Any, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    times, values = _extract_time_and_values(variable, label=label)
    if values.ndim != 3:
        raise ValueError(f"{label} is not a 3D time series; got array with shape {values.shape}.")
    return times, values


def _extract_static_profile(variable: Any, *, label: str) -> np.ndarray:
    _, values = _extract_time_and_values(variable, label=label)
    if values.ndim == 1:
        return values
    if values.ndim != 2:
        raise ValueError(f"{label} is not a static profile; got array with shape {values.shape}.")
    if values.shape[0] > 1 and not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
        raise ValueError(f"{label} changes over time; expected a static coordinate profile.")
    return values[0]


def _require_same_time_axis(reference: np.ndarray, candidate: np.ndarray, *, label: str) -> None:
    if reference.shape != candidate.shape or not np.allclose(reference, candidate, rtol=0.0, atol=1e-12):
        raise ValueError(f"{label} does not share the same time axis as the reference series.")


def _collapse_duplicate_times(time_s: np.ndarray, *series: np.ndarray) -> tuple[np.ndarray, ...]:
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    if time_s.size <= 1:
        return (time_s, *series)

    tolerance = 1e-12
    keep_mask = np.ones(time_s.shape, dtype=bool)
    for index in range(time_s.size - 1):
        current_time = time_s[index]
        next_time = time_s[index + 1]
        if next_time < current_time - tolerance:
            raise ValueError("Reporter time values must be non-decreasing.")
        if np.isclose(next_time, current_time, rtol=0.0, atol=tolerance):
            keep_mask[index] = False

    collapsed_time_s = time_s[keep_mask]
    collapsed_series = tuple(np.asarray(values)[keep_mask, ...] for values in series)
    return (collapsed_time_s, *collapsed_series)


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float).reshape(-1)
    if centers.size == 0:
        raise ValueError("Cannot compute cell edges from an empty center array.")
    if centers.size == 1:
        half_width = 0.5
        return np.array([centers[0] - half_width, centers[0] + half_width], dtype=float)

    deltas = np.diff(centers)
    if np.any(deltas <= 0.0):
        raise ValueError("Coordinate centers must be strictly increasing.")

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * deltas[0]
    edges[-1] = centers[-1] + 0.5 * deltas[-1]
    return edges


def _active_species_indices(outlet_composition: np.ndarray) -> list[int]:
    indices = [
        species_index
        for species_index in range(outlet_composition.shape[1])
        if np.nanmax(np.abs(outlet_composition[:, species_index])) > 1e-10
    ]
    return indices or list(range(outlet_composition.shape[1]))


def extract_run_result_plot_data(run_result: RunResult) -> RunResultPlotData:
    process = _require_process(run_result)

    temperature_variable = _find_report_variable(run_result, "temperature")
    pressure_variable = _find_report_variable(run_result, "pressure")
    composition_variable = _find_report_variable(run_result, "gas_mole_fraction")
    gas_flux_variable = _find_report_variable(run_result, "gas_flux")

    time_s, temperature_profile_k = _extract_matrix_series(
        temperature_variable,
        label="temperature report",
    )
    pressure_time_s, pressure_profile_pa = _extract_matrix_series(
        pressure_variable,
        label="pressure report",
    )
    composition_time_s, gas_mole_fraction = _extract_tensor_series(
        composition_variable,
        label="gas mole fraction report",
    )
    flow_time_s, gas_flux = _extract_tensor_series(
        gas_flux_variable,
        label="gas flux report",
    )

    _require_same_time_axis(time_s, pressure_time_s, label="pressure report")
    _require_same_time_axis(time_s, composition_time_s, label="gas mole fraction report")
    _require_same_time_axis(time_s, flow_time_s, label="gas flux report")

    outlet_pressure_pa: np.ndarray | None = None
    p_out_variable = _find_variable(process, "P_out", required=False)
    if p_out_variable is not None:
        outlet_pressure_time_s, outlet_pressure_pa = _extract_scalar_series(
            p_out_variable,
            label="outlet pressure variable",
        )
        _require_same_time_axis(time_s, outlet_pressure_time_s, label="outlet pressure variable")
        time_s, temperature_profile_k, pressure_profile_pa, gas_mole_fraction, gas_flux, outlet_pressure_pa = (
            _collapse_duplicate_times(
                time_s,
                temperature_profile_k,
                pressure_profile_pa,
                gas_mole_fraction,
                gas_flux,
                outlet_pressure_pa,
            )
        )
    else:
        time_s, temperature_profile_k, pressure_profile_pa, gas_mole_fraction, gas_flux = (
            _collapse_duplicate_times(
                time_s,
                temperature_profile_k,
                pressure_profile_pa,
                gas_mole_fraction,
                gas_flux,
            )
        )

    axial_positions_m = _extract_static_profile(
        _find_variable(process, "xval_cells"),
        label="cell-center coordinates",
    )

    gas_species = tuple(run_result.run_bundle.chemistry.gas_species)
    if temperature_profile_k.shape[1] != axial_positions_m.size:
        raise ValueError("Temperature profile does not align with the axial cell-center coordinates.")
    if pressure_profile_pa.shape[1] != axial_positions_m.size:
        raise ValueError("Pressure profile does not align with the axial cell-center coordinates.")
    if gas_mole_fraction.shape[1] != len(gas_species):
        raise ValueError("Gas mole fraction report does not align with the configured gas species.")
    if gas_mole_fraction.shape[2] != axial_positions_m.size:
        raise ValueError("Gas mole fraction report does not align with the axial cell-center coordinates.")

    outlet_species_flux = gas_flux[:, :, -1]
    if outlet_species_flux.shape[1] != len(gas_species):
        raise ValueError("Gas flux report does not align with the configured gas species.")

    cross_section_area_m2 = np.pi * run_result.run_bundle.run.model.bed_radius_m ** 2
    outlet_species_flow_mol_s = cross_section_area_m2 * outlet_species_flux
    outlet_flowrate_mol_s = outlet_species_flow_mol_s.sum(axis=1)

    outlet_composition = gas_mole_fraction[:, :, -1].copy()
    total_outlet_flow = outlet_species_flow_mol_s.sum(axis=1, keepdims=True)
    has_outlet_flow = np.abs(total_outlet_flow[:, 0]) > 1e-12
    outlet_composition[has_outlet_flow] = np.divide(
        outlet_species_flow_mol_s[has_outlet_flow],
        total_outlet_flow[has_outlet_flow],
    )

    if outlet_pressure_pa is None:
        outlet_pressure_pa = pressure_profile_pa[:, -1]

    return RunResultPlotData(
        time_s=time_s,
        axial_positions_m=axial_positions_m,
        gas_species=gas_species,
        outlet_composition=outlet_composition,
        outlet_temperature_k=temperature_profile_k[:, -1],
        outlet_pressure_pa=outlet_pressure_pa,
        outlet_flowrate_mol_s=outlet_flowrate_mol_s,
        temperature_profile_k=temperature_profile_k,
    )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def render_outlet_composition_plot(
    plot_data: RunResultPlotData,
    output_dir: str | Path,
    *,
    image_format: str = "svg",
) -> Path:
    output_path = Path(output_dir) / f"outlet_composition_vs_time.{image_format}"
    figure, axis = plt.subplots(figsize=(12, 6))

    active_indices = _active_species_indices(plot_data.outlet_composition)
    cmap = plt.get_cmap("tab10", max(len(active_indices), 1))
    for color_index, species_index in enumerate(active_indices):
        axis.plot(
            plot_data.time_s,
            plot_data.outlet_composition[:, species_index],
            linewidth=2,
            color=cmap(color_index),
            label=plot_data.gas_species[species_index],
        )

    axis.set_title("Outlet Composition vs Time")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Mole fraction [-]")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.3)
    axis.margins(x=0.0)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=8,
    )
    figure.tight_layout()
    _save_figure(figure, output_path)
    return output_path


def render_outlet_conditions_plot(
    plot_data: RunResultPlotData,
    output_dir: str | Path,
    *,
    image_format: str = "svg",
) -> Path:
    output_path = Path(output_dir) / f"outlet_conditions_vs_time.{image_format}"
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(plot_data.time_s, plot_data.outlet_temperature_k, linewidth=2, color="#d94841")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title("Outlet Temperature")

    axes[1].plot(plot_data.time_s, plot_data.outlet_pressure_pa, linewidth=2, color="#1d3557")
    axes[1].set_ylabel("Pressure [Pa]")
    axes[1].set_title("Outlet Pressure")

    axes[2].plot(plot_data.time_s, plot_data.outlet_flowrate_mol_s, linewidth=2, color="#2a9d8f")
    axes[2].set_ylabel("Flowrate [mol/s]")
    axes[2].set_title("Overall Outlet Flowrate")
    axes[2].set_xlabel("Time [s]")

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.margins(x=0.0)

    figure.tight_layout()
    _save_figure(figure, output_path)
    return output_path


def render_temperature_profile_plot(
    plot_data: RunResultPlotData,
    output_dir: str | Path,
    *,
    image_format: str = "svg",
) -> Path:
    output_path = Path(output_dir) / f"temperature_profile_vs_time.{image_format}"
    figure, axis = plt.subplots(figsize=(12, 6))

    mesh = axis.pcolormesh(
        _centers_to_edges(plot_data.time_s),
        _centers_to_edges(plot_data.axial_positions_m),
        plot_data.temperature_profile_k.T,
        shading="auto",
        cmap="inferno",
    )
    colorbar = figure.colorbar(mesh, ax=axis)
    colorbar.set_label("Temperature [K]")

    axis.set_title("Temperature Profile vs Time")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Axial position [m]")
    figure.tight_layout()
    _save_figure(figure, output_path)
    return output_path


def render_run_result_plots(
    run_result: RunResult,
    output_dir: str | Path | None = None,
    *,
    image_format: str = "svg",
) -> dict[str, Path]:
    plot_data = extract_run_result_plot_data(run_result)
    resolved_output_dir = Path(output_dir) if output_dir is not None else run_result.run_bundle.artifacts_directory
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    return {
        f"outlet_composition_{image_format}": render_outlet_composition_plot(
            plot_data,
            resolved_output_dir,
            image_format=image_format,
        ),
        f"outlet_conditions_{image_format}": render_outlet_conditions_plot(
            plot_data,
            resolved_output_dir,
            image_format=image_format,
        ),
        f"temperature_profile_{image_format}": render_temperature_profile_plot(
            plot_data,
            resolved_output_dir,
            image_format=image_format,
        ),
    }


__all__ = [
    "RunResultPlotData",
    "extract_run_result_plot_data",
    "render_outlet_composition_plot",
    "render_outlet_conditions_plot",
    "render_run_result_plots",
    "render_temperature_profile_plot",
]
