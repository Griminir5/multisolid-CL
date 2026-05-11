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


_TIME_ATOL = 1e-12
_ACTIVE_SPECIES_ATOL = 1e-10
_FLOW_ATOL = 1e-12


@dataclass(frozen=True)
class RunResultPlotData:
    time_s: np.ndarray
    axial_positions_m: np.ndarray
    gas_species: tuple[str, ...]
    inlet_composition: np.ndarray
    outlet_composition: np.ndarray
    outlet_temperature_k: np.ndarray
    outlet_pressure_pa: np.ndarray
    pressure_drop_pa: np.ndarray
    outlet_flowrate_mol_s: np.ndarray
    temperature_profile_k: np.ndarray
    pressure_profile_pa: np.ndarray


@dataclass(frozen=True)
class _RawPlotSeries:
    time_s: np.ndarray
    temperature_profile_k: np.ndarray
    pressure_profile_pa: np.ndarray
    gas_mole_fraction: np.ndarray
    gas_flux: np.ndarray


@dataclass(frozen=True)
class _BoundaryPressureSeries:
    inlet_pa: np.ndarray | None
    outlet_pa: np.ndarray | None


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


def _find_report_variable(process: Any, report_id: str, *, required: bool = True):
    definition = REPORT_VARIABLE_REGISTRY.get(report_id)
    if definition is None:
        raise ValueError(f"Unknown report id '{report_id}'.")
    if definition.variable_name is None:
        if required:
            raise ValueError(f"Report '{report_id}' does not map to a concrete model variable.")
        return None
    return _find_variable(process, definition.variable_name, required=required)


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
    if values.shape[0] > 1 and not np.allclose(values, values[0], rtol=0.0, atol=_TIME_ATOL):
        raise ValueError(f"{label} changes over time; expected a static coordinate profile.")
    return values[0]


def _require_same_time_axis(reference: np.ndarray, candidate: np.ndarray, *, label: str) -> None:
    if reference.shape != candidate.shape or not np.allclose(reference, candidate, rtol=0.0, atol=_TIME_ATOL):
        raise ValueError(f"{label} does not share the same time axis as the reference series.")


def _collapse_duplicate_times(time_s: np.ndarray, *series: np.ndarray) -> tuple[np.ndarray, ...]:
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    if time_s.size <= 1:
        return (time_s, *series)

    keep_mask = np.ones(time_s.shape, dtype=bool)
    for index in range(time_s.size - 1):
        current_time = time_s[index]
        next_time = time_s[index + 1]
        if next_time < current_time - _TIME_ATOL:
            raise ValueError("Reporter time values must be non-decreasing.")
        if np.isclose(next_time, current_time, rtol=0.0, atol=_TIME_ATOL):
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


def _active_species_indices(*composition_series: np.ndarray) -> list[int]:
    if not composition_series:
        return []

    species_count = composition_series[0].shape[1]
    indices = [
        species_index
        for species_index in range(species_count)
        if any(np.nanmax(np.abs(series[:, species_index])) > _ACTIVE_SPECIES_ATOL for series in composition_series)
    ]
    return indices or list(range(species_count))


def _sample_inlet_composition(run_result: RunResult, time_s: np.ndarray, gas_species: tuple[str, ...]) -> np.ndarray:
    program = run_result.run_bundle.program.inlet_composition.compile_program(
        gas_species,
        repeat=run_result.run_bundle.run.repeat_program,
        time_horizon=run_result.run_bundle.run.time_horizon_s,
    )
    segments = program.build_segments()
    initial_value = np.asarray(program.initial_value, dtype=float)
    sampled = np.empty((time_s.size, initial_value.size), dtype=float)

    if not segments:
        sampled[:, :] = initial_value
        return sampled

    segment_index = 0
    for time_index, time_value in enumerate(time_s):
        while segment_index < len(segments) and time_value > segments[segment_index].end_time + _TIME_ATOL:
            segment_index += 1

        if segment_index >= len(segments):
            sampled[time_index, :] = np.asarray(segments[-1].end_value, dtype=float)
            continue

        segment = segments[segment_index]
        start_value = np.asarray(segment.start_value, dtype=float)
        end_value = np.asarray(segment.end_value, dtype=float)
        duration = segment.end_time - segment.start_time
        if duration <= _TIME_ATOL:
            sampled[time_index, :] = end_value
            continue

        fraction = (time_value - segment.start_time) / duration
        fraction = min(max(fraction, 0.0), 1.0)
        sampled[time_index, :] = start_value + (end_value - start_value) * fraction

    return sampled


def _extract_required_plot_series(process: Any) -> _RawPlotSeries:
    temperature_variable = _find_report_variable(process, "temperature")
    pressure_variable = _find_report_variable(process, "pressure")
    composition_variable = _find_report_variable(process, "gas_mole_fraction")
    gas_flux_variable = _find_report_variable(process, "gas_flux")

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

    return _RawPlotSeries(
        time_s=time_s,
        temperature_profile_k=temperature_profile_k,
        pressure_profile_pa=pressure_profile_pa,
        gas_mole_fraction=gas_mole_fraction,
        gas_flux=gas_flux,
    )


def _extract_optional_scalar_series(
    process: Any,
    variable_name: str,
    *,
    label: str,
    reference_time_s: np.ndarray,
) -> np.ndarray | None:
    variable = _find_variable(process, variable_name, required=False)
    if variable is None:
        return None

    time_s, values = _extract_scalar_series(variable, label=label)
    _require_same_time_axis(reference_time_s, time_s, label=label)
    return values


def _extract_boundary_pressures(process: Any, reference_time_s: np.ndarray) -> _BoundaryPressureSeries:
    return _BoundaryPressureSeries(
        inlet_pa=_extract_optional_scalar_series(
            process,
            "P_in",
            label="inlet pressure variable",
            reference_time_s=reference_time_s,
        ),
        outlet_pa=_extract_optional_scalar_series(
            process,
            "P_out",
            label="outlet pressure variable",
            reference_time_s=reference_time_s,
        ),
    )


def _collapse_named_series(
    time_s: np.ndarray,
    **series_by_name: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, np.ndarray | None]]:
    names: list[str] = []
    series: list[np.ndarray] = []
    for name, values in series_by_name.items():
        if values is None:
            continue
        names.append(name)
        series.append(values)

    collapsed = _collapse_duplicate_times(time_s, *series)
    collapsed_by_name: dict[str, np.ndarray | None] = {name: None for name in series_by_name}
    collapsed_by_name.update(dict(zip(names, collapsed[1:])))
    return collapsed[0], collapsed_by_name


def _require_collapsed_series(series_by_name: dict[str, np.ndarray | None], name: str) -> np.ndarray:
    values = series_by_name.get(name)
    if values is None:
        raise RuntimeError(f"Internal plotting error: required series '{name}' was not retained.")
    return values


def _derive_pressure_outputs(
    pressure_profile_pa: np.ndarray,
    *,
    inlet_pressure_pa: np.ndarray | None,
    outlet_pressure_pa: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_outlet_pressure_pa = outlet_pressure_pa
    if resolved_outlet_pressure_pa is None:
        resolved_outlet_pressure_pa = pressure_profile_pa[:, -1]

    inlet_reference_pa = inlet_pressure_pa
    if inlet_reference_pa is None:
        inlet_reference_pa = pressure_profile_pa[:, 0]

    return resolved_outlet_pressure_pa, inlet_reference_pa - resolved_outlet_pressure_pa


def _validate_plot_series_shapes(
    *,
    axial_positions_m: np.ndarray,
    gas_species: tuple[str, ...],
    temperature_profile_k: np.ndarray,
    pressure_profile_pa: np.ndarray,
    gas_mole_fraction: np.ndarray,
    gas_flux: np.ndarray,
) -> None:
    if temperature_profile_k.shape[1] != axial_positions_m.size:
        raise ValueError("Temperature profile does not align with the axial cell-center coordinates.")
    if pressure_profile_pa.shape[1] != axial_positions_m.size:
        raise ValueError("Pressure profile does not align with the axial cell-center coordinates.")
    if gas_mole_fraction.shape[1] != len(gas_species):
        raise ValueError("Gas mole fraction report does not align with the configured gas species.")
    if gas_mole_fraction.shape[2] != axial_positions_m.size:
        raise ValueError("Gas mole fraction report does not align with the axial cell-center coordinates.")
    if gas_flux.shape[1] != len(gas_species):
        raise ValueError("Gas flux report does not align with the configured gas species.")
    if gas_flux.shape[2] == 0:
        raise ValueError("Gas flux report does not contain any axial faces.")


def _compute_outlet_flows_and_composition(
    run_result: RunResult,
    *,
    gas_mole_fraction: np.ndarray,
    gas_flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    outlet_species_flux = gas_flux[:, :, -1]
    cross_section_area_m2 = np.pi * run_result.run_bundle.run.model.bed_radius_m ** 2
    outlet_species_flow_mol_s = cross_section_area_m2 * outlet_species_flux
    outlet_flowrate_mol_s = outlet_species_flow_mol_s.sum(axis=1)

    outlet_composition = gas_mole_fraction[:, :, -1].copy()
    total_outlet_flow = outlet_species_flow_mol_s.sum(axis=1, keepdims=True)
    has_outlet_flow = np.abs(total_outlet_flow[:, 0]) > _FLOW_ATOL
    outlet_composition[has_outlet_flow] = np.divide(
        outlet_species_flow_mol_s[has_outlet_flow],
        total_outlet_flow[has_outlet_flow],
    )
    return outlet_flowrate_mol_s, outlet_composition


def extract_run_result_plot_data(run_result: RunResult) -> RunResultPlotData:
    process = _require_process(run_result)
    raw_series = _extract_required_plot_series(process)
    boundary_pressures = _extract_boundary_pressures(process, raw_series.time_s)

    time_s, collapsed = _collapse_named_series(
        raw_series.time_s,
        temperature_profile_k=raw_series.temperature_profile_k,
        pressure_profile_pa=raw_series.pressure_profile_pa,
        gas_mole_fraction=raw_series.gas_mole_fraction,
        gas_flux=raw_series.gas_flux,
        inlet_pressure_pa=boundary_pressures.inlet_pa,
        outlet_pressure_pa=boundary_pressures.outlet_pa,
    )
    temperature_profile_k = _require_collapsed_series(collapsed, "temperature_profile_k")
    pressure_profile_pa = _require_collapsed_series(collapsed, "pressure_profile_pa")
    gas_mole_fraction = _require_collapsed_series(collapsed, "gas_mole_fraction")
    gas_flux = _require_collapsed_series(collapsed, "gas_flux")
    inlet_pressure_pa = collapsed["inlet_pressure_pa"]
    outlet_pressure_pa = collapsed["outlet_pressure_pa"]

    outlet_pressure_pa, pressure_drop_pa = _derive_pressure_outputs(
        pressure_profile_pa,
        inlet_pressure_pa=inlet_pressure_pa,
        outlet_pressure_pa=outlet_pressure_pa,
    )

    axial_positions_m = _extract_static_profile(
        _find_variable(process, "xval_cells"),
        label="cell-center coordinates",
    )

    gas_species = tuple(run_result.run_bundle.chemistry.gas_species)
    _validate_plot_series_shapes(
        axial_positions_m=axial_positions_m,
        gas_species=gas_species,
        temperature_profile_k=temperature_profile_k,
        pressure_profile_pa=pressure_profile_pa,
        gas_mole_fraction=gas_mole_fraction,
        gas_flux=gas_flux,
    )
    outlet_flowrate_mol_s, outlet_composition = _compute_outlet_flows_and_composition(
        run_result,
        gas_mole_fraction=gas_mole_fraction,
        gas_flux=gas_flux,
    )

    inlet_composition = _sample_inlet_composition(run_result, time_s, gas_species)

    return RunResultPlotData(
        time_s=time_s,
        axial_positions_m=axial_positions_m,
        gas_species=gas_species,
        inlet_composition=inlet_composition,
        outlet_composition=outlet_composition,
        outlet_temperature_k=temperature_profile_k[:, -1],
        outlet_pressure_pa=outlet_pressure_pa,
        pressure_drop_pa=pressure_drop_pa,
        outlet_flowrate_mol_s=outlet_flowrate_mol_s,
        temperature_profile_k=temperature_profile_k,
        pressure_profile_pa=pressure_profile_pa,
    )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _finalize_time_axes(axes) -> None:
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.margins(x=0.0)


def _plot_time_series_axis(
    axis,
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    ylabel: str,
    title: str,
) -> None:
    axis.plot(time_s, values, linewidth=2, color=color)
    axis.set_ylabel(ylabel)
    axis.set_title(title)


def render_outlet_composition_plot(
    plot_data: RunResultPlotData,
    output_dir: str | Path,
    *,
    image_format: str = "svg",
) -> Path:
    output_path = Path(output_dir) / f"outlet_composition_vs_time.{image_format}"
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    active_indices = _active_species_indices(plot_data.inlet_composition, plot_data.outlet_composition)
    cmap = plt.get_cmap("tab10", max(len(active_indices), 1))

    for axis, title, composition in (
        (axes[0], "Inlet Composition Program", plot_data.inlet_composition),
        (axes[1], "Outlet Composition", plot_data.outlet_composition),
    ):
        for color_index, species_index in enumerate(active_indices):
            axis.plot(
                plot_data.time_s,
                composition[:, species_index],
                linewidth=2,
                color=cmap(color_index),
                label=plot_data.gas_species[species_index],
            )
        axis.set_title(title)
        axis.set_ylabel("Mole fraction [-]")
        axis.set_ylim(-0.02, 1.02)

    axes[1].set_xlabel("Time [s]")
    _finalize_time_axes(axes)
    axes[0].legend(
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

    _plot_time_series_axis(
        axes[0],
        plot_data.time_s,
        plot_data.outlet_temperature_k,
        color="#d94841",
        ylabel="Temperature [K]",
        title="Outlet Temperature",
    )
    _plot_time_series_axis(
        axes[1],
        plot_data.time_s,
        plot_data.pressure_drop_pa,
        color="#1d3557",
        ylabel="Delta P [Pa]",
        title="Pressure Drop (Inlet - Outlet)",
    )
    _plot_time_series_axis(
        axes[2],
        plot_data.time_s,
        plot_data.outlet_flowrate_mol_s,
        color="#2a9d8f",
        ylabel="Flowrate [mol/s]",
        title="Overall Outlet Flowrate",
    )
    axes[2].set_xlabel("Time [s]")
    _finalize_time_axes(axes)

    figure.tight_layout()
    _save_figure(figure, output_path)
    return output_path


def _plot_profile_heatmap(
    figure: plt.Figure,
    axis: plt.Axes,
    plot_data: RunResultPlotData,
    values: np.ndarray,
    *,
    cmap: str,
    colorbar_label: str,
    title: str,
) -> None:
    mesh = axis.pcolormesh(
        _centers_to_edges(plot_data.time_s),
        _centers_to_edges(plot_data.axial_positions_m),
        values.T,
        shading="auto",
        cmap=cmap,
        rasterized=True,
    )
    colorbar = figure.colorbar(mesh, ax=axis)
    colorbar.set_label(colorbar_label)
    axis.set_title(title)
    axis.set_ylabel("Axial position [m]")


def render_profile_plot(
    plot_data: RunResultPlotData,
    output_dir: str | Path,
    *,
    image_format: str = "svg",
) -> Path:
    output_path = Path(output_dir) / f"temperature_profile_vs_time.{image_format}"
    figure, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    _plot_profile_heatmap(
        figure,
        axes[0],
        plot_data,
        plot_data.temperature_profile_k,
        cmap="inferno",
        colorbar_label="Temperature [K]",
        title="Temperature Profile vs Time",
    )
    _plot_profile_heatmap(
        figure,
        axes[1],
        plot_data,
        plot_data.pressure_profile_pa,
        cmap="viridis",
        colorbar_label="Pressure [Pa]",
        title="Pressure Profile vs Time",
    )

    axes[1].set_xlabel("Time [s]")
    figure.tight_layout()
    _save_figure(figure, output_path)
    return output_path


def render_temperature_profile_plot(
    plot_data: RunResultPlotData,
    output_dir: str | Path,
    *,
    image_format: str = "svg",
) -> Path:
    return render_profile_plot(plot_data, output_dir, image_format=image_format)


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
        f"temperature_profile_{image_format}": render_profile_plot(
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
    "render_profile_plot",
    "render_run_result_plots",
    "render_temperature_profile_plot",
]
