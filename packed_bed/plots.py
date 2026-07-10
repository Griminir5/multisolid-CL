"""Static plots that consume the labelled run dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .reports import RunResult, load_dataset


ACTIVE_SPECIES_ATOL = 1.0e-10


def _result_dataset(result: RunResult):
    if result.dataset is not None:
        return result.dataset
    if result.results_path is None:
        raise ValueError("RunResult does not contain a dataset or results path.")
    return load_dataset(result.results_path)


def _edges(centers) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    differences = np.diff(centers)
    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * differences[0]
    edges[-1] = centers[-1] + 0.5 * differences[-1]
    return edges


def _save(figure, path: Path) -> Path:
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _composition_plot(dataset, output_directory: Path, image_format: str) -> Path:
    required = ("inlet_composition", "outlet_composition")
    if any(name not in dataset for name in required):
        raise ValueError("Composition plotting requires gas mole fraction and flux reports.")
    inlet = dataset.inlet_composition.values
    outlet = dataset.outlet_composition.values
    active = [
        index for index in range(dataset.sizes["gas_species"])
        if max(np.nanmax(abs(inlet[:, index])), np.nanmax(abs(outlet[:, index])))
        > ACTIVE_SPECIES_ATOL
    ] or list(range(dataset.sizes["gas_species"]))

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.get_cmap("tab10", max(len(active), 1))
    for axis, title, values in (
        (axes[0], "Inlet Composition Program", inlet),
        (axes[1], "Outlet Composition", outlet),
    ):
        for color_index, species_index in enumerate(active):
            axis.plot(
                dataset.time,
                values[:, species_index],
                color=colors(color_index),
                label=str(dataset.gas_species.values[species_index]),
            )
        axis.set(title=title, ylabel="Mole fraction [-]", ylim=(-0.02, 1.02))
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    axes[1].set_xlabel("Time [s]")
    figure.tight_layout()
    return _save(figure, output_directory / f"outlet_composition_vs_time.{image_format}")


def _outlet_plot(dataset, output_directory: Path, image_format: str) -> Path:
    required = ("outlet_temperature", "pressure_drop", "outlet_flow")
    if any(name not in dataset for name in required):
        raise ValueError("Outlet plotting requires temperature, pressure, and gas flux reports.")
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for axis, variable, color, label, title in (
        (axes[0], "outlet_temperature", "#d94841", "Temperature [K]", "Outlet Temperature"),
        (axes[1], "pressure_drop", "#1d3557", "Delta P [Pa]", "Pressure Drop"),
        (axes[2], "outlet_flow", "#2a9d8f", "Flowrate [mol/s]", "Outlet Flowrate"),
    ):
        axis.plot(dataset.time, dataset[variable], color=color, linewidth=2)
        axis.set(ylabel=label, title=title)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    figure.tight_layout()
    return _save(figure, output_directory / f"outlet_conditions_vs_time.{image_format}")


def _profile_plot(dataset, output_directory: Path, image_format: str) -> Path:
    if "temperature" not in dataset or "pressure" not in dataset:
        raise ValueError("Profile plotting requires temperature and pressure reports.")
    figure, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for axis, variable, color_map, label, title in (
        (axes[0], "temperature", "inferno", "Temperature [K]", "Temperature Profile"),
        (axes[1], "pressure", "viridis", "Pressure [Pa]", "Pressure Profile"),
    ):
        mesh = axis.pcolormesh(
            _edges(dataset.time),
            _edges(dataset.x_cell),
            dataset[variable].transpose("x_cell", "time"),
            shading="auto",
            cmap=color_map,
            rasterized=True,
        )
        figure.colorbar(mesh, ax=axis, label=label)
        axis.set(title=title, ylabel="Axial position [m]")
    axes[-1].set_xlabel("Time [s]")
    figure.tight_layout()
    return _save(figure, output_directory / f"temperature_profile_vs_time.{image_format}")


def render_run_result_plots(
    result: RunResult,
    output_directory: str | Path | None = None,
    *,
    image_format: str = "svg",
) -> dict[str, Path]:
    dataset = _result_dataset(result)
    output = Path(output_directory or result.case.artifacts_directory)
    output.mkdir(parents=True, exist_ok=True)
    return {
        f"outlet_composition_{image_format}": _composition_plot(dataset, output, image_format),
        f"outlet_conditions_{image_format}": _outlet_plot(dataset, output, image_format),
        f"temperature_profile_{image_format}": _profile_plot(dataset, output, image_format),
    }


__all__ = ("render_run_result_plots",)
