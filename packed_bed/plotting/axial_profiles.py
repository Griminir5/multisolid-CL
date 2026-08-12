from __future__ import annotations

from pathlib import Path

from .definitions import PlotSpec, pyplot, save_figure


def _edges(values, *, clamp: bool = False):
    import numpy as np

    values = np.asarray(values, dtype=float)
    if values.size < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError("Profile plot coordinates must contain at least two increasing values.")
    edges = np.empty(values.size + 1)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] if clamp else values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] if clamp else values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def render(dataset, path: Path) -> None:
    plt = pyplot()
    figure, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for axis, variable, color_map, label, title in (
        (axes[0], "temperature", "inferno", "Temperature [K]", "Temperature Profile"),
        (axes[1], "pressure", "viridis", "Pressure [Pa]", "Pressure Profile"),
    ):
        mesh = axis.pcolormesh(
            _edges(dataset.time, clamp=True),
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
    save_figure(figure, path)


PLOT = PlotSpec(
    id="axial_profiles",
    description="Temperature and pressure profiles over time and bed position.",
    required_reports=("temperature", "pressure"),
    filename="axial_profiles_vs_time.svg",
    render=render,
)
