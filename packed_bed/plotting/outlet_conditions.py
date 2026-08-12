from __future__ import annotations

from pathlib import Path

from .definitions import PlotSpec, pyplot, save_figure


def render(dataset, path: Path) -> None:
    plt = pyplot()
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for axis, variable, color, label, title in (
        (axes[0], "outlet_temperature", "#d94841", "Temperature [K]", "Outlet Temperature"),
        (axes[1], "pressure_drop", "#1d3557", "Delta P [Pa]", "Pressure Drop (Inlet - Outlet)"),
        (axes[2], "outlet_flow", "#2a9d8f", "Flowrate [mol/s]", "Overall Outlet Flowrate"),
    ):
        axis.plot(dataset.time, dataset[variable], color=color, linewidth=2)
        axis.set(ylabel=label, title=title)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    figure.tight_layout()
    save_figure(figure, path)


PLOT = PlotSpec(
    id="outlet_conditions",
    description="Outlet temperature, pressure drop, and overall flow over time.",
    required_reports=("temperature", "pressure", "gas_flux"),
    filename="outlet_conditions_vs_time.svg",
    render=render,
)
