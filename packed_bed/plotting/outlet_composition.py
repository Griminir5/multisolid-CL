from __future__ import annotations

from pathlib import Path

from .definitions import PlotSpec, pyplot, save_figure


def render(dataset, path: Path) -> None:
    import numpy as np

    plt = pyplot()
    inlet = dataset.inlet_composition.values
    outlet = dataset.outlet_composition.values
    active = [
        index for index in range(dataset.sizes["gas_species"])
        if max(np.nanmax(abs(inlet[:, index])), np.nanmax(abs(outlet[:, index]))) > 1.0e-10
    ] or list(range(dataset.sizes["gas_species"]))
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.get_cmap("tab10", max(len(active), 1))
    for axis, title, values in (
        (axes[0], "Inlet Composition", inlet),
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
    save_figure(figure, path)


PLOT = PlotSpec(
    id="outlet_composition",
    description="Inlet and outlet gas composition over time.",
    required_reports=("gas_mole_fraction",),
    filename="outlet_composition_vs_time.svg",
    render=render,
)
