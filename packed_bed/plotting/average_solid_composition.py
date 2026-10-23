"""Spatially averaged solid mole fractions on the model's uniform bed grid."""

from __future__ import annotations

from pathlib import Path

from .definitions import PlotSpec, pyplot, save_figure


def average_composition(dataset):
    import numpy as np

    if "solid_mole_fraction" not in dataset:
        raise ValueError("Average solid composition requires the solid_mole_fraction report.")
    fractions = dataset.solid_mole_fraction
    if set(fractions.dims) != {"time", "solid_species", "x_cell"}:
        raise ValueError("Solid mole fractions must have time, solid_species, and x_cell dimensions.")
    if not all(fractions.sizes.values()) or not np.isfinite(fractions.values).all():
        raise ValueError("Average solid composition requires nonempty, finite solid mole fractions.")
    # Every cell has the same volume in the model. Preserve the report's zero
    # fractions for empty cells, and include every selected solid species.
    return fractions.mean("x_cell", skipna=False).transpose("time", "solid_species")


def render(dataset, path: Path) -> None:
    fractions = average_composition(dataset)
    plt = pyplot()
    figure, axis = plt.subplots(figsize=(12, 6))
    colors = plt.get_cmap("tab10", fractions.sizes["solid_species"])
    for index, species in enumerate(fractions.solid_species.values):
        axis.plot(
            fractions.time, fractions.isel(solid_species=index),
            color=colors(index), label=str(species),
        )
    axis.set(
        title="Average Solid Composition of the Bed",
        xlabel="Time [s]", ylabel="Bed-average mole fraction [-]", ylim=(-0.02, 1.02),
    )
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    figure.tight_layout()
    save_figure(figure, path)


PLOT = PlotSpec(
    id="average_solid_composition",
    description="Solid mole fractions averaged over all bed cells, by species, over time.",
    required_reports=("solid_mole_fraction",),
    filename="average_solid_composition_vs_time.svg",
    render=render,
)
