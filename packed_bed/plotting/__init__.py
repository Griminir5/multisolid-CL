"""Registry and coordinator for NetCDF-only post-run plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .axial_profiles import PLOT as AXIAL_PROFILES
from .definitions import PlotSpec
from .outlet_composition import PLOT as OUTLET_COMPOSITION
from .outlet_conditions import PLOT as OUTLET_CONDITIONS


PLOT_REGISTRY: Mapping[str, PlotSpec] = MappingProxyType({
    plot.id: plot
    for plot in (OUTLET_COMPOSITION, OUTLET_CONDITIONS, AXIAL_PROFILES)
})


@dataclass(frozen=True)
class PlotResult:
    paths: dict[str, Path]
    errors: dict[str, str]


def _render_requested_plots(
    results_path: str | Path,
    plot_ids,
    output_directory: str | Path,
) -> PlotResult:
    """Load one results file and independently render each selected plot."""

    import xarray as xr

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(results_path, engine="scipy") as source:
        dataset = source.load()
    paths: dict[str, Path] = {}
    errors: dict[str, str] = {}
    for plot_id in plot_ids:
        spec = PLOT_REGISTRY[plot_id]
        path = output / spec.filename
        try:
            spec.render(dataset, path)
            if not path.is_file():
                raise RuntimeError(f"Plot renderer did not create '{path.name}'.")
        except Exception as exc:
            errors[plot_id] = str(exc)
        else:
            paths[plot_id] = path
    return PlotResult(paths, errors)


__all__ = ("PLOT_REGISTRY", "PlotSpec")
