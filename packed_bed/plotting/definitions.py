"""Import-safe post-run plot definitions and small rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class PlotSpec:
    id: str
    description: str
    required_reports: tuple[str, ...]
    filename: str
    render: Callable[[Any, Path], None]


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(figure, path: Path) -> None:
    plt = pyplot()
    try:
        figure.savefig(path, bbox_inches="tight")
    finally:
        plt.close(figure)


__all__ = ("PlotSpec", "pyplot", "save_figure")
