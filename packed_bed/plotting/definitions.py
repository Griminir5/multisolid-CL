"""Import-safe post-run plot definitions and small rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from xml.dom import minidom


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
        if path.suffix.lower() == ".svg":
            _normalize_svg_paths(path)
    finally:
        plt.close(figure)


def _normalize_svg_paths(path: Path) -> None:
    """Keep empty glyphs invisible while making their paths readable by Qt SVG."""
    # Matplotlib emits paths without `d` for spaces. Qt rejects those paths,
    # then warns for every reference to the glyph. A move draws nothing but
    # keeps the glyph defined, preserving embedded fonts and text placement.
    with minidom.parse(str(path)) as document:
        changed = False
        for node in document.getElementsByTagName("path"):
            if not node.getAttribute("d").strip():
                node.setAttribute("d", "M 0 0")
                changed = True
        if changed:
            path.write_bytes(document.toxml(encoding="utf-8"))


__all__ = ("PlotSpec", "pyplot", "save_figure")
