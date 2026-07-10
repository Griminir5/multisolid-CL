from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IncidenceEntry:
    row_index: int
    column_index: int
    kind: str


@dataclass(frozen=True)
class SolverIncidenceMatrix:
    row_labels: tuple[str, ...]
    row_types: tuple[str, ...]
    column_labels: tuple[str, ...]
    entries: tuple[IncidenceEntry, ...]

    @property
    def row_count(self) -> int:
        return len(self.row_labels)

    @property
    def column_count(self) -> int:
        return len(self.column_labels)

    @property
    def nonzero_count(self) -> int:
        return len(self.entries)

    @property
    def density(self) -> float:
        size = self.row_count * self.column_count
        return self.nonzero_count / size if size else 0.0


def _valid_indexes(values: Any) -> set[int]:
    return {index for value in values if (index := int(value)) >= 0}


def _short_label(label: str, system_name: str | None) -> str:
    if system_name and label.startswith(f"{system_name}."):
        return label[len(system_name) + 1 :]
    return label


def _equation_type_label(equation_type: Any) -> str:
    return str(equation_type).rsplit(".", 1)[-1]


def collect_solver_incidence_matrix(model: Any) -> SolverIncidenceMatrix:
    """Collect the initialized DAETools equation/variable incidence structure."""

    infos = [
        info
        for equation in getattr(model, "Equations", ())
        for info in getattr(equation, "EquationExecutionInfos", ())
    ]
    if not infos:
        raise ValueError(
            "The DAETools model has no equation execution info. "
            "Initialize the simulation first."
        )

    max_row_index = max(int(info.EquationIndex) for info in infos)
    row_labels = [f"equation_{index}" for index in range(max_row_index + 1)]
    row_types = ["unknown" for _index in range(max_row_index + 1)]
    entries: list[IncidenceEntry] = []

    for info in infos:
        row_index = int(info.EquationIndex)
        row_labels[row_index] = str(info.Name)
        row_types[row_index] = _equation_type_label(
            getattr(info, "EquationType", "unknown")
        )
        variable_indexes = _valid_indexes(getattr(info, "VariableIndexes", ()))
        derivative_indexes = _valid_indexes(
            getattr(info, "DiffVariableIndexes", ())
        )
        for column_index in sorted(variable_indexes | derivative_indexes):
            if column_index in variable_indexes and column_index in derivative_indexes:
                kind = "variable_and_time_derivative"
            elif column_index in derivative_indexes:
                kind = "time_derivative"
            else:
                kind = "variable"
            entries.append(IncidenceEntry(row_index, column_index, kind))

    index_name_map = getattr(model, "OverallIndex_BlockIndex_VariableNameMap", {})
    mapped_column_indexes = [int(index) for index in index_name_map]
    max_column_index = max(
        max(mapped_column_indexes, default=-1),
        max((entry.column_index for entry in entries), default=-1),
    )
    column_labels = [f"variable_{index}" for index in range(max_column_index + 1)]
    for overall_index, value in index_name_map.items():
        index = int(overall_index)
        if index < 0:
            continue
        try:
            _block_index, variable_name = value
        except (TypeError, ValueError):
            variable_name = str(value)
        if index >= len(column_labels):
            column_labels.extend(
                f"variable_{item}" for item in range(len(column_labels), index + 1)
            )
        column_labels[index] = str(variable_name)

    return SolverIncidenceMatrix(
        row_labels=tuple(row_labels),
        row_types=tuple(row_types),
        column_labels=tuple(column_labels),
        entries=tuple(entries),
    )


def _write_incidence_csv(
    matrix: SolverIncidenceMatrix,
    path: Path,
    *,
    system_name: str | None,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "row_index",
                "equation",
                "equation_type",
                "column_index",
                "variable",
                "incidence_kind",
            )
        )
        for entry in matrix.entries:
            writer.writerow(
                (
                    entry.row_index,
                    _short_label(matrix.row_labels[entry.row_index], system_name),
                    matrix.row_types[entry.row_index],
                    entry.column_index,
                    _short_label(matrix.column_labels[entry.column_index], system_name),
                    entry.kind,
                )
            )


def _write_incidence_png(matrix: SolverIncidenceMatrix, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    colors = {
        "variable": "#245f8f",
        "time_derivative": "#d76f30",
        "variable_and_time_derivative": "#6f4aa5",
    }
    figure = Figure(figsize=(8, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    largest_axis = max(matrix.row_count, matrix.column_count, 1)
    marker_size = max(0.2, min(16.0, 4000.0 / largest_axis))

    for kind, color in colors.items():
        matching = [entry for entry in matrix.entries if entry.kind == kind]
        if matching:
            axis.scatter(
                [entry.column_index for entry in matching],
                [entry.row_index for entry in matching],
                c=color,
                label=kind.replace("_", " "),
                linewidths=0,
                marker="s",
                s=marker_size,
            )

    axis.set_xlim(-0.5, matrix.column_count - 0.5)
    axis.set_ylim(matrix.row_count - 0.5, -0.5)
    axis.set_xlabel("Solver variable index")
    axis.set_ylabel("Equation index")
    axis.set_title(
        f"Solver incidence matrix: {matrix.row_count} x {matrix.column_count}, "
        f"{matrix.nonzero_count} nonzeros ({matrix.density:.3%})"
    )
    if matrix.entries:
        axis.legend(loc="upper right", fontsize="small")
    figure.savefig(path, dpi=180)


def write_solver_incidence_artifacts(
    *,
    model: Any,
    output_dir: str | Path,
    base_name: str = "solver_incidence_matrix",
) -> dict[str, Path]:
    """Write one labelled edge list and one static incidence image."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    matrix = collect_solver_incidence_matrix(model)
    csv_path = output_path / f"{base_name}.csv"
    png_path = output_path / f"{base_name}.png"
    _write_incidence_csv(matrix, csv_path, system_name=getattr(model, "Name", None))
    _write_incidence_png(matrix, png_path)
    return {
        "solver_incidence_matrix_csv": csv_path,
        "solver_incidence_matrix_png": png_path,
    }


__all__ = [
    "IncidenceEntry",
    "SolverIncidenceMatrix",
    "collect_solver_incidence_matrix",
    "write_solver_incidence_artifacts",
]
