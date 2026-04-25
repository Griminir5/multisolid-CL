from __future__ import annotations

import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_KIND_CODES = {
    "variable": 0,
    "time_derivative": 1,
    "variable_and_time_derivative": 2,
}
_KIND_LABELS = {value: key for key, value in _KIND_CODES.items()}


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


def _safe_int(value: Any) -> int:
    return int(value)


def _valid_indexes(values: Any) -> set[int]:
    return {index for index in (_safe_int(value) for value in values) if index >= 0}


def _short_label(label: str, system_name: str | None) -> str:
    if system_name and label.startswith(f"{system_name}."):
        return label[len(system_name) + 1 :]
    return label


def _group_label(label: str, system_name: str | None) -> str:
    name = _short_label(label, system_name)
    name = re.sub(r"\([^)]*\)$", "", name)
    name = re.sub(r"cell_\d+", "cell_*", name)
    name = re.sub(r"face_\d+", "face_*", name)
    return name


def _equation_type_label(equation_type: Any) -> str:
    label = str(equation_type)
    return label.rsplit(".", 1)[-1]


def _collect_equation_execution_infos(model: Any) -> list[Any]:
    infos: list[Any] = []
    for equation in getattr(model, "Equations", ()):
        infos.extend(getattr(equation, "EquationExecutionInfos", ()))
    return infos


def collect_solver_incidence_matrix(model: Any) -> SolverIncidenceMatrix:
    infos = _collect_equation_execution_infos(model)
    if not infos:
        raise ValueError("The DAETools model has no equation execution info. Initialize the simulation first.")

    max_row_index = max(_safe_int(info.EquationIndex) for info in infos)
    row_labels = [f"equation_{index}" for index in range(max_row_index + 1)]
    row_types = ["unknown" for _index in range(max_row_index + 1)]
    entries: list[IncidenceEntry] = []

    for info in infos:
        row_index = _safe_int(info.EquationIndex)
        row_labels[row_index] = str(info.Name)
        row_types[row_index] = _equation_type_label(getattr(info, "EquationType", "unknown"))

        variable_indexes = _valid_indexes(getattr(info, "VariableIndexes", ()))
        derivative_indexes = _valid_indexes(getattr(info, "DiffVariableIndexes", ()))
        for column_index in sorted(variable_indexes | derivative_indexes):
            if column_index in variable_indexes and column_index in derivative_indexes:
                kind = "variable_and_time_derivative"
            elif column_index in derivative_indexes:
                kind = "time_derivative"
            else:
                kind = "variable"
            entries.append(
                IncidenceEntry(
                    row_index=row_index,
                    column_index=column_index,
                    kind=kind,
                )
            )

    index_name_map = getattr(model, "OverallIndex_BlockIndex_VariableNameMap", {})
    mapped_column_indexes = [_safe_int(index) for index in index_name_map]
    max_column_index = max(
        (
            max(mapped_column_indexes, default=-1),
            max((entry.column_index for entry in entries), default=-1),
        )
    )
    column_labels = [f"variable_{index}" for index in range(max_column_index + 1)]
    for overall_index, value in index_name_map.items():
        index = _safe_int(overall_index)
        if index < 0:
            continue
        try:
            _block_index, variable_name = value
        except (TypeError, ValueError):
            variable_name = str(value)
        if index >= len(column_labels):
            column_labels.extend(f"variable_{item}" for item in range(len(column_labels), index + 1))
        column_labels[index] = str(variable_name)

    return SolverIncidenceMatrix(
        row_labels=tuple(row_labels),
        row_types=tuple(row_types),
        column_labels=tuple(column_labels),
        entries=tuple(entries),
    )


def _axis_nonzero_counts(size: int, entries: tuple[IncidenceEntry, ...], axis: str) -> list[int]:
    counts = [0 for _index in range(size)]
    if axis == "row":
        for entry in entries:
            counts[entry.row_index] += 1
    elif axis == "column":
        for entry in entries:
            counts[entry.column_index] += 1
    else:
        raise ValueError(f"Unknown incidence axis: {axis}")
    return counts


def _group_summaries(labels: tuple[str, ...], counts: list[int], system_name: str | None) -> list[dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    ordered_names: list[str] = []
    for index, label in enumerate(labels):
        group = _group_label(label, system_name)
        if group not in summaries:
            summaries[group] = {
                "name": group,
                "first_index": index,
                "last_index": index,
                "items": 0,
                "nonzeros": 0,
            }
            ordered_names.append(group)
        summary = summaries[group]
        summary["last_index"] = index
        summary["items"] += 1
        summary["nonzeros"] += counts[index] if index < len(counts) else 0
    return [summaries[name] for name in ordered_names]


def _axis_boundaries(labels: tuple[str, ...], system_name: str | None) -> list[int]:
    boundaries: list[int] = []
    previous_group: str | None = None
    for index, label in enumerate(labels):
        group = _group_label(label, system_name)
        if previous_group is not None and group != previous_group:
            boundaries.append(index)
        previous_group = group
    return boundaries


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.4g}"


def _write_incidence_csv(matrix: SolverIncidenceMatrix, path: Path, *, system_name: str | None) -> None:
    row_counts = _axis_nonzero_counts(matrix.row_count, matrix.entries, "row")
    column_counts = _axis_nonzero_counts(matrix.column_count, matrix.entries, "column")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row_index",
                "equation",
                "equation_group",
                "equation_type",
                "equation_nonzeros",
                "column_index",
                "variable",
                "variable_group",
                "variable_nonzeros",
                "incidence_kind",
            ]
        )
        for entry in matrix.entries:
            row_label = matrix.row_labels[entry.row_index]
            column_label = matrix.column_labels[entry.column_index]
            writer.writerow(
                [
                    entry.row_index,
                    _short_label(row_label, system_name),
                    _group_label(row_label, system_name),
                    matrix.row_types[entry.row_index],
                    row_counts[entry.row_index],
                    entry.column_index,
                    _short_label(column_label, system_name),
                    _group_label(column_label, system_name),
                    column_counts[entry.column_index],
                    entry.kind,
                ]
            )


def _group_table(title: str, summaries: list[dict[str, Any]]) -> str:
    rows = []
    for summary in summaries:
        index_range = f"{summary['first_index']}-{summary['last_index']}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(summary['name']))}</td>"
            f"<td>{html.escape(index_range)}</td>"
            f"<td>{_format_number(int(summary['items']))}</td>"
            f"<td>{_format_number(int(summary['nonzeros']))}</td>"
            "</tr>"
        )
    return (
        f"<details open><summary>{html.escape(title)}</summary>"
        "<table>"
        "<thead><tr><th>Group</th><th>Index range</th><th>Items</th><th>Nonzeros</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</details>"
    )


def _write_incidence_html(
    matrix: SolverIncidenceMatrix,
    path: Path,
    *,
    system_name: str | None,
    csv_path: Path,
    xpm_path: Path | None,
    xpm_error: str | None,
) -> None:
    row_counts = _axis_nonzero_counts(matrix.row_count, matrix.entries, "row")
    column_counts = _axis_nonzero_counts(matrix.column_count, matrix.entries, "column")
    row_summaries = _group_summaries(matrix.row_labels, row_counts, system_name)
    column_summaries = _group_summaries(matrix.column_labels, column_counts, system_name)

    payload = {
        "nRows": matrix.row_count,
        "nCols": matrix.column_count,
        "nnz": matrix.nonzero_count,
        "density": matrix.density,
        "rowLabels": [_short_label(label, system_name) for label in matrix.row_labels],
        "rowTypes": list(matrix.row_types),
        "colLabels": [_short_label(label, system_name) for label in matrix.column_labels],
        "entries": [
            [entry.row_index, entry.column_index, _KIND_CODES[entry.kind]]
            for entry in matrix.entries
        ],
        "rowBoundaries": _axis_boundaries(matrix.row_labels, system_name),
        "colBoundaries": _axis_boundaries(matrix.column_labels, system_name),
        "kindLabels": _KIND_LABELS,
    }
    payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    artifact_links = [
        f'<a href="{html.escape(csv_path.name)}">incidence CSV</a>',
    ]
    if xpm_path is not None:
        artifact_links.append(f'<a href="{html.escape(xpm_path.name)}">raw DAETools XPM</a>')
    if xpm_error:
        artifact_links.append(f"<span>raw XPM unavailable: {html.escape(xpm_error)}</span>")

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Solver Incidence Matrix</title>
<style>
:root {{
  color-scheme: light;
  --ink: #1f2933;
  --muted: #64748b;
  --line: #d8dee8;
  --panel: #f8fafc;
  --accent: #245f8f;
  --derivative: #d76f30;
  --both: #6f4aa5;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "Segoe UI", Arial, sans-serif;
  color: var(--ink);
  background: #ffffff;
}}
main {{
  max-width: 1240px;
  margin: 0 auto;
  padding: 28px;
}}
h1 {{
  margin: 0 0 6px;
  font-size: 28px;
  font-weight: 650;
  letter-spacing: 0;
}}
.subtitle {{
  margin: 0 0 22px;
  color: var(--muted);
}}
.stats {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
  margin-bottom: 20px;
}}
.stat {{
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  padding: 12px 14px;
}}
.stat span {{
  display: block;
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}}
.stat strong {{
  display: block;
  margin-top: 4px;
  font-size: 22px;
}}
.layout {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
  gap: 22px;
  align-items: start;
}}
.matrix-panel {{
  min-width: 0;
}}
#matrixCanvas {{
  display: block;
  width: 100%;
  max-width: 880px;
  aspect-ratio: 1 / 1;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
}}
.legend {{
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 12px;
  color: var(--muted);
  font-size: 13px;
}}
.swatch {{
  width: 12px;
  height: 12px;
  border-radius: 2px;
  display: inline-block;
  margin-right: 6px;
  vertical-align: -1px;
}}
.variable {{ background: var(--accent); }}
.time_derivative {{ background: var(--derivative); }}
.variable_and_time_derivative {{ background: var(--both); }}
.side {{
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
  background: var(--panel);
}}
.side h2 {{
  margin: 0 0 10px;
  font-size: 16px;
}}
#hoverReadout {{
  min-height: 118px;
  white-space: pre-wrap;
  font-family: Consolas, "Liberation Mono", monospace;
  font-size: 12px;
  line-height: 1.45;
  color: #111827;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
}}
.links {{
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 13px;
}}
a {{ color: #1d4ed8; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.groups {{
  margin-top: 22px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 18px;
}}
details {{
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}}
summary {{
  cursor: pointer;
  padding: 12px 14px;
  font-weight: 650;
  background: var(--panel);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}}
th, td {{
  padding: 8px 10px;
  border-top: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}}
th {{
  color: var(--muted);
  font-weight: 650;
}}
@media (max-width: 900px) {{
  main {{ padding: 18px; }}
  .layout {{ grid-template-columns: 1fr; }}
  .side {{ order: -1; }}
}}
</style>
</head>
<body>
<main>
  <h1>{html.escape(system_name or "Solver")} Incidence Matrix</h1>
  <p class="subtitle">Rows are expanded DAETools equation executions; columns are solver variables.</p>
  <section class="stats">
    <div class="stat"><span>Rows</span><strong>{_format_number(matrix.row_count)}</strong></div>
    <div class="stat"><span>Columns</span><strong>{_format_number(matrix.column_count)}</strong></div>
    <div class="stat"><span>Nonzeros</span><strong>{_format_number(matrix.nonzero_count)}</strong></div>
    <div class="stat"><span>Density</span><strong>{matrix.density:.3%}</strong></div>
  </section>
  <section class="layout">
    <div class="matrix-panel">
      <canvas id="matrixCanvas" aria-label="Solver incidence matrix"></canvas>
      <div class="legend">
        <span><i class="swatch variable"></i>variable</span>
        <span><i class="swatch time_derivative"></i>time derivative</span>
        <span><i class="swatch variable_and_time_derivative"></i>both</span>
      </div>
    </div>
    <aside class="side">
      <h2>Selection</h2>
      <div id="hoverReadout">Move over the matrix to inspect a row and column.</div>
      <div class="links">{''.join(artifact_links)}</div>
    </aside>
  </section>
  <section class="groups">
    {_group_table("Equation Groups", row_summaries)}
    {_group_table("Variable Groups", column_summaries)}
  </section>
</main>
<script>
const data = {payload_json};
const colors = {{
  0: "#245f8f",
  1: "#d76f30",
  2: "#6f4aa5"
}};
const canvas = document.getElementById("matrixCanvas");
const readout = document.getElementById("hoverReadout");
const ctx = canvas.getContext("2d");
const incidenceByCell = new Map(data.entries.map((entry) => [`${{entry[0]}},${{entry[1]}}`, entry[2]]));

function resizeCanvas() {{
  const cssSize = Math.min(880, canvas.parentElement.clientWidth);
  const ratio = window.devicePixelRatio || 1;
  canvas.style.width = `${{cssSize}}px`;
  canvas.style.height = `${{cssSize}}px`;
  canvas.width = Math.round(cssSize * ratio);
  canvas.height = Math.round(cssSize * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  drawMatrix(cssSize);
}}

function drawMatrix(size) {{
  const cellWidth = size / data.nCols;
  const cellHeight = size / data.nRows;
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, size, size);

  ctx.strokeStyle = "rgba(100, 116, 139, 0.18)";
  ctx.lineWidth = 1;
  for (const boundary of data.colBoundaries) {{
    const x = Math.round(boundary * cellWidth) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, size);
    ctx.stroke();
  }}
  for (const boundary of data.rowBoundaries) {{
    const y = Math.round(boundary * cellHeight) + 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(size, y);
    ctx.stroke();
  }}

  const markWidth = Math.max(0.85, cellWidth);
  const markHeight = Math.max(0.85, cellHeight);
  for (const entry of data.entries) {{
    ctx.fillStyle = colors[entry[2]];
    ctx.fillRect(entry[1] * cellWidth, entry[0] * cellHeight, markWidth, markHeight);
  }}
}}

function kindName(kindCode) {{
  return data.kindLabels[String(kindCode)] || "empty";
}}

canvas.addEventListener("mousemove", (event) => {{
  const rect = canvas.getBoundingClientRect();
  const col = Math.min(data.nCols - 1, Math.max(0, Math.floor((event.clientX - rect.left) / rect.width * data.nCols)));
  const row = Math.min(data.nRows - 1, Math.max(0, Math.floor((event.clientY - rect.top) / rect.height * data.nRows)));
  const kind = incidenceByCell.get(`${{row}},${{col}}`);
  readout.textContent =
    `row ${{row}}: ${{data.rowLabels[row]}}\\n` +
    `type: ${{data.rowTypes[row]}}\\n` +
    `column ${{col}}: ${{data.colLabels[col]}}\\n` +
    `incidence: ${{kind === undefined ? "none" : kindName(kind)}}`;
}});

canvas.addEventListener("mouseleave", () => {{
  readout.textContent = "Move over the matrix to inspect a row and column.";
}});

window.addEventListener("resize", resizeCanvas);
resizeCanvas();
</script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def _try_save_solver_xpm(solver: Any, path: Path) -> str | None:
    exporters = []
    save_matrix = getattr(solver, "SaveMatrixAsXPM", None)
    if callable(save_matrix):
        exporters.append(("daeIDAS.SaveMatrixAsXPM", save_matrix))

    linear_solver = getattr(solver, "LASolver", None)
    save_linear_matrix = getattr(linear_solver, "SaveAsXPM", None)
    if callable(save_linear_matrix):
        exporters.append(("LASolver.SaveAsXPM", save_linear_matrix))

    errors: list[str] = []
    for name, exporter in exporters:
        try:
            exporter(str(path))
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        if path.exists() and path.stat().st_size > 0:
            return None
        errors.append(f"{name}: no file was written")

    return "; ".join(errors) if errors else "no XPM exporter was available"


def write_solver_incidence_artifacts(
    *,
    model: Any,
    solver: Any,
    output_dir: str | Path,
    base_name: str = "solver_incidence_matrix",
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    system_name = getattr(model, "Name", None)
    matrix = collect_solver_incidence_matrix(model)

    csv_path = output_path / f"{base_name}.csv"
    html_path = output_path / f"{base_name}.html"
    xpm_path = output_path / f"{base_name}.xpm"

    _write_incidence_csv(matrix, csv_path, system_name=system_name)
    xpm_error = _try_save_solver_xpm(solver, xpm_path)
    resolved_xpm_path = None if xpm_error else xpm_path
    _write_incidence_html(
        matrix,
        html_path,
        system_name=system_name,
        csv_path=csv_path,
        xpm_path=resolved_xpm_path,
        xpm_error=xpm_error,
    )

    artifacts = {
        "solver_incidence_matrix_html": html_path,
        "solver_incidence_matrix_csv": csv_path,
    }
    if resolved_xpm_path is not None:
        artifacts["solver_incidence_matrix_xpm"] = resolved_xpm_path
    return artifacts


__all__ = [
    "IncidenceEntry",
    "SolverIncidenceMatrix",
    "collect_solver_incidence_matrix",
    "write_solver_incidence_artifacts",
]
