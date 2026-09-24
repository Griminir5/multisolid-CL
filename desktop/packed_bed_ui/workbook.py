"""Dimension-based table selection and atomic Excel export; no Qt dependencies."""

from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
import json
from math import prod
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import warnings

import numpy as np
import xarray as xr
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableColumn, TableStyleInfo
from openpyxl.worksheet.filters import AutoFilter

from packed_bed.report_schema import describe_dataset, quantity_catalog
from packed_bed.reports import RESULTS_FILENAME, TIME_ATOL

from .project import read_documents, read_json


INFORMATION = "Case information"
MAX_ROWS, MAX_COLUMNS = 1_048_576, 16_384


class ExportCancelled(Exception):
    pass


def scalar(value):
    value = value.item() if isinstance(value, np.generic) else value
    return value.decode("utf-8") if isinstance(value, bytes) else value


def text_value(value, precision=15):
    value = scalar(value)
    return format(value, f".{precision}g") if isinstance(value, float) else str(value)


def axis_heading(axis):
    return axis["label"] + (f" ({axis['unit']})" if axis["unit"] else "")


def coordinates(schema, name):
    axis = schema["axes"].get(name)
    if axis is None:
        raise ValueError(f"Axis {name} is unavailable.")
    values = axis["values"]
    if values is None:
        raise ValueError(axis["error"] or f"Coordinates for {name} are unavailable.")
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{axis['label']} has no usable coordinates.")
    if values.dtype.kind in "fci" and not np.isfinite(values).all():
        raise ValueError(f"{axis['label']} contains non-finite coordinates.")
    return values


def coordinate_index(values, value, axis):
    matches = np.flatnonzero(values == value)
    if not len(matches) and values.dtype.kind in "fiu" and isinstance(value, (int, float)):
        scale = np.maximum(np.abs(values.astype(float)), abs(value))
        tolerance = 8 * np.abs(np.spacing(scale))
        if axis == "time":
            tolerance = np.maximum(tolerance, TIME_ATOL)
        matches = np.flatnonzero(np.abs(values - value) <= tolerance)
    if len(matches) != 1:
        reason = "ambiguous" if len(matches) else "not recorded"
        raise ValueError(f"{axis}={text_value(value)} is {reason}; update this selection.")
    return int(matches[0])


def resolve_selector(values, selector, axis, *, rows=False):
    if not isinstance(selector, dict):
        raise ValueError(f"Invalid selection for {axis}.")
    mode = selector.get("mode")
    if mode == "all" and rows:
        return np.arange(len(values))
    if mode in ("first", "last"):
        return np.array([0 if mode == "first" else len(values) - 1])
    if not rows and "index" in selector:
        index = selector["index"]
        if type(index) is not int or not 0 <= index < len(values):
            raise ValueError(f"Position {index} is unavailable on {axis}; update this selection.")
        return np.array([index])
    selected = selector.get("values", []) if rows else [selector.get("value")]
    if (rows and mode != "values") or (not rows and "value" not in selector):
        raise ValueError(f"Choose values for {axis}.")
    if not selected:
        raise ValueError(f"Choose at least one value for {axis}.")
    indices = np.array([coordinate_index(values, value, axis) for value in selected])
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate coordinates selected for {axis}.")
    return indices


def expand_columns(quantity, selections):
    """Expand only one quantity's non-principal dimensions."""
    count = 1
    for values in selections.values():
        count *= len(values)
    if count > MAX_COLUMNS - 1:
        raise ValueError("This selection creates too many Excel columns; narrow it.")
    return [{"quantity": quantity, "fixed": dict(zip(selections, combination))}
            for combination in product(*selections.values())]


def rule_columns(schema, rule):
    """Keep All numeric positions tied to the grid; named and chosen values stay exact."""
    axes = {dim: coordinates(schema, dim) for dim, s in rule["selections"].items() if s["mode"] == "all"}
    count = prod(len(axes[dim]) if s["mode"] == "all" else len(s["values"]) if s["mode"] == "values" else 1
                 for dim, s in rule["selections"].items())
    if count >= MAX_COLUMNS:
        raise ValueError("This selection creates too many Excel columns; narrow it.")
    selections = {}
    for dim, selection in rule["selections"].items():
        mode = selection["mode"]
        if mode == "all":
            values = axes[dim]
            selections[dim] = ([{"index": i} for i in range(len(values))] if values.dtype.kind in "fiu"
                               else [{"value": scalar(value)} for value in values])
        elif mode in ("first", "last"):
            selections[dim] = [{"mode": mode}]
        else:
            selections[dim] = [{"value": value} for value in selection["values"]]
    excluded = {selection_key(fixed) for fixed in rule.get("excluded", [])}
    return [{**column, "rule": rule["id"]} for column in expand_columns(rule["quantity"], selections)
            if selection_key(column["fixed"]) not in excluded]


def selection_key(fixed):
    return json.dumps(fixed, sort_keys=True)


def refresh_columns(schema, sheet):
    """Refresh All selections in place, preserving surviving columns' order and labels.

    Only explicit user removals go in a rule's exclusions. A smaller temporary grid
    must not prevent newly available positions appearing when the grid grows again.
    """
    for rule in sheet.get("column_rules", []):
        generated = rule_columns(schema, rule)
        fixed = {selection_key(column["fixed"]) for column in generated}
        kept = [column for column in sheet["columns"]
                if column.get("rule") != rule["id"] or selection_key(column["fixed"]) in fixed]
        members = {selection_key(column["fixed"]) for column in kept if column.get("rule") == rule["id"]}
        added = [column for column in generated if selection_key(column["fixed"]) not in members]
        if len(kept) + len(added) >= MAX_COLUMNS:
            raise ValueError("Worksheet exceeds Excel's column limit; narrow the selection.")
        end = max((i + 1 for i, column in enumerate(kept) if column.get("rule") == rule["id"]), default=len(kept))
        kept[end:end] = added
        sheet["columns"] = kept


def remove_missing_coordinates(schema, sheet):
    """Drop exact selections invalidated by new inputs, but keep them in unfinished drafts."""
    def available(dim, value):
        axis = schema["axes"].get(dim)
        if axis is None or axis["values"] is None:
            return True
        try:
            coordinate_index(axis["values"], value, dim)
            return True
        except ValueError:
            return False

    removed = 0
    if sheet["rows"].get("mode") == "values":
        values = sheet["rows"].get("values", [])
        kept = [value for value in values if available(sheet["axis"], value)]
        removed += len(values) - len(kept)
        sheet["rows"]["values"] = kept
    for rule in sheet.get("column_rules", []):
        for dim, selection in rule["selections"].items():
            if selection.get("mode") == "values":
                selection["values"] = [value for value in selection["values"] if available(dim, value)]
    columns = sheet["columns"]
    sheet["columns"] = [column for column in columns if all(
        "value" not in selector or available(dim, selector["value"]) for dim, selector in column["fixed"].items())]
    return removed + len(columns) - len(sheet["columns"])


def refresh_report(schema, definition):
    """Adapt to current inputs; keep unresolved selections while a draft is incomplete."""
    removed = 0
    for sheet in definition["sheets"]:
        removed += remove_missing_coordinates(schema, sheet)
        refreshed = deepcopy(sheet)
        try:
            refresh_columns(schema, refreshed)
        except (ValueError, KeyError, TypeError):
            continue  # Preserve the last usable columns; planning displays the input error.
        sheet.update(refreshed)
    return removed


def report_definition(value):
    """Read a portable layout, rejecting malformed data and stripping unrelated metadata."""
    def valid_selector(selector, *, multiple=False):
        if not isinstance(selector, dict):
            return False
        if selector.get("mode") in ("first", "last"):
            return True
        if multiple:
            return selector.get("mode") == "all" or (selector.get("mode") == "values"
                and isinstance(selector.get("values"), list)
                and all(type(v) in (str, int, float) for v in selector["values"]))
        return (type(selector.get("index")) is int and selector["index"] >= 0
                or type(selector.get("value")) in (str, int, float))

    try:
        if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value["sheets"], list):
            raise ValueError("Unsupported report definition version.")
        sheets = []
        for sheet in value["sheets"]:
            if (not isinstance(sheet["name"], str) or not isinstance(sheet["axis"], str)
                    or not valid_selector(sheet["rows"], multiple=True) or not isinstance(sheet["columns"], list)):
                raise ValueError("Malformed worksheet definition.")
            columns = []
            for column in sheet["columns"]:
                if (not isinstance(column["quantity"], str) or not isinstance(column["fixed"], dict)
                        or not isinstance(column.get("label", ""), str)
                        or not all(valid_selector(selector) for selector in column["fixed"].values())):
                    raise ValueError("Malformed column definition.")
                columns.append({key: column[key] for key in ("quantity", "fixed", "label", "rule") if key in column})
            result = {"name": sheet["name"], "axis": sheet["axis"], "rows": sheet["rows"], "columns": columns}
            if "column_rules" in sheet:
                rules = []
                for rule in sheet["column_rules"]:
                    if (not isinstance(rule["id"], str) or not isinstance(rule["quantity"], str)
                            or not isinstance(rule["selections"], dict) or not isinstance(rule.get("excluded", []), list)
                            or not all(valid_selector(s, multiple=True) for s in rule["selections"].values())):
                        raise ValueError("Malformed automatic column selection.")
                    rules.append({key: rule[key] for key in ("id", "quantity", "selections", "excluded") if key in rule})
                if len({r["id"] for r in rules}) != len(rules):
                    raise ValueError("Duplicate automatic column selection.")
                result["column_rules"] = rules
            sheets.append(result)
        return deepcopy({"version": 1, "sheets": sheets})
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed report definition.") from exc


def column_heading(column, quantity, schema, fixed):
    label = column.get("label", quantity["label"]).strip()
    if not label:
        raise ValueError("Column headings cannot be empty.")
    heading = f"{label} ({quantity['unit'] or 'unit unspecified'})"
    if "label" not in column:
        for dim, value in fixed.items():
            axis = schema["axes"][dim]
            displayed = axis.get('labels', {}).get(str(value), text_value(value, axis.get('precision', 15)))
            heading += f" | {axis['label']}={displayed}"
            if axis["unit"]:
                heading += f" {axis['unit']}"
    if len(heading) > 255:
        raise ValueError("Column heading exceeds 255 characters; give the column a shorter label.")
    return heading


def plan_sheet(schema, sheet):
    """Resolve a saved definition without reading any quantity values."""
    sheet = deepcopy(sheet)
    refresh_columns(schema, sheet)
    axis = sheet["axis"]
    values = coordinates(schema, axis)
    rows = resolve_selector(values, sheet["rows"], axis, rows=True)
    columns = sheet["columns"]
    if not columns:
        raise ValueError("Add at least one value column.")
    if len(rows) + 1 > MAX_ROWS or len(columns) + 1 > MAX_COLUMNS:
        raise ValueError("Worksheet exceeds Excel's row/column limits; narrow the selection.")
    resolved, headings = [], [axis_heading(schema["axes"][axis])]
    seen_headings, fixed_indices, coordinate_cache = {headings[0].casefold()}, {}, {}
    for column in columns:
        name = column["quantity"]
        quantity = schema["quantities"].get(name)
        if quantity is None:
            group = quantity_catalog().get(name, {}).get("report")
            hint = f" Enable '{group}' in General and record it in another run." if group else ""
            raise ValueError(f"Quantity '{name}' is unavailable.{hint}")
        if axis not in quantity["dimensions"]:
            raise ValueError(f"{quantity['label']} does not have the sheet's {axis} axis.")
        dimensions = [d for d in quantity["dimensions"] if d != axis]
        if set(column["fixed"]) != set(dimensions):
            raise ValueError(f"Choose exactly one coordinate for every other axis of {name}.")
        indices, fixed = {}, {}
        for dim in dimensions:
            if dim not in coordinate_cache:
                coordinate_cache[dim] = coordinates(schema, dim)
            coords = coordinate_cache[dim]
            selector = column["fixed"][dim]
            key = (dim, repr(selector))
            if key not in fixed_indices:
                fixed_indices[key] = resolve_selector(coords, selector, dim)[0]
            index = fixed_indices[key]
            indices[dim], fixed[dim] = int(index), scalar(coords[index])
        heading = column_heading(column, quantity, schema, fixed)
        if heading.casefold() in seen_headings:
            raise ValueError(f"Duplicate heading '{heading}'; rename a column.")
        seen_headings.add(heading.casefold())
        headings.append(heading)
        resolved.append({"quantity": name, "indices": indices, "fixed": fixed, "unit": quantity["unit"]})
    shown = np.array([schema['axes'][axis].get('labels', {}).get(str(value), value) for value in values[rows]]) if schema['axes'][axis].get('labels') else values[rows]
    return {"name": sheet["name"], "axis": axis, "rows": rows, "values": shown,
            "headings": headings, "columns": resolved, "selection": sheet["rows"]}


def plan_workbook(schema, definition):
    if not isinstance(definition, dict) or definition.get("version") != 1:
        raise ValueError("Unsupported report definition version.")
    if not isinstance(definition.get("sheets"), list) or not definition["sheets"]:
        raise ValueError("Add a data sheet to export.")
    tables, errors, names = [], [], {INFORMATION.casefold()}
    for sheet in definition["sheets"]:
        name = sheet.get("name", "") if isinstance(sheet, dict) else ""
        try:
            if (not isinstance(name, str) or not name.strip() or len(name) > 31
                    or re.search(r"[\\/*?:\[\]\x00-\x1f]", name) or name.startswith("'") or name.endswith("'")
                    or name.casefold() in names or name.casefold() == "history"):
                raise ValueError("Use a unique Excel sheet name (1–31 characters); Case information is reserved.")
            names.add(name.casefold())
            tables.append(plan_sheet(schema, sheet))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append(f"{name or 'Unnamed sheet'}: {exc}")
    if errors:
        raise ValueError("\n".join(errors))
    return tables


def table_rows(dataset, table, *, limit=None, column_limit=None, cancelled=lambda: False):
    columns = table["columns"][:None if column_limit is None else max(0, column_limit - 1)]
    rows = table["rows"][:limit]
    size = max(1, min(256, 65536 // max(1, len(columns))))
    for start in range(0, len(rows), size):
        if cancelled():
            raise ExportCancelled()
        indices = rows[start:start + size]
        arrays = []
        for column in columns:
            selected = dataset[column["quantity"]].isel(column["indices"])
            if selected.dims != (table["axis"],):
                raise ValueError("Quantity did not resolve to the principal axis.")
            arrays.append(selected.isel({table["axis"]: indices}).values)
        for i, row in enumerate(indices):
            yield [scalar(table["values"][start + i]), *[scalar(a[i]) for a in arrays]]


def run_information(run_folder, fingerprint=None):
    snapshot = read_json(run_folder / "snapshot.json")
    if not all(snapshot.get(key) for key in ("case_id", "attempt_id")):
        raise ValueError("The retained run has no readable case/attempt identity.")
    information = {"Run": snapshot, "Inputs": read_documents(run_folder / "inputs"),
                   "Export": {"created_at": datetime.now(timezone.utc).isoformat(),
                              "inputs_changed": snapshot.get("fingerprint") != fingerprint
                              if fingerprint is not None else "Unavailable"}}
    for section, path in (("Status", run_folder / "status.json"),
                          ("Manifest", run_folder / "output" / "manifest.json")):
        try:
            information[section] = read_json(path)
        except (OSError, ValueError):
            information[section] = "Unavailable"
    return information


def information_rows(information, tables, schema):
    def flatten(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from flatten(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                yield from flatten(child, f"{path}[{i}]")
        else:
            yield path, value
    yield ["Section", "Setting", "Value"]
    for section, value in information.items():
        for key, val in flatten(value):
            yield [section, key, val]
    yield []
    yield ["Column dictionary"]
    fixed_axes = list(dict.fromkeys(d for t in tables for c in t["columns"] for d in c["fixed"]))
    yield ["Worksheet", "Column", "Heading", "Quantity", "Value unit", "Principal axis",
           "Row selection", "Rows", "First coordinate", "Last coordinate",
           *[axis_heading(schema["axes"][dim]) for dim in fixed_axes]]
    for table in tables:
        axis = table["axis"]
        for i, column in enumerate([None, *table["columns"]]):
            yield [table["name"], get_column_letter(i + 1), table["headings"][i],
                   column["quantity"] if column else axis,
                   column["unit"] if column else schema["axes"][axis]["unit"],
                   axis_heading(schema["axes"][axis]), table["selection"]["mode"], len(table["rows"]),
                   scalar(table["values"][0]), scalar(table["values"][-1]),
                   *[column["fixed"].get(dim) if column else None for dim in fixed_axes]]


def excel_value(value):
    value = scalar(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, str) and (len(value) > 32767 or re.search(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", value)):
        raise ValueError("A cell contains overlong text or characters Excel cannot store.")
    return value


def validate_information(information, tables, schema):
    for count, row in enumerate(information_rows(information, tables, schema), 1):
        if count > MAX_ROWS or len(row) > MAX_COLUMNS:
            raise ValueError("Case information exceeds Excel's size limits.")
        for value in row:
            excel_value(value)


def write_workbook(run_folder, definition, destination, *, fingerprint=None, cancelled=lambda: False):
    """Export retained data atomically; failures never replace an existing workbook."""
    run_folder, destination = Path(run_folder), Path(destination)
    if destination.suffix.lower() != ".xlsx":
        raise ValueError("Choose an .xlsx file.")
    if destination.resolve().is_relative_to(run_folder.resolve()):
        raise ValueError("Save the workbook outside the replaceable run folder.")
    workbook, temporary = None, None
    try:
        with xr.open_dataset(run_folder / "output" / RESULTS_FILENAME, engine="scipy") as dataset:
            schema = describe_dataset(dataset)
            tables = plan_workbook(schema, definition)
            information = run_information(run_folder, fingerprint)
            validate_information(information, tables, schema)
            if cancelled():
                raise ExportCancelled()
            workbook = Workbook(write_only=True)
            def append(ws, row, *, header=False):
                cells = []
                for value in row:
                    cell = WriteOnlyCell(ws, value=excel_value(value))
                    if isinstance(cell.value, str):
                        cell.data_type = "s"  # User labels and metadata are literal, never formulas.
                    if header:
                        cell.font = Font(bold=True)
                        cell.alignment = Alignment(wrap_text=True)
                    cells.append(cell)
                ws.append(cells)
            ws = workbook.create_sheet(INFORMATION)
            ws.column_dimensions["A"].width = 25
            ws.column_dimensions["B"].width = 48
            ws.column_dimensions["C"].width = 48
            for row in information_rows(information, tables, schema):
                if cancelled():
                    raise ExportCancelled()
                append(ws, row)
            for i, table in enumerate(tables, 1):
                ws = workbook.create_sheet(table["name"])
                ws.freeze_panes = "B2"
                ws.row_dimensions[1].height = 60
                for j in range(len(table["headings"])):
                    ws.column_dimensions[get_column_letter(j + 1)].width = 20 if j == 0 else 32
                native = Table(displayName=f"Report{i}",
                               ref=f"A1:{get_column_letter(len(table['headings']))}{len(table['rows']) + 1}",
                               tableColumns=[TableColumn(id=j, name=h) for j, h in enumerate(table["headings"], 1)])
                native.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
                native.autoFilter = AutoFilter(ref=native.ref)
                # Column definitions are explicit: write-only worksheets cannot infer headers.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="In write-only mode you must add table columns manually")
                    ws.add_table(native)
                append(ws, table["headings"], header=True)
                for row in table_rows(dataset, table, cancelled=cancelled):
                    append(ws, row)
            with NamedTemporaryFile(dir=destination.parent, prefix=".report-", suffix=".xlsx", delete=False) as f:
                temporary = Path(f.name)
            workbook.save(temporary)
            if cancelled():
                raise ExportCancelled()
            temporary.replace(destination)
    finally:
        if workbook:
            for ws in workbook.worksheets:
                if not ws.closed:
                    ws.close()
                if ws._writer:
                    try:
                        ws._writer.cleanup()
                    except FileNotFoundError:
                        pass
            workbook.close()
        if temporary:
            temporary.unlink(missing_ok=True)
