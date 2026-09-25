"""Project reports: Case is an axis, while each run keeps its own native grid."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np
import xarray as xr

from packed_bed.report_schema import axis_description, describe_dataset
from packed_bed.reports import RESULTS_FILENAME

from .workbook import (
    MAX_COLUMNS, MAX_ROWS, ExportCancelled, axis_heading, column_heading, coordinates,
    expand_columns, information_rows, plan_sheet, plan_workbook, report_definition,
    resolve_selector, run_information, scalar, table_rows, write_tables,
)


INFORMATION = "Results information"


def results_definition(value):
    definition = report_definition(value)
    ids = value.get("case_ids", [])
    if not isinstance(ids, list) or not all(isinstance(ident, str) for ident in ids) or len(set(ids)) != len(ids):
        raise ValueError("Malformed project case selection.")
    definition["case_ids"] = ids.copy()
    return definition


def results_template(definition):
    """Keep a layout portable by replacing project case IDs with numbered slots."""
    template = results_definition(definition)
    ids = template.pop("case_ids")
    if not ids:
        raise ValueError("Select cases before saving a results template.")
    slots = {ident: index for index, ident in enumerate(ids)}
    try:
        for sheet in template["sheets"]:
            for column in sheet["columns"]:
                if "case" in column["fixed"]:
                    column["fixed"]["case"] = {"index": slots[column["fixed"]["case"]["value"]]}
            if sheet["axis"] == "case" and sheet["rows"]["mode"] == "values":
                sheet["rows"]["values"] = [slots[ident] for ident in sheet["rows"]["values"]]
    except KeyError as exc:
        raise ValueError("The report refers to an unselected case; repair its selection before saving a template.") from exc
    template.update(kind="project_results_template", case_count=len(ids))
    return template


def apply_results_template(value, case_ids):
    if value.get("kind") != "project_results_template" or type(value.get("case_count")) is not int or value["case_count"] < 1:
        raise ValueError("Choose a project Results template.")
    definition = report_definition(value)
    if len(case_ids) != value["case_count"]:
        raise ValueError(f"Select {value['case_count']} successful cases for this template; {len(case_ids)} are selected.")
    def case_at(index):
        if type(index) is not int or not 0 <= index < len(case_ids):
            raise ValueError("The template contains an invalid case position.")
        return case_ids[index]
    for sheet in definition["sheets"]:
        for column in sheet["columns"]:
            if "case" in column["fixed"]:
                column["fixed"]["case"] = {"value": case_at(column["fixed"]["case"].get("index"))}
        if sheet["axis"] == "case" and sheet["rows"]["mode"] == "values":
            sheet["rows"]["values"] = [case_at(index) for index in sheet["rows"]["values"]]
    definition["case_ids"] = list(case_ids)
    return definition


@contextmanager
def open_results(project, case_ids, *, cancelled=lambda: False):
    """Open only selected runs; no solver or plugin code is needed to read results."""
    cases = {case.id: case for case in project.cases}
    sources = {}
    with ExitStack() as stack:
        for ident in case_ids:
            if cancelled():
                raise ExportCancelled()
            case = cases.get(ident)
            source = sources[ident] = {"name": case.name if case else f"Missing case {ident}", "error": ""}
            if case is None:
                source["error"] = "This case was deleted or replaced. Select its replacement explicitly."
                continue
            try:
                try:
                    fingerprint = case.fingerprint()
                except (OSError, ValueError, TypeError):
                    fingerprint = None
                source["information"] = run_information(case.run_folder, fingerprint)
                if source["information"]["Run"]["case_id"] != ident:
                    raise ValueError("The retained snapshot belongs to another case.")
                status = source["information"]["Status"]
                if not isinstance(status, dict) or status.get("state") != "completed":
                    raise ValueError("Project Results requires a successful run. "
                                     "Use the case's Report tab to inspect any partial data.")
                source["information"]["Case"] = {
                    "name": case.name, "id": ident, "origin": case.metadata.get("origin", "Independent"),
                    "study_id": case.metadata.get("study_id", ""),
                    "study_selections": case.metadata.get("selections", {}),
                }
                source["dataset"] = stack.enter_context(xr.open_dataset(
                    case.run_folder / "output" / RESULTS_FILENAME, engine="scipy"))
                source["schema"] = describe_dataset(source["dataset"])
            except (OSError, ValueError, TypeError, KeyError) as exc:
                source["error"] = f"No readable retained results: {exc}"
        yield describe_results(sources)


def describe_results(sources):
    """Coordinate unions are picker choices only; they never align or reshape data."""
    labels = {ident: source["name"] for ident, source in sources.items()}
    for ident, label in list(labels.items()):
        if list(labels.values()).count(label) > 1:
            for other, source in sources.items():
                if source["name"] == label:
                    labels[other] = f"{label} [{other[:8]}]"
    axes = {"case": {**axis_description("case", list(sources)), "label": "Case", "labels": labels}}
    quantities, values = {}, {}
    for source in sources.values():
        if source["error"]:
            continue
        for name, spec in source["schema"]["quantities"].items():
            quantities.setdefault(name, {**spec, "dimensions": ("case", *spec["dimensions"])})
        for dim, axis in source["schema"]["axes"].items():
            axes.setdefault(dim, deepcopy(axis))
            axes[dim].setdefault("labels", {}).update(axis.get("labels", {}))
            if axis["values"] is not None:
                values.setdefault(dim, []).append(axis["values"])
    for dim, arrays in values.items():
        combined = np.concatenate(arrays)
        combined = np.unique(combined) if combined.dtype.kind in "fiu" else np.array(list(dict.fromkeys(combined)))
        axes[dim].update(axis_description(dim, combined, attrs={
            "long_name": axes[dim]["label"], "units": axes[dim]["unit"]}))
    return {"axes": axes, "quantities": quantities, "sources": sources}


def exact_selector(values, selector, axis, *, rows=False):
    """Cross-case comparisons require recorded coordinates, with no nearest matching."""
    selected = selector.get("values", []) if rows else [selector["value"]] if "value" in selector else []
    for value in selected:
        if np.count_nonzero(values == value) != 1:
            raise ValueError(f"{axis}={value} is not uniquely recorded; choose an available value or First/Last.")
    return resolve_selector(values, selector, axis, rows=rows)


def source_for(schema, ident, quantity):
    source = schema["sources"].get(ident)
    if source is None:
        raise ValueError(f"Case {ident} is not selected. Use Select cases or remove its columns.")
    if source["error"]:
        raise ValueError(f"{source['name']}: {source['error']}")
    spec = source["schema"]["quantities"].get(quantity)
    if spec is None:
        raise ValueError(f"{source['name']}: '{quantity}' was not recorded. Select another quantity or case.")
    expected = schema["quantities"][quantity]
    if spec["unit"] != expected["unit"] or tuple(spec["dimensions"]) != tuple(expected["dimensions"][1:]):
        raise ValueError(f"{source['name']}: '{quantity}' has incompatible units or dimensions.")
    for dim in spec["dimensions"]:
        if source["schema"]["axes"][dim]["unit"] != schema["axes"][dim]["unit"]:
            raise ValueError(f"{source['name']}: '{dim}' has incompatible coordinate units.")
    return source


def project_columns(schema, quantity, selections):
    """Expand All against each case's own coordinates, then save explicit columns."""
    if "case" in selections:
        case_values = coordinates(schema, "case")
        ids = case_values[exact_selector(case_values, selections["case"], "case", rows=True)]
    else:
        ids = [None]  # Case is the row axis.
    columns = []
    for ident in ids:
        native = source_for(schema, str(ident), quantity)["schema"] if ident is not None else schema
        options = {}
        for dim, selection in selections.items():
            if dim == "case":
                options[dim] = [{"value": str(ident)}]
            elif selection["mode"] in ("first", "last"):
                options[dim] = [deepcopy(selection)]
            else:
                coords = coordinates(native, dim)
                indices = exact_selector(coords, selection, dim, rows=True)
                options[dim] = [{"value": scalar(coords[i])} for i in indices]
        columns.extend(expand_columns(quantity, options))
        if len(columns) >= MAX_COLUMNS:
            raise ValueError("This selection creates too many Excel columns; narrow it.")
    return columns


def plan_project_sheet(schema, sheet):
    axis, columns = sheet["axis"], sheet["columns"]
    if not columns:
        raise ValueError("Add at least one value column.")
    if len(columns) >= MAX_COLUMNS:
        raise ValueError("Worksheet exceeds Excel's column limit; narrow the selection.")
    values = None
    if axis == "case":
        cases = coordinates(schema, "case")
        ids = cases[exact_selector(cases, sheet["rows"], "case", rows=True)]
        values = np.array([schema["axes"]["case"]["labels"][ident] for ident in ids])
    resolved, headings = [], [axis_heading(schema["axes"][axis])]
    for column in columns:
        name = column["quantity"]
        spec = schema["quantities"].get(name)
        if spec is None:
            raise ValueError(f"Quantity '{name}' is unavailable in the selected results.")
        if axis not in spec["dimensions"] or set(column["fixed"]) != set(spec["dimensions"]) - {axis}:
            raise ValueError(f"Choose one coordinate for every other axis of {name}.")
        if axis != "case":
            ident = column["fixed"]["case"].get("value")
            if ident is None:
                raise ValueError("Choose a case explicitly for each column.")
            ids = [ident]
        per_case = {}
        for ident in ids:
            source = source_for(schema, ident, name)
            native, indices, fixed = source["schema"], {}, {}
            try:
                for dim, selector in column["fixed"].items():
                    if dim == "case":
                        continue
                    coords = coordinates(native, dim)
                    index = int(exact_selector(coords, selector, dim)[0])
                    indices[dim], fixed[dim] = index, scalar(coords[index])
                if axis == "case":
                    item = {"indices": indices, "fixed": fixed}
                else:
                    coords = coordinates(native, axis)
                    exact_selector(coords, sheet["rows"], axis, rows=True)
                    item = plan_sheet(native, {**sheet, "column_rules": [], "columns": [
                        {"quantity": name, "fixed": {d: {"index": i} for d, i in indices.items()}}]})
                    item["fixed"] = fixed
                    actual = coords[item["rows"]]
                    if values is None:
                        values = actual
                    elif not np.array_equal(values, actual):
                        raise ValueError("Recorded row coordinates differ. Use Split by case for separate worksheets, "
                                         "or select shared recorded row values.")
                per_case[ident] = item
            except ValueError as exc:
                raise ValueError(f"{source['name']}: {exc}") from exc
        if axis == "case":
            fixed = {dim: selector.get("value", selector.get("mode", "Selected"))
                     for dim, selector in column["fixed"].items()}
        else:
            fixed = {"case": ident, **fixed}
        heading = column_heading(column, spec, schema, fixed)
        if heading.casefold() in {h.casefold() for h in headings}:
            raise ValueError(f"Duplicate heading '{heading}'; rename a column.")
        headings.append(heading)
        resolved.append({"quantity": name, "unit": spec["unit"], "fixed": fixed, "sources": per_case})
    if values is None or not len(values):
        raise ValueError("Choose at least one case with recorded results.")
    if len(values) >= MAX_ROWS:
        raise ValueError("Worksheet exceeds Excel's row limit; narrow the selection.")
    return {"name": sheet["name"], "axis": axis, "rows": np.arange(len(values)), "values": values,
            "headings": headings, "columns": resolved, "selection": sheet["rows"]}


def plan_project_workbook(schema, definition):
    for source in schema["sources"].values():
        if source["error"]:
            raise ValueError(f"{source['name']}: {source['error']}")
    return plan_workbook(schema, definition, information_title=INFORMATION, sheet_planner=plan_project_sheet)


def project_table_rows(schema, table, *, limit=None, column_limit=None, cancelled=lambda: False):
    columns = table["columns"][:None if column_limit is None else max(0, column_limit - 1)]
    if table["axis"] == "case":
        ids = list(table["columns"][0]["sources"])
        for i, ident in enumerate(ids[:limit]):
            if cancelled():
                raise ExportCancelled()
            dataset = schema["sources"][ident]["dataset"]
            yield [scalar(table["values"][i]), *[
                scalar(dataset[c["quantity"]].isel(c["sources"][ident]["indices"]).values.item()) for c in columns]]
    else:
        streams = []
        for column in columns:
            ident, native = next(iter(column["sources"].items()))
            streams.append(table_rows(schema["sources"][ident]["dataset"], native, limit=limit, cancelled=cancelled))
        for i, rows in enumerate(zip(*streams)):
            yield [scalar(table["values"][i]), *[row[1] for row in rows]]


def project_information_rows(schema, tables):
    yield ["Section", "Setting", "Value"]
    yield ["Export", "Coordinates", "Actual recorded coordinates only; no interpolation or averaging."]
    yield ["Export", "First / Last", "Resolved independently for each case; exact selections are listed below."]
    for ident, source in schema["sources"].items():
        yield [source["name"], "Case ID", ident]
        if source["error"]:
            yield [source["name"], "Unavailable", source["error"]]
        if "information" in source:
            rows = information_rows(source["information"], [], schema)
            next(rows)
            for row in rows:
                if not row:
                    break
                yield [source["name"], f"{row[0]}.{row[1]}", row[2]]
    yield []
    yield ["Column dictionary"]
    yield ["Worksheet", "Heading", "Quantity", "Unit", "Case", "Case ID", "Run ID", "Row axis",
           "Rows", "First coordinate", "Last coordinate", "Fixed coordinates (with units)"]
    for table in tables:
        for i, column in enumerate(table["columns"], 1):
            for ident, native in column["sources"].items():
                source = schema["sources"][ident]
                fixed = {axis_heading(source["schema"]["axes"][dim]): value for dim, value in native["fixed"].items()}
                yield [table["name"], table["headings"][i], column["quantity"], column["unit"], source["name"],
                       ident, source["information"]["Run"]["attempt_id"], axis_heading(schema["axes"][table["axis"]]),
                       1 if table["axis"] == "case" else len(table["rows"]),
                       source["name"] if table["axis"] == "case" else scalar(table["values"][0]),
                       source["name"] if table["axis"] == "case" else scalar(table["values"][-1]),
                       json.dumps(fixed, ensure_ascii=False)]


def split_by_case(schema, sheet, existing_names):
    if sheet["axis"] == "case":
        raise ValueError("Case already defines the rows. Use Split by case on a time or position worksheet.")
    groups = {}
    for column in sheet["columns"]:
        ident = column["fixed"].get("case", {}).get("value")
        if not ident:
            raise ValueError("Choose a case for each column first.")
        groups.setdefault(ident, []).append(deepcopy(column))
    if not groups:
        raise ValueError("Add columns before splitting the worksheet.")
    names, sheets = {n.casefold() for n in existing_names} | {INFORMATION.casefold(), "history"}, []
    for ident, columns in groups.items():
        label = schema["axes"]["case"].get("labels", {}).get(ident, ident)
        base = re.sub(r"[\\/*?:\[\]\x00-\x1f]", "_", f"{sheet['name']} - {label}").strip(" '")[:31] or "Case"
        name, suffix = base, 1
        while name.casefold() in names:
            suffix += 1
            tail = f" ({suffix})"
            name = base[:31 - len(tail)] + tail
        names.add(name.casefold())
        sheets.append({**deepcopy(sheet), "name": name, "columns": columns, "column_rules": []})
    return sheets


def write_project_workbook(project, definition, destination, *, cancelled=lambda: False):
    definition = results_definition(definition)
    destination = Path(destination)
    for case in project.cases:
        if destination.resolve().is_relative_to(case.run_folder.resolve()):
            raise ValueError("Save the workbook outside replaceable run folders.")
    with open_results(project, definition["case_ids"], cancelled=cancelled) as schema:
        tables = plan_project_workbook(schema, definition)
        write_tables(destination, tables, project_information_rows(schema, tables),
                     lambda table: project_table_rows(schema, table, cancelled=cancelled),
                     information_title=INFORMATION, cancelled=cancelled)
