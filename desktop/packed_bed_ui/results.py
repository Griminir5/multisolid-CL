"""Project Results uses the case report editor, adding case selection and splitting."""

from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
)

from .case_list import result_label
from .editor_widgets import action_button
from .report import ColumnsDialog, ExportDialog, ReportPage, TemplateDialog
from .project import read_json, write_json
from .project_results import (
    INFORMATION, open_results, plan_project_sheet, plan_project_workbook,
    project_columns, project_information_rows, project_table_rows,
    results_definition, results_template, apply_results_template, split_by_case, write_project_workbook,
)


class CaseSelectionDialog(QDialog):
    """Choose successful retained runs without changing the report on Cancel."""

    def __init__(self, project, chosen, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select cases for results")
        self.resize(780, 480)
        layout = QVBoxLayout(self)
        search = QLineEdit()
        search.setPlaceholderText("Filter cases or study factors…")
        layout.addWidget(search)
        note = QLabel("Select successful runs to make available in Results. "
                      "For a failed run's partial data, open that case's Report tab.")
        note.setWordWrap(True)
        layout.addWidget(note)
        tree = QTreeWidget()
        tree.setHeaderLabels(["Case / study", "Latest run"])
        tree.setColumnWidth(0, 380)
        layout.addWidget(tree)
        groups, self.items = {}, {}
        studies = {s["id"]: s["name"] for s in project.metadata.get("studies", [])}
        for case in project.cases:
            study_id = case.metadata.get("study_id")
            parent_item = tree
            if study_id:
                if study_id not in groups:
                    group = QTreeWidgetItem(tree, [studies.get(study_id, case.metadata.get("origin", "Study"))])
                    group.setFlags(group.flags() | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsUserCheckable)
                    groups[study_id] = group
                parent_item = groups[study_id]
            try:
                state = read_json(case.run_folder / "status.json")
            except (OSError, ValueError):
                state = {"state": "not_run"}
            eligible = (state.get("state") == "completed"
                        and (case.run_folder / "output/results.nc").is_file()
                        and (case.run_folder / "snapshot.json").is_file())
            label = result_label(state)
            if state.get("state") == "completed" and not eligible:
                label += " — results unavailable"
            item = QTreeWidgetItem(parent_item, [case.name, label])
            item.setToolTip(0, json.dumps(case.metadata.get("selections", {}), ensure_ascii=False))
            if eligible:
                item.setCheckState(0, Qt.CheckState.Checked if case.id in chosen else Qt.CheckState.Unchecked)
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                item.setDisabled(True)
            self.items[case.id] = item
        tree.expandAll()
        def filter_cases(text):
            text = text.casefold()
            for item in self.items.values():
                group = item.parent().text(0) if item.parent() else ""
                item.setHidden(text not in f"{group} {item.text(0)} {item.toolTip(0)}".casefold())
            for group in groups.values():
                group.setHidden(all(group.child(i).isHidden() for i in range(group.childCount())))
        search.textChanged.connect(filter_cases)
        actions = QHBoxLayout()
        def check_all(state):
            for item in self.items.values():
                if not item.isDisabled():
                    item.setCheckState(0, state)
        actions.addWidget(action_button("Select all", lambda: check_all(Qt.CheckState.Checked)))
        actions.addWidget(action_button("Clear all", lambda: check_all(Qt.CheckState.Unchecked)))
        layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        def update_selection():
            buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(self.selected_cases()))
        tree.itemChanged.connect(update_selection)
        update_selection()

    def selected_cases(self):
        return [ident for ident, item in self.items.items()
                if not item.isDisabled() and item.checkState(0) == Qt.CheckState.Checked]


class ProjectColumnsDialog(ColumnsDialog):
    def load_quantity(self):
        super().load_quantity()
        if self.axis == "case" and "time" in self.pickers:
            self.pickers["time"].set_selection({"mode": "last"})
            self.pickers["time"].update_state()

    def update_count(self):
        self.rule, self.added_columns = None, []
        try:
            if self.quantity.currentIndex() < 0:
                raise ValueError("Select cases with recorded results first.")
            self.added_columns = project_columns(self.schema, self.quantity.currentData(),
                                                {dim: p.selection() for dim, p in self.pickers.items()})
            self.count = len(self.added_columns)
            message = "Narrow the selection to fit Excel's column limit." if self.count > self.capacity else ""
        except (ValueError, KeyError, TypeError) as exc:
            self.count, message = 0, str(exc)
        button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        button.setText(f"Add {self.count:,} columns")
        button.setEnabled(0 < self.count <= self.capacity)
        self.error.setText(message)

    def columns(self):
        return self.added_columns


class ResultsPage(ReportPage):
    back = pyqtSignal()
    information_title = INFORMATION
    plan_sheet = staticmethod(plan_project_sheet)
    columns_dialog = ProjectColumnsDialog

    def __init__(self):
        self.project = None
        super().__init__()
        controls = QHBoxLayout()
        controls.addWidget(action_button("← Back to project", self.back.emit))
        controls.addWidget(action_button("Select cases…", self.select_cases))
        self.selection_note = QLabel("Select cases to build a workbook.")
        self.selection_note.setWordWrap(True)
        controls.addWidget(self.selection_note, 1)
        controls.addWidget(action_button("Refresh results", self.refresh_source))
        self.layout().insertLayout(0, controls)
        hint = QLabel("Choose the row axis, then add columns. Case can be an axis just like time or position. "
                      "First/Last uses each case's own samples. Calculate averages and plot in Excel.")
        hint.setWordWrap(True)
        self.layout().insertWidget(1, hint)
        self.split_button = action_button("Split by case", self.split_sheet)
        self.split_button.setToolTip("Replace this worksheet with one worksheet per case, preserving each recorded grid.")
        self.column_actions.append(self.split_button)
        self.columns.parentWidget().layout().addWidget(self.split_button)

    def clear(self):
        self.debounce.stop()
        self.preview_timer.stop()
        self.project, self.dirty = None, False
        self.schema = {"axes": {}, "quantities": {}}

    def set_project(self, project):
        if not self.save():
            return False
        self.clear()
        self.project = project
        self.definition = deepcopy(project.metadata.get("results_report", {"version": 1, "case_ids": [], "sheets": []}))
        self.setEnabled(True)
        self.save_note.clear()
        self.supported = True
        try:
            self.definition = results_definition(self.definition)
        except ValueError as exc:
            self.supported = False
            self.error.setText(f"{exc} The saved results report has been preserved.")
            self.setEnabled(False)
            return True
        self.refresh_source()
        if not self.definition["sheets"]:
            self.add_sheet(save=False)
        self.rebuild_sheets(1)
        return True

    def changed(self):
        if self.loading or self.project is None:
            return
        self.dirty = True
        self.save_note.setText("Saving results report…")
        try:
            self.project.drafts.write("results", "project", self.definition)
        except (OSError, ValueError) as exc:
            self.save_note.setText(f"Recovery checkpoint could not be saved: {exc}")
        self.debounce.start()
        self.preview_timer.start()

    def save(self):
        self.debounce.stop()
        if self.project is None or not self.dirty:
            return True
        previous = self.project.metadata.get("results_report")
        self.project.metadata["results_report"] = deepcopy(self.definition)
        try:
            self.project.save()
        except (OSError, ValueError) as exc:
            if previous is None:
                self.project.metadata.pop("results_report", None)
            else:
                self.project.metadata["results_report"] = previous
            self.save_note.setText(f"Results report could not be saved: {exc}")
            return False
        self.dirty = False
        self.save_note.setText("Results report saved")
        try:
            self.project.drafts.clear("results", "project")
        except OSError:
            pass  # Recovery ignores a checkpoint identical to saved metadata.
        return True

    def refresh_source(self):
        if self.project is None or not self.supported:
            return
        with open_results(self.project, self.definition["case_ids"]) as schema:
            self.schema = schema
        for source in self.schema["sources"].values():
            source.pop("dataset", None)
        notes, warnings = [], []
        for source in self.schema["sources"].values():
            info = source.get("information", {})
            state = info.get("Status", {})
            state = dict(state) if isinstance(state, dict) else {"state": "Unavailable"}
            state["stale"] = info.get("Export", {}).get("inputs_changed") is True
            label = source["error"] or result_label(state)
            notes.append(f"{source['name']}: {label}")
            if source["error"] or info.get("Export", {}).get("inputs_changed") is not False or state.get("state") != "completed":
                warnings.append(notes[-1])
        self.selection_note.setText(f"{len(notes)} cases selected" if notes else "Select cases to build a workbook.")
        self.selection_note.setToolTip("\n".join(notes))
        self.axis_notice.setText("\n".join(warnings))
        self.axis_notice.setVisible(bool(warnings))
        self.select_sheet()

    def select_cases(self):
        if self.project is None:
            return
        dialog = CaseSelectionDialog(self.project, self.definition["case_ids"], self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_cases():
            self.set_selected_cases(dialog.selected_cases())

    def set_selected_cases(self, case_ids):
        self.definition["case_ids"] = list(case_ids)
        self.refresh_source()
        if self.sheet and not self.sheet["columns"] and self.sheet["axis"] == "case" and "time" in self.schema["axes"]:
            self.sheet["axis"] = "time"
            self.select_sheet()
        self.changed()

    def save_template(self):
        if not self.save():
            return
        try:
            template = results_template(self.definition)
            path, _ = QFileDialog.getSaveFileName(self, "Save results template", "Results template.json",
                                                 "Results template (*.json)")
            if path:
                destination = Path(path)
                if not destination.suffix:
                    destination = destination.with_suffix(".json")
                write_json(destination, template)
                self.save_note.setText(f"Template saved: {destination.name}")
        except (OSError, ValueError) as exc:
            self.error.setText(f"Template could not be saved: {exc}")

    def apply_template(self):
        if not self.save():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Apply results template", "", "Results template (*.json)")
        if not path:
            return
        try:
            template = read_json(Path(path))
            definition = apply_results_template(template, self.definition["case_ids"])
            self.refresh_source()
            mapping = "\n".join(f"Case {i + 1}: {self.schema['axes']['case']['labels'][ident]}"
                                for i, ident in enumerate(definition["case_ids"]))
            note = "Template case positions use the selected cases in this order:\n" + mapping
            dialog = TemplateDialog(self.schema, definition, parent=self, sheet_planner=plan_project_sheet,
                                    workbook_planner=plan_project_workbook, note=note)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            self.definition = definition
            if not definition["sheets"]:
                self.add_sheet(save=False)
            self.rebuild_sheets(1)
            self.changed()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.error.setText(f"Template could not be applied: {exc}")

    def column_identity(self, column):
        return json.dumps([column["quantity"], column["fixed"]], sort_keys=True)

    def row_axis_description(self, sheet):
        self.different_row_grids = False
        if sheet["axis"] == "case" or not sheet["columns"]:
            return super().row_axis_description(sheet)
        ids = dict.fromkeys(c["fixed"].get("case", {}).get("value") for c in sheet["columns"])
        axes = [self.schema["sources"].get(ident, {}).get("schema", {}).get("axes", {}).get(sheet["axis"])
                for ident in ids]
        axes = [axis for axis in axes if axis is not None and axis["values"] is not None]
        if not axes:
            return super().row_axis_description(sheet)
        self.different_row_grids = any(not np.array_equal(axes[0]["values"], axis["values"]) for axis in axes[1:])
        values = axes[0]["values"]
        for axis in axes[1:]:
            values = values[np.isin(values, axis["values"])]
        return {**axes[0], "values": values, "error": "No shared coordinates. Use Split by case."}

    def select_sheet(self):
        super().select_sheet()
        self.update_row_note()

    def change_rows(self):
        super().change_rows()
        self.update_row_note()

    def update_row_note(self):
        if self.sheet is not None and getattr(self, "different_row_grids", False):
            if self.rows.selection()["mode"] in ("all", "first", "last"):
                self.rows.note.setText("Per case")
                self.rows.note.setToolTip("Grids differ. Select shared recorded values or use Split by case.")

    def split_sheet(self):
        if self.sheet is None:
            return
        try:
            index = self.sheets.currentRow() - 1
            sheets = split_by_case(self.schema, self.sheet, [s["name"] for s in self.definition["sheets"]])
            self.definition["sheets"][index:index + 1] = sheets
            self.rebuild_sheets(index + 1)
            self.changed()
        except (ValueError, KeyError, TypeError) as exc:
            self.error.setText(str(exc))

    def preview(self):
        if self.project is None or self.loading or not self.supported:
            return
        self.error.clear()
        self.general_link.hide()
        self.counts.clear()
        self.show_table([], [])
        self.export_button.setEnabled(False)
        try:
            if not self.definition["case_ids"]:
                raise ValueError("Select cases with recorded results, then add columns to a worksheet.")
            with open_results(self.project, self.definition["case_ids"]) as schema:
                if self.sheet is not None:
                    table = plan_project_sheet(schema, self.sheet)
                    self.show_table(table["headings"], project_table_rows(schema, table, limit=20, column_limit=12),
                                    schema["axes"][table["axis"]].get("precision", 15))
                    self.counts.setText(f"Recorded results · {len(table['rows']):,} rows × {len(table['headings']):,} columns "
                                        "· preview up to 20 × 12 · export contains the full selection")
                else:
                    tables = []
                    for sheet in self.definition["sheets"]:
                        try:
                            tables.append(plan_project_sheet(schema, sheet))
                        except (ValueError, KeyError, TypeError):
                            pass
                    rows = iter(project_information_rows(schema, tables))
                    if self.info_view.currentIndex() == 1:
                        for row in rows:
                            if row == ["Column dictionary"]:
                                break
                    self.show_table(next(rows), rows)
                    self.counts.setText("Preview of the first 20 rows · retained run inputs and exact column selections")
                tables = plan_project_workbook(schema, self.definition)
                self.export_button.setEnabled(True)
                self.export_button.setToolTip("\n".join(
                    f"{t['name']}: {len(t['rows']):,} rows × {len(t['headings']):,} columns" for t in tables))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self.error.setText(str(exc))

    def export(self):
        if not self.save():
            return
        self.preview()
        if not self.export_button.isEnabled():
            return
        name = re.sub(r'[\\/:*?"<>|]', "_", self.project.metadata["name"]).strip(". ")[:80] or "Results"
        chooser = QFileDialog(self, "Export project results", str(self.project.root / f"{name} results.xlsx"),
                             "Excel workbook (*.xlsx)")
        chooser.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        chooser.setDefaultSuffix("xlsx")
        if chooser.exec() != QDialog.DialogCode.Accepted:
            return
        path = chooser.selectedFiles()[0]
        dialog = ExportDialog({"project": self.project, "definition": deepcopy(self.definition), "destination": path},
                              self, writer=write_project_workbook)
        dialog.exec()
        dialog.worker.wait()
        if dialog.worker.error:
            QMessageBox.warning(self, "Export failed", dialog.worker.error)
        else:
            self.save_note.setText("Export cancelled" if dialog.worker.cancelled else f"Exported {path}")
        dialog.deleteLater()
