"""Case report authoring using ordinary Qt widgets and the shared table selector."""

from copy import deepcopy
from contextlib import nullcontext
from itertools import islice
import json
from math import prod
from pathlib import Path
import re
from uuid import uuid4

import xarray as xr
from PyQt6.QtCore import QAbstractListModel, QModelIndex, QItemSelectionModel, QSortFilterProxyModel, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListView, QListWidget, QMessageBox,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from packed_bed.report_schema import describe_dataset, describe_inputs
from packed_bed.reports import RESULTS_FILENAME

from .editor_widgets import action_button
from .project import read_json, write_json
from .workbook import (
    INFORMATION, MAX_COLUMNS, ExportCancelled, axis_heading,
    coordinate_index, expand_columns, information_rows, plan_sheet,
    plan_workbook, refresh_report, report_definition,
    resolve_selector, rule_columns, run_information, scalar, table_rows, text_value, validate_information,
    write_workbook,
)


class CoordinateModel(QAbstractListModel):
    """Render coordinate labels lazily, even for long reporting schedules."""
    def __init__(self, values, parent=None, precision=15):
        super().__init__(parent)
        self.values = values
        self.precision = precision

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.values)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return text_value(self.values[index.row()], self.precision)


class CoordinatePicker(QWidget):
    changed = pyqtSignal()

    def __init__(self, axis, selection=None, parent=None):
        super().__init__(parent)
        self.axis = axis
        self.selected = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.mode = QComboBox()
        for label, mode in (("All", "all"), ("First", "first"), ("Last", "last"), ("Selected values", "values")):
            self.mode.addItem(label, mode)
        self.choose = action_button("Choose values…", self.choose_values)
        self.note = QLabel()
        layout.addWidget(self.mode)
        layout.addWidget(self.choose)
        layout.addWidget(self.note, 1)
        self.set_selection(selection or {"mode": "all"})
        self.mode.currentIndexChanged.connect(self.update_state)
        self.update_state()

    def set_selection(self, selection):
        self.mode.blockSignals(True)
        index = self.mode.findData(selection.get("mode"))
        self.mode.setCurrentIndex(index if index >= 0 else 3)
        self.selected = deepcopy(selection.get("values", []))
        self.mode.blockSignals(False)

    def selection(self):
        mode = self.mode.currentData()
        return {"mode": mode, **({"values": self.selected} if mode == "values" else {})}

    def options(self):
        values = self.axis["values"]
        if values is None or not len(values):
            return []
        mode = self.mode.currentData()
        if mode in ("first", "last"):
            return [{"mode": mode}]
        return [{"value": scalar(v)} for v in (values if mode == "all" else self.selected)]

    def count(self):
        values = self.axis["values"]
        if values is None or not len(values):
            return 0
        mode = self.mode.currentData()
        return 1 if mode in ("first", "last") else len(values) if mode == "all" else len(self.selected)

    def update_state(self):
        values, mode = self.axis["values"], self.mode.currentData()
        self.choose.setEnabled(values is not None and len(values) > 0)
        if values is None or not len(values):
            self.note.setText(self.axis["error"] or "No coordinates available")
        elif mode in ("first", "last"):
            value = text_value(values[0 if mode == 'first' else -1], self.axis.get("precision", 15))
            self.note.setText(f"{value} {self.axis['unit']}".strip())
        else:
            self.note.setText(f"{len(values) if mode == 'all' else len(self.selected)} values")
        self.changed.emit()

    def choose_values(self):
        values = self.axis["values"]
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Choose {axis_heading(self.axis)}")
        dialog.resize(470, 500)
        layout = QVBoxLayout(dialog)
        search = QLineEdit()
        search.setPlaceholderText("Filter coordinates…")
        layout.addWidget(search)
        model = CoordinateModel(values, dialog, self.axis.get("precision", 15))
        proxy = QSortFilterProxyModel(dialog)
        proxy.setSourceModel(model)
        proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        view = QListView()
        view.setModel(proxy)
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(view)
        layout.addWidget(QLabel("Use Ctrl or Shift to select several values."))
        chosen, filtering = set(), False
        def remember(selected, deselected):
            if not filtering:
                chosen.difference_update(proxy.mapToSource(i).row() for i in deselected.indexes())
                chosen.update(proxy.mapToSource(i).row() for i in selected.indexes())
        def filter_values(text):
            nonlocal filtering
            filtering = True
            proxy.setFilterFixedString(text)
            for index in chosen:
                visible = proxy.mapFromSource(model.index(index))
                if visible.isValid():
                    view.selectionModel().select(visible, QItemSelectionModel.SelectionFlag.Select)
            filtering = False
        view.selectionModel().selectionChanged.connect(remember)
        search.textChanged.connect(filter_values)
        for value in self.selected:
            try:
                index = coordinate_index(values, value, self.axis["label"])
                view.selectionModel().select(proxy.mapFromSource(model.index(index)), QItemSelectionModel.SelectionFlag.Select)
            except ValueError:
                pass
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.selected = [scalar(values[i]) for i in sorted(chosen)]
            self.mode.setCurrentIndex(self.mode.findData("values"))
            self.update_state()


class ColumnsDialog(QDialog):
    def __init__(self, schema, axis, capacity, parent=None):
        super().__init__(parent)
        self.schema, self.axis, self.capacity = schema, axis, capacity
        self.setWindowTitle("Add columns")
        self.resize(760, 400)
        layout = QVBoxLayout(self)
        self.quantity = QComboBox()
        for name, spec in schema["quantities"].items():
            self.quantity.addItem(f"{spec['label']} ({spec['unit'] or 'unit unspecified'})", name)
            if axis not in spec["dimensions"]:
                item = self.quantity.model().item(self.quantity.count() - 1)
                item.setEnabled(False)
                item.setToolTip(f"Does not have the {axis} axis.")
        layout.addWidget(QLabel(f"Rows represent: {axis_heading(schema['axes'][axis])}"))
        layout.addWidget(self.quantity)
        self.fields = QWidget()
        self.form = QFormLayout(self.fields)
        layout.addWidget(self.fields)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.quantity.currentIndexChanged.connect(self.load_quantity)
        index = next((i for i in range(self.quantity.count()) if self.quantity.model().item(i).isEnabled()), -1)
        self.quantity.setCurrentIndex(index)
        self.load_quantity()

    def load_quantity(self):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.pickers = {}
        spec = self.schema["quantities"].get(self.quantity.currentData())
        if spec and self.axis in spec["dimensions"]:
            for dim in spec["dimensions"]:
                if dim != self.axis:
                    picker = CoordinatePicker(self.schema["axes"][dim])
                    self.pickers[dim] = picker
                    self.form.addRow(axis_heading(self.schema["axes"][dim]), picker)
                    picker.changed.connect(self.update_count)
        self.update_count()

    def update_count(self):
        self.count = prod(p.count() for p in self.pickers.values()) if self.quantity.currentIndex() >= 0 else 0
        button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        button.setText(f"Add {self.count:,} columns")
        button.setEnabled(0 < self.count <= self.capacity)
        self.error.setText("Narrow the selection to fit Excel's column limit." if self.count > self.capacity else "")

    def columns(self):
        selections = {d: p.selection() for d, p in self.pickers.items()}
        self.rule = None
        if any(s["mode"] == "all" for s in selections.values()):
            self.rule = {"id": uuid4().hex, "quantity": self.quantity.currentData(), "selections": selections}
            return rule_columns(self.schema, self.rule)
        return expand_columns(self.quantity.currentData(), {d: p.options() for d, p in self.pickers.items()})


class ExportWorker(QThread):
    def __init__(self, arguments, parent=None):
        super().__init__(parent)
        self.arguments, self.error, self.cancelled = arguments, "", False
        self.cancel_requested = False

    def run(self):
        try:
            write_workbook(**self.arguments, cancelled=lambda: self.cancel_requested or self.isInterruptionRequested())
        except ExportCancelled:
            self.cancelled = True
        except Exception as exc:
            self.error = str(exc)


class ExportDialog(QDialog):
    def __init__(self, arguments, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export workbook")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        layout = QVBoxLayout(self)
        self.message = QLabel("Writing workbook…")
        self.cancel = action_button("Cancel", self.reject)
        layout.addWidget(self.message)
        layout.addWidget(self.cancel)
        self.worker = ExportWorker(arguments, self)
        self.worker.finished.connect(self.accept)
        QTimer.singleShot(0, self.worker.start)

    def reject(self):
        self.worker.cancel_requested = True
        self.worker.requestInterruption()
        self.message.setText("Cancelling…")
        self.cancel.setEnabled(False)

    def closeEvent(self, event):
        self.reject()
        event.ignore()


class TemplateDialog(QDialog):
    """Review a portable layout in this case before replacing the current report."""
    def __init__(self, schema, definition, removed=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Apply report template")
        self.resize(850, 400)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Applying this template replaces the current report configuration."))
        if removed:
            layout.addWidget(QLabel(f"{removed} row selections or columns have coordinates unavailable in this case and will be removed."))
        table = QTableWidget(len(definition["sheets"]), 4)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setHorizontalHeaderLabels(["Worksheet", "Rows represent", "Size", "Validation"])
        names = set()
        for row, sheet in enumerate(definition["sheets"]):
            size, message = "", "Ready"
            try:
                planned = plan_sheet(schema, sheet)
                size = f"{len(planned['rows'])} rows × {len(planned['headings'])} columns"
                plan_workbook(schema, {"version": 1, "sheets": [sheet]})
                if sheet["name"].casefold() in names:
                    raise ValueError("Use a unique worksheet name.")
            except (ValueError, KeyError, TypeError) as exc:
                message = str(exc)
            names.add(sheet["name"].casefold())
            label = axis_heading(schema["axes"][sheet["axis"]]) if sheet["axis"] in schema["axes"] else sheet["axis"]
            for col, value in enumerate((sheet["name"], label, size, message)):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                table.setItem(row, col, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ReportPage(QWidget):
    def __init__(self, editor):
        super().__init__()
        self.editor, self.case = editor, None
        self.definition = {"version": 1, "sheets": []}
        self.schema = {"axes": {}, "quantities": {}}
        self.dirty, self.loading, self.needs_refresh = False, False, True
        self.supported = True
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(350)
        self.debounce.timeout.connect(self.save)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(120)
        self.preview_timer.timeout.connect(self.preview)
        layout = QVBoxLayout(self)
        self.axis_notice = QLabel()
        self.axis_notice.setWordWrap(True)
        self.axis_notice.hide()
        layout.addWidget(self.axis_notice)
        split = QSplitter()
        left, right = QWidget(), QWidget()
        ll, rl = QVBoxLayout(left), QVBoxLayout(right)
        ll.setContentsMargins(0, 0, 0, 0)
        rl.setContentsMargins(0, 0, 0, 0)
        self.sheets = QListWidget()
        self.sheets.setAccessibleName("Worksheets")
        ll.addWidget(self.sheets, 1)
        self.sheet_actions = self.actions(ll, [
            ("Add", self.add_sheet), ("Remove", self.remove_sheet),
            ("↑", lambda: self.move_sheet(-1)), ("↓", lambda: self.move_sheet(1)),
        ])
        self.config = QWidget()
        form = QFormLayout(self.config)
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.name, self.axis = QLineEdit(), QComboBox()
        self.name.setAccessibleName("Worksheet name")
        self.axis.setAccessibleName("Rows represent")
        form.addRow("Name", self.name)
        form.addRow("Rows represent", self.axis)
        self.row_holder = QWidget()
        self.row_layout = QVBoxLayout(self.row_holder)
        self.row_layout.setContentsMargins(0, 0, 0, 0)
        form.addRow("Row values", self.row_holder)
        ll.addWidget(self.config)
        self.columns = QListWidget()
        self.columns.setAccessibleName("Export columns")
        ll.addWidget(self.columns, 2)
        self.column_actions = self.actions(ll, [
            ("Add columns…", self.add_columns), ("Remove", self.remove_columns),
            ("Rename", self.rename_column), ("↑", lambda: self.move_column(-1)),
            ("↓", lambda: self.move_column(1)),
        ])
        self.columns.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.info_view = QComboBox()
        self.info_view.addItems(["Case information", "Column dictionary"])
        rl.addWidget(self.info_view)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(170)
        self.table.setAccessibleName("Report preview")
        rl.addWidget(self.table, 1)
        self.counts, self.error = QLabel(), QLabel()
        self.counts.setWordWrap(True)
        self.error.setWordWrap(True)
        rl.addWidget(self.counts)
        rl.addWidget(self.error)
        self.general_link = action_button("Go to General", lambda: self.editor.tabs.setCurrentWidget(self.editor.general))
        rl.addWidget(self.general_link)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 3)
        split.setSizes([380, 800])
        layout.addWidget(split, 1)
        self.save_note = QLabel()
        bottom = QHBoxLayout()
        bottom.addWidget(self.save_note, 1)
        bottom.addWidget(action_button("Save as template…", self.save_template))
        bottom.addWidget(action_button("Apply template…", self.apply_template))
        self.export_button = action_button("Export workbook…", self.export)
        bottom.addWidget(self.export_button)
        layout.addLayout(bottom)
        self.sheets.currentRowChanged.connect(self.select_sheet)
        self.name.textEdited.connect(self.rename_sheet)
        self.axis.currentIndexChanged.connect(self.change_axis)
        self.info_view.currentIndexChanged.connect(self.preview)
        self.editor.tabs.currentChanged.connect(self.tab_changed)

    def actions(self, layout, entries):
        row = QHBoxLayout()
        buttons = []
        for label, callback in entries:
            button = action_button(label, callback)
            if label in ("↑", "↓"):
                button.setFixedWidth(button.sizeHint().height())
            row.addWidget(button)
            buttons.append(button)
        layout.addLayout(row)
        return buttons

    @property
    def sheet(self):
        if not self.supported:
            return None
        index = self.sheets.currentRow() - 1
        sheets = self.definition.get("sheets", [])
        return sheets[index] if 0 <= index < len(sheets) else None

    def set_case(self, case):
        self.debounce.stop()
        self.preview_timer.stop()
        self.case, self.dirty = case, False
        self.definition = deepcopy(case.metadata.get("report", {"version": 1, "sheets": []}))
        self.save_note.clear()
        self.axis_notice.clear()
        self.axis_notice.hide()
        self.setEnabled(True)
        try:
            report_definition(self.definition)
            self.supported = True
        except ValueError:
            self.supported = False
        if not self.supported:
            self.sheets.blockSignals(True)
            self.sheets.clear()
            self.sheets.blockSignals(False)
            self.columns.clear()
            self.config.hide()
            self.show_table([], [])
            self.counts.clear()
            self.error.setText("Unsupported or malformed report definition; it has been preserved.")
            self.setEnabled(False)
            return
        self.refresh_source()
        if not self.definition["sheets"]:
            self.add_sheet(save=False)
        self.rebuild_sheets(1)

    def tab_changed(self):
        if self.editor.tabs.currentWidget() is self and self.needs_refresh:
            self.refresh_source()

    def invalidate(self):
        self.needs_refresh = True
        if self.editor.tabs.currentWidget() is self:
            self.refresh_source()

    def refresh_source(self):
        if self.case is None or not self.supported:
            return
        self.needs_refresh = False
        try:
            self.schema = describe_inputs(self.case.documents)
            previous = deepcopy(self.definition)
            removed = refresh_report(self.schema, self.definition)
            if removed:
                self.axis_notice.setText(f"Removed {removed} row selections or columns whose coordinates no longer exist in the case inputs.")
                self.axis_notice.show()
            if self.definition != previous:
                self.changed()
        except (OSError, ValueError, TypeError) as exc:
            self.schema = {"axes": {}, "quantities": {}}
            self.select_sheet()
            self.error.setText(f"Cannot read case inputs: {exc}")
            self.export_button.setEnabled(False)
            return False
        self.select_sheet()
        return True

    def rebuild_sheets(self, index):
        self.sheets.blockSignals(True)
        self.sheets.clear()
        self.sheets.addItems([INFORMATION, *[s.get("name", "") for s in self.definition["sheets"]]])
        self.sheets.setCurrentRow(index)
        self.sheets.blockSignals(False)
        self.select_sheet()

    def select_sheet(self):
        self.loading = True
        sheet = self.sheet
        self.config.setVisible(sheet is not None)
        self.columns.setVisible(sheet is not None)
        self.info_view.setVisible(sheet is None)
        for button in self.column_actions:
            button.setVisible(sheet is not None)
        for button in self.sheet_actions[1:]:
            button.setEnabled(sheet is not None)
        self.columns.clear()
        if sheet is not None:
            self.name.setText(sheet["name"])
            self.axis.clear()
            for dim, description in self.schema["axes"].items():
                self.axis.addItem(axis_heading(description), dim)
            index = self.axis.findData(sheet["axis"])
            if index < 0:
                self.axis.addItem(f"{sheet['axis']} — unavailable", sheet["axis"])
                index = self.axis.count() - 1
            self.axis.setCurrentIndex(index)
            self.axis.setEnabled(not sheet["columns"])
            self.axis.setToolTip("Remove the columns before changing the row axis." if sheet["columns"] else "")
            while self.row_layout.count():
                old = self.row_layout.takeAt(0).widget()
                old.hide()
                old.deleteLater()
            axis = self.schema["axes"].get(sheet["axis"], {"label": sheet["axis"], "unit": "", "values": None, "error": "Axis unavailable"})
            self.rows = CoordinatePicker(axis, sheet["rows"])
            self.row_layout.addWidget(self.rows)
            self.rows.changed.connect(self.change_rows)
            for column in sheet["columns"]:
                try:
                    one = {**sheet, "rows": {"mode": "first"}, "columns": [column], "column_rules": []}
                    label = plan_sheet(self.schema, one)["headings"][1]
                except (ValueError, KeyError, TypeError) as exc:
                    label = f"{column.get('label', column.get('quantity', 'Column'))} — {exc}"
                self.columns.addItem(label)
                self.columns.item(self.columns.count() - 1).setToolTip(label)
        self.loading = False
        self.preview()

    def changed(self):
        if self.loading or self.case is None:
            return
        self.dirty = True
        self.save_note.setText("Saving report…")
        self.debounce.start()
        self.preview_timer.start()
        self.editor.checkpoint()

    def save(self):
        self.debounce.stop()
        if not self.dirty or self.case is None:
            return True
        if self.editor.dirty:
            return self.editor.save()  # Commit scientific inputs and their metadata together first.
        previous = self.case.metadata.get("report")
        self.case.metadata["report"] = deepcopy(self.definition)
        try:
            self.case.project.save()
        except (OSError, ValueError) as exc:
            if previous is None:
                self.case.metadata.pop("report", None)
            else:
                self.case.metadata["report"] = previous
            self.save_note.setText(f"Report could not be saved: {exc}")
            return False
        self.dirty = False
        self.save_note.setText("Report saved")
        self.editor.checkpoint()
        return True

    def add_sheet(self, *, save=True):
        names = {s["name"].casefold() for s in self.definition["sheets"]}
        i = 1
        while f"sheet {i}" in names:
            i += 1
        axis = "time" if "time" in self.schema["axes"] else next(iter(self.schema["axes"]), "time")
        self.definition["sheets"].append({"name": f"Sheet {i}", "axis": axis, "rows": {"mode": "all"}, "columns": []})
        self.rebuild_sheets(len(self.definition["sheets"]))
        if save:
            self.changed()

    def remove_sheet(self):
        if self.sheet is not None:
            index = self.sheets.currentRow()
            self.definition["sheets"].pop(index - 1)
            self.rebuild_sheets(min(index, len(self.definition["sheets"])))
            self.changed()

    def move_sheet(self, offset):
        index = self.sheets.currentRow() - 1
        sheets = self.definition["sheets"]
        if 0 <= index < len(sheets) and 0 <= index + offset < len(sheets):
            sheets[index], sheets[index + offset] = sheets[index + offset], sheets[index]
            self.rebuild_sheets(index + offset + 1)
            self.changed()

    def rename_sheet(self, name):
        if not self.loading and self.sheet is not None:
            self.sheet["name"] = name
            self.sheets.currentItem().setText(name)
            self.changed()

    def change_axis(self):
        if not self.loading and self.sheet is not None and not self.sheet["columns"]:
            self.sheet["axis"] = self.axis.currentData()
            self.sheet["rows"] = {"mode": "all"}
            self.select_sheet()
            self.changed()

    def change_rows(self):
        if not self.loading and self.sheet is not None:
            self.sheet["rows"] = self.rows.selection()
            self.changed()

    def add_columns(self):
        sheet = self.sheet
        if sheet is None or sheet["axis"] not in self.schema["axes"]:
            return
        dialog = ColumnsDialog(self.schema, sheet["axis"], MAX_COLUMNS - 1 - len(sheet["columns"]), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            def identity(column):
                try:
                    fixed = {}
                    for dim, selector in column["fixed"].items():
                        values = self.schema["axes"][dim]["values"]
                        fixed[dim] = scalar(values[resolve_selector(values, selector, dim)[0]])
                except (ValueError, KeyError, TypeError):
                    fixed = column["fixed"]  # Unresolved columns remain editable.
                return json.dumps([column["quantity"], fixed], sort_keys=True)
            existing = {identity(c) for c in sheet["columns"]}
            added = False
            for column in dialog.columns():
                key = identity(column)
                if key not in existing:
                    sheet["columns"].append(column)
                    existing.add(key)
                    added = True
                elif dialog.rule:
                    dialog.rule.setdefault("excluded", []).append(column["fixed"])
            if added and dialog.rule:
                sheet.setdefault("column_rules", []).append(dialog.rule)
            self.select_sheet()
            self.changed()

    def remove_columns(self):
        if self.sheet is not None:
            for index in sorted((self.columns.row(item) for item in self.columns.selectedItems()), reverse=True):
                column = self.sheet["columns"].pop(index)
                for rule in self.sheet.get("column_rules", []):
                    if column.get("rule") == rule["id"]:
                        rule.setdefault("excluded", []).append(column["fixed"])
            if "column_rules" in self.sheet:
                self.sheet["column_rules"] = [rule for rule in self.sheet["column_rules"]
                                             if any(c.get("rule") == rule["id"] for c in self.sheet["columns"])]
            self.select_sheet()
            self.changed()

    def rename_column(self):
        index = self.columns.currentRow()
        if self.sheet is not None and index >= 0:
            column = self.sheet["columns"][index]
            label, ok = QInputDialog.getText(self, "Column label", "Label (leave empty for the automatic heading)",
                                             text=column.get("label", ""))
            if ok:
                column.pop("label", None)
                if label.strip():
                    column["label"] = label.strip()
                self.select_sheet()
                self.columns.setCurrentRow(index)
                self.changed()

    def move_column(self, offset):
        index = self.columns.currentRow()
        if self.sheet is not None and 0 <= index + offset < len(self.sheet["columns"]) and index >= 0:
            columns = self.sheet["columns"]
            columns[index], columns[index + offset] = columns[index + offset], columns[index]
            self.select_sheet()
            self.columns.setCurrentRow(index + offset)
            self.changed()

    def show_table(self, headers, rows, coordinate_precision=15):
        self.table.clear()
        self.table.setColumnCount(min(12, len(headers)))
        self.table.setHorizontalHeaderLabels([h.replace(" | ", "\n") for h in headers[:12]])
        for j in range(self.table.columnCount()):
            self.table.horizontalHeaderItem(j).setToolTip(headers[j])
        rows = list(islice(rows, 20))
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row[:12]):
                item = QTableWidgetItem("" if value is None else text_value(value, coordinate_precision if j == 0 else 12))
                item.setToolTip("" if value is None else str(value))
                self.table.setItem(i, j, item)
        for j in range(self.table.columnCount()):
            self.table.setColumnWidth(j, 170)

    def save_template(self):
        if not self.editor.save():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save report template", "Report template.json", "Report template (*.json)")
        if path:
            try:
                destination = Path(path)
                if not destination.suffix:
                    destination = destination.with_suffix(".json")
                write_json(destination, report_definition(self.definition))
                self.save_note.setText(f"Template saved: {destination.name}")
            except (OSError, ValueError) as exc:
                self.error.setText(f"Template could not be saved: {exc}")

    def apply_template(self):
        if not self.editor.save():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Apply report template", "", "Report template (*.json)")
        if not path:
            return
        try:
            definition = report_definition(read_json(Path(path)))
            schema = describe_inputs(self.case.documents)
            removed = refresh_report(schema, definition)
            if TemplateDialog(schema, definition, removed, self).exec() != QDialog.DialogCode.Accepted:
                return
            self.definition, self.schema = definition, schema
            self.axis_notice.clear()
            self.axis_notice.hide()
            if not definition["sheets"]:
                self.add_sheet(save=False)
            self.rebuild_sheets(1)
            self.changed()
        except (OSError, ValueError, TypeError) as exc:
            self.error.setText(f"Template could not be applied: {exc}")

    def preview(self):
        if self.case is None or self.loading or not self.supported:
            return
        self.error.clear()
        self.general_link.hide()
        self.counts.clear()
        self.show_table([], [])
        self.export_button.setEnabled(False)
        path = self.case.run_folder / "output" / RESULTS_FILENAME
        try:
            with xr.open_dataset(path, engine="scipy") if path.is_file() else nullcontext() as dataset:
                schema = describe_dataset(dataset) if dataset is not None else self.schema
                if self.sheet is not None:
                    table = plan_sheet(schema, self.sheet)
                    precision = schema["axes"][table["axis"]].get("precision", 15)
                    if dataset is not None:
                        rows = table_rows(dataset, table, limit=20, column_limit=12)
                        source = "Recorded results ·"
                        note = ""
                    else:
                        rows = ([v, *[None] * min(11, len(table["columns"]))] for v in table["values"][:20])
                        source = "Expected"
                        note = " · No recorded results; run this case to populate values."
                    self.show_table(table["headings"], rows, precision)
                    self.counts.setText(f"{source} {len(table['rows']):,} rows × {len(table['headings']):,} columns · preview up to 20 × 12{note}")
                else:
                    tables = []
                    for sheet in self.definition["sheets"]:
                        try:
                            tables.append(plan_sheet(schema, sheet))
                        except (ValueError, KeyError, TypeError):
                            pass
                    if dataset is not None:
                        info = run_information(self.case.run_folder, self.case.fingerprint())
                        source = "Recorded run information"
                    else:
                        info = {"Planned case": {"name": self.case.name, "id": self.case.id}, "Inputs": self.case.documents}
                        source = "Planned information"
                    rows = iter(information_rows(info, tables, schema))
                    if self.info_view.currentIndex() == 1:
                        for row in rows:
                            if row == ["Column dictionary"]:
                                break
                    self.show_table(next(rows), rows)
                    self.counts.setText(f"Preview of the first 20 rows · {source}")
                plan_workbook(schema, self.definition)
                self.export_button.setEnabled(dataset is not None)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self.error.setText(str(exc))
            self.general_link.setVisible("General" in str(exc) or not self.schema["quantities"])

    def export(self):
        if not self.editor.save():
            return
        try:
            with xr.open_dataset(self.case.run_folder / "output" / RESULTS_FILENAME, engine="scipy") as ds:
                schema = describe_dataset(ds)
                tables = plan_workbook(schema, self.definition)
                validate_information(run_information(self.case.run_folder, self.case.fingerprint()), tables, schema)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.error.setText(str(exc))
            return
        name = re.sub(r'[\\/:*?"<>|]', "_", self.case.name).strip(". ")[:80] or "Report"
        chooser = QFileDialog(self, "Export workbook", str(self.case.project.root / f"{name}.xlsx"), "Excel workbook (*.xlsx)")
        chooser.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        chooser.setDefaultSuffix("xlsx")
        if chooser.exec() != QDialog.DialogCode.Accepted:
            return
        path = chooser.selectedFiles()[0]
        dialog = ExportDialog({"run_folder": self.case.run_folder, "definition": deepcopy(self.definition),
                               "destination": path, "fingerprint": self.case.fingerprint()}, self)
        dialog.exec()
        dialog.worker.wait()
        if dialog.worker.error:
            QMessageBox.warning(self, "Export failed", dialog.worker.error)
        else:
            self.save_note.setText("Export cancelled" if dialog.worker.cancelled else f"Exported {path}")
        dialog.deleteLater()
