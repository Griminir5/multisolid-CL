"""One workspace for study rules, cancellable previews, and explicit complete rebuilds."""

from collections import Counter
from copy import deepcopy
import re
from time import perf_counter
from uuid import uuid4

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QUndoCommand, QUndoStack
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMessageBox, QPlainTextEdit, QSplitter, QTableView,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .definition_editor import DefinitionList, DefinitionLibrary, inspect_inputs
from .editor_widgets import action_button, cell, choices, dialog_buttons, display, message, row, table
from .studies import (Factor, candidate_count, combinations_from_rows, factor_values, get_value,
                      parameter_catalogue, rows_from_combinations)
from .study_store import baseline_eligibility
from .study_tables import CandidateTable, DefinitionDelegate, ExplicitRows, Spreadsheet


def choose_baseline(parent, project, *, new=False):
    dialog = QDialog(parent)
    dialog.setWindowTitle("New parameter study" if new else "Select successful baseline")
    dialog.resize(650, 220)
    form = QFormLayout(dialog)
    name = QLineEdit("New parameter study", accessibleName="Study name")
    if new:
        form.addRow("Study name", name)
    cases = QComboBox(accessibleName="Successful baseline case")
    first = None
    for case in project.cases:
        eligibility = baseline_eligibility(case)
        index = cases.count()
        cases.addItem(case.name + (" · Succeeded" if eligibility.eligible else " · Unavailable"), case.id)
        item = cases.model().item(index)
        item.setEnabled(eligibility.eligible)
        item.setToolTip(eligibility.message or "The latest run succeeded and its inputs are unchanged.")
        if eligibility.eligible and first is None:
            first = index
    cases.setCurrentIndex(first if first is not None else -1)
    form.addRow("Baseline case", cases)
    note = message("The study keeps a fixed copy of these successful inputs. It does not copy results."
                  if first is not None else "Run a case successfully before creating a parameter study.")
    form.addRow(note)
    buttons = dialog_buttons(dialog, QDialogButtonBox.StandardButton.Ok)
    buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(first is not None)
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create study draft" if new else "Use baseline")
    form.addRow(buttons)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return next(case for case in project.cases if case.id == cases.currentData()), name.text()
    return None


def choose_parameter(parent, study):
    dialog = QDialog(parent)
    dialog.setWindowTitle("Add variation")
    dialog.resize(700, 500)
    layout = QVBoxLayout(dialog)
    search = QLineEdit(placeholderText="Search parameters…", accessibleName="Search parameters")
    layout.addWidget(search)
    items = QTreeWidget()
    items.setHeaderLabels(["Parameter", "Baseline value"])
    items.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    groups = {}
    for parameter in parameter_catalogue(study):
        if parameter.group not in groups:
            groups[parameter.group] = QTreeWidgetItem(items, [parameter.group])
        item = QTreeWidgetItem(groups[parameter.group], [parameter.title, display(get_value(study.baseline, parameter.path))])
        item.setData(0, Qt.ItemDataRole.UserRole, parameter.id)
        if not parameter.available:
            item.setDisabled(True)
            item.setToolTip(0, parameter.reason)
    group = QTreeWidgetItem(items, ["Reusable definitions"])
    for kind, title in (("program", "Operating program"), ("bed", "Bed configuration")):
        item = QTreeWidgetItem(group, [title])
        item.setData(0, Qt.ItemDataRole.UserRole, f"definition:{kind}")
    items.expandAll()
    layout.addWidget(items)
    def filter_items(text):
        for index in range(items.topLevelItemCount()):
            group = items.topLevelItem(index)
            for child in range(group.childCount()):
                item = group.child(child)
                item.setHidden(text.casefold() not in (group.text(0) + " " + item.text(0)).casefold())
    search.textChanged.connect(filter_items)
    def accept():
        item = items.currentItem()
        if item is not None and item.data(0, Qt.ItemDataRole.UserRole) and not item.isDisabled():
            dialog.accept()
    items.itemDoubleClicked.connect(lambda *_: accept())
    layout.addWidget(dialog_buttons(dialog, QDialogButtonBox.StandardButton.Ok, accept=accept))
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return items.currentItem().data(0, Qt.ItemDataRole.UserRole)
    return None


class FactorDialog(QDialog):
    def __init__(self, workspace, factor):
        super().__init__(workspace)
        self.workspace, self.factor = workspace, deepcopy(factor)
        self.setWindowTitle(workspace.factor_title(factor))
        self.resize(660, 460)
        layout = QVBoxLayout(self)
        self.parameter = next((p for p in parameter_catalogue(workspace.study) if p.id == factor.target), None)
        self.definition_kind = factor.target.split(":", 1)[1] if factor.target.startswith("definition:") else None
        if self.definition_kind:
            self.definitions = DefinitionList(workspace.store, workspace.study.baseline,
                                              kind=self.definition_kind, selected=factor.values)
            layout.addWidget(self.definitions)
        else:
            layout.addWidget(QLabel("Baseline: " + display(get_value(workspace.study.baseline, self.parameter.path))
                                    + (" " + self.parameter.unit if self.parameter.unit else "")))
            self.mode = choices(["Values", "Range"])
            self.mode.setCurrentIndex(1 if factor.range is not None else 0)
            layout.addWidget(self.mode)
            self.values = QPlainTextEdit(", ".join(map(display, factor.values)), accessibleName="Variation values",
                                         placeholderText="Enter values separated by commas, spaces, or new lines")
            layout.addWidget(self.values)
            self.range_widget = QWidget()
            form = QFormLayout(self.range_widget)
            self.range_fields = {}
            for key, title in (("start", "Start"), ("end", "End"), ("count", "Number of values")):
                control = QLineEdit(display((factor.range or {}).get(key, "")), accessibleName=title)
                self.range_fields[key] = control
                form.addRow(title, control)
                control.textChanged.connect(self.refresh_values)
            layout.addWidget(self.range_widget)
            self.expanded = QPlainTextEdit(readOnly=True, accessibleName="Expanded values", maximumHeight=110)
            layout.addWidget(self.expanded)
            self.values.textChanged.connect(self.refresh_values)
            self.mode.currentIndexChanged.connect(self.refresh_values)
            self.refresh_values()
        layout.addWidget(dialog_buttons(self, accept=self.accept_values, label="Save variation"))

    def refresh_values(self):
        is_range = self.mode.currentIndex() == 1
        self.values.setVisible(not is_range)
        self.range_widget.setVisible(is_range)
        self.factor.range = {key: widget.text() for key, widget in self.range_fields.items()} if is_range else None
        self.factor.values = [value for value in re.split(r"[,;\s]+", self.values.toPlainText().strip()) if value]
        try:
            values = factor_values(self.factor, self.parameter)
            self.expanded.setPlainText(", ".join(map(display, values)))
        except (ValueError, ArithmeticError) as exc:
            self.expanded.setPlainText("Draft — " + str(exc))

    def accept_values(self):
        if self.definition_kind:
            self.factor.values = self.definitions.checked()
        else:
            self.refresh_values()
        self.accept()


class StudyEdit(QUndoCommand):
    def __init__(self, workspace, before, after, label):
        super().__init__(label)
        self.workspace, self.before, self.after = workspace, deepcopy(before), deepcopy(after)

    def undo(self):
        self.workspace.set_draft(self.before)

    def redo(self):
        self.workspace.set_draft(self.after)


class StudyEditor(QWidget):
    changed = pyqtSignal()
    rebuilt = pyqtSignal()
    back = pyqtSignal()
    open_case = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.study = self.store = self.preview = None
        self.loading = False
        self.dirty = False
        self.undo = QUndoStack(self)
        self.debounce = QTimer(self, singleShot=True, interval=350)
        self.debounce.timeout.connect(self.begin_preview)
        self.preview_timer = QTimer(self, interval=0)
        self.preview_timer.timeout.connect(self.preview_batch)
        layout = QVBoxLayout(self)
        self.name = QLineEdit(accessibleName="Study name")
        self.name.textEdited.connect(self.rename)
        self.saved = QLabel()
        layout.addLayout(row(action_button("← Back to project", lambda: self.back.emit()), self.name, self.saved))
        self.baseline_label = QLabel()
        self.add_baseline_button = action_button("Add baseline as independent case", self.add_imported_baseline)
        baseline_row = row(self.baseline_label, action_button("Inspect baseline", self.inspect_baseline),
                           action_button("Replace baseline", self.replace_baseline), self.add_baseline_button)
        baseline_row.setStretch(0, 1)
        layout.addLayout(baseline_row)
        splitter = QSplitter()
        left, right = QWidget(), QWidget()
        self.left_layout, right_layout = QVBoxLayout(left), QVBoxLayout(right)
        self.mode = choices(["All combinations", "Explicit case rows"])
        self.mode.currentIndexChanged.connect(self.change_mode)
        self.left_layout.addLayout(row(self.mode, action_button("Reusable definitions…", self.definitions)))
        self.factors = table(["Parameter", "Values", ""])
        self.factors.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.factors.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.factors.setColumnWidth(1, 135)
        self.factors.setColumnWidth(2, 70)
        self.factors.itemDoubleClicked.connect(lambda item: self.edit_factor(item.row()))
        self.left_layout.addWidget(self.factors)
        self.add_factor_button = action_button("+ Add variation", self.add_factor)
        self.left_layout.addLayout(row(self.add_factor_button, action_button("Edit variation", lambda: self.edit_factor(self.factors.currentRow()))))
        self.rows = Spreadsheet(alternatingRowColors=True, selectionMode=QAbstractItemView.SelectionMode.ExtendedSelection)
        self.row_model = ExplicitRows(self)
        self.rows.setModel(self.row_model)
        self.rows.setItemDelegate(DefinitionDelegate(self))
        self.left_layout.addWidget(self.rows, 1)
        self.row_actions = QWidget()
        row_actions = QHBoxLayout(self.row_actions)
        for label, action in (("+ Row", "add"), ("Duplicate", "duplicate"), ("Remove", "remove"), ("↑", "up"), ("↓", "down")):
            row_actions.addWidget(action_button(label, lambda _, action=action: self.edit_rows(action)))
        self.left_layout.addWidget(self.row_actions)
        self.count = message()
        right_layout.addWidget(self.count)
        self.candidate_table = QTableView(alternatingRowColors=True, selectionBehavior=QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_model = CandidateTable(self)
        self.candidate_table.setModel(self.candidate_model)
        self.candidate_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.candidate_table.selectionModel().currentRowChanged.connect(self.show_issue)
        right_layout.addWidget(self.candidate_table, 1)
        self.issue = QLabel(wordWrap=True, maximumHeight=110, textInteractionFlags=Qt.TextInteractionFlag.TextSelectableByMouse)
        right_layout.addWidget(self.issue)
        self.cancel_preview = action_button("Cancel preview", self.stop_preview)
        self.refresh_button = action_button("Refresh preview", self.begin_preview)
        right_layout.addLayout(row(action_button("Inspect selected case", self.inspect_candidate), self.cancel_preview, self.refresh_button))
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([450, 620])
        layout.addWidget(splitter, 1)
        bottom = QHBoxLayout()
        for label, callback, signal in (("Undo", self.undo.undo, self.undo.canUndoChanged),
                                        ("Redo", self.undo.redo, self.undo.canRedoChanged)):
            button = action_button(label, callback)
            button.setEnabled(False)
            signal.connect(button.setEnabled)
            bottom.addWidget(button)
        for action, shortcut in ((self.undo.createUndoAction(self), QKeySequence.StandardKey.Undo),
                                 (self.undo.createRedoAction(self), QKeySequence.StandardKey.Redo)):
            action.setShortcut(shortcut)
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            self.addAction(action)
        self.replacement_note = message()
        bottom.addWidget(self.replacement_note, 1)
        self.apply_button = action_button("Create cases", self.apply)
        bottom.addWidget(self.apply_button)
        layout.addLayout(bottom)

    def set_study(self, project, study):
        self.finish_cell_edit()
        if not self.save():
            return False
        self.stop_preview()
        self.store = project.study_store
        self.study = deepcopy(study)
        self.dirty = False
        self.undo.clear()
        self.render()
        self.begin_preview()
        return True

    def clear(self):
        self.debounce.stop()
        self.stop_preview()
        self.row_model.beginResetModel()
        self.study = self.store = None
        self.row_model.endResetModel()
        self.candidate_model.reset([])
        self.dirty = False
        self.undo.clear()

    def finish_cell_edit(self):
        """Commit the active spreadsheet cell before navigating away or closing."""
        control = self.rows.focusWidget()
        if control is not None and control is not self.rows:
            control.clearFocus()

    def factor_title(self, factor):
        if factor.target.startswith("definition:"):
            return "Operating program" if factor.target == "definition:program" else "Bed configuration"
        parameter = next((p for p in parameter_catalogue(self.study) if p.id == factor.target), None)
        return parameter.title if parameter else "Unresolved parameter — reselect target"

    def render(self, *, reset_rows=True):
        self.loading = True
        if self.name.text() != self.study.name:
            self.name.setText(self.study.name)
        self.saved.setText("Saving…" if self.dirty else "Saved")
        source = self.study.provenance.get("case_name")
        self.baseline_label.setText(f"Baseline: {source} · successful run · fixed copy" if source else "Baseline needs a successful run")
        self.add_baseline_button.setVisible(not bool(self.study.provenance))
        self.mode.setCurrentIndex(1 if self.study.mode == "rows" else 0)
        self.mode.setEnabled(not bool(self.study.legacy))
        self.add_factor_button.setEnabled(not bool(self.study.legacy))
        factors = self.study.factors
        self.factors.setRowCount(len(factors))
        for index, factor in enumerate(factors):
            title = self.factor_title(factor)
            cell(self.factors, index, 0, title.replace(" → ", "\n"), editable=False, tooltip=title)
            self.factors.setRowHeight(index, 20 * (title.count(" → ") + 1) + 12)
            if factor.range:
                values = f"{factor.range.get('start', '')} → {factor.range.get('end', '')} · {factor.range.get('count', '')} values"
            else:
                values = ", ".join(self.store.definitions[v].name if isinstance(v, str) and v in self.store.definitions else display(v)
                                   for v in factor.values)
            cell(self.factors, index, 1, values, editable=False, tooltip=values)
            if self.study.mode == "rows":
                cell(self.factors, index, 1, "Edit values in the case rows below", editable=False)
            self.factors.setCellWidget(index, 2, action_button("Remove", lambda _, index=index: self.remove_factor(index)))
        if self.study.legacy:
            axes = self.study.legacy["spec"]["axes"]
            self.factors.setRowCount(len(axes))
            for index, axis in enumerate(axes):
                cell(self.factors, index, 0, axis["id"], editable=False)
                cell(self.factors, index, 1, ", ".join(value["id"] for value in axis["values"]), editable=False)
                cell(self.factors, index, 2, "Imported rule", editable=False)
        self.rows.setVisible(self.study.mode == "rows" and not self.study.legacy)
        self.row_actions.setVisible(self.study.mode == "rows" and not self.study.legacy)
        if reset_rows:
            current = self.rows.currentIndex()
            self.row_model.beginResetModel()
            self.row_model.endResetModel()
            if current.isValid() and current.row() < len(self.study.rows) and current.column() < len(self.study.factors):
                self.rows.setCurrentIndex(self.row_model.index(current.row(), current.column()))
        elif self.study.rows and self.study.factors:
            self.row_model.dataChanged.emit(self.row_model.index(0, 0),
                                            self.row_model.index(len(self.study.rows) - 1, len(self.study.factors) - 1))
        self.loading = False

    def change(self, draft, label):
        if draft != self.study:
            self.undo.push(StudyEdit(self, self.study, draft, label))

    def set_draft(self, draft):
        reset_rows = (len(self.study.rows) != len(draft.rows)
                      or [(f.id, f.target) for f in self.study.factors] != [(f.id, f.target) for f in draft.factors])
        self.study = deepcopy(draft)
        self.dirty = True
        self.stop_preview()
        self.render(reset_rows=reset_rows)
        self.debounce.start()
        try:
            self.store.project.drafts.write("study", self.study.id, self.study.rule())
        except (OSError, ValueError) as exc:
            self.issue.setText(f"Recovery copy could not be saved: {exc}")

    def rename(self, text):
        if self.loading or self.study is None:
            return
        draft = deepcopy(self.study)
        draft.name = text
        self.change(draft, "Rename study")

    def save(self):
        self.debounce.stop()
        if self.study is None or not self.dirty:
            return True
        try:
            self.store.save_study(self.study)
            self.store.project.drafts.clear("study", self.study.id)
        except (ValueError, OSError) as exc:
            self.saved.setText("Could not save")
            self.issue.setText(str(exc))
            return False
        self.dirty = False
        self.saved.setText("Saved")
        self.changed.emit()
        return True

    def add_factor(self):
        self.edit_factor()

    def edit_factor(self, index=None):
        if self.study.legacy or (index is not None and not 0 <= index < len(self.study.factors)):
            return
        draft = deepcopy(self.study)
        factor = draft.factors[index] if index is not None else Factor(uuid4().hex, "")
        original_target = factor.target
        if index is None or draft.mode == "rows" or (not factor.target.startswith("definition:") and not any(p.id == factor.target for p in parameter_catalogue(draft))):
            factor.target = choose_parameter(self, draft)
            if not factor.target:
                return
        if draft.mode != "rows":
            dialog = FactorDialog(self, factor)
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            # Definition saves stand independently of saving or cancelling this variation.
            self.changed.emit()
            self.begin_preview()
            if not accepted:
                return
            factor = dialog.factor
        if index is None:
            draft.factors.append(factor)
        else:
            draft.factors[index] = factor
        if factor.target != original_target:
            for values in draft.rows:
                values[factor.id] = ""
        self.change(draft, "Edit variation")

    def remove_factor(self, index):
        draft = deepcopy(self.study)
        factor = draft.factors.pop(index)
        for row in draft.rows:
            row.pop(factor.id, None)
        self.change(draft, "Remove variation")

    def change_mode(self, index):
        if self.loading or self.study is None:
            return
        try:
            if index == 1:
                draft = deepcopy(self.study)
                draft.rows = rows_from_combinations(self.study, self.store.definitions)
                draft.mode = "rows"
            else:
                draft = combinations_from_rows(self.study)
                count = candidate_count(draft, self.store.definitions)
                values = []
                for factor in draft.factors:
                    labels = [self.store.definitions[value].name if isinstance(value, str) and value in self.store.definitions
                              else display(value) for value in factor.values]
                    values.append(f"{self.factor_title(factor)}: " + ", ".join(labels))
                explanation = (f"Use all combinations? This produces {count} cases from {len(self.study.rows)} explicit rows.\n\n"
                               + "\n".join(values))
                answer = QMessageBox.question(self, "All combinations", explanation,
                                              QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
                if answer != QMessageBox.StandardButton.Ok:
                    self.render()
                    return
            self.change(draft, "Change combination mode")
        except (ValueError, ArithmeticError) as exc:
            self.issue.setText(str(exc))
            self.render()

    def edit_rows(self, action):
        draft = deepcopy(self.study)
        index = self.rows.currentIndex().row()
        if action == "add":
            draft.rows.append({factor.id: "" for factor in draft.factors})
        elif 0 <= index < len(draft.rows):
            if action == "duplicate":
                draft.rows.insert(index + 1, deepcopy(draft.rows[index]))
            elif action == "remove":
                draft.rows.pop(index)
            else:
                target = index + (-1 if action == "up" else 1)
                if 0 <= target < len(draft.rows):
                    draft.rows[index], draft.rows[target] = draft.rows[target], draft.rows[index]
        self.change(draft, "Edit case rows")

    def definitions(self):
        if self.save():
            DefinitionLibrary(self.store, self.study.baseline, self).exec()
            self.render()
            self.changed.emit()
            self.begin_preview()

    def inspect_baseline(self):
        inspect_inputs(self, "Study baseline · read-only", self.study.baseline, self.study.editor_metadata)

    def replace_baseline(self):
        if not self.save():
            return
        selection = choose_baseline(self, self.store.project)
        if selection:
            try:
                self.study = self.store.replace_baseline(self.study, selection[0])
                self.undo.clear()
                self.render()
                self.changed.emit()
                self.begin_preview()
            except (ValueError, OSError) as exc:
                self.issue.setText(str(exc))

    def add_imported_baseline(self):
        if self.save():
            try:
                self.open_case.emit(self.store.add_imported_baseline(self.study))
                self.changed.emit()
            except (ValueError, OSError) as exc:
                self.issue.setText(str(exc))

    def stop_preview(self):
        self.preview_timer.stop()
        self.preview = self.pending = None
        self.apply_button.setEnabled(False)
        self.cancel_preview.setEnabled(False)
        self.cancel_preview.hide()

    def begin_preview(self):
        self.stop_preview()
        if self.study is None or not self.save():
            return
        self.issue.clear()
        try:
            self.pending = pending = self.store.preview(self.study, lazy=True)
            columns = [(f.id, self.factor_title(f)) for f in self.study.factors]
            if self.study.legacy:
                columns = [(axis["id"], axis["id"]) for axis in self.study.legacy["spec"]["axes"]]
            self.candidate_model.reset(columns, pending.candidates)
            self.candidate_model.definitions = deepcopy(self.store.definitions)
            self.candidate_table.setColumnWidth(0, 210)
            metrics = self.candidate_table.fontMetrics()
            for index, (_, title) in enumerate(columns, 1):
                width = max(metrics.horizontalAdvance(line) for line in title.split(" → ")) + 22
                self.candidate_table.setColumnWidth(index, min(240, max(105, width)))
            self.candidate_table.setColumnWidth(len(columns) + 1, 100)
            sizes = [factor.value_count() for factor in self.study.factors]
            self.count_prefix = (" × ".join(map(str, sizes)) + f" = {pending.total} cases"
                                 if self.study.mode == "combinations" and not self.study.legacy and len(sizes) > 1
                                 else f"{pending.total} cases")
            self.count.setText(self.count_prefix + " · validating…")
            old_count = len(pending.delete_ids)
            self.replacement_note.setText(
                f"Rebuilding will delete all {old_count} existing generated cases, including {pending.result_count} retained results, and create {pending.total} new cases."
                if old_count else "Creating cases does not run simulations.")
            self.apply_button.setText(f"Replace {old_count} cases with {pending.total}" if old_count else f"Create {pending.total} cases")
            self.cancel_preview.setEnabled(True)
            self.cancel_preview.show()
            self.preview_timer.start()
        except (ValueError, ArithmeticError, TypeError) as exc:
            self.count.setText("Complete the study rule to preview cases.")
            self.issue.setText(str(exc))

    def preview_batch(self):
        batch, pending = [], self.pending
        started = perf_counter()
        try:
            for candidate in pending.remaining:
                batch.append(candidate)
                if len(batch) == 20 or perf_counter() - started >= .025:
                    break
            self.candidate_model.append(batch)
            count = len(self.candidate_model.candidates)
            self.count.setText(self.count_prefix + f" · validated {count}/{pending.total}")
            if count == pending.total:
                self.preview_timer.stop()
                self.cancel_preview.setEnabled(False)
                self.cancel_preview.hide()
                self.preview = pending
                states = Counter(case.inputs for case in self.preview.candidates)
                self.count.setText(self.count_prefix + " · " + " · ".join(f"{n} {state.lower()}" for state, n in states.items()))
                self.store._verify_baseline(self.study)
                entry = next(entry for entry in self.store.project.metadata["studies"] if entry["id"] == self.study.id)
                current = entry.get("generation_signature") == pending.signature and len(pending.delete_ids) == count
                self.apply_button.setEnabled(count > 0 and not current)
                if current:
                    self.replacement_note.setText("Generated cases are up to date.")
        except (ValueError, ArithmeticError, TypeError) as exc:
            self.stop_preview()
            self.issue.setText(str(exc))

    def show_issue(self, index, _previous=None):
        if index.isValid() and index.row() < len(self.candidate_model.candidates):
            candidate = self.candidate_model.candidates[index.row()]
            self.issue.setText(candidate.message or "Inputs ready. Inspect this case to see its program and bed previews.")

    def inspect_candidate(self):
        index = self.candidate_table.currentIndex().row()
        if 0 <= index < len(self.candidate_model.candidates):
            candidate = self.candidate_model.candidates[index]
            inspect_inputs(self, candidate.name + " · read-only", candidate.documents)

    def apply(self):
        if self.preview is None or not self.save():
            return
        try:
            self.store.apply_preview(self.preview)
            self.undo.clear()
            self.rebuilt.emit()
            self.changed.emit()
            self.begin_preview()
        except (ValueError, OSError) as exc:
            self.issue.setText(str(exc))
            self.apply_button.setEnabled(False)
