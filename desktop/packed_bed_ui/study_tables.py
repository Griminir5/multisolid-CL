"""Table models and spreadsheet editing for the study workspace."""

from copy import deepcopy

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QApplication, QComboBox, QLineEdit, QStyledItemDelegate, QTableView

from .editor_widgets import display


class CandidateTable(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.candidates, self.columns = [], []
        self.definitions = {}

    def reset(self, columns, candidates=None):
        self.beginResetModel()
        self.candidates = [] if candidates is None else candidates
        self.columns = columns
        self.endResetModel()

    def append(self, candidates):
        if candidates:
            first = len(self.candidates)
            self.beginInsertRows(QModelIndex(), first, first + len(candidates) - 1)
            self.candidates.extend(candidates)
            self.endInsertRows()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.candidates)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.columns) + 2

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.ToolTipRole and orientation == Qt.Orientation.Horizontal:
            return ["Case name", *[title for _, title in self.columns], "Inputs"][section]
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Vertical:
                return str(section + 1)
            return ["Case name", *[title.replace(" → ", "\n") for _, title in self.columns], "Inputs"][section]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        candidate = self.candidates[index.row()]
        if role == Qt.ItemDataRole.ToolTipRole:
            return candidate.message or candidate.name
        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return candidate.name
            if index.column() == len(self.columns) + 1:
                return candidate.inputs
            value = candidate.selections.get(self.columns[index.column() - 1][0], "")
            definition = self.definitions.get(value) if isinstance(value, str) else None
            return definition.name if definition else display(value)


class ExplicitRows(QAbstractTableModel):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.workspace = workspace

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or self.workspace.study is None else len(self.workspace.study.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or self.workspace.study is None else len(self.workspace.study.factors)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Vertical:
                return str(section + 1)
            return self.workspace.factor_title(self.workspace.study.factors[section]).replace(" → ", "\n")

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        factor = self.workspace.study.factors[index.column()]
        value = self.workspace.study.rows[index.row()].get(factor.id, "")
        if role == Qt.ItemDataRole.EditRole:
            return value
        if role == Qt.ItemDataRole.DisplayRole:
            definition = self.workspace.store.definitions.get(value) if isinstance(value, str) else None
            return definition.name if factor.target.startswith("definition:") and definition else display(value)

    def flags(self, index):
        return super().flags(index) | Qt.ItemFlag.ItemIsEditable

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if self.workspace.study is None or not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        self.paste(index.row(), index.column(), [[value]])
        return True

    def paste(self, row, column, block):
        if self.workspace.study is None:
            return
        draft = deepcopy(self.workspace.study)
        if column + max((len(values) for values in block), default=0) > len(draft.factors):
            self.workspace.issue.setText("The pasted block has more columns than the study. Add the required variations first.")
            return
        for row_offset, values in enumerate(block):
            while len(draft.rows) <= row + row_offset:
                draft.rows.append({factor.id: "" for factor in draft.factors})
            for offset, value in enumerate(values):
                if column + offset >= len(draft.factors):
                    break
                factor = draft.factors[column + offset]
                if factor.target.startswith("definition:"):
                    matches = [d.id for d in self.workspace.store.definitions.values()
                               if d.kind == factor.target.split(":")[1] and (d.id == value or d.name == value)]
                    if len(matches) == 1:
                        value = matches[0]
                draft.rows[row + row_offset][factor.id] = value
        self.workspace.change(draft, "Edit case rows")


class DefinitionDelegate(QStyledItemDelegate):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.workspace = workspace

    def createEditor(self, parent, option, index):
        factor = self.workspace.study.factors[index.column()]
        if not factor.target.startswith("definition:"):
            return QLineEdit(parent)  # Keep unfinished text and full precision; never clamp with a spin box.
        combo = QComboBox(parent)
        combo.addItem("Select definition…", "")
        for definition in self.workspace.store.definitions.values():
            if definition.kind == factor.target.split(":")[1]:
                combo.addItem(definition.name, definition.id)
        return combo

    def setEditorData(self, editor, index):
        if isinstance(editor, QComboBox):
            editor.setCurrentIndex(max(0, editor.findData(index.data(Qt.ItemDataRole.EditRole))))
        elif isinstance(editor, QLineEdit):
            editor.setText(display(index.data(Qt.ItemDataRole.EditRole)))
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentData())
        elif isinstance(editor, QLineEdit):
            model.setData(index, editor.text())
        else:
            super().setModelData(editor, model, index)


class Spreadsheet(QTableView):
    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Paste):
            index = self.currentIndex()
            text = QApplication.clipboard().text().replace("\r\n", "\n").rstrip("\n")
            if text:
                self.model().paste(max(index.row(), 0), max(index.column(), 0),
                                   [line.split("\t") for line in text.split("\n")])
            return
        if event.matches(QKeySequence.StandardKey.Copy):
            selected = self.selectedIndexes()
            if selected:
                rows, columns = [i.row() for i in selected], [i.column() for i in selected]
                text = "\n".join("\t".join(str(self.model().index(row, col).data() or "")
                                          for col in range(min(columns), max(columns) + 1))
                                 for row in range(min(rows), max(rows) + 1))
                QApplication.clipboard().setText(text)
            return
        super().keyPressEvent(event)
