"""Small reusable controls for the case's structured document editors."""

from math import isfinite

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QHeaderView,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
    QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


def number(text):
    """Keep unfinished/invalid numeric input in the saved draft."""
    try:
        value = float(text)
        return value if isfinite(value) else str(text)
    except (ValueError, TypeError):
        return text


def display(value):
    if value is None:
        return ""
    return f"{value:g}" if isinstance(value, float) else str(value)


def select_value(combo, value):
    index = combo.findData(value)
    if index < 0:
        combo.addItem(str(value) if value is not None else "Select…", value)
        index = combo.count() - 1
    combo.setCurrentIndex(index)


def choices(items):
    combo = QComboBox()
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(8)
    for item in items:
        label, value = item if isinstance(item, tuple) else (item, item)
        combo.addItem(label, value)
    return combo


def action_button(text, callback, *, tooltip=None):
    button = QPushButton(text)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setAccessibleName(tooltip or text)
    button.setToolTip(tooltip or text)
    button.clicked.connect(callback)
    return button


def table(headers):
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.verticalHeader().hide()
    widget.verticalHeader().setDefaultSectionSize(34)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    widget.setAlternatingRowColors(True)
    widget.setMinimumSize(0, 80)
    widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
    return widget


def cell(widget, row, column, value, *, editable=True, tooltip=""):
    item = QTableWidgetItem(display(value))
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item.setBackground(widget.palette().alternateBase())
    item.setToolTip(tooltip)
    widget.setItem(row, column, item)
    return item


def choose_items(parent, title, catalog, selected):
    """Choose multiple additions without changing the existing selection."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.resize(480, 440)
    layout = QVBoxLayout(dialog)
    search = QLineEdit()
    search.setPlaceholderText("Filter available items…")
    search.setAccessibleName("Filter available items")
    layout.addWidget(search)
    items = QListWidget()
    for key, (label, description) in catalog.items():
        item = QListWidgetItem(label, items)
        item.setData(Qt.ItemDataRole.UserRole, key)
        item.setToolTip(description)
        item.setCheckState(Qt.CheckState.Checked if key in selected else Qt.CheckState.Unchecked)
        if key in selected:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
    search.textChanged.connect(lambda text: [
        items.item(i).setHidden(text.casefold() not in (
            items.item(i).text() + " " + items.item(i).toolTip()
        ).casefold()) for i in range(items.count())
    ])
    layout.addWidget(items)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add selected")
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return []
    return [items.item(i).data(Qt.ItemDataRole.UserRole) for i in range(items.count())
            if items.item(i).checkState() == Qt.CheckState.Checked
            and items.item(i).data(Qt.ItemDataRole.UserRole) not in selected]


class SelectionList(QWidget):
    changed = pyqtSignal(list)
    show_requested = pyqtSignal(str)

    def __init__(self, catalog, noun, *, show=False, parent=None):
        super().__init__(parent)
        self.catalog, self.noun, self.show = catalog, noun, show
        self.values = []
        self.available = False
        self.show_buttons = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = table([noun.title(), "Show", ""] if show else [noun.title(), ""])
        self.table.horizontalHeader().hide()
        for column in range(1, self.table.columnCount()):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(column, 58 if show and column == 1 else 38)
        layout.addWidget(self.table)

    def set_values(self, values):
        self.values = list(values) if isinstance(values, (list, tuple)) else []
        self.table.clearSpans()
        self.table.clearContents()
        self.table.setRowCount(len(self.values) + 1)
        self.show_buttons = []
        for row, key in enumerate(self.values):
            label, description = self.catalog.get(key, (str(key), "Unavailable definition"))
            cell(self.table, row, 0, label, editable=False, tooltip=description)
            if self.show:
                button = action_button("Show", lambda _, key=key: self.show_requested.emit(key), tooltip=f"Show {label}")
                button.setEnabled(self.available)
                self.table.setCellWidget(row, 1, button)
                self.show_buttons.append(button)
            self.table.setCellWidget(row, self.table.columnCount() - 1, action_button(
                "×", lambda _, key=key: self.remove(key), tooltip=f"Remove {label}",
            ))
        self.table.setSpan(len(self.values), 0, 1, self.table.columnCount())
        self.add_button = action_button(f"+ Add {self.noun}…", self.add)
        self.table.setCellWidget(len(self.values), 0, self.add_button)

    def add(self):
        added = choose_items(self, f"Add {self.noun}", self.catalog, self.values)
        if added:
            self.set_values(self.values + added)
            self.changed.emit(self.values)

    def remove(self, key):
        self.set_values([value for value in self.values if value != key])
        self.changed.emit(self.values)

    def set_results_available(self, available):
        self.available = available
        for button in self.show_buttons:
            button.setEnabled(available)
            if not available:
                button.setToolTip("Run this case to make results available.")


class Preview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(0, 0)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.hide()
        layout.addWidget(self.message)
        layout.addWidget(self.canvas, 1)

    def clear(self, message=""):
        self.figure.clear()
        self.message.setText(message)
        self.message.setVisible(bool(message))
        self.toolbar.update()
        if self.isVisible():
            self.canvas.draw_idle()

    def draw(self):
        self.message.hide()
        self.toolbar.update()
        if self.isVisible():
            self.canvas.draw_idle()
