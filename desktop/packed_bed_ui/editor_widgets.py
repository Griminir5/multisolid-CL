"""Small reusable controls for the case's structured document editors."""

from math import isfinite

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QPalette
from .choice import BinaryChoice
from .theme import NUMERIC_ROLE, LOCKED_ROLE, numeric, numeric_font, style_figure, manager, colors
from .validation import ISSUE_ROLE

from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QGroupBox, QToolButton,
    QSizePolicy, QStyle, QStyledItemDelegate, QStyleOptionViewItem, QTableWidget, QTableWidgetItem,
    QTreeWidget, QVBoxLayout, QWidget,
)


def number(text):
    """Keep unfinished/invalid numeric input in the saved draft."""
    try:
        value = float(text)
        return value if isfinite(value) else str(text)
    except (ValueError, TypeError):
        return text


def display(value):
    """Preserve numeric precision when displayed values are edited and saved."""
    if value is None:
        return ""
    return str(value).removesuffix(".0") if isinstance(value, float) else str(value)


def select_value(combo, value):
    index = combo.findData(value)
    if index < 0:
        combo.addItem(str(value) if value is not None else "Select…", value)
        index = combo.count() - 1
    combo.setCurrentIndex(index)


def choices(items, *, binary=False, vertical=False):
    if binary:
        return BinaryChoice(items, vertical=vertical)
    combo = QComboBox()
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(8)
    for item in items:
        label, value = item if isinstance(item, tuple) else (item, item)
        combo.addItem(label, value)
    return combo


def action_button(text, callback, *, tooltip=None, inspection=False):
    button = QPushButton(text)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setAccessibleName(tooltip or text)
    button.setToolTip(tooltip or text)
    button.clicked.connect(callback)
    button.setProperty("inspectionAction", inspection)
    return button


def table_action(widget, row, text, callback):
    """A full-width trailing action, visually part of the editable table."""
    widget.setSpan(row, 0, 1, widget.columnCount())
    button = action_button(text, callback)
    button.setProperty("role", "tableAction")
    widget.setCellWidget(row, 0, button)
    widget.setRowHeight(row, 36)
    return button


class CollapsibleSection(QGroupBox):
    """An accessible section whose hidden content releases its layout space."""
    expandedChanged = pyqtSignal(bool)

    def __init__(self, title, *, header_widget=None):
        super().__init__()
        self.setProperty("role", "channel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setProperty("role", "channelTitle")
        self.toggle.setProperty("layoutToggle", True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.heading = QHBoxLayout()
        self.heading.addWidget(self.toggle, 1)
        if header_widget is not None:
            self.heading.addWidget(header_widget)
        layout.addLayout(self.heading)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.content, 1)
        self.toggle.toggled.connect(self.set_expanded)
        self.set_expanded(True)

    def set_expanded(self, expanded):
        self.content.setVisible(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle.setToolTip(("Collapse " if expanded else "Expand ") + self.toggle.text().lower())
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding if expanded else QSizePolicy.Policy.Fixed)
        self.expandedChanged.emit(expanded)


def row(*widgets):
    layout = QHBoxLayout()
    for widget in widgets:
        layout.addWidget(widget)
    return layout


def dialog_buttons(dialog, standard=QDialogButtonBox.StandardButton.Save, *, accept=None, label=None):
    buttons = QDialogButtonBox(standard if standard == QDialogButtonBox.StandardButton.Close
                              else standard | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(accept or dialog.accept)
    buttons.rejected.connect(dialog.reject)
    if label:
        buttons.button(standard).setText(label)
    return buttons


def message(text=""):
    label = QLabel(text)
    label.setWordWrap(True)
    return label


class DraftDelegate(QStyledItemDelegate):
    """Keep text cells in the draft while typing, including unfinished numbers."""

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if index.data(NUMERIC_ROLE):
            option.font = numeric_font()
        if index.data(LOCKED_ROLE):
            option.backgroundBrush = option.palette.alternateBase()
        if index.data(ISSUE_ROLE):
            option.backgroundBrush = QBrush(QColor(colors()["missing"]))
            option.palette.setColor(QPalette.ColorRole.Highlight, QColor(colors()["missing"]))
            option.palette.setColor(QPalette.ColorRole.HighlightedText, QColor(colors()["ink"]))

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        self.paint_issue(painter, option, index)

    @staticmethod
    def paint_issue(painter, option, index):
        if index.data(ISSUE_ROLE):
            painter.save()
            painter.setPen(QColor(colors()["error"]))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(option.rect.adjusted(0, 0, -1, -1))
            painter.restore()

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        if isinstance(editor, QLineEdit):
            if index.data(NUMERIC_ROLE):
                numeric(editor)
            editor.textEdited.connect(lambda: self.commitData.emit(editor))
        return editor


def table(headers):
    widget = QTableWidget(0, len(headers))
    widget.setItemDelegate(DraftDelegate(widget))
    widget.setHorizontalHeaderLabels(headers)
    widget.verticalHeader().hide()
    widget.verticalHeader().setDefaultSectionSize(34)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    widget.setAlternatingRowColors(True)
    widget.setMinimumSize(0, 80)
    widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
    return widget


def cell(widget, row, column, value, *, editable=True, tooltip="", numeric_value=False):
    item = QTableWidgetItem(display(value))
    item.setData(NUMERIC_ROLE, numeric_value or isinstance(value, (int, float)))
    if item.data(NUMERIC_ROLE):
        item.setFont(numeric_font())
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item.setData(LOCKED_ROLE, True)
    item.setToolTip(tooltip)
    widget.setItem(row, column, item)
    return item


class _DoubleClickChecks:
    """Toggle checkable rows while preserving ordinary checkbox clicks."""

    def mouseDoubleClickEvent(self, event):
        position = event.position().toPoint()
        index = self.indexAt(position).siblingAtColumn(0)
        flags = index.flags()
        state = index.data(Qt.ItemDataRole.CheckStateRole)
        if (event.button() == Qt.MouseButton.LeftButton and self.isEnabled()
                and flags & Qt.ItemFlag.ItemIsEnabled
                and flags & Qt.ItemFlag.ItemIsUserCheckable
                and (state is not None or flags & Qt.ItemFlag.ItemIsAutoTristate)):
            option = QStyleOptionViewItem()
            option.initFrom(self)
            option.rect = self.visualRect(index)
            option.features = QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
            option.checkState = Qt.CheckState(state) if state is not None else Qt.CheckState.Unchecked
            check = self.style().subElementRect(QStyle.SubElement.SE_ItemViewItemCheckIndicator, option, self)
            # Retain Qt's double-click/release handling without expanding tree groups.
            QAbstractItemView.mouseDoubleClickEvent(self, event)
            # The first click already toggles the checkbox itself. Only row text
            # needs an additional toggle.
            if state is None or not check.contains(position):
                target = Qt.CheckState.Unchecked if option.checkState == Qt.CheckState.Checked else Qt.CheckState.Checked
                self.model().setData(index, target.value, Qt.ItemDataRole.CheckStateRole)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class CheckableListWidget(_DoubleClickChecks, QListWidget):
    """A list whose enabled checkboxes can also be toggled by double-clicking a row."""


class CheckableTreeWidget(_DoubleClickChecks, QTreeWidget):
    """A tree with row double-click toggles and normal expansion for non-checkable headings."""


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
    items = CheckableListWidget()
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
    from .catalogue_widgets import DefinitionDetails
    details = DefinitionDetails()
    details.setMaximumHeight(190)
    items.currentItemChanged.connect(lambda item, _: details.setPlainText(item.toolTip() if item else ''))
    layout.addWidget(details)
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
        self.add_button = action_button(f"+ Add {self.noun}…", self.add)
        layout.addWidget(self.add_button)

    def set_values(self, values):
        self.values = list(values) if isinstance(values, (list, tuple)) else []
        self.table.clearSpans()
        self.table.clearContents()
        self.table.setRowCount(len(self.values))
        self.show_buttons = []
        for row, key in enumerate(self.values):
            label, description = self.catalog.get(key, (str(key), "Unavailable definition"))
            cell(self.table, row, 0, label, editable=False, tooltip=description)
            if self.show:
                button = action_button("Show", lambda _, key=key: self.show_requested.emit(key), tooltip=f"Show {label}", inspection=True)
                button.setEnabled(self.available)
                self.table.setCellWidget(row, 1, button)
                self.show_buttons.append(button)
            self.table.setCellWidget(row, self.table.columnCount() - 1, action_button(
                "×", lambda _, key=key: self.remove(key), tooltip=f"Remove {label}",
            ))

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
        if manager():
            manager().changed.connect(self.restyle)
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

    def restyle(self):
        style_figure(self.figure)
        # Matplotlib chooses icon colours only at construction; refresh existing actions.
        for _, _, image, callback in self.toolbar.toolitems:
            if image and callback in self.toolbar._actions:
                self.toolbar._actions[callback].setIcon(self.toolbar._icon(image + ".png"))
        if self.isVisible():
            self.canvas.draw_idle()

    def showEvent(self, event):
        super().showEvent(event)
        self.canvas.draw_idle()

    def clear(self, message=""):
        self.figure.clear()
        style_figure(self.figure)
        self.message.setText(message)
        self.message.setVisible(bool(message))
        self.toolbar.update()
        if self.isVisible():
            self.canvas.draw_idle()

    def draw(self):
        style_figure(self.figure)
        self.message.hide()
        self.toolbar.update()
        if self.isVisible():
            self.canvas.draw_idle()
