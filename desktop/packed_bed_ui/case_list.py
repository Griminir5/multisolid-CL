"""Case rows grouped by study, with updates that preserve selection and expansion."""

from collections import Counter

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap, QPalette, QTextCharFormat, QTextLayout
from PyQt6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QStyle, QStyledItemDelegate, QToolButton,
    QTreeWidget, QTreeWidgetItem, QWidget, QStyleOptionViewItem,
)


from .theme import colors, manager, numeric_font
import re


STATE_LABELS = {
    "not_run": "Not run", "queued": "Queued", "preparing": "Preparing",
    "checking_cache": "Checking cache", "waiting_for_compilation": "Waiting for compilation",
    "generating": "Generating model", "compiling": "Compiling",
    "initialising": "Initialising", "running": "Running", "writing_results": "Writing results",
    "completed": "Succeeded", "failed": "Failed", "cancelled": "Cancelled", "interrupted": "Interrupted",
}


def result_label(state):
    label = STATE_LABELS.get(state.get("state"), str(state.get("state")))
    if state.get("elapsed_s"):
        label += f" ({state['elapsed_s']:.1f} s)"
    return f"{label} — stale" if state.get("stale") else label


def action_icon(widget, label, theme, fallback):
    paths = {
        "Run": '<path d="M7 4l14 8-14 8z" fill="currentColor"/>',
        "Duplicate": '<rect x="8" y="8" width="12" height="13" rx="1"/><path d="M16 5V3H3v13h2"/>',
        "Edit": '<path d="M4 16L16 4l4 4L8 20l-5 1zM13 7l4 4"/>',
        "Delete": '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>',
    }
    color = widget.palette().windowText().color().name()
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" '
           f'color="{color}" fill="none" stroke="{color}" stroke-width="1.7" '
           f'stroke-linecap="round" stroke-linejoin="round">{paths[label]}</svg>')
    pixmap = QPixmap()
    icon = QIcon(pixmap) if pixmap.loadFromData(svg.encode(), "SVG") else widget.style().standardIcon(fallback)
    return QIcon.fromTheme(theme, icon)


class ThemedIconButton(QToolButton):
    def configure_icon(self, label, theme, fallback):
        self.icon_spec = (label, theme, fallback)
        self.refresh_icon()
        if manager():
            manager().changed.connect(self.refresh_icon)

    def refresh_icon(self):
        self.setIcon(action_icon(self, *self.icon_spec))


def icon_button(widget, label, callback, description):
    theme, fallback = {
        "Run": ("media-playback-start", QStyle.StandardPixmap.SP_MediaPlay),
        "Duplicate": ("edit-copy", QStyle.StandardPixmap.SP_FileDialogNewFolder),
        "Edit": ("document-edit", QStyle.StandardPixmap.SP_FileDialogDetailedView),
        "Delete": ("edit-delete", QStyle.StandardPixmap.SP_TrashIcon),
    }[label]
    button = ThemedIconButton(widget)
    button.configure_icon(label, theme, fallback)
    button.setToolTip(description)
    button.setAccessibleName(description)
    button.setAutoRaise(True)
    button.clicked.connect(callback)
    return button


class CaseItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        if index.column() not in (2, 3):
            return super().paint(painter, option, index)
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text = opt.text
        c = colors()
        token = ("error" if text.startswith(("Failed", "Interrupted")) else
                 "warning" if "stale" in text or text.startswith(("Underdefined", "Needs")) else
                 "success" if text.startswith(("Ready", "Succeeded")) else
                 "active" if text.startswith(("Running", "Queued", "Preparing", "Compiling", "Initialising")) else "ink")
        opt.palette.setColor(QPalette.ColorRole.Text, QColor(c[token]))
        rect = opt.widget.style().subElementRect(QStyle.SubElement.SE_ItemViewItemText, opt, opt.widget)
        text = opt.fontMetrics.elidedText(text, opt.textElideMode, rect.width())
        opt.text = ""
        opt.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
        layout = QTextLayout(text, opt.font)
        formats = []
        for match in re.finditer(r"(?<![\w])\d+(?:[.,]\d+)*(?![\w])", text):
            span = QTextLayout.FormatRange()
            span.start, span.length = match.start(), len(match.group())
            span.format = QTextCharFormat()
            span.format.setFont(numeric_font())
            formats.append(span)
        layout.setFormats(formats)
        layout.beginLayout()
        line = layout.createLine()
        line.setLineWidth(rect.width())
        layout.endLayout()
        painter.save()
        painter.setClipRect(rect)
        painter.setPen(opt.palette.highlightedText().color() if opt.state & QStyle.StateFlag.State_Selected else QColor(c[token]))
        from PyQt6.QtCore import QPointF
        layout.draw(painter, QPointF(rect.left(), rect.top() + (rect.height() - line.height()) / 2))
        painter.restore()

    def createEditor(self, parent, option, index):
        # Restrict text editors without intercepting Qt's checkbox events.
        if index.column() == 1:
            return super().createEditor(parent, option, index)
        return None


class CaseList(QTreeWidget):
    action = pyqtSignal(str, str)
    inclusion_changed = pyqtSignal(list, bool)
    renamed = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(5)
        self.setHeaderLabels(["Include", "Case", "Inputs", "Latest result", "Actions"])
        self.setTreePosition(1)
        self.setItemDelegate(CaseItemDelegate(self))
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.EditKeyPressed)
        self.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        self.setColumnWidth(3, 260)
        self.header().setStretchLastSection(False)
        self.items, self.groups, self.buttons = {}, {}, {}
        self.group_buttons = {}
        self._displayed_states = {}
        self.itemChanged.connect(self._item_changed)
        self.itemDoubleClicked.connect(self._double_clicked)

    def set_project(self, project):
        expansion = {key: group.isExpanded() for key, group in self.groups.items()}
        self.blockSignals(True)
        self.clear()
        self.items, self.groups, self.buttons = {}, {}, {}
        self.group_buttons = {}
        self._displayed_states.clear()
        studies = {study["id"]: study for study in project.metadata.get("studies", [])}
        for study_id, study in studies.items():
            group = QTreeWidgetItem(self)
            group.setData(0, Qt.ItemDataRole.UserRole, ("study", study_id))
            group.setText(1, study["name"])
            group.setExpanded(expansion.get(study_id, True))
            self.groups[study_id] = group
            self.group_buttons[study_id] = self.add_actions(
                group, study_id, f"study {study['name']}", {"Edit": "Study", "Delete": "DeleteStudy"})
        independent_index = 0
        for case in project.cases:
            study_id = case.metadata.get("study_id")
            parent = self.groups.get(study_id, self)
            if parent is self:
                item = QTreeWidgetItem()
                self.insertTopLevelItem(independent_index, item)
                independent_index += 1
            else:
                item = QTreeWidgetItem(parent)
            item.setData(0, Qt.ItemDataRole.UserRole, ("case", case.id))
            item.setText(1, case.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable)
            item.setCheckState(0, Qt.CheckState.Checked if case.metadata.get("included", True) else Qt.CheckState.Unchecked)
            self.items[case.id] = item
            self.buttons[case.id] = self.add_actions(
                item, case.id, case.name, {label: label for label in ("Run", "Duplicate", "Edit", "Delete")})
        for group in self.groups.values():
            group.setText(1, f"{group.text(1)} ({group.childCount()} cases)")
        self.blockSignals(False)

    def add_actions(self, item, key, name, actions):
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)
        buttons = {}
        for label, action in actions.items():
            button = icon_button(self, label, lambda _, a=action: self.action.emit(a, key), f"{label} {name}")
            layout.addWidget(button)
            buttons[label] = button
        self.setItemWidget(item, 4, holder)
        return buttons

    def refresh(self, project, job, active):
        self.blockSignals(True)
        states = {}
        for case in project.cases:
            state = case.state()
            if active and case.id in job.get("cases", {}):
                state.update(job["cases"][case.id])
                if state.get("state") != "queued":
                    state["stale"] = False
            states[case.id] = state
            display_key = (state, active, case.name, case.metadata.get("included", True))
            if self._displayed_states.get(case.id) == display_key:
                continue
            self._displayed_states[case.id] = display_key
            item = self.items[case.id]
            item.setText(1, case.name)
            item.setText(2, state["inputs"])
            item.setToolTip(2, state.get("input_message", ""))
            item.setText(3, result_label(state))
            item.setToolTip(3, state.get("message", ""))
            item.setCheckState(0, Qt.CheckState.Checked if case.metadata.get("included", True) else Qt.CheckState.Unchecked)
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if not active:
                flags |= Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable
            item.setFlags(flags)
            for label, button in self.buttons[case.id].items():
                button.setAccessibleName(f"{label} {case.name}")
                button.setEnabled(not active and (label != "Run" or state["inputs"] == "Ready"))
        for study_id, group in self.groups.items():
            entry = next((entry for entry in project.metadata.get("studies", []) if entry["id"] == study_id), {})
            group.setText(1, f"{entry.get('name', 'Study')} ({group.childCount()} cases)")
            for label, button in self.group_buttons[study_id].items():
                button.setEnabled(not active)
                button.setToolTip(f"{label} study {entry.get('name', 'Study')}")
                button.setAccessibleName(button.toolTip())
            children = [group.child(i) for i in range(group.childCount())]
            checks = {child.checkState(0) for child in children}
            group.setCheckState(0, next(iter(checks)) if len(checks) == 1 else Qt.CheckState.PartiallyChecked if checks else Qt.CheckState.Unchecked)
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            group.setFlags(flags if active or not children else flags | Qt.ItemFlag.ItemIsUserCheckable)
            values = [states[child.data(0, Qt.ItemDataRole.UserRole)[1]] for child in children]
            group.setText(2, ", ".join(f"{count} {label.lower()}" for label, count in Counter(value["inputs"] for value in values).items()))
            group.setText(3, ", ".join(f"{count} {label.lower()}" for label, count in Counter(
                STATE_LABELS.get(value["state"], value["state"]) + (" — stale" if value.get("stale") else "")
                for value in values).items()))
            study = project.study_store.studies.get(study_id)
            if study:
                if project.study_store.needs_update(study_id):
                    group.setText(2, "Needs rebuild")
                elif not study.provenance:
                    group.setText(2, "Baseline needs a successful run")
                elif not children:
                    group.setText(2, "Draft · no generated cases")
            group.setToolTip(3, group.text(3))
        self.blockSignals(False)

    def _item_changed(self, item, column):
        kind, key = item.data(0, Qt.ItemDataRole.UserRole)
        if column == 0:
            keys = [key] if kind == "case" else [item.child(i).data(0, Qt.ItemDataRole.UserRole)[1] for i in range(item.childCount())]
            self.inclusion_changed.emit(keys, item.checkState(0) == Qt.CheckState.Checked)
        elif column == 1 and kind == "case":
            self.renamed.emit(key, item.text(1))

    def _double_clicked(self, item, column):
        kind, key = item.data(0, Qt.ItemDataRole.UserRole)
        self.action.emit("Edit" if kind == "case" else "Study", key)
