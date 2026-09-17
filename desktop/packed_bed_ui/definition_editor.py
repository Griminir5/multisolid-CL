"""Shared input inspection and explicit Save/Cancel editing of reusable definitions."""

from copy import deepcopy
from uuid import uuid4

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout,
                             QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                             QVBoxLayout, QWidget)

from packed_bed.properties import PROPERTY_REGISTRY

from .editor import InputEditor
from .editor_widgets import SelectionList, action_button, choices, dialog_buttons, message, row
from .inputs import apply_definition, definition_payload, empty_documents
from .project import portable_documents
from .studies import ReusableDefinition


def inspect_inputs(parent, title, documents, metadata=None):
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.resize(1100, 800)
    layout = QVBoxLayout(dialog)
    editor = InputEditor()
    editor.set_documents(documents, metadata, read_only=True)
    layout.addWidget(editor)
    layout.addWidget(dialog_buttons(dialog, QDialogButtonBox.StandardButton.Close))
    dialog.exec()


class DefinitionDialog(QDialog):
    def __init__(self, store, kind, context, definition=None, parent=None):
        super().__init__(parent)
        self.store, self.kind, self.definition = store, kind, definition
        self.context = deepcopy(context or portable_documents(empty_documents(uuid4().hex)))
        self.setWindowTitle("Operating program" if kind == "program" else "Bed configuration")
        self.resize(1100, 800)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(definition.name if definition else "New program" if kind == "program" else "New bed configuration")
        self.name.setAccessibleName("Definition name")
        form.addRow("Name", self.name)
        if definition:
            users = store.definition_users(definition.id)
            form.addRow(QLabel("Used by: " + (", ".join(study.name for study in users) or "No studies")))
        self.start = choices(["Copy baseline inputs", "Empty definition"])
        if not definition:
            form.addRow("Start from", self.start)
        layout.addLayout(form)
        self.editor = InputEditor()
        # Definitions expose species and their owned page. Other documents supply preview context.
        for index in range(self.editor.tabs.count()):
            self.editor.tabs.setTabVisible(index, index == (3 if kind == "program" else 2))
        phase = "gas" if kind == "program" else "solid"
        catalog = {key: (key, record.name) for key, record in PROPERTY_REGISTRY.records.items() if record.phase == phase}
        self.species = SelectionList(catalog, f"{phase} species")
        self.species.setMaximumHeight(110)
        self.species.changed.connect(lambda values: self.editor.set_species(phase, values))
        layout.addWidget(self.species)
        layout.addWidget(self.editor, 1)
        layout.addWidget(message("Other baseline inputs supply the preview context. Saving changes only this definition."))
        self.error = message()
        layout.addWidget(self.error)
        layout.addWidget(dialog_buttons(self, accept=self.save_definition))
        self.start.currentIndexChanged.connect(self.load_inputs)
        self.load_inputs()

    def load_inputs(self):
        self.editor.debounce.stop()
        self.editor.dirty = False
        documents = deepcopy(self.context)
        metadata = {}
        if self.definition:
            payload = self.definition.payload
        elif self.start.currentIndex() == 1:
            payload = definition_payload(self.kind, empty_documents(uuid4().hex))
        else:
            payload = definition_payload(self.kind, documents)
        apply_definition(documents, self.kind, payload)
        if self.kind == "program":
            documents["chemistry"]["gas_species"] = deepcopy(payload.get("gas_species", []))
            metadata = payload.get("editor_metadata", {})
        self.editor.set_documents(documents, metadata)
        self.editor.tabs.setCurrentIndex(3 if self.kind == "program" else 2)
        path = ("chemistry", "gas_species") if self.kind == "program" else ("solids", "solid_species")
        self.species.set_values(self.editor.get(path, []))

    def save_definition(self):
        if not self.editor.save():
            return
        definition = ReusableDefinition(self.definition.id if self.definition else uuid4().hex,
                                        self.name.text().strip(), self.kind,
                                        definition_payload(self.kind, self.editor.case.documents, self.editor.case.metadata))
        try:
            self.store.save_definition(definition)
        except (OSError, ValueError) as exc:
            self.error.setText(str(exc))
            return
        self.definition = definition
        self.accept()

    def done(self, result):
        self.editor.debounce.stop()
        super().done(result)


class DefinitionList(QWidget):
    """The same library actions serve standalone management and factor selection."""

    def __init__(self, store, context, *, kind=None, selected=None, parent=None):
        super().__init__(parent)
        self.store, self.context = store, context
        self.checkable = selected is not None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.kind = choices([("Operating programs", "program"), ("Bed configurations", "bed")])
        if kind:
            self.kind.setCurrentIndex(self.kind.findData(kind))
        self.kind.setVisible(kind is None)
        self.kind.currentIndexChanged.connect(lambda: self.refresh())
        layout.addWidget(self.kind)
        self.items = QListWidget()
        self.items.itemDoubleClicked.connect(lambda _: self.manage("Edit"))
        layout.addWidget(self.items)
        actions = ("New", "Edit", "Duplicate") if self.checkable else ("New", "Edit", "Duplicate", "Delete")
        layout.addLayout(row(*(action_button(action, lambda _, action=action: self.manage(action)) for action in actions)))
        self.refresh(selected or [])

    def checked(self):
        return [self.items.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.items.count())
                if self.items.item(i).checkState() == Qt.CheckState.Checked]

    def refresh(self, selected=None):
        selected = self.checked() if selected is None else selected
        self.items.clear()
        for definition in self.store.definitions.values():
            if definition.kind == self.kind.currentData():
                item = QListWidgetItem(definition.name, self.items)
                item.setData(Qt.ItemDataRole.UserRole, definition.id)
                item.setToolTip("Used by: " + (", ".join(s.name for s in self.store.definition_users(definition.id)) or "No studies"))
                if self.checkable:
                    item.setCheckState(Qt.CheckState.Checked if definition.id in selected else Qt.CheckState.Unchecked)

    def manage(self, action):
        item = self.items.currentItem()
        definition = self.store.definitions.get(item.data(Qt.ItemDataRole.UserRole)) if item else None
        if action != "New" and definition is None:
            return
        selected = self.checked()
        try:
            if action == "Delete":
                self.store.delete_definition(definition.id)
            else:
                if action == "New":
                    definition = None
                elif action == "Duplicate":
                    definition = deepcopy(definition)
                    definition.id, definition.name = uuid4().hex, definition.name + " copy"
                dialog = DefinitionDialog(self.store, self.kind.currentData(), self.context, definition, self)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    selected.append(dialog.definition.id)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Reusable definitions", str(exc))
        self.refresh(selected)


class DefinitionLibrary(QDialog):
    def __init__(self, store, context=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reusable definitions")
        self.resize(650, 480)
        layout = QVBoxLayout(self)
        layout.addWidget(DefinitionList(store, context))
        layout.addWidget(dialog_buttons(self, QDialogButtonBox.StandardButton.Close))
