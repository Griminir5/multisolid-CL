"""Project locations and a small per-user index of recently used folders."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Thread

from PyQt6.QtCore import QObject, QSettings, QStandardPaths, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton,
)


def default_parent():
    documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    return Path(documents) / "MultiSolid" if documents else Path.home()


def display_timestamp(value):
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%d %b %Y, %H:%M")
    except ValueError:
        return "Unknown"


class ProjectLocations(QObject):
    changed = pyqtSignal()
    checked = pyqtSignal(str, bool)

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.settings = settings if settings is not None else QSettings("MultiSolid", "MultiSolid")
        self.entries = self._read_entries()
        self.available, self.checking = set(), set()
        self.checked.connect(self._checked)

    def _read_entries(self):
        self.settings.sync()
        try:
            entries = json.loads(self.settings.value("recent_projects", "[]"))
            return [entry for entry in entries if isinstance(entry, dict) and all(
                isinstance(entry.get(key), str) for key in ("path", "name", "last_interaction"))][:12]
        except (TypeError, ValueError):
            return []

    @property
    def creation_parent(self):
        return Path(self.settings.value("creation_parent", str(default_parent())))

    def remember_parent(self, path):
        self.settings.setValue("creation_parent", str(Path(path).expanduser().resolve()))
        self.settings.sync()

    def remember(self, project):
        path = os.path.normcase(str(project.root.resolve()))
        self.entries = [{"path": path, "name": project.metadata["name"],
                         "last_interaction": datetime.now(timezone.utc).isoformat()},
                        *(entry for entry in self._read_entries() if entry["path"] != path)][:12]
        self.settings.setValue("recent_projects", json.dumps(self.entries))
        self.settings.sync()
        self.available.add(path)
        self.changed.emit()

    def visible(self):
        return sorted((entry for entry in self.entries if entry["path"] in self.available),
                      key=lambda entry: entry["last_interaction"], reverse=True)

    def refresh(self):
        # Network/removable folders can stall stat(). Never probe them on the GUI
        # thread, or make closing the application wait for an unavailable drive.
        self.entries = self._read_entries()
        for entry in self.entries:
            path = entry["path"]
            if path not in self.checking:
                self.checking.add(path)
                Thread(target=self._check, args=(path,), daemon=True).start()

    def _check(self, path):
        try:
            available = (Path(path) / "project.json").is_file()
        except OSError:
            available = False
        try:
            self.checked.emit(path, available)
        except RuntimeError:
            pass  # The window closed before a slow drive responded.

    def _checked(self, path, available):
        self.checking.discard(path)
        if available:
            self.available.add(path)
        else:
            self.available.discard(path)
        self.changed.emit()


class NewProjectDialog(QDialog):
    def __init__(self, locations, parent=None, *, creator=None, name="New project"):
        super().__init__(parent)
        self.creator = creator
        self.setWindowTitle("Import project" if creator else "Create new project")
        self.setMinimumWidth(800)
        self.locations = locations
        self.initial_parent = str(locations.creation_parent)
        self.parent_chosen = False
        form = QFormLayout(self)
        self.name = QLineEdit(name)
        self.location = QLineEdit(self.initial_parent)
        self.name.setAccessibleName("Project name")
        self.location.setAccessibleName("Location")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse)
        row = QHBoxLayout()
        row.addWidget(self.location)
        row.addWidget(browse)
        form.addRow("Project name", self.name)
        form.addRow("Location", row)
        self.destination = QLabel()
        self.destination.setTextFormat(Qt.TextFormat.PlainText)
        self.destination.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.destination.setWordWrap(True)
        form.addRow("Project folder", self.destination)
        self.error = QLabel()
        self.error.setWordWrap(True)
        form.addRow(self.error)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import project" if creator else "Create project")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)
        self.name.textChanged.connect(self.update_path)
        self.location.textChanged.connect(self.update_path)
        self.update_path()
        self.resize(860, self.sizeHint().height())

    @property
    def path(self):
        return Path(self.location.text().strip()).expanduser() / self.name.text().strip()

    def update_path(self):
        name = self.name.text().strip()
        valid = bool(name and name not in (".", "..") and not any(c in name for c in '/\\\0'))
        valid = valid and Path(self.location.text().strip()).expanduser().is_absolute()
        self.destination.setText(str(self.path) if valid else "Enter a name and an absolute parent folder.")
        self.location.setToolTip(self.location.text())
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)

    def browse(self):
        path = QFileDialog.getExistingDirectory(self, "Project location", self.location.text())
        if path:
            self.parent_chosen = True
            self.location.setText(path)

    def accept(self):
        from .project import Project
        try:
            self.project = (self.creator or Project.create)(self.path, self.name.text().strip())
            if self.project is None:
                self.error.setText("Import cancelled.")
                return
        except (OSError, ValueError) as exc:
            self.error.setText(str(exc))
            return
        if self.parent_chosen or self.location.text() != self.initial_parent:
            self.locations.remember_parent(self.path.parent)
        super().accept()
