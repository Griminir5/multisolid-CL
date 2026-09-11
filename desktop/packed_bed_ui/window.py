"""Project-first navigation: a case list and a separate selected-case editor."""

from importlib.resources import files
from pathlib import Path

from PyQt6.QtCore import QLockFile, Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QKeySequence
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from .case_list import CaseList, result_label
from .editor import CaseEditor
from .execution import RunController
from .project import Project


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MultiSolid")
        self.resize(1150, 850)
        self.project = None
        self.project_lock = None
        self.closing = False
        self.runner = RunController(self)
        self.runner.changed.connect(self._run_status)
        self.runner.finished.connect(self._run_finished)
        self.pages = QStackedWidget()
        self.setCentralWidget(self.pages)
        self.welcome = QWidget()
        welcome_layout = QVBoxLayout(self.welcome)
        welcome_layout.addStretch()
        title = QLabel("<h1>MultiSolid</h1><p>A project keeps your simulation cases and their latest results together.</p>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(title)
        for label, action in (("Create new project…", self._new_project), ("Open existing project…", self._open)):
            button = QPushButton(label)
            button.clicked.connect(action)
            welcome_layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)
        welcome_layout.addStretch()
        self.pages.addWidget(self.welcome)

        self.home = QWidget()
        layout = QVBoxLayout(self.home)
        self.project_title = QLabel()
        layout.addWidget(self.project_title)
        self.mutation_buttons = []
        menu = self.menuBar().addMenu("Project")
        self.project_actions = []
        for label, callback, shortcut in (
            ("Create New Project…", self._new_project, QKeySequence.StandardKey.New),
            ("Open Project…", self._open, QKeySequence.StandardKey.Open),
            ("Save Project", self._save_project, QKeySequence.StandardKey.Save),
            ("Close Project", self._close_project, QKeySequence.StandardKey.Close),
            ("Open Project Folder", self._show_project_folder, None),
        ):
            action = menu.addAction(label, callback)
            if shortcut is not None:
                action.setShortcut(shortcut)
            self.project_actions.append(action)
        self.case_count = QLabel()
        layout.addWidget(self.case_count)
        self.table = CaseList()
        self.table.action.connect(self._case_action)
        self.table.inclusion_changed.connect(self._include_cases)
        self.table.renamed.connect(self._rename_case)
        layout.addWidget(self.table)
        case_actions = QHBoxLayout()
        for label, callback in (("New Case", self._new_case), ("Import Case", self._add_files),
                                ("New Parameter Study", None), ("Import Parameter Study", self._add_batch)):
            button = QPushButton(label)
            if callback is None:
                button.setEnabled(False)
                button.setToolTip("Parameter study creation is planned for a later step.")
            else:
                button.clicked.connect(callback)
                self.mutation_buttons.append(button)
            case_actions.addWidget(button)
        case_actions.addStretch()
        layout.addLayout(case_actions)
        execution_actions = QHBoxLayout()
        execution_actions.addWidget(QLabel("Maximum workers"))
        self.max_workers = QSpinBox()
        self.max_workers.setRange(1, 1024)
        self.max_workers.setToolTip("Maximum simulations running at once. Each case retains its numerical thread setting.")
        self.max_workers.valueChanged.connect(self._workers_changed)
        execution_actions.addWidget(self.max_workers)
        execution_actions.addWidget(QLabel("concurrent cases · threads are set in each case"))
        execution_actions.addStretch()
        self.run_all_button = QPushButton("Run all cases")
        self.run_all_button.clicked.connect(self._run_all)
        execution_actions.addWidget(self.run_all_button)
        layout.addLayout(execution_actions)
        layout.addWidget(QLabel("Running a case replaces its previous results. Duplicate a case first if you want to keep both."))
        self.pages.addWidget(self.home)

        self.case_page = QWidget()
        case_layout = QVBoxLayout(self.case_page)
        editor_actions = QHBoxLayout()
        back = QPushButton("← Cases")
        back.clicked.connect(self._show_cases)
        editor_actions.addWidget(back)
        self.run_button = QPushButton("Run case")
        self.run_button.clicked.connect(self._start_selected)
        editor_actions.addWidget(self.run_button)
        for label, action in (("Open case folder", self._show_case_folder), ("Open latest run log", self._show_log)):
            button = QPushButton(label)
            button.clicked.connect(action)
            editor_actions.addWidget(button)
        self.case_result = QLabel()
        case_layout.addLayout(editor_actions)
        case_layout.addWidget(self.case_result)
        self.editor = CaseEditor()
        self.editor.changed.connect(self._editor_changed)
        case_layout.addWidget(self.editor)
        self.pages.addWidget(self.case_page)
        self.cancel_button = QPushButton("Cancel execution")
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.cancel_button.setEnabled(False)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self._update_project_actions()
        self.statusBar().showMessage("Create or open a project to begin.")

    def _new_project(self):
        if not self.editor.save():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Create new project — choose a new folder", "New project")
        if path:
            try:
                self._set_project(Project.create(path))
            except (ValueError, OSError) as exc:
                self._error(exc)

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open existing project", filter="MultiSolid project (project.json)")
        if path:
            self.open_project(path)

    def open_project(self, path):
        if not self.editor.save():
            return
        root = Path(path).resolve()
        if root.is_file():
            root = root.parent
        if self.project is not None and root == self.project.root:
            try:
                self._set_project(Project.open(root), self.project_lock)
            except (ValueError, OSError) as exc:
                self._error(exc)
            return
        lock = QLockFile(str(root / ".desktop.lock"))
        lock.setStaleLockTime(0)
        if not lock.tryLock(0):
            self._error("This project is already open in another window, or its folder is not writable.")
            return
        try:
            self._set_project(Project.open(root), lock)
        except (ValueError, OSError) as exc:
            lock.unlock()
            self._error(exc)

    def _set_project(self, project, lock=None):
        if self.runner.active:
            raise ValueError("Wait for the current execution or cancel it before switching projects.")
        if not self.editor.save():
            raise ValueError("Save the current draft before opening another project.")
        if lock is None:
            lock = QLockFile(str(project.root / ".desktop.lock"))
            lock.setStaleLockTime(0)
            if not lock.tryLock(0):
                raise ValueError("This project is already open in another window.")
        try:
            solver_lock = QLockFile(str(project.root / ".solver.lock"))
            solver_lock.setStaleLockTime(0)
            if not solver_lock.tryLock(0):
                raise ValueError("A simulation worker is still using this project. Wait for it to finish before reopening.")
            try:
                project.recover_interrupted()
            finally:
                solver_lock.unlock()
        except Exception:
            if lock is not self.project_lock:
                lock.unlock()
            raise
        if self.project_lock is not None and self.project_lock is not lock:
            self.project_lock.unlock()
        self.project_lock = lock
        self.project = project
        self.editor.case = None
        self.runner.job = {}
        self.max_workers.blockSignals(True)
        self.max_workers.setValue(project.metadata.get("max_workers", 1))
        self.max_workers.blockSignals(False)
        self.table.set_project(project)
        self._update_project_actions()
        self.setWindowTitle(f"{project.metadata['name']} — MultiSolid")
        self.project_title.setText(project.metadata["name"])
        self.statusBar().showMessage("Select a case to edit or preview it. Changes save automatically.")
        self._show_cases()

    def _new_case(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("New case in this project")
        form = QFormLayout(dialog)
        name = QLineEdit("New case")
        template = QComboBox()
        template.addItem("Example — separate channels", "run.yaml")
        template.addItem("Example — feed stream", "run_feed_stream.yaml")
        template.addItem("Empty draft", None)
        form.addRow("Case name", name)
        form.addRow("Start from", template)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create case")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                if template.currentData() is None:
                    case = self.project.add_case(name.text())
                else:
                    source = files("packed_bed") / "examples/default_case" / template.currentData()
                    case = self.project.add_case_from_files(str(source), name.text())
                self._show_case(case)
            except (OSError, ValueError) as exc:
                self._error(exc)

    def _add_files(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Case — select run.yaml", filter="YAML (*.yaml *.yml)")
        if path:
            try:
                self._show_case(self.project.add_case_from_files(path))
            except (OSError, ValueError) as exc:
                self._error(exc)

    def _add_batch(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Parameter Study — select batch.yaml", filter="YAML (*.yaml *.yml)")
        if path:
            try:
                added = self.project.add_cases_from_batch(path)
                self.table.set_project(self.project)
                self._refresh_cases()
                self.statusBar().showMessage(f"Added {len(added)} generated cases. No simulations were started.")
            except (OSError, ValueError) as exc:
                self._error(exc)

    def _case_action(self, action, case_id):
        if self.runner.active:
            return
        case = next(case for case in self.project.cases if case.id == case_id)
        if action == "Run":
            self._start_cases([case])
        elif action == "Edit":
            self._show_case(case)
        elif action == "Duplicate":
            name, ok = QInputDialog.getText(self, "Duplicate Case", "New case name", text=f"{case.name} copy")
            if ok and self.editor.save():
                try:
                    self._show_case(self.project.duplicate_case(case, name))
                except (OSError, ValueError) as exc:
                    self._error(exc)
        elif action == "Delete":
            answer = QMessageBox.question(self, "Delete Case", f"Delete ‘{case.name}’ and its latest results?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                          QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Yes:
                try:
                    self.project.delete_case(case)
                except (OSError, ValueError) as exc:
                    self._error(exc)
                finally:
                    if self.editor.case is case and case not in self.project.cases:
                        self.editor.debounce.stop()
                        self.editor.case = None
                        self.editor.dirty = False
                    self.table.set_project(self.project)
                    self._refresh_cases()

    def _include_cases(self, case_ids, included):
        changed = [case for case in self.project.cases if case.id in case_ids]
        previous = [case.metadata.get("included", True) for case in changed]
        for case in changed:
            case.metadata["included"] = included
        try:
            self.project.save()
        except OSError as exc:
            for case, value in zip(changed, previous):
                case.metadata["included"] = value
            self._error(exc)
        self._refresh_cases()

    def _rename_case(self, case_id, name):
        case = next(case for case in self.project.cases if case.id == case_id)
        previous = case.name
        if name.strip():
            case.metadata["name"] = name.strip()
            try:
                self.project.save()
            except OSError as exc:
                case.metadata["name"] = previous
                self._error(exc)
        self._refresh_cases()

    def _workers_changed(self, value):
        if self.project is None:
            return
        previous = self.project.metadata.get("max_workers", 1)
        self.project.metadata["max_workers"] = value
        try:
            self.project.save()
        except OSError as exc:
            self.project.metadata["max_workers"] = previous
            self.max_workers.blockSignals(True)
            self.max_workers.setValue(previous)
            self.max_workers.blockSignals(False)
            self._error(exc)

    def _update_project_actions(self):
        self.cancel_button.setVisible(self.project is not None)
        for index, action in enumerate(self.project_actions):
            action.setEnabled(not self.runner.active and (index < 2 or self.project is not None))

    def _save_project(self):
        if self.project is not None and self.editor.save():
            try:
                self.project.save()
                self.statusBar().showMessage("Project saved.")
            except OSError as exc:
                self._error(exc)

    def _close_project(self):
        if self.runner.active or not self.editor.save():
            return
        if self.project_lock is not None:
            self.project_lock.unlock()
        self.project_lock = self.project = self.editor.case = None
        self.runner.job = {}
        self.pages.setCurrentWidget(self.welcome)
        self.setWindowTitle("MultiSolid")
        self._update_project_actions()
        self.statusBar().showMessage("Create or open a project to begin.")

    def _show_cases(self):
        if not self.editor.save():
            return
        self._refresh_cases()
        self.pages.setCurrentWidget(self.home)

    def _show_case(self, case):
        if self.editor.set_case(case):
            self._editor_changed()
            self.pages.setCurrentWidget(self.case_page)

    def _refresh_cases(self):
        if self.project is None:
            return
        if set(self.table.items) != {case.id for case in self.project.cases}:
            self.table.set_project(self.project)
        count = sum(case.metadata.get("included", True) for case in self.project.cases)
        replacements = sum(case.metadata.get("included", True) and case.run_folder.exists() for case in self.project.cases)
        self.case_count.setText(f"{len(self.project.cases)} cases · {count} included · {replacements} previous runs will be replaced")
        self.run_all_button.setText(f"Run all included cases ({count})")
        self.run_all_button.setEnabled(count > 0 and not self.runner.active)
        self.table.refresh(self.project, self.runner.job, self.runner.active)

    _result_label = staticmethod(result_label)

    def _editor_changed(self):
        if self.editor.case is None:
            return
        state = self.editor.case.state()
        self.run_button.setEnabled(state["inputs"] == "Ready" and not self.editor.dirty and not self.runner.active)
        self.case_result.setText(f"Latest result: {self._result_label(state)}. Running replaces the previous results.")

    def _start_selected(self):
        if self.editor.case is not None:
            self._start_cases([self.editor.case])

    def _run_all(self):
        self._start_cases([case for case in self.project.cases if case.metadata.get("included", True)])

    def _start_cases(self, cases):
        if self.runner.active or not self.editor.save():
            return
        try:
            path = self.project.prepare_execution(cases, max_workers=self.max_workers.value())
            self.runner.start(path)
        except (OSError, ValueError) as exc:
            self._error(exc)
            return
        self._set_running(self.runner.active)
        self.pages.setCurrentWidget(self.home)

    def _set_running(self, active):
        for button in self.mutation_buttons:
            button.setEnabled(not active)
        self.editor.setEnabled(not active)
        self.max_workers.setEnabled(not active)
        self._update_project_actions()
        self.cancel_button.setEnabled(active and not self.runner.cancelling)
        self._refresh_cases()
        self._editor_changed()

    def _run_status(self, job):
        self._refresh_cases()
        self.cancel_button.setEnabled(self.runner.active and not self.runner.cancelling)
        complete = sum(value["state"] in ("completed", "failed", "cancelled") for value in job.get("cases", {}).values())
        state = "cancelling" if self.runner.active and self.runner.cancelling else job.get("state", "queued")
        message = job.get("message") or f"Execution {state} · {complete}/{len(job.get('cases', {}))} cases finished · {job.get('elapsed_s', 0):.1f} s"
        if not self.runner.active:
            errors = [value.get("message") for value in job.get("cases", {}).values() if value.get("message")]
            if errors:
                message += f" · {errors[0]}"
        self.statusBar().showMessage(message)

    def _run_finished(self):
        self._set_running(False)
        if self.closing:
            self.close()

    def _show_project_folder(self):
        if self.project is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.root)))

    def _show_case_folder(self):
        if self.editor.case is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.editor.case.root)))

    def _show_log(self):
        if self.editor.case is not None:
            path = self.editor.case.run_folder / "worker.log"
            if path.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            else:
                self.statusBar().showMessage("This case does not have a run log yet.")

    def _error(self, error):
        QMessageBox.warning(self, "MultiSolid", str(error))

    def closeEvent(self, event):
        if not self.editor.save():
            event.ignore()
        elif self.runner.active:
            self.closing = True
            self.runner.cancel()
            event.ignore()
        else:
            if self.project_lock is not None:
                self.project_lock.unlock()
            event.accept()
