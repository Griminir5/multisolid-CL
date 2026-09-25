"""Project-first navigation: a case list and a separate selected-case editor."""

from pathlib import Path

from PyQt6.QtCore import QLockFile, Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHeaderView, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QSpinBox, QStackedWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from .case_list import CaseList, icon_button, result_label
from .editor import CaseEditor
from .execution import RunController
from .project import Project
from .study_editor import StudyEditor, choose_baseline
from .definition_editor import DefinitionLibrary
from .navigation import NewProjectDialog, ProjectLocations, display_timestamp
from .results import ResultsPage, CaseSelectionDialog


class MainWindow(QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self.setWindowTitle("MultiSolid")
        self.resize(1150, 850)
        self.project = None
        self.project_lock = None
        self.closing = False
        self.locations = ProjectLocations(self, settings)
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
        welcome_layout.addWidget(QLabel("Recent projects"))
        self.recent_list = QTreeWidget()
        self.recent_list.setAccessibleName("Recent projects")
        self.recent_list.setHeaderLabels(["Name", "Last interaction", "Folder path"])
        self.recent_list.setRootIsDecorated(False)
        self.recent_list.setAlternatingRowColors(True)
        self.recent_list.setUniformRowHeights(True)
        self.recent_list.setStyleSheet("QTreeView::item { padding: 6px 8px; }")
        self.recent_list.setColumnWidth(0, 260)
        self.recent_list.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.recent_list.header().setStretchLastSection(True)
        self.recent_list.headerItem().setToolTip(1, "Date and time in your local time zone.")
        self.recent_list.itemClicked.connect(lambda item, _: self.open_project(item.data(0, Qt.ItemDataRole.UserRole)))
        self.recent_list.itemActivated.connect(lambda item, _: self.open_project(item.data(0, Qt.ItemDataRole.UserRole)))
        welcome_layout.addWidget(self.recent_list)
        self.recent_empty = QLabel("No recent projects available.")
        welcome_layout.addWidget(self.recent_empty)
        welcome_layout.addStretch()
        self.pages.addWidget(self.welcome)

        self.home = QWidget()
        layout = QVBoxLayout(self.home)
        self.project_title = QLabel()
        layout.addWidget(self.project_title)
        self.results_button = QPushButton("Results…")
        self.results_button.clicked.connect(self._show_results)
        self.mutation_buttons = []
        menu = self.menuBar().addMenu("Project")
        self.project_actions = []
        for label, callback, shortcut in (
            ("Create New Project…", self._new_project, QKeySequence.StandardKey.New),
            ("Open Existing Project…", self._open, QKeySequence.StandardKey.Open),
            ("Save Project", self._save_project, QKeySequence.StandardKey.Save),
            ("Close Project", self._close_project, QKeySequence.StandardKey.Close),
            ("Open Project Folder", self._show_project_folder, None),
            ("Reusable definitions…", self._definitions, None),
        ):
            action = menu.addAction(label, callback)
            action.setProperty("requires_project", callback not in (self._new_project, self._open))
            if shortcut is not None:
                action.setShortcut(shortcut)
            self.project_actions.append(action)
        for label, callback, required in (("Export project…", self._export_archive, True),
                                          ("Import project archive…", self._import_archive, False)):
            action = menu.addAction(label, callback)
            action.setProperty("requires_project", required)
            self.project_actions.append(action)
        self.recent_menu = menu.addMenu("Recent projects")
        self.plugins_action = self.menuBar().addAction('Plugins', self._plugins)
        self.plugins_action.setEnabled(False)
        self.recent_menu.aboutToShow.connect(self.locations.refresh)
        self.case_count = QLabel()
        layout.addWidget(self.case_count)
        self.table = CaseList()
        self.table.action.connect(self._case_action)
        self.table.inclusion_changed.connect(self._include_cases)
        self.table.renamed.connect(self._rename_case)
        layout.addWidget(self.table)
        case_actions = QHBoxLayout()
        for label, callback in (("New Case", self._new_case), ("Import Case", self._add_files),
                                ("New Parameter Study", self._new_study), ("Import Parameter Study", self._add_batch)):
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
        execution_actions.addWidget(self.results_button)
        layout.addLayout(execution_actions)
        layout.addWidget(QLabel("Running a case replaces its previous results. Duplicate a case first if you want to keep both."))
        self.pages.addWidget(self.home)
        self.results = ResultsPage()
        self.results.back.connect(self._show_cases)
        self.pages.addWidget(self.results)

        self.case_page = QWidget()
        case_layout = QVBoxLayout(self.case_page)
        editor_actions = QHBoxLayout()
        back = QPushButton("← Back to project")
        back.clicked.connect(self._show_cases)
        editor_actions.addWidget(back)
        self.case_result = QLabel()
        editor_actions.addWidget(self.case_result, 1)
        for label, action in (("Open case folder", self._show_case_folder), ("Open latest run log", self._show_log)):
            button = QPushButton(label)
            button.clicked.connect(action)
            editor_actions.addWidget(button)
        case_layout.addLayout(editor_actions)
        self.editor = CaseEditor()
        self.editor.changed.connect(self._editor_changed)
        case_layout.addWidget(self.editor)
        self.pages.addWidget(self.case_page)
        self.study_editor = StudyEditor()
        self.study_editor.back.connect(self._show_cases)
        self.study_editor.changed.connect(self._refresh_cases)
        self.study_editor.rebuilt.connect(self._study_rebuilt)
        self.study_editor.open_case.connect(self._show_case)
        self.pages.addWidget(self.study_editor)
        self.edit_study_button = icon_button(
            self, "Edit", lambda: self._show_study(self.editor.case.metadata["study_id"]), "Edit study")
        editor_actions.addWidget(self.edit_study_button)
        self.independent_button = QPushButton("Duplicate as independent case")
        self.independent_button.clicked.connect(lambda: self._case_action("Duplicate", self.editor.case.id))
        editor_actions.addWidget(self.independent_button)
        self.edit_study_button.hide()
        self.independent_button.hide()
        self.cancel_button = QPushButton("Cancel execution")
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.cancel_button.setEnabled(False)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self._update_project_actions()
        self.locations.changed.connect(self._refresh_recents)
        self._refresh_recents()
        self.locations.refresh()
        self.statusBar().showMessage("Create or open a project to begin.")

    def _save_editors(self):
        control = QApplication.focusWidget()
        if control is not None and self.isAncestorOf(control):
            control.clearFocus()  # Commit delegates and editingFinished before saving.
        saved = self.editor.save() and self.study_editor.save() and self.results.save()
        if not saved:
            reason = self.editor.validation.text() if self.editor.dirty else (
                self.editor.report.save_note.text() if self.editor.report.dirty else
                self.study_editor.issue.text() if self.study_editor.dirty else self.results.save_note.text())
            self.statusBar().showMessage(f"Navigation cannot complete until the draft is saved. {reason}")
        return saved

    def _refresh_recents(self):
        self.recent_list.clear()
        self.recent_menu.clear()
        entries = self.locations.visible()
        for entry in entries:
            values = [entry["name"], display_timestamp(entry["last_interaction"]), entry["path"]]
            item = QTreeWidgetItem(values)
            item.setData(0, Qt.ItemDataRole.UserRole, entry["path"])
            for column, value in enumerate(values):
                item.setToolTip(column, value)
            self.recent_list.addTopLevelItem(item)
            action = self.recent_menu.addAction(" · ".join(values).replace("&", "&&"))
            action.triggered.connect(lambda _, path=entry["path"]: self.open_project(path))
        self.recent_empty.setVisible(not entries)
        if not entries:
            self.recent_menu.addAction("No recent projects available").setEnabled(False)

    def _remember_project(self):
        if self.project is not None:
            self.locations.remember(self.project)

    def _recover_drafts(self, project):
        drafts = project.drafts.pending()
        if not drafts:
            return True
        prompt = QMessageBox(self)
        prompt.setWindowTitle("Recover unfinished drafts")
        prompt.setText(f"‘{project.metadata['name']}’ has {len(drafts)} unsaved recovery draft(s).")
        prompt.setInformativeText("Recover these edits or keep the saved inputs. Retained run snapshots and results are preserved.")
        prompt.setDetailedText("\n".join(f"{draft['name']} — {draft['updated_at']}" for draft in drafts))
        recover = prompt.addButton("Recover drafts", QMessageBox.ButtonRole.AcceptRole)
        discard = prompt.addButton("Discard drafts", QMessageBox.ButtonRole.DestructiveRole)
        prompt.addButton(QMessageBox.StandardButton.Cancel)
        prompt.setDefaultButton(recover)
        prompt.exec()
        if prompt.clickedButton() is recover:
            for draft in drafts:
                project.drafts.apply(draft)
        elif prompt.clickedButton() is discard:
            for draft in drafts:
                project.drafts.discard(draft)
        else:
            return False
        return True

    def _new_study(self):
        if not self._save_editors():
            return
        selection = choose_baseline(self, self.project, new=True)
        if selection:
            try:
                study = self.project.study_store.create(selection[1], selection[0])
                self._show_study(study.id)
                self._refresh_cases()
            except (ValueError, OSError) as exc:
                self._error(exc)

    def _show_study(self, study_id):
        if not self._save_editors():
            return
        study = self.project.study_store.studies.get(study_id)
        if study is None:
            self._error("The saved study rule is unavailable.")
            return
        if self.study_editor.set_study(self.project, study):
            self.pages.setCurrentWidget(self.study_editor)
            self.setWindowTitle(f"{study.name} — {self.project.metadata['name']} — MultiSolid")
            self.statusBar().showMessage("Study edits save automatically. Review the preview before creating or replacing cases.")

    def _study_rebuilt(self):
        ids = {case.id for case in self.project.cases}
        if self.editor.case is not None and self.editor.case.id not in ids:
            self.editor.debounce.stop()
            self.editor.case, self.editor.dirty = None, False
            self.editor.report.debounce.stop()
            self.editor.report.case, self.editor.report.dirty = None, False
        for ident in list(self.runner.job.get("cases", {})):
            if ident not in ids:
                self.runner.job["cases"].pop(ident)
        self.table.set_project(self.project)
        self._refresh_cases()

    def _definitions(self):
        if self.project is not None and self._save_editors():
            context = self.study_editor.study.baseline if self.pages.currentWidget() is self.study_editor else (
                self.editor.case.documents if self.editor.case is not None else
                self.project.cases[0].documents if self.project.cases else None)
            DefinitionLibrary(self.project.study_store, context, self).exec()
            self._refresh_cases()
            if self.pages.currentWidget() is self.study_editor:
                self.study_editor.render()
                self.study_editor.begin_preview()

    def _plugins(self):
        if self.project is None or self.runner.active or not self._save_editors():
            return
        from .plugins_dialog import PluginsDialog
        try:
            PluginsDialog(self.project, self).exec()
            if self.editor.case is not None:
                self.editor.set_case(self.editor.case)
            self._refresh_cases()
            if self.pages.currentWidget() is self.study_editor:
                self.study_editor.render()
                self.study_editor.begin_preview()
        except (ValueError, OSError) as exc:
            self._error(exc)

    def _new_project(self):
        if self.runner.active or not self._save_editors():
            return
        dialog = NewProjectDialog(self.locations, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self._set_project(dialog.project)
            except (ValueError, OSError) as exc:
                self._error(exc)

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open existing project", filter="MultiSolid project (project.json)")
        if path:
            self.open_project(path)

    def _archive_transfer(self, writer, title, **arguments):
        from .report import ExportDialog
        dialog = ExportDialog(arguments, self, writer, title=title, message=f"{title}…")
        dialog.exec()
        dialog.worker.wait()
        dialog.deleteLater()
        if dialog.worker.error:
            raise ValueError(dialog.worker.error)
        return None if dialog.worker.cancelled else dialog.worker.value

    def _export_archive(self):
        if self.project is None or self.runner.active or not self._save_editors():
            return
        from .project_archive import export_project
        chooser = QFileDialog(self, "Export project inputs — results excluded",
            str(self.project.root.with_name(self.project.root.name + ".msproject")), "MultiSolid project archive (*.msproject)")
        chooser.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        chooser.setDefaultSuffix("msproject")
        if chooser.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            result = self._archive_transfer(export_project, "Export project", project=self.project,
                                           destination=chooser.selectedFiles()[0])
            self.statusBar().showMessage(f"Exported {result}" if result else "Export cancelled.")
        except (OSError, ValueError) as exc:
            self._error(exc)

    def _import_archive(self):
        if self.runner.active or not self._save_editors():
            return
        from .project_archive import archive_name, import_project
        path, _ = QFileDialog.getOpenFileName(self, "Import project archive", "",
                                             "MultiSolid project archive (*.msproject *.zip)")
        if not path:
            return
        try:
            name = archive_name(path)
            def create(destination, name):
                return self._archive_transfer(import_project, "Import project", source=path,
                                              destination=destination, name=name)
            dialog = NewProjectDialog(self.locations, self, creator=create, name=name)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.open_project(dialog.path)
        except (OSError, ValueError) as exc:
            self._error(exc)

    def open_project(self, path):
        if self.runner.active:
            self._error("Wait for the current execution or cancel it before switching projects.")
            return
        if not self._save_editors():
            return
        root = Path(path).resolve()
        if root.is_file():
            root = root.parent
        lock = self.project_lock if self.project is not None and root == self.project.root else None
        if lock is None:
            lock = QLockFile(str(root / ".desktop.lock"))
            lock.setStaleLockTime(0)
            if not lock.tryLock(0):
                self._error("This project is already open in another window, or its folder is not writable.")
                return
        try:
            # Check for a surviving worker before transaction recovery reads or
            # repairs project folders, not just before displaying the project.
            solver_lock = QLockFile(str(root / ".solver.lock"))
            solver_lock.setStaleLockTime(0)
            if not solver_lock.tryLock(0):
                raise ValueError("A simulation worker is still using this project. Wait for it to finish before reopening.")
            try:
                project = Project.open(root)
            finally:
                solver_lock.unlock()
            self._set_project(project, lock)
        except (ValueError, OSError) as exc:
            if lock is not self.project_lock:
                lock.unlock()
            self._error(exc)

    def _set_project(self, project, lock=None):
        if self.runner.active:
            raise ValueError("Wait for the current execution or cancel it before switching projects.")
        if not self._save_editors():
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
            if not self._recover_drafts(project):
                if lock is not self.project_lock:
                    lock.unlock()
                return
        except Exception:
            if lock is not self.project_lock:
                lock.unlock()
            raise
        if self.project_lock is not None and self.project_lock is not lock:
            self.project_lock.unlock()
        self.project_lock = lock
        self.study_editor.clear()
        self.results.clear()
        if self.project is not None:
            self.project.on_edit = None
        self.project = project
        self.editor.case = self.editor.report.case = None
        project.on_edit = self._remember_project
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
        self._remember_project()

    def _new_case(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("New case in this project")
        form = QFormLayout(dialog)
        name = QLineEdit("New case")
        form.addRow("Case name", name)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create case")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                if self._save_editors():
                    self._show_case(self.project.add_case(name.text()))
            except (OSError, ValueError) as exc:
                self._error(exc)

    def _add_files(self):
        if not self._save_editors():
            return
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
                if not self._save_editors():
                    return
                study = self.project.import_study(path)
                self._show_study(study.id)
                self._refresh_cases()
                self.statusBar().showMessage("Study imported. Run its baseline successfully before generating cases.")
            except (OSError, ValueError) as exc:
                self._error(exc)

    def _case_action(self, action, case_id):
        if self.runner.active:
            return
        if action == "Study":
            self._show_study(case_id)
            return
        if action == "DeleteStudy":
            self._delete_study(case_id)
            return
        case = next(case for case in self.project.cases if case.id == case_id)
        if action == "Run":
            self._start_cases([case])
        elif action == "Edit":
            self._show_case(case)
        elif action == "Duplicate":
            name, ok = QInputDialog.getText(self, "Duplicate Case", "New case name", text=f"{case.name} copy")
            if ok and self._save_editors():
                try:
                    self._show_case(self.project.duplicate_case(case, name))
                except (OSError, ValueError) as exc:
                    self._error(exc)
        elif action == "Delete":
            answer = QMessageBox.question(self, "Delete Case", f"Delete ‘{case.name}’ and its latest results?" + ("\nThe study can recreate this combination when rebuilt." if case.metadata.get("study_id") else ""),
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
                        self.editor.report.debounce.stop()
                        self.editor.report.case, self.editor.report.dirty = None, False
                    self.table.set_project(self.project)
                    self._refresh_cases()

    def _delete_study(self, study_id):
        if not self._save_editors():
            return
        entry = next(entry for entry in self.project.metadata["studies"] if entry["id"] == study_id)
        cases = self.project.study_store.existing_cases(study_id)
        answer = QMessageBox.question(
            self, "Delete Study",
            f"Delete ‘{entry['name']}’, its {len(cases)} generated cases and "
            f"{sum(case.has_results for case in cases)} retained results?\n"
            "Independent cases and reusable definitions will be kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Yes:
            try:
                self.project.study_store.delete_study(study_id)
            except (OSError, ValueError) as exc:
                self._error(exc)
            finally:
                if not any(entry["id"] == study_id for entry in self.project.metadata["studies"]):
                    if self.study_editor.study is not None and self.study_editor.study.id == study_id:
                        self.study_editor.clear()
                    self._study_rebuilt()
                    self._show_cases()

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
        self.plugins_action.setEnabled(self.project is not None and not self.runner.active)
        self.cancel_button.setVisible(self.project is not None)
        for action in self.project_actions:
            action.setEnabled(not self.runner.active and (not action.property("requires_project") or self.project is not None))
        self.recent_menu.setEnabled(not self.runner.active)

    def _save_project(self):
        if self.project is not None and self._save_editors():
            try:
                self.project.save()
                self.statusBar().showMessage("Project saved.")
            except OSError as exc:
                self._error(exc)

    def _close_project(self):
        if self.runner.active or not self._save_editors():
            return
        if self.project_lock is not None:
            self.project_lock.unlock()
        self.study_editor.clear()
        self.results.clear()
        if self.project is not None:
            self.project.on_edit = None
        self.project_lock = self.project = self.editor.case = None
        self.editor.report.case = None
        self.runner.job = {}
        self.pages.setCurrentWidget(self.welcome)
        self.setWindowTitle("MultiSolid")
        self._update_project_actions()
        self.locations.refresh()
        self.statusBar().showMessage("Create or open a project to begin.")

    def _show_cases(self):
        if not self._save_editors():
            return
        self.study_editor.stop_preview()
        self._refresh_cases()
        if self.project is not None:
            self.setWindowTitle(f"{self.project.metadata['name']} — MultiSolid")
        self.pages.setCurrentWidget(self.home)

    def _show_results(self):
        if self.project is None or self.runner.active or not self._save_editors():
            return
        self.study_editor.stop_preview()
        chosen = self.project.metadata.get("results_report", {}).get("case_ids", [])
        dialog = CaseSelectionDialog(self.project, chosen, self)
        if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.selected_cases():
            return
        if self.results.set_project(self.project):
            self.results.set_selected_cases(dialog.selected_cases())
            self.pages.setCurrentWidget(self.results)
            self.setWindowTitle(f"Results — {self.project.metadata['name']} — MultiSolid")

    def _show_case(self, case):
        if not self._save_editors():
            return
        self.study_editor.stop_preview()
        if self.editor.set_case(case):
            generated = bool(case.metadata.get("study_id"))
            self.edit_study_button.setVisible(generated)
            self.independent_button.setVisible(generated)
            self.setWindowTitle(f"{case.name} — {self.project.metadata['name']} — MultiSolid")
            self._editor_changed()
            self.pages.setCurrentWidget(self.case_page)

    def _refresh_cases(self):
        if self.project is None:
            return
        if (set(self.table.items) != {case.id for case in self.project.cases}
                or set(self.table.groups) != {entry["id"] for entry in self.project.metadata.get("studies", [])}):
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
        if self.runner.active and self.editor.case.id in self.runner.job.get("cases", {}):
            state.update(self.runner.job["cases"][self.editor.case.id])
            if state.get("state") != "queued":
                state["stale"] = False
        self.case_result.setText(self._result_label(state))

    def _run_all(self):
        self._start_cases([case for case in self.project.cases if case.metadata.get("included", True)])

    def _start_cases(self, cases):
        if self.runner.active or not self._save_editors():
            return
        try:
            path = self.project.prepare_execution(cases, max_workers=self.max_workers.value())
            self.runner.start(path)
            self._remember_project()
        except (OSError, ValueError) as exc:
            self._error(exc)
            return
        self._set_running(self.runner.active)
        self.pages.setCurrentWidget(self.home)

    def _set_running(self, active):
        if self.project is not None:
            self.project.executing = active
        self.study_editor.setEnabled(not active)
        self.results_button.setEnabled(not active)
        self.results.setEnabled(not active and self.results.supported)
        if active:
            self.study_editor.stop_preview()
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
        self._editor_changed()
        self.editor.update_results()
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
        if not self._save_editors():
            event.ignore()
        elif self.runner.active:
            self.closing = True
            self.runner.cancel()
            event.ignore()
        else:
            self.study_editor.stop_preview()
            if self.project is not None:
                self.project.on_edit = None
            if self.project_lock is not None:
                self.project_lock.unlock()
            event.accept()
