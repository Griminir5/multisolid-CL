from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from threading import Event
from time import monotonic, sleep

import pytest

from packed_bed_ui.project import Project, read_json, write_json
from packed_bed_ui.recovery import case_payload
from packed_bed_ui.worker import activate_snapshot


def wait_for(app, predicate):
    deadline = monotonic() + 5
    while not predicate():
        assert monotonic() < deadline
        app.processEvents()
        sleep(.005)


@pytest.fixture
def saved_case(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    case.documents["run"]["simulation"]["repeat_program"] = True
    case.save()
    job = read_json(project.prepare_execution([case]))
    activate_snapshot(case.root / f".pending-{job['attempt_id']}")
    write_json(case.run_folder / "status.json", {"state": "completed"})
    (case.run_folder / "result.txt").write_text("retained result")
    return case


def test_creation_form_defaults_cancel_custom_location_and_existing_folder(qt_app, tmp_path, monkeypatch):
    from PyQt6.QtCore import QStandardPaths
    from PyQt6.QtWidgets import QDialog
    from packed_bed_ui.navigation import NewProjectDialog, ProjectLocations, default_parent

    documents = tmp_path / "Documents from OS"
    monkeypatch.setattr(QStandardPaths, "writableLocation", lambda _: str(documents))
    assert default_parent() == documents / "MultiSolid"
    locations = ProjectLocations()
    dialog = NewProjectDialog(locations)
    assert dialog.path == documents / "MultiSolid" / "New project"
    dialog.reject()
    assert not documents.exists()
    monkeypatch.setattr(QStandardPaths, "writableLocation", lambda _: "")
    assert default_parent() == Path.home()
    dialog = NewProjectDialog(locations)
    dialog.name.setText("A project")
    dialog.location.setText(str(tmp_path / "custom"))
    assert not dialog.path.exists()
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert Project.open(dialog.path).metadata["name"] == "A project"
    assert ProjectLocations().creation_parent == tmp_path / "custom"
    other = Project.create(tmp_path / "elsewhere", "A project")
    locations.remember(other)
    assert ProjectLocations().creation_parent == tmp_path / "custom"
    retry = NewProjectDialog(locations)
    retry.name.setText("A project")
    retry.accept()
    assert retry.result() != QDialog.DialogCode.Accepted
    assert retry.error.text()
    assert Project.open(dialog.path).cases == []


def test_recents_persist_deduplicate_aliases_and_handle_moved_folders(qt_app, tmp_path):
    from packed_bed_ui.navigation import ProjectLocations

    locations = ProjectLocations()
    other_window = ProjectLocations()
    first = Project.create(tmp_path / "first", "Same name")
    second = Project.create(tmp_path / "second", "Same name")
    locations.remember(first)
    other_window.remember(second)
    assert len(other_window.entries) == 2
    locations.refresh()
    wait_for(qt_app, lambda: not locations.checking)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(first.root, target_is_directory=True)
    except OSError:  # Windows can require privileges to create directory symlinks.
        alias = first.root
    locations.remember(Project.open(alias))
    assert [entry["path"] for entry in locations.visible()] == [str(first.root), str(second.root)]
    assert len({entry["name"] for entry in locations.visible()}) == 1
    moved = tmp_path / "moved"
    first.root.rename(moved)
    restarted = ProjectLocations()
    restarted.refresh()
    wait_for(qt_app, lambda: not restarted.checking)
    assert [entry["path"] for entry in restarted.visible()] == [str(second.root)]
    restarted.remember(Project.open(moved))
    assert [entry["path"] for entry in restarted.visible()] == [str(moved), str(second.root)]


def test_recent_probes_do_not_block_ui_or_read_project_contents(qt_app, tmp_path, monkeypatch):
    from packed_bed_ui.navigation import ProjectLocations

    locations = ProjectLocations()
    slow = Project.create(tmp_path / "slow")
    fast = Project.create(tmp_path / "fast")
    locations.remember(slow)
    locations.remember(fast)
    locations = ProjectLocations()
    started, release = Event(), Event()
    is_file = Path.is_file

    def probe(path):
        if path.parent == slow.root:
            started.set()
            release.wait(5)
            return False
        return is_file(path)

    monkeypatch.setattr(Path, "is_file", probe)
    monkeypatch.setattr(Project, "open", lambda *_: pytest.fail("Listing must not load projects"))
    try:
        locations.refresh()
        wait_for(qt_app, lambda: started.is_set() and len(locations.visible()) == 1)
        assert locations.visible()[0]["path"] == str(fast.root)
        assert str(slow.root) in locations.checking
    finally:
        release.set()
        wait_for(qt_app, lambda: not locations.checking)


def test_recency_changes_for_edits_and_execution_not_background_status(qt_app, saved_case, monkeypatch):
    from packed_bed_ui.window import MainWindow

    window = MainWindow()
    window._set_project(saved_case.project)
    first = deepcopy(window.locations.entries)
    window._run_status({"state": "completed", "cases": {}})
    saved_case.project.recover_interrupted()
    assert window.locations.entries == first
    window._rename_case(saved_case.id, "Renamed")
    edited = deepcopy(window.locations.entries)
    assert edited[0]["last_interaction"] > first[0]["last_interaction"]
    monkeypatch.setattr(window.runner, "start", lambda _: None)
    window._start_cases([saved_case])
    assert window.locations.entries[0]["last_interaction"] > edited[0]["last_interaction"]
    window._close_project()
    assert window.cancel_button.isHidden()
    window.recent_list.itemClicked.emit(window.recent_list.topLevelItem(0), 0)
    assert window.project.root == saved_case.project.root
    assert window.project.metadata["cases"][0]["name"] == "Renamed"
    assert window.recent_menu.actions()[0].text().startswith(window.project.metadata["name"])
    window.close()


def test_active_cell_and_report_edits_flush_on_close_and_reopen(qt_app, saved_case):
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLineEdit
    from packed_bed_ui.window import MainWindow

    window = MainWindow()
    window._set_project(saved_case.project)
    window._show_case(saved_case)
    window.show()
    editor = window.editor
    editor.tabs.setCurrentWidget(editor.bed)
    table = editor.bed.zones
    table.editItem(table.item(0, 2))
    field = table.findChild(QLineEdit)
    field.selectAll()
    QTest.keyClicks(field, "unfinished")
    # No focus change or debounce: active text is already recoverable.
    draft = Project.open(saved_case.project.root).drafts.pending()[0]
    assert draft["data"]["documents"]["solids"]["initial_profile"]["zones"][0]["e_b"] == "unfinished"
    editor.report.add_sheet()
    assert window.close()
    reopened = Project.open(saved_case.project.root)
    case = reopened.cases[0]
    assert case.documents["solids"]["initial_profile"]["zones"][0]["e_b"] == "unfinished"
    assert case.metadata["report"]["sheets"][0]["name"] == "Sheet 1"
    assert case.state()["inputs"] != "Ready"
    assert case.state()["stale"]
    assert case.state()["state"] == "completed"
    assert (case.run_folder / "result.txt").read_text() == "retained result"
    assert reopened.drafts.pending() == []


@pytest.mark.parametrize("choice", ["Recover drafts", "Discard drafts", "Cancel"])
def test_interrupted_session_recovery_requires_choice(qt_app, saved_case, monkeypatch, choice):
    from PyQt6.QtWidgets import QMessageBox
    from packed_bed_ui.window import MainWindow

    root = saved_case.project.root
    snapshot = (saved_case.run_folder / "snapshot.json").read_bytes()
    # Exit without Qt cleanup/autosave, leaving both a checkpoint and a stale lock.
    script = '''
import os, sys
from PyQt6.QtCore import QLockFile
from PyQt6.QtWidgets import QApplication
from packed_bed_ui.project import Project
from packed_bed_ui.editor import CaseEditor
app = QApplication([])
project = Project.open(sys.argv[1])
lock = QLockFile(str(project.root / '.desktop.lock'))
assert lock.tryLock(0)
editor = CaseEditor()
editor.set_case(project.cases[0])
editor.fields[('simulation', 'time_horizon_s')].setText('unfinished')
editor.report.add_sheet()
os._exit(0)
'''
    subprocess.run([sys.executable, "-c", script, str(root)], check=True, timeout=30)
    saved = Project.open(root)
    assert saved.cases[0].documents["run"]["simulation"]["time_horizon_s"] == .01
    assert "report" not in saved.cases[0].metadata
    assert len(saved.drafts.pending()) == 1

    def choose(dialog):
        button = next(button for button in dialog.buttons() if button.text() == choice)
        button.click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", choose)
    window = MainWindow()
    window.open_project(root)
    if choice == "Cancel":
        assert window.project is None
        assert Project.open(root).drafts.pending()
    else:
        assert window.project is not None
        case = window.project.cases[0]
        assert not window.project.drafts.pending()
        recovered = choice == "Recover drafts"
        assert case.documents["run"]["simulation"]["time_horizon_s"] == ("unfinished" if recovered else .01)
        assert ("report" in case.metadata) == recovered
        assert case.state()["stale"] == recovered
        assert case.state()["state"] == "completed"
    assert (saved_case.run_folder / "snapshot.json").read_bytes() == snapshot
    assert (saved_case.run_folder / "result.txt").read_text() == "retained result"
    window.close()


def test_report_only_recovery_preserves_scientific_fingerprint(qt_app, saved_case):
    from packed_bed_ui.editor import CaseEditor

    editor = CaseEditor()
    editor.set_case(saved_case)
    editor.report.add_sheet()
    editor.report.debounce.stop()
    editor.report.preview_timer.stop()
    project = Project.open(saved_case.project.root)
    project.drafts.apply(project.drafts.pending()[0])
    assert project.cases[0].metadata["report"]["sheets"]
    assert not project.cases[0].state()["stale"]
    editor.close()


def test_report_autosave_flushes_pending_scientific_metadata_with_inputs(qt_app, saved_case):
    from packed_bed_ui.editor import CaseEditor

    editor = CaseEditor()
    editor.set_case(saved_case)
    editor.program.mode.setCurrentIndex(editor.program.mode.findData("feed_stream"))
    editor.report.add_sheet()
    assert editor.dirty and editor.report.dirty
    assert editor.report.save()
    reopened = Project.open(saved_case.project.root)
    assert reopened.cases[0].documents["run"]["simulation"]["program_mode"] == "feed_stream"
    assert reopened.cases[0].metadata["program_modes"]
    assert reopened.cases[0].metadata["report"]["sheets"]
    assert not reopened.drafts.pending()
    editor.close()


def test_case_write_failure_rolls_back_inputs_and_retains_recoverable_edits(qt_app, saved_case, monkeypatch):
    from packed_bed_ui.window import MainWindow
    import packed_bed_ui.study_store as storage

    project = saved_case.project
    original = deepcopy(saved_case.documents)
    window = MainWindow()
    window._set_project(project)
    window._show_case(saved_case)
    window.editor.fields[("simulation", "time_horizon_s")].setText("unfinished")
    original_write = storage.write_json

    def fail_metadata(path, value):
        if path == project.root / "project.json":
            raise PermissionError("test disk is read-only")
        original_write(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "write_json", fail_metadata)
        window._close_project()
        assert window.project is project
        assert window.editor.dirty
        assert "Navigation cannot complete" in window.statusBar().currentMessage()
        assert "read-only" in window.statusBar().currentMessage()
        assert not window.close()
        reopened = Project.open(project.root)
        assert reopened.cases[0].documents == original
        assert reopened.drafts.pending()
    assert window.close()
    reopened = Project.open(project.root)
    assert reopened.cases[0].documents["run"]["simulation"]["time_horizon_s"] == "unfinished"
    assert not reopened.drafts.pending()


def test_study_active_cell_recovery_keeps_baseline_and_retained_results(qt_app, saved_case):
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLineEdit
    from packed_bed_ui.studies import Factor
    from packed_bed_ui.study_editor import StudyEditor

    project = saved_case.project
    study = project.study_store.create("Sweep", saved_case)
    study.mode, study.factors, study.rows = "rows", [Factor("length", "bed_length_m")], [{"length": "1"}]
    project.study_store.save_study(study)
    editor = StudyEditor()
    editor.set_study(project, study)
    editor.show()
    editor.rows.edit(editor.row_model.index(0, 0))
    field = editor.rows.findChild(QLineEdit)
    field.selectAll()
    QTest.keyClicks(field, "unfinished")
    editor.debounce.stop()
    editor.stop_preview()
    reopened = Project.open(project.root)
    assert reopened.study_store.studies[study.id].rows == [{"length": "1"}]
    reopened.drafts.apply(reopened.drafts.pending()[0])
    recovered = reopened.study_store.studies[study.id]
    assert recovered.rows == [{"length": "unfinished"}]
    assert recovered.baseline == study.baseline
    assert not reopened.cases[0].state()["stale"]
    editor.clear()
    editor.close()


def test_old_or_deleted_drafts_do_not_replace_saved_state(saved_case):
    project = saved_case.project
    project.drafts.write("case", saved_case.id, case_payload(saved_case))
    assert project.drafts.pending() == []
    saved_case.documents["run"]["simulation"]["time_horizon_s"] = "unfinished"
    project.drafts.write("case", saved_case.id, case_payload(saved_case))
    project.delete_case(saved_case)
    assert project.drafts.pending() == []


def test_interrupted_multi_document_save_rolls_back_before_offering_draft(qt_app, saved_case, monkeypatch):
    from packed_bed_ui.editor import CaseEditor
    import packed_bed_ui.study_store as storage

    original = deepcopy(saved_case.documents)
    editor = CaseEditor()
    editor.set_case(saved_case)
    editor.put(("run", "model", "bed_length_m"), 2.0)
    editor.put(("program", "inlet_temperature", "initial"), "unfinished")
    editor.debounce.stop()
    write = storage.write_json

    def interrupt_before_commit(path, value):
        if path == saved_case.project.root / "project.json":
            raise KeyboardInterrupt("Simulate process exit during commit")
        write(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "write_json", interrupt_before_commit)
        with pytest.raises(KeyboardInterrupt):
            editor.save()
    reopened = Project.open(saved_case.project.root)
    assert reopened.cases[0].documents == original
    drafts = reopened.drafts.pending()
    assert len(drafts) == 1
    reopened.drafts.apply(drafts[0])
    assert reopened.cases[0].documents["run"]["model"]["bed_length_m"] == 2.0
    assert reopened.cases[0].documents["program"]["inlet_temperature"]["initial"] == "unfinished"
    assert (reopened.cases[0].run_folder / "result.txt").read_text() == "retained result"
    editor.close()
