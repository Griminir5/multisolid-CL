from time import monotonic, sleep
import sys

import pytest

from packed_bed_ui.project import Project, input_hashes, read_json, write_json
from packed_bed_ui.worker import activate_snapshot


def wait_until(app, condition, timeout=10):
    deadline = monotonic() + timeout
    while not condition():
        assert monotonic() < deadline, "Timed out waiting for the Qt worker"
        app.processEvents()
        sleep(0.01)


def test_window_opens_on_welcome_then_project_case_list(qt_app, tmp_path, source_case):
    from packed_bed_ui.window import MainWindow
    from PyQt6.QtWidgets import QPushButton

    window = MainWindow()
    assert window.pages.currentWidget() is window.welcome
    assert [button.text() for button in window.welcome.findChildren(QPushButton)] == [
        "Create new project…", "Open existing project…",
    ]
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case, "Benchmark")
    project.duplicate_case(case, "Exploratory")
    project.add_case("Underdefined")
    window._set_project(project)
    assert window.pages.currentWidget() is window.home
    assert window.table.topLevelItemCount() == 3
    assert window.table.topLevelItem(2).text(2) == "Underdefined"
    assert window.table.topLevelItem(0).text(3) == "Not run"
    window.close()


def test_editor_saves_invalid_drafts_and_keeps_other_cases_unchanged(qt_app, tmp_path, source_case):
    from packed_bed_ui.window import MainWindow

    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    case.documents["run"]["simulation"]["repeat_program"] = True
    other = project.duplicate_case(case, "Other")
    window = MainWindow()
    window._set_project(project)
    window._show_case(case)
    assert case.state()["inputs"] == "Ready"
    field = window.editor.fields[("simulation", "time_horizon_s")]
    field.setText("unfinished")
    window.editor.queue_edit()
    assert window.editor.dirty
    wait_until(qt_app, lambda: not window.editor.dirty)
    assert "cannot run" in window.editor.validation.text()
    assert case.state()["inputs"] in {"Underdefined", "Invalid"}
    assert not window.editor.figures[0].axes
    reopened = Project.open(project.root)
    assert reopened.cases[0].documents["run"]["simulation"]["time_horizon_s"] == "unfinished"
    assert reopened.cases[1].resolve().run.simulation.time_horizon_s == 0.01
    window._show_case(other)
    assert other.state()["inputs"] == "Ready"
    window.close()


@pytest.mark.parametrize("action", ["cancel", "crash", "failed_start"])
def test_controller_stops_workers_and_retains_results_of_unstarted_cases(qt_app, tmp_path, source_case, monkeypatch, action):
    from packed_bed_ui.execution import RunController

    project = Project.create(tmp_path / "project")
    first = project.add_case_from_files(source_case)
    second = project.duplicate_case(first, "Second")
    # The second case already has results. A queued cancellation must keep them.
    job = read_json(project.prepare_execution([second]))
    activate_snapshot(second.root / f".pending-{job['attempt_id']}")
    write_json(second.run_folder / "status.json", {"state": "completed"})
    (second.run_folder / "result.txt").write_text("keep")
    hashes = input_hashes(second.run_folder / "inputs")
    path = project.prepare_execution(project.cases)
    controller = RunController()
    start = controller.process.start
    if action == "failed_start":
        monkeypatch.setattr(controller.process, "start", lambda *_: start(str(tmp_path / "missing-python"), []))
    else:
        script = '''
import time, sys
from pathlib import Path
from packed_bed_ui.project import read_json, write_json
from packed_bed_ui.worker import activate_snapshot
path = Path(sys.argv[1])
job = read_json(path)
case_id = next(iter(job['cases']))
folder = activate_snapshot(path.parent / 'cases' / case_id / ('.pending-' + job['attempt_id']))
write_json(folder / 'status.json', {'state': 'running'})
job['cases'][case_id]['state'] = 'running'
write_json(path, job)
print('ready', flush=True)
while not (path.parent / (".cancel-" + job["attempt_id"])).exists():
    time.sleep(0.02)
''' if action == "cancel" else "raise SystemExit(7)"
        monkeypatch.setattr(controller.process, "start", lambda *_: start(sys.executable, ["-c", script, str(path)]))
    controller.start(path)
    if action == "cancel":
        log = project.root / "execution.log"
        wait_until(qt_app, lambda: log.exists() and "ready" in log.read_text())
        controller.cancel()
    wait_until(qt_app, lambda: not controller.active)
    assert read_json(path)["state"] == ("cancelled" if action == "cancel" else "failed")
    assert (second.run_folder / "result.txt").read_text() == "keep"
    assert second.state()["state"] == "completed"
    assert input_hashes(second.run_folder / "inputs") == hashes
    if action == "cancel":
        assert first.state()["state"] == "cancelled"
    assert not list(project.root.glob("cases/*/.pending-*"))


def test_unsaved_draft_blocks_close_and_project_switch(qt_app, tmp_path, source_case, monkeypatch):
    from packed_bed_ui.window import MainWindow

    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    case.documents["run"]["simulation"]["repeat_program"] = True
    other = Project.create(tmp_path / "other")
    window = MainWindow()
    window._set_project(project)
    window._show_case(case)
    window.show()
    window.editor.fields[("simulation", "time_horizon_s")].setText("0.02")
    window.editor.queue_edit()

    def fail():
        raise PermissionError("test destination is read-only")

    with monkeypatch.context() as patch:
        patch.setattr(case, "save", fail)
        with pytest.raises(ValueError, match="Save the current draft"):
            window._set_project(other)
        assert window.project is project
        assert not window.close()
        assert window.editor.dirty
    assert window.close()
    assert Project.open(project.root).cases[0].documents["run"]["simulation"]["time_horizon_s"] == 0.02


def test_closing_window_stops_active_worker(qt_app, tmp_path, source_case, monkeypatch):
    from packed_bed_ui.window import MainWindow

    window = MainWindow()
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    window._set_project(project)
    window.show()
    start = window.runner.process.start
    monkeypatch.setattr(window.runner.process, "start", lambda *_: start(
        sys.executable, ["-c", "import time, sys; from pathlib import Path; "
                         "from packed_bed_ui.project import read_json; "
                         "p = Path(sys.argv[1]); j = read_json(p); print('ready', flush=True); "
                         "exec(\"while not (p.parent / ('.cancel-' + j['attempt_id'])).exists(): time.sleep(0.02)\")",
                         str(project.root / "execution.json")],
    ))
    window._start_cases([case])
    log = project.root / "execution.log"
    wait_until(qt_app, lambda: log.exists() and "ready" in log.read_text())
    assert not window.close()
    wait_until(qt_app, lambda: not window.runner.active)
    assert not window.isVisible()
    assert read_json(project.root / "execution.json")["state"] == "cancelled"


def test_run_all_validates_included_cases_before_replacing_any_results(qt_app, tmp_path, source_case, monkeypatch):
    from packed_bed_ui.window import MainWindow
    from PyQt6.QtCore import Qt

    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    draft = project.add_case("Draft")
    window = MainWindow()
    window._set_project(project)
    errors = []
    monkeypatch.setattr(window, "_error", lambda error: errors.append(str(error)))
    window._run_all()
    assert not window.runner.active
    assert "Draft" in errors[0]
    assert not (project.root / "execution.json").exists()
    window.table.topLevelItem(1).setCheckState(0, Qt.CheckState.Unchecked)
    assert not draft.metadata["included"]
    started = []
    monkeypatch.setattr(window.runner, "start", lambda path: started.append(read_json(path)))
    window._run_all()
    assert list(started[0]["cases"]) == [case.id]
    window.close()


@pytest.mark.parametrize("target", ["case", "study", "study_case"])
def test_inclusion_mouse_and_keyboard_controls_persist_and_select_run_all(qt_app, tmp_path, source_case, monkeypatch, target):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QStyle, QStyleOptionViewItem
    from packed_bed_ui.window import MainWindow

    project = Project.create(tmp_path / "project")
    independent = project.add_case_from_files(source_case, "Independent")
    children = [project.duplicate_case(independent, f"Variant {i}") for i in range(2)]
    project.metadata["studies"].append({"id": "study", "name": "Study"})
    for child in children:
        child.metadata["study_id"] = "study"
    window = MainWindow()
    window._set_project(project)
    window.show()
    qt_app.processEvents()
    tree = window.table
    item, excluded = {
        "case": (tree.items[independent.id], [independent]),
        "study": (tree.groups["study"], children),
        "study_case": (tree.items[children[0].id], [children[0]]),
    }[target]

    def click_checkbox():
        option = QStyleOptionViewItem()
        option.initFrom(tree)
        option.rect = tree.visualRect(tree.indexFromItem(item, 0))
        option.features = QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
        option.checkState = item.checkState(0)
        check = tree.style().subElementRect(QStyle.SubElement.SE_ItemViewItemCheckIndicator, option, tree)
        assert check.isValid()
        QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=check.center())

    try:
        click_checkbox()
        assert all(not case.metadata["included"] for case in excluded)
        assert window.run_all_button.text() == f"Run all included cases ({3 - len(excluded)})"
        tree.setCurrentItem(item, 0)
        QTest.keyClick(tree, Qt.Key.Key_Space)
        assert all(case.metadata["included"] for case in project.cases)
        click_checkbox()
        included = [case.id for case in project.cases if case not in excluded]
        assert [case.id for case in Project.open(project.root).cases if case.metadata["included"]] == included
        assert window.pages.currentWidget() is window.home

        window.runner.active = True
        window._set_running(True)
        click_checkbox()
        QTest.keyClick(tree, Qt.Key.Key_Space)
        assert all(not case.metadata["included"] for case in excluded)
        window.runner.active = False
        window._set_running(False)

        jobs = []
        monkeypatch.setattr(window.runner, "start", lambda path: jobs.append(read_json(path)))
        window._run_all()
        assert list(jobs[0]["cases"]) == included
    finally:
        window.runner.active = False
        window.close()


def test_project_lock_prevents_two_editors(qt_app, tmp_path):
    from packed_bed_ui.window import MainWindow

    project = Project.create(tmp_path / "project")
    first, second = MainWindow(), MainWindow()
    first._set_project(project)
    with pytest.raises(ValueError, match="already open"):
        second._set_project(Project.open(project.root))
    first.close()
    second._set_project(Project.open(project.root))
    second.close()


def test_opening_current_project_reloads_external_case_edits(qt_app, tmp_path, source_case):
    from packed_bed_ui.window import MainWindow
    import yaml

    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    window = MainWindow()
    window._set_project(project)
    path = case.root / "inputs/run.yaml"
    document = yaml.safe_load(path.read_text())
    document["model"]["axial_cells"] = 5
    path.write_text(yaml.safe_dump(document))
    window.open_project(project.root)
    assert window.project.cases[0].resolve().run.model.axial_cells == 5
    window.close()


def test_study_groups_keep_selection_and_collapse_during_live_updates(qt_app, tmp_path, source_case):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QPushButton
    from packed_bed_ui.window import MainWindow

    project = Project.create(tmp_path / "project")
    first = project.add_case_from_files(source_case, "Benchmark")
    children = [project.duplicate_case(first, f"Variant {i}") for i in range(3)]
    project.metadata["studies"].append({"id": "study", "name": "Program × solids"})
    for child in children:
        child.metadata["study_id"] = "study"
    window = MainWindow()
    window._set_project(project)
    assert window.menuBar().actions()[0].text() == "Project"
    assert window.table.topLevelItemCount() == 2
    assert [window.table.headerItem().text(i) for i in range(5)] == ["Include", "Case", "Inputs", "Latest result", "Actions"]
    group = window.table.groups["study"]
    assert group.childCount() == 3
    group.setExpanded(False)
    window.table.setCurrentItem(window.table.items[first.id])
    group.setCheckState(0, Qt.CheckState.Unchecked)
    assert all(not child.metadata["included"] for child in children)
    assert first.metadata["included"]
    group.child(0).setCheckState(0, Qt.CheckState.Checked)
    assert group.checkState(0) == Qt.CheckState.PartiallyChecked
    window.runner.active = True
    window.runner.job = {"cases": {children[0].id: {"state": "running", "elapsed_s": 2.0}}}
    window._run_status(window.runner.job)
    assert not group.isExpanded()
    assert window.table.currentItem() is window.table.items[first.id]
    assert "1 running" in group.text(3)
    assert window.table.items[children[0].id].text(3) == "Running (2.0 s)"
    window.runner.active = False
    window._set_running(False)
    assert list(window.table.buttons[first.id]) == ["Run", "Duplicate", "Edit", "Delete"]
    assert all(not button.icon().isNull() for button in window.table.buttons[first.id].values())
    buttons = {button.text(): button for button in window.home.findChildren(QPushButton)}
    assert not buttons["New Parameter Study"].isEnabled()
    assert buttons["Import Parameter Study"].isEnabled()
    window.max_workers.setValue(3)
    assert Project.open(project.root).metadata["max_workers"] == 3
    window.close()


def test_elapsed_time_advances_without_solver_status_updates(qt_app, tmp_path, monkeypatch):
    from packed_bed_ui.execution import RunController
    import packed_bed_ui.execution as execution

    controller = RunController()
    controller.path = tmp_path / "execution.json"
    controller.active = True
    controller.started = 10.0
    write_json(controller.path, {"cases": {
        "active": {"state": "running", "started_at": 11.0, "elapsed_s": 0.2},
        "finished": {"state": "completed", "started_at": 11.0, "elapsed_s": 0.5},
        "queued": {"state": "queued", "elapsed_s": 0.0},
    }})
    for now in (12.0, 15.0):
        monkeypatch.setattr(execution, "perf_counter", lambda: now)
        cases = controller._read_status()["cases"]
        assert cases["active"]["elapsed_s"] == now - 11.0
        assert cases["finished"]["elapsed_s"] == 0.5
        assert cases["queued"]["elapsed_s"] == 0.0
    controller.active = False


def test_cancel_reaps_parallel_workers_and_keeps_queued_results(qt_app, tmp_path, source_case, monkeypatch):
    import os
    from pathlib import Path
    from packed_bed_ui.execution import RunController

    project = Project.create(tmp_path / "project")
    first = project.add_case_from_files(source_case)
    second = project.duplicate_case(first, "Second")
    queued = project.duplicate_case(first, "Queued")
    previous = read_json(project.prepare_execution([queued]))
    activate_snapshot(queued.root / f".pending-{previous['attempt_id']}")
    write_json(queued.run_folder / "status.json", {"state": "completed"})
    (queued.run_folder / "result.txt").write_text("keep")
    path = project.prepare_execution(project.cases, max_workers=2)
    script = tmp_path / "slow_job.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from test_worker import slow_prepared_case\n"
        "from packed_bed_ui.worker import run_project_job\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(run_project_job(sys.argv[1], case_worker=slow_prepared_case))\n"
    )
    controller = RunController()
    start = controller.process.start
    monkeypatch.setattr(controller.process, "start", lambda *_: start(sys.executable, [str(script), str(path)]))
    controller.start(path)
    wait_until(qt_app, lambda: all((case.run_folder / "pid.json").exists() for case in (first, second)))
    pids = [read_json(case.run_folder / "pid.json")["pid"] for case in (first, second)]
    controller.cancel()
    wait_until(qt_app, lambda: not controller.active)
    assert read_json(path)["state"] == "cancelled"
    assert [case.state()["state"] for case in project.cases] == ["cancelled", "cancelled", "completed"]
    assert (queued.run_folder / "result.txt").read_text() == "keep"
    assert not list(project.root.glob("cases/*/.pending-*"))
    if os.name == "posix":
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
