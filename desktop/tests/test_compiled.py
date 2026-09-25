from copy import deepcopy

import pytest

from packed_bed_ui.project import Project


@pytest.fixture
def compiled_editor(qt_app, tmp_path, source_case):
    from packed_bed_ui.editor import CaseEditor
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    editor = CaseEditor()
    editor.set_case(case)
    yield editor, case
    editor.close()
    editor.deleteLater()
    qt_app.processEvents()


def test_backend_switch_repairs_are_one_explicit_transaction(compiled_editor, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    editor, case = compiled_editor
    backend = editor.general.backend
    backend.setCurrentIndex(backend.findData("compiled"))
    case.documents["run"]["solver"].update(name="band", band_reciprocals=True, vector_exponentials=True)
    editor.set_case(case)
    before = deepcopy(case.documents)
    prompts = []

    def cancel(*args):
        prompts.append(args[2])
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", cancel)
    backend.setCurrentIndex(backend.findData("daetools"))
    assert case.documents == before
    assert backend.currentData() == "compiled"
    assert len(prompts) == 1 and "band reciprocals" in prompts[0]
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Ok)
    backend.setCurrentIndex(backend.findData("daetools"))
    settings = case.documents["run"]["solver"]
    assert settings["backend"] == "daetools" and settings["name"] == "superlu"
    assert not settings["band_reciprocals"] and not settings["vector_exponentials"]


def test_klu_choice_and_imported_alias_preserve_documents(compiled_editor):
    editor, case = compiled_editor
    combo = editor.general.solver
    assert combo.itemText(combo.findData("klu")) == "KLU"
    case.documents["run"]["solver"]["name"] = "trilinos_klu"
    before = deepcopy(case.documents)
    editor.set_case(case)
    assert case.documents == before
    assert not editor.dirty


@pytest.mark.parametrize("name", (
    "superlu_mt", "trilinos_umfpack", "trilinos_lapack", "trilinos_aztecoo",
    "trilinos_aztecoo_ifpack", "trilinos_aztecoo_ml", "sundials_gmres_ifpack",
))
def test_bundled_standard_solver_can_be_selected_and_reopened(compiled_editor, monkeypatch, name):
    import packed_bed.solver_support as support
    monkeypatch.setattr(support, "find_spec", lambda _: object())
    editor, case = compiled_editor
    editor.general.update_solver_choices()
    combo = editor.general.solver
    assert combo.model().item(combo.findData(name)).isEnabled()
    combo.setCurrentIndex(combo.findData(name))
    assert case.documents["run"]["solver"]["name"] == name
    assert case.documents["run"]["solver"]["backend"] == "daetools"
    before = deepcopy(case.documents)
    editor.set_case(case)
    assert case.documents == before
    assert not editor.dirty
    for excluded in ("band", "intel_pardiso"):
        assert not combo.model().item(combo.findData(excluded)).isEnabled()


def test_missing_trilinos_disables_choices_without_rewriting_import(compiled_editor, monkeypatch):
    import packed_bed.solver_support as support
    monkeypatch.setattr(support, "find_spec", lambda module: None if module.endswith(".trilinos") else object())
    editor, case = compiled_editor
    case.documents["run"]["solver"]["name"] = "trilinos_umfpack"
    before = deepcopy(case.documents)
    editor.set_case(case)
    combo = editor.general.solver
    assert case.documents == before
    assert combo.currentData() == "trilinos_umfpack"
    for name in ("klu", "trilinos_umfpack", "trilinos_lapack", "trilinos_aztecoo",
                 "trilinos_aztecoo_ifpack", "trilinos_aztecoo_ml", "sundials_gmres_ifpack"):
        item = combo.model().item(combo.findData(name))
        assert not item.isEnabled()
        assert "component is missing" in item.toolTip()
    assert combo.model().item(combo.findData("superlu")).isEnabled()


def test_standard_only_solver_switch_to_compiled_requires_confirmation(compiled_editor, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    editor, case = compiled_editor
    case.documents["run"]["solver"]["name"] = "trilinos_umfpack"
    editor.set_case(case)
    before = deepcopy(case.documents)
    backend = editor.general.backend
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Cancel)
    backend.setCurrentIndex(backend.findData("compiled"))
    assert case.documents == before
    assert backend.currentData() == "daetools"
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Ok)
    backend.setCurrentIndex(backend.findData("compiled"))
    assert case.documents["run"]["solver"]["name"] == "superlu"
    item = editor.general.solver.model().item(editor.general.solver.findData("trilinos_umfpack"))
    assert not item.isEnabled()


def test_desktop_project_cache_overrides_environment(qt_app, tmp_path, source_case, monkeypatch):
    from packed_bed_ui.execution import RunController
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    controller = RunController()
    calls = []
    monkeypatch.setattr(controller.process, "start", lambda *args: calls.append(args))
    monkeypatch.setenv("PACKED_BED_COMPILED_CACHE", str(tmp_path / "unrelated"))
    controller.start(project.prepare_execution([case]))
    assert controller.process.processEnvironment().value("PACKED_BED_COMPILED_CACHE") == str(project.root / ".packed_bed_cache")
    controller.poll.stop()
    controller.active = False


def test_cache_does_not_change_fingerprint_or_archive_payload(tmp_path, source_case):
    from packed_bed_ui.project_archive import _payload
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    before = case.fingerprint()
    cache = project.root / ".packed_bed_cache"
    cache.mkdir()
    (cache / "native.dll").write_bytes(b"not an input")
    assert case.fingerprint() == before
    # Export enumerates owned definitions rather than the whole project tree.
    payload = _payload(project.root, exporting=True)
    assert ".packed_bed_cache" not in repr(payload)


def test_windowed_worker_captures_python_diagnostics(tmp_path, monkeypatch):
    import sys
    from packed_bed_ui.worker import diagnostic_log
    path = tmp_path / "worker.log"
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    with diagnostic_log(path):
        print("compiler diagnostics", file=sys.stderr)
    assert sys.stdout is None and sys.stderr is None
    assert "compiler diagnostics" in path.read_text()


def test_unwritable_cache_blocks_before_replacing_results(tmp_path, source_case, monkeypatch):
    import packed_bed.solver_support as support
    import packed_bed_ui.project as projects
    monkeypatch.setattr(support, "require_desktop_solver", lambda *a, **k: None)
    monkeypatch.setattr(projects, "require_desktop_solver", lambda *a, **k: None)
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    case.documents["run"]["solver"]["backend"] = "compiled"
    case.run_folder.mkdir()
    marker = case.run_folder / "retained-result"
    marker.write_text("previous result")
    # A file at the required directory path fails identically on both platforms.
    (project.root / ".packed_bed_cache").write_text("cannot create directory")
    with pytest.raises(ValueError, match="writable project cache"):
        project.prepare_execution([case])
    assert marker.read_text() == "previous result"
    assert not list(case.root.glob(".pending-*"))
