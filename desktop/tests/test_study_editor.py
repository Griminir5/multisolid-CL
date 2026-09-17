from copy import deepcopy
from time import monotonic

import pytest

from packed_bed_ui.project import Project, read_json, write_json
from packed_bed_ui.studies import Factor, parameter_catalogue
from packed_bed_ui.worker import activate_snapshot


@pytest.fixture
def workspace_project(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case, "Benchmark")
    case.documents["program"]["inlet_temperature"]["steps"] = [{"kind": "hold", "duration_s": .01}]
    case.save()
    job = read_json(project.prepare_execution([case]))
    folder = activate_snapshot(case.root / f".pending-{job['attempt_id']}")
    write_json(folder / "status.json", {"state": "completed"})
    study = project.study_store.create("Sweep", case)
    return project, case, study


def finish_preview(widget, qt_app):
    widget.begin_preview()
    deadline = monotonic() + 10
    while widget.preview_timer.isActive() and monotonic() < deadline:
        qt_app.processEvents()
    assert not widget.preview_timer.isActive()
    assert widget.preview is not None, widget.issue.text()


def test_baseline_inspection_is_readonly_without_normalization(qt_app, workspace_project):
    from PyQt6.QtWidgets import QAbstractItemView
    from packed_bed_ui.editor import InputEditor
    project, base, study = workspace_project
    # Deliberately non-normalized values must be displayed faithfully, including invalid drafts.
    documents = deepcopy(study.baseline)
    documents["solids"]["initial_profile"]["zones"][0]["x_end_m"] = .5
    documents["run"]["simulation"]["time_horizon_s"] = .25
    before = deepcopy(documents)
    editor = InputEditor()
    editor.set_documents(documents, study.editor_metadata, read_only=True)
    editor.put(("run", "model", "bed_length_m"), 10)
    editor.set_species("gas", ["O2"])
    editor.save()
    assert editor.case.documents == before == documents
    assert editor.general.horizon.text() == "0.25"
    assert editor.bed.zones.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    assert editor.bed.length.isReadOnly()
    assert not editor.dirty
    assert not editor.debounce.isActive()
    assert base.state()["stale"] is False
    editor.close()


def test_workspace_preview_and_complete_rebuild(qt_app, workspace_project):
    from packed_bed_ui.study_editor import StudyEditor
    project, base, study = workspace_project
    editor = StudyEditor()
    editor.set_study(project, study)
    draft = deepcopy(editor.study)
    draft.factors = [Factor("length", "bed_length_m", [.4, .6, .8])]
    editor.change(draft, "Add lengths")
    finish_preview(editor, qt_app)
    assert editor.preview.candidates[0].inputs == "Ready"
    assert editor.apply_button.text() == "Create 3 cases"
    assert len(project.cases) == 1
    editor.apply_button.click()
    finish_preview(editor, qt_app)
    first_ids = {case.id for case in project.cases if case.id != base.id}
    assert len(first_ids) == 3
    assert not editor.apply_button.isEnabled()
    draft = deepcopy(editor.study)
    draft.factors[0].values = [.4, 1.2]
    editor.change(draft, "Change lengths")
    finish_preview(editor, qt_app)
    assert "delete all 3" in editor.replacement_note.text()
    assert editor.apply_button.text() == "Replace 3 cases with 2"
    editor.apply_button.click()
    assert len(project.cases) == 3
    assert not first_ids & {case.id for case in project.cases}
    editor.clear()
    editor.close()


def test_explicit_rows_paste_and_undo_are_one_operation(qt_app, workspace_project):
    from packed_bed_ui.study_editor import StudyEditor
    project, _, study = workspace_project
    study.factors = [Factor("length", "bed_length_m", [.4, .8]), Factor("cells", "axial_cells", [3])]
    project.study_store.save_study(study)
    editor = StudyEditor()
    editor.set_study(project, study)
    editor.change_mode(1)
    assert len(editor.study.rows) == 2
    before = deepcopy(editor.study.rows)
    count = editor.undo.count()
    editor.row_model.paste(0, 0, [[".5", "4"], ["1.0", "8"]])
    assert editor.undo.count() == count + 1
    finish_preview(editor, qt_app)
    assert len(editor.preview.candidates) == 2
    assert [c.documents["run"]["model"]["axial_cells"] for c in editor.preview.candidates] == [4, 8]
    editor.undo.undo()
    assert editor.study.rows == before
    editor.undo.redo()
    assert editor.study.rows[0] == {"length": ".5", "cells": "4"}
    editor.clear()
    editor.close()


def test_spreadsheet_tab_keeps_cell_navigation(qt_app, workspace_project):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLineEdit
    from packed_bed_ui.study_editor import StudyEditor
    project, _, study = workspace_project
    study.factors = [Factor("length", "bed_length_m"), Factor("cells", "axial_cells")]
    study.mode, study.rows = "rows", [{"length": .5, "cells": 3}]
    project.study_store.save_study(study)
    editor = StudyEditor()
    editor.resize(1100, 800)
    editor.set_study(project, study)
    editor.show()
    qt_app.processEvents()
    index = editor.row_model.index(0, 0)
    editor.rows.setCurrentIndex(index)
    editor.rows.edit(index)
    control = editor.rows.findChild(QLineEdit)
    assert control is not None
    control.selectAll()
    QTest.keyClicks(control, "0.6")
    QTest.keyClick(control, Qt.Key.Key_Tab)
    qt_app.processEvents()
    assert editor.study.rows[0]["length"] == "0.6"
    assert editor.rows.currentIndex().column() == 1
    control = qt_app.focusWidget()
    assert isinstance(control, QLineEdit)
    control.selectAll()
    QTest.keyClicks(control, "7")
    editor.finish_cell_edit()
    assert editor.study.rows[0]["cells"] == "7"
    assert editor.save()
    assert project.study_store.studies[study.id].rows[0]["cells"] == "7"
    editor.clear()
    editor.close()


def test_definition_dialog_saves_owned_inputs_and_cancel_changes_nothing(qt_app, workspace_project):
    from packed_bed_ui.definition_editor import DefinitionDialog
    project, base, study = workspace_project
    original = deepcopy(base.documents)
    dialog = DefinitionDialog(project.study_store, "bed", study.baseline)
    dialog.name.setText("Narrow bed")
    dialog.editor.fields[("model", "bed_radius_m")].setText("0.005")
    dialog.save_definition()
    assert dialog.definition.payload["model"]["bed_radius_m"] == .005
    assert base.documents == original
    again = DefinitionDialog(project.study_store, "bed", study.baseline, dialog.definition)
    again.editor.fields[("model", "bed_radius_m")].setText("20")
    again.reject()
    assert project.study_store.definitions[dialog.definition.id].payload["model"]["bed_radius_m"] == .005
    program = DefinitionDialog(project.study_store, "program", study.baseline)
    program.save_definition()
    assert program.definition.payload["simulation"]["program_mode"] == "separate_channels"
    assert base.documents == original


@pytest.mark.parametrize("empty", [False, True])
def test_bed_definition_edits_and_retains_all_bed_settings(qt_app, workspace_project, empty):
    from packed_bed_ui.definition_editor import DefinitionDialog
    project, base, study = workspace_project
    original = deepcopy(base.documents)
    dialog = DefinitionDialog(project.study_store, "bed", study.baseline)
    if empty:
        dialog.start.setCurrentIndex(1)
    fields = dialog.editor.fields
    temperature = fields[("model", "ambient_temperature_k")]
    coefficient = fields[("model", "heat_transfer_coefficient_w_per_m2_k")]
    voidage = fields[("model", "gas_voidage_mode")]
    reversible = fields[("simulation", "interior_flow_mode")]
    assert all(control.isEnabled() for control in (temperature, coefficient, voidage, reversible))
    temperature.setText("650")
    coefficient.setText("42")
    voidage.setCurrentIndex(voidage.findData("bed_only"))
    reversible.setChecked(True)
    dialog.save_definition()
    reopened = Project.open(project.root)
    definition = reopened.study_store.definitions[dialog.definition.id]
    assert definition.payload["model"]["ambient_temperature_k"] == 650
    assert definition.payload["model"]["heat_transfer_coefficient_w_per_m2_k"] == 42
    assert definition.payload["model"]["gas_voidage_mode"] == "bed_only"
    assert definition.payload["simulation"] == {"interior_flow_mode": "reversible"}
    assert "axial_cells" not in definition.payload["model"]
    again = DefinitionDialog(reopened.study_store, "bed", study.baseline, definition)
    assert again.editor.fields[("model", "ambient_temperature_k")].text() == "650"
    assert again.editor.fields[("simulation", "interior_flow_mode")].isChecked()
    again.editor.fields[("model", "ambient_temperature_k")].setText("700")
    again.editor.fields[("simulation", "interior_flow_mode")].setChecked(False)
    again.reject()
    assert reopened.study_store.definitions[definition.id] == definition
    assert base.documents == original


def test_project_navigation_and_readonly_generated_case(qt_app, workspace_project):
    from packed_bed_ui.window import MainWindow
    project, base, study = workspace_project
    window = MainWindow()
    window._set_project(project)
    assert window.table.groups[study.id].childCount() == 0
    window._case_action("Study", study.id)
    assert window.pages.currentWidget() is window.study_editor
    draft = deepcopy(window.study_editor.study)
    draft.factors = [Factor("cells", "axial_cells", [3])]
    window.study_editor.change(draft, "Add variation")
    finish_preview(window.study_editor, qt_app)
    window.study_editor.apply()
    generated = project.cases[-1]
    window._show_case(generated)
    assert window.editor.read_only
    assert not window.edit_study_button.isHidden()
    assert not window.independent_button.isHidden()
    assert not generated.run_folder.exists()
    window._show_case(base)
    assert not window.editor.read_only
    assert not window.editor.bed.length.isReadOnly()
    window.close()


def test_preview_cancellation_does_not_create_cases(qt_app, workspace_project):
    from packed_bed_ui.study_editor import StudyEditor
    project, _, study = workspace_project
    study.factors = [Factor("cells", "axial_cells", list(range(3, 203)))]
    project.study_store.save_study(study)
    editor = StudyEditor()
    editor.set_study(project, study)
    editor.preview_batch()
    assert len(editor.candidate_model.candidates) < 200
    editor.stop_preview()
    qt_app.processEvents()
    assert not editor.preview_timer.isActive() and editor.preview is None
    assert len(project.cases) == 1
    assert not editor.apply_button.isEnabled()
    editor.clear()
    editor.close()


def test_definition_selection_and_library_share_save_cancel_and_duplicate(qt_app, workspace_project, monkeypatch):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QDialog
    from packed_bed_ui.definition_editor import DefinitionDialog, DefinitionList
    project, _, study = workspace_project
    picker = DefinitionList(project.study_store, study.baseline, kind="bed", selected=[])

    def save_dialog(dialog):
        dialog.name.setText("Chosen bed")
        dialog.save_definition()
        return dialog.result()

    monkeypatch.setattr(DefinitionDialog, "exec", save_dialog)
    picker.manage("New")
    original, = picker.checked()
    picker.items.setCurrentRow(0)
    picker.manage("Duplicate")
    assert len(picker.checked()) == len(project.study_store.definitions) == 2
    assert original in picker.checked()
    assert len(set(picker.checked())) == 2

    monkeypatch.setattr(DefinitionDialog, "exec", lambda dialog: QDialog.DialogCode.Rejected)
    picker.items.setCurrentRow(0)
    picker.manage("New")
    assert len(project.study_store.definitions) == 2
    picker.items.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert original not in picker.checked()
    assert original in project.study_store.definitions

    library = DefinitionList(project.study_store, study.baseline, kind="bed")
    library.items.setCurrentRow(0)
    library.manage("Delete")
    assert len(project.study_store.definitions) == 1
    picker.close()
    library.close()


def test_add_and_edit_variations_use_the_same_dialog(qt_app, workspace_project, monkeypatch):
    from PyQt6.QtWidgets import QDialog
    from packed_bed_ui import study_editor
    project, _, study = workspace_project
    editor = study_editor.StudyEditor()
    editor.set_study(project, study)
    targets = []

    def choose_target(*args):
        targets.append("bed_length_m")
        return targets[-1]

    def edit_values(dialog):
        dialog.values.setPlainText(".4, .8")
        dialog.accept_values()
        return dialog.result()

    monkeypatch.setattr(study_editor, "choose_parameter", choose_target)
    monkeypatch.setattr(study_editor.FactorDialog, "exec", edit_values)
    editor.add_factor()
    editor.edit_factor(0)
    assert targets == ["bed_length_m"]
    assert len(editor.study.factors) == 1
    finish_preview(editor, qt_app)
    assert len(editor.preview.candidates) == 2
    assert len(project.cases) == 1

    monkeypatch.setattr(study_editor.FactorDialog, "exec", lambda dialog: QDialog.DialogCode.Rejected)
    editor.add_factor()
    assert len(editor.study.factors) == 1
    editor.clear()
    editor.close()
