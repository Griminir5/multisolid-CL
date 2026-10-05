"""Field feedback follows the scientific rules and never rewrites a draft."""

from copy import deepcopy

import pytest

from packed_bed.config.load import input_field_issues
from packed_bed_ui.project import Project


@pytest.fixture
def editing(qt_app, tmp_path, source_case):
    from packed_bed_ui.editor import CaseEditor
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    case.documents["run"]["simulation"].update(time_horizon_s=1.0, repeat_program=True)
    for channel in case.documents["program"].values():
        channel["steps"] = [{"kind": "hold", "duration_s": 1.0}]
    case.save()
    editor = CaseEditor()
    editor.set_case(case)
    yield editor, case
    editor.field_validation.stop()
    editor.debounce.stop()
    editor.close()
    editor.deleteLater()
    qt_app.processEvents()


def test_simultaneous_errors_clear_without_mutating_drafts(editing):
    from packed_bed_ui.validation import ISSUE_ROLE
    editor, case = editing
    radius = editor.fields[("model", "bed_radius_m")]
    tolerance = editor.fields[("solver", "relative_tolerance")]
    radius.setText("")
    tolerance.setText("-1")
    editor.bed.zones.item(0, 2).setText("1.2")
    editor.program.channels["inlet_flow"].table.item(0, 2).setText("wrong")
    before = deepcopy(case.documents)
    assert editor.save()
    assert case.documents == before
    assert radius.property("invalidInput")
    assert "number" in radius.toolTip()
    assert tolerance.property("invalidInput")
    assert editor.bed.zones.item(0, 2).data(ISSUE_ROLE)
    assert editor.program.channels["inlet_flow"].table.item(0, 2).data(ISSUE_ROLE)
    assert not editor.tabs.tabIcon(0).isNull()
    assert not editor.tabs.tabIcon(2).isNull()
    assert not editor.tabs.tabIcon(3).isNull()
    assert editor.tabs.tabIcon(1).isNull()
    radius.setText("0.01")
    tolerance.setText("1e-5")
    editor.bed.zones.item(0, 2).setText("0.4")
    editor.program.channels["inlet_flow"].table.item(0, 2).setText("1e-8")
    editor.save()
    assert not radius.property("invalidInput")
    assert radius.toolTip() == ""
    assert not tolerance.property("invalidInput")
    assert not editor.bed.zones.item(0, 2).data(ISSUE_ROLE)
    assert all(editor.tabs.tabIcon(i).isNull() for i in range(4))
    assert not editor.dirty


def test_feedback_while_typing_preserves_focus_and_tooltips(editing):
    from PyQt6.QtTest import QTest
    editor, case = editing
    editor.show()
    field = editor.general.horizon
    original_tip = field.toolTip()
    field.setFocus()
    field.selectAll()
    QTest.keyClicks(field, "-2")
    QTest.qWait(250)
    assert field.property("invalidInput")
    assert field.hasFocus()
    assert field.text() == "-2"
    assert original_tip in field.toolTip()
    field.setText("2")
    QTest.qWait(250)
    assert not field.property("invalidInput")
    assert field.toolTip() == original_tip


def test_ramp_errors_point_to_cells_and_survive_row_rebuild(editing):
    from packed_bed_ui.validation import ISSUE_ROLE
    editor, case = editing
    channel = editor.program.channels["inlet_temperature"]
    channel.change_kind(0, "ramp")
    channel.table.item(1, 1).setText("0")
    editor.save()
    assert channel.table.item(1, 1).data(ISSUE_ROLE)
    assert channel.table.item(1, 2).data(ISSUE_ROLE)
    assert not channel.table.cellWidget(1, 0).property("invalidInput")
    channel.add_step()
    editor.save()
    assert channel.table.item(2, 1).data(ISSUE_ROLE)
    channel.remove_step(0)
    channel.table.item(1, 1).setText("2")
    editor.save()
    assert not channel.table.item(1, 1).data(ISSUE_ROLE)
    assert not channel.table.item(1, 2).data(ISSUE_ROLE)
    assert not channel.toggle.property("invalidInput")


def test_table_validation_does_not_interrupt_typing(editing):
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLineEdit
    from packed_bed_ui.validation import ISSUE_ROLE
    editor, case = editing
    editor.show()
    editor.tabs.setCurrentWidget(editor.program)
    table = editor.program.channels["inlet_flow"].table
    item = table.item(0, 2)
    table.editItem(item)
    field = table.findChild(QLineEdit)
    field.selectAll()
    QTest.keyClicks(field, "1e-")
    QTest.qWait(250)
    assert item.data(ISSUE_ROLE)
    assert field.text() == "1e-"
    assert field.cursorPosition() == 3
    QTest.keyClicks(field, "8")
    QTest.qWait(250)
    assert not item.data(ISSUE_ROLE)
    assert case.documents["program"]["inlet_flow"]["initial"] == 1e-8


def test_related_fields_and_zone_gaps(editing):
    from packed_bed_ui.validation import ISSUE_ROLE
    editor, case = editing
    interval = editor.fields[("simulation", "reporting_interval_s")]
    interval.setText("2")
    editor.save()
    assert interval.property("invalidInput")
    assert editor.general.horizon.property("invalidInput")
    assert not editor.fields[("solver", "relative_tolerance")].property("invalidInput")
    interval.setText("0.1")
    zone = deepcopy(case.documents["solids"]["initial_profile"]["zones"][0])
    case.documents["solids"]["initial_profile"]["zones"][0]["x_end_m"] = 0.4
    zone["x_start_m"] = 0.5
    case.documents["solids"]["initial_profile"]["zones"].append(zone)
    editor.bed.load_zones()
    editor.refresh()
    assert editor.bed.zones.item(0, 1).data(ISSUE_ROLE)
    assert editor.bed.zones.item(1, 0).data(ISSUE_ROLE)
    assert not editor.bed.zones.item(0, 2).data(ISSUE_ROLE)


def test_channel_timing_marks_durations(editing):
    from packed_bed_ui.validation import ISSUE_ROLE
    editor, case = editing
    editor.program.repeat.setChecked(False)
    editor.program.channels["inlet_flow"].table.item(1, 1).setText("2")
    editor.save()
    temperature = editor.program.channels["inlet_temperature"]
    assert "must sum" in temperature.table.item(1, 1).data(ISSUE_ROLE)
    assert not temperature.table.item(0, 2).data(ISSUE_ROLE)
    assert not editor.program.channels["inlet_flow"].table.item(1, 1).data(ISSUE_ROLE)


def test_empty_draft_marks_required_inputs(qt_app, tmp_path):
    from packed_bed_ui.editor import CaseEditor
    project = Project.create(tmp_path / "project")
    case = project.add_case("Draft")
    editor = CaseEditor()
    editor.set_case(case)
    assert editor.chemistry.gas_list.table.property("invalidInput")
    assert editor.chemistry.solid_list.table.property("invalidInput")
    assert editor.bed.add_button.property("invalidInput")
    assert editor.bed.zones.property("invalidInput")
    assert editor.program.channels["feed_stream"].table.cellWidget(0, 2).property("invalidInput")
    assert all(not editor.tabs.tabIcon(i).isNull() for i in range(4))
    assert not editor.dirty
    editor.deleteLater()
    qt_app.processEvents()


@pytest.mark.parametrize("dark", [False, True])
def test_empty_zone_highlight_is_red_and_clears_on_add(editing, qt_app, dark):
    from PyQt6.QtGui import QColor
    from packed_bed_ui.theme import manager, colors

    editor, case = editing
    manager().apply(dark)
    editor.show()
    editor.tabs.setCurrentWidget(editor.bed)
    editor.bed.remove_zone(0)
    qt_app.processEvents()
    assert editor.bed.zones.property("invalidInput")
    button = editor.bed.add_button
    assert button.property("invalidInput")
    # Check rendered paint, not only the property: table-action styling used to
    # override the error fill even though invalidInput was correctly set.
    picture = button.grab().toImage()
    assert picture.pixelColor(picture.width() - 12, picture.height() // 2) == QColor(colors()["missing"])
    assert "At least one solid zone" in button.toolTip()
    editor.save()
    assert not editor.tabs.tabIcon(2).isNull()

    editor.bed.add_button.click()
    assert len(case.documents["solids"]["initial_profile"]["zones"]) == 1
    assert not editor.bed.zones.property("invalidInput")
    assert not editor.bed.add_button.property("invalidInput")
    editor.save()
    assert "At least one solid zone" not in editor.tabs.tabToolTip(2)
    # Removing the final zone brings the highlight back immediately.
    editor.bed.remove_zone(0)
    assert editor.bed.zones.property("invalidInput")
    assert editor.bed.add_button.property("invalidInput")


@pytest.mark.parametrize("mode,key", [("separate_channels", "inlet_composition"), ("feed_stream", "feed_stream")])
def test_composition_dialog_highlights_and_clears_group_error(editing, monkeypatch, mode, key):
    from PyQt6.QtWidgets import QDialog, QLineEdit
    editor, case = editing
    editor.program.mode.setCurrentIndex(editor.program.mode.findData(mode))
    before = deepcopy(case.documents)

    def inspect(dialog):
        fraction = dialog.findChild(QLineEdit, "N2")
        fraction.setText("0.5")
        assert fraction.property("invalidInput")
        assert "sum to 1" in fraction.toolTip()
        fraction.setText("bad")
        assert fraction.property("invalidInput")
        fraction.setText("1")
        assert not fraction.property("invalidInput")
        if key == "feed_stream":
            flow = dialog.findChild(QLineEdit, "flow")
            flow.setText("0")
            assert flow.property("invalidInput")
            assert not fraction.property("invalidInput")
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    editor.program.channels[key].edit_state(0)
    assert case.documents == before


def test_optional_feed_values_are_not_required(editing, monkeypatch):
    from PyQt6.QtWidgets import QCheckBox, QDialog, QLineEdit
    editor, case = editing
    editor.program.mode.setCurrentIndex(editor.program.mode.findData("feed_stream"))
    channel = editor.program.channels["feed_stream"]
    channel.add_step()
    channel.change_kind(0, "ramp")

    def inspect(dialog):
        flow = dialog.findChild(QLineEdit, "flow")
        fraction = dialog.findChild(QLineEdit, "N2")
        assert not flow.isEnabled()
        assert not flow.property("invalidInput")
        assert not fraction.property("invalidInput")
        check = next(c for c in dialog.findChildren(QCheckBox) if c.text().startswith("Flow"))
        assert check.property("invalidInput")  # An entirely empty ramp needs at least one target.
        check.setChecked(True)
        assert flow.property("invalidInput")
        flow.setText("1e-8")
        assert not flow.property("invalidInput")
        assert not fraction.property("invalidInput")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", inspect)
    channel.edit_state(1)
    assert case.documents["program"]["feed_stream"]["steps"][0]["target"] == {"flow": 1e-8}


def test_theme_changes_and_case_switch_do_not_keep_old_errors(editing):
    from packed_bed_ui.theme import manager
    editor, case = editing
    original = deepcopy(case.documents)
    editor.fields[("solver", "relative_tolerance")].setText("")
    editor.save()
    for dark in (True, False):
        manager().apply(dark)
        assert editor.fields[("solver", "relative_tolerance")].property("invalidInput")
        assert not editor.dirty
    clean = case.project.add_case("Clean", original)
    editor.set_case(clean)
    assert all(editor.tabs.tabIcon(i).isNull() for i in range(4))
    assert not editor.fields[("solver", "relative_tolerance")].property("invalidInput")
    assert editor.case.documents == original
    from packed_bed_ui.editor import InputEditor
    inspection = InputEditor()
    inspection.set_documents(case.documents, read_only=True)
    assert inspection.fields[("solver", "relative_tolerance")].property("invalidInput")
    assert inspection.case.documents == case.documents
    inspection.deleteLater()


def test_structured_locations_include_advanced_and_composition_errors(editing):
    editor, case = editing
    case.documents["run"]["solver"]["nonlinear_convergence_coefficient"] = -1.0
    case.documents["program"]["inlet_composition"]["steps"] = [
        {"kind": "ramp", "duration_s": 1.0, "target": {"N2": 0.5}}]
    paths = [path for issue in input_field_issues(case.documents) for path in issue.paths]
    assert ("run", "solver", "nonlinear_convergence_coefficient") in paths
    assert ("program", "inlet_composition", "steps", 0, "target") in paths
    editor.refresh()
    assert editor.general.advanced_button.property("invalidInput")
