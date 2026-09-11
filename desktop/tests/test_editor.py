"""Case-authoring interactions and scientific input round trips, without a solver."""

from copy import deepcopy

import pytest

from packed_bed_ui.project import Project, write_json


@pytest.fixture
def editing(qt_app, tmp_path, source_case):
    from packed_bed_ui.editor import CaseEditor

    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    case.documents["run"]["simulation"]["time_horizon_s"] = 1.0
    for channel in case.documents["program"].values():
        channel["steps"] = [{"kind": "hold", "duration_s": 1.0}]
    case.save()
    editor = CaseEditor()
    assert editor.set_case(case)
    yield editor, case
    editor.save()
    editor.close()
    editor.deleteLater()
    qt_app.processEvents()


def test_five_tabs_and_toolbar(qt_app, tmp_path):
    from PyQt6.QtWidgets import QPushButton
    from packed_bed_ui.window import MainWindow

    window = MainWindow()
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files("packed_bed/examples/default_case/run.yaml", "Benchmark")
    window._set_project(project)
    window._show_case(case)
    assert [window.editor.tabs.tabText(i) for i in range(5)] == ["General", "Chemistry", "Bed", "Program", "Results"]
    buttons = [window.case_page.layout().itemAt(0).layout().itemAt(i).widget()
               for i in range(window.case_page.layout().itemAt(0).layout().count())]
    assert [button.text() for button in buttons if isinstance(button, QPushButton)] == [
        "← Back to project", "Open case folder", "Open latest run log",
    ]
    assert window.case_result.text() == "Not run"
    assert "Benchmark" in window.windowTitle()
    assert not window.editor.dirty  # Browsing the cyclic example does not rewrite inputs.
    window.close()


def test_general_fields_persist_and_preserve_advanced_settings(editing):
    editor, case = editing
    case.documents["run"]["solver"].update(maximum_order=3, concentration_absolute_tolerance=2e-8)
    editor.fields[("simulation", "reporting_interval_s")].setText("0.2")
    editor.fields[("model", "axial_cells")].setValue(9)
    editor.fields[("solver", "threads")].setValue(4)
    editor.fields[("solver", "relative_tolerance")].setText("1e-6")
    editor.fields[("solver", "suppress_algebraic_errors")].setChecked(True)
    combo = editor.fields[("simulation", "mass_scheme")]
    combo.setCurrentIndex(combo.findData("upwind1"))
    assert editor.save()
    resolved = Project.open(case.project.root).cases[0].resolve()
    assert resolved.run.simulation.reporting_interval_s == .2
    assert resolved.run.simulation.mass_scheme == "upwind1"
    assert resolved.run.model.axial_cells == 9
    assert resolved.run.solver.threads == 4
    assert resolved.run.solver.relative_tolerance == 1e-6
    assert resolved.run.solver.suppress_algebraic_errors
    assert resolved.run.solver.maximum_order == 3
    assert resolved.run.solver.concentration_absolute_tolerance == 2e-8
    assert not editor.general.backend.model().item(editor.general.backend.findData("compiled")).isEnabled()


def test_advanced_solver_dialog_applies_only_edits(editing, qt_app):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QDialog, QLineEdit, QSpinBox

    editor, case = editing
    original = deepcopy(case.documents["run"]["solver"])
    def accept():
        dialog = qt_app.activeModalWidget()
        assert isinstance(dialog, QDialog)
        dialog.findChild(QLineEdit, "concentration_absolute_tolerance").setText("2e-7")
        dialog.findChild(QSpinBox, "maximum_order").setValue(3)
        dialog.accept()
    QTimer.singleShot(0, accept)
    editor.general.advanced()
    editor.save()
    solver = Project.open(case.project.root).cases[0].documents["run"]["solver"]
    assert solver == {**original, "concentration_absolute_tolerance": 2e-7, "maximum_order": 3}


def test_horizon_tracks_timing_and_repeat_mode(editing):
    editor, case = editing
    horizon = editor.general.horizon
    assert not horizon.isEnabled()
    assert horizon.text() == "1"
    flow = editor.program.channels["inlet_flow"]
    flow.table.item(1, 1).setText("2.5")
    assert horizon.text() == "2.5"
    assert "must match" in editor.program.timing.text()
    for key in ("inlet_temperature", "inlet_composition", "outlet_pressure"):
        editor.program.channels[key].table.item(1, 1).setText("2.5")
    assert editor.save()
    assert case.resolve().run.simulation.time_horizon_s == 2.5
    flow.table.item(1, 1).setText("unfinished")
    assert horizon.text() == ""
    assert editor.save()
    assert "cannot run" in editor.validation.text()
    editor.program.repeat.setChecked(True)
    assert horizon.isEnabled()
    horizon.setText("8")
    flow.table.item(1, 1).setText("2")
    assert editor.save()
    assert case.resolve().run.simulation.time_horizon_s == 8
    editor.program.repeat.setChecked(False)
    assert not horizon.isEnabled()
    assert horizon.text() == "2.5"


def test_zero_step_program_has_zero_derived_horizon(editing):
    editor, case = editing
    for key in ("inlet_flow", "inlet_temperature", "inlet_composition", "outlet_pressure"):
        editor.program.channels[key].remove_step(0)
    assert editor.general.horizon.text() == "0"
    assert "Add a timed hold" in editor.program.timing.text()
    assert editor.save()
    assert case.state()["inputs"] == "Invalid"


def test_report_selection_dialog_and_removal_persist(editing, qt_app):
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtWidgets import QListWidget

    editor, case = editing
    def choose():
        dialog = qt_app.activeModalWidget()
        items = dialog.findChild(QListWidget)
        for i in range(items.count()):
            item = items.item(i)
            if item.data(Qt.ItemDataRole.UserRole) in ("velocity", "gas_flux"):
                item.setCheckState(Qt.CheckState.Checked)
        dialog.accept()
    QTimer.singleShot(0, choose)
    editor.general.reports.add()
    editor.general.reports.remove("pressure")
    assert editor.save()
    reports = Project.open(case.project.root).cases[0].documents["run"]["outputs"]["requested_reports"]
    assert set(reports) == {"temperature", "gas_mole_fraction", "velocity", "gas_flux"}
    # The add row must become a normal label after adding the first item.
    editor.general.plots.set_values([])
    editor.general.plots.set_values(["outlet_composition"])
    assert editor.general.plots.table.cellWidget(0, 0) is None
    assert not editor.general.plots.show_buttons[0].isEnabled()


def test_species_and_families_update_graph_and_dependent_drafts(editing, monkeypatch):
    from PyQt6.QtCore import Qt
    from packed_bed.kinetics import FAMILY_REGISTRY

    editor, case = editing
    monkeypatch.setattr("packed_bed_ui.chemistry.choose_items", lambda *_: ["nickel_medrano"])
    editor.chemistry.add_families()
    family = FAMILY_REGISTRY["nickel_medrano"]
    editor.chemistry.add_requirements(family)
    assert set(family.required_gas_species) <= set(case.documents["chemistry"]["gas_species"])
    assert "" in case.documents["program"]["inlet_composition"]["initial"].values()
    assert case.documents["solids"]["initial_profile"]["zones"][0]["values"]["NiO"] == ""
    root = editor.chemistry.families.topLevelItem(0)
    first_reaction = root.child(0).data(0, Qt.ItemDataRole.UserRole)
    root.child(0).setCheckState(0, Qt.CheckState.Unchecked)
    assert first_reaction not in case.documents["chemistry"]["reaction_ids"]
    assert editor.chemistry.graph.scene().items()
    editor.chemistry.remove_family("nickel_medrano")
    assert not case.documents["chemistry"]["reaction_ids"]
    editor.set_species("gas", ["N2"])
    editor.set_species("solid", ["Ni"])
    assert editor.save()
    reopened = Project.open(case.project.root).cases[0]
    assert reopened.resolve().chemistry.gas_species == ("N2",)
    assert reopened.documents["program"]["inlet_composition"]["initial"] == {"N2": 1.0}


def test_zones_add_edit_delete_and_anchor_to_length(editing):
    from PyQt6.QtCore import Qt

    editor, case = editing
    editor.bed.add_zone()
    zones = case.documents["solids"]["initial_profile"]["zones"]
    assert [(zone["x_start_m"], zone["x_end_m"]) for zone in zones] == [(0, .5), (.5, 1)]
    assert zones[1]["values"]["Ni"] == ""
    assert not editor.bed.zones.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable
    assert not editor.bed.zones.item(1, 1).flags() & Qt.ItemFlag.ItemIsEditable
    for col, value in ((2, ".4"), (3, ".5"), (4, ".002"), (5, "4")):
        editor.bed.zones.item(1, col).setText(value)
    assert editor.save()
    assert case.resolve().solids.initial_profile.zones[1].values["Ni"] == 4
    editor.bed.remove_zone(0)
    editor.bed.length.setText("2")
    editor.bed.resize_zones()
    assert editor.save()
    resolved = Project.open(case.project.root).cases[0].resolve()
    assert resolved.solids.initial_profile.zones[0].x_start_m == 0
    assert resolved.solids.initial_profile.zones[0].x_end_m == 2
    assert resolved.run.model.bed_length_m == 2


def test_length_change_can_scale_internal_boundaries(editing, qt_app):
    from PyQt6.QtCore import QTimer

    editor, case = editing
    editor.bed.add_zone()
    editor.bed.length.setText("3")
    def scale():
        dialog = qt_app.activeModalWidget()
        next(button for button in dialog.buttons() if button.text() == "Scale proportionally").click()
    QTimer.singleShot(0, scale)
    editor.bed.resize_zones()
    assert [(zone["x_start_m"], zone["x_end_m"]) for zone in case.documents["solids"]["initial_profile"]["zones"]] == [(0, 1.5), (1.5, 3)]


def test_hold_ramp_targets_and_duration_round_trip(editing):
    editor, case = editing
    flow = editor.program.channels["inlet_flow"]
    flow.change_kind(0, "ramp")
    assert flow.channel()["steps"][0]["target"] == ""
    flow.table.item(1, 2).setText("2e-8")
    editor.save()
    assert case.resolve().program.inlet_flow.steps[0].target == 2e-8
    flow.change_kind(0, "hold")
    editor.save()
    assert case.documents["program"]["inlet_flow"]["steps"] == [{"kind": "hold", "duration_s": 1.0}]


def test_modes_preserve_steps_and_shared_outlet_across_reopen(editing, qt_app):
    editor, case = editing
    original = deepcopy(case.documents["program"])
    editor.resize(1100, 760)
    editor.tabs.setCurrentWidget(editor.program)
    editor.show()
    qt_app.processEvents()
    pressure_geometry = editor.program.channels["outlet_pressure"].geometry()
    editor.program.mode.setCurrentIndex(editor.program.mode.findData("feed_stream"))
    qt_app.processEvents()
    assert editor.program.channels["outlet_pressure"].geometry() == pressure_geometry
    assert set(case.documents["program"]) == {"feed_stream", "outlet_pressure"}
    assert case.documents["program"]["feed_stream"]["initial"]["composition"] == {"N2": 1}
    editor.program.channels["outlet_pressure"].table.item(0, 2).setText("110000")
    editor.save()
    reopened = Project.open(case.project.root).cases[0]
    editor.set_case(reopened)
    editor.program.mode.setCurrentIndex(editor.program.mode.findData("separate_channels"))
    assert case.documents["program"]["outlet_pressure"]["initial"] == 110000
    for key in ("inlet_flow", "inlet_temperature", "inlet_composition"):
        assert reopened.documents["program"][key] == original[key]
    assert reopened.documents["program"]["outlet_pressure"]["initial"] == 110000


def test_flow_basis_converts_physical_flow_in_both_modes(editing):
    editor, case = editing
    for mode in ("separate_channels", "feed_stream"):
        editor.program.mode.setCurrentIndex(editor.program.mode.findData(mode))
        flow = case.resolve().inlet_flow_program.value_at(0, smooth_ramp_width_s=1)
        editor.program.flow_basis.setCurrentIndex(editor.program.flow_basis.findData("ghsv_per_h"))
        assert case.resolve().inlet_flow_program.value_at(0, smooth_ramp_width_s=1) == pytest.approx(flow)
        editor.program.flow_basis.setCurrentIndex(editor.program.flow_basis.findData("mol_per_s"))
        assert case.resolve().inlet_flow_program.value_at(0, smooth_ramp_width_s=1) == pytest.approx(flow)


def test_feed_partial_target_survives_editing_other_fields(qt_app, tmp_path):
    from packed_bed_ui.editor import CaseEditor

    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files("packed_bed/examples/default_case/run_feed_stream.yaml")
    editor = CaseEditor()
    editor.set_case(case)
    original = deepcopy(case.documents["program"])
    editor.fields[("solver", "threads")].setValue(2)
    editor.save()
    assert Project.open(project.root).cases[0].documents["program"] == original
    editor.close()


def test_new_empty_draft_can_be_completed_through_controls(qt_app, tmp_path):
    from packed_bed_ui.editor import CaseEditor

    project = Project.create(tmp_path / "project")
    case = project.add_case("Draft")
    editor = CaseEditor()
    editor.set_case(case)
    editor.fields[("simulation", "reporting_interval_s")].setText("1")
    editor.bed.length.setText("1")
    editor.fields[("model", "bed_radius_m")].setText(".01")
    editor.set_species("gas", ["N2"])
    editor.set_species("solid", ["Ni"])
    editor.bed.add_zone()
    for col, value in ((2, ".4"), (3, ".5"), (4, ".001"), (5, "1")):
        editor.bed.zones.item(0, col).setText(value)
    for key, value in (("inlet_flow", "1e-8"), ("inlet_temperature", "300"), ("outlet_pressure", "100000")):
        channel = editor.program.channels[key]
        channel.table.item(0, 2).setText(value)
    # Composition fields are edited through the same modal used for a feed target.
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QLineEdit
    def composition():
        dialog = qt_app.activeModalWidget()
        dialog.findChild(QLineEdit, "N2").setText("1")
        dialog.accept()
    QTimer.singleShot(0, composition)
    editor.program.channels["inlet_composition"].edit_state(0)
    channel = editor.program.channels["inlet_flow"]
    channel.add_step()
    channel.table.item(1, 1).setText("10")
    assert editor.save()
    assert Project.open(project.root).cases[0].state()["inputs"] == "Ready"
    editor.close()


def test_plot_opens_from_retained_data_even_with_invalid_current_inputs(editing, qt_app):
    import xarray as xr
    from PyQt6.QtSvgWidgets import QSvgWidget

    editor, case = editing
    folder = case.run_folder / "output"
    folder.mkdir(parents=True)
    write_json(case.run_folder / "snapshot.json", {"fingerprint": case.fingerprint()})
    write_json(case.run_folder / "status.json", {"state": "completed"})
    dataset = xr.Dataset({
        "inlet_composition": (("time", "gas_species"), [[1.0], [1.0]]),
        "outlet_composition": (("time", "gas_species"), [[1.0], [1.0]]),
    }, coords={"time": [0.0, 1.0], "gas_species": ["N2"]})
    path = folder / "results.nc"
    dataset.to_netcdf(path, engine="scipy")
    original = path.read_bytes()
    editor.fields[("simulation", "reporting_interval_s")].setText("invalid")
    editor.save()
    editor.general.plots.set_values(["outlet_composition"])
    assert editor.general.plots.show_buttons[0].isEnabled()
    editor.show_plot("outlet_composition")
    qt_app.processEvents()
    assert len(editor.plot_windows) == 1
    assert editor.plot_windows[0].findChild(QSvgWidget).renderer().isValid()
    assert path.read_bytes() == original
    assert case.state()["stale"]
    editor.plot_windows[0].close()
    qt_app.processEvents()
