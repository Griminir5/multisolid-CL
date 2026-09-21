from copy import deepcopy
import shutil
from uuid import uuid4

import pytest
import yaml

from packed_bed_ui.inputs import definition_payload
from packed_bed_ui.project import Project, read_json, write_json
from packed_bed_ui.studies import (Factor, ReusableDefinition, StudyError, expand_study, factor_values,
                                   generation_signature, parameter_catalogue)
from packed_bed_ui.study_store import baseline_eligibility
from packed_bed_ui.worker import activate_snapshot


def succeed(project, case):
    job = read_json(project.prepare_execution([case]))
    folder = activate_snapshot(case.root / f".pending-{job['attempt_id']}")
    write_json(folder / "status.json", {"state": "completed"})
    (folder / "retained-result.txt").write_text("result")
    return folder


@pytest.fixture
def study_project(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case, "Benchmark")
    case.documents["program"]["inlet_temperature"]["steps"] = [
        {"kind": "hold", "duration_s": .004}, {"kind": "ramp", "duration_s": .006, "target": 400.0}]
    case.save()
    succeed(project, case)
    study = project.study_store.create("Sweep", case)
    return project, case, study


def test_baseline_requires_latest_unchanged_success(study_project):
    project, case, study = study_project
    assert baseline_eligibility(case).eligible
    original = deepcopy(study.baseline)
    case.documents["run"]["model"]["bed_radius_m"] *= 2
    case.save()
    assert not baseline_eligibility(case).eligible
    with pytest.raises(StudyError, match="changed"):
        project.study_store.create("Unproven", case)
    assert project.study_store.studies[study.id].baseline == original
    case.documents["run"]["model"]["bed_radius_m"] /= 2
    for status in ("not_run", "failed", "cancelled", "running"):
        write_json(case.run_folder / "status.json", {"state": status})
        assert not baseline_eligibility(case).eligible
    write_json(case.run_folder / "status.json", {"state": "completed"})
    (case.run_folder / "inputs/run.yaml").write_text("tampered: true")
    assert not baseline_eligibility(case).eligible


def test_numeric_product_and_length_scaling(study_project):
    project, case, study = study_project
    study.baseline["solids"]["initial_profile"]["zones"] = [
        {**study.baseline["solids"]["initial_profile"]["zones"][0], "x_end_m": .4},
        {**study.baseline["solids"]["initial_profile"]["zones"][0], "x_start_m": .4},
    ]
    temperature = next(p for p in parameter_catalogue(study) if p.label == "Inlet temperature → Initial value")
    study.factors = [Factor("temperature", temperature.id, [800, 900, 1000]), Factor("length", "bed_length_m", [.4, .6, .8])]
    candidates = list(expand_study(study, {}))
    assert len(candidates) == 9
    assert all(c.inputs == "Ready" for c in candidates)
    zones = candidates[0].documents["solids"]["initial_profile"]["zones"]
    assert zones[0]["x_end_m"] == pytest.approx(.16)
    assert zones[1]["x_end_m"] == pytest.approx(.4)
    assert zones[0]["values"] == study.baseline["solids"]["initial_profile"]["zones"][0]["values"]


def test_replacement_requires_success_and_reselecting_steps(study_project):
    project, base, study = study_project
    target = next(p for p in parameter_catalogue(study) if "Step 2 ramp → Target" in p.label)
    study.factors = [Factor("target", target.id, [450])]
    project.study_store.save_study(study)
    replacement = project.duplicate_case(base, "New baseline")
    with pytest.raises(StudyError, match="succeeded"):
        project.study_store.replace_baseline(study, replacement)
    succeed(project, replacement)
    changed = project.study_store.replace_baseline(study, replacement)
    with pytest.raises(StudyError, match="no longer exists"):
        list(expand_study(changed, {}))
    changed.factors[0].target = next(p.id for p in parameter_catalogue(changed) if "Step 2 ramp → Target" in p.label)
    project.study_store.save_study(changed)
    fixed = deepcopy(changed.baseline)
    project.delete_case(base)
    project.delete_case(replacement)
    reopened = Project.open(project.root)
    retained = reopened.study_store.studies[study.id]
    assert retained.baseline == fixed
    assert len(reopened.study_store.apply_preview(reopened.study_store.preview(retained))) == 1


def test_invalid_inputs_and_unreadable_snapshots_cannot_be_baselines(study_project):
    _, base, _ = study_project
    original = deepcopy(base.documents)
    base.documents["run"]["model"]["bed_length_m"] = -1
    assert not baseline_eligibility(base).eligible
    base.documents = original
    (base.run_folder / "snapshot.json").write_text("{broken")
    assert not baseline_eligibility(base).eligible


def test_ranges_reject_rounding_duplicates_and_nonfinite(study_project):
    _, _, study = study_project
    cells = next(p for p in parameter_catalogue(study) if p.id == "axial_cells")
    factor = Factor("cells", cells.id, range={"start": "3", "end": "6", "count": "3"})
    with pytest.raises(StudyError, match="whole numbers"):
        factor_values(factor, cells)
    for values in ([3, 3], ["NaN"], ["Inf"], [""], [True]):
        with pytest.raises(StudyError):
            factor_values(Factor("cells", cells.id, values), cells)
    assert factor_values(Factor("cells", cells.id, range={"start": "3", "end": "7", "count": "3"}), cells) == [3, 5, 7]


def test_step_targets_and_two_edits_in_same_list(study_project):
    _, _, study = study_project
    parameters = parameter_catalogue(study)
    duration = next(p for p in parameters if "Step 2 ramp → Duration" in p.label)
    target = next(p for p in parameters if "Step 2 ramp → Target" in p.label)
    study.factors = [Factor("duration", duration.id, [.008]), Factor("target", target.id, [450])]
    candidate, = expand_study(study, {})
    assert candidate.documents["program"]["inlet_temperature"]["steps"][1] == {
        "kind": "ramp", "duration_s": .008, "target": 450}
    assert candidate.documents["run"]["simulation"]["time_horizon_s"] == pytest.approx(.012)
    study.baseline["program"]["inlet_temperature"]["steps"].pop(0)
    study.editor_metadata["step_ids"]["inlet_temperature"].pop(0)
    candidate, = expand_study(study, {})
    assert candidate.documents["program"]["inlet_temperature"]["steps"][0]["target"] == 450
    study.baseline["program"]["inlet_temperature"]["steps"].pop()
    study.editor_metadata["step_ids"]["inlet_temperature"].pop()
    with pytest.raises(StudyError, match="no longer exists"):
        list(expand_study(study, {}))


def test_explicit_rows_and_overlap(study_project):
    _, _, study = study_project
    study.factors = [Factor("length", "bed_length_m"), Factor("cells", "axial_cells")]
    study.mode = "rows"
    study.rows = [{"length": .4, "cells": 4}, {"length": .8, "cells": 8}]
    assert len(list(expand_study(study, {}))) == 2
    study.rows[0]["cells"] = ""
    with pytest.raises(StudyError, match="finite number"):
        list(expand_study(study, {}))
    study.factors.append(Factor("bed", "definition:bed"))
    with pytest.raises(StudyError, match="overlaps"):
        list(expand_study(study, {}))


def test_rebuild_deletes_every_case_and_result_but_keeps_library(study_project):
    project, base, study = study_project
    independent = project.duplicate_case(base, "Exploratory")
    definition = ReusableDefinition(uuid4().hex, "Bed A", "bed", definition_payload("bed", base.documents))
    project.study_store.save_definition(definition)
    study.factors = [Factor("cells", "axial_cells", [3, 5])]
    project.study_store.save_study(study)
    previous = project.study_store.apply_preview(project.study_store.preview(study))
    result = succeed(project, previous[0])
    old_ids = {case.id for case in previous}
    old_inputs = deepcopy(previous[0].documents)
    study.factors[0].values = [3, 7, 9]
    project.study_store.save_study(study)
    assert all(case.state()["inputs"] == "Needs update" for case in previous)
    with pytest.raises(ValueError, match="No cases were started"):
        project.prepare_execution([base, previous[0]])
    assert result.exists()
    current = project.study_store.apply_preview(project.study_store.preview(study))
    assert len(current) == 3 and not old_ids & {case.id for case in current}
    assert all(not case.root.exists() for case in previous)
    assert all(not case.run_folder.exists() for case in current)
    assert current[0].documents == old_inputs
    assert base.run_folder.exists() and independent.root.exists()
    reopened = Project.open(project.root)
    assert definition.id in reopened.study_store.definitions
    assert len(reopened.cases) == 5
    assert not (project.root / ".study-transaction").exists()
    assert not old_ids & read_json(project.root / "execution.json")["cases"].keys()
    # Reintroducing a removed selection creates another fresh case without restored results.
    study.factors[0].values = [5]
    project.study_store.save_study(study)
    restored, = project.study_store.apply_preview(project.study_store.preview(study))
    assert restored.id not in old_ids | {case.id for case in current}
    assert restored.documents["run"]["model"]["axial_cells"] == 5
    assert not restored.run_folder.exists()


def test_study_reports_are_fixed_copies_and_adapt_to_generated_grids(study_project):
    from packed_bed.report_schema import describe_inputs
    from packed_bed_ui.workbook import plan_sheet, rule_columns

    project, base, study = study_project
    rule = {"id": "temperature", "quantity": "temperature", "selections": {"x_cell": {"mode": "all"}}}
    sheet = {"name": "Temperatures", "axis": "time", "rows": {"mode": "all"},
             "columns": rule_columns(describe_inputs(base.documents), rule), "column_rules": [rule]}
    base.metadata["report"] = {"version": 1, "sheets": [sheet]}
    base.save()
    study = project.study_store.replace_baseline(study, base)
    study.factors = [Factor("cells", "axial_cells", [3, 5])]
    project.study_store.save_study(study)
    signature = project.study_store.signature(study)
    base.metadata["report"]["sheets"][0]["name"] = "Source edited later"
    base.save()
    project.delete_case(base)
    project = Project.open(project.root)
    store = project.study_store
    study = store.studies[study.id]
    cases = store.apply_preview(store.preview(study))
    assert len(cases) == 2
    for case, cells in zip(cases, (3, 5)):
        layout = case.metadata["report"]["sheets"][0]
        assert layout["name"] == "Temperatures"
        assert len(plan_sheet(describe_inputs(case.documents), layout)["columns"]) == cells
    cases[0].metadata["report"]["sheets"][0]["name"] = "Local report edit"
    cases[0].save()
    assert cases[1].metadata["report"]["sheets"][0]["name"] == "Temperatures"
    assert store.signature(study) == signature and not store.needs_update(study.id)
    replacement = store.apply_preview(store.preview(study))
    assert replacement[0].metadata["report"]["sheets"][0]["name"] == "Temperatures"


def test_existing_study_can_capture_report_before_source_is_deleted(study_project):
    project, base, study = study_project
    study.editor_metadata.pop("report")  # An existing study from before report inheritance.
    study.factors = [Factor("cells", "axial_cells", [3])]
    project.study_store.save_study(study, _replace_baseline=True)
    base.metadata["report"] = {"version": 1, "sheets": []}
    base.save()
    generated, = project.study_store.apply_preview(project.study_store.preview(study))
    assert generated.metadata["report"] == base.metadata["report"]
    project.delete_case(base)
    reopened = Project.open(project.root)
    store = reopened.study_store
    generated, = store.apply_preview(store.preview(store.studies[study.id]))
    assert generated.metadata["report"] == {"version": 1, "sheets": []}


def test_delete_study_removes_cases_results_and_queue_but_keeps_library(study_project):
    project, base, study = study_project
    store = project.study_store
    other = store.create("Other study", base)
    other.factors = [Factor("cells", "axial_cells", [5])]
    store.save_study(other)
    other_case, = store.apply_preview(store.preview(other))
    definition = ReusableDefinition(uuid4().hex, "Bed", "bed", definition_payload("bed", base.documents))
    store.save_definition(definition)
    study.factors = [Factor("bed", "definition:bed", [definition.id])]
    store.save_study(study)
    generated, = store.apply_preview(store.preview(study))
    succeed(project, generated)
    project.executing = True
    with pytest.raises(StudyError, match="execution"):
        store.delete_study(study.id)
    project.executing = False
    store.delete_study(study.id)
    assert not generated.root.exists()
    assert not (project.root / "studies" / study.id).exists()
    assert not (project.root / ".study-transaction").exists()
    assert generated.id not in read_json(project.root / "execution.json")["cases"]
    reopened = Project.open(project.root)
    assert {case.id for case in reopened.cases} == {base.id, other_case.id}
    assert set(reopened.study_store.studies) == {other.id}
    assert definition.id in reopened.study_store.definitions


def test_delete_study_rolls_back_when_metadata_commit_fails(study_project, monkeypatch):
    import packed_bed_ui.study_store as storage

    project, base, study = study_project
    study.factors = [Factor("cells", "axial_cells", [3])]
    project.study_store.save_study(study)
    generated, = project.study_store.apply_preview(project.study_store.preview(study))
    succeed(project, generated)
    before = (project.root / "project.json").read_bytes()
    write = storage.write_json
    def fail(path, value):
        if path == project.root / "project.json":
            raise OSError("Commit interrupted")
        write(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(storage, "write_json", fail)
        with pytest.raises(OSError, match="interrupted"):
            project.study_store.delete_study(study.id)
    reopened = Project.open(project.root)
    assert (project.root / "project.json").read_bytes() == before
    assert {case.id for case in reopened.cases} == {base.id, generated.id}
    assert study.id in reopened.study_store.studies
    assert (generated.run_folder / "retained-result.txt").read_text() == "result"
    assert not (project.root / ".study-transaction").exists()


def test_scientific_errors_create_drafts_and_program_horizons_follow_ownership(study_project):
    project, base, study = study_project
    study.factors = [Factor("length", "bed_length_m", [-1])]
    project.study_store.save_study(study)
    invalid, = project.study_store.apply_preview(project.study_store.preview(study))
    assert invalid.state()["inputs"] == "Invalid"
    with pytest.raises(ValueError):
        invalid.validate_for_run()
    definition = ReusableDefinition(uuid4().hex, "Other gas", "program", definition_payload("program", base.documents))
    definition.payload["program"]["inlet_composition"]["initial"] = {"H2": 1.0}
    study.factors = [Factor("program", "definition:program", [definition.id])]
    candidate, = expand_study(study, {definition.id: definition})
    assert candidate.inputs == "Invalid" and "H2" in candidate.message
    definition.payload = definition_payload("program", base.documents)
    definition.payload["simulation"]["repeat_program"] = True
    study.baseline["run"]["simulation"]["time_horizon_s"] = .25
    candidate, = expand_study(study, {definition.id: definition})
    assert candidate.documents["run"]["simulation"]["time_horizon_s"] == .25
    definition.payload["simulation"]["repeat_program"] = False
    candidate, = expand_study(study, {definition.id: definition})
    assert candidate.documents["run"]["simulation"]["time_horizon_s"] == .01


def test_revert_and_rename_do_not_require_rebuild(study_project):
    project, _, study = study_project
    study.factors = [Factor("cells", "axial_cells", [3, 5])]
    project.study_store.save_study(study)
    cases = project.study_store.apply_preview(project.study_store.preview(study))
    study.name = "Renamed"
    study.factors[0].values.reverse()
    project.study_store.save_study(study)
    assert not project.study_store.needs_update(study.id)
    study.factors[0].values.append(7)
    project.study_store.save_study(study)
    assert cases[0].state()["inputs"] == "Needs update"
    study.factors[0].values.remove(7)
    project.study_store.save_study(study)
    assert cases[0].state()["inputs"] == "Ready"
    study.factors[0].values = ["3.0", "5"]
    project.study_store.save_study(study)
    assert cases[0].state()["inputs"] == "Ready"



def test_stale_preview_and_unverified_baseline_cannot_delete_cases(study_project):
    project, _, study = study_project
    study.factors = [Factor("cells", "axial_cells", [3])]
    project.study_store.save_study(study)
    preview = project.study_store.preview(study)
    study.factors[0].values = [5]
    project.study_store.save_study(study)
    with pytest.raises(StudyError, match="changed after"):
        project.study_store.apply_preview(preview)
    study.baseline["run"]["model"]["bed_length_m"] = 20
    with pytest.raises(StudyError, match="read-only"):
        project.study_store.save_study(study)


def test_incremental_preview_keeps_the_reviewed_sources(study_project):
    project, _, study = study_project
    study.factors = [Factor("cells", "axial_cells", [3, 5])]
    project.study_store.save_study(study)
    preview = project.study_store.preview(study, lazy=True)
    assert preview.total == 2 and not preview.candidates
    study.factors[0].values = [7]
    project.study_store.save_study(study)
    preview.candidates.extend(preview.remaining)
    assert [candidate.documents["run"]["model"]["axial_cells"] for candidate in preview.candidates] == [3, 5]
    with pytest.raises(StudyError, match="changed after"):
        project.study_store.apply_preview(preview)
    assert len(project.cases) == 1


def test_definition_changes_invalidate_all_users_and_removal_keeps_library(study_project):
    project, base, first = study_project
    definition = ReusableDefinition(uuid4().hex, "Bed", "bed", definition_payload("bed", base.documents))
    project.study_store.save_definition(definition)
    second = project.study_store.create("Other", base)
    for study in (first, second):
        study.factors = [Factor("bed", "definition:bed", [definition.id])]
        project.study_store.save_study(study)
        project.study_store.apply_preview(project.study_store.preview(study))
    definition.payload["model"]["bed_radius_m"] *= 2
    project.study_store.save_definition(definition)
    assert all(project.study_store.needs_update(s.id) for s in (first, second))
    with pytest.raises(StudyError, match="used by"):
        project.study_store.delete_definition(definition.id)
    first.factors = [Factor("cells", "axial_cells", [3])]
    project.study_store.save_study(first)
    project.study_store.apply_preview(project.study_store.preview(first))
    assert definition.id in project.study_store.definitions


@pytest.mark.parametrize("bed_first", [True, False])
def test_bed_settings_apply_alongside_a_program_definition(study_project, bed_first):
    project, base, study = study_project
    bed = ReusableDefinition(uuid4().hex, "Thermal bed", "bed", definition_payload("bed", base.documents))
    bed.payload["model"].update(ambient_temperature_k=650, heat_transfer_coefficient_w_per_m2_k=42, gas_voidage_mode="bed_only")
    bed.payload["simulation"]["interior_flow_mode"] = "reversible"
    program = ReusableDefinition(uuid4().hex, "Cyclic program", "program", definition_payload("program", base.documents))
    program.payload["simulation"]["repeat_program"] = True
    for definition in (bed, program):
        project.study_store.save_definition(definition)
    study.factors = [Factor("bed", "definition:bed", [bed.id]), Factor("program", "definition:program", [program.id])]
    if not bed_first:
        study.factors.reverse()
    project.study_store.save_study(study)
    generated, = project.study_store.apply_preview(project.study_store.preview(study))
    case = generated.resolve()
    assert generated.state()["inputs"] == "Ready"
    assert case.run.model.ambient_temperature_k == 650
    assert case.run.model.heat_transfer_coefficient_w_per_m2_k == 42
    assert case.run.model.gas_voidage_mode == "bed_only"
    assert case.run.simulation.interior_flow_mode == "reversible"
    assert case.run.simulation.repeat_program is True
    assert case.run.simulation.time_horizon_s == base.resolve().run.simulation.time_horizon_s
    for section, key, value in (("model", "ambient_temperature_k", 700),
                                 ("model", "heat_transfer_coefficient_w_per_m2_k", 0),
                                 ("model", "gas_voidage_mode", "bed_and_particle"),
                                 ("simulation", "interior_flow_mode", "forward_only")):
        bed.payload[section][key] = value
        project.study_store.save_definition(bed)
        assert generated.state()["inputs"] == "Needs update"
        previous = generated
        generated, = project.study_store.apply_preview(project.study_store.preview(study))
        assert not previous.root.exists()
        assert generated.documents["run"][section][key] == value


@pytest.mark.parametrize("target", ["ambient_temperature_k", "heat_transfer_coefficient_w_per_m2_k"])
def test_thermal_sweeps_only_conflict_with_beds_that_own_those_settings(study_project, target):
    project, base, study = study_project
    # Old definitions continue inheriting their thermal and flow settings until saved in the editor.
    payload = {"model": {key: base.documents["run"]["model"][key] for key in ("bed_length_m", "bed_radius_m")},
               "solids": deepcopy(base.documents["solids"])}
    bed = ReusableDefinition(uuid4().hex, "Old bed", "bed", payload)
    project.study_store.save_definition(bed)
    study.factors = [Factor("bed", "definition:bed", [bed.id]), Factor("thermal", target, [350])]
    project.study_store.save_study(study)
    case, = project.study_store.apply_preview(project.study_store.preview(study))
    assert case.documents["run"]["model"][target] == 350
    reopened = Project.open(project.root)
    assert not reopened.study_store.needs_update(study.id)
    bed.payload = definition_payload("bed", base.documents)
    project.study_store.save_definition(bed)
    with pytest.raises(StudyError, match="overlaps"):
        project.study_store.preview(study)


def test_import_bed_presets_keeps_thermal_and_voidage_values(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    path = source_case.parent / "batch.yaml"
    path.write_text(yaml.safe_dump({"base_case": "run.yaml", "output_directory": "unused",
        "geometries": {"hot": {"model": {"ambient_temperature_k": 700,
            "heat_transfer_coefficient_w_per_m2_k": 25, "gas_voidage_mode": "bed_only"}}},
        "axes": [{"id": "bed", "values": [{"id": "hot", "geometry": "hot"}]}]}))
    study = project.import_study(path)
    assert study.legacy is None
    baseline = project.study_store.add_imported_baseline(study)
    succeed(project, baseline)
    study = project.study_store.replace_baseline(study, baseline)
    generated, = project.study_store.apply_preview(project.study_store.preview(study))
    model = generated.resolve().run.model
    assert model.ambient_temperature_k == 700
    assert model.heat_transfer_coefficient_w_per_m2_k == 25
    assert model.gas_voidage_mode == "bed_only"


@pytest.mark.parametrize("fail_after_commit", [False, True])
def test_rebuild_recovers_consistently(study_project, monkeypatch, fail_after_commit):
    from packed_bed_ui import study_store
    project, _, study = study_project
    study.factors = [Factor("cells", "axial_cells", [3, 5])]
    project.study_store.save_study(study)
    previous = project.study_store.apply_preview(project.study_store.preview(study))
    old_ids = {case.id for case in previous}
    study.factors[0].values = [7]
    project.study_store.save_study(study)
    preview = project.study_store.preview(study)
    original = study_store.write_json
    def fail(path, data):
        if path == project.root / "project.json":
            if fail_after_commit:
                original(path, data)
            raise OSError("simulated interruption")
        return original(path, data)
    with monkeypatch.context() as patch:
        patch.setattr(study_store, "write_json", fail)
        with pytest.raises(OSError):
            project.study_store.apply_preview(preview)
    assert len([case for case in project.cases if case.metadata.get("study_id") == study.id]) == (1 if fail_after_commit else 2)
    reopened = Project.open(project.root)
    cases = [case for case in reopened.cases if case.metadata.get("study_id") == study.id]
    assert len(cases) == (1 if fail_after_commit else 2)
    assert bool(old_ids & {case.id for case in cases}) is not fail_after_commit
    assert all(case.root.exists() for case in cases)
    assert not (project.root / ".study-transaction").exists()


def test_import_native_and_advanced_rules_require_baseline_success(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    path = source_case.parent / "batch.yaml"
    spec = {"base_case": "run.yaml", "output_directory": "unused", "axes": [
        {"id": "cells", "values": [{"id": "coarse", "patch": {"run": {"model": {"axial_cells": 3}}}},
                                    {"id": "fine", "patch": {"run": {"model": {"axial_cells": 5}}}}]},
    ]}
    path.write_text(yaml.safe_dump(spec))
    native = project.import_study(path)
    assert native.legacy is None
    assert not project.cases
    with pytest.raises(StudyError, match="successful"):
        project.study_store.apply_preview(project.study_store.preview(native))
    baseline = project.study_store.add_imported_baseline(native)
    assert not baseline_eligibility(baseline).eligible
    succeed(project, baseline)
    native = project.study_store.replace_baseline(native, baseline)
    generated = project.study_store.apply_preview(project.study_store.preview(native))
    assert [case.resolve().run.model.axial_cells for case in generated] == [3, 5]
    # Overlapping legacy axes keep their original later-axis-wins semantics.
    spec["axes"].append({"id": "override", "values": [{"id": "seven", "patch": {"run": {"model": {"axial_cells": 7}}}}]})
    path.write_text(yaml.safe_dump(spec))
    advanced = project.import_study(path)
    assert advanced.legacy is not None
    advanced = project.study_store.replace_baseline(advanced, baseline)
    shutil.rmtree(source_case.parent)
    reopened = Project.open(project.root)
    study = reopened.study_store.studies[advanced.id]
    generated = reopened.study_store.apply_preview(reopened.study_store.preview(study))
    assert [case.resolve().run.model.axial_cells for case in generated] == [7, 7]


def test_format_two_studies_migrate_without_changing_cases_or_results(tmp_path, source_case):
    from packed_bed.batch import expand_batch_cases, load_batch_spec
    project = Project.create(tmp_path / "project")
    path = source_case.parent / "batch.yaml"
    path.write_text(yaml.safe_dump({"base_case": "run.yaml", "output_directory": "unused", "axes": [
        {"id": "cells", "values": [{"id": "three", "patch": {"run": {"model": {"axial_cells": 3}}}}]},
    ]}))
    study = project.import_study(path)
    folder = project.root / "studies" / study.id
    expanded, = expand_batch_cases(load_batch_spec(folder / "batch.yaml"))
    case = project.add_case("Old generated", {name: getattr(expanded, name) for name in ("run", "program", "chemistry", "solids")})
    case.metadata["study_id"] = study.id
    case.metadata["selections"] = expanded.selections
    project.save()
    run = succeed(project, case)
    before = deepcopy(case.documents)
    project.metadata["format_version"] = 2
    project.metadata["studies"] = [{"id": study.id, "name": study.name}]
    project.save()
    (folder / "study.json").unlink()
    shutil.rmtree(folder / "baseline")
    reopened = Project.open(project.root)
    assert reopened.metadata["format_version"] == 3
    assert read_json(project.root / "project-v2.json")["format_version"] == 2
    retained, = reopened.cases
    assert retained.id == case.id and retained.documents == before
    assert (run / "retained-result.txt").read_text() == "result"
    assert retained.state()["inputs"] == "Ready"
    migrated = reopened.study_store.studies[study.id]
    assert not migrated.provenance
    with pytest.raises(StudyError, match="successful"):
        reopened.study_store.apply_preview(reopened.study_store.preview(migrated))


def test_feed_ramp_units_timing_and_dependency_errors(study_project):
    from packed_bed_ui.inputs import new_step_ids
    project, base, study = study_project
    study.baseline["run"]["simulation"]["program_mode"] = "feed_stream"
    study.baseline["program"] = {
        "feed_stream": {"basis": "ghsv_per_h", "initial": {"flow": 100, "temperature": 300, "composition": {"N2": 1}},
                        "steps": [{"kind": "ramp", "duration_s": .01, "target": {"flow": 200, "temperature": 400}}]},
        "outlet_pressure": {"initial": 100000, "steps": [{"kind": "hold", "duration_s": .01}]},
    }
    study.editor_metadata = {"step_ids": new_step_ids(study.baseline["program"])}
    catalogue = parameter_catalogue(study)
    flow = next(p for p in catalogue if p.label == "Feed → Step 1 ramp → Flow")
    duration = next(p for p in catalogue if p.label == "Feed → Step 1 ramp → Duration")
    assert flow.unit == "h⁻¹"
    study.factors = [Factor("flow", flow.id, [300]), Factor("length", "bed_length_m", [.5, 1.0])]
    candidates = list(expand_study(study, {}))
    assert all(c.inputs == "Ready" for c in candidates)
    from packed_bed_ui.inputs import resolve_documents
    flows = [resolve_documents(c.documents).inlet_flow_program for c in candidates]
    # The engine converts the same GHSV using each candidate's physical volume.
    assert flows[1].initial_value == pytest.approx(2 * flows[0].initial_value)
    study.factors.append(Factor("duration", duration.id, [.02]))
    assert all(c.inputs == "Invalid" for c in expand_study(study, {}))
    study.baseline["program"]["feed_stream"]["basis"] = "mol_per_s"
    with pytest.raises(StudyError, match="no longer exists"):
        list(expand_study(study, {}))
