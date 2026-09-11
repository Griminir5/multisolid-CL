from pathlib import Path
import shutil

import pytest
import yaml

from packed_bed.config import load_case
from packed_bed_ui.project import Project, input_hashes, read_documents, read_json, write_documents, write_json
from packed_bed_ui.worker import activate_snapshot


def test_project_starts_empty_and_owns_multiple_independent_cases(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    assert project.cases == []
    benchmark = project.add_case_from_files(source_case, "Benchmark")
    exploratory = project.duplicate_case(benchmark, "Exploratory")
    assert benchmark.id != exploratory.id
    assert benchmark.root != exploratory.root
    assert not exploratory.run_folder.exists()
    reopened = Project.open(project.root / "project.json")
    assert [case.name for case in reopened.cases] == ["Benchmark", "Exploratory"]
    assert reopened.metadata["extensions"] == []
    assert not (project.root / "inputs").exists()


@pytest.mark.parametrize("filename", ["run.yaml", "run_feed_stream.yaml"])
def test_example_import_is_portable_and_preserves_scientific_settings(tmp_path, filename):
    source = Path(__file__).resolve().parents[2] / "packed_bed/examples/default_case" / filename
    original = load_case(source)
    hashes = input_hashes(source.parent)
    project = Project.create(tmp_path / "project")
    project.add_case_from_files(source)
    shutil.move(project.root, tmp_path / "moved")
    case = Project.open(tmp_path / "moved").cases[0].resolve()
    for name in ("chemistry", "program", "solids"):
        assert getattr(case, name) == getattr(original, name)
    for name in ("simulation", "solver", "model"):
        assert getattr(case.run, name) == getattr(original.run, name)
    assert input_hashes(source.parent) == hashes


def test_empty_and_invalid_drafts_survive_save_but_block_execution(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    draft = project.add_case("Draft")
    assert draft.state()["inputs"] == "Underdefined"
    case = project.add_case_from_files(source_case)
    case.documents["run"]["simulation"]["time_horizon_s"] = "unfinished"
    case.save()
    reopened = Project.open(project.root)
    assert reopened.cases[1].state()["inputs"] == "Invalid"
    with pytest.raises(ValueError, match="No cases were started"):
        reopened.prepare_execution(reopened.cases)
    assert not (project.root / "execution.json").exists()
    assert not list(project.root.glob("cases/*/.pending-*"))


def test_latest_run_replaced_only_after_preparation_and_edits_show_staleness(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    path = project.prepare_execution([case])
    first = read_json(path)["attempt_id"]
    folder = activate_snapshot(case.root / f".pending-{first}")
    write_json(folder / "status.json", {"state": "completed"})
    (folder / "old-results.txt").write_text("previous results")
    original_hashes = input_hashes(folder / "inputs")
    assert not case.state()["stale"]
    case.documents["run"]["model"]["axial_cells"] = 5
    case.save()
    assert case.state()["stale"]
    assert case.state()["state"] == "completed"
    case.documents["run"]["model"]["axial_cells"] = 3
    assert not case.state()["stale"]
    case.documents["run"]["model"]["axial_cells"] = 5
    assert input_hashes(folder / "inputs") == original_hashes
    second = read_json(project.prepare_execution([case]))["attempt_id"]
    assert (folder / "old-results.txt").is_file()
    assert activate_snapshot(case.root / f".pending-{second}") == folder
    assert not (folder / "old-results.txt").exists()
    assert not (case.root / ".previous-run").exists()
    assert load_case(folder / "inputs/run.yaml").run.model.axial_cells == 5
    assert not case.state()["stale"]
    assert not (case.root / "runs").exists()


def test_invalid_and_tampered_reruns_preserve_previous_results(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    job = read_json(project.prepare_execution([case]))
    folder = activate_snapshot(case.root / f".pending-{job['attempt_id']}")
    marker = folder / "result.txt"
    marker.write_text("keep")
    case.documents["run"]["model"]["axial_cells"] = -1
    with pytest.raises(ValueError):
        project.prepare_execution([case])
    assert marker.read_text() == "keep"
    case.documents["run"]["model"]["axial_cells"] = 3
    job = read_json(project.prepare_execution([case]))
    pending = case.root / f".pending-{job['attempt_id']}"
    with (pending / "inputs/run.yaml").open("a") as stream:
        stream.write("\n# tampered\n")
    with pytest.raises(ValueError, match="Snapshot inputs have changed"):
        activate_snapshot(pending)
    assert marker.read_text() == "keep"


def test_duplicate_and_renaming_do_not_copy_results_or_invent_staleness(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    job = read_json(project.prepare_execution([case]))
    folder = activate_snapshot(case.root / f".pending-{job['attempt_id']}")
    write_json(folder / "status.json", {"state": "completed"})
    duplicate = project.duplicate_case(case, "Copy")
    assert duplicate.state()["state"] == "not_run"
    case.metadata["name"] = "Renamed"
    case.documents["run"]["outputs"]["directory"] = "/different/output"
    case.save()
    assert not case.state()["stale"]
    project.metadata["extensions"] = [{"id": "custom", "version": "1", "sha256": "example"}]
    assert case.state()["stale"]
    with pytest.raises(ValueError, match="requires extensions"):
        case.validate_for_run()


def test_legacy_project_migration_keeps_only_latest_run_in_new_case_and_backs_up_originals(tmp_path, source_case):
    root = tmp_path / "legacy"
    write_documents(root / "inputs", read_documents(source_case.parent))
    write_json(root / "project.json", {"format_version": 1, "name": "Original"})
    hashes = input_hashes(root / "inputs")
    for number in (1, 2):
        folder = root / "runs" / f"run-{number}"
        write_documents(folder / "inputs", read_documents(source_case.parent))
        write_json(folder / "snapshot.json", {"format_version": 1, "input_hashes": hashes,
                                               "created_at": f"2026-01-01T00:00:00.00000{number}+00:00"})
        write_json(folder / "status.json", {"state": "completed"})
        (folder / "result.txt").write_text(str(number))
    (root / "runs/run-1").rename(root / "runs/zzz-first")
    project = Project.open(root)
    assert len(project.cases) == 1
    assert (project.cases[0].run_folder / "result.txt").read_text() == "2"
    assert input_hashes(root / "inputs") == hashes
    assert (root / "runs/zzz-first/result.txt").read_text() == "1"
    assert read_json(root / "project-v1.json")["format_version"] == 1
    assert read_json(root / "project.json")["format_version"] == 2
    assert len(Project.open(root).cases) == 1


def test_interrupted_replacement_and_pending_inputs_recover(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    job = read_json(project.prepare_execution([case]))
    folder = activate_snapshot(case.root / f".pending-{job['attempt_id']}")
    write_json(folder / "status.json", {"state": "running"})
    folder.rename(case.root / ".previous-run")
    project.prepare_execution([case])
    project.recover_interrupted()
    assert case.state()["state"] == "interrupted"
    assert not list(case.root.glob(".pending-*"))


def test_mixed_project_has_two_independent_plus_nine_generated_cases(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    base = project.add_case_from_files(source_case, "Benchmark")
    project.duplicate_case(base, "Exploratory")
    source = source_case.parent
    programs, geometries = {}, {}
    for index in range(3):
        program = yaml.safe_load((source / "program.yaml").read_text())
        program["inlet_flow"]["initial"] = 1e-8 * (index + 1)
        name = f"program-{index}"
        (source / f"{name}.yaml").write_text(yaml.safe_dump(program))
        programs[name] = f"{name}.yaml"
        solids = yaml.safe_load((source / "solids.yaml").read_text())
        solids["initial_profile"]["zones"][0]["values"]["Ni"] = float(index + 1)
        name = f"solids-{index}"
        (source / f"{name}.yaml").write_text(yaml.safe_dump(solids))
        geometries[name] = {"solids_file": f"{name}.yaml"}
    batch = {"base_case": "run.yaml", "output_directory": "must-not-be-written", "programs": programs,
             "geometries": geometries, "axes": [
                 {"id": "program", "values": [{"id": name, "program": name} for name in programs]},
                 {"id": "solids", "values": [{"id": name, "geometry": name} for name in geometries]},
             ]}
    path = source / "study.yaml"
    path.write_text(yaml.safe_dump(batch))
    added = project.add_cases_from_batch(path)
    assert len(added) == 9
    assert len(project.cases) == 11
    assert len({tuple(c.metadata["selections"].values()) for c in added}) == 9
    assert all(case.state()["inputs"] == "Ready" for case in added)
    assert not (source / "must-not-be-written").exists()
    shutil.rmtree(source)
    reopened = Project.open(project.root)
    job = read_json(reopened.prepare_execution(reopened.cases))
    assert len(job["cases"]) == 11
    from packed_bed.batch import expand_batch_cases, load_batch_spec
    study_path = next((project.root / "studies").glob("*/batch.yaml"))
    assert len(expand_batch_cases(load_batch_spec(study_path))) == 9


def test_unknown_project_version_and_external_references_are_rejected(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    case.documents["run"]["references"]["chemistry_file"] = str(source_case.parent / "chemistry.yaml")
    with pytest.raises(ValueError, match="Case inputs must reference"):
        case.prepare("test")
    write_json(project.root / "project.json", {"format_version": 999})
    with pytest.raises(ValueError, match="Unsupported project format"):
        Project.open(project.root)


def test_delete_case_removes_only_its_inputs_and_result(tmp_path, source_case, monkeypatch):
    project = Project.create(tmp_path / "project")
    first = project.add_case_from_files(source_case, "First")
    second = project.duplicate_case(first, "Second")
    job = read_json(project.prepare_execution([first]))
    activate_snapshot(first.root / f".pending-{job['attempt_id']}")
    with monkeypatch.context() as patch:
        def fail():
            raise PermissionError("Cannot save metadata")
        patch.setattr(project, "save", fail)
        with pytest.raises(PermissionError):
            project.delete_case(first)
    assert first.root.exists() and first.run_folder.exists()
    assert len(Project.open(project.root).cases) == 2
    project.delete_case(first)
    assert not first.root.exists()
    assert [case.id for case in Project.open(project.root).cases] == [second.id]
    assert second.resolve()


def test_worker_limit_validation_and_case_threads_are_independent(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    case.documents["run"]["solver"]["threads"] = 3
    for invalid in (0, -1, True, 2.5, "2"):
        with pytest.raises(ValueError, match="positive integer"):
            project.prepare_execution([case], max_workers=invalid)
    assert not list(case.root.glob(".pending-*"))
    job = read_json(project.prepare_execution([case], max_workers=2))
    assert job["max_workers"] == 2
    staged = case.root / f".pending-{job['attempt_id']}" / "inputs/run.yaml"
    assert yaml.safe_load(staged.read_text())["solver"]["threads"] == 3
