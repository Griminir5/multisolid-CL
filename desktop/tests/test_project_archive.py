from copy import deepcopy
import json
from pathlib import Path
import shutil
import stat
from zipfile import ZipFile, ZipInfo

import pytest
import yaml

from packed_bed.plugins.catalogue import builtin_manifest
from packed_bed.plugins.storage import package_hash
from packed_bed_ui import project_archive as archives
from packed_bed_ui.inputs import definition_payload
from packed_bed_ui.project import Project, read_json, write_json
from packed_bed_ui.studies import Factor, ReusableDefinition, generation_signature
from packed_bed_ui.worker import activate_snapshot


EXAMPLES = Path(__file__).parents[2] / "packed_bed/examples"


@pytest.fixture
def project(tmp_path, source_case):
    project = Project.create(tmp_path / "workspace", "Original")
    case = project.add_case_from_files(source_case)
    job = read_json(project.prepare_execution([case]))
    run = activate_snapshot(case.root / f".pending-{job['attempt_id']}")
    write_json(run / "status.json", {"state": "completed"})
    (run / "results.txt").write_text("retained results")
    return project


def tree(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def transfer(project, tmp_path):
    archive = archives.export_project(project, tmp_path / "transfer.msproject")
    assert archives.archive_name(archive) == project.metadata["name"]
    destination = archives.import_project(archive, tmp_path / "imported", "Imported")
    return Project.open(destination)


def rewrite(archive, edit):
    with ZipFile(archive) as source:
        files = {name: source.read(name) for name in source.namelist()}
    edit(files)
    with ZipFile(archive, "w") as target:
        for name, value in files.items():
            target.writestr(name, value)


def test_empty_project(tmp_path):
    project = Project.create(tmp_path / "empty")
    imported = transfer(project, tmp_path)
    assert imported.cases == []
    assert set(tree(imported.root)) == {"project.json"}


def test_mixed_round_trip_keeps_definitions_and_excludes_execution(project, tmp_path):
    case = project.cases[0]
    case.metadata.update(report={"version": 1, "sheets": []}, included=False,
                         step_ids={"inlet_flow": ["step_1"]}, program_modes={"pressure": {"draft": ""}})
    case.save()
    study = project.study_store.create("Sweep", case)
    study.factors = [Factor("cells", "axial_cells", [3, 4])]
    project.study_store.save_study(study)
    generated = project.study_store.apply_preview(project.study_store.preview(study))
    project.study_store.save_definition(ReusableDefinition("unused", "Unused", "program",
                                       definition_payload("program", case.documents)))
    project.add_case("Unfinished")
    project.metadata["results_report"] = {"version": 1, "case_ids": [case.id, "deleted"], "sheets": []}
    project.save()
    # Exclude unrelated files even when they resemble inputs.
    for name in ("notes.yaml", ".drafts/case.json", ".study-transaction/new/file", "backup/project.json"):
        path = project.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded")
    original = tree(project.root)
    imported = transfer(project, tmp_path)
    assert tree(project.root) == original
    expected = deepcopy(project.metadata)
    expected.pop("study_transaction", None)
    expected["name"] = "Imported"
    assert imported.metadata == expected
    assert [c.documents for c in imported.cases] == [c.documents for c in project.cases]
    assert imported.study_store.studies[study.id] == project.study_store.studies[study.id]
    assert imported.study_store.definitions == project.study_store.definitions
    assert not imported.study_store.needs_update(study.id)
    assert [c.id for c in imported.cases if c.metadata.get("study_id")] == [c.id for c in generated]
    assert all(c.state()["state"] == "not_run" and not c.run_folder.exists() for c in imported.cases)
    assert imported.cases[-1].state()["inputs"] == "Underdefined"
    assert not any("run/" in name or name.startswith((".", "backup")) for name in tree(imported.root))
    shutil.rmtree(project.root)
    assert Project.open(imported.root).cases[0].resolve()


@pytest.mark.parametrize("needs_update", [False, True])
def test_baseline_survives_deleted_source_and_preserves_rebuild_state(project, tmp_path, needs_update):
    case = project.cases[0]
    study = project.study_store.create("Established", case)
    study.factors = [Factor("cells", "axial_cells", [3])]
    project.study_store.save_study(study)
    project.study_store.apply_preview(project.study_store.preview(study))
    if needs_update:
        study.factors[0].values = [4]
        project.study_store.save_study(study)
    project.delete_case(case)
    imported = transfer(project, tmp_path)
    assert imported.study_store.needs_update(study.id) == needs_update
    baseline = imported.study_store.studies[study.id]
    assert baseline.provenance == study.provenance
    # Established provenance still permits rebuilding, without the source run.
    generated = imported.study_store.apply_preview(imported.study_store.preview(baseline))
    assert len(generated) == 1


def test_pending_advanced_study_keeps_portable_sources(tmp_path):
    project = Project.create(tmp_path / "source")
    source = tmp_path / "batch"
    shutil.copytree(EXAMPLES / "default_batch_case", source)
    path = source / "batch.yaml"
    data = yaml.safe_load(path.read_text())
    data["axes"][0]["values"][0]["patch"] = {"run": {"solver": {"maximum_order": 2}}}
    path.write_text(yaml.safe_dump(data))
    study = project.import_study(path)
    assert study.legacy is not None
    imported = transfer(project, tmp_path)
    shutil.rmtree(source)
    assert tree(imported.root / "studies") == tree(project.root / "studies")
    restored = imported.study_store.studies[study.id]
    assert restored == study
    with pytest.raises(ValueError, match="successful"):
        imported.study_store.apply_preview(imported.study_store.preview(restored))


def test_external_study_sources_are_copied_and_keep_generation_signature(tmp_path):
    project = Project.create(tmp_path / "project")
    source = tmp_path / "batch"
    shutil.copytree(EXAMPLES / "default_batch_case", source)
    path = source / "batch.yaml"
    data = yaml.safe_load(path.read_text())
    data["axes"][0]["values"][0]["patch"] = {"run": {"solver": {"maximum_order": 2}}}
    path.write_text(yaml.safe_dump(data))
    study = project.import_study(path)
    folder = project.root / "studies" / study.id
    batch = yaml.safe_load((folder / "batch.yaml").read_text())
    batch["base_case"] = str(folder / batch["base_case"])
    batch["programs"] = {key: str(folder / ref) for key, ref in batch["programs"].items()}
    for item in batch["geometries"].values():
        if item.get("solids_file"):
            item["solids_file"] = str(folder / item["solids_file"])
    (folder / "batch.yaml").write_text(yaml.safe_dump(batch))
    study.legacy["spec"] = deepcopy(batch)
    project.study_store.save_study(study)
    project.metadata["studies"][0]["generation_signature"] = generation_signature(
        study, project.study_store.definitions, [])
    project.save()
    imported = transfer(project, tmp_path)
    assert not imported.study_store.needs_update(study.id)
    assert not project.study_store.needs_update(study.id)
    shutil.rmtree(project.root)
    reexported = archives.export_project(imported, tmp_path / "again.msproject")
    assert reexported.is_file()


def test_external_case_inputs_and_advanced_drafts_are_preserved(project, tmp_path):
    case = project.cases[0]
    path = case.root / "inputs/run.yaml"
    run = yaml.safe_load(path.read_text())
    outside = tmp_path / "chemistry.yaml"
    outside.write_text(yaml.safe_dump(case.documents["chemistry"]))
    run["references"]["chemistry_file"] = str(outside)
    run["solver"].update(backend="compiled", maximum_order=3)
    run["model"]["bed_radius_m"] = "unfinished"
    path.write_text(yaml.safe_dump(run))
    imported = transfer(project, tmp_path)
    outside.unlink()
    restored = imported.cases[0].documents
    assert restored["chemistry"] == case.documents["chemistry"]
    assert restored["run"]["solver"] == run["solver"]
    assert restored["run"]["model"]["bed_radius_m"] == "unfinished"
    assert restored["run"]["references"]["chemistry_file"] == "chemistry.yaml"


def test_plugins_use_current_bytes_and_keep_historical_baseline(project, tmp_path):
    folder = tmp_path / "plugin"
    folder.mkdir()
    manifest = {"id": "lab", "name": "Lab", "species": {
        "N2": builtin_manifest().species["N2"].model_dump(mode="json")}}
    (folder / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    entry = project.plugins.add(folder)
    project.plugins.replace_uses("species", "builtin:N2", "lab:N2")
    case = project.cases[0]
    job = read_json(project.prepare_execution([case]))
    run = activate_snapshot(case.root / f".pending-{job['attempt_id']}")
    write_json(run / "status.json", {"state": "completed"})
    study = project.study_store.create("Before edit", case)
    installed = project.root / "plugins/lab/current"
    manifest["species"]["N2"]["mw"] *= 1.1
    (installed / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    imported = transfer(project, tmp_path)
    assert imported.metadata["plugins"][0]["hash"] == package_hash(installed) != entry["hash"]
    assert project.metadata["plugins"][0]["hash"] == entry["hash"]
    assert imported.study_store.studies[study.id].provenance == study.provenance
    assert imported.cases[0].validate_for_run()


def test_import_and_browsing_never_run_code_or_copy_approval(project, tmp_path, monkeypatch):
    import packed_bed.plugins.storage as storage
    monkeypatch.setenv("MULTISOLID_PLUGIN_APPROVALS", str(tmp_path / "approvals.json"))
    folder = tmp_path / "code"
    shutil.copytree(EXAMPLES / "plugins/enthalpy_overrides", folder)
    module = folder / "correlation.py"
    module.write_text('raise RuntimeError("archive executed code")\n' + module.read_text())
    entry = project.plugins.add(folder)
    entry["enabled"] = True  # Enablement copied from another computer grants no permission.
    project.metadata["plugins"][0]["enabled"] = True
    case = project.cases[0]
    case.documents["chemistry"].update(gas_species=["CO2_demo"], species_definitions={
        "CO2_demo": f"{entry['id']}:CO2"})
    case.documents["program"]["inlet_composition"]["initial"] = {"CO2_demo": 1.0}
    case.save()
    with monkeypatch.context() as patch:
        patch.setattr(storage, "approved_hashes", lambda: pytest.fail("Archive accessed local approvals"))
        imported = transfer(project, tmp_path)
    assert not (tmp_path / "approvals.json").exists()
    assert imported.plugins.browse_catalogue()[0].manifests
    assert imported.cases[0].resolve()
    with pytest.raises(ValueError, match="approval"):
        imported.cases[0].validate_for_run()


def test_disabled_unused_plugin_is_preserved(project, tmp_path):
    project.plugins.add(EXAMPLES / "plugins/enthalpy_overrides")
    assert not project.plugins.entries[0]["enabled"]
    imported = transfer(project, tmp_path)
    assert imported.plugins.entries == project.plugins.entries
    assert tree(imported.root / "plugins") == tree(project.root / "plugins")


@pytest.mark.parametrize("path", ["../escaped", "/absolute", "C:/escape", "a\\b", "NUL.txt",
                                  "a./file", "a /file", "a//b", "a/./b", "a:b"])
def test_unsafe_archive_paths_rejected(tmp_path, path):
    archive = tmp_path / "bad.msproject"
    with ZipFile(archive, "w") as out:
        out.writestr("archive.json", json.dumps(archives.ARCHIVE))
        out.writestr(path, "x")
    with pytest.raises(ValueError, match="Unsafe|portable"):
        archives.import_project(archive, tmp_path / "new", "New")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("names", [("a/x", "A/y"), ("a", "a/x"), ("a/x", "a"),
                                   ("é/x", "e\u0301/y"), ("same", "same")])
@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")
def test_duplicate_and_colliding_archive_paths(tmp_path, names):
    archive = tmp_path / "collision.zip"
    with ZipFile(archive, "w") as out:
        for name in names:
            out.writestr(name, "x")
    with pytest.raises(ValueError, match="[Cc]ollide|[Cc]ollision|[Dd]uplicate"):
        archives.import_project(archive, tmp_path / "new", "New")


@pytest.mark.parametrize("mode", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFDIR])
def test_special_archive_members_rejected(tmp_path, mode):
    archive = tmp_path / "special.zip"
    with ZipFile(archive, "w") as out:
        item = ZipInfo("special")
        item.create_system = 3
        item.external_attr = (mode | 0o777) << 16
        out.writestr(item, "target")
    with pytest.raises(ValueError, match="Unsupported archive member"):
        archives.import_project(archive, tmp_path / "new", "New")


@pytest.mark.parametrize("damage", ["version", "project_version", "missing", "execution", "identity", "reference",
                                    "hash", "missing_cases", "plugin_api", "missing_plugin"])
def test_damaged_payloads_are_rejected_before_destination_creation(project, tmp_path, damage):
    project.plugins.add(EXAMPLES / "plugins/nitrogen_oxides")
    archive = archives.export_project(project, tmp_path / "project.msproject")
    def edit(files):
        if damage == "version":
            files["archive.json"] = json.dumps({**archives.ARCHIVE, "format_version": 99})
        elif damage == "missing":
            files.pop(f"cases/{project.cases[0].id}/inputs/solids.yaml")
        elif damage == "execution":
            files[".study-transaction/transaction.json"] = "{}"
        elif damage in ("plugin_api", "missing_plugin"):
            manifest = next(name for name in files if name.endswith("/manifest.yaml"))
            if damage == "missing_plugin":
                files.pop(manifest)
            else:
                data = yaml.safe_load(files[manifest])
                data["api_version"] = 999
                files[manifest] = yaml.safe_dump(data)
        elif damage == "reference":
            path = f"cases/{project.cases[0].id}/inputs/run.yaml"
            data = yaml.safe_load(files[path])
            data["references"]["chemistry_file"] = "/outside.yaml"
            files[path] = yaml.safe_dump(data)
        else:
            metadata = json.loads(files["project.json"])
            if damage == "identity":
                metadata["cases"][0]["id"] = "../outside"
            elif damage == "hash":
                metadata["plugins"][0]["hash"] = "0" * 64
            elif damage == "missing_cases":
                metadata.pop("cases")
            else:
                metadata["format_version"] = 99
            files["project.json"] = json.dumps(metadata)
    rewrite(archive, edit)
    with pytest.raises(ValueError):
        archives.import_project(archive, tmp_path / "new", "New")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("limit", ["MAX_BYTES", "MAX_FILES"])
def test_archive_limits(project, tmp_path, monkeypatch, limit):
    archive = archives.export_project(project, tmp_path / "project.msproject")
    monkeypatch.setattr(archives, limit, 1)
    with pytest.raises(ValueError, match="limit"):
        archives.import_project(archive, tmp_path / "new", "New")


def test_corrupt_zip(tmp_path):
    archive = tmp_path / "bad.msproject"
    archive.write_bytes(b"broken zip")
    with pytest.raises(ValueError, match="Invalid project archive"):
        archives.archive_name(archive)


def test_nul_filename_is_rejected_before_extraction(tmp_path):
    archive = tmp_path / "nul.msproject"
    with ZipFile(archive, "w") as target:
        target.writestr("archive.json", json.dumps(archives.ARCHIVE))
        target.writestr("badXname", "x")
    archive.write_bytes(archive.read_bytes().replace(b"badXname", b"bad\0name"))
    with pytest.raises(ValueError, match="Unsupported archive member"):
        archives.import_project(archive, tmp_path / "new", "New")
    assert not (tmp_path / "new").exists()


def test_plugin_changes_during_copy_do_not_publish_an_archive(project, tmp_path, monkeypatch):
    project.plugins.add(EXAMPLES / "plugins/enthalpy_overrides")
    destination = tmp_path / "existing.msproject"
    destination.write_bytes(b"keep existing")
    original_copy = archives._copy
    def copy(source, target, cancelled):
        original_copy(source, target, cancelled)
        if str(getattr(source, "name", "")).endswith("correlation.py"):
            target.write(b"\n# changed while copying\n")
    monkeypatch.setattr(archives, "_copy", copy)
    with pytest.raises(ValueError, match="hash"):
        archives.export_project(project, destination)
    assert destination.read_bytes() == b"keep existing"


def test_cancellation_and_failed_copy_preserve_existing_data(project, tmp_path, monkeypatch):
    archive = archives.export_project(project, tmp_path / "project.msproject")
    original = archive.read_bytes()
    with pytest.raises(InterruptedError):
        archives.export_project(project, archive, cancelled=lambda: bool(list(tmp_path.glob(".project.msproject.*.tmp"))))
    assert archive.read_bytes() == original
    assert not list(tmp_path.glob(".project.msproject.*.tmp"))
    destination = tmp_path / "new"
    with pytest.raises(InterruptedError):
        archives.import_project(archive, destination, "New", cancelled=destination.exists)
    assert not destination.exists()
    write = archives._write_files
    def fail(folder, files, cancelled):
        if folder == destination:
            (folder / "partial").write_text("partial")
            raise OSError("disk full")
        return write(folder, files, cancelled)
    monkeypatch.setattr(archives, "_write_files", fail)
    with pytest.raises(OSError, match="disk full"):
        archives.import_project(archive, destination, "New")
    assert not destination.exists()
    for populated in (False, True):
        destination.mkdir(exist_ok=True)
        if populated:
            (destination / "existing").write_text("keep")
        before = tree(destination)
        with pytest.raises(FileExistsError):
            archives.import_project(archive, destination, "New")
        assert tree(destination) == before


def test_source_symlinks_and_missing_plugins_block_export(project, tmp_path):
    case = project.cases[0]
    case.documents["chemistry"]["species_definitions"] = {"N2": "missing:N2"}
    case.save()
    with pytest.raises(ValueError, match="Required plugin is missing"):
        archives.export_project(project, tmp_path / "project.msproject")
    case.documents["chemistry"].pop("species_definitions")
    case.save()
    source = case.root / "inputs/solids.yaml"
    outside = tmp_path / "outside.yaml"
    source.rename(outside)
    try:
        source.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks unavailable")
    with pytest.raises(ValueError, match="regular file"):
        archives.export_project(project, tmp_path / "project.msproject")


def test_import_menu_location_dialog_and_recent_registration(qt_app, project, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    from packed_bed_ui.navigation import NewProjectDialog
    from packed_bed_ui.window import MainWindow
    archive = archives.export_project(project, tmp_path / "project.msproject")
    window = MainWindow()
    actions = {action.text(): action for action in window.project_actions}
    assert actions["Import project archive…"].isEnabled()
    assert not actions["Export project…"].isEnabled()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **kw: (str(archive), ""))
    def choose(dialog):
        assert dialog.name.text() == "Original"
        assert dialog.windowTitle() == "Import project"
        dialog.name.setText("Chosen name")
        dialog.location.setText(str(tmp_path))
        dialog.accept()
        return dialog.result()
    monkeypatch.setattr(NewProjectDialog, "exec", choose)
    window._import_archive()
    assert window.project.metadata["name"] == "Chosen name"
    assert window.project.root == tmp_path / "Chosen name"
    assert window.locations.entries[0]["path"] == str(window.project.root)
    assert actions["Export project…"].isEnabled()
    window.close()


def test_archive_actions_flush_edits_and_leave_current_project_on_cancel(qt_app, project, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    from packed_bed_ui.window import MainWindow
    window = MainWindow()
    window._set_project(project)
    previous = deepcopy(window.locations.entries)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **kw: ("", ""))
    window._import_archive()
    assert window.project is project and window.locations.entries == previous
    monkeypatch.setattr(window, "_save_editors", lambda: False)
    monkeypatch.setattr(QFileDialog, "exec", lambda *a, **kw: pytest.fail("Unsaved export"))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **kw: pytest.fail("Unsaved import"))
    window._export_archive()
    window._import_archive()
    monkeypatch.setattr(window, "_save_editors", lambda: True)
    window.close()


def test_export_menu_saves_pending_edits(qt_app, project, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog, QDialog
    from packed_bed_ui.window import MainWindow
    window = MainWindow()
    window._set_project(project)
    case = project.cases[0]
    window._show_case(case)
    case.documents["run"]["model"]["bed_length_m"] = 2.0
    window.editor.dirty = True
    destination = tmp_path / "edited.msproject"
    def choose(dialog):
        assert dialog.defaultSuffix() == "msproject"
        dialog.selectFile(str(destination.with_suffix("")))
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QFileDialog, "exec", choose)
    window._export_archive()
    imported = Project.open(archives.import_project(destination, tmp_path / "new", "New"))
    assert imported.cases[0].documents["run"]["model"]["bed_length_m"] == 2.0
    window.close()
