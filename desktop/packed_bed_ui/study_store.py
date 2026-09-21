"""Project-owned study storage and recoverable, complete case replacement."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
from uuid import uuid4

import yaml

from packed_bed.batch import load_batch_spec
from packed_bed.config.load import read_yaml_mapping, resolve_path

from .inputs import BED_RUN_FIELDS, definition_payload, new_step_ids
from .project import (DOCUMENTS, ProjectCase, input_hashes, portable_documents, read_documents,
                      read_json, scientific_fingerprint, write_json, write_text)
from .studies import (ExistingCase, Factor, ReusableDefinition, Study, StudyError, expand_study,
                      generation_signature, parameter_catalogue, prepared_factors,
                      preview_study, referenced_definitions)


TRANSACTION = ".study-transaction"


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    message: str = ""


def baseline_eligibility(case):
    state = case.state()
    if state["state"] != "completed":
        return Eligibility(False, "The latest run must have succeeded.")
    if state["inputs"] != "Ready":
        return Eligibility(False, "The current inputs must be ready and up to date.")
    if state["stale"]:
        return Eligibility(False, "Inputs have changed since the successful run. Run this case again.")
    try:
        snapshot = read_json(case.run_folder / "snapshot.json")
        documents = read_documents(case.run_folder / "inputs")
        if not snapshot.get("attempt_id") or snapshot.get("case_id") != case.id:
            raise ValueError("Run identity is missing or does not match this case.")
        if snapshot.get("input_hashes") != input_hashes(case.run_folder / "inputs"):
            raise ValueError("The successful input snapshot has changed.")
        if scientific_fingerprint(documents, snapshot.get("extensions", [])) != case.fingerprint():
            raise ValueError("The successful snapshot does not match the current inputs.")
    except (ValueError, OSError) as exc:
        return Eligibility(False, f"Run this case again: {exc}")
    return Eligibility(True)


def _identity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
        raise ValueError("Invalid study or definition identity.")
    return value


def _inside(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path == root:
        raise ValueError("Study transaction paths must stay inside the project.")
    return path


def recover_study_transaction(root):
    """Called while holding the project lock, before loading any case inputs."""
    root = Path(root)
    folder = root / TRANSACTION
    if not folder.exists():
        return
    manifest = folder / "transaction.json"
    if not manifest.exists():
        shutil.rmtree(folder)  # Only staging occurred; active files were never touched.
        return
    transaction = read_json(manifest)
    committed = read_json(root / "project.json").get("study_transaction") == transaction["id"]
    if not committed:
        for index, operation in reversed(list(enumerate(transaction["operations"]))):
            target = _inside(root, operation["path"])
            previous = folder / "old" / str(index)
            staged = folder / "new" / str(index)
            if previous.exists():
                if target.exists():
                    shutil.rmtree(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                previous.rename(target)
            elif not operation["existed"] and not staged.exists() and target.exists():
                shutil.rmtree(target)
    else:
        # Execution summaries are diagnostic, and must not reference deleted cases.
        path = root / "execution.json"
        if path.exists():
            try:
                job = read_json(path)
            except ValueError:
                path.unlink()  # A corrupt diagnostic summary must not prevent recovery.
            else:
                for ident in transaction.get("deleted_case_ids", []):
                    job.get("cases", {}).pop(ident, None)
                write_json(path, job)
    shutil.rmtree(folder)


def _json(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def _documents(documents, prefix=""):
    return {f"{prefix}{name}.yaml": yaml.safe_dump(documents[name], sort_keys=False) for name in DOCUMENTS}


class StudyStore:
    def __init__(self, project):
        self.project = project
        self._signatures = {}
        self.reload()

    def reload(self):
        self.studies, self.definitions = {}, {}
        for entry in self.project.metadata.get("studies", []):
            folder = self.project.root / "studies" / _identity(entry["id"])
            if (folder / "study.json").exists():
                self.studies[entry["id"]] = Study.from_rule(read_json(folder / "study.json"), read_documents(folder / "baseline"))
        for entry in self.project.metadata.get("definitions", []):
            folder = self.project.root / "definitions" / _identity(entry["id"])
            data = read_json(folder / "definition.json")
            document = "program" if data["kind"] == "program" else "solids"
            data["payload"][document] = read_yaml_mapping(folder / f"{document}.yaml", document)
            self.definitions[entry["id"]] = ReusableDefinition(**data)
        self.invalidate()

    def invalidate(self):
        self._signatures.clear()

    def signature(self, study):
        # Cache only until a source save; previews always compute their own fresh signature.
        if study.id not in self._signatures:
            self._signatures[study.id] = generation_signature(study, self.definitions,
                                                             self.project.metadata.get("extensions", []))
        return self._signatures[study.id]

    def needs_update(self, study_id):
        study = self.studies.get(study_id)
        if study is None:
            return False
        entry = next(entry for entry in self.project.metadata["studies"] if entry["id"] == study_id)
        applied = entry.get("generation_signature")
        return applied is not None and applied != self.signature(study)

    def _require_idle(self):
        if getattr(self.project, "executing", False) or (self.project.root / ".solver.lock").exists():
            raise StudyError("Wait for execution to finish before changing studies or definitions.")

    def _commit(self, folders, metadata, deleted_case_ids=()):
        """Stage all folders, record rollback information, then commit project.json last."""
        self._require_idle()
        root = self.project.root
        recover_study_transaction(root)
        transaction = root / TRANSACTION
        transaction.mkdir()
        identity = uuid4().hex
        operations = []
        try:
            for index, (relative, contents) in enumerate(folders.items()):
                target = _inside(root, relative)
                operations.append({"path": relative, "existed": target.exists(), "replacement": contents is not None})
                if contents is not None:
                    staged = transaction / "new" / str(index)
                    staged.mkdir(parents=True)
                    for name, text in contents.items():
                        path = _inside(staged, name)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        write_text(path, text)
            write_json(transaction / "transaction.json", {"id": identity, "operations": operations,
                                                          "deleted_case_ids": list(deleted_case_ids)})
            for index, operation in enumerate(operations):
                target = _inside(root, operation["path"])
                if operation["existed"]:
                    previous = transaction / "old" / str(index)
                    previous.parent.mkdir(parents=True, exist_ok=True)
                    target.rename(previous)
                if operation["replacement"]:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    (transaction / "new" / str(index)).rename(target)
            metadata = deepcopy(metadata)
            metadata["study_transaction"] = identity
            write_json(root / "project.json", metadata)
        except Exception:
            recover_study_transaction(root)
            self._adopt_metadata(read_json(root / "project.json"))
            raise
        # Once metadata commits, the replacement is authoritative even if cleanup is interrupted.
        self._adopt_metadata(metadata)
        recover_study_transaction(root)
        self.project.edited()

    def save_case(self, case):
        # Reuse the existing folder transaction so a crash cannot save only some
        # of the four YAML documents. The retained run folder is never replaced.
        metadata = deepcopy(case.metadata)
        try:
            self._commit({f"cases/{case.id}/inputs": _documents(case.documents)}, self.project.metadata)
        except Exception:
            # Transaction rollback reloads saved metadata; keep the editor's draft.
            case.metadata.clear()
            case.metadata.update(metadata)
            raise

    def _adopt_metadata(self, metadata):
        self.project.metadata = metadata
        existing = {case.id: case for case in self.project.cases}
        cases = []
        for entry in metadata["cases"]:
            case = existing.get(entry["id"])
            if case is None:
                case = ProjectCase(self.project, entry, read_documents(self.project.root / "cases" / entry["id"] / "inputs"))
            else:
                case.metadata = entry
            cases.append(case)
        self.project.cases = cases
        self.reload()

    def _study_files(self, study):
        files = {"study.json": _json(study.rule()), **_documents(study.baseline, "baseline/")}
        # Keep original portable batch sources for advanced imports and compatibility.
        folder = self.project.root / "studies" / study.id
        if folder.exists():
            for path in folder.rglob("*"):
                if path.is_file() and path.relative_to(folder).parts[0] not in ("baseline", "study.json"):
                    files[path.relative_to(folder).as_posix()] = path.read_text(encoding="utf-8")
        return files

    def save_study(self, study, *, _replace_baseline=False):
        if not study.name.strip():
            raise StudyError("Enter a study name.")
        _identity(study.id)
        previous = self.studies.get(study.id)
        if previous and not _replace_baseline and any(
                getattr(study, key) != getattr(previous, key) for key in ("baseline", "provenance", "editor_metadata")):
            raise StudyError("The baseline is read-only. Select another successful case to replace it.")
        metadata = deepcopy(self.project.metadata)
        entry = next((value for value in metadata["studies"] if value["id"] == study.id), None)
        if entry is None:
            entry = {"id": study.id}
            metadata["studies"].append(entry)
        entry["name"] = study.name.strip()
        self._commit({f"studies/{study.id}": self._study_files(study)}, metadata)

    def create(self, name, case):
        return self.replace_baseline(Study(uuid4().hex, name.strip(), {}), case)

    def delete_study(self, study_id):
        _identity(study_id)
        metadata = deepcopy(self.project.metadata)
        metadata["studies"] = [entry for entry in metadata["studies"] if entry["id"] != study_id]
        removed = [case.id for case in self.project.cases if case.metadata.get("study_id") == study_id]
        metadata["cases"] = [entry for entry in metadata["cases"] if entry["id"] not in removed]
        folders = {f"studies/{study_id}": None, **{f"cases/{ident}": None for ident in removed}}
        self._commit(folders, metadata, removed)

    def replace_baseline(self, study, case):
        if case.project is not self.project or not any(case is member for member in self.project.cases):
            raise StudyError("Select a case belonging to this project.")
        eligibility = baseline_eligibility(case)
        if not eligibility.eligible:
            raise StudyError(eligibility.message)
        study = deepcopy(study)
        snapshot = read_json(case.run_folder / "snapshot.json")
        study.baseline = portable_documents(case.documents)
        study.provenance = {"case_id": case.id, "case_name": case.name,
                            "attempt_id": snapshot["attempt_id"], "fingerprint": case.fingerprint(),
                            "extensions": deepcopy(snapshot.get("extensions", []))}
        study.editor_metadata = {"step_ids": new_step_ids(study.baseline["program"]),
                                 "report": deepcopy(case.metadata.get("report"))}
        self.save_study(study, _replace_baseline=True)
        return study

    def _verify_baseline(self, study):
        provenance = study.provenance
        if not provenance.get("attempt_id") or scientific_fingerprint(
                study.baseline, provenance.get("extensions", [])) != provenance.get("fingerprint"):
            raise StudyError("Select a successful, unchanged baseline before generating study cases.")

    def preview(self, study, *, lazy=False):
        return preview_study(study, self.definitions, self.existing_cases(study.id),
                             self.project.metadata.get("extensions", []), lazy=lazy)

    def existing_cases(self, study_id):
        return [ExistingCase(case.id, study_id, case.run_folder.exists()) for case in self.project.cases
                if case.metadata.get("study_id") == study_id]

    def apply_preview(self, preview):
        self._require_idle()
        # Read committed sources again: a stale preview must never authorize deletion.
        self.reload()
        study = self.studies[preview.study_id]
        self._verify_baseline(study)
        current = self.preview(study, lazy=True)
        if current.signature != preview.signature or set(current.delete_ids) != set(preview.delete_ids):
            raise StudyError("The study changed after this preview. Review a fresh preview before replacing cases.")
        if not preview.candidates:
            raise StudyError("Add at least one case before generating the study.")
        # Verify the reviewed documents as well; callers cannot substitute an arbitrary candidate list.
        if list(current.remaining) != preview.candidates:
            raise StudyError("The candidate inputs changed. Review a fresh preview.")
        metadata = deepcopy(self.project.metadata)
        metadata["cases"] = [entry for entry in metadata["cases"] if entry["id"] not in preview.delete_ids]
        folders = {f"cases/{ident}": None for ident in current.delete_ids}
        report = study.editor_metadata.get("report")
        # Studies created before report inheritance can still copy their source's layout.
        if "report" not in study.editor_metadata:
            source = next((case for case in self.project.cases if case.id == study.provenance.get("case_id")), None)
            report = deepcopy(source.metadata.get("report")) if source is not None else None
            study.editor_metadata["report"] = report
            folders[f"studies/{study.id}"] = self._study_files(study)
        new_ids = []
        for candidate in preview.candidates:
            ident = uuid4().hex
            new_ids.append(ident)
            entry = {"id": ident, "name": candidate.name, "included": True, "origin": study.name,
                     "study_id": study.id, "selections": candidate.selections}
            if report is not None:
                entry["report"] = deepcopy(report)
            metadata["cases"].append(entry)
            folders[f"cases/{ident}"] = _documents(portable_documents(candidate.documents), "inputs/")
        next(entry for entry in metadata["studies"] if entry["id"] == study.id)["generation_signature"] = current.signature
        self._commit(folders, metadata, preview.delete_ids)
        return [case for case in self.project.cases if case.id in new_ids]

    def definition_users(self, definition_id):
        return [study for study in self.studies.values() if definition_id in referenced_definitions(study)]

    @staticmethod
    def _definition_files(definition):
        data = asdict(definition)
        document = "program" if definition.kind == "program" else "solids"
        inputs = data["payload"].pop(document)
        return {"definition.json": _json(data), f"{document}.yaml": yaml.safe_dump(inputs, sort_keys=False)}

    def save_definition(self, definition):
        _identity(definition.id)
        if definition.kind not in ("program", "bed") or not definition.name.strip():
            raise StudyError("Enter a definition name and choose Program or Bed configuration.")
        metadata = deepcopy(self.project.metadata)
        entries = metadata.setdefault("definitions", [])
        entries[:] = [value for value in entries if value["id"] != definition.id]
        entries.append({"id": definition.id, "name": definition.name, "kind": definition.kind})
        self._commit({f"definitions/{definition.id}": self._definition_files(definition)}, metadata)

    def delete_definition(self, ident):
        users = self.definition_users(ident)
        if users:
            raise StudyError("This definition is used by: " + ", ".join(study.name for study in users))
        metadata = deepcopy(self.project.metadata)
        metadata["definitions"] = [entry for entry in metadata.get("definitions", []) if entry["id"] != ident]
        self._commit({f"definitions/{_identity(ident)}": None}, metadata)

    def add_imported_baseline(self, study):
        return self.project.add_case(f"{study.name} baseline", study.baseline)

    def _read_import(self, path, name, study_id=None):
        document = load_batch_spec(path)
        base_path = resolve_path(document.base_dir, document.spec.base_case)
        run = read_yaml_mapping(base_path, "run")
        baseline = {"run": run, **{key: read_yaml_mapping(resolve_path(base_path.parent, run["references"][f"{key}_file"]), key)
                                  for key in DOCUMENTS[1:]}}
        study = Study(study_id or uuid4().hex, name, portable_documents(baseline),
                      editor_metadata={"step_ids": new_step_ids(baseline["program"])})
        definitions, programs, beds = {}, {}, {}
        portable = document.spec.model_dump(mode="json")
        portable.update(base_case="base/run.yaml", output_directory="unused-output")
        source_files = _documents(study.baseline, "base/")
        for label, source in document.spec.programs.items():
            program = read_yaml_mapping(resolve_path(document.base_dir, source), label)
            payload = definition_payload("program", {**baseline, "program": program})
            definition = ReusableDefinition(uuid4().hex, label, "program", payload)
            definitions[definition.id] = definition
            programs[label] = definition.id
            destination = f"program-{definition.id}.yaml"
            source_files[destination] = yaml.safe_dump(program)
            portable["programs"][label] = destination
        for label, preset in document.spec.geometries.items():
            payload = definition_payload("bed", baseline)
            # Keep omitted settings inherited so importing preserves the authored rule exactly.
            for section in BED_RUN_FIELDS:
                payload[section] = {key: value for key, value in payload[section].items()
                                    if key in baseline["run"].get(section, {})}
            payload["model"].update({key: value for key, value in preset.model.items()
                                     if key in BED_RUN_FIELDS["model"]})
            if preset.solids_file:
                payload["solids"] = read_yaml_mapping(resolve_path(document.base_dir, preset.solids_file), label)
            definition = ReusableDefinition(uuid4().hex, label, "bed", payload)
            definitions[definition.id] = definition
            beds[label] = definition.id
            if preset.solids_file:
                destination = f"solids-{definition.id}.yaml"
                source_files[destination] = yaml.safe_dump(payload["solids"])
                portable["geometries"][label]["solids_file"] = destination
        source_files["batch.yaml"] = yaml.safe_dump(portable, sort_keys=False)
        study.legacy = {"spec": portable, "programs": programs, "beds": beds}
        self._convert_native(study, definitions)
        return study, definitions, source_files

    @staticmethod
    def _convert_native(study, definitions):
        """Convert only a provably identical subset; everything else keeps its original rule."""
        native = deepcopy(study)
        native.legacy = None
        catalogue = parameter_catalogue(native)
        for axis in study.legacy["spec"]["axes"]:
            target, values = None, []
            for value in axis["values"]:
                effects = [key for key in ("program", "geometry", "patch") if value.get(key)]
                if len(effects) != 1:
                    return
                effect = effects[0]
                if effect == "program":
                    this_target, raw = "definition:program", study.legacy["programs"][value[effect]]
                elif effect == "geometry":
                    preset = study.legacy["spec"]["geometries"][value[effect]]
                    if set(preset["model"]) - set(BED_RUN_FIELDS["model"]):
                        return
                    this_target, raw = "definition:bed", study.legacy["beds"][value[effect]]
                else:
                    patch = {key: val for key, val in value["patch"].items() if val}
                    path = []
                    while isinstance(patch, dict) and len(patch) == 1:
                        key, patch = next(iter(patch.items()))
                        path.append(key)
                    parameter = next((parameter for parameter in catalogue if parameter.path == tuple(path)), None)
                    if parameter is None or not isinstance(patch, (int, float)):
                        return
                    this_target, raw = parameter.id, patch
                if target is not None and target != this_target:
                    return
                target = this_target
                values.append(raw)
            native.factors.append(Factor(uuid4().hex, target, values))
        try:
            prepared_factors(native, definitions)
            # Compare documents, including derived timing and geometry. Ignore only display labels.
            for first, second in zip(expand_study(study, definitions), expand_study(native, definitions), strict=True):
                if first.documents != second.documents:
                    return
        except (ValueError, TypeError):
            return
        study.factors, study.legacy = native.factors, None

    def _stage_import(self, metadata, study, definitions, sources):
        folders = {f"studies/{study.id}": {**sources, **self._study_files(study)}}
        for definition in definitions.values():
            metadata.setdefault("definitions", []).append({"id": definition.id, "name": definition.name, "kind": definition.kind})
            folders[f"definitions/{definition.id}"] = self._definition_files(definition)
        return folders

    def import_batch(self, path, name=None):
        study, definitions, sources = self._read_import(path, name or Path(path).stem)
        metadata = deepcopy(self.project.metadata)
        metadata["studies"].append({"id": study.id, "name": study.name})
        self._commit(self._stage_import(metadata, study, definitions, sources), metadata)
        return deepcopy(self.studies[study.id])

    def migrate_v2(self):
        metadata = deepcopy(self.project.metadata)
        folders = {}
        for entry in metadata.get("studies", []):
            path = self.project.root / "studies" / _identity(entry["id"]) / "batch.yaml"
            if path.exists():
                study, definitions, sources = self._read_import(path, entry["name"], entry["id"])
                entry["generation_signature"] = generation_signature(study, definitions, metadata.get("extensions", []))
                folders.update(self._stage_import(metadata, study, definitions, sources))
        metadata["format_version"] = 3
        self._commit(folders, metadata)
