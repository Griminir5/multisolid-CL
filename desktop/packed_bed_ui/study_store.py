"""Project-owned study storage and recoverable, complete case replacement."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
from uuid import uuid4

import yaml

from packed_bed.config.load import read_yaml_mapping
from packed_bed.file_io import retry_file_operation

from .inputs import new_step_ids
from .project import (DOCUMENTS, ProjectCase, input_hashes, portable_documents, read_documents,
                      read_json, scientific_fingerprint, write_json, write_text)
from .studies import (ExistingCase, ReusableDefinition, Study, StudyError,
                      generation_signature, preview_study, referenced_definitions)


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
        retry_file_operation(shutil.rmtree, folder)  # Only staging occurred; active files were never touched.
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
                    retry_file_operation(shutil.rmtree, target)
                target.parent.mkdir(parents=True, exist_ok=True)
                retry_file_operation(previous.rename, target)
            elif not operation["existed"] and not staged.exists() and target.exists():
                retry_file_operation(shutil.rmtree, target)
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
    retry_file_operation(shutil.rmtree, folder)


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
                                                             study.provenance.get("extensions", []))
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

    def _commit(self, folders, metadata, deleted_case_ids=(), *, cancelled=lambda: False, progress=lambda *_: None):
        """Stage all folders, record rollback information, then commit project.json last."""
        self._require_idle()
        root = self.project.root
        recover_study_transaction(root)
        original_manifest = (root / "project.json").read_bytes()
        transaction = root / TRANSACTION
        transaction.mkdir()
        identity = uuid4().hex
        operations = []
        try:
            entries = folders.items() if isinstance(folders, dict) else folders
            for index, (relative, contents) in enumerate(entries):
                if cancelled():
                    raise InterruptedError("Study generation cancelled; existing cases are unchanged.")
                target = _inside(root, relative)
                operations.append({"path": relative, "existed": target.exists(), "replacement": contents is not None})
                if contents is not None:
                    staged = transaction / "new" / str(index)
                    staged.mkdir(parents=True)
                    for name, text in contents.items():
                        path = _inside(staged, name)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if isinstance(text, bytes):
                            path.write_bytes(text)
                        else:
                            write_text(path, text)
            if cancelled():
                raise InterruptedError("Study generation cancelled; existing cases are unchanged.")
            if (root / "project.json").read_bytes() != original_manifest:
                raise StudyError("The project changed during staging. Review a fresh preview before replacing cases.")
            progress("Committing generated cases…")
            write_json(transaction / "transaction.json", {"id": identity, "operations": operations,
                                                          "deleted_case_ids": list(deleted_case_ids)})
            for index, operation in enumerate(operations):
                target = _inside(root, operation["path"])
                if operation["existed"]:
                    previous = transaction / "old" / str(index)
                    previous.parent.mkdir(parents=True, exist_ok=True)
                    retry_file_operation(target.rename, previous)
                if operation["replacement"]:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    retry_file_operation((transaction / "new" / str(index)).rename, target)
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

    def _adopt_metadata(self, metadata, case_documents=None):
        self.project.metadata = metadata
        existing = {case.id: case for case in self.project.cases}
        cases = []
        for entry in metadata["cases"]:
            case = existing.get(entry["id"])
            if case is None:
                documents = (case_documents[entry["id"]] if case_documents is not None else
                             read_documents(self.project.root / "cases" / entry["id"] / "inputs"))
                case = ProjectCase(self.project, entry, documents)
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
        lock = study.provenance.get('extensions', [])
        return preview_study(study, self.definitions, self.existing_cases(study.id),
                             lock, lazy=lazy, catalogue=self.project.plugins.catalogue())

    def existing_cases(self, study_id):
        return [ExistingCase(case.id, study_id, case.run_folder.exists()) for case in self.project.cases
                if case.metadata.get("study_id") == study_id]

    def apply_preview(self, preview, *, cancelled=lambda: False, progress=lambda *_: None):
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
        from itertools import zip_longest
        for index, (actual, reviewed) in enumerate(zip_longest(current.remaining, preview.candidates)):
            if cancelled():
                raise InterruptedError("Study generation cancelled; existing cases are unchanged.")
            if actual != reviewed:
                raise StudyError("The candidate inputs changed. Review a fresh preview.")
            progress(f"Checking case {index + 1}/{preview.total}…")
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
        def staged_folders():
            yield from folders.items()
            for index, candidate in enumerate(preview.candidates):
                ident = uuid4().hex
                new_ids.append(ident)
                entry = {"id": ident, "name": candidate.name, "included": True, "origin": study.name,
                         "study_id": study.id, "selections": candidate.selections}
                if report is not None:
                    entry["report"] = deepcopy(report)
                metadata["cases"].append(entry)
                progress(f"Staging case {index + 1}/{preview.total}…")
                yield f"cases/{ident}", _documents(portable_documents(candidate.documents), "inputs/")
        next(entry for entry in metadata["studies"] if entry["id"] == study.id)["generation_signature"] = current.signature
        self._commit(staged_folders(), metadata, preview.delete_ids, cancelled=cancelled, progress=progress)
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
        # Compatibility for pending studies saved before desktop batch import was removed.
        return self.project.add_case(f"{study.name} baseline", study.baseline)
