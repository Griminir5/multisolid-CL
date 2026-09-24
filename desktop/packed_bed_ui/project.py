"""Projects contain editable cases; each case has one replaceable run folder."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from uuid import uuid4

from pydantic import ValidationError
import yaml

from packed_bed.config import Case, CaseInputs
from packed_bed.config.load import read_yaml_mapping, inspect_case, inspect_case_file
from packed_bed.parameters import plain

from .inputs import scientific_documents


DOCUMENTS = ("run", "chemistry", "program", "solids")
REFERENCES = {f"{name}_file": f"{name}.yaml" for name in DOCUMENTS[1:]}
PROJECT_VERSION = 3
SNAPSHOT_VERSION = 1
TERMINAL_STATES = {"completed", "failed", "cancelled", "interrupted"}


class NeedsStudyUpdate(ValueError):
    pass


def write_json(path: Path, value: dict) -> None:
    write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def input_hashes(folder: Path) -> dict[str, str]:
    return {name: hashlib.sha256((folder / f"{name}.yaml").read_bytes()).hexdigest() for name in DOCUMENTS}


def read_documents(folder: Path) -> dict[str, dict]:
    return {name: read_yaml_mapping(folder / f"{name}.yaml", name) for name in DOCUMENTS}


def write_documents(folder: Path, documents: dict[str, dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in DOCUMENTS:
        write_text(folder / f"{name}.yaml", yaml.safe_dump(documents[name], sort_keys=False))


def portable_documents(documents: dict[str, dict]) -> dict[str, dict]:
    documents = deepcopy(documents)
    run = documents["run"]
    run["references"] = dict(REFERENCES)
    run.setdefault("outputs", {}).update(directory="../output", artifacts_directory="../output/artifacts")
    return documents


def scientific_fingerprint(documents: dict[str, dict], extensions: list) -> str:
    """Ignore file destinations, YAML formatting, and project display names."""
    payload = yaml.safe_dump({"inputs": scientific_documents(documents), "extensions": extensions}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def require_desktop_solver(case: Case | CaseInputs) -> None:
    if case.run.solver.backend != "daetools" or case.run.solver.name != "superlu":
        raise ValueError("This starter supports DAETools / SuperLU. Select it explicitly in the case inputs, or use the CLI for this solver.")


@dataclass
class ProjectCase:
    project: Project
    metadata: dict
    documents: dict[str, dict]

    @property
    def id(self) -> str:
        return self.metadata["id"]

    @property
    def name(self) -> str:
        return self.metadata["name"]

    @property
    def root(self) -> Path:
        return self.project.root / "cases" / self.id

    @property
    def run_folder(self) -> Path:
        return self.root / "run"

    def resolve(self) -> CaseInputs:
        if self.documents["run"].get("references") != REFERENCES:
            raise ValueError("Case inputs must reference chemistry.yaml, program.yaml and solids.yaml in inputs/.")
        return inspect_case(catalogue=self.catalogue(),
            **{f"{name}_path": self.root / "inputs" / f"{name}.yaml" for name in DOCUMENTS},
            **{f"{name}_data": self.documents[name] for name in DOCUMENTS},
        )

    def catalogue(self):
        return self.project.plugins.catalogue(self.definition_lock())

    def definition_lock(self):
        return self.project.plugins.lock_for(self.documents)

    def validate_for_run(self) -> CaseInputs:
        study_id = self.metadata.get("study_id")
        if study_id and self.project.study_store.needs_update(study_id):
            raise NeedsStudyUpdate("This study changed. Open Edit study and rebuild its generated cases before running.")
        case = self.resolve()
        if self.project.metadata.get("extensions"):
            raise ValueError("This project requires extensions in an unsupported legacy format.")
        require_desktop_solver(case)
        from packed_bed.plugins.storage import require_approval
        require_approval(self.catalogue(), case.selection.lock)
        return case

    def save(self) -> None:
        """Save incomplete drafts without altering this case's latest results."""
        changed = self.documents != read_documents(self.root / "inputs")
        if self.metadata.get("study_id"):
            if changed:
                raise ValueError("Generated inputs are read-only. Edit the study or duplicate this case.")
        if changed:
            self.project.study_store.save_case(self)
        else:
            self.project.save()

    def fingerprint(self) -> str:
        return scientific_fingerprint(self.documents, self.definition_lock())

    def state(self) -> dict:
        readiness, message = "Ready", ""
        try:
            self.validate_for_run()
        except (ValueError, OSError) as exc:
            message = str(exc)
            errors = exc.__cause__.errors() if isinstance(exc.__cause__, ValidationError) else []
            missing = any(error["type"] == "missing" or error.get("input") in (None, "") for error in errors)
            readiness = "Needs update" if isinstance(exc, NeedsStudyUpdate) else "Underdefined" if missing else "Invalid"
        result = {"state": "not_run", "elapsed_s": 0.0}
        stale = False
        if self.run_folder.exists():
            try:
                result = read_json(self.run_folder / "status.json")
                snapshot = read_json(self.run_folder / "snapshot.json")
                stale = snapshot.get("fingerprint") != self.fingerprint()
            except (OSError, ValueError) as exc:
                result = {"state": "interrupted", "message": f"Cannot read the latest run: {exc}"}
                stale = True
        return {"inputs": readiness, "input_message": message, "stale": stale, **result}

    def prepare(self, attempt_id: str) -> Path:
        """Stage a complete input snapshot. Existing results remain until execution starts."""
        resolved = self.validate_for_run()
        self.save()
        folder = self.root / f".pending-{attempt_id}"
        folder.mkdir(exist_ok=False)
        try:
            write_documents(folder / "inputs", portable_documents(self.documents))
            from packed_bed.plugins.storage import copy_package
            catalogue = self.catalogue()
            for provider in resolved.selection.lock['plugins']:
                copy_package(catalogue.paths[provider], folder)
            write_json(folder / 'inputs' / 'definitions.json', {'root': '..', 'lock': plain(resolved.selection.lock)})
            write_json(folder / "snapshot.json", {
                "format_version": SNAPSHOT_VERSION,
                "case_id": self.id,
                "case_name": self.name,
                "attempt_id": attempt_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "fingerprint": scientific_fingerprint(self.documents, plain(resolved.selection.lock)),
                "extensions": plain(resolved.selection.lock),
                "definitions": resolved.selection.to_dict(),
                "input_hashes": input_hashes(folder / "inputs"),
            })
            write_json(folder / "status.json", {"state": "queued", "elapsed_s": 0.0})
        except Exception:
            shutil.rmtree(folder)
            raise
        return folder


@dataclass
class Project:
    root: Path
    metadata: dict
    cases: list[ProjectCase] = field(default_factory=list)
    on_edit: object = field(default=None, repr=False, compare=False)

    @property
    def drafts(self):
        from .recovery import Drafts
        return Drafts(self)

    def edited(self):
        if self.on_edit is not None:
            self.on_edit()

    @property
    def study_store(self):
        from .study_store import StudyStore
        if not hasattr(self, "_study_store"):
            self._study_store = StudyStore(self)
        return self._study_store

    @property
    def plugins(self):
        from .plugins_store import PluginStore
        return PluginStore(self)

    @classmethod
    def create(cls, destination: str | Path, name: str | None = None) -> Project:
        root = Path(destination).resolve()
        root.mkdir(parents=True, exist_ok=False)
        project = cls(root, {"format_version": PROJECT_VERSION, "name": name or root.name,
                             "cases": [], "extensions": [], "plugins": [], "studies": [], "definitions": [], "max_workers": 1})
        try:
            project.save()
        except Exception:
            shutil.rmtree(root)
            raise
        return project

    @classmethod
    def open(cls, path: str | Path) -> Project:
        root = Path(path).resolve()
        if root.is_file():
            root = root.parent
        from .study_store import recover_study_transaction
        recover_study_transaction(root)
        metadata = read_json(root / "project.json")
        if metadata.get("format_version") != PROJECT_VERSION:
            raise ValueError("Unsupported project format.")
        if not isinstance(metadata.get("cases"), list):
            raise ValueError("Project cases must be a list.")
        project = cls(root, metadata)
        seen = set()
        for entry in metadata["cases"]:
            if not isinstance(entry, dict) or not re.fullmatch(r"[a-zA-Z0-9_-]+", str(entry.get("id", ""))):
                raise ValueError("Invalid case identity in project.json.")
            if entry["id"] in seen or not isinstance(entry.get("name"), str):
                raise ValueError("Case identities must be unique and each case must have a name.")
            seen.add(entry["id"])
            folder = root / "cases" / entry["id"]
            if not folder.resolve().is_relative_to(root):
                raise ValueError("Case folders must stay inside the project.")
            project.cases.append(ProjectCase(project, entry, read_documents(folder / "inputs")))
        return project

    def save(self) -> None:
        write_json(self.root / "project.json", self.metadata)
        if hasattr(self, "_study_store"):
            self._study_store.invalidate()
        self.edited()

    def add_case(self, name: str, documents: dict[str, dict] | None = None, *, origin="Independent", report=None) -> ProjectCase:
        if not name.strip():
            raise ValueError("Enter a case name.")
        entry = {"id": uuid4().hex, "name": name.strip(), "included": True, "origin": origin}
        if report is not None:
            entry["report"] = deepcopy(report)
        if documents is None:
            from .inputs import empty_documents
            documents = empty_documents(entry["id"])
        case = ProjectCase(self, entry, portable_documents(documents))
        write_documents(case.root / "inputs", case.documents)
        self.metadata["cases"].append(entry)
        self.cases.append(case)
        try:
            self.save()
        except Exception:
            self.metadata["cases"].remove(entry)
            self.cases.remove(case)
            shutil.rmtree(case.root)
            raise
        return case

    def add_case_from_files(self, run_path: str | Path, name: str | None = None) -> ProjectCase:
        from packed_bed.plugins.storage import catalogue_for_case
        catalogue = catalogue_for_case(run_path)
        resolved = inspect_case_file(run_path, catalogue=catalogue)
        existing, _ = self.plugins.browse_catalogue(include_disabled=True)
        registered = {entry['id'] for entry in self.plugins.entries}
        for provider in resolved.selection.lock['plugins']:
            if provider in registered and existing.hashes.get(provider) != catalogue.hashes[provider]:
                raise ValueError(f'Plugin {provider} already exists with different contents. '
                                 'Use Register plugin to replace it before importing this case.')
        for provider in resolved.selection.lock['plugins']:
            self.plugins.add(catalogue.paths[provider])
        documents = {key: read_yaml_mapping(getattr(resolved, f"{key}_path"), key) for key in DOCUMENTS}
        return self.add_case(name or resolved.run.simulation.system_name, documents)

    def duplicate_case(self, case: ProjectCase, name: str) -> ProjectCase:
        return self.add_case(name, case.documents, report=case.metadata.get("report"))

    def import_study(self, batch_path: str | Path, *, name: str | None = None):
        """Import a pending study. Generation requires a successful baseline case."""
        return self.study_store.import_batch(batch_path, name)

    def delete_case(self, case: ProjectCase) -> None:
        """Remove this case and its latest run, rolling back if metadata cannot save."""
        if case not in self.cases:
            raise ValueError("This case does not belong to the project.")
        index = self.cases.index(case)
        removed = self.root / f".deleted-{case.id}"
        case.root.rename(removed)
        self.cases.pop(index)
        self.metadata["cases"].pop(index)
        try:
            self.save()
        except Exception:
            self.cases.insert(index, case)
            self.metadata["cases"].insert(index, case.metadata)
            removed.rename(case.root)
            raise
        shutil.rmtree(removed)

    def prepare_execution(self, cases: list[ProjectCase], *, max_workers: int | None = None) -> Path:
        if max_workers is None:
            max_workers = self.metadata.get("max_workers", 1)
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("Maximum workers must be a positive integer.")
        if len({case.id for case in cases}) != len(cases):
            raise ValueError("A case can only appear once in an execution.")
        if not cases:
            raise ValueError("Include at least one case to run.")
        errors = []
        for case in cases:
            if case.project is not self or not any(member is case for member in self.cases):
                raise ValueError("All cases must belong to this project.")
            try:
                case.validate_for_run()
            except (ValueError, OSError) as exc:
                errors.append(f"{case.name}: {exc}")
        if errors:
            raise ValueError("No cases were started. Resolve these inputs first:\n\n" + "\n\n".join(errors))
        attempt_id = uuid4().hex
        pending = []
        try:
            for case in cases:
                pending.append(case.prepare(attempt_id))
            job = {"attempt_id": attempt_id, "state": "queued", "max_workers": max_workers, "cases": {
                case.id: {"name": case.name, "state": "queued", "elapsed_s": 0.0} for case in cases
            }}
            path = self.root / "execution.json"
            write_json(path, job)
            return path
        except Exception:
            for folder in pending:
                shutil.rmtree(folder)
            raise

    def recover_interrupted(self) -> None:
        """Called only after the GUI acquires this project's exclusive lock."""
        for case in self.cases:
            previous = case.root / ".previous-run"
            if previous.exists() and not case.run_folder.exists():
                previous.rename(case.run_folder)
            if case.run_folder.exists():
                path = case.run_folder / "status.json"
                try:
                    status = read_json(path)
                except (OSError, ValueError):
                    continue
                if status.get("state") not in TERMINAL_STATES:
                    status.update(state="interrupted", message="The application closed before this run completed.")
                    write_json(path, status)
            if previous.exists():
                shutil.rmtree(previous)
            for pending in case.root.glob(".pending-*"):
                if pending.is_dir():
                    shutil.rmtree(pending)
