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

from packed_bed.config import Case, load_case, resolve_case
from packed_bed.config.load import read_yaml_mapping


DOCUMENTS = ("run", "chemistry", "program", "solids")
REFERENCES = {f"{name}_file": f"{name}.yaml" for name in DOCUMENTS[1:]}
PROJECT_VERSION = 2
SNAPSHOT_VERSION = 1
TERMINAL_STATES = {"completed", "failed", "cancelled", "interrupted"}


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
    values = deepcopy(documents)
    values["run"].pop("references", None)
    outputs = values["run"].get("outputs", {})
    if isinstance(outputs, dict):
        outputs.pop("directory", None)
        outputs.pop("artifacts_directory", None)
    payload = yaml.safe_dump({"inputs": values, "extensions": extensions}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def require_desktop_solver(case: Case) -> None:
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

    def resolve(self) -> Case:
        if self.documents["run"].get("references") != REFERENCES:
            raise ValueError("Case inputs must reference chemistry.yaml, program.yaml and solids.yaml in inputs/.")
        return resolve_case(
            **{f"{name}_path": self.root / "inputs" / f"{name}.yaml" for name in DOCUMENTS},
            **{f"{name}_data": self.documents[name] for name in DOCUMENTS},
        )

    def validate_for_run(self) -> Case:
        case = self.resolve()
        if self.project.metadata.get("extensions"):
            raise ValueError("This project requires extensions. Extension loading is not available in the starter yet.")
        require_desktop_solver(case)
        return case

    def save(self) -> None:
        """Save incomplete drafts without altering this case's latest results."""
        write_documents(self.root / "inputs", self.documents)
        self.project.save()

    def fingerprint(self) -> str:
        return scientific_fingerprint(self.documents, self.project.metadata.get("extensions", []))

    def state(self) -> dict:
        readiness, message = "Ready", ""
        try:
            self.validate_for_run()
        except (ValueError, OSError) as exc:
            message = str(exc)
            errors = exc.__cause__.errors() if isinstance(exc.__cause__, ValidationError) else []
            missing = any(error["type"] == "missing" or error.get("input") in (None, "") for error in errors)
            readiness = "Underdefined" if missing else "Invalid"
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
        self.validate_for_run()
        self.save()
        folder = self.root / f".pending-{attempt_id}"
        folder.mkdir(exist_ok=False)
        try:
            write_documents(folder / "inputs", portable_documents(self.documents))
            write_json(folder / "snapshot.json", {
                "format_version": SNAPSHOT_VERSION,
                "case_id": self.id,
                "case_name": self.name,
                "attempt_id": attempt_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "fingerprint": self.fingerprint(),
                "extensions": deepcopy(self.project.metadata.get("extensions", [])),
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

    @classmethod
    def create(cls, destination: str | Path, name: str | None = None) -> Project:
        root = Path(destination).resolve()
        root.mkdir(parents=True, exist_ok=False)
        project = cls(root, {"format_version": PROJECT_VERSION, "name": name or root.name,
                             "cases": [], "extensions": [], "studies": [], "max_workers": 1})
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
        metadata = read_json(root / "project.json")
        if metadata.get("format_version") == 1:
            return cls._migrate(root, metadata)
        if metadata.get("format_version") != PROJECT_VERSION:
            raise ValueError("Unsupported project format. Open it with a compatible application version.")
        if not isinstance(metadata.get("cases"), list) or not isinstance(metadata.get("extensions", []), list):
            raise ValueError("Project cases and extensions must be lists.")
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

    @classmethod
    def _migrate(cls, root: Path, old: dict) -> Project:
        """Copy the old project into one case, retaining original files as a migration backup."""
        backup = root / "project-v1.json"
        if not backup.exists():
            write_json(backup, old)
        project = cls(root, {**old, "format_version": PROJECT_VERSION, "cases": [],
                             "extensions": old.get("extensions", []), "studies": []})
        # Commit project.json last, so a failed migration can be retried.
        case_id = f"original-{uuid4().hex}"
        folder = root / "cases" / case_id
        entry = {"id": case_id, "name": old.get("name", "Original case"), "included": True, "origin": "Independent"}
        case = ProjectCase(project, entry, read_documents(root / "inputs"))
        write_documents(folder / "inputs", case.documents)
        def created_at(path):
            value = read_json(path / "snapshot.json").get("created_at")
            try:
                return datetime.fromisoformat(value).timestamp()
            except (TypeError, ValueError):
                return path.stat().st_mtime

        old_runs = sorted((path for path in (root / "runs").glob("*") if (path / "snapshot.json").is_file()), key=created_at)
        if old_runs:
            shutil.copytree(old_runs[-1], case.run_folder)
            snapshot = read_json(case.run_folder / "snapshot.json")
            snapshot.update(case_id=case_id, case_name=case.name,
                            fingerprint=scientific_fingerprint(read_documents(case.run_folder / "inputs"), project.metadata["extensions"]))
            write_json(case.run_folder / "snapshot.json", snapshot)
        project.metadata["cases"].append(entry)
        project.cases.append(case)
        project.save()
        return project

    def save(self) -> None:
        write_json(self.root / "project.json", self.metadata)

    def add_case(self, name: str, documents: dict[str, dict] | None = None, *, origin="Independent") -> ProjectCase:
        if not name.strip():
            raise ValueError("Enter a case name.")
        entry = {"id": uuid4().hex, "name": name.strip(), "included": True, "origin": origin}
        if documents is None:
            documents = {name: {} for name in DOCUMENTS}
            documents["run"] = {"simulation": {}, "model": {}, "solver": {"threads": 1}}
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
        resolved = load_case(run_path)
        documents = {key: read_yaml_mapping(getattr(resolved, f"{key}_path"), key) for key in DOCUMENTS}
        return self.add_case(name or resolved.run.simulation.system_name, documents)

    def duplicate_case(self, case: ProjectCase, name: str) -> ProjectCase:
        return self.add_case(name, case.documents)

    def add_cases_from_batch(self, batch_path: str | Path, *, name: str | None = None) -> list[ProjectCase]:
        """Import expanded cases and retain a portable copy of their generation rule."""
        from packed_bed.batch import expand_batch_cases, load_batch_spec
        from packed_bed.config.load import resolve_path

        document = load_batch_spec(batch_path)
        study_id = uuid4().hex
        folder = self.root / "studies" / study_id
        folder.mkdir(parents=True)
        entries = []
        study = {"id": study_id, "name": name or Path(batch_path).stem}
        try:
            spec = document.spec.model_dump(mode="json")
            base_path = resolve_path(document.base_dir, document.spec.base_case)
            run = read_yaml_mapping(base_path, "run")
            base = {"run": run, **{
                name: read_yaml_mapping(resolve_path(base_path.parent, run["references"][f"{name}_file"]), name)
                for name in DOCUMENTS[1:]
            }}
            write_documents(folder / "base", portable_documents(base))
            spec["base_case"] = "base/run.yaml"
            spec["output_directory"] = "unused-output"
            for index, (name, source) in enumerate(document.spec.programs.items()):
                destination = f"program-{index}.yaml"
                write_text(folder / destination, yaml.safe_dump(read_yaml_mapping(resolve_path(document.base_dir, source), name)))
                spec["programs"][name] = destination
            for index, (name, preset) in enumerate(document.spec.geometries.items()):
                if preset.solids_file is not None:
                    destination = f"solids-{index}.yaml"
                    write_text(folder / destination, yaml.safe_dump(read_yaml_mapping(resolve_path(document.base_dir, preset.solids_file), name)))
                    spec["geometries"][name]["solids_file"] = destination
            write_text(folder / "batch.yaml", yaml.safe_dump(spec, sort_keys=False))
            for expanded in expand_batch_cases(load_batch_spec(folder / "batch.yaml")):
                entry = {"id": uuid4().hex, "name": " / ".join(expanded.selections.values()),
                         "included": True, "origin": study["name"], "study_id": study_id,
                         "selections": expanded.selections}
                case = ProjectCase(self, entry, portable_documents({name: getattr(expanded, name) for name in DOCUMENTS}))
                write_documents(case.root / "inputs", case.documents)
                entries.append(case)
            self.cases.extend(entries)
            self.metadata["cases"].extend(case.metadata for case in entries)
            self.metadata.setdefault("studies", []).append(study)
            self.save()
        except Exception:
            for case in entries:
                if case in self.cases:
                    self.cases.remove(case)
                    self.metadata["cases"].remove(case.metadata)
                shutil.rmtree(case.root)
            if study in self.metadata.get("studies", []):
                self.metadata["studies"].remove(study)
            shutil.rmtree(folder)
            raise
        return entries

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
