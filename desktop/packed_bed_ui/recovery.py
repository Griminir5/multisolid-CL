"""Small atomic checkpoints, separate from saved inputs and retained results."""

from copy import deepcopy
from datetime import datetime, timezone

from .project import DOCUMENTS, read_json, write_json
from .studies import Study


def case_payload(case, report=None):
    metadata = deepcopy(case.metadata)
    if report is not None:
        metadata["report"] = deepcopy(report)
    return {"metadata": metadata, "documents": deepcopy(case.documents)}


class Drafts:
    def __init__(self, project):
        self.project = project
        self.folder = project.root / ".drafts"

    def path(self, kind, ident):
        return self.folder / f"{kind}-{ident}.json"

    def write(self, kind, ident, data):
        self.folder.mkdir(exist_ok=True)
        write_json(self.path(kind, ident), {"version": 1, "kind": kind, "id": ident,
                   "updated_at": datetime.now(timezone.utc).isoformat(), "data": data})

    def discard(self, draft):
        self.path(draft["kind"], draft["id"]).unlink(missing_ok=True)

    def clear(self, kind, ident):
        self.path(kind, ident).unlink(missing_ok=True)

    def pending(self):
        """Compare checkpoint contents, never result or filesystem timestamps."""
        cases = {case.id: case for case in self.project.cases}
        pending = []
        for path in sorted(self.folder.glob("*.json")):
            draft = read_json(path)
            kind, ident, data = draft.get("kind"), draft.get("id"), draft.get("data")
            if (draft.get("version") != 1 or kind not in ("case", "study") or not isinstance(data, dict)
                    or not isinstance(ident, str) or path != self.path(kind, ident)
                    or not isinstance(draft.get("updated_at"), str)):
                raise ValueError(f"Cannot read recovery draft: {path.name}")
            if kind == "case" and ident in cases:
                case = cases[ident]
                if (not isinstance(data.get("metadata"), dict) or data["metadata"].get("id") != ident
                        or not isinstance(data["metadata"].get("name"), str)
                        or not isinstance(data.get("documents"), dict)
                        or not all(isinstance(data["documents"].get(key), dict) for key in DOCUMENTS)
                        or data["metadata"].get("study_id") != case.metadata.get("study_id")):
                    raise ValueError(f"Invalid case recovery draft: {path.name}")
                current, name = case_payload(case), case.name
            elif kind == "study" and ident in self.project.study_store.studies:
                if data.get("id") != ident:
                    raise ValueError(f"Invalid study recovery draft: {path.name}")
                study = self.project.study_store.studies[ident]
                current, name = study.rule(), study.name
            else:
                continue  # Never recreate a deleted case or study from a checkpoint.
            if data != current:
                pending.append({**draft, "name": name})
        return pending

    def apply(self, draft):
        """Only called after the user chooses recovery. No execution state changes."""
        data, ident = deepcopy(draft["data"]), draft["id"]
        if draft["kind"] == "case":
            case = next(case for case in self.project.cases if case.id == ident)
            before = case_payload(case)
            case.documents = data["documents"]
            case.metadata.clear()
            case.metadata.update(data["metadata"])
            try:
                case.save()
            except Exception:
                case.documents = before["documents"]
                case.metadata.clear()
                case.metadata.update(before["metadata"])
                raise
        else:
            store = self.project.study_store
            study = store.studies[ident]
            store.save_study(Study.from_rule(data, deepcopy(study.baseline)))
        self.discard(draft)
