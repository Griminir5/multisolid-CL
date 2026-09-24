"""Parameter-study calculations. Documents enter and leave as ordinary Python data."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from typing import Iterator

from packed_bed.batch import BatchSpec, apply_batch_value, iter_selections

from .inputs import (apply_definition, get_value, input_readiness, program_duration, scale_zones,
                     scientific_documents, set_value)


class StudyError(ValueError):
    """A rule cannot be expanded without guessing the user's intention."""


@dataclass
class Factor:
    id: str
    target: str
    values: list = field(default_factory=list)
    range: dict | None = None

    def value_count(self):
        if self.range is None or self.target.startswith("definition:"):
            return len(self.values)
        count = numeric_value(self.range.get("count"), True)
        if count < 2:
            raise StudyError("A range needs at least two values.")
        return count


@dataclass
class ReusableDefinition:
    id: str
    name: str
    kind: str
    payload: dict


@dataclass
class Study:
    id: str
    name: str
    baseline: dict
    provenance: dict = field(default_factory=dict)
    editor_metadata: dict = field(default_factory=dict)
    factors: list[Factor] = field(default_factory=list)
    mode: str = "combinations"
    rows: list[dict] = field(default_factory=list)
    legacy: dict | None = None

    def rule(self):
        data = asdict(self)
        data.pop("baseline")
        return data

    @classmethod
    def from_rule(cls, data, baseline):
        data = deepcopy(data)
        data["factors"] = [Factor(**factor) for factor in data.get("factors", [])]
        return cls(baseline=baseline, **data)


@dataclass(frozen=True)
class Parameter:
    id: str
    label: str
    group: str
    unit: str
    path: tuple
    integer: bool = False
    available: bool = True
    reason: str = ""

    @property
    def title(self):
        return f"{self.label} ({self.unit})" if self.unit else self.label


@dataclass
class Candidate:
    name: str
    documents: dict
    selections: dict
    inputs: str
    message: str = ""


@dataclass(frozen=True)
class ExistingCase:
    id: str
    study_id: str
    has_results: bool


@dataclass
class StudyPreview:
    study_id: str
    signature: str
    candidates: list[Candidate]
    delete_ids: list[str]
    result_count: int
    total: int
    remaining: Iterator[Candidate] = field(repr=False, compare=False)


def parameter_catalogue(study):
    """An explicit list of supported fields, plus the baseline's identified steps."""
    parameters = []
    for key, label, unit in (
        ("bed_length_m", "Bed length", "m"), ("bed_radius_m", "Bed radius", "m"),
        ("ambient_temperature_k", "Ambient temperature", "K"),
        ("heat_transfer_coefficient_w_per_m2_k", "Heat-transfer coefficient", "W/m²/K"),
    ):
        parameters.append(Parameter(key, label, "Bed", unit, ("run", "model", key)))
    for section, key, label, unit, integer in (
        ("model", "axial_cells", "Number of cells", "", True),
        ("solver", "threads", "Number of threads", "", True),
        ("solver", "relative_tolerance", "Relative tolerance", "", False),
        ("solver", "concentration_absolute_tolerance", "Concentration tolerance", "mol/m³", False),
        ("simulation", "reporting_interval_s", "Reporting interval", "s", False),
        ("simulation", "time_horizon_s", "Repeating-program horizon", "s", False),
    ):
        available = key != "time_horizon_s" or (
            get_value(study.baseline, ("run", "simulation", "repeat_program"), False)
            and not any(f.target == "definition:program" for f in study.factors))
        parameters.append(Parameter(key, label, "Numerics", unit, ("run", section, key), integer,
                                    bool(available), "Horizon sweeps require a repeating baseline without a program variation."))

    mode = get_value(study.baseline, ("run", "simulation", "program_mode"), "separate_channels")
    flow_key = "feed_stream" if mode == "feed_stream" else "inlet_flow"
    basis = get_value(study.baseline, ("program", flow_key, "basis"), "mol_per_s")
    flow_unit = "h⁻¹" if basis == "ghsv_per_h" else "mol/s"
    channels = [("outlet_pressure", "Outlet pressure", "Pa")]
    if mode == "feed_stream":
        channels += [("feed_stream", "Feed", "")]
    else:
        channels += [("inlet_flow", "Inlet flow", flow_unit),
                     ("inlet_temperature", "Inlet temperature", "K"),
                     ("inlet_composition", "Inlet composition", "")]
    for channel, label, unit in channels:
        data = study.baseline["program"].get(channel, {})
        fields = [("flow", "Flow", flow_unit), ("temperature", "Temperature", "K")] if channel == "feed_stream" else [(None, "Value", unit)]
        if channel != "inlet_composition":
            for subfield, field_label, field_unit in fields:
                path = ("program", channel, "initial") + ((subfield,) if subfield else ())
                ident = f"program:{mode}:{channel}:{basis if channel in ('inlet_flow', 'feed_stream') else ''}:initial:{subfield or 'value'}"
                parameters.append(Parameter(ident, f"{label} → Initial {field_label.lower()}", "Program", field_unit, path))
        identities = study.editor_metadata.get("step_ids", {}).get(channel, [])
        for index, (step, step_id) in enumerate(zip(data.get("steps", []), identities)):
            prefix = f"program:{mode}:{channel}:{basis if channel in ('inlet_flow', 'feed_stream') else ''}:{step_id}:{step.get('kind')}"
            title = f"{label} → Step {index + 1} {step.get('kind', '')}"
            path = ("program", channel, "steps", index)
            parameters.append(Parameter(prefix + ":duration", title + " → Duration", "Program", "s", path + ("duration_s",)))
            if step.get("kind") == "ramp" and channel != "inlet_composition":
                for subfield, field_label, field_unit in fields:
                    parameters.append(Parameter(prefix + f":{subfield or 'target'}", title + f" → {field_label if subfield else 'Target'}",
                                                "Program", field_unit, path + ("target",) + ((subfield,) if subfield else ())))
    return parameters


def numeric_value(raw, integer=False):
    try:
        if isinstance(raw, bool) or raw is None or str(raw).strip() == "":
            raise InvalidOperation
        number = Decimal(str(raw).strip())
        if not number.is_finite() or not math.isfinite(float(number)):
            raise InvalidOperation
        if integer and number != number.to_integral_value():
            raise StudyError("Enter whole numbers for this parameter; values are never rounded.")
        return int(number) if integer else float(number)
    except (InvalidOperation, ValueError, OverflowError) as exc:
        if isinstance(exc, StudyError):
            raise
        raise StudyError("Enter a finite number in every value cell.") from exc


def factor_values(factor, parameter=None):
    if factor.target.startswith("definition:"):
        values = list(factor.values)
    elif factor.range is not None:
        spec = factor.range
        count = factor.value_count()
        # Decimal avoids endpoint drift; integer targets are checked before float conversion.
        start = Decimal(str(spec.get("start", "")))
        end = Decimal(str(spec.get("end", "")))
        numeric_value(start)
        numeric_value(end)
        values = [numeric_value(start + (end - start) * i / (count - 1), parameter.integer)
                  for i in range(count)]
    else:
        values = [numeric_value(value, parameter.integer) for value in factor.values]
    if not values:
        raise StudyError("Add at least one value to each variation.")
    if len(set(values)) != len(values):
        raise StudyError("Remove duplicate values from the variation.")
    return values


def prepared_factors(study, definitions):
    if not study.factors:
        raise StudyError("Add at least one variation.")
    if study.mode not in ("combinations", "rows"):
        raise StudyError("Choose All combinations or Explicit case rows.")
    catalogue = {parameter.id: parameter for parameter in parameter_catalogue(study)}
    targets, factors = set(), {}
    for factor in study.factors:
        if factor.id in factors or factor.target in targets:
            raise StudyError("Two variations target the same parameter. Remove one of them.")
        targets.add(factor.target)
        if factor.target.startswith("definition:"):
            kind = factor.target.split(":", 1)[1]
            if kind not in ("program", "bed"):
                raise StudyError("Unknown reusable-definition kind.")
            parameter = None
        else:
            parameter = catalogue.get(factor.target)
            if parameter is None:
                raise StudyError("A parameter target no longer exists. Reselect its step, mode or flow basis.")
            if not parameter.available:
                raise StudyError(parameter.reason)
        factors[factor.id] = (factor, parameter)
    if "definition:program" in targets and any(target.startswith("program:") for target in targets):
        raise StudyError("A whole-program variation overlaps program-field variations. Choose one approach.")
    if "definition:bed" in targets:
        bed_fields = {"bed_length_m", "bed_radius_m"}
        for ident in referenced_definitions(study):
            definition = definitions.get(ident)
            if definition and definition.kind == "bed":
                bed_fields.update(definition.payload.get("model", {}))
        overlaps = targets & bed_fields
        if overlaps:
            labels = ", ".join(catalogue[target].label for target in sorted(overlaps))
            raise StudyError(f"A bed-configuration variation overlaps {labels}. Choose one approach.")
    return factors


def selections_for(study, definitions):
    factors = prepared_factors(study, definitions)
    try:
        if study.mode == "rows":
            if not study.rows:
                raise StudyError("Add at least one explicit case row.")
            for row in study.rows:
                yield {ident: row.get(ident) if parameter is None else numeric_value(row.get(ident), parameter.integer)
                       for ident, (_, parameter) in factors.items()}
        else:
            yield from iter_selections({ident: factor_values(factor, parameter)
                                        for ident, (factor, parameter) in factors.items()})
    except (InvalidOperation, TypeError) as exc:
        raise StudyError("Complete the variation values or range.") from exc


def candidate_count(study, definitions):
    if study.legacy:
        return math.prod(len(axis["values"]) for axis in study.legacy["spec"]["axes"])
    prepared_factors(study, definitions)
    return len(study.rows) if study.mode == "rows" else math.prod(factor.value_count() for factor in study.factors)


def referenced_definitions(study):
    references = set()
    for factor in study.factors:
        if factor.target.startswith("definition:"):
            if study.mode == "rows":
                references.update(row[factor.id] for row in study.rows if row.get(factor.id))
            else:
                references.update(value for value in factor.values if value)
    if study.legacy:
        references.update(study.legacy.get("programs", {}).values())
        references.update(study.legacy.get("beds", {}).values())
    return references


def generation_signature(study, definitions, extensions=()):
    """Ignore labels and table ordering; include every source that affects generation."""
    content = {}
    for ident in sorted(referenced_definitions(study)):
        definition = definitions.get(ident)
        payload = deepcopy(definition.payload) if definition else {"missing": ident}
        payload.pop("editor_metadata", None)
        content[ident] = payload
    catalogue = {parameter.id: parameter for parameter in parameter_catalogue(study)}

    def canonical_value(factor, value):
        parameter = catalogue.get(factor.target)
        try:
            return numeric_value(value, parameter.integer) if parameter else value
        except StudyError:
            return value  # Invalid draft text still invalidates previously generated cases.

    factors = []
    for factor in study.factors:
        values = sorted((canonical_value(factor, value) for value in factor.values), key=str)
        range_spec = deepcopy(factor.range)
        if range_spec:
            for key, raw in range_spec.items():
                try:
                    range_spec[key] = numeric_value(raw, key == "count")
                except StudyError:
                    pass
        parameter = catalogue.get(factor.target)
        factors.append({"id": factor.id, "target": factor.target,
                        "path": parameter.path if parameter else None,
                        "values": values if study.mode == "combinations" and factor.range is None else [],
                        "range": range_spec if study.mode == "combinations" else None})
    rows = [{factor.id: canonical_value(factor, row.get(factor.id, "")) for factor in study.factors}
            for row in study.rows]
    payload = {"baseline": scientific_documents(study.baseline), "mode": study.mode,
               "factors": sorted(factors, key=lambda value: value["id"]),
               "rows": sorted(rows, key=lambda row: json.dumps(row, sort_keys=True)) if study.mode == "rows" else [],
               "definitions": content, "legacy": study.legacy, "extensions": extensions}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _legacy_candidates(study, definitions, extensions, catalogue=None):
    legacy = study.legacy
    spec = BatchSpec.model_validate(legacy["spec"])
    for name, ident in legacy["beds"].items():
        # Advanced rules retain their original ownership: only originally overridden model keys change.
        model = definitions[ident].payload["model"]
        spec.geometries[name].model.update({key: model[key] for key in spec.geometries[name].model if key in model})
    programs = {name: definitions[ident].payload["program"] for name, ident in legacy["programs"].items()}
    solids = {name: definitions[ident].payload["solids"] for name, ident in legacy["beds"].items()
              if spec.geometries[name].solids_file is not None}
    for selection in iter_selections({axis.id: axis.values for axis in spec.axes}):
        documents = deepcopy(study.baseline)
        for value in selection.values():
            documents = apply_batch_value(documents, value, programs, spec.geometries, solids)
        selected = {key: value.id for key, value in selection.items()}
        readiness, message = input_readiness(documents, extensions, catalogue=catalogue)
        yield Candidate(" / ".join(selected.values()), documents, selected, readiness, message)


def expand_study(study, definitions, extensions=(), *, catalogue=None):
    if study.legacy:
        try:
            yield from _legacy_candidates(study, definitions, extensions, catalogue)
        except KeyError as exc:
            raise StudyError("An imported rule references a missing definition.") from exc
        return
    factors = prepared_factors(study, definitions)
    for selection in selections_for(study, definitions):
        documents = deepcopy(study.baseline)
        labels = []
        changed_timing = False
        # Whole definitions establish inherited inputs before scalar variations apply.
        for ident, (factor, parameter) in factors.items():
            if parameter is None:
                kind = factor.target.split(":", 1)[1]
                definition = definitions.get(selection[ident])
                if definition is None or definition.kind != kind:
                    raise StudyError("Select an existing reusable definition in every definition cell.")
                apply_definition(documents, kind, definition.payload)
                changed_timing |= kind == "program"
        for ident, (factor, parameter) in factors.items():
            value = selection[ident]
            if parameter is None:
                label = definitions[value].name
            else:
                set_value(documents, parameter.path, value)
                label = f"{parameter.label}: {value:g}" + (f" {parameter.unit}" if parameter.unit else "")
                changed_timing |= parameter.path[-1] == "duration_s"
            labels.append(label)
        if any(factor.target == "bed_length_m" for factor, _ in factors.values()):
            try:
                scale_zones(documents, study.baseline["run"]["model"]["bed_length_m"],
                            documents["run"]["model"]["bed_length_m"])
            except ValueError as exc:
                yield Candidate(" / ".join(labels), documents, selection, "Invalid", str(exc))
                continue
        if changed_timing and not documents["run"]["simulation"].get("repeat_program", False):
            duration = program_duration(documents["program"])
            documents["run"]["simulation"]["time_horizon_s"] = duration if duration is not None else ""
        readiness, message = input_readiness(documents, extensions, catalogue=catalogue)
        yield Candidate(" / ".join(labels), documents, selection, readiness, message)


def preview_study(study, definitions, existing_cases, extensions=(), *, lazy=False, catalogue=None):
    """Capture one reviewed source state; Qt may consume its candidates incrementally."""
    study, definitions, extensions = deepcopy((study, definitions, extensions))
    cases = [case for case in existing_cases if case.study_id == study.id]
    preview = StudyPreview(study.id, generation_signature(study, definitions, extensions), [],
                           [case.id for case in cases], sum(case.has_results for case in cases),
                           candidate_count(study, definitions), expand_study(study, definitions, extensions, catalogue=catalogue))
    if not lazy:
        preview.candidates.extend(preview.remaining)
    return preview


def rows_from_combinations(study, definitions):
    return list(selections_for(study, definitions))


def combinations_from_rows(study):
    """Return a draft product rule; the UI previews its count before accepting it."""
    replacement = deepcopy(study)
    for factor in replacement.factors:
        factor.range = None
        factor.values = list(dict.fromkeys(row.get(factor.id, "") for row in study.rows))
    replacement.mode = "combinations"
    replacement.rows = []
    return replacement
