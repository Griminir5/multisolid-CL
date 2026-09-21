"""Shared document operations for case editors and parameter studies (no Qt)."""

from copy import deepcopy
import math
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from packed_bed.config import resolve_case
from packed_bed.config.models import ModelConfig, SimulationConfig


BED_RUN_FIELDS = {
    "model": ("bed_length_m", "bed_radius_m", "ambient_temperature_k",
              "heat_transfer_coefficient_w_per_m2_k", "gas_voidage_mode"),
    "simulation": ("interior_flow_mode",),
}


def empty_documents(case_id):
    """Initialize an authored draft explicitly, never as a side effect of inspection."""
    return {
        "run": {
            "simulation": {"system_name": "Case_" + case_id, "time_horizon_s": 0.0,
                           "reporting_interval_s": 1.0, "mass_scheme": "weno3", "heat_scheme": "weno3",
                           "report_time_derivatives": False, "repeat_program": True, "program_mode": "feed_stream"},
            "model": {"axial_cells": 20},
            "solver": {"backend": "daetools", "name": "superlu", "threads": 1, "relative_tolerance": 1e-5},
            "outputs": {"requested_reports": [], "requested_plots": []},
        },
        "chemistry": {"gas_species": [], "reaction_families": [], "reaction_ids": []},
        "solids": {"solid_species": [], "initial_profile": {"basis": "bed", "zones": []}},
        "program": {"feed_stream": {"basis": "mol_per_s", "initial": {"flow": "", "temperature": "", "composition": {}}, "steps": []},
                    "outlet_pressure": {"initial": "", "steps": []}},
    }


def program_duration(program):
    """Longest authored channel, or None when a duration is unfinished."""
    try:
        durations = [math.fsum(float(step["duration_s"]) for step in channel.get("steps", []))
                     for channel in program.values() if isinstance(channel, dict)]
        if any(not math.isfinite(value) or value < 0 for value in durations):
            return None
        return max(durations, default=0.0)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def new_step_ids(program):
    return {channel: [uuid4().hex for _ in data.get("steps", [])]
            for channel, data in program.items() if isinstance(data, dict)}


def ensure_step_ids(program, metadata):
    identities = metadata.setdefault("step_ids", {})
    for channel, data in program.items():
        if not isinstance(data, dict):
            continue
        steps = data.get("steps", [])
        if len(identities.get(channel, [])) != len(steps):
            identities[channel] = [uuid4().hex for _ in steps]
    return identities


def scale_zones(documents, old_length, new_length):
    if not isinstance(old_length, (int, float)) or not math.isfinite(old_length) or old_length <= 0:
        raise ValueError("The baseline needs a positive bed length to scale solid zones.")
    zones = documents["solids"].get("initial_profile", {}).get("zones", [])
    for zone in zones:
        for key in ("x_start_m", "x_end_m"):
            value = zone.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Complete all baseline zone boundaries before sweeping bed length.")
            zone[key] = value * new_length / old_length


def resolve_documents(documents):
    # resolve_case accepts in-memory data and never reads these diagnostic paths.
    return resolve_case(**{f"{name}_path": Path("inputs") / f"{name}.yaml" for name in documents},
                        **{f"{name}_data": value for name, value in documents.items()})


def input_readiness(documents, extensions=()):
    try:
        case = resolve_documents(documents)
        if extensions:
            raise ValueError("This project requires extensions. Extension loading is not available yet.")
        if case.run.solver.backend != "daetools" or case.run.solver.name != "superlu":
            raise ValueError("Select DAETools / SuperLU for the desktop runtime.")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        errors = exc.__cause__.errors() if isinstance(exc.__cause__, ValidationError) else []
        missing = any(error["type"] == "missing" or error.get("input") in (None, "") for error in errors)
        return ("Underdefined" if missing else "Invalid"), str(exc)
    return "Ready", ""


def definition_payload(kind, documents, metadata=None):
    if kind == "program":
        simulation = documents["run"].get("simulation", {})
        return {"program": deepcopy(documents["program"]),
                "simulation": {key: simulation.get(key, default) for key, default in
                               (("program_mode", "separate_channels"), ("repeat_program", False))},
                "gas_species": deepcopy(documents["chemistry"].get("gas_species", [])),
                "editor_metadata": deepcopy(metadata or {})}
    payload = {"solids": deepcopy(documents["solids"])}
    for section, keys in BED_RUN_FIELDS.items():
        schema = ModelConfig if section == "model" else SimulationConfig
        payload[section] = {}
        for key in keys:
            field = schema.model_fields[key]
            default = "" if field.is_required() else field.default
            payload[section][key] = documents["run"].get(section, {}).get(key, default)
    return payload


def apply_definition(documents, kind, payload):
    document = "program" if kind == "program" else "solids"
    documents[document] = deepcopy(payload[document])
    for section in ("model", "simulation"):
        if section in payload:
            documents["run"].setdefault(section, {}).update(deepcopy(payload[section]))


def get_value(documents, path, default=None):
    value = documents
    try:
        for key in path:
            value = value[key]
        return value
    except (KeyError, IndexError, TypeError):
        return default


def set_value(documents, path, value):
    target = documents
    try:
        for key in path[:-1]:
            if isinstance(target, dict):
                target = target.setdefault(key, {})
            else:
                target = target[key]
        target[path[-1]] = value
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("The selected input must have the expected mapping or step structure.") from exc


def scientific_documents(documents):
    documents = deepcopy(documents)
    documents["run"].pop("references", None)
    outputs = documents["run"].get("outputs", {})
    if isinstance(outputs, dict):
        for key in ("directory", "artifacts_directory"):
            outputs.pop(key, None)
    return documents
