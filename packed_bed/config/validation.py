from __future__ import annotations

import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import ValidationError
from packed_bed.programs import sum_step_durations

from .errors import PackedBedValidationError
from .program import CompositionRampStep
from .validators import _require_exact_keys

if TYPE_CHECKING:
    from .bundle import RunBundle


_PROGRAM_DURATION_SUM_ABS_TOLERANCE_S = 1.0e-9


REQUIRED_PROPERTIES_BY_PHASE = {
    "gas": (
        ("mw", "molecular weight"),
        ("enthalpy", "an enthalpy correlation"),
        ("viscosity", "a viscosity correlation"),
    ),
    "solid": (
        ("mw", "molecular weight"),
        ("enthalpy", "an enthalpy correlation"),
    ),
}


def validate_config_model(model_type, data: dict[str, Any], label: str, path: Path):
    try:
        return model_type.model_validate(data)
    except ValidationError as exc:
        raise _format_validation_error(label, path, exc) from exc


def validate_bundle_shape(run_bundle: RunBundle) -> RunBundle:
    errors: list[str] = []

    _validate_phase_disjointness(run_bundle, errors)
    _validate_solid_zones_cover_bed(run_bundle, errors)
    _validate_program_species(run_bundle, errors)
    _validate_program_durations(run_bundle, errors)

    if errors:
        raise PackedBedValidationError("\n".join(errors))
    return run_bundle


def validate_run_bundle(
    run_bundle: RunBundle,
    property_registry=None,
    reaction_catalog=None,
    report_variable_registry=None,
) -> RunBundle:
    if property_registry is None:
        from packed_bed.properties import PROPERTY_REGISTRY

        property_registry = PROPERTY_REGISTRY
    if reaction_catalog is None:
        from packed_bed.reactions import REACTION_CATALOG

        reaction_catalog = REACTION_CATALOG
    if report_variable_registry is None:
        from packed_bed.reporting import REPORT_VARIABLE_REGISTRY

        report_variable_registry = REPORT_VARIABLE_REGISTRY

    from packed_bed.reactions import build_reaction_network

    errors: list[str] = []
    has_unknown_reaction_ids = False
    gas_species = set(run_bundle.chemistry.gas_species)
    solid_species = set(run_bundle.solids.solid_species)
    selected_species = gas_species | solid_species

    _validate_species_group(
        errors,
        species_ids=run_bundle.chemistry.gas_species,
        expected_phase="gas",
        property_registry=property_registry,
    )
    _validate_species_group(
        errors,
        species_ids=run_bundle.solids.solid_species,
        expected_phase="solid",
        property_registry=property_registry,
    )

    for reaction_id in run_bundle.chemistry.reaction_ids:
        reaction = reaction_catalog.get(reaction_id)
        if reaction is None:
            errors.append(f"Unknown reaction id '{reaction_id}'.")
            has_unknown_reaction_ids = True
            continue

        missing = sorted(
            species_id
            for species_id in reaction.all_species
            if species_id not in selected_species
        )
        if missing:
            errors.append(f"Reaction '{reaction_id}' requires unselected species: {', '.join(missing)}.")

        invalid_refs = sorted(
            species_id
            for species_id in reaction.stoichiometry
            if species_id not in selected_species
        )
        for species_id in invalid_refs:
            errors.append(
                f"Reaction '{reaction_id}' references species '{species_id}' that is not selected in the current gas/solid configuration."
            )

    if not has_unknown_reaction_ids:
        try:
            build_reaction_network(
                run_bundle.chemistry.reaction_ids,
                run_bundle.chemistry.gas_species,
                run_bundle.solids.solid_species,
                reaction_catalog=reaction_catalog,
            )
        except ValueError as exc:
            error_message = str(exc)
            if error_message not in errors:
                errors.append(error_message)

    unknown_reports = sorted(
        report_id
        for report_id in run_bundle.run.outputs.requested_reports
        if report_id not in report_variable_registry
    )
    if unknown_reports:
        errors.append(f"outputs.requested_reports contains unknown ids: {', '.join(unknown_reports)}.")

    if errors:
        raise PackedBedValidationError("\n".join(errors))
    return run_bundle


def _format_validation_error(label: str, path: Path, exc: ValidationError) -> PackedBedValidationError:
    lines = [f"{label} is invalid: {path}"]
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"]) or "<root>"
        lines.append(f"- {location}: {error['msg']}")
    return PackedBedValidationError("\n".join(lines))


def _validate_phase_disjointness(run_bundle: RunBundle, errors: list[str]) -> None:
    overlap = sorted(set(run_bundle.chemistry.gas_species) & set(run_bundle.solids.solid_species))
    if overlap:
        errors.append(
            "Gas and solid species identifiers must be disjoint. "
            f"Found duplicates: {', '.join(overlap)}."
        )


def _validate_solid_zones_cover_bed(run_bundle: RunBundle, errors: list[str]) -> None:
    zones = run_bundle.solids.initial_profile.zones
    previous_end: float | None = None

    for zone_index, zone in enumerate(zones):
        if zone_index == 0 and not math.isclose(zone.x_start_m, 0.0, rel_tol=0.0, abs_tol=1e-12):
            errors.append("solids.initial_profile.zones must start at x = 0.")
        if previous_end is not None and not math.isclose(zone.x_start_m, previous_end, rel_tol=0.0, abs_tol=1e-12):
            errors.append("solids.initial_profile.zones must be contiguous without gaps or overlaps.")
        previous_end = zone.x_end_m

    if previous_end is None:
        errors.append("solids.initial_profile.zones must not be empty.")
        return
    if not math.isclose(previous_end, run_bundle.run.model.bed_length_m, rel_tol=0.0, abs_tol=1e-12):
        errors.append("solids.initial_profile.zones must end at model.bed_length_m.")


def _validate_program_species(run_bundle: RunBundle, errors: list[str]) -> None:
    try:
        _require_exact_keys(
            set(run_bundle.program.inlet_composition.initial),
            run_bundle.chemistry.gas_species,
            "program.inlet_composition.initial",
        )
        for step_index, step in enumerate(run_bundle.program.inlet_composition.steps):
            if isinstance(step, CompositionRampStep):
                _require_exact_keys(
                    set(step.target),
                    run_bundle.chemistry.gas_species,
                    f"program.inlet_composition.steps[{step_index}].target",
                )
    except ValueError as exc:
        errors.append(str(exc))


def _validate_program_durations(run_bundle: RunBundle, errors: list[str]) -> None:
    simulation = run_bundle.run.simulation
    if simulation.repeat_program:
        return

    horizon = simulation.time_horizon_s
    cycle_durations = (
        ("program.inlet_flow.steps", sum_step_durations(run_bundle.program.inlet_flow.steps)),
        ("program.inlet_temperature.steps", sum_step_durations(run_bundle.program.inlet_temperature.steps)),
        ("program.outlet_pressure.steps", sum_step_durations(run_bundle.program.outlet_pressure.steps)),
        ("program.inlet_composition.steps", sum_step_durations(run_bundle.program.inlet_composition.steps)),
    )
    for label, duration in cycle_durations:
        if duration == 0.0:
            continue
        if not math.isclose(
            duration,
            horizon,
            rel_tol=0.0,
            abs_tol=_PROGRAM_DURATION_SUM_ABS_TOLERANCE_S,
        ):
            difference = duration - horizon
            errors.append(
                f"{label} must sum to time_horizon_s ({horizon:.16g}) within "
                f"{_PROGRAM_DURATION_SUM_ABS_TOLERANCE_S:.1e} s, got {duration:.17g} "
                f"(difference {difference:+.3e} s)."
            )


def _validate_species_group(
    errors: list[str],
    *,
    species_ids: tuple[str, ...],
    expected_phase: str,
    property_registry,
) -> None:
    for species_id in species_ids:
        if not property_registry.has_species(species_id):
            errors.append(f"Unknown {expected_phase} species '{species_id}'.")
            continue

        record = property_registry.get_record(species_id)
        if record.phase != expected_phase:
            errors.append(f"Species '{species_id}' is phase '{record.phase}', not {expected_phase}.")

        for property_name, description in REQUIRED_PROPERTIES_BY_PHASE[expected_phase]:
            if getattr(record, property_name) is None:
                errors.append(f"{expected_phase.capitalize()} species '{species_id}' must define {description}.")
