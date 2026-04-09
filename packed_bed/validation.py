from __future__ import annotations

import math

from .axial_schemes import SUPPORTED_SCHEMES
from .config import RunBundle
from .programs import coerce_composition_mapping
from .properties import DEFAULT_PROPERTY_REGISTRY
from .reactions import DEFAULT_REACTION_CATALOG
from .reporting import REPORT_VARIABLE_REGISTRY


class PackedBedValidationError(ValueError):
    pass


def _validate_unique(items, label, errors):
    duplicates = sorted({item for item in items if items.count(item) > 1})
    if duplicates:
        errors.append(f"{label} must not contain duplicates: {', '.join(duplicates)}.")


def _validate_finite_scalar(value, label, errors):
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be numeric.")
        return None
    if not math.isfinite(scalar):
        errors.append(f"{label} must be finite.")
        return None
    return scalar


def _validate_positive_scalar(value, label, errors):
    scalar = _validate_finite_scalar(value, label, errors)
    if scalar is not None and scalar <= 0.0:
        errors.append(f"{label} must be positive.")
    return scalar


def _validate_unit_interval_scalar(value, label, errors):
    scalar = _validate_finite_scalar(value, label, errors)
    if scalar is not None and not (0.0 < scalar < 1.0):
        errors.append(f"{label} must stay within (0, 1).")
    return scalar


def _validate_scalar_channel(channel, label, errors):
    if channel is None:
        return
    initial_value = _validate_positive_scalar(channel.initial, f"{label} initial value", errors)
    for index, step in enumerate(channel.steps):
        _validate_positive_scalar(step.duration, f"{label} step {index} duration", errors)
        if step.kind == "ramp":
            target = _validate_positive_scalar(
                step.target,
                f"{label} step {index} ramp target",
                errors,
            )


def _validate_composition_channel(channel, gas_species, label, errors):
    if channel is None:
        return
    try:
        coerce_composition_mapping(channel.initial, gas_species, label=f"{label} initial value")
    except ValueError as exc:
        errors.append(str(exc))

    for index, step in enumerate(channel.steps):
        _validate_positive_scalar(step.duration, f"{label} step {index} duration", errors)
        if step.kind == "ramp":
            try:
                coerce_composition_mapping(
                    step.target,
                    gas_species,
                    label=f"{label} step {index} target",
                )
            except ValueError as exc:
                errors.append(str(exc))


def _validate_solid_profile(run_bundle: RunBundle, errors):
    solid_species = list(run_bundle.solids.solid_species)
    bed_length = _validate_positive_scalar(run_bundle.run.model.bed_length_m, "model.bed_length_m", errors)
    zones = list(run_bundle.solids.initial_profile_zones)

    if not solid_species:
        errors.append("solids.solid_species must contain at least one species.")
        return

    if run_bundle.solids.concentration_basis not in {"solid", "bed"}:
        errors.append(
            "solids.initial_profile concentration_basis must be one of: solid, bed."
        )

    if not zones:
        errors.append("solids.initial_profile.zones must contain at least one zone.")
        return

    expected_species = set(solid_species)
    previous_end = None
    for index, zone in enumerate(zones):
        x_start = _validate_finite_scalar(zone.x_start_m, f"solids.initial_profile zone {index} x_start_m", errors)
        x_end = _validate_finite_scalar(zone.x_end_m, f"solids.initial_profile zone {index} x_end_m", errors)
        
        if x_start is not None and x_start < 0.0:
            errors.append(f"solids.initial_profile zone {index} x_start_m must be non-negative.")
        if x_start is not None and x_end is not None and x_end <= x_start:
            errors.append(f"solids.initial_profile zone {index} must satisfy x_end_m > x_start_m.")
        value_species = set(zone.values_mol_per_m3.keys())
        missing = sorted(expected_species - value_species)
        unexpected = sorted(value_species - expected_species)
        if missing:
            errors.append(
                f"solids.initial_profile zone {index} is missing species values for: {', '.join(missing)}."
            )
        if unexpected:
            errors.append(
                f"solids.initial_profile zone {index} contains unexpected species values: {', '.join(unexpected)}."
            )
        nonfinite_species = []
        negative_species = []
        for species_id, value in zone.values_mol_per_m3.items():
            try:
                scalar_value = float(value)
            except (TypeError, ValueError):
                nonfinite_species.append(species_id)
                continue
            if not math.isfinite(scalar_value):
                nonfinite_species.append(species_id)
                continue
            if scalar_value < 0.0:
                negative_species.append(species_id)
        nonfinite_species.sort()
        if nonfinite_species:
            errors.append(
                f"solids.initial_profile zone {index} contains non-finite concentrations for: {', '.join(nonfinite_species)}."
            )
        negative_species.sort()
        if negative_species:
            errors.append(
                f"solids.initial_profile zone {index} contains negative concentrations for: {', '.join(negative_species)}."
            )
        _validate_unit_interval_scalar(zone.e_b, f"solids.initial_profile zone {index} e_b", errors)
        _validate_unit_interval_scalar(zone.e_p, f"solids.initial_profile zone {index} e_p", errors)
        _validate_positive_scalar(zone.d_p, f"solids.initial_profile zone {index} d_p", errors)
        if index == 0 and x_start is not None and abs(x_start) > 1e-9:
            errors.append("solids.initial_profile must start at x = 0.")
        if previous_end is not None and x_start is not None and abs(x_start - previous_end) > 1e-9:
            errors.append("solids.initial_profile zones must be contiguous without gaps or overlaps.")
        previous_end = x_end

    if previous_end is not None and bed_length is not None and abs(previous_end - bed_length) > 1e-9:
        errors.append("solids.initial_profile must end at x = model.bed_length_m.")


def validate_run_bundle(
    run_bundle: RunBundle,
    *,
    property_registry=DEFAULT_PROPERTY_REGISTRY,
    reaction_catalog=DEFAULT_REACTION_CATALOG,
):
    errors: list[str] = []

    gas_species = list(run_bundle.chemistry.gas_species)
    solid_species = list(run_bundle.solids.solid_species)
    reaction_ids = list(run_bundle.chemistry.reaction_ids)

    if not gas_species:
        errors.append("chemistry.gas_species must contain at least one species.")
    if not solid_species:
        errors.append("chemistry.solid_species must contain at least one species.")

    _validate_unique(gas_species, "chemistry.gas_species", errors)
    _validate_unique(solid_species, "chemistry.gas_species", errors)
    _validate_unique(reaction_ids, "chemistry.reaction_ids", errors)

    for species_id in gas_species:
        if not property_registry.has_species(species_id):
            errors.append(f"Unknown gas species '{species_id}'.")
            continue
        record = property_registry.get_record(species_id)
        if record.phase != "gas":
            errors.append(f"Species '{species_id}' is phase '{record.phase}', not gas.")
        if record.mw is None:
            errors.append(f"Gas species '{species_id}' must define molecular weight.")
        if record.enthalpy is None:
            errors.append(f"Gas species '{species_id}' must define an enthalpy correlation.")
        if record.viscosity is None:
            errors.append(f"Gas species '{species_id}' must define a viscosity correlation.")

    for species_id in solid_species:
        if not property_registry.has_species(species_id):
            errors.append(f"Unknown solid species '{species_id}'.")
            continue
        record = property_registry.get_record(species_id)
        if record.phase != "solid":
            errors.append(f"Species '{species_id}' is phase '{record.phase}', not solid.")
        if record.mw is None:
            errors.append(f"Solid species '{species_id}' must define molecular weight.")
        if record.enthalpy is None:
            errors.append(f"Solid species '{species_id}' must define an enthalpy correlation.")

    _validate_solid_profile(run_bundle, errors)

    selected_species = set(gas_species) | set(solid_species)
    for reaction_id in reaction_ids:
        reaction = reaction_catalog.get(reaction_id)
        if reaction is None:
            errors.append(f"Unknown reaction id '{reaction_id}'.")
            continue
        missing = [species_id for species_id in reaction.required_species if species_id not in selected_species]
        if missing:
            errors.append(
                f"Reaction '{reaction_id}' requires unselected species: {', '.join(missing)}."
            )
        for species_id in reaction.stoichiometry:
            if species_id not in selected_species:
                errors.append(
                    f"Reaction '{reaction_id}' references species '{species_id}' "
                    "that is not selected in the current gas/solid configuration."
                )

    _validate_scalar_channel(
        run_bundle.program.inlet_flow,
        "program.inlet_flow",
        errors,
    )
    _validate_scalar_channel(
        run_bundle.program.inlet_temperature,
        "program.inlet_temperature",
        errors,
    )
    _validate_scalar_channel(
        run_bundle.program.outlet_pressure,
        "program.outlet_pressure",
        errors,
    )
    _validate_composition_channel(
        run_bundle.program.inlet_composition,
        run_bundle.chemistry.gas_species,
        "program.inlet_composition",
        errors,
    )

    time_horizon = _validate_positive_scalar(run_bundle.run.time_horizon_s, "simulation.time_horizon_s", errors)
    reporting_interval = _validate_positive_scalar(
        run_bundle.run.reporting_interval_s,
        "simulation.reporting_interval_s",
        errors,
    )
    if (
        time_horizon is not None
        and reporting_interval is not None
        and reporting_interval > time_horizon
    ):
        errors.append("simulation.reporting_interval_s must not exceed simulation.time_horizon_s.")
    if run_bundle.run.mass_scheme not in SUPPORTED_SCHEMES:
        errors.append(
            f"simulation.mass_scheme must be one of: {', '.join(SUPPORTED_SCHEMES)}."
        )
    if run_bundle.run.heat_scheme not in SUPPORTED_SCHEMES:
        errors.append(
            f"simulation.heat_scheme must be one of: {', '.join(SUPPORTED_SCHEMES)}."
        )
    if run_bundle.run.model.axial_cells < 1:
        errors.append("model.axial_cells must be at least 1.")
    _validate_positive_scalar(run_bundle.run.model.bed_radius_m, "model.bed_radius_m", errors)
    _validate_positive_scalar(run_bundle.run.model.particle_diameter_m, "model.particle_diameter_m", errors)
    _validate_unit_interval_scalar(
        run_bundle.run.model.interparticle_voidage,
        "model.interparticle_voidage",
        errors,
    )
    _validate_unit_interval_scalar(
        run_bundle.run.model.intraparticle_voidage,
        "model.intraparticle_voidage",
        errors,
    )
    _validate_positive_scalar(run_bundle.run.model.gas_constant, "model.gas_constant", errors)
    _validate_positive_scalar(run_bundle.run.model.pi_value, "model.pi_value", errors)
    _validate_positive_scalar(run_bundle.run.solver.relative_tolerance, "solver.relative_tolerance", errors)

    unknown_reports = [
        report_id
        for report_id in run_bundle.run.outputs.requested_reports
        if report_id not in REPORT_VARIABLE_REGISTRY
    ]
    if unknown_reports:
        errors.append(
            f"outputs.requested_reports contains unknown ids: {', '.join(unknown_reports)}."
        )

    if errors:
        raise PackedBedValidationError("\n".join(errors))

    return run_bundle
