from __future__ import annotations

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


def _validate_scalar_channel(channel, label, errors):
    if channel is None:
        return
    for index, step in enumerate(channel.steps):
        if step.duration <= 0.0:
            errors.append(f"{label} step {index} duration must be positive.")
        if step.kind == "ramp":
            try:
                float(step.target)
            except (TypeError, ValueError):
                errors.append(f"{label} step {index} ramp target must be numeric.")


def _validate_composition_channel(channel, gas_species, label, errors):
    if channel is None:
        return
    try:
        coerce_composition_mapping(channel.initial, gas_species, label=f"{label} initial value")
    except ValueError as exc:
        errors.append(str(exc))

    for index, step in enumerate(channel.steps):
        if step.duration <= 0.0:
            errors.append(f"{label} step {index} duration must be positive.")
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
    bed_length = run_bundle.run.model.bed_length_m
    zones = list(run_bundle.solids.initial_profile_zones)

    if not solid_species:
        errors.append("solids.solid_species must contain at least one species.")
        return

    _validate_unique(solid_species, "solids.solid_species", errors)

    if run_bundle.solids.concentration_unit not in {"mol_per_m3_solid", "mol_per_m3_bed"}:
        errors.append(
            "solids.initial_profile units must be one of: mol_per_m3_solid, mol_per_m3_bed."
        )

    if not zones:
        errors.append("solids.initial_profile.zones must contain at least one zone.")
        return

    expected_species = set(solid_species)
    previous_end = None
    for index, zone in enumerate(zones):
        if zone.x_start_m < 0.0:
            errors.append(f"solids.initial_profile zone {index} x_start_m must be non-negative.")
        if zone.x_end_m <= zone.x_start_m:
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
        negative_species = sorted(
            species_id for species_id, value in zone.values_mol_per_m3.items() if value < 0.0
        )
        if negative_species:
            errors.append(
                f"solids.initial_profile zone {index} contains negative concentrations for: {', '.join(negative_species)}."
            )
        if not (0.0 < zone.e_b < 1.0):
            errors.append(f"solids.initial_profile zone {index} e_b must stay within (0, 1).")
        if not (0.0 < zone.e_p < 1.0):
            errors.append(f"solids.initial_profile zone {index} e_p must stay within (0, 1).")
        if zone.d_p <= 0.0:
            errors.append(f"solids.initial_profile zone {index} d_p must be positive.")
        if index == 0 and abs(zone.x_start_m) > 1e-9:
            errors.append("solids.initial_profile must start at x = 0.")
        if previous_end is not None and abs(zone.x_start_m - previous_end) > 1e-9:
            errors.append("solids.initial_profile zones must be contiguous without gaps or overlaps.")
        previous_end = zone.x_end_m

    if previous_end is not None and abs(previous_end - bed_length) > 1e-9:
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

    _validate_unique(gas_species, "chemistry.gas_species", errors)
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

    _validate_scalar_channel(run_bundle.program.inlet_flow, "program.inlet_flow", errors)
    _validate_scalar_channel(run_bundle.program.inlet_temperature, "program.inlet_temperature", errors)
    _validate_scalar_channel(run_bundle.program.outlet_pressure, "program.outlet_pressure", errors)
    _validate_composition_channel(
        run_bundle.program.inlet_composition,
        run_bundle.chemistry.gas_species,
        "program.inlet_composition",
        errors,
    )

    if run_bundle.run.time_horizon_s <= 0.0:
        errors.append("simulation.time_horizon_s must be positive.")
    if run_bundle.run.reporting_interval_s <= 0.0:
        errors.append("simulation.reporting_interval_s must be positive.")
    if run_bundle.run.reporting_interval_s > run_bundle.run.time_horizon_s:
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
    if run_bundle.run.model.bed_length_m <= 0.0:
        errors.append("model.bed_length_m must be positive.")
    if run_bundle.run.model.bed_radius_m <= 0.0:
        errors.append("model.bed_radius_m must be positive.")
    if run_bundle.run.model.particle_diameter_m <= 0.0:
        errors.append("model.particle_diameter_m must be positive.")
    if not (0.0 < run_bundle.run.model.interparticle_voidage < 1.0):
        errors.append("model.interparticle_voidage must stay within (0, 1).")
    if not (0.0 < run_bundle.run.model.intraparticle_voidage < 1.0):
        errors.append("model.intraparticle_voidage must stay within (0, 1).")

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
