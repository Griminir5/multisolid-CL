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


def validate_run_bundle(
    run_bundle: RunBundle,
    *,
    property_registry=DEFAULT_PROPERTY_REGISTRY,
    reaction_catalog=DEFAULT_REACTION_CATALOG,
):
    errors: list[str] = []

    gas_species = list(run_bundle.chemistry.gas_species)
    solid_species = list(run_bundle.chemistry.solid_species)
    reaction_ids = list(run_bundle.chemistry.reaction_ids)

    if not gas_species:
        errors.append("chemistry.gas_species must contain at least one species.")
    if not solid_species:
        errors.append("chemistry.solid_species must contain at least one species.")

    _validate_unique(gas_species, "chemistry.gas_species", errors)
    _validate_unique(solid_species, "chemistry.solid_species", errors)
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
                    "that is not selected in chemistry.yaml."
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

    initial_solids = dict(run_bundle.run.model.initial_solid_concentration_mol_per_m3)
    missing_initial_solids = [
        species_id for species_id in run_bundle.chemistry.solid_species if species_id not in initial_solids
    ]
    unexpected_initial_solids = [
        species_id for species_id in initial_solids if species_id not in set(run_bundle.chemistry.solid_species)
    ]
    if missing_initial_solids:
        errors.append(
            "model.initial_solid_concentration_mol_per_m3 is missing entries for: "
            + ", ".join(missing_initial_solids)
            + "."
        )
    if unexpected_initial_solids:
        errors.append(
            "model.initial_solid_concentration_mol_per_m3 contains unexpected entries: "
            + ", ".join(unexpected_initial_solids)
            + "."
        )

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
