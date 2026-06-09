from __future__ import annotations

from typing import TYPE_CHECKING

from packed_bed.properties import PROPERTY_REGISTRY
from packed_bed.reactions import REACTION_CATALOG, build_reaction_network
from packed_bed.reporting import REPORT_VARIABLE_REGISTRY

from .errors import PackedBedValidationError

if TYPE_CHECKING:
    from .models import RunBundle


PROGRAM_DURATION_SUM_ABS_TOLERANCE_S = 1.0e-9


def validate_run_bundle(
    run_bundle: RunBundle,
    property_registry=PROPERTY_REGISTRY,
    reaction_catalog=REACTION_CATALOG,
    report_variable_registry=REPORT_VARIABLE_REGISTRY,
) -> RunBundle:
    errors: list[str] = []
    has_unknown_reaction_ids = False
    gas_species = set(run_bundle.chemistry.gas_species)
    solid_species = set(run_bundle.solids.solid_species)
    selected_species = gas_species | solid_species

    for species_id in run_bundle.chemistry.gas_species:
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

    for species_id in run_bundle.solids.solid_species:
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
