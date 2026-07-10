from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal, Mapping


ReactionPhase = Literal["gas_gas", "gas_solid", "solid_solid"]
ReactionRateBasis = Literal["bed_volume", "gas_volume", "solid_volume", "catalyst_volume"]
KineticsHook = Callable[[Any], Any]


def _unique_ordered(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class ReactionDefinition:
    id: str
    name: str
    phase: ReactionPhase
    stoichiometry: Mapping[str, float]
    required_species: tuple[str, ...]
    source_reference: str
    reversible: bool = False
    catalyst_species: tuple[str, ...] = ()
    rate_basis: ReactionRateBasis = "bed_volume"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id or self.id != self.id.strip():
            raise ValueError("Reaction ids must not be blank or padded.")
        if not self.name or self.name != self.name.strip():
            raise ValueError(f"Reaction '{self.id}' must have a readable name.")
        if not self.source_reference or self.source_reference != self.source_reference.strip():
            raise ValueError(f"Reaction '{self.id}' must have a source reference.")
        if any(not species or species != species.strip() for species in self.required_species):
            raise ValueError(f"Reaction '{self.id}' contains an invalid required species id.")
        if len(self.required_species) != len(set(self.required_species)):
            raise ValueError(f"Reaction '{self.id}' contains duplicate required species.")
        if len(self.catalyst_species) != len(set(self.catalyst_species)):
            raise ValueError(f"Reaction '{self.id}' contains duplicate catalyst species.")
        if not self.stoichiometry:
            raise ValueError(f"Reaction '{self.id}' must define a non-empty stoichiometry mapping.")
        invalid_coefficients = sorted(
            species_id
            for species_id, coefficient in self.stoichiometry.items()
            if coefficient == 0.0 or not math.isfinite(coefficient)
        )
        if invalid_coefficients:
            raise ValueError(
                f"Reaction '{self.id}' contains zero or non-finite stoichiometric "
                f"coefficients: {', '.join(invalid_coefficients)}."
            )

        stoichiometric_species = set(self.stoichiometry)
        catalysts = set(self.catalyst_species)
        overlap = sorted(stoichiometric_species & catalysts)
        if overlap:
            raise ValueError(
                f"Reaction '{self.id}' lists catalyst species in stoichiometry: "
                f"{', '.join(overlap)}."
            )
        missing_required = sorted(
            (stoichiometric_species | catalysts) - set(self.required_species)
        )
        if missing_required:
            raise ValueError(
                f"Reaction '{self.id}' required_species must include all stoichiometric "
                f"and catalyst species: {', '.join(missing_required)}."
            )

    @property
    def participating_species(self) -> tuple[str, ...]:
        return tuple(self.stoichiometry)

    @property
    def all_species(self) -> tuple[str, ...]:
        return _unique_ordered(self.required_species + self.catalyst_species)

    @property
    def reactants(self) -> Mapping[str, float]:
        return {
            species_id: -coefficient
            for species_id, coefficient in self.stoichiometry.items()
            if coefficient < 0.0
        }

    @property
    def products(self) -> Mapping[str, float]:
        return {
            species_id: coefficient
            for species_id, coefficient in self.stoichiometry.items()
            if coefficient > 0.0
        }

    def source_coefficient(self, species_id: str) -> float:
        return float(self.stoichiometry.get(species_id, 0.0))

    def has_catalyst(self, species_id: str) -> bool:
        return species_id in self.catalyst_species


@dataclass(frozen=True)
class ReactionFamily:
    name: str
    reactions: tuple[ReactionDefinition, ...]
    required_gas_species: tuple[str, ...]
    required_solid_species: tuple[str, ...]
    kinetics_hooks: Mapping[str, KineticsHook]

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("Reaction family names must not be blank or padded.")
        reaction_ids = tuple(reaction.id for reaction in self.reactions)
        if len(reaction_ids) != len(set(reaction_ids)):
            raise ValueError(f"Reaction family '{self.name}' contains duplicate reaction ids.")
        if len(self.required_gas_species) != len(set(self.required_gas_species)):
            raise ValueError(f"Reaction family '{self.name}' contains duplicate gas requirements.")
        if len(self.required_solid_species) != len(set(self.required_solid_species)):
            raise ValueError(f"Reaction family '{self.name}' contains duplicate solid requirements.")
        if any(
            not species or species != species.strip()
            for species in self.required_gas_species + self.required_solid_species
        ):
            raise ValueError(f"Reaction family '{self.name}' contains an invalid species requirement.")
        phase_overlap = sorted(
            set(self.required_gas_species) & set(self.required_solid_species)
        )
        if phase_overlap:
            raise ValueError(
                f"Reaction family '{self.name}' declares species in both phases: "
                f"{', '.join(phase_overlap)}."
            )

        requirements = set(self.required_gas_species) | set(self.required_solid_species)
        missing_requirements = sorted(
            species_id
            for reaction in self.reactions
            for species_id in reaction.all_species
            if species_id not in requirements
        )
        if missing_requirements:
            raise ValueError(
                f"Reaction family '{self.name}' does not declare requirements for: "
                f"{', '.join(dict.fromkeys(missing_requirements))}."
            )

        missing_hooks = sorted(set(reaction_ids) - set(self.kinetics_hooks))
        extra_hooks = sorted(set(self.kinetics_hooks) - set(reaction_ids))
        if missing_hooks or extra_hooks:
            differences = []
            if missing_hooks:
                differences.append(f"missing {', '.join(missing_hooks)}")
            if extra_hooks:
                differences.append(f"unexpected {', '.join(extra_hooks)}")
            raise ValueError(
                f"Reaction family '{self.name}' kinetics hook mismatch: "
                f"{'; '.join(differences)}."
            )
        non_callable_hooks = sorted(
            reaction_id
            for reaction_id, hook in self.kinetics_hooks.items()
            if not callable(hook)
        )
        if non_callable_hooks:
            raise ValueError(
                f"Reaction family '{self.name}' has non-callable kinetics hooks: "
                f"{', '.join(non_callable_hooks)}."
            )

    @property
    def reaction_ids(self) -> tuple[str, ...]:
        return tuple(reaction.id for reaction in self.reactions)


@dataclass(frozen=True)
class ReactionNetwork:
    gas_species: tuple[str, ...]
    solid_species: tuple[str, ...]
    reactions: tuple[ReactionDefinition, ...]
    gas_source_matrix: tuple[tuple[float, ...], ...]
    solid_source_matrix: tuple[tuple[float, ...], ...]

    @property
    def reaction_ids(self) -> tuple[str, ...]:
        return tuple(reaction.id for reaction in self.reactions)

    @property
    def reaction_count(self) -> int:
        return len(self.reactions)

    @property
    def has_reactions(self) -> bool:
        return bool(self.reactions)

    def gas_coefficients(self, gas_species_id: str) -> tuple[float, ...]:
        return self.gas_source_matrix[self.gas_species.index(gas_species_id)]

    def solid_coefficients(self, solid_species_id: str) -> tuple[float, ...]:
        return self.solid_source_matrix[self.solid_species.index(solid_species_id)]


def reaction_catalog(families: tuple[ReactionFamily, ...]) -> dict[str, ReactionDefinition]:
    catalog: dict[str, ReactionDefinition] = {}
    owners: dict[str, str] = {}
    for family in families:
        for reaction in family.reactions:
            if reaction.id in catalog:
                raise ValueError(
                    f"Reaction '{reaction.id}' is provided by both '{owners[reaction.id]}' "
                    f"and '{family.name}'."
                )
            catalog[reaction.id] = reaction
            owners[reaction.id] = family.name
    return catalog


def _validate_reaction_phase_membership(
    reaction: ReactionDefinition,
    gas_species: tuple[str, ...],
    solid_species: tuple[str, ...],
) -> None:
    gas_species_set = set(gas_species)
    solid_species_set = set(solid_species)
    selected_species = gas_species_set | solid_species_set
    missing_species = sorted(set(reaction.all_species) - selected_species)
    if missing_species:
        raise ValueError(
            f"Reaction '{reaction.id}' requires unselected species: "
            f"{', '.join(missing_species)}."
        )

    stoichiometric_species = set(reaction.participating_species)
    gas_members = sorted(stoichiometric_species & gas_species_set)
    solid_members = sorted(stoichiometric_species & solid_species_set)
    if reaction.phase == "gas_gas":
        if not gas_members:
            raise ValueError(
                f"Reaction '{reaction.id}' is marked gas_gas but has no selected gas species."
            )
        if solid_members:
            raise ValueError(
                f"Reaction '{reaction.id}' is marked gas_gas but references selected "
                f"solid species: {', '.join(solid_members)}."
            )
    elif reaction.phase == "gas_solid":
        if not gas_members or not solid_members:
            missing_phase = "gas" if not gas_members else "solid"
            raise ValueError(
                f"Reaction '{reaction.id}' is marked gas_solid but has no selected "
                f"{missing_phase} species."
            )
    elif gas_members:
        raise ValueError(
            f"Reaction '{reaction.id}' is marked solid_solid but references selected "
            f"gas species: {', '.join(gas_members)}."
        )
    elif not solid_members:
        raise ValueError(
            f"Reaction '{reaction.id}' is marked solid_solid but has no selected solid species."
        )


def build_reaction_network(
    reaction_ids: tuple[str, ...] | list[str],
    gas_species: tuple[str, ...] | list[str],
    solid_species: tuple[str, ...] | list[str],
    *,
    families: tuple[ReactionFamily, ...],
) -> ReactionNetwork:
    gas_species_tuple = tuple(gas_species)
    solid_species_tuple = tuple(solid_species)
    catalog = reaction_catalog(families)
    reactions = tuple(catalog[reaction_id] for reaction_id in reaction_ids)
    for reaction in reactions:
        _validate_reaction_phase_membership(reaction, gas_species_tuple, solid_species_tuple)

    return ReactionNetwork(
        gas_species=gas_species_tuple,
        solid_species=solid_species_tuple,
        reactions=reactions,
        gas_source_matrix=tuple(
            tuple(reaction.source_coefficient(species_id) for reaction in reactions)
            for species_id in gas_species_tuple
        ),
        solid_source_matrix=tuple(
            tuple(reaction.source_coefficient(species_id) for reaction in reactions)
            for species_id in solid_species_tuple
        ),
    )


__all__ = (
    "KineticsHook",
    "ReactionDefinition",
    "ReactionFamily",
    "ReactionNetwork",
    "build_reaction_network",
    "reaction_catalog",
)
