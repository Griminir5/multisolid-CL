from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


ReactionPhase = Literal["gas_gas", "gas_solid", "solid_solid"]
ReactionRateBasis = Literal["bed_volume", "gas_volume", "solid_volume", "catalyst_volume"]


def _unique_ordered(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return tuple(unique)


@dataclass(frozen=True)
class ReactionDefinition:
    id: str
    name: str
    phase: ReactionPhase
    stoichiometry: Mapping[str, float]
    required_species: tuple[str, ...]
    source_reference: str
    kinetics_hook: str | None = None
    reversible: bool = False
    catalyst_species: tuple[str, ...] = ()
    rate_basis: ReactionRateBasis = "bed_volume"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.stoichiometry:
            raise ValueError(f"Reaction '{self.id}' must define a non-empty stoichiometry mapping.")

        zero_species = sorted(
            species_id
            for species_id, coefficient in self.stoichiometry.items()
            if coefficient == 0.0
        )
        if zero_species:
            raise ValueError(
                f"Reaction '{self.id}' contains zero stoichiometric coefficients: {', '.join(zero_species)}."
            )

        stoich_species = set(self.stoichiometry)
        catalyst_species = set(self.catalyst_species)
        overlap = sorted(stoich_species & catalyst_species)
        if overlap:
            raise ValueError(
                f"Reaction '{self.id}' lists catalyst species in stoichiometry: {', '.join(overlap)}."
            )

        missing_required = sorted((stoich_species | catalyst_species) - set(self.required_species))
        if missing_required:
            raise ValueError(
                f"Reaction '{self.id}' required_species must include all stoichiometric and catalyst species: "
                f"{', '.join(missing_required)}."
            )

    @property
    def participating_species(self) -> tuple[str, ...]:
        return tuple(self.stoichiometry)

    @property
    def all_species(self) -> tuple[str, ...]:
        return _unique_ordered(tuple(self.required_species) + tuple(self.catalyst_species))

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


def _validate_reaction_phase_membership(
    reaction: ReactionDefinition,
    gas_species: tuple[str, ...],
    solid_species: tuple[str, ...],
) -> None:
    gas_species_set = set(gas_species)
    solid_species_set = set(solid_species)
    involved_species = set(reaction.participating_species)

    gas_members = sorted(involved_species & gas_species_set)
    solid_members = sorted(involved_species & solid_species_set)

    if reaction.phase == "gas_gas":
        if not gas_members:
            raise ValueError(f"Reaction '{reaction.id}' is marked gas_gas but has no selected gas species.")
        if solid_members:
            raise ValueError(
                f"Reaction '{reaction.id}' is marked gas_gas but references selected solid species: "
                f"{', '.join(solid_members)}."
            )

    if reaction.phase == "gas_solid":
        if not gas_members:
            raise ValueError(f"Reaction '{reaction.id}' is marked gas_solid but has no selected gas species.")
        if not solid_members:
            raise ValueError(f"Reaction '{reaction.id}' is marked gas_solid but has no selected solid species.")

    if reaction.phase == "solid_solid":
        if gas_members:
            raise ValueError(
                f"Reaction '{reaction.id}' is marked solid_solid but references selected gas species: "
                f"{', '.join(gas_members)}."
            )
        if not solid_members:
            raise ValueError(f"Reaction '{reaction.id}' is marked solid_solid but has no selected solid species.")


def build_reaction_network(
    reaction_ids: tuple[str, ...] | list[str],
    gas_species: tuple[str, ...] | list[str],
    solid_species: tuple[str, ...] | list[str],
    *,
    reaction_catalog: Mapping[str, ReactionDefinition],
) -> ReactionNetwork:
    gas_species_tuple = tuple(gas_species)
    solid_species_tuple = tuple(solid_species)
    reactions = tuple(reaction_catalog[reaction_id] for reaction_id in reaction_ids)

    for reaction in reactions:
        _validate_reaction_phase_membership(reaction, gas_species_tuple, solid_species_tuple)

    gas_source_matrix = tuple(
        tuple(reaction.source_coefficient(species_id) for reaction in reactions)
        for species_id in gas_species_tuple
    )
    solid_source_matrix = tuple(
        tuple(reaction.source_coefficient(species_id) for reaction in reactions)
        for species_id in solid_species_tuple
    )

    return ReactionNetwork(
        gas_species=gas_species_tuple,
        solid_species=solid_species_tuple,
        reactions=reactions,
        gas_source_matrix=gas_source_matrix,
        solid_source_matrix=solid_source_matrix,
    )


REACTION_CATALOG = {
    "ni_reduction_h2_medrano": ReactionDefinition(
        id="ni_reduction_h2_medrano",
        name="NiO reduction by H2",
        phase="gas_solid",
        stoichiometry={
            "H2": -1.0,
            "NiO": -1.0,
            "Ni": 1.0,
            "H2O": 1.0,
        },
        required_species=("H2", "H2O", "Ni", "NiO"),
        source_reference="Medrano et al., Applied Energy 2015, https://doi.org/10.1016/j.apenergy.2015.08.078",
        kinetics_hook=None,
        reversible=False,
        notes="Catalogued as metadata only in v1; assembly must reject it until kinetics are implemented.",
    ),
    "ni_reduction_co_medrano": ReactionDefinition(
        id="ni_reduction_co_medrano",
        name="NiO reduction by CO",
        phase="gas_solid",
        stoichiometry={
            "CO": -1.0,
            "NiO": -1.0,
            "Ni": 1.0,
            "CO2": 1.0,
        },
        required_species=("CO", "CO2", "Ni", "NiO"),
        source_reference="Medrano et al., Applied Energy 2015, https://doi.org/10.1016/j.apenergy.2015.08.078",
        kinetics_hook=None,
        reversible=False,
        notes="Catalogued as metadata only in v1; assembly must reject it until kinetics are implemented.",
    ),
    "ni_oxidation_o2_medrano": ReactionDefinition(
        id="ni_oxidation_o2_medrano",
        name="Ni oxidation by O2",
        phase="gas_solid",
        stoichiometry={
            "O2": -0.5,
            "Ni": -1.0,
            "NiO": 1.0,
        },
        required_species=("O2", "Ni", "NiO"),
        source_reference="Medrano et al., Applied Energy 2015, https://doi.org/10.1016/j.apenergy.2015.08.078",
        kinetics_hook=None,
        reversible=False,
        notes="Catalogued as metadata only in v1; assembly must reject it until kinetics are implemented.",
    ),
    "smr_reaction_numaguchi": ReactionDefinition(
        id="smr_reaction_numaguchi",
        name="Steam methane reforming on Ni",
        phase="gas_gas",
        stoichiometry={"CH4": -1.0, "H2O": -1.0, "CO": 1.0, "H2": 3.0},
        required_species=("CH4", "H2O", "CO", "H2", "Ni"),
        catalyst_species=("Ni",),
        reversible=True,
        kinetics_hook=None,
        source_reference="...",
        notes="Nickel-catalysed reversible steam methane reforming with kinetics metadata retained for future work.",
    ),
    "wgs_reaction_numaguchi": ReactionDefinition(
        id="wgs_reaction_numaguchi",
        name="Water-gas shift on Ni",
        phase="gas_gas",
        stoichiometry={"CO": -1.0, "H2O": -1.0, "CO2": 1.0, "H2": 1.0},
        required_species=("CO", "H2O", "CO2", "H2", "Ni"),
        catalyst_species=("Ni",),
        reversible=True,
        kinetics_hook=None,
        source_reference="...",
        notes="Nickel-catalysed reversible water-gas shift with kinetics metadata retained for future work.",
    ),
    "wgs_reaction_iron": ReactionDefinition(
        id="wgs_reaction_iron",
        name="Water-gas shift on Fe",
        phase="gas_gas",
        stoichiometry={"CO": -1.0, "H2O": -1.0, "CO2": 1.0, "H2": 1.0},
        required_species=("CO", "H2O", "CO2", "H2", "Fe"),
        catalyst_species=("Fe",),
        reversible=True,
        kinetics_hook=None,
        source_reference="...",
        notes="Iron-catalysed reversible water-gas shift reaction with kinetics taken from TBD.",
    ),
    "smr_reaction_xu_froment": ReactionDefinition(
        id="smr_reaction_xu_froment",
        name="Steam methane reforming on Ni (Xu-Froment)",
        phase="gas_gas",
        stoichiometry={"CH4": -1.0, "H2O": -1.0, "CO": 1.0, "H2": 3.0},
        required_species=("CH4", "H2O", "CO", "H2", "Ni"),
        catalyst_species=("Ni",),
        reversible=True,
        kinetics_hook="xu_froment_smr",
        source_reference="Xu and Froment, AIChE Journal 1989, https://doi.org/10.1002/aic.690350109",
        notes="One-reaction-one-hook Xu-Froment steam methane reforming rate expression.",
    ),
    "wgs_reaction_xu_froment": ReactionDefinition(
        id="wgs_reaction_xu_froment",
        name="Water-gas shift on Ni (Xu-Froment)",
        phase="gas_gas",
        stoichiometry={"CO": -1.0, "H2O": -1.0, "CO2": 1.0, "H2": 1.0},
        required_species=("CO", "H2O", "CO2", "H2", "Ni"),
        catalyst_species=("Ni",),
        reversible=True,
        kinetics_hook="xu_froment_wgs",
        source_reference="Xu and Froment, AIChE Journal 1989, https://doi.org/10.1002/aic.690350109",
        notes="One-reaction-one-hook Xu-Froment water-gas shift rate expression.",
    ),
    "overall_reforming_xu_froment": ReactionDefinition(
        id="overall_reforming_xu_froment",
        name="Overall steam reforming on Ni (Xu-Froment)",
        phase="gas_gas",
        stoichiometry={"CH4": -1.0, "H2O": -2.0, "CO2": 1.0, "H2": 4.0},
        required_species=("CH4", "H2O", "CO2", "H2", "Ni"),
        catalyst_species=("Ni",),
        reversible=True,
        kinetics_hook="xu_froment_overall",
        source_reference="Xu and Froment, AIChE Journal 1989, https://doi.org/10.1002/aic.690350109",
        notes="One-reaction-one-hook Xu-Froment overall reforming rate expression.",
    ),
}

DEFAULT_REACTION_CATALOG = REACTION_CATALOG
