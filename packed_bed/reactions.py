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
    selected_species = gas_species_set | solid_species_set

    missing_species = sorted(
        species_id
        for species_id in reaction.all_species
        if species_id not in selected_species
    )
    if missing_species:
        raise ValueError(
            f"Reaction '{reaction.id}' requires unselected species: {', '.join(missing_species)}."
        )

    stoichiometric_species = set(reaction.participating_species)
    gas_members = sorted(stoichiometric_species & gas_species_set)
    solid_members = sorted(stoichiometric_species & solid_species_set)

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
        kinetics_hook="medrano_reduction_h2",
        reversible=False,
        notes="Implemented with corrected Medrano shrinking-core redox kinetics.",
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
        kinetics_hook="medrano_reduction_co",
        reversible=False,
        notes="Implemented with corrected Medrano shrinking-core redox kinetics.",
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
        kinetics_hook="medrano_oxidation_o2",
        reversible=False,
        notes="Implemented with corrected Medrano shrinking-core redox kinetics.",
    ),
    "ni_reduction_h2_medrano_an": ReactionDefinition(
        id="ni_reduction_h2_medrano_an",
        name="NiO reduction by H2 (Medrano AN)",
        phase="gas_solid",
        stoichiometry={
            "H2": -1.0,
            "NiO": -1.0,
            "Ni": 1.0,
            "H2O": 1.0,
        },
        required_species=("H2", "H2O", "Ni", "NiO"),
        source_reference="Andrew Wright, Chemical Looping Reactor Modelling - 2D, Technical Report",
        kinetics_hook="medrano_an_reduction_h2",
        reversible=False,
        notes="Excerpt-based Medrano shrinking-core redox kinetics using rational fractional-power approximations.",
    ),
    "ni_reduction_co_medrano_an": ReactionDefinition(
        id="ni_reduction_co_medrano_an",
        name="NiO reduction by CO (Medrano AN)",
        phase="gas_solid",
        stoichiometry={
            "CO": -1.0,
            "NiO": -1.0,
            "Ni": 1.0,
            "CO2": 1.0,
        },
        required_species=("CO", "CO2", "Ni", "NiO"),
        source_reference="Andrew Wright, Chemical Looping Reactor Modelling - 2D, Technical Report",
        kinetics_hook="medrano_an_reduction_co",
        reversible=False,
        notes="Excerpt-based Medrano shrinking-core redox kinetics using rational fractional-power approximations.",
    ),
    "ni_oxidation_o2_medrano_an": ReactionDefinition(
        id="ni_oxidation_o2_medrano_an",
        name="Ni oxidation by O2 (Medrano AN)",
        phase="gas_solid",
        stoichiometry={
            "O2": -0.5,
            "Ni": -1.0,
            "NiO": 1.0,
        },
        required_species=("O2", "Ni", "NiO"),
        source_reference="Andrew Wright, Chemical Looping Reactor Modelling - 2D, Technical Report",
        kinetics_hook="medrano_an_oxidation_o2",
        reversible=False,
        notes="Excerpt-based Medrano shrinking-core redox kinetics using rational fractional-power approximations.",
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
    "smr_reaction_numaguchi_an": ReactionDefinition(
        id="smr_reaction_numaguchi_an",
        name="Steam methane reforming on Ni (Numaguchi and Kikuchi) as documented by Andrew Wright",
        phase="gas_gas",
        stoichiometry={"CH4": -1.0, "H2O": -1.0, "CO": 1.0, "H2": 3.0},
        required_species=("CH4", "H2O", "CO", "H2", "Ni"),
        catalyst_species=("Ni",),
        reversible=True,
        kinetics_hook="numaguchi_smr_an",
        source_reference="Andrew Wright, Chemical Looping Reactor Modelling – 2D, Technical Report",
        notes="This appears to be different from the actual paper by Numaguchi and Kikuchi",
    ),
    "wgs_reaction_numaguchi_an": ReactionDefinition(
        id="wgs_reaction_numaguchi_an",
        name="Water-gas shift on Ni (Numaguchi and Kikuchi) as documented by Andrew Wright",
        phase="gas_gas",
        stoichiometry={"CO": -1.0, "H2O": -1.0, "CO2": 1.0, "H2": 1.0},
        required_species=("CO", "H2O", "CO2", "H2", "Ni"),
        catalyst_species=("Ni",),
        reversible=True,
        kinetics_hook="numaguchi_wgs_an",
        source_reference="Andrew Wright, Chemical Looping Reactor Modelling – 2D, Technical Report",
        notes="This appears to be different from the actual paper by Numaguchi and Kikuchi",
    ),
    "cuo_h2_reduction_san_pio": ReactionDefinition(
        id="cuo_h2_reduction_san_pio",
        name="CuO reduction to Cu2O by H2",
        phase="gas_solid",
        stoichiometry={
            "H2": -1.0,
            "CuO": -2.0,
            "Cu2O": 1.0,
            "H2O": 1.0,
        },
        required_species=("H2", "H2O", "CuO", "Cu2O"),
        kinetics_hook="san_pio_cuo_h2_reduction_ph",
        reversible=False,
        source_reference="San Pio et al., Chemical Engineering Science 175 (2018) 56-71",
        notes="Pseudo-homogeneous CuO reduction reaction rred1 = kred1 * C_CuO from Table 4.",
    ),
    "cu2o_h2_reduction_san_pio": ReactionDefinition(
        id="cu2o_h2_reduction_san_pio",
        name="Cu2O reduction to Cu by H2",
        phase="gas_solid",
        stoichiometry={
            "H2": -1.0,
            "Cu2O": -1.0,
            "Cu": 2.0,
            "H2O": 1.0,
        },
        required_species=("H2", "H2O", "Cu2O", "Cu"),
        kinetics_hook="san_pio_cu2o_h2_reduction_ph",
        reversible=False,
        source_reference="San Pio et al., Chemical Engineering Science 175 (2018) 56-71",
        notes="Pseudo-homogeneous Cu2O reduction reaction rred2 = kred2 * C_Cu2O from Table 4.",
    ),
    "cu_al2o3_spinel_reduction_1_san_pio": ReactionDefinition(
        id="cu_al2o3_spinel_reduction_1_san_pio",
        name="CuAl2O4 reduction to Cu on CuO/Al2O3",
        phase="gas_solid",
        stoichiometry={
            "H2": -1.0,
            "CuAl2O4": -1.0,
            "Cu": 1.0,
            "Al2O3": 1.0,
            "H2O": 1.0,
        },
        required_species=("H2", "H2O", "CuAl2O4", "Cu", "Al2O3"),
        kinetics_hook="san_pio_cu_al2o3_sp1_ph",
        reversible=False,
        source_reference="San Pio et al., Chemical Engineering Science 175 (2018) 56-71",
        notes="Pseudo-homogeneous spinel reduction reaction rsp1 = ksp1 * C_CuAl2O4 for CuO/Al2O3.",
    ),

    "cu_al2o3_spinel_reduction_2_san_pio": ReactionDefinition(
        id="cu_al2o3_spinel_reduction_2_san_pio",
        name="CuAl2O4 reduction to CuAlO2 on CuO/Al2O3",
        phase="gas_solid",
        stoichiometry={
            "H2": -1.0,
            "CuAl2O4": -2.0,
            "CuAlO2": 2.0,
            "Al2O3": 1.0,
            "H2O": 1.0,
        },
        required_species=("H2", "H2O", "CuAl2O4", "CuAlO2", "Al2O3"),
        kinetics_hook="san_pio_cu_al2o3_sp2_ph",
        reversible=False,
        source_reference="San Pio et al., Chemical Engineering Science 175 (2018) 56-71",
        notes="Pseudo-homogeneous spinel reduction reaction rsp2 = ksp2 * C_CuAl2O4 for CuO/Al2O3.",
    ),

    "cu_al2o3_spinel_reduction_3_san_pio": ReactionDefinition(
        id="cu_al2o3_spinel_reduction_3_san_pio",
        name="CuAlO2 reduction to Cu on CuO/Al2O3",
        phase="gas_solid",
        stoichiometry={
            "H2": -1.0,
            "CuAlO2": -2.0,
            "Cu": 2.0,
            "Al2O3": 1.0,
            "H2O": 1.0,
        },
        required_species=("H2", "H2O", "CuAlO2", "Cu", "Al2O3"),
        kinetics_hook="san_pio_cu_al2o3_sp3_ph",
        reversible=False,
        source_reference="San Pio et al., Chemical Engineering Science 175 (2018) 56-71",
        notes="Pseudo-homogeneous spinel reduction reaction rsp3 = ksp3 * C_CuAlO2 for CuO/Al2O3.",
    ),

    "cu_al2o3_oxidation_1_san_pio": ReactionDefinition(
        id="cu_al2o3_oxidation_1_san_pio",
        name="Cu oxidation to CuO on CuO/Al2O3",
        phase="gas_solid",
        stoichiometry={
            "O2": -0.5,
            "Cu": -1.0,
            "CuO": 1.0,
        },
        required_species=("O2", "Cu", "CuO"),
        kinetics_hook="san_pio_cu_al2o3_ox1_ph",
        reversible=False,
        source_reference="San Pio et al., Chemical Engineering Science 175 (2018) 56-71",
        notes="Pseudo-homogeneous oxidation reaction rox1 = kox1 * C_Cu * P_O2^0.5 for CuO/Al2O3.",
    ),

    "cu_al2o3_oxidation_2_san_pio": ReactionDefinition(
        id="cu_al2o3_oxidation_2_san_pio",
        name="CuO reaction with Al2O3 to form CuAl2O4",
        phase="solid_solid",
        stoichiometry={
            "CuO": -1.0,
            "Al2O3": -1.0,
            "CuAl2O4": 1.0,
        },
        required_species=("CuO", "Al2O3", "CuAl2O4"),
        kinetics_hook="san_pio_cu_al2o3_ox2_ph",
        reversible=False,
        source_reference="San Pio et al., Chemical Engineering Science 175 (2018) 56-71",
        notes="Pseudo-homogeneous oxidation reaction rox2 = kox2 * C_CuO * C_Al2O3 for CuO/Al2O3.",
    ),

    "cu_al2o3_oxidation_3_san_pio": ReactionDefinition(
        id="cu_al2o3_oxidation_3_san_pio",
        name="CuAlO2 oxidation with Al2O3 to form CuAl2O4",
        phase="gas_solid",
        stoichiometry={
            "O2": -0.5,
            "CuAlO2": -2.0,
            "Al2O3": -1.0,
            "CuAl2O4": 2.0,
        },
        required_species=("O2", "CuAlO2", "Al2O3", "CuAl2O4"),
        kinetics_hook="san_pio_cu_al2o3_ox3_ph",
        reversible=False,
        source_reference="San Pio et al., Chemical Engineering Science 175 (2018) 56-71",
        notes="Pseudo-homogeneous oxidation reaction rox3 = kox3 * C_CuAlO2 * C_Al2O3 * P_O2^0.5 for CuO/Al2O3.",
    ),
    "fe2o3_h2_reduction_he_2023": ReactionDefinition(
        id="fe2o3_h2_reduction_he_2023",
        name="Fe2O3 reduction to Fe3O4 by H2",
        phase="gas_solid",
        stoichiometry={
            "Fe2O3": -3.0,
            "H2": -1.0,
            "Fe3O4": 2.0,
            "H2O": 1.0,
        },
        required_species=("Fe2O3", "Fe3O4", "H2", "H2O"),
        source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
        kinetics_hook="he_fe2o3_h2_reduction",
        reversible=False,
        notes="First-stage Fe reduction by H2. Paper treats Fe2O3->Fe3O4 as effectively irreversible.",
    ),

    "fe3o4_h2_reduction_he_2023": ReactionDefinition(
        id="fe3o4_h2_reduction_he_2023",
        name="Fe3O4 reduction to FeO by H2",
        phase="gas_solid",
        stoichiometry={
            "Fe3O4": -1.0,
            "H2": -1.0,
            "FeO": 3.0,
            "H2O": 1.0,
        },
        required_species=("Fe3O4", "FeO", "H2", "H2O"),
        source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
        kinetics_hook="he_fe3o4_h2_reduction",
        reversible=True,
        notes="Second-stage Fe reduction by H2 with equilibrium driving force.",
    ),

    "feo_h2_reduction_he_2023": ReactionDefinition(
        id="feo_h2_reduction_he_2023",
        name="FeO reduction to Fe by H2",
        phase="gas_solid",
        stoichiometry={
            "FeO": -1.0,
            "H2": -1.0,
            "Fe": 1.0,
            "H2O": 1.0,
        },
        required_species=("FeO", "Fe", "H2", "H2O"),
        source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
        kinetics_hook="he_feo_h2_reduction",
        reversible=True,
        notes="Third-stage Fe reduction by H2 with equilibrium driving force.",
    ),

    "fe2o3_co_reduction_he_2023": ReactionDefinition(
        id="fe2o3_co_reduction_he_2023",
        name="Fe2O3 reduction to Fe3O4 by CO",
        phase="gas_solid",
        stoichiometry={
            "Fe2O3": -3.0,
            "CO": -1.0,
            "Fe3O4": 2.0,
            "CO2": 1.0,
        },
        required_species=("Fe2O3", "Fe3O4", "CO", "CO2"),
        source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
        kinetics_hook="he_fe2o3_co_reduction",
        reversible=False,
        notes="First-stage Fe reduction by CO. Paper treats Fe2O3->Fe3O4 as effectively irreversible.",
    ),

    "fe3o4_co_reduction_he_2023": ReactionDefinition(
        id="fe3o4_co_reduction_he_2023",
        name="Fe3O4 reduction to FeO by CO",
        phase="gas_solid",
        stoichiometry={
            "Fe3O4": -1.0,
            "CO": -1.0,
            "FeO": 3.0,
            "CO2": 1.0,
        },
        required_species=("Fe3O4", "FeO", "CO", "CO2"),
        source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
        kinetics_hook="he_fe3o4_co_reduction",
        reversible=True,
        notes="Second-stage Fe reduction by CO with equilibrium driving force.",
    ),

    "feo_co_reduction_he_2023": ReactionDefinition(
        id="feo_co_reduction_he_2023",
        name="FeO reduction to Fe by CO",
        phase="gas_solid",
        stoichiometry={
            "FeO": -1.0,
            "CO": -1.0,
            "Fe": 1.0,
            "CO2": 1.0,
        },
        required_species=("FeO", "Fe", "CO", "CO2"),
        source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
        kinetics_hook="he_feo_co_reduction",
        reversible=True,
        notes="Third-stage Fe reduction by CO using the random pore model. This reaction entry is correct, but it will only run if the kinetics hook is completed with symbolic logarithm support.",
    ),

    "fe2o3_ch4_reduction_he_2023": ReactionDefinition(
        id="fe2o3_ch4_reduction_he_2023",
        name="Fe2O3 reduction to Fe3O4 by CH4",
        phase="gas_solid",
        stoichiometry={
            "Fe2O3": -12.0,
            "CH4": -1.0,
            "Fe3O4": 8.0,
            "CO2": 1.0,
            "H2O": 2.0,
        },
        required_species=("Fe2O3", "Fe3O4", "CH4", "CO2", "H2O"),
        source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
        kinetics_hook="he_fe2o3_ch4_reduction",
        reversible=False,
        notes="Paper includes only CH4 reduction of hematite; CH4 reduction of Fe3O4 and FeO is neglected.",
    ),

    "fe_o2_oxidation_he_2023": ReactionDefinition(
        id="fe_o2_oxidation_he_2023",
        name="Fe oxidation by O2 to equivalent Fe2O3",
        phase="gas_solid",
        stoichiometry={
            "Fe": -1.0,
            "O2": -0.75,
            "Fe2O3": 0.5,
        },
        required_species=("Fe", "O2", "Fe2O3"),
        source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
        kinetics_hook="he_fe_o2_oxidation",
        reversible=False,
        notes="Empirical oxidation law from the paper. The paper models oxidation as a single fast Fe+O2 step and handles Fe2O3/Fe3O4/FeO redistribution separately by solid-state transformation logic.",
    ),
}

DEFAULT_REACTION_CATALOG = REACTION_CATALOG
