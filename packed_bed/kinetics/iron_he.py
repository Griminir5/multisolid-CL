from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..properties import PROPERTY_REGISTRY
from ..reactions import ReactionDefinition, ReactionFamily
from . import KineticsContext
from .runtime import Constant, Exp, K, Log, Pa, Sqrt, m, mol, s


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
PRESSURE_PA_PER_BAR = 1.0e5

FE2O3_MW_KG_PER_MOL = PROPERTY_REGISTRY.get_record("Fe2O3").mw
FE3O4_MW_KG_PER_MOL = PROPERTY_REGISTRY.get_record("Fe3O4").mw
FEO_MW_KG_PER_MOL = PROPERTY_REGISTRY.get_record("FeO").mw
FE_MW_KG_PER_MOL = PROPERTY_REGISTRY.get_record("Fe").mw


# C_fullOx_Fe2O3 = (0.20 * 2500) / MW_Fe2O3
FULLY_OXIDIZED_FE2O3_CONCENTRATION_MOL_PER_M3 = (0.20 * 2500.0) / FE2O3_MW_KG_PER_MOL

# H2 reduction via shrinking-core model
AVERAGE_GRAIN_RADIUS_M = 5.3e-8
MOLAR_DENSITY_FE2O3_MOL_PER_M3 = 7.82e3

H2_REDUCTION_A_FACTORS = {
    "Fe2O3": 3.0,
    "Fe3O4": 1.0,
    "FeO": 1.0,
}
H2_REDUCTION_PREEXPONENTIALS = {
    "Fe2O3": 1.2e8,   # m/s
    "Fe3O4": 2.2e2,   # m/s
    "FeO": 1.5e2,     # m/s
}
H2_REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL = {
    "Fe2O3": 191.1e3,
    "Fe3O4": 129.2e3,
    "FeO": 120.2e3,
}
H2_DIFFUSIVITY_PREEXPONENTIALS = {
    "Fe2O3": 2.8e-12,  # m2/s
    "Fe3O4": 1.8e-9,   # m2/s
    "FeO": 3.8e-9,     # m2/s
}
H2_DIFFUSIVITY_ACTIVATION_ENERGIES_J_PER_MOL = {
    "Fe2O3": 37.5e3,
    "Fe3O4": 86.9e3,
    "FeO": 99.0e3,
}

# CO reduction
CO_REDUCTION_PREEXPONENTIALS = {
    "Fe2O3": 3.88e3,   # m3/kgOC/s
    "Fe3O4": 1.85e4,   # m3/kgOC/s
    "FeO": 1.1e-6,     # m4/mol/s
}
CO_REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL = {
    "Fe2O3": 75.0e3,
    "Fe3O4": 94.0e3,
    "FeO": 64.0e3,
}

# FeO + CO random pore model constants
FEO_CO_S0_PER_M = 2.5e6
FEO_CO_L0_PER_M2 = 7.5e12
FEO_CO_EPSILON0 = 0.6
FEO_MOLAR_VOLUME_M3_PER_MOL = 1.25e-5
FEO_CO_DIFFUSIVITY_PREEXPONENTIAL_M2_PER_S = 2.91e-7
FEO_CO_DIFFUSIVITY_ACTIVATION_ENERGY_J_PER_MOL = 90.0e3

# alpha = ratio of molar volume of solid product to reactant
# FeO -> Fe gives alpha = Vm(Fe) / Vm(FeO)
# Paper denotes alpha explicitly in Eq. 12 but does not list a numeric value in Table 3.
# Use literature-consistent value based on molar volumes.
FEO_CO_ALPHA = 0.567

# CH4 reduction of Fe2O3 only
CH4_REDUCTION_PREEXPONENTIAL = 1.46e10
CH4_REDUCTION_ACTIVATION_ENERGY_J_PER_MOL = 257.0e3

# Equilibrium constants, Table 2
H2_EQUILIBRIUM_COEFFICIENTS = {
    "Fe3O4_to_FeO": (7.563, -7393.9),
    "FeO_to_Fe": (1.239, -2023.8),
}
CO_EQUILIBRIUM_COEFFICIENTS = {
    "Fe3O4_to_FeO": (3.661, -3132.5),
    "FeO_to_Fe": (-2.667, 2240.6),
}

# Oxidation Eq. 22, written on O2-consumption basis
OXIDATION_PREEXPONENTIAL_O2_BASIS = 3.436e3  # mol_O2 / (kgOC * bar * s)
OXIDATION_ACTIVATION_ENERGY_J_PER_MOL = 32.0e3


@dataclass(frozen=True)
class FeReductionTerms:
    temperature_k: Any
    c_h2_mol_per_m3: Any
    c_h2o_mol_per_m3: Any
    c_co_mol_per_m3: Any
    c_co2_mol_per_m3: Any
    c_ch4_mol_per_m3: Any
    p_o2_bar: Any
    c_fe2o3_mol_per_m3: Any
    c_fe3o4_mol_per_m3: Any
    c_feo_mol_per_m3: Any
    c_fe_mol_per_m3: Any
    oc_mass_density_kg_per_m3: Any
    x_fe2o3: Any
    x_fe3o4: Any
    x_feo: Any
    x_fe: Any


FEO_CO_STRUCTURAL_PARAMETER = (
    4.0
    * math.pi
    * FEO_CO_L0_PER_M2
    * (1.0 - FEO_CO_EPSILON0)
    / FEO_CO_S0_PER_M**2
)


def _temperature_k_expression(temperature) -> Any:
    return temperature / Constant(1.0 * K)


def _pressure_bar_expression(pressure) -> Any:
    return pressure / Constant(PRESSURE_PA_PER_BAR * Pa)


def _arrhenius_expression(preexponential: float, activation_energy_j_per_mol: float, temperature_k) -> Any:
    return Constant(preexponential) * Exp(
        -Constant(activation_energy_j_per_mol / GAS_CONSTANT_J_PER_MOL_K) / temperature_k
    )


def _equilibrium_constant_expression(a: float, b: float, temperature_k) -> Any:
    return Exp(Constant(a) + Constant(b) / temperature_k)


def _solid_concentration_expression(context: KineticsContext, species_id: str):
    species_idx = context.solid_index(species_id)
    return context.model.c_sol(species_idx, context.idx_cell) / Constant(1.0 * mol / m**3)


def _optional_solid_concentration_expression(context: KineticsContext, species_id: str):
    try:
        species_idx = context.solid_index(species_id)
    except Exception:
        return Constant(0.0)
    concentration = context.model.c_sol(species_idx, context.idx_cell) / Constant(1.0 * mol / m**3)
    return _positive_part_expression(concentration)


def _gas_concentration_expression(context: KineticsContext, species_id: str):
    species_idx = context.gas_index(species_id)
    concentration = context.model.c_gas(species_idx, context.idx_cell) / Constant(1.0 * mol / m**3)
    return _positive_part_expression(concentration)


def _partial_pressure_bar_expression(context: KineticsContext, species_id: str):
    species_idx = context.gas_index(species_id)
    return _pressure_bar_expression(context.model.P(context.idx_cell)) * context.model.y_gas(
        species_idx, context.idx_cell
    )


def _positive_part_expression(expression):
    eps = Constant(1.0e-16)
    return Constant(0.5) * (expression + Sqrt(expression ** 2 + eps))


def _safe_one_minus_conversion(x_expr):
    eps = Constant(1.0e-12)
    return _positive_part_expression(Constant(1.0) - x_expr) + eps


def _pow_minus_two_thirds(one_minus_x):
    return one_minus_x ** Constant(-2.0 / 3.0)


def _pow_minus_one_third(one_minus_x):
    return one_minus_x ** Constant(-1.0 / 3.0)


def _oc_mass_density_expression(context: KineticsContext):
    return (
        _optional_solid_concentration_expression(context, "Fe2O3") * Constant(FE2O3_MW_KG_PER_MOL)
        + _optional_solid_concentration_expression(context, "Fe3O4") * Constant(FE3O4_MW_KG_PER_MOL)
        + _optional_solid_concentration_expression(context, "FeO") * Constant(FEO_MW_KG_PER_MOL)
        + _optional_solid_concentration_expression(context, "Fe") * Constant(FE_MW_KG_PER_MOL)
    )


def _fe_conversions(context: KineticsContext):
    c_fe2o3 = _optional_solid_concentration_expression(context, "Fe2O3")
    c_fe3o4 = _optional_solid_concentration_expression(context, "Fe3O4")
    c_feo = _optional_solid_concentration_expression(context, "FeO")
    c_fe = _optional_solid_concentration_expression(context, "Fe")
    c_full = Constant(FULLY_OXIDIZED_FE2O3_CONCENTRATION_MOL_PER_M3)

    # Eqs. 14-16 from paper
    x_fe2o3 = Constant(1.0) - c_fe2o3 / c_full
    x_fe3o4 = x_fe2o3 - Constant(1.5) * c_fe3o4 / c_full
    x_feo = x_fe3o4 - Constant(0.5) * c_feo / c_full

    # Eq. 33 for oxidation-side Fe conversion
    x_fe = Constant(1.0) - c_fe / (Constant(2.0) * c_full)

    return x_fe2o3, x_fe3o4, x_feo, x_fe


def _fe_terms(context: KineticsContext) -> FeReductionTerms:
    x_fe2o3, x_fe3o4, x_feo, x_fe = _fe_conversions(context)
    return FeReductionTerms(
        temperature_k=_temperature_k_expression(context.model.T(context.idx_cell)),
        c_h2_mol_per_m3=_gas_concentration_expression(context, "H2"),
        c_h2o_mol_per_m3=_gas_concentration_expression(context, "H2O"),
        c_co_mol_per_m3=_gas_concentration_expression(context, "CO"),
        c_co2_mol_per_m3=_gas_concentration_expression(context, "CO2"),
        c_ch4_mol_per_m3=_gas_concentration_expression(context, "CH4"),
        p_o2_bar=_partial_pressure_bar_expression(context, "O2"),
        c_fe2o3_mol_per_m3=_optional_solid_concentration_expression(context, "Fe2O3"),
        c_fe3o4_mol_per_m3=_optional_solid_concentration_expression(context, "Fe3O4"),
        c_feo_mol_per_m3=_optional_solid_concentration_expression(context, "FeO"),
        c_fe_mol_per_m3=_optional_solid_concentration_expression(context, "Fe"),
        oc_mass_density_kg_per_m3=_oc_mass_density_expression(context),
        x_fe2o3=x_fe2o3,
        x_fe3o4=x_fe3o4,
        x_feo=x_feo,
        x_fe=x_fe,
    )


def he_fe2o3_h2_reduction(context: KineticsContext):
    terms = _fe_terms(context)
    one_minus_x = _safe_one_minus_conversion(terms.x_fe2o3)

    denominator = (
        Constant(1.0)
        / _arrhenius_expression(
            H2_REDUCTION_PREEXPONENTIALS["Fe2O3"],
            H2_REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL["Fe2O3"],
            terms.temperature_k,
        )
        * _pow_minus_two_thirds(one_minus_x)
        + Constant(AVERAGE_GRAIN_RADIUS_M)
        / _arrhenius_expression(
            H2_DIFFUSIVITY_PREEXPONENTIALS["Fe2O3"],
            H2_DIFFUSIVITY_ACTIVATION_ENERGIES_J_PER_MOL["Fe2O3"],
            terms.temperature_k,
        )
        * (_pow_minus_one_third(one_minus_x) - Constant(1.0))
    )

    rate_expression = (
        Constant(3.0 / (AVERAGE_GRAIN_RADIUS_M * MOLAR_DENSITY_FE2O3_MOL_PER_M3))
        * Constant(H2_REDUCTION_A_FACTORS["Fe2O3"])
        * terms.c_fe2o3_mol_per_m3
        * _positive_part_expression(terms.c_h2_mol_per_m3)
        / denominator
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


def he_fe3o4_h2_reduction(context: KineticsContext):
    terms = _fe_terms(context)
    one_minus_x = _safe_one_minus_conversion(terms.x_fe3o4)
    keq = _equilibrium_constant_expression(*H2_EQUILIBRIUM_COEFFICIENTS["Fe3O4_to_FeO"], terms.temperature_k)
    driving_force = _positive_part_expression(terms.c_h2_mol_per_m3 - terms.c_h2o_mol_per_m3 / keq)

    denominator = (
        Constant(1.0)
        / _arrhenius_expression(
            H2_REDUCTION_PREEXPONENTIALS["Fe3O4"],
            H2_REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL["Fe3O4"],
            terms.temperature_k,
        )
        * _pow_minus_two_thirds(one_minus_x)
        + Constant(AVERAGE_GRAIN_RADIUS_M)
        / _arrhenius_expression(
            H2_DIFFUSIVITY_PREEXPONENTIALS["Fe3O4"],
            H2_DIFFUSIVITY_ACTIVATION_ENERGIES_J_PER_MOL["Fe3O4"],
            terms.temperature_k,
        )
        * (_pow_minus_one_third(one_minus_x) - Constant(1.0))
    )

    rate_expression = (
        Constant(3.0 / (AVERAGE_GRAIN_RADIUS_M * MOLAR_DENSITY_FE2O3_MOL_PER_M3))
        * Constant(H2_REDUCTION_A_FACTORS["Fe3O4"])
        * terms.c_fe3o4_mol_per_m3
        * driving_force
        / denominator
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


def he_feo_h2_reduction(context: KineticsContext):
    terms = _fe_terms(context)
    one_minus_x = _safe_one_minus_conversion(terms.x_feo)
    keq = _equilibrium_constant_expression(*H2_EQUILIBRIUM_COEFFICIENTS["FeO_to_Fe"], terms.temperature_k)
    driving_force = _positive_part_expression(terms.c_h2_mol_per_m3 - terms.c_h2o_mol_per_m3 / keq)

    denominator = (
        Constant(1.0)
        / _arrhenius_expression(
            H2_REDUCTION_PREEXPONENTIALS["FeO"],
            H2_REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL["FeO"],
            terms.temperature_k,
        )
        * _pow_minus_two_thirds(one_minus_x)
        + Constant(AVERAGE_GRAIN_RADIUS_M)
        / _arrhenius_expression(
            H2_DIFFUSIVITY_PREEXPONENTIALS["FeO"],
            H2_DIFFUSIVITY_ACTIVATION_ENERGIES_J_PER_MOL["FeO"],
            terms.temperature_k,
        )
        * (_pow_minus_one_third(one_minus_x) - Constant(1.0))
    )

    rate_expression = (
        Constant(3.0 / (AVERAGE_GRAIN_RADIUS_M * MOLAR_DENSITY_FE2O3_MOL_PER_M3))
        * Constant(H2_REDUCTION_A_FACTORS["FeO"])
        * terms.c_feo_mol_per_m3
        * driving_force
        / denominator
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


def he_fe2o3_co_reduction(context: KineticsContext):
    terms = _fe_terms(context)

    rate_expression = (
        terms.oc_mass_density_kg_per_m3
        * _arrhenius_expression(
            CO_REDUCTION_PREEXPONENTIALS["Fe2O3"],
            CO_REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL["Fe2O3"],
            terms.temperature_k,
        )
        * _positive_part_expression(terms.c_co_mol_per_m3)
        * _safe_one_minus_conversion(terms.x_fe2o3) ** Constant(0.4)
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


def he_fe3o4_co_reduction(context: KineticsContext):
    terms = _fe_terms(context)
    keq = _equilibrium_constant_expression(*CO_EQUILIBRIUM_COEFFICIENTS["Fe3O4_to_FeO"], terms.temperature_k)
    driving_force = _positive_part_expression(terms.c_co_mol_per_m3 - terms.c_co2_mol_per_m3 / keq)

    rate_expression = (
        terms.oc_mass_density_kg_per_m3
        * _arrhenius_expression(
            CO_REDUCTION_PREEXPONENTIALS["Fe3O4"],
            CO_REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL["Fe3O4"],
            terms.temperature_k,
        )
        * driving_force
        * _safe_one_minus_conversion(terms.x_fe3o4) ** Constant(1.2)
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


def he_feo_co_reduction(context: KineticsContext):
    terms = _fe_terms(context)
    one_minus_x = _safe_one_minus_conversion(terms.x_feo)

    keq = _equilibrium_constant_expression(*CO_EQUILIBRIUM_COEFFICIENTS["FeO_to_Fe"], terms.temperature_k)
    driving_force = _positive_part_expression(terms.c_co_mol_per_m3 - terms.c_co2_mol_per_m3 / keq)

    psi = Constant(FEO_CO_STRUCTURAL_PARAMETER)

    structure_term = Sqrt(Constant(1.0) - psi * Log(one_minus_x))

    k_feo_co = _arrhenius_expression(
        CO_REDUCTION_PREEXPONENTIALS["FeO"],
        CO_REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL["FeO"],
        terms.temperature_k,
    )
    d_feo_co = _arrhenius_expression(
        FEO_CO_DIFFUSIVITY_PREEXPONENTIAL_M2_PER_S,
        FEO_CO_DIFFUSIVITY_ACTIVATION_ENERGY_J_PER_MOL,
        terms.temperature_k,
    )

    beta = (
        Constant(2.0 * (1.0 - FEO_CO_EPSILON0))
        * k_feo_co
        / (Constant(FEO_MOLAR_VOLUME_M3_PER_MOL * FEO_CO_S0_PER_M) * d_feo_co)
    )

    denominator = Constant(1.0) + beta * Constant(FEO_CO_ALPHA) / psi * (structure_term - Constant(1.0))

    rate_expression = (
        Constant(FEO_CO_S0_PER_M / (1.0 - FEO_CO_EPSILON0))
        * terms.c_feo_mol_per_m3
        * k_feo_co
        * structure_term
        / denominator
        * driving_force
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


def he_fe2o3_ch4_reduction(context: KineticsContext):
    terms = _fe_terms(context)

    rate_expression = (
        terms.oc_mass_density_kg_per_m3
        * _arrhenius_expression(
            CH4_REDUCTION_PREEXPONENTIAL,
            CH4_REDUCTION_ACTIVATION_ENERGY_J_PER_MOL,
            terms.temperature_k,
        )
        * _positive_part_expression(terms.c_ch4_mol_per_m3)
        * _safe_one_minus_conversion(terms.x_fe2o3) ** Constant(2.0 / 3.0)
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


def he_fe_o2_oxidation(context: KineticsContext):
    terms = _fe_terms(context)

    r_o2_expression = (
        terms.oc_mass_density_kg_per_m3
        * _arrhenius_expression(
            OXIDATION_PREEXPONENTIAL_O2_BASIS,
            OXIDATION_ACTIVATION_ENERGY_J_PER_MOL,
            terms.temperature_k,
        )
        * _safe_one_minus_conversion(terms.x_fe)
        * _positive_part_expression(terms.p_o2_bar)
    )

    # paper: r_O2 = 0.75 r_Fe
    rate_expression = r_o2_expression / Constant(0.75)
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


FAMILY = ReactionFamily(
    name="iron_he",
    required_gas_species=("H2", "H2O", "CO", "CO2", "CH4", "O2"),
    required_solid_species=("Fe", "FeO", "Fe3O4", "Fe2O3"),
    reactions=(
        ReactionDefinition(
            id="fe2o3_h2_reduction_he_2023",
            name="Fe2O3 reduction to Fe3O4 by H2",
            phase="gas_solid",
            stoichiometry={"Fe2O3": -3.0, "H2": -1.0, "Fe3O4": 2.0, "H2O": 1.0},
            required_species=("Fe2O3", "Fe3O4", "H2", "H2O"),
            source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
            notes="First-stage Fe reduction by H2. Paper treats Fe2O3->Fe3O4 as effectively irreversible.",
        ),
        ReactionDefinition(
            id="fe3o4_h2_reduction_he_2023",
            name="Fe3O4 reduction to FeO by H2",
            phase="gas_solid",
            stoichiometry={"Fe3O4": -1.0, "H2": -1.0, "FeO": 3.0, "H2O": 1.0},
            required_species=("Fe3O4", "FeO", "H2", "H2O"),
            source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
            reversible=True,
            notes="Second-stage Fe reduction by H2 with equilibrium driving force.",
        ),
        ReactionDefinition(
            id="feo_h2_reduction_he_2023",
            name="FeO reduction to Fe by H2",
            phase="gas_solid",
            stoichiometry={"FeO": -1.0, "H2": -1.0, "Fe": 1.0, "H2O": 1.0},
            required_species=("FeO", "Fe", "H2", "H2O"),
            source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
            reversible=True,
            notes="Third-stage Fe reduction by H2 with equilibrium driving force.",
        ),
        ReactionDefinition(
            id="fe2o3_co_reduction_he_2023",
            name="Fe2O3 reduction to Fe3O4 by CO",
            phase="gas_solid",
            stoichiometry={"Fe2O3": -3.0, "CO": -1.0, "Fe3O4": 2.0, "CO2": 1.0},
            required_species=("Fe2O3", "Fe3O4", "CO", "CO2"),
            source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
            notes="First-stage Fe reduction by CO. Paper treats Fe2O3->Fe3O4 as effectively irreversible.",
        ),
        ReactionDefinition(
            id="fe3o4_co_reduction_he_2023",
            name="Fe3O4 reduction to FeO by CO",
            phase="gas_solid",
            stoichiometry={"Fe3O4": -1.0, "CO": -1.0, "FeO": 3.0, "CO2": 1.0},
            required_species=("Fe3O4", "FeO", "CO", "CO2"),
            source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
            reversible=True,
            notes="Second-stage Fe reduction by CO with equilibrium driving force.",
        ),
        ReactionDefinition(
            id="feo_co_reduction_he_2023",
            name="FeO reduction to Fe by CO",
            phase="gas_solid",
            stoichiometry={"FeO": -1.0, "CO": -1.0, "Fe": 1.0, "CO2": 1.0},
            required_species=("FeO", "Fe", "CO", "CO2"),
            source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
            reversible=True,
            notes="Third-stage Fe reduction by CO using the random pore model. This reaction entry is correct, but it will only run if the kinetics hook is completed with symbolic logarithm support.",
        ),
        ReactionDefinition(
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
            notes="Paper includes only CH4 reduction of hematite; CH4 reduction of Fe3O4 and FeO is neglected.",
        ),
        ReactionDefinition(
            id="fe_o2_oxidation_he_2023",
            name="Fe oxidation by O2 to equivalent Fe2O3",
            phase="gas_solid",
            stoichiometry={"Fe": -1.0, "O2": -0.75, "Fe2O3": 0.5},
            required_species=("Fe", "O2", "Fe2O3"),
            source_reference="He et al., Energy Conversion and Management 293 (2023) 117525",
            notes="Empirical oxidation law from the paper. The paper models oxidation as a single fast Fe+O2 step and handles Fe2O3/Fe3O4/FeO redistribution separately by solid-state transformation logic.",
        ),
    ),
    kinetics_hooks={
        "fe2o3_h2_reduction_he_2023": he_fe2o3_h2_reduction,
        "fe3o4_h2_reduction_he_2023": he_fe3o4_h2_reduction,
        "feo_h2_reduction_he_2023": he_feo_h2_reduction,
        "fe2o3_co_reduction_he_2023": he_fe2o3_co_reduction,
        "fe3o4_co_reduction_he_2023": he_fe3o4_co_reduction,
        "feo_co_reduction_he_2023": he_feo_co_reduction,
        "fe2o3_ch4_reduction_he_2023": he_fe2o3_ch4_reduction,
        "fe_o2_oxidation_he_2023": he_fe_o2_oxidation,
    },
)


__all__ = ("FAMILY",)
