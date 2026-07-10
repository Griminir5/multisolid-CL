from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..properties import PROPERTY_REGISTRY
from ..reactions import ReactionDefinition, ReactionFamily
from . import KineticsContext
from .runtime import Constant, Exp, K, Max, Pa, Sqrt, m, mol, s


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
PRESSURE_PA_PER_BAR = 1.0e5
MIN_STEAM_PARTIAL_PRESSURE_BAR = 1.0e-2
STEAM_REFORMING_H2O_ORDER = 1.596
NI_MW_KG_PER_MOL = PROPERTY_REGISTRY.get_record("Ni").mw

NUMAGUCHI_RATE_COEFFICIENTS = {
    "smr": 3.65e5,
    "wgs": 2.45e5,
}
NUMAGUCHI_ACTIVATION_ENERGIES_J_PER_MOL = {
    "smr": 42800.0,
    "wgs": 54531.0,
}
SMR_EQUILIBRIUM_INTERCEPT = 30.114
SMR_EQUILIBRIUM_TEMPERATURE_TERM = -26830.0
WGS_EQUILIBRIUM_INTERCEPT = -4.036
WGS_EQUILIBRIUM_TEMPERATURE_TERM = 4400.0


@dataclass(frozen=True)
class NumaguchiTerms:
    temperature_k: Any
    p_ch4_bar: Any
    p_co_bar: Any
    p_co2_bar: Any
    p_h2_bar: Any
    p_h2o_bar: Any
    p_h2o_bar_safe: Any
    catalyst_mass_density_kg_per_m3: Any


def _temperature_k_expression(temperature) -> Any:
    return temperature / Constant(1.0 * K)


def _pressure_bar_expression(pressure) -> Any:
    return pressure / Constant(PRESSURE_PA_PER_BAR * Pa)


def _rate_constant_expression(rate_key: str, temperature_k, catalyst_mass_density_kg_per_m3) -> Any:
    coefficient = NUMAGUCHI_RATE_COEFFICIENTS[rate_key]
    activation_energy = NUMAGUCHI_ACTIVATION_ENERGIES_J_PER_MOL[rate_key]
    return (
        catalyst_mass_density_kg_per_m3
        * Constant(coefficient)
        * Exp(-Constant(activation_energy / GAS_CONSTANT_J_PER_MOL_K) / temperature_k)
    )


def _equilibrium_constant_expression(intercept: float, temperature_term: float, temperature_k) -> Any:
    return Exp(Constant(intercept) + Constant(temperature_term) / temperature_k)


def _equilibrium_constant_smr_expression(temperature_k) -> Any:
    return _equilibrium_constant_expression(
        SMR_EQUILIBRIUM_INTERCEPT,
        SMR_EQUILIBRIUM_TEMPERATURE_TERM,
        temperature_k,
    )


def _equilibrium_constant_wgs_expression(temperature_k) -> Any:
    return _equilibrium_constant_expression(
        WGS_EQUILIBRIUM_INTERCEPT,
        WGS_EQUILIBRIUM_TEMPERATURE_TERM,
        temperature_k,
    )


def _partial_pressure_bar_expression(context: KineticsContext, species_id: str):
    species_idx = context.gas_index(species_id)
    return _pressure_bar_expression(context.model.P(context.idx_cell)) * context.model.y_gas(species_idx, context.idx_cell)


def _catalyst_mass_density_expression(context: KineticsContext):
    ni_idx = context.solid_index("Ni")
    ni_concentration = context.model.c_sol(ni_idx, context.idx_cell) / Constant(1.0 * mol / m**3)
    return Max(ni_concentration, Constant(0.0)) * Constant(NI_MW_KG_PER_MOL)


def _safe_steam_partial_pressure_bar_expression(partial_pressure_bar):
    return Constant(0.5) * (
        partial_pressure_bar
        + Sqrt(partial_pressure_bar**2 + Constant((2.0 * MIN_STEAM_PARTIAL_PRESSURE_BAR) ** 2))
    )


def _numaguchi_terms(context: KineticsContext) -> NumaguchiTerms:
    temperature_k = _temperature_k_expression(context.model.T(context.idx_cell))
    p_h2o_bar = _partial_pressure_bar_expression(context, "H2O")
    return NumaguchiTerms(
        temperature_k=temperature_k,
        p_ch4_bar=_partial_pressure_bar_expression(context, "CH4"),
        p_co_bar=_partial_pressure_bar_expression(context, "CO"),
        p_co2_bar=_partial_pressure_bar_expression(context, "CO2"),
        p_h2_bar=_partial_pressure_bar_expression(context, "H2"),
        p_h2o_bar=p_h2o_bar,
        p_h2o_bar_safe=_safe_steam_partial_pressure_bar_expression(p_h2o_bar),
        catalyst_mass_density_kg_per_m3=_catalyst_mass_density_expression(context),
    )


def numaguchi_smr(context: KineticsContext):
    terms = _numaguchi_terms(context)
    driving_force = terms.p_ch4_bar * terms.p_h2o_bar - (
        terms.p_h2_bar**3 * terms.p_co_bar / _equilibrium_constant_smr_expression(terms.temperature_k)
    )
    rate_expression = _rate_constant_expression(
        "smr",
        terms.temperature_k,
        terms.catalyst_mass_density_kg_per_m3,
    ) * driving_force / terms.p_h2o_bar_safe**STEAM_REFORMING_H2O_ORDER
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


def numaguchi_wgs(context: KineticsContext):
    terms = _numaguchi_terms(context)
    driving_force = terms.p_co_bar * terms.p_h2o_bar - (
        terms.p_h2_bar * terms.p_co2_bar / _equilibrium_constant_wgs_expression(terms.temperature_k)
    )
    rate_expression = _rate_constant_expression(
        "wgs",
        terms.temperature_k,
        terms.catalyst_mass_density_kg_per_m3,
    ) * driving_force / terms.p_h2o_bar_safe
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


FAMILY = ReactionFamily(
    name="reforming_numaguchi",
    required_gas_species=("CH4", "H2O", "CO", "CO2", "H2"),
    required_solid_species=("Ni",),
    reactions=(
        ReactionDefinition(
            id="smr_reaction_numaguchi",
            name="Steam methane reforming on Ni (Numaguchi and Kikuchi) as documented by Andrew Wright",
            phase="gas_gas",
            stoichiometry={"CH4": -1.0, "H2O": -1.0, "CO": 1.0, "H2": 3.0},
            required_species=("CH4", "H2O", "CO", "H2", "Ni"),
            catalyst_species=("Ni",),
            reversible=True,
            source_reference="Andrew Wright, Chemical Looping Reactor Modelling – 2D, Technical Report",
            notes="This appears to be different from the actual paper by Numaguchi and Kikuchi",
        ),
        ReactionDefinition(
            id="wgs_reaction_numaguchi",
            name="Water-gas shift on Ni (Numaguchi and Kikuchi) as documented by Andrew Wright",
            phase="gas_gas",
            stoichiometry={"CO": -1.0, "H2O": -1.0, "CO2": 1.0, "H2": 1.0},
            required_species=("CO", "H2O", "CO2", "H2", "Ni"),
            catalyst_species=("Ni",),
            reversible=True,
            source_reference="Andrew Wright, Chemical Looping Reactor Modelling – 2D, Technical Report",
            notes="This appears to be different from the actual paper by Numaguchi and Kikuchi",
        ),
    ),
    kinetics_hooks={
        "smr_reaction_numaguchi": numaguchi_smr,
        "wgs_reaction_numaguchi": numaguchi_wgs,
    },
)


__all__ = ("FAMILY",)
