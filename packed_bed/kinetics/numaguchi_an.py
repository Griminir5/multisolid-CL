from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from daetools.pyDAE import Constant, Exp, Sqrt

from pyUnits import K, Pa, m, mol, s

from ..properties import PROPERTY_REGISTRY
from . import KineticsContext, register_kinetics_hook


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
PRESSURE_PA_PER_BAR = 1.0e5
MIN_STEAM_PARTIAL_PRESSURE_BAR = 1.0e-2
STEAM_REFORMING_H2O_ORDER = 1.596
NI_MW_KG_PER_MOL = PROPERTY_REGISTRY.get_record("Ni").mw

NUMAGUCHI_RATE_COEFFICIENTS = {
    "smr": 3.65e2,
    "wgs": 2.45e2,
}
NUMAGUCHI_ACTIVATION_ENERGIES_J_PER_MOL = {
    "smr": 42800.0,
    "wgs": 54500.0,
}


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


def partial_pressure_value(total_pressure_pa: float, mole_fraction: float) -> float:
    return total_pressure_pa * mole_fraction


def pressure_bar_value(pressure_pa: float) -> float:
    return pressure_pa / PRESSURE_PA_PER_BAR


def partial_pressure_bar_value(total_pressure_pa: float, mole_fraction: float) -> float:
    return pressure_bar_value(partial_pressure_value(total_pressure_pa, mole_fraction))


def safe_steam_partial_pressure_bar_value(partial_pressure_bar: float) -> float:
    return 0.5 * (
        partial_pressure_bar
        + math.sqrt(partial_pressure_bar**2 + (2.0 * MIN_STEAM_PARTIAL_PRESSURE_BAR) ** 2)
    )


def catalyst_mass_density_value(ni_concentration_mol_per_m3: float) -> float:
    return ni_concentration_mol_per_m3 * NI_MW_KG_PER_MOL


def rate_constant_value(
    rate_key: str,
    *,
    catalyst_mass_density_kg_per_m3: float,
    temperature_k: float,
) -> float:
    coefficient = NUMAGUCHI_RATE_COEFFICIENTS[rate_key]
    activation_energy = NUMAGUCHI_ACTIVATION_ENERGIES_J_PER_MOL[rate_key]
    return catalyst_mass_density_kg_per_m3 * coefficient * math.exp(
        -activation_energy / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
    )


def equilibrium_constant_smr_value(temperature_k: float) -> float:
    return math.exp(-2683.0 / temperature_k + 30.114)


def equilibrium_constant_wgs_value(temperature_k: float) -> float:
    return math.exp(4400.0 / temperature_k - 4.036)



def smr_rate_value(
    *,
    temperature_k: float,
    p_ch4_pa: float,
    p_h2o_pa: float,
    p_co_pa: float,
    p_h2_pa: float,
    catalyst_mass_density_kg_per_m3: float,
) -> float:
    p_ch4_bar = pressure_bar_value(p_ch4_pa)
    p_h2o_bar = pressure_bar_value(p_h2o_pa)
    p_co_bar = pressure_bar_value(p_co_pa)
    p_h2_bar = pressure_bar_value(p_h2_pa)
    driving_force = p_ch4_bar * p_h2o_bar - (p_h2_bar**3 * p_co_bar) / equilibrium_constant_smr_value(temperature_k)
    return (
        rate_constant_value(
            "smr",
            catalyst_mass_density_kg_per_m3=catalyst_mass_density_kg_per_m3,
            temperature_k=temperature_k,
        )
        * driving_force
        / safe_steam_partial_pressure_bar_value(p_h2o_bar) ** STEAM_REFORMING_H2O_ORDER
    )


def wgs_rate_value(
    *,
    temperature_k: float,
    p_co_pa: float,
    p_h2o_pa: float,
    p_co2_pa: float,
    p_h2_pa: float,
    p_ch4_pa: float = 0.0,
    catalyst_mass_density_kg_per_m3: float,
) -> float:
    p_co_bar = pressure_bar_value(p_co_pa)
    p_h2o_bar = pressure_bar_value(p_h2o_pa)
    p_co2_bar = pressure_bar_value(p_co2_pa)
    p_h2_bar = pressure_bar_value(p_h2_pa)
    driving_force = p_co_bar * p_h2o_bar - (p_h2_bar * p_co2_bar) / equilibrium_constant_wgs_value(temperature_k)
    return (
        rate_constant_value(
            "wgs",
            catalyst_mass_density_kg_per_m3=catalyst_mass_density_kg_per_m3,
            temperature_k=temperature_k,
        )
        * driving_force
        / safe_steam_partial_pressure_bar_value(p_h2o_bar)
    )


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


def _equilibrium_constant_smr_expression(temperature_k) -> Any:
    return Exp(-Constant(2683.0) / temperature_k + Constant(30.114))


def _equilibrium_constant_wgs_expression(temperature_k) -> Any:
    return Exp(Constant(4400.0) / temperature_k - Constant(4.036))


def _partial_pressure_bar_expression(context: KineticsContext, species_id: str):
    species_idx = context.gas_index(species_id)
    return _pressure_bar_expression(context.model.P(context.idx_cell)) * context.model.y_gas(species_idx, context.idx_cell)


def _catalyst_mass_density_expression(context: KineticsContext):
    ni_idx = context.solid_index("Ni")
    return context.model.c_sol(ni_idx, context.idx_cell) / Constant(1.0 * mol / m**3) * Constant(NI_MW_KG_PER_MOL)


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


@register_kinetics_hook("numaguchi_smr_an")
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


@register_kinetics_hook("numaguchi_wgs_an")
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


__all__ = [
    "NUMAGUCHI_ACTIVATION_ENERGIES_J_PER_MOL",
    "NUMAGUCHI_RATE_COEFFICIENTS",
    "catalyst_mass_density_value",
    "equilibrium_constant_smr_value",
    "equilibrium_constant_wgs_value",
    "safe_steam_partial_pressure_bar_value",
    "partial_pressure_value",
    "partial_pressure_bar_value",
    "pressure_bar_value",
    "rate_constant_value",
    "smr_rate_value",
    "wgs_rate_value",
]
