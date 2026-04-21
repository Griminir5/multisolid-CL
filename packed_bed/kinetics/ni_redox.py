from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from daetools.pyDAE import Constant, Exp, Sqrt

from pyUnits import K, m, mol, s

from . import KineticsContext, register_kinetics_hook


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
PRESSURE_PA_PER_BAR = 1.0e5
MIN_ONE_MINUS_CONVERSION = 1.0e-12
MIN_GAS_CONCENTRATION_MOL_PER_M3 = 1.0e-12


NI_REDOX_RATE_COEFFICIENTS = {
    "h2_reduction": 9.00e-4,
    "co_reduction": 3.50e-3,
    "o2_oxidation": 1.20e-3,
}

NI_REDOX_ACTIVATION_ENERGIES_J_PER_MOL = {
    "h2_reduction": 30.0e3,
    "co_reduction": 45.0e3,
    "o2_oxidation": 7.0e3,
}

NI_REDOX_DIFFUSIVITY_COEFFICIENTS = {
    "h2_reduction": 1.70e-3,
    "co_reduction": 7.40e6,
    "o2_oxidation": 1.0,
}

NI_REDOX_DIFFUSION_ACTIVATION_ENERGIES_J_PER_MOL = {
    "h2_reduction": 150.0e3,
    "co_reduction": 300.0e3,
    "o2_oxidation": 0.0,
}

NI_REDOX_REACTION_ORDERS = {
    "h2_reduction": 0.6,
    "co_reduction": 0.65,
    "o2_oxidation": 0.9,
}

NI_REDOX_GRAIN_RADII_M = {
    "h2_reduction": 3.13e-8,
    "co_reduction": 3.13e-8,
    "o2_oxidation": 5.8e-7,
}

NI_REDOX_SOLID_CONCENTRATIONS_MOL_PER_M3 = {
    "h2_reduction": 89960.0,
    "co_reduction": 89960.0,
    "o2_oxidation": 151200.0,
}

NI_REDOX_KX = {
    "h2_reduction": 5.0,
    "co_reduction": 15.0,
    "o2_oxidation": 0.0,
}

NI_REDOX_B = {
    "h2_reduction": 1.0,
    "co_reduction": 1.0,
    "o2_oxidation": 2.0,
}

@dataclass(frozen=True)
class NiRedoxTerms:
    temperature_k: Any
    c_h2_mol_per_m3: Any
    c_co_mol_per_m3: Any
    c_o2_mol_per_m3: Any
    c_nio_mol_per_m3: Any
    c_ni_mol_per_m3: Any
    c_active_ni_mol_per_m3: Any
    x_reduction: Any
    x_oxidation: Any


def partial_pressure_value(total_pressure_pa: float, mole_fraction: float) -> float:
    return total_pressure_pa * mole_fraction


def pressure_bar_value(pressure_pa: float) -> float:
    return pressure_pa / PRESSURE_PA_PER_BAR


def partial_pressure_bar_value(total_pressure_pa: float, mole_fraction: float) -> float:
    return pressure_bar_value(partial_pressure_value(total_pressure_pa, mole_fraction))


def active_ni_concentration_value(
    ni_concentration_mol_per_m3: float,
    nio_concentration_mol_per_m3: float,
) -> float:
    return ni_concentration_mol_per_m3 + nio_concentration_mol_per_m3


def rate_constant_value(
    rate_key: str,
    *,
    temperature_k: float,
) -> float:
    coefficient = NI_REDOX_RATE_COEFFICIENTS[rate_key]
    activation_energy = NI_REDOX_ACTIVATION_ENERGIES_J_PER_MOL[rate_key]
    return coefficient * math.exp(-activation_energy / (GAS_CONSTANT_J_PER_MOL_K * temperature_k))


def diffusivity_value(
    rate_key: str,
    *,
    temperature_k: float,
    conversion: float,
) -> float:
    coefficient = NI_REDOX_DIFFUSIVITY_COEFFICIENTS[rate_key]
    activation_energy = NI_REDOX_DIFFUSION_ACTIVATION_ENERGIES_J_PER_MOL[rate_key]
    kx = NI_REDOX_KX[rate_key]
    return (
        coefficient
        * math.exp(-activation_energy / (GAS_CONSTANT_J_PER_MOL_K * temperature_k))
        * math.exp(-kx * conversion)
    )


def conversion_rate_value(
    rate_key: str,
    *,
    temperature_k: float,
    gas_concentration_mol_per_m3: float,
    conversion: float,
) -> float:
    conversion_safe = min(max(conversion, 0.0), 1.0 - MIN_ONE_MINUS_CONVERSION)
    one_minus_x = 1.0 - conversion_safe
    c_s = NI_REDOX_SOLID_CONCENTRATIONS_MOL_PER_M3[rate_key]
    r0 = NI_REDOX_GRAIN_RADII_M[rate_key]
    n = NI_REDOX_REACTION_ORDERS[rate_key]
    b = NI_REDOX_B[rate_key]

    k_value = rate_constant_value(rate_key, temperature_k=temperature_k)
    d_value = diffusivity_value(rate_key, temperature_k=temperature_k, conversion=conversion_safe)

    denominator = (
        (1.0 / k_value) * one_minus_x ** (-2.0 / 3.0) + (r0 / d_value) * (one_minus_x ** (-1.0 / 3.0) - 1.0)
    )

    gas_concentration_safe = max(gas_concentration_mol_per_m3, MIN_GAS_CONCENTRATION_MOL_PER_M3)
    return 3.0 * gas_concentration_safe**n / (b * r0 * c_s * denominator)


def h2_reduction_rate_value(
    *,
    temperature_k: float,
    c_h2_mol_per_m3: float,
    c_active_ni_mol_per_m3: float,
    x_reduction: float,
) -> float:
    dxdt = conversion_rate_value(
        "h2_reduction",
        temperature_k=temperature_k,
        gas_concentration_mol_per_m3=c_h2_mol_per_m3,
        conversion=x_reduction,
    )
    return c_active_ni_mol_per_m3 * dxdt


def co_reduction_rate_value(
    *,
    temperature_k: float,
    c_co_mol_per_m3: float,
    c_active_ni_mol_per_m3: float,
    x_reduction: float,
) -> float:
    dxdt = conversion_rate_value(
        "co_reduction",
        temperature_k=temperature_k,
        gas_concentration_mol_per_m3=c_co_mol_per_m3,
        conversion=x_reduction,
    )
    return c_active_ni_mol_per_m3 * dxdt


def o2_oxidation_rate_value(
    *,
    temperature_k: float,
    c_o2_mol_per_m3: float,
    c_active_ni_mol_per_m3: float,
    x_oxidation: float,
) -> float:
    dxdt = conversion_rate_value(
        "o2_oxidation",
        temperature_k=temperature_k,
        gas_concentration_mol_per_m3=c_o2_mol_per_m3,
        conversion=x_oxidation,
    )
    return c_active_ni_mol_per_m3 * dxdt


def _temperature_k_expression(temperature) -> Any:
    return temperature / Constant(1.0 * K)


def _rate_constant_expression(rate_key: str, temperature_k) -> Any:
    coefficient = NI_REDOX_RATE_COEFFICIENTS[rate_key]
    activation_energy = NI_REDOX_ACTIVATION_ENERGIES_J_PER_MOL[rate_key]
    return Constant(coefficient) * Exp(-Constant(activation_energy / GAS_CONSTANT_J_PER_MOL_K) / temperature_k)


def _diffusivity_expression(rate_key: str, temperature_k, conversion) -> Any:
    coefficient = NI_REDOX_DIFFUSIVITY_COEFFICIENTS[rate_key]
    activation_energy = NI_REDOX_DIFFUSION_ACTIVATION_ENERGIES_J_PER_MOL[rate_key]
    kx = NI_REDOX_KX[rate_key]
    return (
        Constant(coefficient)
        * Exp(-Constant(activation_energy / GAS_CONSTANT_J_PER_MOL_K) / temperature_k)
        * Exp(-Constant(kx) * conversion)
    )


def _gas_concentration_expression(context: KineticsContext, species_id: str):
    species_idx = context.gas_index(species_id)
    concentration = context.model.c_gas(species_idx, context.idx_cell) / Constant(1.0 * mol / m**3)
    return Constant(0.5) * (
        concentration + Sqrt(concentration**2 + Constant((2.0 * MIN_GAS_CONCENTRATION_MOL_PER_M3) ** 2))
    )


def _solid_concentration_expression(context: KineticsContext, species_id: str):
    species_idx = context.solid_index(species_id)
    return context.model.c_sol(species_idx, context.idx_cell) / Constant(1.0 * mol / m**3)


def _safe_division(numerator, denominator):
    return numerator / (denominator + Constant(1.0e-20))


def _ni_redox_terms(context: KineticsContext) -> NiRedoxTerms:
    temperature_k = _temperature_k_expression(context.model.T(context.idx_cell))
    c_nio = _solid_concentration_expression(context, "NiO")
    c_ni = _solid_concentration_expression(context, "Ni")
    c_active_ni = c_nio + c_ni

    return NiRedoxTerms(
        temperature_k=temperature_k,
        c_h2_mol_per_m3=_gas_concentration_expression(context, "H2"),
        c_co_mol_per_m3=_gas_concentration_expression(context, "CO"),
        c_o2_mol_per_m3=_gas_concentration_expression(context, "O2"),
        c_nio_mol_per_m3=c_nio,
        c_ni_mol_per_m3=c_ni,
        c_active_ni_mol_per_m3=c_active_ni,
        x_reduction=_safe_division(c_ni, c_active_ni),
        x_oxidation=_safe_division(c_nio, c_active_ni),
    )


def _conversion_rate_expression(rate_key: str, temperature_k, gas_concentration_mol_per_m3, conversion) -> Any:
    c_s = NI_REDOX_SOLID_CONCENTRATIONS_MOL_PER_M3[rate_key]
    r0 = NI_REDOX_GRAIN_RADII_M[rate_key]
    n = NI_REDOX_REACTION_ORDERS[rate_key]
    b = NI_REDOX_B[rate_key]

    one_minus_x = Constant(1.0) - conversion + Constant(MIN_ONE_MINUS_CONVERSION)

    denominator = (
        Constant(1.0)/ _rate_constant_expression(rate_key, temperature_k) * one_minus_x ** Constant(-2.0 / 3.0)
        + Constant(r0)/ _diffusivity_expression(rate_key, temperature_k, conversion) * (one_minus_x ** Constant(-1.0 / 3.0) - Constant(1.0))
    )

    return Constant(3.0 / (b * r0 * c_s)) * gas_concentration_mol_per_m3 ** Constant(n) / denominator


@register_kinetics_hook("ni_redox_reduction_h2")
def ni_redox_reduction_h2(context: KineticsContext):
    terms = _ni_redox_terms(context)
    dxdt_expression = _conversion_rate_expression(
        "h2_reduction",
        terms.temperature_k,
        terms.c_h2_mol_per_m3,
        terms.x_reduction,
    )
    return Constant(1.0 * mol / (m**3 * s)) * terms.c_active_ni_mol_per_m3 * dxdt_expression


@register_kinetics_hook("ni_redox_reduction_co")
def ni_redox_reduction_co(context: KineticsContext):
    terms = _ni_redox_terms(context)
    dxdt_expression = _conversion_rate_expression(
        "co_reduction",
        terms.temperature_k,
        terms.c_co_mol_per_m3,
        terms.x_reduction,
    )
    return Constant(1.0 * mol / (m**3 * s)) * terms.c_active_ni_mol_per_m3 * dxdt_expression


@register_kinetics_hook("ni_redox_oxidation_o2")
def ni_redox_oxidation_o2(context: KineticsContext):
    terms = _ni_redox_terms(context)
    dxdt_expression = _conversion_rate_expression(
        "o2_oxidation",
        terms.temperature_k,
        terms.c_o2_mol_per_m3,
        terms.x_oxidation,
    )
    return Constant(1.0 * mol / (m**3 * s)) * terms.c_active_ni_mol_per_m3 * dxdt_expression


__all__ = [
    "NI_REDOX_ACTIVATION_ENERGIES_J_PER_MOL",
    "NI_REDOX_B",
    "NI_REDOX_DIFFUSION_ACTIVATION_ENERGIES_J_PER_MOL",
    "NI_REDOX_DIFFUSIVITY_COEFFICIENTS",
    "NI_REDOX_GRAIN_RADII_M",
    "NI_REDOX_KX",
    "MIN_GAS_CONCENTRATION_MOL_PER_M3",
    "NI_REDOX_RATE_COEFFICIENTS",
    "NI_REDOX_REACTION_ORDERS",
    "NI_REDOX_SOLID_CONCENTRATIONS_MOL_PER_M3",
    "active_ni_concentration_value",
    "co_reduction_rate_value",
    "conversion_rate_value",
    "diffusivity_value",
    "h2_reduction_rate_value",
    "o2_oxidation_rate_value",
    "partial_pressure_bar_value",
    "partial_pressure_value",
    "pressure_bar_value",
    "rate_constant_value",
]
