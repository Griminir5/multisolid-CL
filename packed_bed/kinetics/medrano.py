from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from daetools.pyDAE import Constant, Exp, Max, Min, Sqrt

from pyUnits import K, m, mol, s

from . import KineticsContext, register_kinetics_hook


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324

POS_EPS = 1.0e-10
F_FLOOR = 1.0e-4
F_GATE = 1.0e-3
CG_FLOOR = 1.0e-8
CG_GATE = 1.0e-8

CS_MOL_PER_M3 = {
    "H2": 89960.0,
    "CO": 89960.0,
    "O2": 151200.0,
}
R0_M = {
    "H2": 3.13e-8,
    "CO": 3.13e-8,
    "O2": 5.8e-7,
}
K0_M_PER_S = {
    "H2": 9.0e-4,
    "CO": 3.5e-3,
    "O2": 1.2e-3,
}
ACTIVATION_ENERGY_J_PER_MOL = {
    "H2": 30000.0,
    "CO": 45000.0,
    "O2": 7000.0,
}
REACTION_ORDER = {
    "H2": 0.6,
    "CO": 0.65,
    "O2": 0.9,
}
D0_M2_PER_S = {
    "H2": 1.7e-3,
    "CO": 7.4e6,
    "O2": 1.0,
}
ED_J_PER_MOL = {
    "H2": 150000.0,
    "CO": 300000.0,
    "O2": 0.0,
}
KX = {
    "H2": 5.0,
    "CO": 15.0,
    "O2": 0.0,
}
B = {
    "H2": 1.0,
    "CO": 1.0,
    "O2": 2.0,
}

@dataclass(frozen=True)
class MedranoTerms:
    temperature_k: Any
    gas_conc_molm3: Any
    ni_conc_molm3: Any
    nio_conc_molm3: Any
    frac_reduced: Any
    frac_oxidised: Any

def _smooth_pos_value(x):
    return 0.5 * (x + math.sqrt(x * x + POS_EPS * POS_EPS))

def _smooth_pos_expr(x):
    return Constant(0.5) * (x + Sqrt(x * x + Constant(POS_EPS * POS_EPS)))

def _available_value(x):
    return max(0.0, x)

def _available_expr(x):
    return Max(x, Constant(0.0))

def _bounded_fraction_value(x):
    return min(1.0, max(0.0, x))

def _bounded_fraction_expr(x):
    return Min(Constant(1.0), Max(x, Constant(0.0)))

def _temperature_k_expression(temperature) -> Any:
    return temperature / Constant(1.0 * K)

def _concentration_expression(conc) -> Any:
    return conc / Constant(1.0 * mol / m**3)

def _medrano_terms(context: KineticsContext, gas_species_id: str) -> MedranoTerms:
    nickel_idx = context.solid_index("Ni")
    nickel_oxide_idx = context.solid_index("NiO")
    gas_idx = context.gas_index(gas_species_id)
    temperature_k = _temperature_k_expression(context.model.T(context.idx_cell))
    c_ni  = _available_expr(_concentration_expression(context.model.c_sol(nickel_idx, context.idx_cell)))
    c_nio = _available_expr(_concentration_expression(context.model.c_sol(nickel_oxide_idx, context.idx_cell)))
    c_gas  = _concentration_expression(context.model.c_gas(gas_idx, context.idx_cell))
    c_solid_total = c_ni + c_nio

    return MedranoTerms(
        temperature_k=temperature_k,
        gas_conc_molm3=c_gas,
        ni_conc_molm3=c_ni,
        nio_conc_molm3=c_nio,
        frac_reduced=c_ni / (c_solid_total + Constant(POS_EPS)),
        frac_oxidised=c_nio / (c_solid_total + Constant(POS_EPS)),
    )

def k_value(
    comp_key: str,
    *,
    temperature_k: float,
) -> float:
    preexp = K0_M_PER_S[comp_key]
    activation_energy = ACTIVATION_ENERGY_J_PER_MOL[comp_key]
    return preexp * math.exp(
        -activation_energy / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
        )


def k_expr(
    comp_key: str,
    *,
    temperature_k,
) -> Any:
    preexp = K0_M_PER_S[comp_key]
    activation_energy = ACTIVATION_ENERGY_J_PER_MOL[comp_key]
    return Constant(preexp) * Exp(-Constant(activation_energy / GAS_CONSTANT_J_PER_MOL_K) / temperature_k)


def D_value(
    comp_key: str,
    *,
    temperature_k: float,
    conv: float,
) -> float:
    preexp = D0_M2_PER_S[comp_key]
    activation_energy = ED_J_PER_MOL[comp_key]
    kx = KX[comp_key]
    return preexp * math.exp(
        -activation_energy / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
        ) * math.exp(
            -kx*conv
        )


def D_expr(
    comp_key: str,
    *,
    temperature_k,
    conv,
) -> Any:
    preexp = D0_M2_PER_S[comp_key]
    activation_energy = ED_J_PER_MOL[comp_key]
    kx = KX[comp_key]
    return Constant(preexp) * Exp(
        -Constant(activation_energy / GAS_CONSTANT_J_PER_MOL_K) / temperature_k) * Exp(-Constant(kx)*conv)


def safe_gas_concentration_value(concentration_molm3: float) -> float:
    return _available_value(concentration_molm3) + CG_FLOOR


def medrano_conversion_rate_value(
    comp_key: str,
    *,
    temperature_k: float,
    gas_concentration_molm3: float,
    conversion: float,
    reactant_fraction: float,
) -> float:
    k = k_value(comp_key, temperature_k=temperature_k)
    conversion_bounded = _bounded_fraction_value(conversion)
    diffusivity = D_value(comp_key, temperature_k=temperature_k, conv=conversion_bounded)
    gas_available = _available_value(gas_concentration_molm3)
    reactant_available = _bounded_fraction_value(reactant_fraction)
    gas_concentration_safe = gas_available + CG_FLOOR
    f_power = F_FLOOR + (1.0 - F_FLOOR) * reactant_available
    gas_gate = gas_available / (gas_available + CG_GATE) if gas_available > 0.0 else 0.0
    solid_gate = reactant_available / (reactant_available + F_GATE) if reactant_available > 0.0 else 0.0
    numerator = (
        3.0
        * B[comp_key]
        * gas_concentration_safe ** REACTION_ORDER[comp_key]
        / (R0_M[comp_key] * CS_MOL_PER_M3[comp_key])
    )
    denominator = (
        (1.0 / k) * f_power ** (-2.0 / 3.0)
        + (R0_M[comp_key] / diffusivity) * (f_power ** (-1.0 / 3.0) - 1.0)
    )
    return gas_gate * solid_gate * numerator / denominator


def medrano_reaction_rate_value(
    comp_key: str,
    *,
    temperature_k: float,
    gas_concentration_molm3: float,
    conversion: float,
    reactant_fraction: float,
    active_inventory_molm3: float,
) -> float:
    return active_inventory_molm3 * medrano_conversion_rate_value(
        comp_key,
        temperature_k=temperature_k,
        gas_concentration_molm3=gas_concentration_molm3,
        conversion=conversion,
        reactant_fraction=reactant_fraction,
    )


def _safe_gas_concentration_expr(concentration_molm3):
    return _available_expr(concentration_molm3) + Constant(CG_FLOOR)


def _medrano_conversion_rate_expr(
    comp_key: str,
    *,
    temperature_k,
    gas_concentration_molm3,
    conversion,
    reactant_fraction,
):
    k = k_expr(comp_key, temperature_k=temperature_k)
    conversion_bounded = _bounded_fraction_expr(conversion)
    diffusivity = D_expr(comp_key, temperature_k=temperature_k, conv=conversion_bounded)
    gas_available = _available_expr(gas_concentration_molm3)
    reactant_available = _bounded_fraction_expr(reactant_fraction)
    gas_concentration_safe = gas_available + Constant(CG_FLOOR)
    f_power = Constant(F_FLOOR) + Constant(1.0 - F_FLOOR) * reactant_available
    gas_gate = gas_available / (gas_available + Constant(CG_GATE))
    solid_gate = reactant_available / (reactant_available + Constant(F_GATE))
    numerator = (
        Constant(3.0 * B[comp_key])
        * gas_concentration_safe ** REACTION_ORDER[comp_key]
        / Constant(R0_M[comp_key] * CS_MOL_PER_M3[comp_key])
    )
    denominator = (
        Constant(1.0) / k * f_power ** (-2.0 / 3.0)
        + Constant(R0_M[comp_key]) / diffusivity * (f_power ** (-1.0 / 3.0) - Constant(1.0))
    )
    return gas_gate * solid_gate * numerator / denominator


def _medrano_reaction_rate_expr(
    comp_key: str,
    *,
    temperature_k,
    gas_concentration_molm3,
    conversion,
    reactant_fraction,
    active_inventory_molm3,
):
    return active_inventory_molm3 * _medrano_conversion_rate_expr(
        comp_key,
        temperature_k=temperature_k,
        gas_concentration_molm3=gas_concentration_molm3,
        conversion=conversion,
        reactant_fraction=reactant_fraction,
    )


@register_kinetics_hook("medrano_reduction_h2")
def medrano_reduction_h2(context: KineticsContext):
    terms = _medrano_terms(context, "H2")
    rate_expression = _medrano_reaction_rate_expr(
        "H2",
        temperature_k=terms.temperature_k,
        gas_concentration_molm3=terms.gas_conc_molm3,
        conversion=terms.frac_reduced,
        reactant_fraction=terms.frac_oxidised,
        active_inventory_molm3=terms.ni_conc_molm3 + terms.nio_conc_molm3,
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


@register_kinetics_hook("medrano_reduction_co")
def medrano_reduction_co(context: KineticsContext):
    terms = _medrano_terms(context, "CO")
    rate_expression = _medrano_reaction_rate_expr(
        "CO",
        temperature_k=terms.temperature_k,
        gas_concentration_molm3=terms.gas_conc_molm3,
        conversion=terms.frac_reduced,
        reactant_fraction=terms.frac_oxidised,
        active_inventory_molm3=terms.ni_conc_molm3 + terms.nio_conc_molm3,
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


@register_kinetics_hook("medrano_oxidation_o2")
def medrano_oxidation_o2(context: KineticsContext):
    terms = _medrano_terms(context, "O2")
    rate_expression = _medrano_reaction_rate_expr(
        "O2",
        temperature_k=terms.temperature_k,
        gas_concentration_molm3=terms.gas_conc_molm3,
        conversion=terms.frac_oxidised,
        reactant_fraction=terms.frac_reduced,
        active_inventory_molm3=terms.ni_conc_molm3 + terms.nio_conc_molm3,
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


__all__ = [
    "ACTIVATION_ENERGY_J_PER_MOL",
    "B",
    "CG_FLOOR",
    "CG_GATE",
    "CS_MOL_PER_M3",
    "D0_M2_PER_S",
    "ED_J_PER_MOL",
    "F_FLOOR",
    "F_GATE",
    "GAS_CONSTANT_J_PER_MOL_K",
    "K0_M_PER_S",
    "KX",
    "MedranoTerms",
    "POS_EPS",
    "R0_M",
    "REACTION_ORDER",
    "D_value",
    "k_value",
    "medrano_conversion_rate_value",
    "medrano_reaction_rate_value",
    "safe_gas_concentration_value",
]
