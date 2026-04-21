from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from daetools.pyDAE import Constant, Exp, Max, Min, Abs

from pyUnits import K, Pa, m, mol, s

from . import KineticsContext, register_kinetics_hook


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
DENOMINATOR_FLOOR = 1.0e-16
POS_EPS = 1.0e-10

ONE_THIRD = 1.0 / 3.0
TWO_THIRDS = 2.0 / 3.0

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
    "H2": 0.60,
    "CO": 0.65,
    "O2": 0.90,
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
KXE_J_PER_MOL = {
    "H2": 0.0,
    "CO": 0.0,
    "O2": 0.0,
}
B = {
    "H2": 1.0,
    "CO": 1.0,
    "O2": 2.0,
}

# Two-term rational approximations a*x/(1+b*|x|) + c*x/(1+d*|x|), fit over [0, 1].
RATIONAL_POWER_COEFFICIENTS = {
    ONE_THIRD:  (1.36709714, 1.19782338, 38.81369122, 103.2164461),
    TWO_THIRDS: (1.10119253, 0.31786513, 3.171007230, 18.51100221),
    0.60:       (1.13885038, 0.42300595, 4.952297600, 24.11159535),
    0.65:       (1.10997490, 0.34224903, 3.545189580, 19.72954101),
    0.90:       (1.01682681, 0.06978578, 0.468259710, 8.536784090),
}


@dataclass(frozen=True)
class MedranoANTerms:
    temperature_k: Any
    total_gas_conc_molm3: Any
    gas_mole_fraction: Any
    ni_conc_molm3: Any
    nio_conc_molm3: Any
    total_solid_inventory_molm3: Any
    frac_reduced: Any
    frac_oxidised: Any


@dataclass(frozen=True)
class MedranoANReactionState:
    conversion: Any
    unreacted_fraction: Any
    total_solid_inventory_molm3: Any


def _temperature_k_expression(temperature) -> Any:
    return temperature / Constant(1.0 * K)


def _pressure_pa_expression(pressure) -> Any:
    return pressure / Constant(1.0 * Pa)


def _concentration_expression(conc) -> Any:
    return conc / Constant(1.0 * mol / m**3)


def _available_value(x: float) -> float:
    return max(0.0, x)


def _available_expr(x):
    return Max(x, Constant(0.0))


def _bounded_fraction_value(x: float) -> float:
    return min(1.0, max(0.0, x))


def _bounded_fraction_expr(x):
    return Min(Constant(1.0), Max(x, Constant(0.0)))


def _medrano_an_terms(context: KineticsContext, gas_species_id: str) -> MedranoANTerms:
    nickel_idx = context.solid_index("Ni")
    nickel_oxide_idx = context.solid_index("NiO")
    gas_idx = context.gas_index(gas_species_id)
    temperature_k = _temperature_k_expression(context.model.T(context.idx_cell))
    pressure_pa = _pressure_pa_expression(context.model.P(context.idx_cell))
    c_ni = _available_expr(_concentration_expression(context.model.c_sol(nickel_idx, context.idx_cell)))
    c_nio = _available_expr(_concentration_expression(context.model.c_sol(nickel_oxide_idx, context.idx_cell)))
    c_solid_total = c_ni + c_nio
    c_solid_denominator = c_solid_total + Constant(POS_EPS)

    return MedranoANTerms(
        temperature_k=temperature_k,
        total_gas_conc_molm3=pressure_pa / (Constant(GAS_CONSTANT_J_PER_MOL_K) * temperature_k),
        gas_mole_fraction=context.model.y_gas(gas_idx, context.idx_cell),
        ni_conc_molm3=c_ni,
        nio_conc_molm3=c_nio,
        total_solid_inventory_molm3=c_solid_total,
        frac_reduced=c_ni / c_solid_denominator,
        frac_oxidised=c_nio / c_solid_denominator,
    )


def medrano_an_reaction_state_value(
    comp_key: str,
    *,
    ni_concentration_molm3: float,
    nio_concentration_molm3: float,
) -> MedranoANReactionState:
    ni_available = _available_value(ni_concentration_molm3)
    nio_available = _available_value(nio_concentration_molm3)
    total_solid_inventory = ni_available + nio_available
    denominator = total_solid_inventory + POS_EPS
    frac_reduced = ni_available / denominator
    frac_oxidised = nio_available / denominator

    if comp_key == "O2":
        return MedranoANReactionState(
            conversion=frac_oxidised,
            unreacted_fraction=frac_reduced,
            total_solid_inventory_molm3=total_solid_inventory,
        )
    if comp_key in {"H2", "CO"}:
        return MedranoANReactionState(
            conversion=frac_reduced,
            unreacted_fraction=frac_oxidised,
            total_solid_inventory_molm3=total_solid_inventory,
        )
    raise KeyError(f"Unsupported Medrano AN component key: {comp_key}")


def _medrano_an_reaction_state_expr(comp_key: str, terms: MedranoANTerms) -> MedranoANReactionState:
    if comp_key == "O2":
        return MedranoANReactionState(
            conversion=terms.frac_oxidised,
            unreacted_fraction=terms.frac_reduced,
            total_solid_inventory_molm3=terms.total_solid_inventory_molm3,
        )
    if comp_key in {"H2", "CO"}:
        return MedranoANReactionState(
            conversion=terms.frac_reduced,
            unreacted_fraction=terms.frac_oxidised,
            total_solid_inventory_molm3=terms.total_solid_inventory_molm3,
        )
    raise KeyError(f"Unsupported Medrano AN component key: {comp_key}")


def rational_power_value(power: float, x: float) -> float:
    a, b, c, d = RATIONAL_POWER_COEFFICIENTS[power]
    raw_value = a * x / (1.0 + b * abs(x)) + c * x / (1.0 + d * abs(x))
    raw_value_at_one = a / (1.0 + b) + c / (1.0 + d)
    return raw_value / raw_value_at_one


def _rational_power_expr(power: float, x):
    a, b, c, d = RATIONAL_POWER_COEFFICIENTS[power]
    raw_value = (
        Constant(a) * x / (Constant(1.0) + Constant(b) * Abs(x))
        + Constant(c) * x / (Constant(1.0) + Constant(d) * Abs(x))
    )
    raw_value_at_one = a / (1.0 + b) + c / (1.0 + d)
    return raw_value / Constant(raw_value_at_one)


def gas_concentration_power_value(
    *,
    total_gas_concentration_molm3: float,
    gas_mole_fraction: float,
    power: float,
) -> float:
    return total_gas_concentration_molm3**power * rational_power_value(
        power,
        _available_value(gas_mole_fraction),
    )


def _gas_concentration_power_expr(*, total_gas_concentration_molm3, gas_mole_fraction, power: float):
    return total_gas_concentration_molm3**power * _rational_power_expr(
        power,
        _available_expr(gas_mole_fraction),
    )


def k_value(
    comp_key: str,
    *,
    temperature_k: float,
) -> float:
    return K0_M_PER_S[comp_key] * math.exp(
        -ACTIVATION_ENERGY_J_PER_MOL[comp_key] / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
    )


def k_expr(
    comp_key: str,
    *,
    temperature_k,
) -> Any:
    return Constant(K0_M_PER_S[comp_key]) * Exp(
        -Constant(ACTIVATION_ENERGY_J_PER_MOL[comp_key] / GAS_CONSTANT_J_PER_MOL_K) / temperature_k
    )


def D_value(
    comp_key: str,
    *,
    temperature_k: float,
    conversion: float,
) -> float:
    conversion_factor = KX[comp_key] * math.exp(
        -KXE_J_PER_MOL[comp_key] / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
    )
    return D0_M2_PER_S[comp_key] * math.exp(
        -ED_J_PER_MOL[comp_key] / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
    ) * math.exp(-conversion_factor * conversion)


def D_expr(
    comp_key: str,
    *,
    temperature_k,
    conversion,
) -> Any:
    diffusivity = Constant(D0_M2_PER_S[comp_key])
    activation_energy = ED_J_PER_MOL[comp_key]
    if activation_energy != 0.0:
        diffusivity = diffusivity * Exp(-Constant(activation_energy / GAS_CONSTANT_J_PER_MOL_K) / temperature_k)

    kx = KX[comp_key]
    if kx != 0.0:
        conversion_factor = Constant(kx)
        kxe = KXE_J_PER_MOL[comp_key]
        if kxe != 0.0:
            conversion_factor = conversion_factor * Exp(-Constant(kxe / GAS_CONSTANT_J_PER_MOL_K) / temperature_k)
        diffusivity = diffusivity * Exp(-conversion_factor * conversion)

    return diffusivity


def denominator_safe_value(denominator: float) -> float:
    return max(DENOMINATOR_FLOOR, denominator)


def _denominator_safe_expr(denominator):
    return Max(denominator, Constant(DENOMINATOR_FLOOR))


def medrano_an_conversion_rate_value(
    comp_key: str,
    *,
    temperature_k: float,
    total_gas_concentration_molm3: float,
    gas_mole_fraction: float,
    conversion: float,
    unreacted_fraction: float,
) -> float:
    order = REACTION_ORDER[comp_key]
    k_reaction = k_value(comp_key, temperature_k=temperature_k)
    conversion_bounded = _bounded_fraction_value(conversion)
    unreacted_available = _bounded_fraction_value(unreacted_fraction)
    diffusivity = D_value(comp_key, temperature_k=temperature_k, conversion=conversion_bounded)
    c_power_kinetic = gas_concentration_power_value(
        total_gas_concentration_molm3=total_gas_concentration_molm3,
        gas_mole_fraction=gas_mole_fraction,
        power=order,
    )
    c_power_diffusive = gas_concentration_power_value(
        total_gas_concentration_molm3=total_gas_concentration_molm3,
        gas_mole_fraction=gas_mole_fraction,
        power=order,
    )
    f_one_third = rational_power_value(ONE_THIRD, unreacted_available)
    f_two_thirds = rational_power_value(TWO_THIRDS, unreacted_available)
    numerator = (
        3.0
        * B[comp_key]
        * f_two_thirds
        * k_reaction
        * c_power_kinetic
        * diffusivity
        * c_power_diffusive
        / (R0_M[comp_key] * CS_MOL_PER_M3[comp_key])
    )
    denominator = (
        diffusivity * c_power_diffusive
        + R0_M[comp_key] * k_reaction * c_power_kinetic * (f_one_third - f_two_thirds)
    )
    return numerator / denominator_safe_value(denominator)


def medrano_an_reaction_rate_value(
    comp_key: str,
    *,
    temperature_k: float,
    total_gas_concentration_molm3: float,
    gas_mole_fraction: float,
    conversion: float,
    unreacted_fraction: float,
    total_solid_inventory_molm3: float,
) -> float:
    return total_solid_inventory_molm3 * medrano_an_conversion_rate_value(
        comp_key,
        temperature_k=temperature_k,
        total_gas_concentration_molm3=total_gas_concentration_molm3,
        gas_mole_fraction=gas_mole_fraction,
        conversion=conversion,
        unreacted_fraction=unreacted_fraction,
    )


def _medrano_an_conversion_rate_expr(
    comp_key: str,
    *,
    temperature_k,
    total_gas_concentration_molm3,
    gas_mole_fraction,
    conversion,
    unreacted_fraction,
):
    order = REACTION_ORDER[comp_key]
    k_reaction = k_expr(comp_key, temperature_k=temperature_k)
    conversion_bounded = _bounded_fraction_expr(conversion)
    unreacted_available = _bounded_fraction_expr(unreacted_fraction)
    diffusivity = D_expr(comp_key, temperature_k=temperature_k, conversion=conversion_bounded)
    c_power_kinetic = _gas_concentration_power_expr(
        total_gas_concentration_molm3=total_gas_concentration_molm3,
        gas_mole_fraction=gas_mole_fraction,
        power=order,
    )
    c_power_diffusive = _gas_concentration_power_expr(
        total_gas_concentration_molm3=total_gas_concentration_molm3,
        gas_mole_fraction=gas_mole_fraction,
        power=order,
    )
    f_one_third = _rational_power_expr(ONE_THIRD, unreacted_available)
    f_two_thirds = _rational_power_expr(TWO_THIRDS, unreacted_available)
    numerator = (
        Constant(3.0 * B[comp_key])
        * f_two_thirds
        * k_reaction
        * c_power_kinetic
        * diffusivity
        * c_power_diffusive
        / Constant(R0_M[comp_key] * CS_MOL_PER_M3[comp_key])
    )
    denominator = (
        diffusivity * c_power_diffusive
        + Constant(R0_M[comp_key]) * k_reaction * c_power_kinetic * (f_one_third - f_two_thirds)
    )
    return numerator / _denominator_safe_expr(denominator)


def _medrano_an_reaction_rate_expr(
    comp_key: str,
    *,
    temperature_k,
    total_gas_concentration_molm3,
    gas_mole_fraction,
    conversion,
    unreacted_fraction,
    total_solid_inventory_molm3,
):
    return total_solid_inventory_molm3 * _medrano_an_conversion_rate_expr(
        comp_key,
        temperature_k=temperature_k,
        total_gas_concentration_molm3=total_gas_concentration_molm3,
        gas_mole_fraction=gas_mole_fraction,
        conversion=conversion,
        unreacted_fraction=unreacted_fraction,
    )


@register_kinetics_hook("medrano_an_reduction_h2")
def medrano_an_reduction_h2(context: KineticsContext):
    terms = _medrano_an_terms(context, "H2")
    state = _medrano_an_reaction_state_expr("H2", terms)
    rate_expression = _medrano_an_reaction_rate_expr(
        "H2",
        temperature_k=terms.temperature_k,
        total_gas_concentration_molm3=terms.total_gas_conc_molm3,
        gas_mole_fraction=terms.gas_mole_fraction,
        conversion=state.conversion,
        unreacted_fraction=state.unreacted_fraction,
        total_solid_inventory_molm3=state.total_solid_inventory_molm3,
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


@register_kinetics_hook("medrano_an_reduction_co")
def medrano_an_reduction_co(context: KineticsContext):
    terms = _medrano_an_terms(context, "CO")
    state = _medrano_an_reaction_state_expr("CO", terms)
    rate_expression = _medrano_an_reaction_rate_expr(
        "CO",
        temperature_k=terms.temperature_k,
        total_gas_concentration_molm3=terms.total_gas_conc_molm3,
        gas_mole_fraction=terms.gas_mole_fraction,
        conversion=state.conversion,
        unreacted_fraction=state.unreacted_fraction,
        total_solid_inventory_molm3=state.total_solid_inventory_molm3,
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


@register_kinetics_hook("medrano_an_oxidation_o2")
def medrano_an_oxidation_o2(context: KineticsContext):
    terms = _medrano_an_terms(context, "O2")
    state = _medrano_an_reaction_state_expr("O2", terms)
    rate_expression = _medrano_an_reaction_rate_expr(
        "O2",
        temperature_k=terms.temperature_k,
        total_gas_concentration_molm3=terms.total_gas_conc_molm3,
        gas_mole_fraction=terms.gas_mole_fraction,
        conversion=state.conversion,
        unreacted_fraction=state.unreacted_fraction,
        total_solid_inventory_molm3=state.total_solid_inventory_molm3,
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


__all__ = [
    "ACTIVATION_ENERGY_J_PER_MOL",
    "B",
    "CS_MOL_PER_M3",
    "D0_M2_PER_S",
    "DENOMINATOR_FLOOR",
    "ED_J_PER_MOL",
    "GAS_CONSTANT_J_PER_MOL_K",
    "K0_M_PER_S",
    "KX",
    "KXE_J_PER_MOL",
    "MedranoANReactionState",
    "MedranoANTerms",
    "ONE_THIRD",
    "R0_M",
    "RATIONAL_POWER_COEFFICIENTS",
    "REACTION_ORDER",
    "TWO_THIRDS",
    "D_value",
    "denominator_safe_value",
    "gas_concentration_power_value",
    "k_value",
    "medrano_an_reaction_state_value",
    "medrano_an_conversion_rate_value",
    "medrano_an_reaction_rate_value",
    "rational_power_value",
]
