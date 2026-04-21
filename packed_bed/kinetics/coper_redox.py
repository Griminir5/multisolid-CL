from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from daetools.pyDAE import Constant, Exp, Sqrt

from pyUnits import K, Pa, m, mol, s

from . import KineticsContext, register_kinetics_hook


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
PRESSURE_PA_PER_BAR = 1.0e5
MIN_PARTIAL_PRESSURE_BAR = 1.0e-12
GAS_AVAILABILITY_PRESSURE_BAR = 1.0e-6


# CuO/Cu2O reduction kinetics, Table 4, pseudo-homogeneous zero-order H2 model.
CU_REDUCTION_COEFFICIENTS = {
    "red1": 1.54e-1,  # s^-1
    "red2": 1.53e-2,  # s^-1
}
CU_REDUCTION_EA_J_PER_MOL = {
    "red1": 0.15e3,
    "red2": 1.79e3,
}

# Spinel reduction kinetics, Table 8
CU_AL2O3_SPINEL_REDUCTION_COEFFICIENTS = {
    "sp1": 1.93e10,   # s^-1
    "sp2": 5.4e-1,    # s^-1
    "sp3": 4.87e-3,   # s^-1
}
CU_AL2O3_SPINEL_REDUCTION_EA_J_PER_MOL = {
    "sp1": 241.75e3,
    "sp2": 0.37e3,
    "sp3": 8.85e3,
}

# Oxidation kinetics, Table 9
CU_AL2O3_OXIDATION_COEFFICIENTS = {
    "ox1": 8.54e-1,   # s^-1 bar^-1/2
    "ox2": 1.27e-6,   # m^3 mol^-1 s^-1
    "ox3": 1.98e-5,   # m^3 mol^-1 s^-1 bar^-1/2
}
CU_AL2O3_OXIDATION_EA_J_PER_MOL = {
    "ox1": 0.83e3,
    "ox2": 1.18e3,
    "ox3": 0.71e3,
}


@dataclass(frozen=True)
class CuAl2O3Terms:
    temperature_k: Any
    p_h2_bar: Any
    p_o2_bar: Any
    c_cuo_mol_per_m3: Any
    c_cu2o_mol_per_m3: Any
    c_cu_mol_per_m3: Any
    c_cual2o4_mol_per_m3: Any
    c_cualo2_mol_per_m3: Any
    c_al2o3_mol_per_m3: Any


def pressure_bar_value(pressure_pa: float) -> float:
    return pressure_pa / PRESSURE_PA_PER_BAR


def partial_pressure_value(total_pressure_pa: float, mole_fraction: float) -> float:
    return total_pressure_pa * mole_fraction


def partial_pressure_bar_value(total_pressure_pa: float, mole_fraction: float) -> float:
    return pressure_bar_value(partial_pressure_value(total_pressure_pa, mole_fraction))


def rate_constant_value(
    coefficient: float,
    activation_energy_j_per_mol: float,
    *,
    temperature_k: float,
) -> float:
    return coefficient * math.exp(
        -activation_energy_j_per_mol / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
    )


def cu_al2o3_spinel_reduction_rate_constant_value(rate_key: str, *, temperature_k: float) -> float:
    return rate_constant_value(
        CU_AL2O3_SPINEL_REDUCTION_COEFFICIENTS[rate_key],
        CU_AL2O3_SPINEL_REDUCTION_EA_J_PER_MOL[rate_key],
        temperature_k=temperature_k,
    )


def cu_reduction_rate_constant_value(rate_key: str, *, temperature_k: float) -> float:
    return rate_constant_value(
        CU_REDUCTION_COEFFICIENTS[rate_key],
        CU_REDUCTION_EA_J_PER_MOL[rate_key],
        temperature_k=temperature_k,
    )


def cu_al2o3_oxidation_rate_constant_value(rate_key: str, *, temperature_k: float) -> float:
    return rate_constant_value(
        CU_AL2O3_OXIDATION_COEFFICIENTS[rate_key],
        CU_AL2O3_OXIDATION_EA_J_PER_MOL[rate_key],
        temperature_k=temperature_k,
    )


def gas_availability_value(partial_pressure_bar: float) -> float:
    pressure = max(partial_pressure_bar, 0.0)
    return pressure / (pressure + GAS_AVAILABILITY_PRESSURE_BAR)


def cuo_h2_reduction_rate_value(
    *,
    temperature_k: float,
    c_cuo_mol_per_m3: float,
    p_h2_pa: float,
) -> float:
    return (
        cu_reduction_rate_constant_value("red1", temperature_k=temperature_k)
        * c_cuo_mol_per_m3
        * gas_availability_value(pressure_bar_value(p_h2_pa))
    )


def cu2o_h2_reduction_rate_value(
    *,
    temperature_k: float,
    c_cu2o_mol_per_m3: float,
    p_h2_pa: float,
) -> float:
    return (
        cu_reduction_rate_constant_value("red2", temperature_k=temperature_k)
        * c_cu2o_mol_per_m3
        * gas_availability_value(pressure_bar_value(p_h2_pa))
    )


def cu_al2o3_sp1_rate_value(
    *,
    temperature_k: float,
    c_cual2o4_mol_per_m3: float,
    p_h2_pa: float,
) -> float:
    return (
        cu_al2o3_spinel_reduction_rate_constant_value("sp1", temperature_k=temperature_k)
        * c_cual2o4_mol_per_m3
        * gas_availability_value(pressure_bar_value(p_h2_pa))
    )


def cu_al2o3_sp2_rate_value(
    *,
    temperature_k: float,
    c_cual2o4_mol_per_m3: float,
    p_h2_pa: float,
) -> float:
    return (
        cu_al2o3_spinel_reduction_rate_constant_value("sp2", temperature_k=temperature_k)
        * c_cual2o4_mol_per_m3
        * gas_availability_value(pressure_bar_value(p_h2_pa))
    )


def cu_al2o3_sp3_rate_value(
    *,
    temperature_k: float,
    c_cualo2_mol_per_m3: float,
    p_h2_pa: float,
) -> float:
    return (
        cu_al2o3_spinel_reduction_rate_constant_value("sp3", temperature_k=temperature_k)
        * c_cualo2_mol_per_m3
        * gas_availability_value(pressure_bar_value(p_h2_pa))
    )


def cu_al2o3_ox1_rate_value(
    *,
    temperature_k: float,
    c_cu_mol_per_m3: float,
    p_o2_pa: float,
) -> float:
    p_o2_bar = pressure_bar_value(p_o2_pa)
    return (
        cu_al2o3_oxidation_rate_constant_value("ox1", temperature_k=temperature_k)
        * c_cu_mol_per_m3
        * math.sqrt(max(p_o2_bar, 0.0))
    )


def cu_al2o3_ox2_rate_value(
    *,
    temperature_k: float,
    c_cuo_mol_per_m3: float,
    c_al2o3_mol_per_m3: float,
) -> float:
    return (
        cu_al2o3_oxidation_rate_constant_value("ox2", temperature_k=temperature_k)
        * c_cuo_mol_per_m3
        * c_al2o3_mol_per_m3
    )


def cu_al2o3_ox3_rate_value(
    *,
    temperature_k: float,
    c_cualo2_mol_per_m3: float,
    c_al2o3_mol_per_m3: float,
    p_o2_pa: float,
) -> float:
    p_o2_bar = pressure_bar_value(p_o2_pa)
    return (
        cu_al2o3_oxidation_rate_constant_value("ox3", temperature_k=temperature_k)
        * c_cualo2_mol_per_m3
        * c_al2o3_mol_per_m3
        * math.sqrt(max(p_o2_bar, 0.0))
    )


def _temperature_k_expression(temperature) -> Any:
    return temperature / Constant(1.0 * K)


def _pressure_bar_expression(pressure) -> Any:
    return pressure / Constant(PRESSURE_PA_PER_BAR * Pa)


def _solid_concentration_expression(context: KineticsContext, species_id: str):
    species_idx = context.solid_index(species_id)
    return context.model.c_sol(species_idx, context.idx_cell) / Constant(1.0 * mol / m**3)


def _partial_pressure_bar_expression(context: KineticsContext, species_id: str):
    species_idx = context.gas_index(species_id)
    return _pressure_bar_expression(context.model.P(context.idx_cell)) * context.model.y_gas(
        species_idx, context.idx_cell
    )


def _sqrt_bar_expression(bar_pressure_expression):
    positive_pressure = Constant(0.5) * (
        bar_pressure_expression
        + Sqrt(bar_pressure_expression**2 + Constant((2.0 * MIN_PARTIAL_PRESSURE_BAR) ** 2))
    )
    return Sqrt(positive_pressure)


def _gas_availability_expression(partial_pressure_bar):
    positive_pressure = Constant(0.5) * (
        partial_pressure_bar + Sqrt(partial_pressure_bar**2 + Constant((2.0 * MIN_PARTIAL_PRESSURE_BAR) ** 2))
    )
    return positive_pressure / (positive_pressure + Constant(GAS_AVAILABILITY_PRESSURE_BAR))


def _rate_constant_expression(
    coefficient: float,
    activation_energy_j_per_mol: float,
    temperature_k,
) -> Any:
    return Constant(coefficient) * Exp(
        -Constant(activation_energy_j_per_mol / GAS_CONSTANT_J_PER_MOL_K) / temperature_k
    )


def _cu_al2o3_terms(context: KineticsContext) -> CuAl2O3Terms:
    return CuAl2O3Terms(
        temperature_k=_temperature_k_expression(context.model.T(context.idx_cell)),
        p_h2_bar=_partial_pressure_bar_expression(context, "H2"),
        p_o2_bar=_partial_pressure_bar_expression(context, "O2"),
        c_cuo_mol_per_m3=_solid_concentration_expression(context, "CuO"),
        c_cu2o_mol_per_m3=_solid_concentration_expression(context, "Cu2O"),
        c_cu_mol_per_m3=_solid_concentration_expression(context, "Cu"),
        c_cual2o4_mol_per_m3=_solid_concentration_expression(context, "CuAl2O4"),
        c_cualo2_mol_per_m3=_solid_concentration_expression(context, "CuAlO2"),
        c_al2o3_mol_per_m3=_solid_concentration_expression(context, "Al2O3"),
    )



@register_kinetics_hook("san_pio_cuo_h2_reduction_ph")
def san_pio_cuo_h2_reduction_ph(context: KineticsContext):
    terms = _cu_al2o3_terms(context)
    rate_expression = (
        _rate_constant_expression(
            CU_REDUCTION_COEFFICIENTS["red1"],
            CU_REDUCTION_EA_J_PER_MOL["red1"],
            terms.temperature_k,
        )
        * terms.c_cuo_mol_per_m3
        * _gas_availability_expression(terms.p_h2_bar)
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


@register_kinetics_hook("san_pio_cu2o_h2_reduction_ph")
def san_pio_cu2o_h2_reduction_ph(context: KineticsContext):
    terms = _cu_al2o3_terms(context)
    rate_expression = (
        _rate_constant_expression(
            CU_REDUCTION_COEFFICIENTS["red2"],
            CU_REDUCTION_EA_J_PER_MOL["red2"],
            terms.temperature_k,
        )
        * terms.c_cu2o_mol_per_m3
        * _gas_availability_expression(terms.p_h2_bar)
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


@register_kinetics_hook("san_pio_cu_al2o3_sp1_ph")
def san_pio_cu_al2o3_sp1_ph(context: KineticsContext):
    terms = _cu_al2o3_terms(context)
    rate_expression = (
        _rate_constant_expression(
            CU_AL2O3_SPINEL_REDUCTION_COEFFICIENTS["sp1"],
            CU_AL2O3_SPINEL_REDUCTION_EA_J_PER_MOL["sp1"],
            terms.temperature_k,
        )
        * terms.c_cual2o4_mol_per_m3
        * _gas_availability_expression(terms.p_h2_bar)
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


@register_kinetics_hook("san_pio_cu_al2o3_sp2_ph")
def san_pio_cu_al2o3_sp2_ph(context: KineticsContext):
    terms = _cu_al2o3_terms(context)
    rate_expression = (
        _rate_constant_expression(
            CU_AL2O3_SPINEL_REDUCTION_COEFFICIENTS["sp2"],
            CU_AL2O3_SPINEL_REDUCTION_EA_J_PER_MOL["sp2"],
            terms.temperature_k,
        )
        * terms.c_cual2o4_mol_per_m3
        * _gas_availability_expression(terms.p_h2_bar)
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


@register_kinetics_hook("san_pio_cu_al2o3_sp3_ph")
def san_pio_cu_al2o3_sp3_ph(context: KineticsContext):
    terms = _cu_al2o3_terms(context)
    rate_expression = (
        _rate_constant_expression(
            CU_AL2O3_SPINEL_REDUCTION_COEFFICIENTS["sp3"],
            CU_AL2O3_SPINEL_REDUCTION_EA_J_PER_MOL["sp3"],
            terms.temperature_k,
        )
        * terms.c_cualo2_mol_per_m3
        * _gas_availability_expression(terms.p_h2_bar)
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression



@register_kinetics_hook("san_pio_cu_al2o3_ox1_ph")
def san_pio_cu_al2o3_ox1_ph(context: KineticsContext):
    terms = _cu_al2o3_terms(context)
    rate_expression = (
        _rate_constant_expression(
            CU_AL2O3_OXIDATION_COEFFICIENTS["ox1"],
            CU_AL2O3_OXIDATION_EA_J_PER_MOL["ox1"],
            terms.temperature_k,
        )
        * terms.c_cu_mol_per_m3
        * _sqrt_bar_expression(terms.p_o2_bar)
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


@register_kinetics_hook("san_pio_cu_al2o3_ox2_ph")
def san_pio_cu_al2o3_ox2_ph(context: KineticsContext):
    terms = _cu_al2o3_terms(context)
    rate_expression = (
        _rate_constant_expression(
            CU_AL2O3_OXIDATION_COEFFICIENTS["ox2"],
            CU_AL2O3_OXIDATION_EA_J_PER_MOL["ox2"],
            terms.temperature_k,
        )
        * terms.c_cuo_mol_per_m3
        * terms.c_al2o3_mol_per_m3
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


@register_kinetics_hook("san_pio_cu_al2o3_ox3_ph")
def san_pio_cu_al2o3_ox3_ph(context: KineticsContext):
    terms = _cu_al2o3_terms(context)
    rate_expression = (
        _rate_constant_expression(
            CU_AL2O3_OXIDATION_COEFFICIENTS["ox3"],
            CU_AL2O3_OXIDATION_EA_J_PER_MOL["ox3"],
            terms.temperature_k,
        )
        * terms.c_cualo2_mol_per_m3
        * terms.c_al2o3_mol_per_m3
        * _sqrt_bar_expression(terms.p_o2_bar)
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


__all__ = [
    "CU_REDUCTION_COEFFICIENTS",
    "CU_REDUCTION_EA_J_PER_MOL",
    "CU_AL2O3_SPINEL_REDUCTION_COEFFICIENTS",
    "CU_AL2O3_SPINEL_REDUCTION_EA_J_PER_MOL",
    "CU_AL2O3_OXIDATION_COEFFICIENTS",
    "CU_AL2O3_OXIDATION_EA_J_PER_MOL",
    "partial_pressure_value",
    "partial_pressure_bar_value",
    "pressure_bar_value",
    "rate_constant_value",
    "gas_availability_value",
    "cu_reduction_rate_constant_value",
    "cu_al2o3_spinel_reduction_rate_constant_value",
    "cu_al2o3_oxidation_rate_constant_value",
    "cuo_h2_reduction_rate_value",
    "cu2o_h2_reduction_rate_value",
    "cu_al2o3_sp1_rate_value",
    "cu_al2o3_sp2_rate_value",
    "cu_al2o3_sp3_rate_value",
    "cu_al2o3_ox1_rate_value",
    "cu_al2o3_ox2_rate_value",
    "cu_al2o3_ox3_rate_value",
]
