from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from daetools.pyDAE import Constant, Exp

from pyUnits import K, Pa, m, mol, s

from ..properties import PROPERTY_REGISTRY
from . import KineticsContext, register_kinetics_hook


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
MIN_H2_MOLE_FRACTION = 0.02
LOW_H2_BLEND_BASE = 8.43458496001593e-11
NI_MW_KG_PER_MOL = PROPERTY_REGISTRY.get_record("Ni").mw

if NI_MW_KG_PER_MOL is None:
    raise ValueError("Nickel molecular weight must be available for Xu-Froment kinetics.")

XU_FROMENT_RATE_COEFFICIENTS = {
    "smr": 1.17e15,
    "wgs": 5.43e5,
    "overall": 2.81e14,
}
XU_FROMENT_ACTIVATION_ENERGIES_J_PER_MOL = {
    "smr": 240100.0,
    "wgs": 67130.0,
    "overall": 243900.0,
}
XU_FROMENT_ADSORPTION_COEFFICIENTS = {
    "CO": 8.23e-10,
    "H2": 6.12e-14,
    "CH4": 6.65e-9,
    "H2O": 1.77e5,
}
XU_FROMENT_ADSORPTION_ENERGIES_J_PER_MOL = {
    "CO": 70650.0,
    "H2": 82900.0,
    "CH4": 38280.0,
    "H2O": -88680.0,
}


@dataclass(frozen=True)
class XuFromentTerms:
    temperature_k: Any
    p_ch4_pa: Any
    p_co_pa: Any
    p_co2_pa: Any
    p_h2_pa: Any
    p_h2o_pa: Any
    p_inv_h2_pa_inv: Any
    denominator: Any
    catalyst_mass_density_kg_per_m3: Any


def partial_pressure_value(total_pressure_pa: float, mole_fraction: float) -> float:
    return total_pressure_pa * mole_fraction


def hydrogen_inverse_pressure_value(total_pressure_pa: float, hydrogen_mole_fraction: float) -> float:
    return 1.0 / (
        total_pressure_pa
        * (hydrogen_mole_fraction + MIN_H2_MOLE_FRACTION * (LOW_H2_BLEND_BASE**hydrogen_mole_fraction))
    )


def catalyst_mass_density_value(ni_concentration_mol_per_m3: float) -> float:
    return ni_concentration_mol_per_m3 * NI_MW_KG_PER_MOL


def rate_constant_value(
    rate_key: str,
    *,
    catalyst_mass_density_kg_per_m3: float,
    temperature_k: float,
) -> float:
    coefficient = XU_FROMENT_RATE_COEFFICIENTS[rate_key]
    activation_energy = XU_FROMENT_ACTIVATION_ENERGIES_J_PER_MOL[rate_key]
    return catalyst_mass_density_kg_per_m3 * coefficient * math.exp(
        -activation_energy / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
    )


def adsorption_constant_value(species_id: str, *, temperature_k: float) -> float:
    coefficient = XU_FROMENT_ADSORPTION_COEFFICIENTS[species_id]
    adsorption_energy = XU_FROMENT_ADSORPTION_ENERGIES_J_PER_MOL[species_id]
    return coefficient * math.exp(adsorption_energy / (GAS_CONSTANT_J_PER_MOL_K * temperature_k))


def _temperature_polynomial_value(temperature_k: float, coefficients: tuple[float, float, float, float, float]) -> float:
    temperature_kilo = temperature_k / 1000.0
    c0, c1, c2, c3, c4 = coefficients
    return c0 + c1 * temperature_kilo + c2 * temperature_kilo**2 + c3 * temperature_kilo**3 + c4 * temperature_kilo**4


def equilibrium_constant_smr_value(temperature_k: float) -> float:
    f_smr = _temperature_polynomial_value(
        temperature_k,
        (-109.009403073377, 282.095558262226, -280.329186487247, 136.639283940245, -26.1188462628166),
    )
    return math.exp(f_smr)


def equilibrium_constant_wgs_value(temperature_k: float) -> float:
    f_wgs = _temperature_polynomial_value(
        temperature_k,
        (15.3296851277332, -30.4525115685455, 19.943075765819, -4.24085991611101, -0.232844426047115),
    )
    return math.exp(f_wgs)


def equilibrium_constant_overall_value(temperature_k: float) -> float:
    f_global = _temperature_polynomial_value(
        temperature_k,
        (-88.3330905278377, 229.002065362964, -225.776305234964, 109.686395839527, -20.9330826181798),
    )
    return math.exp(f_global)


def xu_froment_denominator_value(
    *,
    p_co_pa: float,
    p_h2_pa: float,
    p_ch4_pa: float,
    p_h2o_pa: float,
    p_inv_h2_pa_inv: float,
    temperature_k: float,
) -> float:
    return (
        1.0
        + adsorption_constant_value("CO", temperature_k=temperature_k) * p_co_pa
        + adsorption_constant_value("H2", temperature_k=temperature_k) * p_h2_pa
        + adsorption_constant_value("CH4", temperature_k=temperature_k) * p_ch4_pa
        + adsorption_constant_value("H2O", temperature_k=temperature_k) * p_h2o_pa * p_inv_h2_pa_inv
    )


def smr_rate_value(
    *,
    temperature_k: float,
    p_ch4_pa: float,
    p_h2o_pa: float,
    p_co_pa: float,
    p_h2_pa: float,
    p_inv_h2_pa_inv: float,
    catalyst_mass_density_kg_per_m3: float,
) -> float:
    denominator = xu_froment_denominator_value(
        p_co_pa=p_co_pa,
        p_h2_pa=p_h2_pa,
        p_ch4_pa=p_ch4_pa,
        p_h2o_pa=p_h2o_pa,
        p_inv_h2_pa_inv=p_inv_h2_pa_inv,
        temperature_k=temperature_k,
    )
    driving_force = p_ch4_pa * p_h2o_pa - (p_h2_pa**3 * p_co_pa) / (1.0e10 * equilibrium_constant_smr_value(temperature_k))
    return (
        rate_constant_value(
            "smr",
            catalyst_mass_density_kg_per_m3=catalyst_mass_density_kg_per_m3,
            temperature_k=temperature_k,
        )
        * (p_inv_h2_pa_inv**2.5 / (10.0**-2.5))
        * driving_force
        / denominator**2
    )


def wgs_rate_value(
    *,
    temperature_k: float,
    p_co_pa: float,
    p_h2o_pa: float,
    p_co2_pa: float,
    p_h2_pa: float,
    p_inv_h2_pa_inv: float,
    p_ch4_pa: float = 0.0,
    catalyst_mass_density_kg_per_m3: float,
) -> float:
    denominator = xu_froment_denominator_value(
        p_co_pa=p_co_pa,
        p_h2_pa=p_h2_pa,
        p_ch4_pa=p_ch4_pa,
        p_h2o_pa=p_h2o_pa,
        p_inv_h2_pa_inv=p_inv_h2_pa_inv,
        temperature_k=temperature_k,
    )
    driving_force = p_co_pa * p_h2o_pa - (p_h2_pa * p_co2_pa) / equilibrium_constant_wgs_value(temperature_k)
    return (
        rate_constant_value(
            "wgs",
            catalyst_mass_density_kg_per_m3=catalyst_mass_density_kg_per_m3,
            temperature_k=temperature_k,
        )
        * (p_inv_h2_pa_inv / 1.0e5)
        * driving_force
        / denominator**2
    )


def overall_reforming_rate_value(
    *,
    temperature_k: float,
    p_ch4_pa: float,
    p_h2o_pa: float,
    p_co_pa: float,
    p_co2_pa: float,
    p_h2_pa: float,
    p_inv_h2_pa_inv: float,
    catalyst_mass_density_kg_per_m3: float,
) -> float:
    denominator = xu_froment_denominator_value(
        p_co_pa=p_co_pa,
        p_h2_pa=p_h2_pa,
        p_ch4_pa=p_ch4_pa,
        p_h2o_pa=p_h2o_pa,
        p_inv_h2_pa_inv=p_inv_h2_pa_inv,
        temperature_k=temperature_k,
    )
    driving_force = p_ch4_pa * p_h2o_pa**2 - (p_h2_pa**4 * p_co2_pa) / (
        1.0e10 * equilibrium_constant_overall_value(temperature_k)
    )
    return (
        rate_constant_value(
            "overall",
            catalyst_mass_density_kg_per_m3=catalyst_mass_density_kg_per_m3,
            temperature_k=temperature_k,
        )
        * (p_inv_h2_pa_inv**3.5 / (10.0**-2.5))
        * driving_force
        / denominator**2
    )


def _temperature_k_expression(temperature) -> Any:
    return temperature / Constant(1.0 * K)


def _pressure_pa_expression(pressure) -> Any:
    return pressure / Constant(1.0 * Pa)


def _rate_constant_expression(rate_key: str, temperature_k, catalyst_mass_density_kg_per_m3) -> Any:
    coefficient = XU_FROMENT_RATE_COEFFICIENTS[rate_key]
    activation_energy = XU_FROMENT_ACTIVATION_ENERGIES_J_PER_MOL[rate_key]
    return (
        catalyst_mass_density_kg_per_m3
        * Constant(coefficient)
        * Exp(-Constant(activation_energy / GAS_CONSTANT_J_PER_MOL_K) / temperature_k)
    )


def _adsorption_constant_expression(species_id: str, temperature_k) -> Any:
    coefficient = XU_FROMENT_ADSORPTION_COEFFICIENTS[species_id]
    adsorption_energy = XU_FROMENT_ADSORPTION_ENERGIES_J_PER_MOL[species_id]
    return Constant(coefficient) * Exp(Constant(adsorption_energy / GAS_CONSTANT_J_PER_MOL_K) / temperature_k)


def _temperature_polynomial_expression(temperature_k, coefficients: tuple[float, float, float, float, float]) -> Any:
    temperature_kilo = temperature_k / Constant(1000.0)
    c0, c1, c2, c3, c4 = coefficients
    return (
        Constant(c0)
        + Constant(c1) * temperature_kilo
        + Constant(c2) * temperature_kilo**2
        + Constant(c3) * temperature_kilo**3
        + Constant(c4) * temperature_kilo**4
    )


def _equilibrium_constant_smr_expression(temperature_k) -> Any:
    return Exp(
        _temperature_polynomial_expression(
            temperature_k,
            (-109.009403073377, 282.095558262226, -280.329186487247, 136.639283940245, -26.1188462628166),
        )
    )


def _equilibrium_constant_wgs_expression(temperature_k) -> Any:
    return Exp(
        _temperature_polynomial_expression(
            temperature_k,
            (15.3296851277332, -30.4525115685455, 19.943075765819, -4.24085991611101, -0.232844426047115),
        )
    )


def _equilibrium_constant_overall_expression(temperature_k) -> Any:
    return Exp(
        _temperature_polynomial_expression(
            temperature_k,
            (-88.3330905278377, 229.002065362964, -225.776305234964, 109.686395839527, -20.9330826181798),
        )
    )


def _partial_pressure_pa_expression(context: KineticsContext, species_id: str):
    species_idx = context.gas_index(species_id)
    return _pressure_pa_expression(context.model.P(context.idx_cell)) * context.model.y_gas(species_idx, context.idx_cell)


def _hydrogen_inverse_pressure_expression(context: KineticsContext):
    pressure_pa = _pressure_pa_expression(context.model.P(context.idx_cell))
    hydrogen_mole_fraction = context.model.y_gas(context.gas_index("H2"), context.idx_cell)
    blended_floor = Constant(MIN_H2_MOLE_FRACTION) * Exp(Constant(math.log(LOW_H2_BLEND_BASE)) * hydrogen_mole_fraction)
    return Constant(1.0) / (pressure_pa * (hydrogen_mole_fraction + blended_floor))


def _catalyst_mass_density_expression(context: KineticsContext):
    ni_idx = context.solid_index("Ni")
    return context.model.c_sol(ni_idx, context.idx_cell) / Constant(1.0 * mol / m**3) * Constant(NI_MW_KG_PER_MOL)


def _xu_froment_terms(context: KineticsContext) -> XuFromentTerms:
    temperature_k = _temperature_k_expression(context.model.T(context.idx_cell))
    return XuFromentTerms(
        temperature_k=temperature_k,
        p_ch4_pa=_partial_pressure_pa_expression(context, "CH4"),
        p_co_pa=_partial_pressure_pa_expression(context, "CO"),
        p_co2_pa=_partial_pressure_pa_expression(context, "CO2"),
        p_h2_pa=_partial_pressure_pa_expression(context, "H2"),
        p_h2o_pa=_partial_pressure_pa_expression(context, "H2O"),
        p_inv_h2_pa_inv=_hydrogen_inverse_pressure_expression(context),
        denominator=(
            Constant(1.0)
            + _adsorption_constant_expression("CO", temperature_k) * _partial_pressure_pa_expression(context, "CO")
            + _adsorption_constant_expression("H2", temperature_k) * _partial_pressure_pa_expression(context, "H2")
            + _adsorption_constant_expression("CH4", temperature_k) * _partial_pressure_pa_expression(context, "CH4")
            + _adsorption_constant_expression("H2O", temperature_k)
            * _partial_pressure_pa_expression(context, "H2O")
            * _hydrogen_inverse_pressure_expression(context)
        ),
        catalyst_mass_density_kg_per_m3=_catalyst_mass_density_expression(context),
    )


@register_kinetics_hook("xu_froment_smr")
def xu_froment_smr(context: KineticsContext):
    terms = _xu_froment_terms(context)
    driving_force = terms.p_ch4_pa * terms.p_h2o_pa - (
        terms.p_h2_pa**3 * terms.p_co_pa / (Constant(1.0e10) * _equilibrium_constant_smr_expression(terms.temperature_k))
    )
    rate_expression = _rate_constant_expression(
        "smr",
        terms.temperature_k,
        terms.catalyst_mass_density_kg_per_m3,
    ) * (terms.p_inv_h2_pa_inv**2.5 / Constant(10.0**-2.5)) * driving_force / terms.denominator**2
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


@register_kinetics_hook("xu_froment_wgs")
def xu_froment_wgs(context: KineticsContext):
    terms = _xu_froment_terms(context)
    driving_force = terms.p_co_pa * terms.p_h2o_pa - (
        terms.p_h2_pa * terms.p_co2_pa / _equilibrium_constant_wgs_expression(terms.temperature_k)
    )
    rate_expression = _rate_constant_expression(
        "wgs",
        terms.temperature_k,
        terms.catalyst_mass_density_kg_per_m3,
    ) * (terms.p_inv_h2_pa_inv / Constant(1.0e5)) * driving_force / terms.denominator**2
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


@register_kinetics_hook("xu_froment_overall")
def xu_froment_overall(context: KineticsContext):
    terms = _xu_froment_terms(context)
    driving_force = terms.p_ch4_pa * terms.p_h2o_pa**2 - (
        terms.p_h2_pa**4
        * terms.p_co2_pa
        / (Constant(1.0e10) * _equilibrium_constant_overall_expression(terms.temperature_k))
    )
    rate_expression = _rate_constant_expression(
        "overall",
        terms.temperature_k,
        terms.catalyst_mass_density_kg_per_m3,
    ) * (terms.p_inv_h2_pa_inv**3.5 / Constant(10.0**-2.5)) * driving_force / terms.denominator**2
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression


__all__ = [
    "LOW_H2_BLEND_BASE",
    "MIN_H2_MOLE_FRACTION",
    "XU_FROMENT_ACTIVATION_ENERGIES_J_PER_MOL",
    "XU_FROMENT_ADSORPTION_COEFFICIENTS",
    "XU_FROMENT_ADSORPTION_ENERGIES_J_PER_MOL",
    "XU_FROMENT_RATE_COEFFICIENTS",
    "adsorption_constant_value",
    "catalyst_mass_density_value",
    "equilibrium_constant_overall_value",
    "equilibrium_constant_smr_value",
    "equilibrium_constant_wgs_value",
    "hydrogen_inverse_pressure_value",
    "overall_reforming_rate_value",
    "partial_pressure_value",
    "rate_constant_value",
    "smr_rate_value",
    "wgs_rate_value",
    "xu_froment_denominator_value",
]
