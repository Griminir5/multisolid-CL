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

POS_EPS = 1.0e-10
F_FLOOR = 1.0e-4
F_GATE = 1.0e-3
CG_FLOOR = 1.0e-8

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
    h2_conc_molm3: Any
    co_conc_molm3: Any
    o2_conc_molm3: Any
    ni_conc_molm3: Any
    nio_conc_molm3: Any
    frac_reduced: Any
    frac_oxidised: Any

def _smooth_pos_value(x):
    return 0.5 * (x + math.sqrt(x * x + POS_EPS * POS_EPS))

def _smooth_pos_expr(x):
    return Constant(0.5) * (x + Sqrt(x * x + Constant(POS_EPS * POS_EPS)))

def _temperature_k_expression(temperature) -> Any:
    return temperature / Constant(1.0 * K)

def _concentration_expression(conc) -> Any:
    return conc / Constant(1.0 * mol * m**3)

def _medrano_terms(context: KineticsContext) -> MedranoTerms:
    nickel_idx = context.solid_index("Ni")
    nickel_oxide_idx = context.solid_index("NiO")
    h2_idx = context.gas_index("H2")
    co_idx = context.gas_index("CO")
    o2_idx = context.gas_index("O2")
    temperature_k = _temperature_k_expression(context.model.T(context.idx_cell))
    c_ni  = _concentration_expression(context.model.c_sol(nickel_idx, context.idx_cell))
    c_nio = _concentration_expression(context.model.c_sol(nickel_oxide_idx, context.idx_cell))
    c_h2  = _concentration_expression(context.model.c_gas(h2_idx, context.idx_cell))
    c_co  = _concentration_expression(context.model.c_gas(co_idx, context.idx_cell))
    c_o2  = _concentration_expression(context.model.c_gas(o2_idx, context.idx_cell))

    return MedranoTerms(
        temperature_k=temperature_k,
        h2_conc_molm3=c_h2,
        co_conc_molm3=c_co,
        o2_conc_molm3=c_o2,
        ni_conc_molm3=c_ni,
        ni_conc_molm3=c_nio,
        frac_reduced=c_ni / (c_ni+c_nio),
        frac_oxidised=c_nio / (c_ni+c_nio),
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
    activation_energy = ACTIVATION_ENERGY_J_PER_MOL[comp_key]
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
    activation_energy = ACTIVATION_ENERGY_J_PER_MOL[comp_key]
    kx = KX[comp_key]
    return Constant(preexp) * Exp(
        -Constant(activation_energy / GAS_CONSTANT_J_PER_MOL_K) / temperature_k) * Exp(-Constant(kx)*conv)
