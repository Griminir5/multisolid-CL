from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from daetools.pyDAE import Abs, Constant, Exp, Log, Sqrt

from pyUnits import K, Pa, m, mol, s

from ..properties import PROPERTY_REGISTRY
from . import KineticsContext, register_kinetics_hook


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324

FE2O3_MW_KG_PER_MOL = PROPERTY_REGISTRY.get_record("Fe2O3").mw
FE_MW_KG_PER_MOL = PROPERTY_REGISTRY.get_record("Fe").mw

'''
Fe + 0.75 O2 -> 0.5 Fe2O3

For oxidation, the conversion (X) is defined in the PDF as 1 when the entirety of iron is converted to Fe2O3.
Because of this, the approximation to the power function must be equal to 0  when X is 1 (and 1-X is 0) to correctly
stop the reaction when all metallic iron is exhausted.
Following the definintion of conversion, the oxidation equation can be rewritten in terms of molar concentrations
of Fe and Fe2O3 as follows: 

X = C_Fe2O3/(C_Fe2O3 + 0.5*C_Fe)
dX/dt = K_MT * (1-X)^n * G(O2), where G is an availabilty gate function that is 1 when O2 is available and 0 if not.

d(C_Fe2O3/(C_Fe2O3 + 0.5*C_Fe))/dt = K_MT * (1 - C_Fe2O3/(C_Fe2O3 + 0.5*C_Fe))^n * G(O2)
d(C_Fe2O3)/dt * 1/(C_Fe2O3 + 0.5*C_Fe) = K_MT * (1 - C_Fe2O3/(C_Fe2O3 + 0.5*C_Fe))^n * G(O2)
d(C_Fe2O3)/dt = (C_Fe2O3 + 0.5*C_Fe) * dX/dt
d(C_Fe2O3)/dt = 0.5 * R

R = 2 * (C_Fe2O3 + 0.5*C_Fe) * K_MT * (1 - C_Fe2O3/(C_Fe2O3 + 0.5*C_Fe))^n * G(O2)
'''


'''
Fe2O3 + 3 H2 -> 2 Fe + 3H2O

For reduction, the conversion (X) is defined as 1 when the entirety of iron is converted to Fe.
Thus, the reduction equation can be rewritten in terms of molar concentrations of Fe and Fe2O3 as follows:

X = C_Fe/(C_Fe + 2*C_Fe2O3)
k_rxn = k0 * exp(-Ea/(R*T)) * H2^m
k_eff = k_rxn/(1 + k_rxn/K_MT)
dX/dt = k_eff * (1-X)^n * (-ln(1-X))^b

Reduction requires a bit more coercion to get into a numerically friendly form:
The (1-X)^n terms is rewritten as Ra(1-X)*(1-X)^4, where Ra() is the rational approximation to ^(n-4)
The Avrami term gets replaced by a (2,1) Pade approximation. It is positive and finite at X=0 and X=1,
has finite slope at X=0 and X=1, and the pole is at around -0.01, where the solver
is not expected to reach.

Reaction rate can be derived like for the oxidation reaction:

d(C_Fe/(C_Fe + 2*C_Fe2O3)) = k_eff * (1 - C_Fe/(C_Fe + 2*C_Fe2O3))^n * (-ln(1 - C_Fe/(C_Fe + 2*C_Fe2O3)))^b
d(C_Fe)/dt * 1/(C_Fe + 2*C_Fe2O3) = k_eff * (1 - C_Fe/(C_Fe + 2*C_Fe2O3))^n * (-ln(1 - C_Fe/(C_Fe + 2*C_Fe2O3)))^b
d(C_Fe)/dt = (C_Fe + 2*C_Fe2O3) * dX/dt
d(C_Fe)/dt = 2*R
R = 0.5*(C_Fe + 2*C_Fe2O3) * k_eff * (1 - C_Fe/(C_Fe + 2*C_Fe2O3))^n * (-ln(1 - C_Fe/(C_Fe + 2*C_Fe2O3)))^b
'''

K0 = { #s^-1 (mol/m^3)^-m
    "H2": 0.1382,
    "O2": None,
}
ACTIVATION_ENERGY_J_PER_MOL = {
    "H2": 37318.0,
    "O2": None,
}
SOLID_REACTION_ORDER = {
    "H2": 4.7769,
    "O2": 0.7882,
}
GAS_REACTION_ORDER = {
    "H2": 0.7554,
    "O2": None,
}
K_MT = {
    "H2": 0.0180,
    "O2": 0.0126,
}
AVRAMI_ORDER = { # not actually used, here for reference
    "H2": 0.1489,
    "O2": None,
}
CONC_GATE = 0.1
REDUCTION_SOLID_REMAINDER_ORDER = 0.7769
H2_POWER_OFFSET_MOLM3 = 0.1

# Two-term rational approximations a*x/(1+b*|x|) + c*x/(1+d*|x|), fit over [0, 1]
# with f(0) = 0 and f(1) = 1.
RATIONAL_POWER_COEFFICIENTS = {
    0.7882:       (1.333, 11.423, 1.034, 0.159), # oxidation solid term
    REDUCTION_SOLID_REMAINDER_ORDER:       (1.452, 11.849, 1.038, 0.170), # reduction solid term, approximating n-4 because that can be easily added back in
}

PADE_NUMER = (0.2476, 67.3172, 43.4567)
PADE_DENOM = (1.0000, 92.6243)

@dataclass(frozen=True)
class FeWasteTerms:
    temperature_k: Any
    o2_gas_conc_molm3: Any
    h2_gas_conc_molm3: Any    
    Fe_conc_molm3: Any
    Fe2O3_conc_molm3: Any


def _temperature_k_expression(temperature) -> Any:
    return temperature / Constant(1.0 * K)

def _concentration_expression(conc) -> Any:
    return conc / Constant(1.0 * mol / m**3)

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

def _conc_gate_value(x: float) -> float:

    return x / (x + CONC_GATE)

def _conc_gate_expr(x):

    return x / (x + Constant(CONC_GATE))

def _pade_value(x: float):
    numer = PADE_NUMER[0] + PADE_NUMER[1]*x + PADE_NUMER[2]*x*x
    denom = PADE_DENOM[0] + PADE_DENOM[1]*x
    return numer/denom

def _pade_expr(x):
    numer = Constant(PADE_NUMER[0]) + Constant(PADE_NUMER[1])*x + Constant(PADE_NUMER[2])*x*x
    denom = Constant(PADE_DENOM[0]) + Constant(PADE_DENOM[1])*x
    return numer/denom

def oxidation_conversion_value(*, Fe_conc_molm3: float, Fe2O3_conc_molm3: float) -> float:
    return Fe2O3_conc_molm3 / (Fe2O3_conc_molm3 + 0.5 * Fe_conc_molm3)

def reduction_conversion_value(*, Fe_conc_molm3: float, Fe2O3_conc_molm3: float) -> float:
    return Fe_conc_molm3 / (Fe_conc_molm3 + 2.0 * Fe2O3_conc_molm3)

def h2_concentration_power_value(h2_gas_conc_molm3: float) -> float:
    order = GAS_REACTION_ORDER["H2"]
    return (h2_gas_conc_molm3 + H2_POWER_OFFSET_MOLM3) ** order - H2_POWER_OFFSET_MOLM3 ** order

def _h2_concentration_power_expr(h2_gas_conc_molm3):
    order = Constant(GAS_REACTION_ORDER["H2"])
    offset = Constant(H2_POWER_OFFSET_MOLM3)
    return (h2_gas_conc_molm3 + offset) ** order - offset ** order

def h2_reaction_constant_value(*, temperature_k: float, h2_gas_conc_molm3: float) -> float:
    return (
        K0["H2"]
        * math.exp(-ACTIVATION_ENERGY_J_PER_MOL["H2"] / (GAS_CONSTANT_J_PER_MOL_K * temperature_k))
        * h2_concentration_power_value(h2_gas_conc_molm3)
    )

def h2_effective_rate_constant_value(*, temperature_k: float, h2_gas_conc_molm3: float) -> float:
    k_rxn = h2_reaction_constant_value(
        temperature_k=temperature_k,
        h2_gas_conc_molm3=h2_gas_conc_molm3,
    )
    return k_rxn / (1.0 + k_rxn / K_MT["H2"])

def _h2_effective_rate_constant_expr(*, temperature_k, h2_gas_conc_molm3):
    k_rxn = (
        Constant(K0["H2"])
        * Exp(-Constant(ACTIVATION_ENERGY_J_PER_MOL["H2"] / GAS_CONSTANT_J_PER_MOL_K) / temperature_k)
        * _h2_concentration_power_expr(h2_gas_conc_molm3)
    )
    return k_rxn / (Constant(1.0) + k_rxn / Constant(K_MT["H2"]))

def _oxidation_rate(*, Fe_conc_molm3, Fe2O3_conc_molm3, o2_gas_conc_molm3, constant, rational_power, conc_gate):
    total_sites = Fe2O3_conc_molm3 + constant(0.5) * Fe_conc_molm3
    x_ox = Fe2O3_conc_molm3 / total_sites
    solid_term = rational_power(SOLID_REACTION_ORDER["O2"], constant(1.0) - x_ox)
    gas_term = conc_gate(o2_gas_conc_molm3)
    return constant(2.0) * total_sites * constant(K_MT["O2"]) * solid_term * gas_term

def _reduction_rate(
    *,
    temperature_k,
    Fe_conc_molm3,
    Fe2O3_conc_molm3,
    h2_gas_conc_molm3,
    constant,
    rational_power,
    pade,
    h2_effective_rate_constant,
):
    total_sites = Fe_conc_molm3 + constant(2.0) * Fe2O3_conc_molm3
    x_red = Fe_conc_molm3 / total_sites
    one_minus_x = constant(1.0) - x_red
    solid_term = rational_power(REDUCTION_SOLID_REMAINDER_ORDER, one_minus_x) * one_minus_x**4
    avrami_term = pade(x_red)
    k_eff = h2_effective_rate_constant(
        temperature_k=temperature_k,
        h2_gas_conc_molm3=h2_gas_conc_molm3,
    )
    return constant(0.5) * total_sites * k_eff * solid_term * avrami_term

def _fe_waste_terms(context: KineticsContext) -> FeWasteTerms:
    iron_idx = context.solid_index("Fe")
    iron_oxide_idx = context.solid_index("Fe2O3")
    Fe_conc = _concentration_expression(context.model.c_sol(iron_idx, context.idx_cell))
    Fe2O3_conc = _concentration_expression(context.model.c_sol(iron_oxide_idx, context.idx_cell))    
    o2_idx = context.gas_index("O2")
    h2_idx = context.gas_index("H2")
    o2_conc = _concentration_expression(context.model.c_gas(o2_idx, context.idx_cell))
    h2_conc = _concentration_expression(context.model.c_gas(h2_idx, context.idx_cell))
    temperature_k = _temperature_k_expression(context.model.T(context.idx_cell))

    return FeWasteTerms(
        temperature_k=temperature_k,
        Fe_conc_molm3=Fe_conc,
        Fe2O3_conc_molm3=Fe2O3_conc,
        o2_gas_conc_molm3=o2_conc,
        h2_gas_conc_molm3=h2_conc,
    )

def fe_waste_oxidation_value(
    *,
    Fe_conc_molm3: float,
    Fe2O3_conc_molm3: float,
    o2_gas_conc_molm3: float,
) -> float:
    return _oxidation_rate(
        Fe_conc_molm3=Fe_conc_molm3,
        Fe2O3_conc_molm3=Fe2O3_conc_molm3,
        o2_gas_conc_molm3=o2_gas_conc_molm3,
        constant=lambda value: value,
        rational_power=rational_power_value,
        conc_gate=_conc_gate_value,
    )

@register_kinetics_hook("fe_waste_oxidation")
def fe_waste_oxidation_expr(context: KineticsContext):
    terms = _fe_waste_terms(context)
    rate_expression = _oxidation_rate(
        Fe_conc_molm3=terms.Fe_conc_molm3,
        Fe2O3_conc_molm3=terms.Fe2O3_conc_molm3,
        o2_gas_conc_molm3=terms.o2_gas_conc_molm3,
        constant=Constant,
        rational_power=_rational_power_expr,
        conc_gate=_conc_gate_expr,
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression



def fe_waste_reduction_value(
    *,
    temperature_k: float,
    Fe_conc_molm3: float,
    Fe2O3_conc_molm3: float,
    h2_gas_conc_molm3: float,
) -> float:
    return _reduction_rate(
        temperature_k=temperature_k,
        Fe_conc_molm3=Fe_conc_molm3,
        Fe2O3_conc_molm3=Fe2O3_conc_molm3,
        h2_gas_conc_molm3=h2_gas_conc_molm3,
        constant=lambda value: value,
        rational_power=rational_power_value,
        pade=_pade_value,
        h2_effective_rate_constant=h2_effective_rate_constant_value,
    )

@register_kinetics_hook("fe_waste_reduction")
def fe_waste_reduction_expr(context: KineticsContext):
    terms = _fe_waste_terms(context)
    rate_expression = _reduction_rate(
        temperature_k=terms.temperature_k,
        Fe_conc_molm3=terms.Fe_conc_molm3,
        Fe2O3_conc_molm3=terms.Fe2O3_conc_molm3,
        h2_gas_conc_molm3=terms.h2_gas_conc_molm3,
        constant=Constant,
        rational_power=_rational_power_expr,
        pade=_pade_expr,
        h2_effective_rate_constant=_h2_effective_rate_constant_expr,
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression
