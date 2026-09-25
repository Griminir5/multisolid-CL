"""San Pio pseudo-homogeneous redox mechanism for CuO on inert SiO2."""

from __future__ import annotations

from ..reactions import KineticsContext, ReactionDefinition, ReactionFamily
from .runtime import Constant, Exp, K, Pa, Sqrt, m, mol, s


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
PRESSURE_PA_PER_BAR = 1.0e5
MIN_PARTIAL_PRESSURE_BAR = 1.0e-12
GAS_AVAILABILITY_PRESSURE_BAR = 1.0e-6

REDUCTION_COEFFICIENTS = {
    "cuo": 1.54e-1,
    "cu2o": 1.53e-2,
}
REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL = {
    "cuo": 0.15e3,
    "cu2o": 1.79e3,
}
# Table 5 uses the same Cu oxidation rate law as Al2O3, with a separate SiO2 fit.
OXIDATION_COEFFICIENTS = {
    "cu_to_cuo": 1.93e-1,
}
OXIDATION_ACTIVATION_ENERGIES_J_PER_MOL = {
    "cu_to_cuo": 0.18e3,
}


def _temperature_k(context: KineticsContext):
    return context.model.T(context.idx_cell) / Constant(1.0 * K)


def _solid_concentration(context: KineticsContext, species_id: str):
    return context.model.c_sol(
        context.solid_index(species_id), context.idx_cell
    ) / Constant(1.0 * mol / m**3)


def _partial_pressure_bar(context: KineticsContext, species_id: str):
    pressure_bar = context.model.P(context.idx_cell) / Constant(PRESSURE_PA_PER_BAR * Pa)
    return pressure_bar * context.model.y_gas(
        context.gas_index(species_id), context.idx_cell
    )


def _positive_pressure(partial_pressure_bar):
    return Constant(0.5) * (
        partial_pressure_bar
        + Sqrt(
            partial_pressure_bar**2
            + Constant((2.0 * MIN_PARTIAL_PRESSURE_BAR) ** 2)
        )
    )


def _arrhenius(coefficient: float, activation_energy: float, temperature_k):
    return Constant(coefficient) * Exp(
        -Constant(activation_energy / GAS_CONSTANT_J_PER_MOL_K) / temperature_k
    )


def _reduction_rate(context: KineticsContext, reactant: str, rate_key: str):
    temperature_k = _temperature_k(context)
    rate_constant = Constant(context.parameters["REDUCTION_COEFFICIENTS"][rate_key]) * Exp(
        -Constant(
            context.parameters["REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL"][rate_key]
            / GAS_CONSTANT_J_PER_MOL_K
        )
        / temperature_k
    )
    hydrogen_pressure = _positive_pressure(_partial_pressure_bar(context, "H2"))
    availability = hydrogen_pressure / (
        hydrogen_pressure + Constant(GAS_AVAILABILITY_PRESSURE_BAR)
    )
    return (
        Constant(1.0 * mol / (m**3 * s))
        * rate_constant
        * _solid_concentration(context, reactant)
        * availability
    )


def reduce_cuo(context: KineticsContext):
    return _reduction_rate(context, "CuO", "cuo")


def reduce_cu2o(context: KineticsContext):
    return _reduction_rate(context, "Cu2O", "cu2o")


def oxidize_cu(context: KineticsContext):
    oxygen_pressure = _positive_pressure(_partial_pressure_bar(context, "O2"))
    rate = (
        _arrhenius(
            context.parameters["OXIDATION_COEFFICIENTS"]["cu_to_cuo"],
            context.parameters["OXIDATION_ACTIVATION_ENERGIES_J_PER_MOL"]["cu_to_cuo"],
            _temperature_k(context),
        )
        * _solid_concentration(context, "Cu")
        * Sqrt(oxygen_pressure)
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate


FAMILY = ReactionFamily(
    name="copper_sio2_san_pio",
    required_gas_species=("H2", "H2O", "O2"),
    required_solid_species=("Cu", "Cu2O", "CuO"),
    reactions=(
        ReactionDefinition(
            id="cuo_h2_reduction_sio2_san_pio",
            name="CuO reduction to Cu2O by H2 on SiO2",
            phase="gas_solid",
            stoichiometry={"H2": -1.0, "CuO": -2.0, "Cu2O": 1.0, "H2O": 1.0},
            required_species=("H2", "H2O", "CuO", "Cu2O"),
            source_reference="San Pio et al., Chemical Engineering Science 175 (2018) 56-71",
            notes="Pseudo-homogeneous support-inert tenorite reduction from Table 4.",
        ),
        ReactionDefinition(
            id="cu2o_h2_reduction_sio2_san_pio",
            name="Cu2O reduction to Cu by H2 on SiO2",
            phase="gas_solid",
            stoichiometry={"H2": -1.0, "Cu2O": -1.0, "Cu": 2.0, "H2O": 1.0},
            required_species=("H2", "H2O", "Cu2O", "Cu"),
            source_reference="San Pio et al., Chemical Engineering Science 175 (2018) 56-71",
            notes="Pseudo-homogeneous support-inert cuprite reduction from Table 4.",
        ),
        ReactionDefinition(
            id="cu_sio2_oxidation_1_san_pio",
            name="Cu oxidation to CuO on CuO/SiO2",
            phase="gas_solid",
            stoichiometry={"O2": -0.5, "Cu": -1.0, "CuO": 1.0},
            required_species=("O2", "Cu", "CuO"),
            source_reference="San Pio et al., Chemical Engineering Science 175 (2018) 56-71",
            notes="Pseudo-homogeneous Cu oxidation from Eq. (28), with SiO2 parameters from Table 5.",
        ),
    ),
    kinetics_hooks={
        "cuo_h2_reduction_sio2_san_pio": reduce_cuo,
        "cu2o_h2_reduction_sio2_san_pio": reduce_cu2o,
        "cu_sio2_oxidation_1_san_pio": oxidize_cu,
    },
)


__all__ = ("FAMILY",)


# Explicit authoring contract; undeclared implementation constants stay fixed.
from ..parameters import parameter_group

PARAMETERS = {
    **parameter_group("REDUCTION_COEFFICIENTS", REDUCTION_COEFFICIENTS, "1/s", "Reduction coefficients", minimum=0),
    **parameter_group("REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL", REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL, "J/mol", "Reduction activation energies j per mol", minimum=0),
    **parameter_group("OXIDATION_COEFFICIENTS.cu_to_cuo", OXIDATION_COEFFICIENTS['cu_to_cuo'], "1/s", "Cu oxidation; pressure factor is sqrt(p/(100000 Pa))", minimum=0),
    **parameter_group("OXIDATION_ACTIVATION_ENERGIES_J_PER_MOL", OXIDATION_ACTIVATION_ENERGIES_J_PER_MOL, "J/mol", "Oxidation activation energies j per mol", minimum=0),
}
