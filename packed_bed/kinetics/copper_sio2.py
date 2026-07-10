"""San Pio pseudo-homogeneous reduction mechanism for CuO on inert SiO2."""

from __future__ import annotations

from ..reactions import ReactionDefinition, ReactionFamily
from . import KineticsContext
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


def _reduction_rate(context: KineticsContext, reactant: str, rate_key: str):
    temperature_k = _temperature_k(context)
    rate_constant = Constant(REDUCTION_COEFFICIENTS[rate_key]) * Exp(
        -Constant(
            REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL[rate_key]
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


FAMILY = ReactionFamily(
    name="copper_sio2_san_pio",
    required_gas_species=("H2", "H2O"),
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
    ),
    kinetics_hooks={
        "cuo_h2_reduction_sio2_san_pio": reduce_cuo,
        "cu2o_h2_reduction_sio2_san_pio": reduce_cu2o,
    },
)


__all__ = ("FAMILY",)
