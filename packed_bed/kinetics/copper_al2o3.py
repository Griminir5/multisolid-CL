"""San Pio pseudo-homogeneous redox mechanism for CuO on Al2O3."""

from __future__ import annotations

from ..reactions import ReactionDefinition, ReactionFamily
from . import KineticsContext
from .runtime import Constant, Exp, K, Pa, Sqrt, m, mol, s


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
PRESSURE_PA_PER_BAR = 1.0e5
MIN_PARTIAL_PRESSURE_BAR = 1.0e-12
GAS_AVAILABILITY_PRESSURE_BAR = 1.0e-6

# The paper assumes these CuO/Cu2O steps are independent of the support.
REDUCTION_COEFFICIENTS = {
    "cuo": 1.54e-1,
    "cu2o": 1.53e-2,
    "spinel_to_cu": 1.93e10,
    "spinel_to_cualo2": 5.4e-1,
    "cualo2_to_cu": 4.87e-3,
}
REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL = {
    "cuo": 0.15e3,
    "cu2o": 1.79e3,
    "spinel_to_cu": 241.75e3,
    "spinel_to_cualo2": 0.37e3,
    "cualo2_to_cu": 8.85e3,
}
OXIDATION_COEFFICIENTS = {
    "cu_to_cuo": 8.54e-1,
    "cuo_to_spinel": 1.27e-6,
    "cualo2_to_spinel": 1.98e-5,
}
OXIDATION_ACTIVATION_ENERGIES_J_PER_MOL = {
    "cu_to_cuo": 0.83e3,
    "cuo_to_spinel": 1.18e3,
    "cualo2_to_spinel": 0.71e3,
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
    rate_constant = _arrhenius(
        REDUCTION_COEFFICIENTS[rate_key],
        REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL[rate_key],
        _temperature_k(context),
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


def reduce_spinel_to_cu(context: KineticsContext):
    return _reduction_rate(context, "CuAl2O4", "spinel_to_cu")


def reduce_spinel_to_cualo2(context: KineticsContext):
    return _reduction_rate(context, "CuAl2O4", "spinel_to_cualo2")


def reduce_cualo2_to_cu(context: KineticsContext):
    return _reduction_rate(context, "CuAlO2", "cualo2_to_cu")


def oxidize_cu(context: KineticsContext):
    oxygen_pressure = _positive_pressure(_partial_pressure_bar(context, "O2"))
    rate = (
        _arrhenius(
            OXIDATION_COEFFICIENTS["cu_to_cuo"],
            OXIDATION_ACTIVATION_ENERGIES_J_PER_MOL["cu_to_cuo"],
            _temperature_k(context),
        )
        * _solid_concentration(context, "Cu")
        * Sqrt(oxygen_pressure)
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate


def form_spinel_from_cuo(context: KineticsContext):
    rate = (
        _arrhenius(
            OXIDATION_COEFFICIENTS["cuo_to_spinel"],
            OXIDATION_ACTIVATION_ENERGIES_J_PER_MOL["cuo_to_spinel"],
            _temperature_k(context),
        )
        * _solid_concentration(context, "CuO")
        * _solid_concentration(context, "Al2O3")
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate


def oxidize_cualo2(context: KineticsContext):
    oxygen_pressure = _positive_pressure(_partial_pressure_bar(context, "O2"))
    rate = (
        _arrhenius(
            OXIDATION_COEFFICIENTS["cualo2_to_spinel"],
            OXIDATION_ACTIVATION_ENERGIES_J_PER_MOL["cualo2_to_spinel"],
            _temperature_k(context),
        )
        * _solid_concentration(context, "CuAlO2")
        * _solid_concentration(context, "Al2O3")
        * Sqrt(oxygen_pressure)
    )
    return Constant(1.0 * mol / (m**3 * s)) * rate


SOURCE = "San Pio et al., Chemical Engineering Science 175 (2018) 56-71"

FAMILY = ReactionFamily(
    name="copper_al2o3_san_pio",
    required_gas_species=("H2", "H2O", "O2"),
    required_solid_species=("Cu", "Cu2O", "CuO", "Al2O3", "CuAlO2", "CuAl2O4"),
    reactions=(
        ReactionDefinition(
            id="cuo_h2_reduction_al2o3_san_pio",
            name="CuO reduction to Cu2O by H2 on Al2O3",
            phase="gas_solid",
            stoichiometry={"H2": -1.0, "CuO": -2.0, "Cu2O": 1.0, "H2O": 1.0},
            required_species=("H2", "H2O", "CuO", "Cu2O"),
            source_reference=SOURCE,
            notes="Support-independent tenorite reduction extended to the Al2O3 mechanism.",
        ),
        ReactionDefinition(
            id="cu2o_h2_reduction_al2o3_san_pio",
            name="Cu2O reduction to Cu by H2 on Al2O3",
            phase="gas_solid",
            stoichiometry={"H2": -1.0, "Cu2O": -1.0, "Cu": 2.0, "H2O": 1.0},
            required_species=("H2", "H2O", "Cu2O", "Cu"),
            source_reference=SOURCE,
            notes="Support-independent cuprite reduction extended to the Al2O3 mechanism.",
        ),
        ReactionDefinition(
            id="cu_al2o3_spinel_reduction_1_san_pio",
            name="CuAl2O4 reduction to Cu on CuO/Al2O3",
            phase="gas_solid",
            stoichiometry={
                "H2": -1.0,
                "CuAl2O4": -1.0,
                "Cu": 1.0,
                "Al2O3": 1.0,
                "H2O": 1.0,
            },
            required_species=("H2", "H2O", "CuAl2O4", "Cu", "Al2O3"),
            source_reference=SOURCE,
        ),
        ReactionDefinition(
            id="cu_al2o3_spinel_reduction_2_san_pio",
            name="CuAl2O4 reduction to CuAlO2 on CuO/Al2O3",
            phase="gas_solid",
            stoichiometry={
                "H2": -1.0,
                "CuAl2O4": -2.0,
                "CuAlO2": 2.0,
                "Al2O3": 1.0,
                "H2O": 1.0,
            },
            required_species=("H2", "H2O", "CuAl2O4", "CuAlO2", "Al2O3"),
            source_reference=SOURCE,
        ),
        ReactionDefinition(
            id="cu_al2o3_spinel_reduction_3_san_pio",
            name="CuAlO2 reduction to Cu on CuO/Al2O3",
            phase="gas_solid",
            stoichiometry={
                "H2": -1.0,
                "CuAlO2": -2.0,
                "Cu": 2.0,
                "Al2O3": 1.0,
                "H2O": 1.0,
            },
            required_species=("H2", "H2O", "CuAlO2", "Cu", "Al2O3"),
            source_reference=SOURCE,
        ),
        ReactionDefinition(
            id="cu_al2o3_oxidation_1_san_pio",
            name="Cu oxidation to CuO on CuO/Al2O3",
            phase="gas_solid",
            stoichiometry={"O2": -0.5, "Cu": -1.0, "CuO": 1.0},
            required_species=("O2", "Cu", "CuO"),
            source_reference=SOURCE,
        ),
        ReactionDefinition(
            id="cu_al2o3_oxidation_2_san_pio",
            name="CuO reaction with Al2O3 to form CuAl2O4",
            phase="solid_solid",
            stoichiometry={"CuO": -1.0, "Al2O3": -1.0, "CuAl2O4": 1.0},
            required_species=("CuO", "Al2O3", "CuAl2O4"),
            source_reference=SOURCE,
        ),
        ReactionDefinition(
            id="cu_al2o3_oxidation_3_san_pio",
            name="CuAlO2 oxidation with Al2O3 to form CuAl2O4",
            phase="gas_solid",
            stoichiometry={
                "O2": -0.5,
                "CuAlO2": -2.0,
                "Al2O3": -1.0,
                "CuAl2O4": 2.0,
            },
            required_species=("O2", "CuAlO2", "Al2O3", "CuAl2O4"),
            source_reference=SOURCE,
        ),
    ),
    kinetics_hooks={
        "cuo_h2_reduction_al2o3_san_pio": reduce_cuo,
        "cu2o_h2_reduction_al2o3_san_pio": reduce_cu2o,
        "cu_al2o3_spinel_reduction_1_san_pio": reduce_spinel_to_cu,
        "cu_al2o3_spinel_reduction_2_san_pio": reduce_spinel_to_cualo2,
        "cu_al2o3_spinel_reduction_3_san_pio": reduce_cualo2_to_cu,
        "cu_al2o3_oxidation_1_san_pio": oxidize_cu,
        "cu_al2o3_oxidation_2_san_pio": form_spinel_from_cuo,
        "cu_al2o3_oxidation_3_san_pio": oxidize_cualo2,
    },
)


__all__ = ("FAMILY",)
