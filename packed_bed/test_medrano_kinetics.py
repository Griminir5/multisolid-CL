from __future__ import annotations

import math
import unittest

try:
    from .kinetics import resolve_kinetics_hooks
    from .kinetics import medrano
    from .reactions import REACTION_CATALOG, build_reaction_network
except ImportError:  # pragma: no cover - supports unittest discovery with -s packed_bed
    from packed_bed.kinetics import resolve_kinetics_hooks
    from packed_bed.kinetics import medrano
    from packed_bed.reactions import REACTION_CATALOG, build_reaction_network


class MedranoKineticsTests(unittest.TestCase):
    def test_diffusivity_uses_diffusion_activation_energy(self) -> None:
        temperature_k = 700.0
        conversion = 0.4

        value = medrano.D_value("H2", temperature_k=temperature_k, conv=conversion)

        expected = medrano.D0_M2_PER_S["H2"] * math.exp(
            -medrano.ED_J_PER_MOL["H2"] / (medrano.GAS_CONSTANT_J_PER_MOL_K * temperature_k)
        ) * math.exp(-medrano.KX["H2"] * conversion)
        wrong_activation_energy = medrano.D0_M2_PER_S["H2"] * math.exp(
            -medrano.ACTIVATION_ENERGY_J_PER_MOL["H2"]
            / (medrano.GAS_CONSTANT_J_PER_MOL_K * temperature_k)
        ) * math.exp(-medrano.KX["H2"] * conversion)

        self.assertAlmostEqual(value, expected)
        self.assertNotAlmostEqual(value, wrong_activation_energy)

    def test_oxygen_rate_uses_corrected_b_multiplier(self) -> None:
        temperature_k = 900.0
        gas_concentration_molm3 = 2.0
        conversion = 0.3
        reactant_fraction = 0.7

        rate = medrano.medrano_conversion_rate_value(
            "O2",
            temperature_k=temperature_k,
            gas_concentration_molm3=gas_concentration_molm3,
            conversion=conversion,
            reactant_fraction=reactant_fraction,
        )

        gas_safe = medrano.safe_gas_concentration_value(gas_concentration_molm3)
        f_power = medrano.F_FLOOR + (1.0 - medrano.F_FLOOR) * reactant_fraction
        gas_gate = gas_concentration_molm3 / (gas_concentration_molm3 + medrano.CG_GATE)
        solid_gate = reactant_fraction / (reactant_fraction + medrano.F_GATE)
        k = medrano.k_value("O2", temperature_k=temperature_k)
        diffusivity = medrano.D_value("O2", temperature_k=temperature_k, conv=conversion)
        denominator = (
            (1.0 / k) * f_power ** (-2.0 / 3.0)
            + (medrano.R0_M["O2"] / diffusivity) * (f_power ** (-1.0 / 3.0) - 1.0)
        )
        uncorrected_b_rate = gas_gate * solid_gate * (
            3.0
            * gas_safe ** medrano.REACTION_ORDER["O2"]
            / (medrano.R0_M["O2"] * medrano.CS_MOL_PER_M3["O2"])
        ) / denominator

        self.assertAlmostEqual(rate, 2.0 * uncorrected_b_rate)

    def test_reduction_rates_go_to_zero_without_nio(self) -> None:
        for comp_key in ("H2", "CO"):
            with self.subTest(comp_key=comp_key):
                rate = medrano.medrano_reaction_rate_value(
                    comp_key,
                    temperature_k=900.0,
                    gas_concentration_molm3=2.0,
                    conversion=1.0,
                    reactant_fraction=0.0,
                    active_inventory_molm3=1000.0,
                )

                self.assertEqual(rate, 0.0)

    def test_oxidation_rate_goes_to_zero_without_ni(self) -> None:
        rate = medrano.medrano_reaction_rate_value(
            "O2",
            temperature_k=900.0,
            gas_concentration_molm3=2.0,
            conversion=1.0,
            reactant_fraction=0.0,
            active_inventory_molm3=1000.0,
        )

        self.assertEqual(rate, 0.0)

    def test_active_inventory_uses_current_solid_reactant(self) -> None:
        self.assertEqual(
            medrano.active_solid_inventory_value(
                "O2",
                ni_concentration_molm3=2.0,
                nio_concentration_molm3=5.0,
            ),
            2.0,
        )
        self.assertEqual(
            medrano.active_solid_inventory_value(
                "H2",
                ni_concentration_molm3=2.0,
                nio_concentration_molm3=5.0,
            ),
            5.0,
        )
        self.assertEqual(
            medrano.active_solid_inventory_value(
                "CO",
                ni_concentration_molm3=2.0,
                nio_concentration_molm3=5.0,
            ),
            5.0,
        )

    def test_slightly_negative_reactant_fraction_within_floor_is_finite(self) -> None:
        rate = medrano.medrano_reaction_rate_value(
            "H2",
            temperature_k=900.0,
            gas_concentration_molm3=2.0,
            conversion=1.0,
            reactant_fraction=-0.5 * medrano.F_FLOOR,
            active_inventory_molm3=1000.0,
        )

        self.assertTrue(math.isfinite(rate))

    def test_negative_gas_concentration_is_finite(self) -> None:
        rate = medrano.medrano_reaction_rate_value(
            "CO",
            temperature_k=900.0,
            gas_concentration_molm3=-2.0,
            conversion=0.5,
            reactant_fraction=0.5,
            active_inventory_molm3=1000.0,
        )

        self.assertTrue(math.isfinite(rate))

    def test_no_gas_reactant_gives_zero_rate(self) -> None:
        rate = medrano.medrano_reaction_rate_value(
            "H2",
            temperature_k=900.0,
            gas_concentration_molm3=0.0,
            conversion=0.5,
            reactant_fraction=0.5,
            active_inventory_molm3=1000.0,
        )

        self.assertEqual(rate, 0.0)

    def test_negative_reactant_fraction_gives_zero_rate(self) -> None:
        rate = medrano.medrano_reaction_rate_value(
            "H2",
            temperature_k=900.0,
            gas_concentration_molm3=2.0,
            conversion=1.0,
            reactant_fraction=-0.5 * medrano.F_FLOOR,
            active_inventory_molm3=1000.0,
        )

        self.assertEqual(rate, 0.0)

    def test_positive_available_solid_and_gas_produce_positive_rate(self) -> None:
        rate = medrano.medrano_reaction_rate_value(
            "CO",
            temperature_k=900.0,
            gas_concentration_molm3=2.0,
            conversion=0.5,
            reactant_fraction=0.5,
            active_inventory_molm3=1000.0,
        )

        self.assertTrue(math.isfinite(rate))
        self.assertGreater(rate, 0.0)

    def test_reactant_fraction_at_one_keeps_co_reduction_positive(self) -> None:
        rate = medrano.medrano_reaction_rate_value(
            "CO",
            temperature_k=600.0,
            gas_concentration_molm3=1.0e-4,
            conversion=0.0,
            reactant_fraction=1.0,
            active_inventory_molm3=1143.0,
        )

        self.assertTrue(math.isfinite(rate))
        self.assertGreaterEqual(rate, 0.0)


class MedranoRegistryTests(unittest.TestCase):
    def test_medrano_reaction_ids_resolve_to_hooks_in_network_order(self) -> None:
        reaction_ids = (
            "ni_reduction_h2_medrano",
            "ni_reduction_co_medrano",
            "ni_oxidation_o2_medrano",
        )
        network = build_reaction_network(
            reaction_ids,
            ("H2", "H2O", "CO", "CO2", "O2"),
            ("Ni", "NiO"),
            reaction_catalog=REACTION_CATALOG,
        )

        hooks = resolve_kinetics_hooks(network)

        self.assertEqual(
            [hook.__name__ for hook in hooks],
            ["medrano_reduction_h2", "medrano_reduction_co", "medrano_oxidation_o2"],
        )

    def test_single_medrano_reduction_resolves_with_only_required_gases(self) -> None:
        network = build_reaction_network(
            ("ni_reduction_h2_medrano",),
            ("H2", "H2O"),
            ("Ni", "NiO"),
            reaction_catalog=REACTION_CATALOG,
        )

        hooks = resolve_kinetics_hooks(network)

        self.assertEqual([hook.__name__ for hook in hooks], ["medrano_reduction_h2"])


if __name__ == "__main__":
    unittest.main()
