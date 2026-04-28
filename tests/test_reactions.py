import unittest

from packed_bed.reactions import REACTION_CATALOG, build_reaction_network


class BuildReactionNetworkValidationTests(unittest.TestCase):
    def test_rejects_missing_stoichiometric_species(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires unselected species: H2O"):
            build_reaction_network(
                ("ni_reduction_h2_medrano",),
                ("H2",),
                ("Ni", "NiO"),
                reaction_catalog=REACTION_CATALOG,
            )

    def test_rejects_missing_catalyst_species(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires unselected species: Ni"):
            build_reaction_network(
                ("smr_reaction_xu_froment",),
                ("CH4", "H2O", "CO", "H2"),
                (),
                reaction_catalog=REACTION_CATALOG,
            )

    def test_accepts_selected_solid_catalyst_for_gas_reaction(self) -> None:
        network = build_reaction_network(
            ("smr_reaction_xu_froment",),
            ("CH4", "H2O", "CO", "H2"),
            ("Ni",),
            reaction_catalog=REACTION_CATALOG,
        )

        self.assertEqual(network.reaction_ids, ("smr_reaction_xu_froment",))


if __name__ == "__main__":
    unittest.main()
