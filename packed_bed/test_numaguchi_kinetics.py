from __future__ import annotations

import unittest

try:
    from .kinetics import numaguchi_an
except ImportError:  # pragma: no cover - supports unittest discovery with -s packed_bed
    from packed_bed.kinetics import numaguchi_an


class NumaguchiKineticsTests(unittest.TestCase):
    def test_negative_nickel_inventory_gives_zero_catalyst_density(self) -> None:
        density = numaguchi_an.catalyst_mass_density_value(-1.0e-5)

        self.assertEqual(density, 0.0)


if __name__ == "__main__":
    unittest.main()
