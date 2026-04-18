from __future__ import annotations

import unittest

try:
    from .axial_schemes import reconstruct_face_states
except ImportError:  # pragma: no cover - supports unittest discovery with -s packed_bed
    from packed_bed.axial_schemes import reconstruct_face_states


class AxialSchemeTests(unittest.TestCase):
    def test_high_order_reconstructions_preserve_constant_state(self) -> None:
        values = [2.5] * 12

        for scheme_name in ("linear_upwind5", "weno5", "weno7"):
            with self.subTest(scheme_name=scheme_name):
                left_value, right_value = reconstruct_face_states(
                    lambda cell_index: values[cell_index],
                    face_index=5,
                    n_cells=len(values),
                    scheme_name=scheme_name,
                    small_eps=1.0e-8,
                )

                self.assertAlmostEqual(left_value, 2.5)
                self.assertAlmostEqual(right_value, 2.5)


if __name__ == "__main__":
    unittest.main()
