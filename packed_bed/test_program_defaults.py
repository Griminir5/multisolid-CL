from __future__ import annotations

import unittest

from .config import load_run_bundle
from .visualization import _series_from_segments, _vector_series_from_segments


class ProgramDefaultStepTests(unittest.TestCase):
    def test_load_run_bundle_allows_omitted_steps_and_holds_initial_value(self) -> None:
        run_bundle = load_run_bundle("packed_bed/examples/numaguchi_case/run.yaml")

        self.assertEqual(run_bundle.program.inlet_flow.steps, ())
        self.assertEqual(run_bundle.program.outlet_pressure.steps, ())

        inlet_flow_program = run_bundle.program.inlet_flow.compile_program(
            repeat=run_bundle.run.repeat_program,
            time_horizon=run_bundle.run.time_horizon_s,
        )
        outlet_pressure_program = run_bundle.program.outlet_pressure.compile_program(
            repeat=run_bundle.run.repeat_program,
            time_horizon=run_bundle.run.time_horizon_s,
        )

        self.assertEqual(inlet_flow_program.build_segments(), ())
        self.assertEqual(outlet_pressure_program.build_segments(), ())
        self.assertEqual(inlet_flow_program.initial_value, 1.0)
        self.assertEqual(outlet_pressure_program.initial_value, 500000.0)


class OperatingProgramSeriesTests(unittest.TestCase):
    def test_scalar_series_extends_constant_value_to_horizon(self) -> None:
        times, values = _series_from_segments((), 1.5, final_time=10.0)

        self.assertEqual(times, [0.0, 10.0])
        self.assertEqual(values, [1.5, 1.5])

    def test_vector_series_extends_constant_value_to_horizon(self) -> None:
        times, values_by_component = _vector_series_from_segments((), (0.2, 0.8), final_time=10.0)

        self.assertEqual(times, [0.0, 10.0])
        self.assertEqual(values_by_component, [[0.2, 0.2], [0.8, 0.8]])


if __name__ == "__main__":
    unittest.main()
