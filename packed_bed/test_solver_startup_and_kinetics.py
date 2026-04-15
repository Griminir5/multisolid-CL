from __future__ import annotations

import unittest

from .kinetics.numaguchi_an import (
    catalyst_mass_density_value,
    safe_steam_partial_pressure_bar_value,
    smr_rate_value,
    wgs_rate_value,
)
from .solver import (
    _integrate_until_time_with_breakpoints,
    _warm_start_first_reporting_interval,
)


class _FakeSimulation:
    def __init__(self, reporting_interval: float, time_horizon: float, current_time: float = 0.0) -> None:
        self.ReportingInterval = reporting_interval
        self.TimeHorizon = time_horizon
        self.CurrentTime = current_time
        self.integrated_targets: list[tuple[float, bool]] = []
        self.reported_times: list[float] = []

    def IntegrateUntilTime(self, time: float, _stop_criterion, report_data_around_discontinuities: bool) -> None:
        self.integrated_targets.append((time, report_data_around_discontinuities))
        self.CurrentTime = time

    def ReportData(self, current_time: float) -> None:
        self.reported_times.append(current_time)


class _BreakpointSensitiveSimulation(_FakeSimulation):
    def __init__(
        self,
        reporting_interval: float,
        time_horizon: float,
        current_time: float = 0.0,
        *,
        breakpoint_times: tuple[float, ...],
        direct_step_threshold_s: float = 1.0e-3,
    ) -> None:
        super().__init__(reporting_interval=reporting_interval, time_horizon=time_horizon, current_time=current_time)
        self.breakpoint_times = breakpoint_times
        self.direct_step_threshold_s = direct_step_threshold_s

    def IntegrateUntilTime(self, time: float, _stop_criterion, report_data_around_discontinuities: bool) -> None:
        for breakpoint_time in self.breakpoint_times:
            if abs(self.CurrentTime - breakpoint_time) <= 1.0e-12 and time > breakpoint_time + self.direct_step_threshold_s:
                raise RuntimeError(f"Cannot step directly away from breakpoint at t={breakpoint_time}.")
        super().IntegrateUntilTime(time, _stop_criterion, report_data_around_discontinuities)


class SolverStartupWarmupTests(unittest.TestCase):
    def test_warm_start_substeps_the_first_reporting_interval(self) -> None:
        simulation = _FakeSimulation(reporting_interval=1.0, time_horizon=5.0)

        _warm_start_first_reporting_interval(simulation, (), max_step_s=0.1)

        self.assertEqual(len(simulation.integrated_targets), 10)
        self.assertAlmostEqual(simulation.integrated_targets[0][0], 0.1)
        self.assertAlmostEqual(simulation.integrated_targets[-1][0], 1.0)
        self.assertTrue(all(report_flag is False for _, report_flag in simulation.integrated_targets))
        self.assertEqual(len(simulation.reported_times), 1)
        self.assertAlmostEqual(simulation.reported_times[0], 1.0)
        self.assertAlmostEqual(simulation.CurrentTime, 1.0)

    def test_warm_start_skips_short_first_interval(self) -> None:
        simulation = _FakeSimulation(reporting_interval=0.1, time_horizon=5.0)

        _warm_start_first_reporting_interval(simulation, (), max_step_s=0.1)

        self.assertEqual(simulation.integrated_targets, [])
        self.assertEqual(simulation.reported_times, [])
        self.assertAlmostEqual(simulation.CurrentTime, 0.0)

    def test_integration_helper_nudges_past_breakpoints_before_continuing(self) -> None:
        simulation = _BreakpointSensitiveSimulation(
            reporting_interval=1.0,
            time_horizon=5.0,
            current_time=2.0,
            breakpoint_times=(2.0,),
        )

        _integrate_until_time_with_breakpoints(simulation, 3.0, (2.0,))

        self.assertGreater(len(simulation.integrated_targets), 1)
        self.assertGreater(simulation.integrated_targets[0][0], 2.0)
        self.assertLess(simulation.integrated_targets[0][0], 3.0)
        self.assertAlmostEqual(simulation.integrated_targets[-1][0], 3.0)
        self.assertAlmostEqual(simulation.CurrentTime, 3.0)

class NumaguchiKineticsTests(unittest.TestCase):
    def test_safe_steam_regularization_stays_positive_for_small_negative_overshoots(self) -> None:
        for partial_pressure_bar in (0.0, -1.0e-6, -1.0e-4, -1.0e-3, -1.0e-2):
            self.assertGreater(safe_steam_partial_pressure_bar_value(partial_pressure_bar), 0.0)

    def test_numaguchi_rates_are_finite_without_steam_or_hydrogen(self) -> None:
        catalyst_density = catalyst_mass_density_value(75000.0)

        smr_rate = smr_rate_value(
            temperature_k=900.0,
            p_ch4_pa=0.0,
            p_h2o_pa=0.0,
            p_co_pa=0.0,
            p_h2_pa=0.0,
            catalyst_mass_density_kg_per_m3=catalyst_density,
        )
        wgs_rate = wgs_rate_value(
            temperature_k=900.0,
            p_co_pa=0.0,
            p_h2o_pa=0.0,
            p_co2_pa=0.0,
            p_h2_pa=0.0,
            catalyst_mass_density_kg_per_m3=catalyst_density,
        )

        self.assertEqual(smr_rate, 0.0)
        self.assertEqual(wgs_rate, 0.0)

    def test_numaguchi_rates_match_corrected_packed_bed_form(self) -> None:
        catalyst_density = catalyst_mass_density_value(75000.0)

        smr_rate = smr_rate_value(
            temperature_k=900.0,
            p_ch4_pa=1.0e5,
            p_h2o_pa=2.5e5,
            p_co_pa=2.5e4,
            p_h2_pa=0.0,
            catalyst_mass_density_kg_per_m3=catalyst_density,
        )
        wgs_rate = wgs_rate_value(
            temperature_k=900.0,
            p_co_pa=2.5e4,
            p_h2o_pa=2.5e5,
            p_co2_pa=2.5e4,
            p_h2_pa=0.0,
            catalyst_mass_density_kg_per_m3=catalyst_density,
        )

        self.assertAlmostEqual(smr_rate, 3053.299225308567, delta=0.1)
        self.assertAlmostEqual(wgs_rate, 185.2327522671585, delta=0.1)


if __name__ == "__main__":
    unittest.main()
