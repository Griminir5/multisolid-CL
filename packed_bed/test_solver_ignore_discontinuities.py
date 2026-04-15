from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from . import solver_ignore_discontinuities as solver_alt


class _FakeModel:
    def __init__(self) -> None:
        self.reporting_flags: list[bool] = []

    def SetReportingOn(self, enabled: bool) -> None:
        self.reporting_flags.append(enabled)


class _FakeSimulation:
    def __init__(self, reporting_interval: float, time_horizon: float, current_time: float = 0.0) -> None:
        self.ReportingInterval = reporting_interval
        self.TimeHorizon = time_horizon
        self.CurrentTime = current_time
        self.ReportTimeDerivatives = None
        self.model = _FakeModel()
        self.integrated_targets: list[tuple[float, tuple[object, ...]]] = []
        self.reported_times: list[float] = []
        self.initialize_args = None
        self.solve_initial_called = False

    def Initialize(self, solver, reporter, log) -> None:
        self.initialize_args = (solver, reporter, log)

    def SolveInitial(self) -> None:
        self.solve_initial_called = True

    def IntegrateUntilTime(self, time: float, *args) -> float:
        self.integrated_targets.append((time, args))
        self.CurrentTime = time
        return time

    def ReportData(self, current_time: float) -> None:
        self.reported_times.append(current_time)

    def Run(self) -> None:
        raise AssertionError("The ignore-discontinuity solver should not call Run().")


class IgnoreDiscontinuitySolverTests(unittest.TestCase):
    def test_warm_start_substeps_without_breakpoint_handling(self) -> None:
        simulation = _FakeSimulation(reporting_interval=1.0, time_horizon=5.0)

        solver_alt._warm_start_first_reporting_interval(simulation, max_step_s=0.1)

        self.assertEqual(len(simulation.integrated_targets), 10)
        self.assertAlmostEqual(simulation.integrated_targets[0][0], 0.1)
        self.assertAlmostEqual(simulation.integrated_targets[-1][0], 1.0)
        self.assertTrue(all(len(args) == 1 for _, args in simulation.integrated_targets))
        self.assertEqual(simulation.reported_times, [1.0])
        self.assertAlmostEqual(simulation.CurrentTime, 1.0)

    def test_reporting_loop_hits_interval_endpoints_and_final_horizon(self) -> None:
        simulation = _FakeSimulation(reporting_interval=1.0, time_horizon=2.5)

        solver_alt._run_with_reporting_times(simulation)

        self.assertEqual([time for time, _ in simulation.integrated_targets], [1.0, 2.0, 2.5])
        self.assertEqual(simulation.reported_times, [1.0, 2.0, 2.5])
        self.assertAlmostEqual(simulation.CurrentTime, 2.5)

    def test_run_assembled_simulation_uses_manual_integrate_loop(self) -> None:
        simulation = _FakeSimulation(reporting_interval=1.0, time_horizon=3.0)
        assembly = SimpleNamespace(
            run_bundle=SimpleNamespace(
                run=SimpleNamespace(
                    report_time_derivatives=False,
                    reporting_interval_s=1.0,
                    time_horizon_s=3.0,
                    solver=SimpleNamespace(relative_tolerance=1.0e-6),
                )
            ),
            simulation=simulation,
        )
        fake_solver = object()
        fake_reporter = object()
        fake_log = SimpleNamespace(PrintProgress=True)

        with (
            patch.object(solver_alt, "build_idas_solver", return_value=fake_solver),
            patch.object(solver_alt, "daeNoOpDataReporter", return_value=fake_reporter),
            patch.object(solver_alt, "daePythonStdOutLog", return_value=fake_log),
        ):
            reporter = solver_alt.run_assembled_simulation(assembly)

        self.assertIs(reporter, fake_reporter)
        self.assertEqual(simulation.model.reporting_flags, [True])
        self.assertEqual(simulation.initialize_args, (fake_solver, fake_reporter, fake_log))
        self.assertTrue(simulation.solve_initial_called)
        self.assertEqual(simulation.reported_times, [1.0, 2.0, 3.0])
        self.assertEqual([time for time, _ in simulation.integrated_targets][-2:], [2.0, 3.0])
        self.assertFalse(fake_log.PrintProgress)


if __name__ == "__main__":
    unittest.main()
