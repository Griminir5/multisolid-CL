from __future__ import annotations

import unittest

try:
    from .benchmark import build_benchmark_specs
    from .solver import _restore_reporting_on, _set_reporting_on
except ImportError:  # pragma: no cover - supports unittest discovery with -s packed_bed
    from packed_bed.benchmark import build_benchmark_specs
    from packed_bed.solver import _restore_reporting_on, _set_reporting_on


class _Reportable:
    def __init__(self) -> None:
        self.ReportingOn = True


class _FakeModel:
    def __init__(self) -> None:
        self.reporting_flags: list[bool] = []
        self.dictVariables = {
            "temp_bed": _Reportable(),
            "pres_bed": _Reportable(),
            "y_gas": _Reportable(),
            "N_gas_face": _Reportable(),
            "P_out": _Reportable(),
            "heat_in_total": _Reportable(),
            "heat_out_total": _Reportable(),
            "heat_bed_total": _Reportable(),
        }
        self.dictParameters = {
            "xval_cells": _Reportable(),
        }

    def SetReportingOn(self, enabled: bool) -> None:
        self.reporting_flags.append(enabled)
        for reportable in (*self.dictVariables.values(), *self.dictParameters.values()):
            reportable.ReportingOn = enabled


class _FakeSimulation:
    def __init__(self) -> None:
        self.model = _FakeModel()


class ReportingSelectionTests(unittest.TestCase):
    def test_requested_reports_enable_only_requested_variables(self) -> None:
        simulation = _FakeSimulation()

        _set_reporting_on(simulation, ("temperature",), include_plot_variables=False)

        self.assertEqual(simulation.model.reporting_flags, [False])
        self.assertTrue(simulation.model.dictVariables["temp_bed"].ReportingOn)
        self.assertFalse(simulation.model.dictVariables["pres_bed"].ReportingOn)
        self.assertFalse(simulation.model.dictVariables["N_gas_face"].ReportingOn)
        self.assertFalse(simulation.model.dictParameters["xval_cells"].ReportingOn)

    def test_plot_reporting_adds_plot_dependencies(self) -> None:
        simulation = _FakeSimulation()

        _set_reporting_on(simulation, ("temperature",), include_plot_variables=True)

        enabled_variables = {
            name
            for name, variable in simulation.model.dictVariables.items()
            if variable.ReportingOn
        }
        self.assertEqual(
            enabled_variables,
            {"temp_bed", "pres_bed", "y_gas", "N_gas_face", "P_out"},
        )
        self.assertTrue(simulation.model.dictParameters["xval_cells"].ReportingOn)

    def test_restore_preserves_selective_reporting_after_temporary_disable(self) -> None:
        simulation = _FakeSimulation()
        _set_reporting_on(simulation, ("temperature",), include_plot_variables=True)

        simulation.model.SetReportingOn(False)
        _restore_reporting_on(simulation)

        self.assertTrue(simulation.model.dictVariables["temp_bed"].ReportingOn)
        self.assertTrue(simulation.model.dictVariables["N_gas_face"].ReportingOn)
        self.assertFalse(simulation.model.dictVariables["heat_in_total"].ReportingOn)

    def test_heat_balance_expands_to_heat_totals(self) -> None:
        simulation = _FakeSimulation()

        _set_reporting_on(simulation, ("heat_balance",), include_plot_variables=False)

        self.assertTrue(simulation.model.dictVariables["heat_in_total"].ReportingOn)
        self.assertTrue(simulation.model.dictVariables["heat_out_total"].ReportingOn)
        self.assertTrue(simulation.model.dictVariables["heat_bed_total"].ReportingOn)
        self.assertFalse(simulation.model.dictVariables["temp_bed"].ReportingOn)

    def test_medrano_benchmark_tiers_are_available(self) -> None:
        self.assertEqual([spec.name for spec in build_benchmark_specs("medrano-short")], ["medrano-short"])
        self.assertEqual(len(build_benchmark_specs("medrano-scale")), 4)
        self.assertEqual(len(build_benchmark_specs("medrano-reporting")), 6)
        self.assertEqual(len(build_benchmark_specs("medrano-runner")), 8)


if __name__ == "__main__":
    unittest.main()
