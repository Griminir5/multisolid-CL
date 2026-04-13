from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .cli import run_simulation
from .config import RunResult, load_run_bundle
from .result_plots import extract_run_result_plot_data, render_run_result_plots


class _FakeVariable:
    def __init__(self, name: str, time_values, values):
        self.Name = name
        self.TimeValues = np.asarray(time_values, dtype=float)
        self.Values = np.asarray(values, dtype=float)


class _FakeProcess:
    def __init__(self, variables: dict[str, _FakeVariable]):
        self.dictVariables = variables


class _FakeReporter:
    def __init__(self, variables: dict[str, _FakeVariable]):
        self.Process = _FakeProcess(variables)


class ResultPlotTests(unittest.TestCase):
    def _build_run_result(self, time_s: np.ndarray | None = None) -> RunResult:
        run_bundle = load_run_bundle("packed_bed/examples/default_case/run.yaml")
        system_name = run_bundle.run.system_name

        if time_s is None:
            time_s = np.array([0.0, 5.0, 10.0])
        axial_positions_m = np.array([0.5, 1.5, 2.5])
        gas_species = run_bundle.chemistry.gas_species
        n_species = len(gas_species)

        temperature_profile_k = np.column_stack(
            (
                700.0 + np.arange(time_s.size) * 5.0,
                720.0 + np.arange(time_s.size) * 5.0,
                740.0 + np.arange(time_s.size) * 5.0,
            )
        )
        pressure_profile_pa = np.column_stack(
            (
                105000.0 - np.arange(time_s.size) * 500.0,
                103000.0 - np.arange(time_s.size) * 500.0,
                101000.0 - np.arange(time_s.size) * 500.0,
            )
        )
        p_out_pa = pressure_profile_pa[:, -1].copy()

        gas_mole_fraction = np.zeros((time_s.size, n_species, axial_positions_m.size), dtype=float)
        base_fluxes_by_time = [
            {"CH4": 1.0, "CO": 3.0},
            {"CH4": 2.0, "CO": 2.0},
            {"CH4": 1.0, "CO2": 2.0, "H2": 1.0},
        ]
        fluxes_by_time = [base_fluxes_by_time[min(index, len(base_fluxes_by_time) - 1)] for index in range(time_s.size)]
        gas_flux = np.zeros((time_s.size, n_species, axial_positions_m.size + 1), dtype=float)

        for time_index, species_fluxes in enumerate(fluxes_by_time):
            total_flux = sum(species_fluxes.values())
            for species_index, species_id in enumerate(gas_species):
                flux = float(species_fluxes.get(species_id, 0.0))
                gas_flux[time_index, species_index, -1] = flux
                gas_mole_fraction[time_index, species_index, -1] = 0.0 if total_flux == 0.0 else flux / total_flux

        variables = {
            f"{system_name}.temp_bed": _FakeVariable(
                f"{system_name}.temp_bed",
                time_s,
                temperature_profile_k,
            ),
            f"{system_name}.pres_bed": _FakeVariable(
                f"{system_name}.pres_bed",
                time_s,
                pressure_profile_pa,
            ),
            f"{system_name}.y_gas": _FakeVariable(
                f"{system_name}.y_gas",
                time_s,
                gas_mole_fraction,
            ),
            f"{system_name}.N_gas_face": _FakeVariable(
                f"{system_name}.N_gas_face",
                time_s,
                gas_flux,
            ),
            f"{system_name}.P_out": _FakeVariable(
                f"{system_name}.P_out",
                time_s,
                p_out_pa,
            ),
            f"{system_name}.xval_cells": _FakeVariable(
                f"{system_name}.xval_cells",
                np.array([0.0]),
                axial_positions_m[np.newaxis, :],
            ),
        }

        return RunResult(
            run_bundle=run_bundle,
            output_directory=run_bundle.output_directory,
            success=True,
            reporter=_FakeReporter(variables),
        )

    def test_extract_plot_data_uses_outlet_fluxes_and_profiles(self) -> None:
        run_result = self._build_run_result()

        plot_data = extract_run_result_plot_data(run_result)

        expected_area = np.pi * run_result.run_bundle.run.model.bed_radius_m ** 2
        np.testing.assert_allclose(plot_data.outlet_temperature_k, [740.0, 745.0, 750.0])
        np.testing.assert_allclose(plot_data.outlet_pressure_pa, [101000.0, 100500.0, 100000.0])
        np.testing.assert_allclose(plot_data.outlet_flowrate_mol_s, expected_area * np.array([4.0, 4.0, 4.0]))

        ch4_index = plot_data.gas_species.index("CH4")
        co_index = plot_data.gas_species.index("CO")
        co2_index = plot_data.gas_species.index("CO2")
        h2_index = plot_data.gas_species.index("H2")

        np.testing.assert_allclose(plot_data.outlet_composition[:, ch4_index], [0.25, 0.5, 0.25])
        np.testing.assert_allclose(plot_data.outlet_composition[:, co_index], [0.75, 0.5, 0.0])
        np.testing.assert_allclose(plot_data.outlet_composition[:, co2_index], [0.0, 0.0, 0.5])
        np.testing.assert_allclose(plot_data.outlet_composition[:, h2_index], [0.0, 0.0, 0.25])

    def test_extract_plot_data_collapses_duplicate_report_times(self) -> None:
        run_result = self._build_run_result(time_s=np.array([0.0, 0.0, 5.0, 10.0, 10.0]))

        plot_data = extract_run_result_plot_data(run_result)

        np.testing.assert_allclose(plot_data.time_s, [0.0, 5.0, 10.0])
        np.testing.assert_allclose(plot_data.outlet_temperature_k, [745.0, 750.0, 760.0])
        np.testing.assert_allclose(plot_data.outlet_pressure_pa, [100500.0, 100000.0, 99000.0])

    def test_render_run_result_plots_writes_requested_figures(self) -> None:
        run_result = self._build_run_result()

        with tempfile.TemporaryDirectory() as temp_dir:
            plot_paths = render_run_result_plots(run_result, output_dir=temp_dir)

            self.assertEqual(
                set(plot_paths),
                {"outlet_composition_svg", "outlet_conditions_svg", "temperature_profile_svg"},
            )
            for path in plot_paths.values():
                self.assertTrue(Path(path).exists())


class CliIntegrationTests(unittest.TestCase):
    def test_run_simulation_adds_result_plot_artifacts(self) -> None:
        run_bundle = load_run_bundle("packed_bed/examples/default_case/run.yaml")
        fake_assembly = SimpleNamespace(simulation=object())
        fake_reporter = object()
        base_artifact_paths = {
            "system_graph_svg": Path("C:/tmp/system_graph.svg"),
        }
        result_plot_paths = {
            "outlet_composition_svg": Path("C:/tmp/outlet_composition_vs_time.svg"),
            "outlet_conditions_svg": Path("C:/tmp/outlet_conditions_vs_time.svg"),
            "temperature_profile_svg": Path("C:/tmp/temperature_profile_vs_time.svg"),
        }

        with (
            patch("packed_bed.cli.assemble_simulation", return_value=fake_assembly) as assemble_simulation_mock,
            patch("packed_bed.cli.run_assembled_simulation", return_value=fake_reporter) as run_assembled_mock,
            patch("packed_bed.cli.render_run_result_plots", return_value=result_plot_paths) as render_plots_mock,
        ):
            run_result = run_simulation(run_bundle, artifact_paths=base_artifact_paths)

        assemble_simulation_mock.assert_called_once()
        run_assembled_mock.assert_called_once_with(fake_assembly)
        render_plots_mock.assert_called_once()

        plotted_run_result = render_plots_mock.call_args.args[0]
        self.assertIs(plotted_run_result.reporter, fake_reporter)
        self.assertIs(plotted_run_result.simulation, fake_assembly.simulation)
        self.assertEqual(run_result.artifact_paths, {**base_artifact_paths, **result_plot_paths})


if __name__ == "__main__":
    unittest.main()
