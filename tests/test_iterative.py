from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from packed_bed.config import load_case
from packed_bed.programs import compile_program_channels

pytestmark = pytest.mark.skipif(
    find_spec("daetools") is None, reason="DAETools is not installed"
)


def _reactive_case(tmp_path, solver_name):
    root = Path(__file__).resolve().parents[1]
    case = load_case(root / "packed_bed/examples/default_case/run.yaml")
    # Include the inlet ramp and a measurable reaction/temperature change.
    horizon = 100.0
    model = case.run.model.model_copy(update={"axial_cells": 3})
    simulation = case.run.simulation.model_copy(update={"time_horizon_s": horizon})
    solver = case.run.solver.model_copy(
        update={
            "name": solver_name,
            "threads": 1,
            "relative_tolerance": 1e-6,
            "max_nonlinear_iterations": 12,
            "nonlinear_convergence_coefficient": 0.03,
        }
    )
    output = tmp_path / solver_name
    outputs = case.run.outputs.model_copy(
        update={
            "directory": str(output),
            "artifacts_directory": str(output / "artifacts"),
            "requested_plots": (),
        }
    )
    programs = compile_program_channels(
        case.program,
        case.chemistry.gas_species,
        model,
        repeat=True,
        time_horizon=horizon,
    )
    return replace(
        case,
        run=case.run.model_copy(
            update={
                "simulation": simulation,
                "model": model,
                "solver": solver,
                "outputs": outputs,
            }
        ),
        inlet_flow_program=programs[0],
        inlet_composition_program=programs[1],
        inlet_temperature_program=programs[2],
        outlet_pressure_program=programs[3],
    )


def test_gmres_reactive_trajectory_and_actual_krylov_iterations(tmp_path):
    from packed_bed.reports import load_dataset
    from packed_bed.simulation import run_case

    direct = run_case(_reactive_case(tmp_path, "superlu"))
    iterative = run_case(_reactive_case(tmp_path, "sundials_gmres_ifpack"))
    expected = load_dataset(direct.results_path)
    actual = load_dataset(iterative.results_path)
    assert actual.time.values[-1] == 100.0
    assert (
        float(
            abs(
                actual.temperature.isel(time=-1) - actual.temperature.isel(time=0)
            ).max()
        )
        > 1
    )
    np.testing.assert_allclose(
        actual.temperature, expected.temperature, rtol=0, atol=0.01
    )
    np.testing.assert_allclose(
        actual.gas_mole_fraction, expected.gas_mole_fraction, rtol=0, atol=1e-4
    )
    assert iterative.solver_stats["integrator"]["NumLinIters"] > 0
    assert iterative.solver_stats["integrator"]["NumPrecSolves"] > 0
    assert float(abs(actual.heat_balance_error).max()) < 0.01


def test_iterative_order_places_a_dependency_on_every_diagonal(tmp_path):
    from daetools.pyDAE import daeIDAS, daeNoOpDataReporter, daePythonStdOutLog

    from packed_bed.incidence_matrix import collect_solver_incidence_matrix
    from packed_bed.properties import PROPERTY_REGISTRY
    from packed_bed.simulation import (
        PackedBedSimulation,
        configure_idas,
        configure_threads,
    )

    case = _reactive_case(tmp_path, "sundials_gmres_ifpack")
    configure_threads(1)
    configure_idas(case.run.solver)
    simulation = PackedBedSimulation(case, PROPERTY_REGISTRY)
    simulation.TimeHorizon = 100
    simulation.ReportingInterval = 1
    solver = daeIDAS()
    log = daePythonStdOutLog()
    log.Enabled = False
    simulation.Initialize(solver, daeNoOpDataReporter(), log)
    try:
        matrix = collect_solver_incidence_matrix(simulation.model)
        diagonal = {
            e.row_index for e in matrix.entries if e.row_index == e.column_index
        }
        assert diagonal == set(range(matrix.row_count))
    finally:
        simulation.Finalize()


def test_iterative_example_loads_with_tight_settings():
    root = Path(__file__).resolve().parents[1]
    case = load_case(root / "packed_bed/examples/default_case/run_iterative.yaml")
    assert case.run.solver.name == "sundials_gmres_ifpack"
    assert case.run.solver.relative_tolerance == 1e-6
    assert case.run.simulation.time_horizon_s == 12000
