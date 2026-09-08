from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from packed_bed.config import load_case
from packed_bed.programs import compile_program_channels


def _small_reactive_case(tmp_path, backend, gas_voidage_mode="bed_and_particle"):
    root = Path(__file__).resolve().parents[2]
    case = load_case(root / "packed_bed/examples/default_case/run.yaml")
    simulation = case.run.simulation.model_copy(
        update={"time_horizon_s": 100.0, "reporting_interval_s": 0.7}
    )
    solver = case.run.solver.model_copy(
        update={
            "backend": backend,
            "threads": 1,
            "relative_tolerance": 1e-6,
            "maximum_order": 5,
            "max_nonlinear_iterations": 4,
            "nonlinear_convergence_coefficient": 0.03,
        }
    )
    outputs = case.run.outputs.model_copy(
        update={
            "directory": str(tmp_path / backend),
            "artifacts_directory": str(tmp_path / backend / "artifacts"),
            "requested_plots": (),
        }
    )
    programs = compile_program_channels(
        case.program,
        case.chemistry.gas_species,
        case.run.model,
        repeat=True,
        time_horizon=100.0,
    )
    return replace(
        case,
        run=case.run.model_copy(
            update={
                "simulation": simulation,
                "solver": solver,
                "outputs": outputs,
                "model": case.run.model.model_copy(update={"axial_cells": 3, "gas_voidage_mode": gas_voidage_mode}),
            }
        ),
        inlet_flow_program=programs[0],
        inlet_composition_program=programs[1],
        inlet_temperature_program=programs[2],
        outlet_pressure_program=programs[3],
    )


@pytest.mark.parametrize("linear_solver, gas_voidage_mode", (("superlu", "bed_and_particle"), ("band", "bed_only")))
def test_reactive_run_reports_cache_and_tight_reference(
    native_tools, tmp_path, monkeypatch, linear_solver, gas_voidage_mode
):
    from packed_bed.reports import load_dataset
    from packed_bed.simulation import run_case

    monkeypatch.setenv("PACKED_BED_COMPILED_CACHE", str(tmp_path / "cache"))
    reference = run_case(_small_reactive_case(tmp_path, "daetools", gas_voidage_mode))
    case = _small_reactive_case(tmp_path, "compiled", gas_voidage_mode)
    case = replace(
        case,
        run=case.run.model_copy(
            update={
                "solver": case.run.solver.model_copy(
                    update={
                        "name": linear_solver,
                        "step_growth_threshold": 1.25,
                        "nonlinear_refresh_interval": 4,
                    }
                ),
            }
        ),
    )
    first = run_case(case, retain_reporter=True)
    second = run_case(case, retain_reporter=True)
    assert not first.solver_stats["cache_hit"]
    assert second.solver_stats["cache_hit"]
    assert (
        first.solver_stats["fixed_states_removed"] == 7
    )  # He, inert solid, heat loss.
    assert first.solver_stats["linear_solver"] == linear_solver
    assert first.solver_stats["step_growth_threshold"] == 1.25
    assert first.solver_stats["nonlinear_refresh_interval"] == 4
    assert first.solver_stats["nonlinear_solver_implementation"] == (
        "Newton with early Jacobian refresh"
    )
    assert first.solver_stats["kernel_layout"] == "shared cells"
    assert (
        second.solver_stats["reduced_unknowns"]
        < second.solver_stats["original_unknowns"]
    )
    expected = load_dataset(reference.results_path)
    actual = load_dataset(second.results_path)
    assert (
        float(
            abs(
                expected.temperature.isel(time=-1) - expected.temperature.isel(time=0)
            ).max()
        )
        > 1.0
    )
    assert actual.time.values[-1] == 100.0  # Include a nonintegral reporting endpoint.
    assert set(actual.data_vars) == set(expected.data_vars)
    np.testing.assert_allclose(
        actual.temperature, expected.temperature, rtol=0, atol=0.01
    )
    np.testing.assert_allclose(
        actual.gas_mole_fraction, expected.gas_mole_fraction, rtol=0, atol=1e-4
    )
    assert np.max(abs(actual.heat_balance_error)) < 0.01
    assert (
        first.reporter.Process.dictVariables.keys()
        == second.reporter.Process.dictVariables.keys()
    )
    for key, variable in first.reporter.Process.dictVariables.items():
        np.testing.assert_array_equal(
            variable.Values, second.reporter.Process.dictVariables[key].Values
        )
    manifest = json.loads(second.manifest_path.read_text())
    assert manifest["solver_stats"]["cache_hit"]
    from daetools.dae_plotter.data_receiver_io import dataReceiverProcess

    plotter_process = dataReceiverProcess(second.reporter.Process)
    assert set(plotter_process.dictVariables) == set(
        second.reporter.Process.dictVariables
    )
    # Same shape, changed physical constant: old kernels must not be reused.
    changed = replace(
        case,
        run=case.run.model_copy(
            update={
                "model": case.run.model.model_copy(
                    update={"heat_transfer_coefficient_w_per_m2_k": 10.0}
                ),
            }
        ),
    )
    third = run_case(changed)
    assert third.solver_stats["kernel_sha256"] != first.solver_stats["kernel_sha256"]
    assert third.solver_stats["fixed_states_removed"] == 6  # Heat loss now evolves.


def test_compiler_update_invalidates_the_model_cache(native_tools, tmp_path, monkeypatch):
    from packed_bed.compiled import compiler
    from packed_bed.simulation import run_case

    monkeypatch.setenv("PACKED_BED_COMPILED_CACHE", str(tmp_path / "cache"))
    case = _small_reactive_case(tmp_path, "compiled")
    first = run_case(case, retain_reporter=True)
    cached = run_case(case)
    assert cached.solver_stats["cache_hit"]

    toolchain, version = compiler.compiler_identity()
    monkeypatch.setattr(compiler, "compiler_identity", lambda: (toolchain, version + "-updated"))
    updated = run_case(case, retain_reporter=True)
    assert not updated.solver_stats["cache_hit"]
    assert updated.solver_stats["kernel_sha256"] != first.solver_stats["kernel_sha256"]
    assert updated.solver_stats["compiler_version"] == version + "-updated"
    for name, variable in first.reporter.Process.dictVariables.items():
        np.testing.assert_array_equal(variable.Values, updated.reporter.Process.dictVariables[name].Values)


def test_concentration_tolerance_is_scoped_to_each_case(native_tools, tmp_path):
    from packed_bed.model import molar_conc_sol_type, molar_conc_type
    from packed_bed.properties import PROPERTY_REGISTRY
    from packed_bed.simulation import PackedBedSimulation

    original = _small_reactive_case(tmp_path, "daetools")
    tight = replace(original, run=original.run.model_copy(update={
        "solver": original.run.solver.model_copy(update={"concentration_absolute_tolerance": 1e-11}),
    }))
    first = PackedBedSimulation(tight, PROPERTY_REGISTRY)
    second = PackedBedSimulation(original, PROPERTY_REGISTRY)
    for name in ("c_gas", "c_sol", "ct_gas", "ct_sol"):
        assert getattr(first.model, name).VariableType.AbsoluteTolerance == 1e-11
        assert getattr(second.model, name).VariableType.AbsoluteTolerance == 1e-5
    assert molar_conc_type.AbsoluteTolerance == molar_conc_sol_type.AbsoluteTolerance == 1e-5


@pytest.mark.parametrize("case_name,scalar", (
    ("copper_sio2_san_pio", True),
    ("copper_al2o3_san_pio", False),
    ("iron_he_mixed_feed", False),
    ("mixed_solid_zones", False),
))
def test_trace_chemistry_and_heterogeneous_solids_match_reference(
    native_tools, tmp_path, monkeypatch, case_name, scalar
):
    import packed_bed.compiled as compiled
    from packed_bed.compiled import band
    from packed_bed.reports import _scheduled_time, load_dataset
    from packed_bed.simulation import run_case
    from tools.benchmark_compiled import prepare_case
    from tools.performance_cases import cases

    monkeypatch.setenv("PACKED_BED_COMPILED_CACHE", str(tmp_path / "cache"))
    if scalar:
        # Exercise a complete reactor without AVX2, including the CPU fallback
        # for the optional SLEEF exponential implementation.
        monkeypatch.setattr(compiled, "supports_avx2", lambda: False)
        monkeypatch.setattr(band, "supports_avx2", lambda: False)
    documents, _ = cases()[case_name]
    reports = {}
    for profile in ("reference", "compiled"):
        run_file, _ = prepare_case(documents, tmp_path / profile, profile, 1e-6, 1e-11)
        case = load_case(run_file)
        result = run_case(case)
        reports[profile] = load_dataset(result.results_path)
        np.testing.assert_allclose(reports[profile].time, _scheduled_time(case), rtol=0, atol=1e-12)
        assert all(bool(np.isfinite(v).all()) for v in reports[profile].data_vars.values())
    expected, actual = reports["reference"], reports["compiled"]
    assert set(actual.data_vars) == set(expected.data_vars)
    np.testing.assert_allclose(actual.temperature, expected.temperature, rtol=0, atol=.02)
    np.testing.assert_allclose(actual.gas_mole_fraction, expected.gas_mole_fraction, rtol=0, atol=1e-4)
    np.testing.assert_allclose(actual.solid_mole_fraction, expected.solid_mole_fraction, rtol=0, atol=1e-4)
    assert float(actual.gas_mole_fraction.min()) > -1e-5
    assert float(abs(actual.temperature - actual.temperature.isel(time=0)).max()) > 1
    assert float(abs(actual.solid_mole_fraction - actual.solid_mole_fraction.isel(time=0)).max()) > 1e-3
    if "reaction_rate" in actual:
        assert bool((abs(actual.reaction_rate).max(dim=("time", "x_cell")) > 1e-9).all())
    if scalar:
        assert result.solver_stats["residual_lanes"] == 1


def test_later_helium_feed_and_changed_inert_initial_state(
    native_tools,
    tmp_path,
    monkeypatch,
):
    from packed_bed.config import ProgramConfig
    from packed_bed.reports import load_dataset
    from packed_bed.simulation import run_case

    monkeypatch.setenv("PACKED_BED_COMPILED_CACHE", str(tmp_path / "cache"))
    case = _small_reactive_case(tmp_path, "compiled")
    program_data = case.program.model_dump(mode="python")
    air = dict(program_data["inlet_composition"]["steps"][0]["target"])
    air.update(He=0.02, N2=air["N2"] - 0.02)
    program_data["inlet_composition"]["steps"] = [
        {"kind": "hold", "duration_s": 20.0},
        {"kind": "ramp", "duration_s": 5.0, "target": air},
    ]
    program = ProgramConfig.model_validate(program_data)
    channels = compile_program_channels(
        program,
        case.chemistry.gas_species,
        case.run.model,
        repeat=False,
        time_horizon=100.0,
    )
    case = replace(
        case,
        program=program,
        inlet_flow_program=channels[0],
        inlet_composition_program=channels[1],
        inlet_temperature_program=channels[2],
        outlet_pressure_program=channels[3],
        run=case.run.model_copy(
            update={
                "solver": case.run.solver.model_copy(update={"name": "band"}),
                "model": case.run.model.model_copy(update={"axial_cells": 5}),
            }
        ),
    )
    first = run_case(case)
    actual = load_dataset(first.results_path)
    helium = actual.gas_mole_fraction.sel(gas_species="He")
    # The model's smooth ramps have a small tail before the declared start.
    assert program.inlet_composition.initial["He"] == 0.0
    assert float(abs(helium.isel(time=0)).max()) < 1e-4
    assert float(helium.isel(time=-1).max()) > 0.001
    assert first.solver_stats["fixed_states_removed"] == 6
    reference_case = replace(
        case,
        run=case.run.model_copy(
            update={
                "solver": case.run.solver.model_copy(
                    update={"backend": "daetools", "name": "superlu"}
                ),
            }
        ),
    )
    reference = load_dataset(run_case(reference_case).results_path)
    np.testing.assert_allclose(
        actual.gas_mole_fraction, reference.gas_mole_fraction, rtol=0, atol=1e-4
    )
    profile = case.solids.initial_profile
    zone = profile.zones[0]
    changed_zone = zone.model_copy(
        update={"values": {**zone.values, "CaAl2O4": 5000.0}}
    )
    changed = replace(
        case,
        solids=case.solids.model_copy(
            update={
                "initial_profile": profile.model_copy(
                    update={"zones": (changed_zone,)}
                ),
            }
        ),
    )
    second = run_case(changed)
    assert not second.solver_stats["cache_hit"]
    assert second.solver_stats["kernel_sha256"] != first.solver_stats["kernel_sha256"]
    changed_report = load_dataset(second.results_path)
    expected_inert_fraction = 5000.0 / (5000.0 + zone.values["Ni"])
    np.testing.assert_allclose(
        changed_report.solid_mole_fraction.sel(solid_species="CaAl2O4"),
        expected_inert_fraction,
        rtol=0,
        atol=1e-12,
    )
