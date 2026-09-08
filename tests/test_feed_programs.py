"""Feed programs conserve species flow before normalizing inlet conditions."""

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest
from pydantic import ValidationError

from packed_bed.config import FeedProgramConfig, PackedBedValidationError, load_case
from packed_bed.programs import (
    NORMAL_MOLAR_DENSITY_MOL_PER_M3, RatioProgram, compile_feed_stream,
)
from test_config import _case_documents, _write_case


AIR_FLOW = 2736.774269930825
STEAM_FLOW = 333.050610370752
SPECIES = ("N2", "H2O")


def _feed_documents():
    documents = _case_documents()
    documents["run.yaml"]["simulation"]["program_mode"] = "feed_stream"
    documents["chemistry.yaml"]["gas_species"] = list(SPECIES)
    documents["program.yaml"] = {
        "feed_stream": {
            "initial": {"flow": AIR_FLOW, "temperature": 773.15,
                        "composition": {"N2": 1.0, "H2O": 0.0}},
            "steps": [
                {"kind": "ramp", "duration_s": 5.0,
                 "target": {"flow": STEAM_FLOW, "temperature": 673.15,
                            "composition": {"N2": 0.0, "H2O": 1.0}}},
                {"kind": "hold", "duration_s": 5.0},
            ],
        },
        "outlet_pressure": {"initial": 100000.0, "steps": []},
    }
    return documents


def _compile(document=None, **kwargs):
    config = FeedProgramConfig.model_validate(document or _feed_documents()["program.yaml"])
    return compile_feed_stream(config.feed_stream, SPECIES, **kwargs)


def _value(program, t):
    return np.asarray(program.value_at(t, smooth_ramp_width_s=1.0))


def test_species_flow_is_bounded_through_smoothed_air_to_steam_transition():
    flow, composition, temperature = _compile()
    for t in np.linspace(-10, 30, 401):
        f, y, temp = (_value(p, t) for p in (flow, composition, temperature))
        assert STEAM_FLOW - 1e-8 <= f <= AIR_FLOW + 1e-8
        assert y.sum() == pytest.approx(1, abs=1e-12)
        assert np.min(y) >= -1e-12
        assert 0 <= f * y[1] <= STEAM_FLOW + 1e-8
        assert 673.15 <= temp <= 773.15
        np.testing.assert_allclose(f * y, _value(composition.numerator, t), atol=1e-10)
    f, y, temp = (_value(p, 2.5) for p in (flow, composition, temperature))
    assert f * y[1] == pytest.approx(STEAM_FLOW / 2)
    assert f * y[0] == pytest.approx(AIR_FLOW / 2)
    assert y[1] == pytest.approx(0.1084917294494119)
    assert temp == pytest.approx(762.3008270550588)


@pytest.mark.parametrize("target,expected", [
    ({"flow": 20.0}, (15.0, (.75, .25), 500.0)),
    ({"temperature": 700.0}, (10.0, (.75, .25), 600.0)),
    ({"composition": {"N2": 0.0, "H2O": 1.0}}, (10.0, (.375, .625), 500.0)),
    ({"flow": 20.0, "temperature": 700.0, "composition": {"N2": 0.0, "H2O": 1.0}},
     (15.0, (.25, .75), 19000.0 / 30)),
])
def test_partial_targets_retain_other_fields(target, expected):
    document = _feed_documents()["program.yaml"]
    document["feed_stream"]["initial"] = {
        "flow": 10.0, "temperature": 500.0, "composition": {"N2": .75, "H2O": .25}}
    document["feed_stream"]["steps"][0]["target"] = target
    for program, value in zip(_compile(document), expected, strict=True):
        np.testing.assert_allclose(_value(program, 2.5), value)


@pytest.mark.parametrize("steps", [[], [{"kind": "hold", "duration_s": 10.0}]])
def test_constant_feed_and_holds(steps):
    document = _feed_documents()["program.yaml"]
    document["feed_stream"]["steps"] = steps
    for program, expected in zip(_compile(document), [AIR_FLOW, (1.0, 0.0), 773.15], strict=True):
        np.testing.assert_allclose(program.initial_value, expected)
        np.testing.assert_allclose(_value(program, 100), expected)


def test_partial_targets_carry_forward_across_repetitions():
    document = _feed_documents()["program.yaml"]
    document["feed_stream"] = {
        "initial": {"flow": 10.0, "temperature": 500.0, "composition": {"N2": 1.0, "H2O": 0.0}},
        "steps": [
            {"kind": "ramp", "duration_s": 2.0, "target": {"flow": 20.0}},
            {"kind": "hold", "duration_s": 1.0},
            {"kind": "ramp", "duration_s": 2.0,
             "target": {"flow": 10.0, "temperature": 700.0, "composition": {"N2": 0.0, "H2O": 1.0}}},
        ],
    }
    original = deepcopy(document)
    flow, composition, temperature = _compile(document, repeat=True, time_horizon=12.0)
    assert flow.duration_s == composition.duration_s == temperature.duration_s == 12.0
    second_ramp = 3
    assert flow.segments[second_ramp].start_time == 5.0
    assert temperature.numerator.segments[second_ramp].start_value == 10 * 700
    assert temperature.numerator.segments[second_ramp].end_value == 20 * 700
    assert composition.numerator.segments[second_ramp].end_value == (0.0, 20.0)
    assert document == original


def test_clipped_ramp_interpolates_flows_before_dividing():
    flow, composition, temperature = _compile(time_horizon=2.5)
    f = flow.segments[-1].end_value
    assert f == pytest.approx((AIR_FLOW + STEAM_FLOW) / 2)
    assert composition.numerator.segments[-1].end_value == pytest.approx((AIR_FLOW / 2, STEAM_FLOW / 2))
    assert temperature.numerator.segments[-1].end_value / f == pytest.approx(762.3008270550588)
    assert flow.duration_s == composition.duration_s == temperature.duration_s == 2.5


def test_feed_ghsv_conversion_scales_all_molar_quantities(tmp_path):
    documents = _feed_documents()
    documents["program.yaml"]["feed_stream"]["basis"] = "ghsv_per_h"
    case = load_case(_write_case(tmp_path, documents))
    scale = np.pi * .01**2 * NORMAL_MOLAR_DENSITY_MOL_PER_M3 / 3600.0
    mol_programs = _compile()
    for t in (0, 2.5, 10):
        assert _value(case.inlet_flow_program, t) == pytest.approx(_value(mol_programs[0], t) * scale)
        np.testing.assert_allclose(_value(case.inlet_composition_program, t), _value(mol_programs[1], t))
        assert _value(case.inlet_temperature_program, t) == pytest.approx(_value(mol_programs[2], t))


def test_feed_mode_loads_with_independent_pressure_schedule(tmp_path):
    documents = _feed_documents()
    documents["program.yaml"]["outlet_pressure"]["steps"] = [
        {"kind": "hold", "duration_s": 8.0},
        {"kind": "ramp", "duration_s": 2.0, "target": 120000.0},
    ]
    case = load_case(_write_case(tmp_path, documents))
    assert isinstance(case.program, FeedProgramConfig)
    assert isinstance(case.inlet_composition_program, RatioProgram)
    assert isinstance(case.inlet_temperature_program, RatioProgram)
    assert case.inlet_composition_program.denominator is case.inlet_flow_program
    assert case.outlet_pressure_program.segments[-1].start_time == 8.0
    assert _value(case.outlet_pressure_program, 9) == pytest.approx(110000.0)


@pytest.mark.parametrize("target", [
    {}, {"flow": 0.0}, {"flow": -1.0}, {"temperature": 0.0},
    {"temperature": float("nan")}, {"flow": float("inf")},
    {"composition": {"N2": .5, "H2O": .4}},
    {"composition": {"N2": -1.0, "H2O": 2.0}}, {"temp": 500.0},
])
def test_invalid_feed_targets_are_rejected(target):
    document = _feed_documents()["program.yaml"]
    document["feed_stream"]["steps"][0]["target"] = target
    with pytest.raises(ValidationError):
        FeedProgramConfig.model_validate(document)


@pytest.mark.parametrize("change,match", [
    ("wrong_mode", "program.inlet_flow"),
    ("mixed", "program.inlet_flow"),
    ("initial_species", "initial.composition species mismatch"),
    ("target_species", "target.composition species mismatch"),
    ("duration", "feed_stream.steps must sum"),
    ("pressure_duration", "outlet_pressure.steps must sum"),
    ("zero_initial_flow", "feed_stream.initial.flow"),
])
def test_feed_case_validation(tmp_path, change, match):
    documents = _feed_documents()
    program = documents["program.yaml"]
    if change == "wrong_mode":
        documents["run.yaml"]["simulation"].pop("program_mode")
    elif change == "mixed":
        program["inlet_flow"] = {"initial": 1.0}
    elif change == "initial_species":
        program["feed_stream"]["initial"]["composition"] = {"N2": 1.0}
    elif change == "target_species":
        program["feed_stream"]["steps"][0]["target"]["composition"] = {"O2": 0.0, "H2O": 1.0}
    elif change == "duration":
        program["feed_stream"]["steps"][1]["duration_s"] = 6.0
    elif change == "pressure_duration":
        program["outlet_pressure"]["steps"] = [{"kind": "hold", "duration_s": 9.0}]
    elif change == "zero_initial_flow":
        program["feed_stream"]["initial"]["flow"] = 0.0
    with pytest.raises(PackedBedValidationError, match=match):
        load_case(_write_case(tmp_path, documents))


def test_separate_channel_mode_remains_default_and_rejects_feed_mode_mismatch(tmp_path):
    documents = _case_documents()
    case = load_case(_write_case(tmp_path, documents))
    assert case.run.simulation.program_mode == "separate_channels"
    documents["run.yaml"]["simulation"]["program_mode"] = "feed_stream"
    with pytest.raises(PackedBedValidationError, match="program.feed_stream"):
        load_case(_write_case(tmp_path, documents))


def test_feed_initialization_plots_and_manifest_use_derived_conditions(tmp_path):
    from packed_bed.artifacts import _series_from_smoothed_program, _smoothed_program_sample_times, render_operating_program
    from packed_bed.initialization import calculate_initial_state
    from packed_bed.properties import PROPERTY_REGISTRY
    from packed_bed.reports import RunResult, write_run_manifest

    documents = _feed_documents()
    # A laboratory-scale inert case keeps the pressure initialization modest.
    feed = documents["program.yaml"]["feed_stream"]
    feed["initial"]["flow"] = .008
    feed["steps"][0]["target"]["flow"] = .002
    case = load_case(_write_case(tmp_path, documents))
    state = calculate_initial_state(case, PROPERTY_REGISTRY)
    assert state.inlet_flow_mol_s == pytest.approx(_value(case.inlet_flow_program, 0))
    assert state.inlet_temperature_k == pytest.approx(_value(case.inlet_temperature_program, 0))
    np.testing.assert_allclose(state.inlet_composition, _value(case.inlet_composition_program, 0))
    assert state.inlet_temperature_k != pytest.approx(feed["initial"]["temperature"])

    programs = (case.inlet_flow_program, case.inlet_composition_program,
                case.inlet_temperature_program, case.outlet_pressure_program)
    times = _smoothed_program_sample_times(programs, final_time=10, smooth_ramp_width_s=1)
    assert {0, 5, 10} <= set(times)
    fractions = _series_from_smoothed_program(programs[1], times, smooth_ramp_width_s=1)
    np.testing.assert_allclose(fractions.sum(axis=1), 1, atol=1e-12)
    artifacts = render_operating_program(case, tmp_path / "artifacts")
    assert artifacts["operating_program_svg"].is_file()
    manifest = json.loads(write_run_manifest(RunResult(
        case=case, output_directory=tmp_path, artifact_paths=artifacts)).read_text())
    compiled = manifest["configuration"]["compiled_programs"]
    assert manifest["configuration"]["run"]["simulation"]["program_mode"] == "feed_stream"
    assert manifest["configuration"]["program"]["feed_stream"]["initial"]["flow"] == .008
    assert compiled["inlet_composition"]["denominator"] == compiled["inlet_flow"]
    assert compiled["inlet_temperature"]["numerator"]["initial_value"] == pytest.approx(.008 * 773.15)


def _small_feed_case(directory, backend="daetools", solver="superlu"):
    directory.mkdir(parents=True, exist_ok=True)
    documents = _feed_documents()
    documents["run.yaml"]["simulation"]["reporting_interval_s"] = .5
    documents["run.yaml"]["solver"].update(
        backend=backend, name=solver, threads=1, relative_tolerance=1e-7,
        suppress_algebraic_errors=True, max_nonlinear_iterations=12,
    )
    documents["run.yaml"]["outputs"]["requested_reports"] = [
        "temperature", "pressure", "gas_mole_fraction", "gas_flux", "heat_balance", "mass_balance"]
    feed = documents["program.yaml"]["feed_stream"]
    feed["initial"]["flow"] = .008
    feed["steps"][0]["target"]["flow"] = .002
    documents["program.yaml"]["outlet_pressure"]["steps"] = [
        {"kind": "hold", "duration_s": 8.0},
        {"kind": "ramp", "duration_s": 2.0, "target": 102000.0},
    ]
    return load_case(_write_case(directory, documents))


def test_feed_program_drives_daetools_inlet(tmp_path):
    pytest.importorskip("daetools.pyDAE")
    from packed_bed.reports import load_dataset
    from packed_bed.simulation import run_case

    case = _small_feed_case(tmp_path)
    result = run_case(case)
    dataset = load_dataset(result.results_path)
    assert result.status == "success"
    for name, program, tolerance in (
        ("inlet_flow", case.inlet_flow_program, 1e-10),
        ("inlet_composition", case.inlet_composition_program, 1e-8),
        ("inlet_temperature", case.inlet_temperature_program, 1e-5),
        ("outlet_pressure", case.outlet_pressure_program, 1e-3),
    ):
        expected = np.array([_value(program, float(t)) for t in dataset.time.values])
        np.testing.assert_allclose(dataset[name], expected, rtol=0, atol=tolerance)
    steam = dataset.inlet_flow * dataset.inlet_composition.sel(gas_species="H2O")
    assert float(steam.min()) >= 0
    assert float(steam.max()) <= .002 + 1e-10
    assert float(abs(dataset.temperature - dataset.temperature.isel(time=0)).max()) > .1
    assert all(np.isfinite(variable).all() for variable in dataset.data_vars.values())


@pytest.mark.parametrize("linear_solver", ("superlu", "band"))
def test_feed_stream_matches_compiled_backend(tmp_path, monkeypatch, linear_solver):
    pytest.importorskip("daetools.pyDAE")
    pytest.importorskip("sksundae")
    from packed_bed.compiled.compiler import find_toolchain
    from packed_bed.compiled.runtime import check_runtime
    from packed_bed.reports import load_dataset
    from packed_bed.simulation import run_case

    try:
        find_toolchain()
        check_runtime()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    monkeypatch.setenv("PACKED_BED_COMPILED_CACHE", str(tmp_path / "cache"))
    reference = run_case(_small_feed_case(tmp_path / "reference"))
    compiled = run_case(_small_feed_case(tmp_path / "compiled", "compiled", linear_solver))
    expected = load_dataset(reference.results_path)
    actual = load_dataset(compiled.results_path)
    assert reference.status == compiled.status == "success"
    assert set(actual.data_vars) == set(expected.data_vars)
    for name in ("inlet_flow", "inlet_temperature", "inlet_composition", "outlet_pressure"):
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(actual.temperature, expected.temperature, rtol=0, atol=.01)
    np.testing.assert_allclose(actual.gas_mole_fraction, expected.gas_mole_fraction, rtol=0, atol=1e-5)
    steam = actual.inlet_flow * actual.inlet_composition.sel(gas_species="H2O")
    assert float(steam.min()) >= 0
    assert float(steam.max()) <= .002 + 1e-10
    assert all(np.isfinite(variable).all() for variable in actual.data_vars.values())
