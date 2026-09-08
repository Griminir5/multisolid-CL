from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys

from pydantic import ValidationError
import pytest
import yaml

import packed_bed.config as config
from packed_bed.config import Case, PackedBedValidationError, ProgramConfig, load_case
from packed_bed.programs import (
    NORMAL_MOLAR_DENSITY_MOL_PER_M3,
    CompiledProgram,
    compile_composition_channel,
    compile_scalar_channel,
)


def _case_documents(
    *,
    flow_duration_s: float | None = None,
    composition_duration_s: float | None = None,
) -> dict[str, dict]:
    flow_steps = (
        []
        if flow_duration_s is None
        else [{"kind": "hold", "duration_s": flow_duration_s}]
    )
    composition_steps = (
        []
        if composition_duration_s is None
        else [{"kind": "hold", "duration_s": composition_duration_s}]
    )
    return {
        "run.yaml": {
            "references": {
                "chemistry_file": "chemistry.yaml",
                "program_file": "program.yaml",
                "solids_file": "solids.yaml",
            },
            "simulation": {
                "system_name": "test",
                "time_horizon_s": 10.0,
                "reporting_interval_s": 1.0,
                "repeat_program": False,
                "interior_flow_mode": "forward_only",
                "mass_scheme": "weno3",
                "heat_scheme": "weno3",
                "report_time_derivatives": False,
            },
            "model": {
                "bed_length_m": 1.0,
                "bed_radius_m": 0.01,
                "axial_cells": 3,
                "ambient_temperature_k": 300.0,
                "heat_transfer_coefficient_w_per_m2_k": 0.0,
            },
            "solver": {"relative_tolerance": 1.0e-5},
            "outputs": {
                "directory": "output",
                "artifacts_directory": "output/artifacts",
                "requested_plots": [],
                "requested_reports": [],
            },
        },
        "chemistry.yaml": {
            "gas_species": ["N2"],
            "reaction_families": [],
            "reaction_ids": [],
        },
        "program.yaml": {
            "inlet_flow": {"initial": 1.0, "steps": flow_steps},
            "inlet_temperature": {"initial": 300.0, "steps": []},
            "outlet_pressure": {"initial": 100000.0, "steps": []},
            "inlet_composition": {
                "initial": {"N2": 1.0},
                "steps": composition_steps,
            },
        },
        "solids.yaml": {
            "solid_species": ["Ni"],
            "initial_profile": {
                "basis": "bed",
                "zones": [
                    {
                        "x_start_m": 0.0,
                        "x_end_m": 1.0,
                        "e_b": 0.4,
                        "e_p": 0.5,
                        "d_p": 0.001,
                        "values": {"Ni": 1.0},
                    }
                ],
            },
        },
    }


def _write_case(tmp_path: Path, documents: dict[str, dict] | None = None) -> Path:
    for filename, document in (documents or _case_documents()).items():
        (tmp_path / filename).write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
    return tmp_path / "run.yaml"


def test_load_case_returns_one_resolved_case_with_compiled_programs(
    tmp_path: Path,
) -> None:
    run_path = _write_case(
        tmp_path,
        _case_documents(flow_duration_s=10.0, composition_duration_s=10.0),
    )

    case = load_case(run_path)

    assert isinstance(case, Case)
    assert isinstance(case.inlet_flow_program, CompiledProgram)
    assert case.inlet_flow_program.duration_s == 10.0
    assert case.inlet_composition_program.initial_value == (1.0,)
    assert case.program.inlet_flow.initial == case.inlet_flow_program.initial_value
    assert case.output_directory == (tmp_path / "output").resolve()


def test_interior_flow_mode_defaults_to_forward_only_and_accepts_reversible(
    tmp_path: Path,
) -> None:
    documents = _case_documents()
    documents["run.yaml"]["simulation"].pop("interior_flow_mode")
    default_directory = tmp_path / "default"
    default_directory.mkdir()
    default_case = load_case(_write_case(default_directory, documents))

    documents = _case_documents()
    documents["run.yaml"]["simulation"]["interior_flow_mode"] = "reversible"
    reversible_directory = tmp_path / "reversible"
    reversible_directory.mkdir()
    reversible_case = load_case(_write_case(reversible_directory, documents))

    assert default_case.run.simulation.interior_flow_mode == "forward_only"
    assert reversible_case.run.simulation.interior_flow_mode == "reversible"


def test_solver_controls_default_to_daetools_values_and_accept_tuning(
    tmp_path: Path,
) -> None:
    default_directory = tmp_path / "default"
    default_directory.mkdir()
    default_case = load_case(_write_case(default_directory, _case_documents()))

    assert default_case.run.solver.model_dump() == {
        "backend": "daetools",
        "name": "trilinos_klu",
        "threads": 0,
        "relative_tolerance": 1.0e-5,
        "concentration_absolute_tolerance": 1.0e-5,
        "suppress_algebraic_errors": False,
        "max_nonlinear_iterations": 4,
        "nonlinear_convergence_coefficient": 0.33,
        "maximum_order": 5,
        "scale_residuals": False,
        "step_growth_threshold": 2.0,
        "nonlinear_refresh_interval": 0,
        "vector_exponentials": False,
        "band_reciprocals": False,
    }

    documents = _case_documents()
    documents["run.yaml"]["solver"].update(
        name="superlu",
        threads=2,
        relative_tolerance=1.0e-3,
        concentration_absolute_tolerance=1.0e-11,
        suppress_algebraic_errors=True,
        max_nonlinear_iterations=12,
        nonlinear_convergence_coefficient=1.0,
    )
    tuned_directory = tmp_path / "tuned"
    tuned_directory.mkdir()
    tuned_case = load_case(_write_case(tuned_directory, documents))

    assert tuned_case.run.solver.model_dump() == {
        "backend": "daetools",
        "name": "superlu",
        "threads": 2,
        "relative_tolerance": 1.0e-3,
        "concentration_absolute_tolerance": 1.0e-11,
        "suppress_algebraic_errors": True,
        "max_nonlinear_iterations": 12,
        "nonlinear_convergence_coefficient": 1.0,
        "maximum_order": 5,
        "scale_residuals": False,
        "step_growth_threshold": 2.0,
        "nonlinear_refresh_interval": 0,
        "vector_exponentials": False,
        "band_reciprocals": False,
    }


@pytest.mark.parametrize(
    "invalid",
    (
        "solver",
        "derivatives",
        "incidence",
        "order",
        "scaling",
        "band_backend",
        "step_growth_backend",
        "nonlinear_refresh_backend",
        "vector_exp_backend",
        "reciprocal_backend",
        "reciprocal_sparse",
    ),
)
def test_compiled_backend_rejects_unsupported_configuration(tmp_path, invalid):
    documents = _case_documents()
    run = documents["run.yaml"]
    run["solver"].update(backend="compiled", name="superlu")
    if invalid == "solver":
        run["solver"]["name"] = "trilinos_klu"
    elif invalid == "derivatives":
        run["simulation"]["report_time_derivatives"] = True
    elif invalid == "incidence":
        run["outputs"]["solver_incidence_matrix"] = True
    elif invalid == "scaling":
        run["solver"].update(backend="daetools", scale_residuals=True)
    elif invalid == "band_backend":
        run["solver"].update(backend="daetools", name="band")
    elif invalid == "step_growth_backend":
        run["solver"].update(backend="daetools", step_growth_threshold=1.25)
    elif invalid == "nonlinear_refresh_backend":
        run["solver"].update(backend="daetools", nonlinear_refresh_interval=4)
    elif invalid == "vector_exp_backend":
        run["solver"].update(backend="daetools", vector_exponentials=True)
    elif invalid == "reciprocal_backend":
        run["solver"].update(backend="daetools", band_reciprocals=True)
    elif invalid == "reciprocal_sparse":
        run["solver"].update(backend="compiled", name="superlu", band_reciprocals=True)
    else:
        run["solver"]["maximum_order"] = 6
    with pytest.raises(PackedBedValidationError):
        load_case(_write_case(tmp_path, documents))


@pytest.mark.parametrize("threshold", (0.0, 0.99, float("inf"), float("nan")))
def test_step_growth_threshold_rejects_invalid_values(tmp_path, threshold):
    documents = _case_documents()
    documents["run.yaml"]["solver"].update(
        backend="compiled", name="band", step_growth_threshold=threshold
    )
    with pytest.raises(PackedBedValidationError, match="step_growth_threshold"):
        load_case(_write_case(tmp_path, documents))


@pytest.mark.parametrize("interval", (-1, 1.5, True, float("inf")))
def test_nonlinear_refresh_interval_rejects_invalid_values(tmp_path, interval):
    documents = _case_documents()
    documents["run.yaml"]["solver"].update(
        backend="compiled", name="band", nonlinear_refresh_interval=interval
    )
    with pytest.raises(PackedBedValidationError, match="nonlinear_refresh_interval"):
        load_case(_write_case(tmp_path, documents))


@pytest.mark.parametrize("value", (1, "true", None))
def test_band_reciprocals_requires_boolean(tmp_path, value):
    documents = _case_documents()
    documents["run.yaml"]["solver"].update(
        backend="compiled", name="band", band_reciprocals=value
    )
    with pytest.raises(PackedBedValidationError, match="band_reciprocals"):
        load_case(_write_case(tmp_path, documents))


@pytest.mark.parametrize("value", (1, "true", None))
def test_vector_exponentials_requires_boolean(tmp_path, value):
    documents = _case_documents()
    documents["run.yaml"]["solver"].update(
        backend="compiled", name="band", vector_exponentials=value
    )
    with pytest.raises(PackedBedValidationError, match="vector_exponentials"):
        load_case(_write_case(tmp_path, documents))




@pytest.mark.parametrize("filename,solver", (("run_compiled.yaml", "superlu"), ("run_band.yaml", "band")))
def test_compiled_example_preserves_default_physics(filename, solver):
    directory = Path(__file__).resolve().parents[1] / "packed_bed/examples/default_case"
    original = load_case(directory / "run.yaml")
    compiled = load_case(directory / filename)
    assert compiled.run.model == original.run.model
    assert compiled.chemistry == original.chemistry
    assert compiled.solids == original.solids
    assert compiled.program == original.program
    assert compiled.run.simulation == original.run.simulation
    assert compiled.run.solver.backend == "compiled"
    assert compiled.run.solver.name == solver
    controls = {"backend", "name", "step_growth_threshold", "nonlinear_refresh_interval"}
    assert compiled.run.solver.model_dump(exclude=controls) == original.run.solver.model_dump(exclude=controls)
    assert compiled.run.solver.nonlinear_refresh_interval == (4 if solver == "band" else 0)
    assert compiled.output_directory != original.output_directory


@pytest.mark.parametrize("profile", ("superlu", "compiled_superlu", "compiled_band"))
def test_benchmark_copies_inputs_and_matches_state_tolerances(tmp_path, profile):
    from copy import deepcopy
    from tools.benchmark_solvers import prepare_case, scenarios

    source = Path(__file__).resolve().parents[1] / "packed_bed/examples/default_case/run.yaml"
    original = source.read_bytes()
    _, _, documents, _ = next(scenarios("default"))
    documents["run"]["solver"].update(relative_tolerance=1e-6, concentration_absolute_tolerance=1e-11)
    before = deepcopy(documents)
    run_file = prepare_case(documents, tmp_path / profile, profile)
    case = load_case(run_file)
    assert case.run.solver.relative_tolerance == 1e-6
    assert case.run.solver.concentration_absolute_tolerance == 1e-11
    assert case.output_directory == tmp_path / profile / "output"
    assert case.run.solver.backend == ("daetools" if profile == "superlu" else "compiled")
    assert source.read_bytes() == original
    assert documents == before


def test_load_case_is_side_effect_free(tmp_path: Path) -> None:
    run_path = _write_case(tmp_path)
    files_before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    load_case(run_path)

    assert (
        sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
        == files_before
    )
    assert not (tmp_path / "output").exists()




def test_ghsv_inlet_flow_is_compiled_for_an_ordinary_case(tmp_path: Path) -> None:
    documents = _case_documents()
    documents["program.yaml"]["inlet_flow"].update(
        basis="ghsv_per_h",
        initial=3600.0,
        steps=[{"kind": "ramp", "duration_s": 10.0, "target": 7200.0}],
    )
    run_path = _write_case(tmp_path, documents)

    case = load_case(run_path)

    empty_bed_volume_m3 = math.pi * 0.01**2 * 1.0
    assert case.inlet_flow_program.initial_value == pytest.approx(
        empty_bed_volume_m3 * NORMAL_MOLAR_DENSITY_MOL_PER_M3
    )
    assert case.inlet_flow_program.segments[0].end_value == pytest.approx(
        2.0 * empty_bed_volume_m3 * NORMAL_MOLAR_DENSITY_MOL_PER_M3
    )


def test_load_case_does_not_import_daetools(tmp_path: Path) -> None:
    run_path = _write_case(tmp_path)
    script = """
import builtins
import sys

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'daetools' or name.startswith('daetools.') or name == 'pyUnits':
        raise AssertionError(f'forbidden solver import: {name}')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from packed_bed.config import load_case
load_case(sys.argv[1])
assert not any(name == 'daetools' or name.startswith('daetools.') for name in sys.modules)
assert 'pyUnits' not in sys.modules
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]))

    subprocess.run(
        [sys.executable, "-c", script, str(run_path)],
        check=True,
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )


def test_duplicate_yaml_keys_are_rejected_with_file_and_line(tmp_path: Path) -> None:
    run_path = _write_case(tmp_path)
    (tmp_path / "program.yaml").write_text(
        """inlet_flow: {initial: 1.0, steps: []}
inlet_flow: {initial: 2.0, steps: []}
""",
        encoding="utf-8",
    )

    with pytest.raises(PackedBedValidationError) as caught:
        load_case(run_path)

    message = str(caught.value)
    assert f"program is invalid: {tmp_path / 'program.yaml'}" in message
    assert "duplicate key 'inlet_flow' at line 2" in message


@pytest.mark.parametrize("source", [b"? [x, y]\n: 1\n", b"\xff"])
def test_invalid_yaml_uses_a_configuration_error(tmp_path, source):
    run_path = _write_case(tmp_path)
    (tmp_path / "program.yaml").write_bytes(source)
    with pytest.raises(PackedBedValidationError, match="program contains invalid YAML"):
        load_case(run_path)


def test_missing_reference_files_are_reported_together(tmp_path: Path) -> None:
    documents = _case_documents()
    (tmp_path / "run.yaml").write_text(
        yaml.safe_dump(documents["run.yaml"], sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(PackedBedValidationError) as caught:
        load_case(tmp_path / "run.yaml")

    message = str(caught.value)
    assert "run.references.chemistry_file does not exist" in message
    assert "run.references.program_file does not exist" in message
    assert "run.references.solids_file does not exist" in message


def test_structural_errors_use_configuration_paths_and_are_aggregated(
    tmp_path: Path,
) -> None:
    documents = _case_documents()
    documents["chemistry.yaml"]["gas_species"] = ["N2", "H2"]
    zone = documents["solids.yaml"]["initial_profile"]["zones"][0]
    zone["x_start_m"] = 0.1
    zone["x_end_m"] = 0.9
    run_path = _write_case(tmp_path, documents)

    with pytest.raises(PackedBedValidationError) as caught:
        load_case(run_path)

    message = str(caught.value)
    assert "solids.initial_profile.zones must start at x = 0" in message
    assert "solids.initial_profile.zones must end at run.model.bed_length_m" in message
    assert "program.inlet_composition.initial species mismatch: missing H2" in message


def test_component_reaction_and_report_references_are_aggregated(
    tmp_path: Path,
) -> None:
    documents = _case_documents()
    documents["chemistry.yaml"] = {
        "gas_species": ["MysteryGas"],
        "reaction_families": [],
        "reaction_ids": ["mystery_reaction"],
    }
    documents["program.yaml"]["inlet_composition"]["initial"] = {"MysteryGas": 1.0}
    documents["run.yaml"]["outputs"]["requested_reports"] = ["mystery_report"]
    documents["run.yaml"]["outputs"]["requested_plots"] = [
        "mystery_plot",
        "outlet_conditions",
    ]
    run_path = _write_case(tmp_path, documents)

    with pytest.raises(PackedBedValidationError) as caught:
        load_case(run_path)

    message = str(caught.value)
    assert "Unknown gas species 'MysteryGas'" in message
    assert "chemistry.reaction_ids contains unknown id 'mystery_reaction'" in message
    assert (
        "run.outputs.requested_reports contains unknown ids: mystery_report" in message
    )
    assert "run.outputs.requested_plots contains unknown ids: mystery_plot" in message
    assert (
        "'outlet_conditions' requires requested_reports: gas_flux, pressure, temperature"
        in message
    )


def test_reaction_rate_report_requires_a_selected_reaction(tmp_path: Path) -> None:
    documents = _case_documents()
    documents["run.yaml"]["outputs"]["requested_reports"] = ["reaction_rate"]

    with pytest.raises(
        PackedBedValidationError, match="requires at least one selected reaction"
    ):
        load_case(_write_case(tmp_path, documents))


def test_requested_plots_is_required_and_unique(tmp_path: Path) -> None:
    missing = _case_documents()
    missing["run.yaml"]["outputs"].pop("requested_plots")
    missing_directory = tmp_path / "missing"
    missing_directory.mkdir()
    with pytest.raises(PackedBedValidationError, match="run.outputs.requested_plots"):
        load_case(_write_case(missing_directory, missing))

    duplicate = _case_documents()
    duplicate["run.yaml"]["outputs"]["requested_plots"] = [
        "axial_profiles",
        "axial_profiles",
    ]
    duplicate["run.yaml"]["outputs"]["requested_reports"] = ["temperature", "pressure"]
    duplicate_directory = tmp_path / "duplicate"
    duplicate_directory.mkdir()
    with pytest.raises(
        PackedBedValidationError, match="contains duplicates: axial_profiles"
    ):
        load_case(_write_case(duplicate_directory, duplicate))


def test_unknown_reaction_family_uses_configuration_path(tmp_path: Path) -> None:
    documents = _case_documents()
    documents["chemistry.yaml"]["reaction_families"] = ["mystery_family"]
    run_path = _write_case(tmp_path, documents)

    with pytest.raises(PackedBedValidationError) as caught:
        load_case(run_path)

    assert (
        "chemistry.reaction_families: Unknown reaction families: mystery_family"
        in str(caught.value)
    )


def test_reaction_must_belong_to_a_selected_family(tmp_path: Path) -> None:
    documents = _case_documents()
    documents["chemistry.yaml"]["reaction_families"] = ["nickel_medrano"]
    documents["chemistry.yaml"]["reaction_ids"] = ["smr_reaction_xu_froment"]
    run_path = _write_case(tmp_path, documents)

    with pytest.raises(PackedBedValidationError) as caught:
        load_case(run_path)

    assert (
        "chemistry.reaction_ids contains unknown id 'smr_reaction_xu_froment'"
        in str(caught.value)
    )


def test_accepts_sub_nanosecond_duration_sum_drift(tmp_path: Path) -> None:
    run_path = _write_case(
        tmp_path,
        _case_documents(flow_duration_s=9.999999999999998, composition_duration_s=10.0),
    )

    assert load_case(run_path).run.simulation.time_horizon_s == 10.0


def test_rejects_material_duration_sum_mismatch(tmp_path: Path) -> None:
    run_path = _write_case(
        tmp_path,
        _case_documents(flow_duration_s=9.999, composition_duration_s=10.0),
    )

    with pytest.raises(PackedBedValidationError, match=r"difference -1\.000e-03 s"):
        load_case(run_path)


def test_accepts_integer_literals_for_float_fields() -> None:
    program = ProgramConfig.model_validate(
        {
            "inlet_flow": {"initial": 1, "steps": []},
            "inlet_temperature": {"initial": 300, "steps": []},
            "outlet_pressure": {"initial": 100000, "steps": []},
            "inlet_composition": {"initial": {"N2": 1}, "steps": []},
        }
    )

    assert program.inlet_flow.initial == 1.0
    assert program.inlet_composition.initial["N2"] == 1.0


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_rejects_invalid_float_fields(value: float) -> None:
    with pytest.raises(ValidationError):
        ProgramConfig.model_validate(
            {
                "inlet_flow": {"initial": value, "steps": []},
                "inlet_temperature": {"initial": 300.0, "steps": []},
                "outlet_pressure": {"initial": 100000.0, "steps": []},
                "inlet_composition": {"initial": {"N2": 1.0}, "steps": []},
            }
        )


def test_program_channel_compilers_return_the_runtime_representation() -> None:
    program_config = ProgramConfig.model_validate(
        {
            "inlet_flow": {"initial": 1.0, "steps": []},
            "inlet_temperature": {"initial": 300.0, "steps": []},
            "outlet_pressure": {"initial": 100000.0, "steps": []},
            "inlet_composition": {
                "initial": {"N2": 0.75, "H2": 0.25},
                "steps": [],
            },
        }
    )

    scalar = compile_scalar_channel(program_config.inlet_flow)
    composition = compile_composition_channel(
        program_config.inlet_composition,
        ("H2", "N2"),
    )

    assert isinstance(scalar, CompiledProgram)
    assert scalar.value_at(10.0, smooth_ramp_width_s=1.0) == 1.0
    assert composition.initial_value == (0.25, 0.75)


def test_config_public_surface_has_no_compatibility_bundle() -> None:
    assert hasattr(config, "load_case")
    assert hasattr(config, "validate_case")
    assert not hasattr(config, "RunBundle")
    assert not hasattr(config, "load_run_bundle")
    assert not hasattr(config, "DEFAULT_SMOOTH_RAMP_WIDTH_S")
