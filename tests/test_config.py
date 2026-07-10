from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from pydantic import ValidationError
import pytest
import yaml

import packed_bed.config as config
from packed_bed.config import Case, PackedBedValidationError, ProgramConfig, load_case
from packed_bed.programs import CompiledProgram, compile_composition_channel, compile_scalar_channel


def _case_documents(
    *,
    flow_duration_s: float | None = None,
    composition_duration_s: float | None = None,
) -> dict[str, dict]:
    flow_steps = [] if flow_duration_s is None else [{"kind": "hold", "duration_s": flow_duration_s}]
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
                "requested_reports": [],
            },
        },
        "chemistry.yaml": {"gas_species": ["N2"], "reaction_ids": []},
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
        (tmp_path / filename).write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return tmp_path / "run.yaml"


def test_load_case_returns_one_resolved_case_with_compiled_programs(tmp_path: Path) -> None:
    run_path = _write_case(
        tmp_path,
        _case_documents(flow_duration_s=10.0, composition_duration_s=10.0),
    )

    case = load_case(run_path)

    assert isinstance(case, Case)
    assert isinstance(case.inlet_flow_program, CompiledProgram)
    assert case.inlet_flow_program.duration_s == 10.0
    assert case.inlet_composition_program.initial_value == (1.0,)
    assert not hasattr(case, "program")
    assert case.output_directory == (tmp_path / "output").resolve()


def test_load_case_is_side_effect_free(tmp_path: Path) -> None:
    run_path = _write_case(tmp_path)
    files_before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    load_case(run_path)

    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == files_before
    assert not (tmp_path / "output").exists()


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
inlet_temperature: {initial: 300.0, steps: []}
outlet_pressure: {initial: 100000.0, steps: []}
inlet_composition: {initial: {N2: 1.0}, steps: []}
""",
        encoding="utf-8",
    )

    with pytest.raises(PackedBedValidationError) as caught:
        load_case(run_path)

    message = str(caught.value)
    assert f"program is invalid: {tmp_path / 'program.yaml'}" in message
    assert "duplicate key 'inlet_flow' at line 2" in message


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


def test_structural_errors_use_configuration_paths_and_are_aggregated(tmp_path: Path) -> None:
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


def test_component_reaction_and_report_references_are_aggregated(tmp_path: Path) -> None:
    documents = _case_documents()
    documents["chemistry.yaml"] = {
        "gas_species": ["MysteryGas"],
        "reaction_ids": ["mystery_reaction"],
    }
    documents["program.yaml"]["inlet_composition"]["initial"] = {"MysteryGas": 1.0}
    documents["run.yaml"]["outputs"]["requested_reports"] = ["mystery_report"]
    run_path = _write_case(tmp_path, documents)

    with pytest.raises(PackedBedValidationError) as caught:
        load_case(run_path)

    message = str(caught.value)
    assert "Unknown gas species 'MysteryGas'" in message
    assert "chemistry.reaction_ids contains unknown id 'mystery_reaction'" in message
    assert "run.outputs.requested_reports contains unknown ids: mystery_report" in message


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
