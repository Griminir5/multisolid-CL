import unittest
from pathlib import Path

from pydantic import ValidationError

import packed_bed.config as config
from packed_bed.config import (
    ChemistryConfig,
    PackedBedValidationError,
    ProgramConfig,
    RunBundle,
    RunConfig,
    SolidConfig,
    validate_bundle_shape,
)
from packed_bed.programs import compile_composition_channel, compile_scalar_channel


def build_run_bundle(flow_duration_s: float, composition_duration_s: float | None = None) -> RunBundle:
    composition_duration_s = flow_duration_s if composition_duration_s is None else composition_duration_s
    chemistry = ChemistryConfig.model_validate({"gas_species": ["N2"], "reaction_ids": []})
    solids = SolidConfig.model_validate(
        {
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
                        "values": {"Ni": 0.0},
                    }
                ],
            },
        }
    )
    program = ProgramConfig.model_validate(
        {
            "inlet_flow": {"initial": 1.0, "steps": [{"kind": "hold", "duration_s": flow_duration_s}]},
            "inlet_temperature": {"initial": 300.0, "steps": []},
            "outlet_pressure": {"initial": 100000.0, "steps": []},
            "inlet_composition": {
                "initial": {"N2": 1.0},
                "steps": [{"kind": "hold", "duration_s": composition_duration_s}],
            },
        }
    )
    run = RunConfig.model_validate(
        {
            "references": {
                "chemistry_file": "chemistry.yaml",
                "program_file": "program.yaml",
                "solids_file": "solids.yaml",
            },
            "simulation": {
                "system_name": "test",
                "time_horizon_s": 12000.0,
                "reporting_interval_s": 1.0,
                "repeat_program": False,
                "mass_scheme": "weno3",
                "heat_scheme": "weno3",
                "report_time_derivatives": False,
            },
            "model": {
                "bed_length_m": 1.0,
                "bed_radius_m": 0.01,
                "axial_cells": 1,
                "ambient_temperature_k": 300.0,
                "heat_transfer_coefficient_w_per_m2_k": 0.0,
            },
            "solver": {"relative_tolerance": 1.0e-5},
            "outputs": {
                "directory": "output",
                "artifacts_directory": "output/artifacts",
                "requested_reports": [],
            },
        }
    )
    return RunBundle(
        run_path=Path("run.yaml"),
        chemistry_path=Path("chemistry.yaml"),
        solids_path=Path("solids.yaml"),
        program_path=Path("program.yaml"),
        chemistry=chemistry,
        solids=solids,
        program=program,
        run=run,
    )


class RunBundleValidationTests(unittest.TestCase):
    def test_accepts_sub_nanosecond_duration_sum_drift(self) -> None:
        bundle = build_run_bundle(11999.999999999998)

        self.assertIs(validate_bundle_shape(bundle), bundle)
        self.assertEqual(bundle.run.simulation.time_horizon_s, 12000.0)

    def test_rejects_material_duration_sum_mismatch(self) -> None:
        bundle = build_run_bundle(11999.999, composition_duration_s=12000.0)

        with self.assertRaisesRegex(PackedBedValidationError, r"difference -1\.000e-03 s"):
            validate_bundle_shape(bundle)


class NumericValidationTests(unittest.TestCase):
    def test_accepts_integer_literals_for_float_fields(self) -> None:
        program = ProgramConfig.model_validate(
            {
                "inlet_flow": {"initial": 1, "steps": []},
                "inlet_temperature": {"initial": 300, "steps": []},
                "outlet_pressure": {"initial": 100000, "steps": []},
                "inlet_composition": {"initial": {"N2": 1}, "steps": []},
            }
        )

        self.assertEqual(program.inlet_flow.initial, 1.0)
        self.assertEqual(program.inlet_composition.initial["N2"], 1.0)

    def test_rejects_bool_for_float_fields(self) -> None:
        with self.assertRaises(ValidationError):
            ProgramConfig.model_validate(
                {
                    "inlet_flow": {"initial": True, "steps": []},
                    "inlet_temperature": {"initial": 300.0, "steps": []},
                    "outlet_pressure": {"initial": 100000.0, "steps": []},
                    "inlet_composition": {"initial": {"N2": 1.0}, "steps": []},
                }
            )

    def test_rejects_nonfinite_float_fields(self) -> None:
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    ProgramConfig.model_validate(
                        {
                            "inlet_flow": {"initial": value, "steps": []},
                            "inlet_temperature": {"initial": 300.0, "steps": []},
                            "outlet_pressure": {"initial": 100000.0, "steps": []},
                            "inlet_composition": {"initial": {"N2": 1.0}, "steps": []},
                        }
                    )


class ProgramCompilationTests(unittest.TestCase):
    def test_compiles_scalar_channel_with_function_api(self) -> None:
        program_config = ProgramConfig.model_validate(
            {
                "inlet_flow": {"initial": 1.0, "steps": []},
                "inlet_temperature": {"initial": 300.0, "steps": []},
                "outlet_pressure": {"initial": 100000.0, "steps": []},
                "inlet_composition": {"initial": {"N2": 1.0}, "steps": []},
            }
        )

        program = compile_scalar_channel(program_config.inlet_flow)

        self.assertEqual(program.segments, ())
        self.assertEqual(program.value_at(10.0, smooth_ramp_width_s=1.0), 1.0)

    def test_compiles_composition_channel_with_function_api(self) -> None:
        program_config = ProgramConfig.model_validate(
            {
                "inlet_flow": {"initial": 1.0, "steps": []},
                "inlet_temperature": {"initial": 300.0, "steps": []},
                "outlet_pressure": {"initial": 100000.0, "steps": []},
                "inlet_composition": {"initial": {"N2": 0.75, "H2": 0.25}, "steps": []},
            }
        )

        program = compile_composition_channel(program_config.inlet_composition, ("H2", "N2"))

        self.assertEqual(program.initial_value, (0.25, 0.75))
        self.assertEqual(program.value_at(10.0, smooth_ramp_width_s=1.0), (0.25, 0.75))


class PublicImportTests(unittest.TestCase):
    def test_config_public_surface_is_small(self) -> None:
        self.assertTrue(hasattr(config, "load_run_bundle"))
        self.assertTrue(hasattr(config, "validate_bundle_shape"))
        self.assertFalse(hasattr(config, "RunResult"))
        self.assertFalse(hasattr(config, "DEFAULT_SMOOTH_RAMP_WIDTH_S"))
        self.assertFalse(hasattr(config, "ScalarProgram"))


if __name__ == "__main__":
    unittest.main()
