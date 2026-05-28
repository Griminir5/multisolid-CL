import unittest
from pathlib import Path

from pydantic import ValidationError

from packed_bed.config import ChemistryConfig, ProgramConfig, RunBundle, RunConfig, SolidConfig


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

        self.assertEqual(bundle.run.time_horizon_s, 12000.0)

    def test_rejects_material_duration_sum_mismatch(self) -> None:
        with self.assertRaisesRegex(ValidationError, r"difference -1\.000e-03 s"):
            build_run_bundle(11999.999, composition_duration_s=12000.0)


if __name__ == "__main__":
    unittest.main()
