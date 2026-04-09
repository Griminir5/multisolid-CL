from __future__ import annotations

import importlib.util
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from packed_bed import load_run_bundle, run_simulation, validate_run_bundle
from packed_bed.config import OutputConfig
from packed_bed.solver import assemble_simulation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_YAML = REPO_ROOT / "packed_bed" / "examples" / "default_case" / "run.yaml"


class PackedBedIntegrationTests(unittest.TestCase):
    def test_default_example_runs_and_exports_expected_files(self):
        bundle = load_run_bundle(DEFAULT_RUN_YAML)
        validate_run_bundle(bundle)

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_simulation(
                DEFAULT_RUN_YAML,
                output_dir=tmp_path / "output",
                artifacts_dir=tmp_path / "artifacts",
            )

            self.assertTrue(result.success)
            self.assertTrue(result.summary_path.exists())
            self.assertTrue(result.balances_path.exists())
            self.assertTrue(result.artifact_paths["system_graph_png"].exists())
            self.assertTrue(result.artifact_paths["operating_program_png"].exists())
            self.assertTrue(result.artifact_paths["initial_solid_profile_png"].exists())

            expected_reports = {f"{report_id}.csv" for report_id in bundle.run.outputs.requested_reports}
            actual_reports = {path.name for path in result.report_paths.values()}
            self.assertEqual(expected_reports, actual_reports)

    def test_metadata_only_reaction_fails_before_solver_initialization(self):
        bundle = load_run_bundle(DEFAULT_RUN_YAML)
        chemistry = replace(
            bundle.chemistry,
            gas_species=("H2", "H2O"),
            reaction_ids=("ni_reduction_h2_medrano",),
        )
        solids = replace(
            bundle.solids,
            solid_species=("Ni", "NiO"),
            initial_profile_zones=tuple(
                replace(zone, values_mol_per_m3={"Ni": 0.0, "NiO": 100000.0})
                for zone in bundle.solids.initial_profile_zones
            ),
        )
        program = replace(
            bundle.program,
            inlet_composition=replace(
                bundle.program.inlet_composition,
                initial={"H2": 1.0, "H2O": 0.0},
                steps=(
                    replace(
                        bundle.program.inlet_composition.steps[0],
                        target=None,
                    ),
                    replace(
                        bundle.program.inlet_composition.steps[1],
                        target={"H2": 0.5, "H2O": 0.5},
                    ),
                    replace(
                        bundle.program.inlet_composition.steps[2],
                        target=None,
                    ),
                ),
            ),
        )
        with TemporaryDirectory() as tmp:
            outputs = OutputConfig(
                directory=Path(tmp) / "output",
                artifacts_directory=Path(tmp) / "artifacts",
                requested_reports=bundle.run.outputs.requested_reports,
            )
            guarded_bundle = replace(
                bundle,
                chemistry=chemistry,
                solids=solids,
                program=program,
                run=replace(bundle.run, outputs=outputs),
            )
            validate_run_bundle(guarded_bundle)
            with self.assertRaisesRegex(NotImplementedError, "ni_reduction_h2_medrano"):
                assemble_simulation(guarded_bundle)

    def test_legacy_operation_program_file_delegates_to_package(self):
        legacy_path = REPO_ROOT / "Packed Bed Models" / "CLBed_MHMM_operation_program.py"
        spec = importlib.util.spec_from_file_location("legacy_operation_program", legacy_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        from packed_bed.programs import ScalarProgram
        from packed_bed.solver import simBed

        self.assertIs(module.simBed, simBed)
        self.assertIs(module.ScalarProgram, ScalarProgram)

    def test_solver_reuses_shared_program_types(self):
        import packed_bed.solver as solver_module
        from packed_bed.programs import ProgramSegment, ProgramStep, ScalarProgram, VectorProgram

        self.assertIs(solver_module.ProgramStep, ProgramStep)
        self.assertIs(solver_module.ProgramSegment, ProgramSegment)
        self.assertIs(solver_module.ScalarProgram, ScalarProgram)
        self.assertIs(solver_module.VectorProgram, VectorProgram)

    def test_run_simulation_rejects_run_bundle_argument(self):
        bundle = load_run_bundle(DEFAULT_RUN_YAML)
        with self.assertRaisesRegex(TypeError, "expects a run.yaml path"):
            run_simulation(bundle)

    def test_cli_runs_default_example(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            artifacts_dir = Path(tmp) / "artifacts"
            command = [
                "python",
                "-m",
                "packed_bed",
                str(DEFAULT_RUN_YAML),
                "--output-dir",
                str(output_dir),
                "--artifacts-dir",
                str(artifacts_dir),
            ]
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Simulation finished successfully", completed.stdout)
            self.assertTrue((output_dir / "run_summary.csv").exists())
            self.assertTrue((artifacts_dir / "system_graph.png").exists())
            self.assertTrue((artifacts_dir / "initial_solid_profile.png").exists())


if __name__ == "__main__":
    unittest.main()
