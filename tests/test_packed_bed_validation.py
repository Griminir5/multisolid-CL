from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from packed_bed import load_run_bundle
from packed_bed.config import OutputConfig
from packed_bed.properties import (
    CpZerothMolar,
    DEFAULT_PROPERTY_REGISTRY,
    PropertyRegistry,
    SpeciesPropertyRecord,
)
from packed_bed.validation import PackedBedValidationError, validate_run_bundle


DEFAULT_CHEMISTRY = {
    "gas_species": ["AR", "H2", "H2O"],
    "solid_species": ["Ni", "NiO", "CaAl2O4"],
    "reaction_ids": [],
}

DEFAULT_PROGRAM = {
    "inlet_flow": {
        "initial": 0.785,
        "steps": [{"kind": "hold", "duration_s": 5.0}],
    },
    "inlet_temperature": {
        "initial": 500.0,
        "steps": [{"kind": "hold", "duration_s": 5.0}],
    },
    "outlet_pressure": {
        "initial": 101325.0,
        "steps": [{"kind": "hold", "duration_s": 5.0}],
    },
    "inlet_composition": {
        "initial": {"AR": 1.0, "H2": 0.0, "H2O": 0.0},
        "steps": [{"kind": "hold", "duration_s": 5.0}],
    },
}

DEFAULT_RUN = {
    "references": {
        "chemistry_file": "chemistry.yaml",
        "program_file": "program.yaml",
    },
    "simulation": {
        "time_horizon_s": 10.0,
        "reporting_interval_s": 5.0,
        "mass_scheme": "weno3",
        "heat_scheme": "central",
        "report_time_derivatives": False,
    },
    "model": {
        "bed_length_m": 2.5,
        "bed_radius_m": 0.1,
        "particle_diameter_m": 0.01,
        "axial_cells": 5,
        "interparticle_voidage": 0.5,
        "intraparticle_voidage": 0.5,
        "initial_solid_concentration_mol_per_m3": {
            "Ni": 100000.0,
            "NiO": 0.0,
            "CaAl2O4": 0.0,
        },
    },
    "solver": {"relative_tolerance": 1e-6},
    "outputs": {
        "directory": "output",
        "artifacts_directory": "artifacts",
        "requested_reports": ["temperature", "pressure"],
    },
}


def _write_yaml(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _make_bundle(temp_dir: Path, chemistry=None, program=None, run=None):
    chemistry_path = temp_dir / "chemistry.yaml"
    program_path = temp_dir / "program.yaml"
    run_path = temp_dir / "run.yaml"
    _write_yaml(chemistry_path, DEFAULT_CHEMISTRY if chemistry is None else chemistry)
    _write_yaml(program_path, DEFAULT_PROGRAM if program is None else program)
    _write_yaml(run_path, DEFAULT_RUN if run is None else run)
    bundle = load_run_bundle(run_path)
    outputs = OutputConfig(
        directory=temp_dir / "output",
        artifacts_directory=temp_dir / "artifacts",
        requested_reports=bundle.run.outputs.requested_reports,
    )
    return replace(bundle, run=replace(bundle.run, outputs=outputs))


class PackedBedValidationTests(unittest.TestCase):
    def test_rejects_unknown_species(self):
        with TemporaryDirectory() as tmp:
            chemistry = dict(DEFAULT_CHEMISTRY)
            chemistry["gas_species"] = ["AR", "UNKNOWN"]
            bundle = _make_bundle(Path(tmp), chemistry=chemistry)
            with self.assertRaisesRegex(PackedBedValidationError, "Unknown gas species 'UNKNOWN'"):
                validate_run_bundle(bundle)

    def test_rejects_duplicate_species(self):
        with TemporaryDirectory() as tmp:
            chemistry = dict(DEFAULT_CHEMISTRY)
            chemistry["solid_species"] = ["Ni", "Ni", "CaAl2O4"]
            bundle = _make_bundle(Path(tmp), chemistry=chemistry)
            with self.assertRaisesRegex(PackedBedValidationError, "must not contain duplicates"):
                validate_run_bundle(bundle)

    def test_rejects_missing_required_correlations(self):
        with TemporaryDirectory() as tmp:
            chemistry = dict(DEFAULT_CHEMISTRY)
            chemistry["gas_species"] = ["BROKEN_GAS"]
            run = dict(DEFAULT_RUN)
            run["model"] = dict(DEFAULT_RUN["model"])
            bundle = _make_bundle(Path(tmp), chemistry=chemistry, run=run)
            custom_registry = PropertyRegistry(
                records={
                    **DEFAULT_PROPERTY_REGISTRY.records,
                    "BROKEN_GAS": SpeciesPropertyRecord(
                        name="Broken Gas",
                        phase="gas",
                        mw=1.0e-3,
                        enthalpy=CpZerothMolar(h_form_ref=0.0, a0=10.0),
                        viscosity=None,
                    ),
                }
            )
            with self.assertRaisesRegex(PackedBedValidationError, "must define a viscosity correlation"):
                validate_run_bundle(bundle, property_registry=custom_registry)

    def test_rejects_phase_mismatch(self):
        with TemporaryDirectory() as tmp:
            chemistry = dict(DEFAULT_CHEMISTRY)
            chemistry["gas_species"] = ["Ni"]
            bundle = _make_bundle(Path(tmp), chemistry=chemistry)
            with self.assertRaisesRegex(PackedBedValidationError, "not gas"):
                validate_run_bundle(bundle)

    def test_rejects_incompatible_reaction_selection(self):
        with TemporaryDirectory() as tmp:
            chemistry = dict(DEFAULT_CHEMISTRY)
            chemistry["reaction_ids"] = ["ni_reduction_h2_medrano"]
            chemistry["gas_species"] = ["AR"]
            bundle = _make_bundle(Path(tmp), chemistry=chemistry)
            with self.assertRaisesRegex(PackedBedValidationError, "requires unselected species"):
                validate_run_bundle(bundle)

    def test_rejects_unknown_report_id(self):
        with TemporaryDirectory() as tmp:
            run = dict(DEFAULT_RUN)
            run["outputs"] = dict(DEFAULT_RUN["outputs"])
            run["outputs"]["requested_reports"] = ["temperature", "unknown_report"]
            bundle = _make_bundle(Path(tmp), run=run)
            with self.assertRaisesRegex(PackedBedValidationError, "unknown ids: unknown_report"):
                validate_run_bundle(bundle)

    def test_rejects_nonpositive_step_duration(self):
        with TemporaryDirectory() as tmp:
            program = dict(DEFAULT_PROGRAM)
            program["inlet_flow"] = {
                "initial": 0.785,
                "steps": [{"kind": "hold", "duration_s": 0.0}],
            }
            bundle = _make_bundle(Path(tmp), program=program)
            with self.assertRaisesRegex(PackedBedValidationError, "duration must be positive"):
                validate_run_bundle(bundle)

    def test_rejects_composition_with_missing_species(self):
        with TemporaryDirectory() as tmp:
            program = dict(DEFAULT_PROGRAM)
            program["inlet_composition"] = {
                "initial": {"AR": 1.0, "H2": 0.0},
                "steps": [{"kind": "hold", "duration_s": 5.0}],
            }
            bundle = _make_bundle(Path(tmp), program=program)
            with self.assertRaisesRegex(PackedBedValidationError, "missing entries for: H2O"):
                validate_run_bundle(bundle)

    def test_rejects_composition_that_does_not_sum_to_one(self):
        with TemporaryDirectory() as tmp:
            program = dict(DEFAULT_PROGRAM)
            program["inlet_composition"] = {
                "initial": {"AR": 0.8, "H2": 0.3, "H2O": 0.0},
                "steps": [{"kind": "hold", "duration_s": 5.0}],
            }
            bundle = _make_bundle(Path(tmp), program=program)
            with self.assertRaisesRegex(PackedBedValidationError, "must sum to 1"):
                validate_run_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
