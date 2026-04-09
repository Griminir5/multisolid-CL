from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from packed_bed import load_run_bundle
from packed_bed.config import OutputConfig, SolidConfig, SolidZoneConfig
from packed_bed.properties import (
    CpZerothMolar,
    DEFAULT_PROPERTY_REGISTRY,
    PropertyRegistry,
    SpeciesPropertyRecord,
)
from packed_bed.solid_profiles import build_face_scalar_profile
from packed_bed.solver import _convert_solid_profile_to_bed_volume
from packed_bed.validation import PackedBedValidationError, validate_run_bundle


DEFAULT_CHEMISTRY = {
    "gas_species": ["AR", "H2", "H2O"],
    "reaction_ids": [],
}

DEFAULT_SOLIDS = {
    "solid_species": ["Ni", "NiO", "CaAl2O4"],
    "initial_profile": {
        "units": "mol_per_m3_solid",
        "zones": [
            {
                "x_start_m": 0.0,
                "x_end_m": 2.5,
                "e_b": 0.5,
                "e_p": 0.5,
                "d_p": 0.01,
                "values": {
                    "Ni": 100000.0,
                    "NiO": 0.0,
                    "CaAl2O4": 0.0,
                },
            }
        ],
    },
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
        "solids_file": "solids.yaml",
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


def _make_bundle(temp_dir: Path, chemistry=None, solids=None, program=None, run=None):
    chemistry_path = temp_dir / "chemistry.yaml"
    solids_path = temp_dir / "solids.yaml"
    program_path = temp_dir / "program.yaml"
    run_path = temp_dir / "run.yaml"
    _write_yaml(chemistry_path, DEFAULT_CHEMISTRY if chemistry is None else chemistry)
    _write_yaml(solids_path, DEFAULT_SOLIDS if solids is None else solids)
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
            solids = dict(DEFAULT_SOLIDS)
            solids["solid_species"] = ["Ni", "Ni", "CaAl2O4"]
            bundle = _make_bundle(Path(tmp), solids=solids)
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
            solids = {
                "solid_species": ["Ni"],
                "initial_profile": {
                    "units": "mol_per_m3_solid",
                    "zones": [
                        {
                            "x_start_m": 0.0,
                            "x_end_m": 2.5,
                            "e_b": 0.5,
                            "e_p": 0.5,
                            "d_p": 0.01,
                            "values": {"Ni": 100000.0},
                        }
                    ],
                },
            }
            bundle = _make_bundle(Path(tmp), chemistry=chemistry, solids=solids)
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

    def test_rejects_noncontiguous_solid_profile(self):
        with TemporaryDirectory() as tmp:
            solids = {
                "solid_species": ["Ni", "NiO", "CaAl2O4"],
                "initial_profile": {
                    "units": "mol_per_m3_solid",
                    "zones": [
                        {
                            "x_start_m": 0.0,
                            "x_end_m": 1.0,
                            "e_b": 0.5,
                            "e_p": 0.5,
                            "d_p": 0.01,
                            "values": {"Ni": 100000.0, "NiO": 0.0, "CaAl2O4": 0.0},
                        },
                        {
                            "x_start_m": 1.2,
                            "x_end_m": 2.5,
                            "e_b": 0.5,
                            "e_p": 0.5,
                            "d_p": 0.01,
                            "values": {"Ni": 100000.0, "NiO": 0.0, "CaAl2O4": 0.0},
                        },
                    ],
                },
            }
            bundle = _make_bundle(Path(tmp), solids=solids)
            with self.assertRaisesRegex(PackedBedValidationError, "contiguous"):
                validate_run_bundle(bundle)

    def test_rejects_invalid_zone_voidage_or_particle_length(self):
        with TemporaryDirectory() as tmp:
            solids = {
                "solid_species": ["Ni", "NiO", "CaAl2O4"],
                "initial_profile": {
                    "units": "mol_per_m3_solid",
                    "zones": [
                        {
                            "x_start_m": 0.0,
                            "x_end_m": 2.5,
                            "e_b": 1.1,
                            "e_p": 0.5,
                            "d_p": 0.0,
                            "values": {"Ni": 100000.0, "NiO": 0.0, "CaAl2O4": 0.0},
                        },
                    ],
                },
            }
            bundle = _make_bundle(Path(tmp), solids=solids)
            with self.assertRaises(PackedBedValidationError) as context:
                validate_run_bundle(bundle)
            self.assertIn("e_b must stay within", str(context.exception))
            self.assertIn("d_p must be positive", str(context.exception))

    def test_converts_solid_phase_profile_to_bed_volume_basis(self):
        solids = SolidConfig(
            solid_species=("Ni", "NiO"),
            concentration_unit="mol_per_m3_solid",
            initial_profile_zones=(
                SolidZoneConfig(
                    x_start_m=0.0,
                    x_end_m=2.0,
                    values_mol_per_m3={"Ni": 400000.0, "NiO": 0.0},
                    e_b=0.5,
                    e_p=0.5,
                    d_p=0.01,
                ),
            ),
        )
        converted = _convert_solid_profile_to_bed_volume(
            solids,
            cell_centers_m=[0.5, 1.5],
            solid_fraction=[0.25, 0.25],
            solid_species=("Ni", "NiO"),
        )
        self.assertEqual(converted.shape, (2, 2))
        self.assertAlmostEqual(float(converted[0, 0]), 100000.0)
        self.assertAlmostEqual(float(converted[0, 1]), 100000.0)

    def test_averages_face_particle_length_at_zone_boundary(self):
        solids = SolidConfig(
            solid_species=("Ni",),
            concentration_unit="mol_per_m3_solid",
            initial_profile_zones=(
                SolidZoneConfig(
                    x_start_m=0.0,
                    x_end_m=1.0,
                    values_mol_per_m3={"Ni": 100000.0},
                    e_b=0.45,
                    e_p=0.35,
                    d_p=0.008,
                ),
                SolidZoneConfig(
                    x_start_m=1.0,
                    x_end_m=2.0,
                    values_mol_per_m3={"Ni": 100000.0},
                    e_b=0.55,
                    e_p=0.25,
                    d_p=0.012,
                ),
            ),
        )
        d_p = build_face_scalar_profile(solids, face_positions_m=[0.0, 1.0, 2.0], attribute_name="d_p")
        self.assertAlmostEqual(float(d_p[0]), 0.008)
        self.assertAlmostEqual(float(d_p[1]), 0.01)
        self.assertAlmostEqual(float(d_p[2]), 0.012)


if __name__ == "__main__":
    unittest.main()
