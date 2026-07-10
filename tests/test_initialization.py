from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from packed_bed.config import PackedBedValidationError, load_case
from packed_bed.initialization import calculate_initial_state
from packed_bed.properties import PROPERTY_REGISTRY
from test_config import _case_documents, _write_case


def _write_inert_case(tmp_path: Path, *, axial_cells: int = 3) -> Path:
    documents = _case_documents()
    documents["program.yaml"]["inlet_flow"]["initial"] = 1.0e-8
    documents["run.yaml"]["model"]["axial_cells"] = axial_cells
    documents["run.yaml"]["simulation"]["time_horizon_s"] = 0.01
    documents["run.yaml"]["simulation"]["reporting_interval_s"] = 0.01
    return _write_case(tmp_path, documents)


def test_initial_state_calculation_is_pure_and_consistently_shaped(tmp_path: Path) -> None:
    case = load_case(_write_inert_case(tmp_path))
    paths_before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    state = calculate_initial_state(case, PROPERTY_REGISTRY)

    assert state.face_coordinates_m.shape == (4,)
    assert state.interparticle_voidage.shape == (3,)
    assert state.particle_diameter_m.shape == (4,)
    assert state.gas_concentration_mol_m3.shape == (1, 3)
    assert state.solid_concentration_mol_m3.shape == (1, 3)
    assert state.face_velocity_m_s.shape == (4,)
    assert np.all(np.isfinite(state.face_velocity_m_s))
    assert state.inlet_pressure_pa > state.outlet_pressure_pa
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == paths_before


def test_daetools_grid_minimum_is_validated_before_initialization(tmp_path: Path) -> None:
    with pytest.raises(PackedBedValidationError, match="model.axial_cells"):
        load_case(_write_inert_case(tmp_path, axial_cells=2))
