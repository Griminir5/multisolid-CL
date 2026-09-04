from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from packed_bed.config import PackedBedValidationError, load_case
from packed_bed.initialization import calculate_initial_state
from packed_bed.properties import PROPERTY_REGISTRY
from packed_bed.solid_profiles import build_cell_profiles, build_face_scalar_profile
from test_config import _case_documents, _write_case


def _write_inert_case(tmp_path: Path, *, axial_cells: int = 3, inlet_flow: float = 1.0e-8) -> Path:
    documents = _case_documents()
    documents["program.yaml"]["inlet_flow"]["initial"] = inlet_flow
    documents["run.yaml"]["model"]["axial_cells"] = axial_cells
    documents["run.yaml"]["simulation"]["time_horizon_s"] = 0.01
    documents["run.yaml"]["simulation"]["reporting_interval_s"] = 0.01
    return _write_case(tmp_path, documents)


@pytest.mark.parametrize("inlet_flow", (1.0e-8, 0.1))
def test_initial_state_calculation_and_pressure_bracketing(tmp_path: Path, inlet_flow) -> None:
    case = load_case(_write_inert_case(tmp_path, inlet_flow=inlet_flow))
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
    assert np.all(state.gas_density_kg_m3 > 0.0)
    assert np.all(np.diff(state.gas_density_kg_m3) < 0.0)
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == paths_before


def test_daetools_grid_minimum_is_validated_before_initialization(tmp_path: Path) -> None:
    with pytest.raises(PackedBedValidationError, match="model.axial_cells"):
        load_case(_write_inert_case(tmp_path, axial_cells=2))


@pytest.mark.parametrize("basis, expected", (("bed", [1.0, 4.0]), ("solid", [0.3, 1.6])))
def test_cell_zone_endpoints_basis_and_face_averaging(tmp_path, basis, expected):
    documents = _case_documents()
    profile = documents["solids.yaml"]["initial_profile"]
    profile["basis"] = basis
    left = profile["zones"][0]
    left["x_end_m"] = 0.5
    right = dict(left, x_start_m=0.5, x_end_m=1.0, e_b=0.5, e_p=0.2, d_p=0.003, values={"Ni": 4.0})
    profile["zones"].append(right)
    solids = load_case(_write_case(tmp_path, documents)).solids
    e_b, e_p, concentrations = build_cell_profiles(solids, [0.0, 0.5 - 0.5e-12, 0.5, 1.0])
    np.testing.assert_allclose(e_b, [0.4, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(e_p, [0.5, 0.2, 0.2, 0.2])
    np.testing.assert_allclose(concentrations, [[expected[0], *[expected[1]] * 3]])
    np.testing.assert_allclose(
        build_face_scalar_profile(solids, [0.0, 0.5 - 0.5e-12, 0.5, 0.5 + 0.5e-12, 1.0], "d_p"),
        [0.001, 0.002, 0.002, 0.002, 0.003],
    )
    with pytest.raises(ValueError, match="cover"):
        build_cell_profiles(solids, [1.01])
