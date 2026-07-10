from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

from packed_bed.config import load_case
from packed_bed.properties import PROPERTY_REGISTRY
from test_initialization import _write_inert_case


EXPECTED_INERT_EQUATIONS = (
    "species_balance_cell_0_N2",
    "species_balance_cell_1_N2",
    "species_balance_cell_2_N2",
    "solid_species_balance_cell_0_Ni",
    "solid_species_balance_cell_1_Ni",
    "solid_species_balance_cell_2_Ni",
    "total_concentration_closure",
    "solid_total_concentration_closure",
    "molar_fraction_calc",
    "lhs_boundary_flux_N2",
    "face_flux_1_N2",
    "face_flux_2_N2",
    "rhs_boundary_flux_N2",
    "gas_component_enthalpy_N2",
    "energy_balance_cell_0",
    "energy_balance_cell_1",
    "energy_balance_cell_2",
    "total_cell_enthalpy",
    "solid_component_enthalpy_Ni",
    "lhs_boundary_enthalpy_flux_N2",
    "face_enthalpy_flux_1_N2",
    "face_enthalpy_flux_2_N2",
    "rhs_boundary_enthalpy_flux_N2",
    "axial_dispersion_face",
    "ergun_face_0",
    "ergun_face_1",
    "ergun_face_2",
    "ergun_face_3",
    "gas_equation_of_state",
    "gas_mixture_viscosity",
    "gas_density_closure",
    "mass_in_total_accumulation",
    "mass_out_total_accumulation",
    "mass_bed_total_definition",
    "heat_in_total_accumulation",
    "heat_out_total_accumulation",
    "heat_loss_total_accumulation",
    "heat_bed_total_definition",
    "Active_inlet_flow_smooth",
    "Active_inlet_composition_0_smooth",
    "Active_inlet_temperature_smooth",
    "inlet_pressure_from_flow",
    "Active_outlet_pressure_smooth",
)


@pytest.mark.skipif(find_spec("daetools") is None, reason="DAETools is not installed")
def test_tiny_inert_case_preserves_equation_order_and_executes(tmp_path: Path) -> None:
    from packed_bed.cli import run_simulation

    case = load_case(_write_inert_case(tmp_path))
    result = run_simulation(case, property_registry=PROPERTY_REGISTRY)
    equation_names = tuple(equation.Name for equation in result.simulation.model.Equations)

    assert result.success
    assert equation_names == EXPECTED_INERT_EQUATIONS
