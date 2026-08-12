from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec
import json
from pathlib import Path
from types import SimpleNamespace

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


def _with_reports(case, reports):
    outputs = case.run.outputs.model_copy(update={"requested_reports": tuple(reports)})
    return replace(case, run=case.run.model_copy(update={"outputs": outputs}))


def _with_plots(case, plots):
    outputs = case.run.outputs.model_copy(update={"requested_plots": tuple(plots)})
    return replace(case, run=case.run.model_copy(update={"outputs": outputs}))


def _with_interior_flow_mode(case, mode):
    simulation = case.run.simulation.model_copy(update={"interior_flow_mode": mode})
    return replace(case, run=case.run.model_copy(update={"simulation": simulation}))


@pytest.mark.skipif(find_spec("daetools") is None, reason="DAETools is not installed")
def test_configure_idas_maps_every_per_case_control(monkeypatch) -> None:
    import packed_bed.simulation as simulation_module

    observed = {}

    class RecordingConfig:
        def SetBoolean(self, name, value):
            observed[name] = value

        def SetInteger(self, name, value):
            observed[name] = value

        def SetFloat(self, name, value):
            observed[name] = value

    monkeypatch.setattr(simulation_module, "daeGetConfig", RecordingConfig)
    simulation_module.configure_idas(
        SimpleNamespace(
            suppress_algebraic_errors=True,
            max_nonlinear_iterations=12,
            nonlinear_convergence_coefficient=1.0,
        )
    )

    assert observed == {
        "daetools.IDAS.SuppressAlg": True,
        "daetools.IDAS.MaxNonlinIters": 12,
        "daetools.IDAS.NonlinConvCoef": 1.0,
    }


@pytest.mark.skipif(find_spec("daetools") is None, reason="DAETools is not installed")
def test_tiny_inert_case_preserves_equation_order_and_executes(tmp_path: Path) -> None:
    from packed_bed.reports import create_dataset_reporter, load_dataset
    from packed_bed.simulation import PackedBedSimulation, execute_simulation

    case = _with_reports(
        load_case(_write_inert_case(tmp_path)),
        ("solid_mole_fraction", "heat_balance", "mass_balance"),
    )
    simulation = PackedBedSimulation(case, PROPERTY_REGISTRY)
    reporter = create_dataset_reporter(case)
    equation_names = ()

    def record_equations(initialized_simulation, _solver):
        nonlocal equation_names
        equation_names = tuple(
            equation.Name for equation in initialized_simulation.model.Equations
        )

    execute_simulation(
        simulation,
        data_reporter=reporter,
        after_initialize=record_equations,
    )

    assert reporter.results_path.is_file()
    assert "solid_mole_fraction" in load_dataset(reporter.results_path)
    assert equation_names == EXPECTED_INERT_EQUATIONS


@pytest.mark.skipif(find_spec("daetools") is None, reason="DAETools is not installed")
def test_ordinary_run_writes_one_dataset_and_manifest(tmp_path: Path) -> None:
    from packed_bed.reports import load_dataset
    from packed_bed.simulation import run_case

    case = load_case(_write_inert_case(tmp_path))
    result = run_case(case, property_registry=PROPERTY_REGISTRY)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.status == "success"
    assert result.results_path.is_file()
    assert not hasattr(result, "dataset")
    assert set(load_dataset(result.results_path).coords) == {"time"}
    assert not load_dataset(result.results_path).data_vars
    assert result.balance_errors == {}
    assert manifest["status"] == "success"
    assert manifest["outputs"]["results"]["path"] == str(result.results_path)


@pytest.mark.skipif(find_spec("daetools") is None, reason="DAETools is not installed")
def test_dae_plotter_retention_cannot_change_reports_or_netcdf(tmp_path: Path) -> None:
    from packed_bed.reports import load_dataset
    from packed_bed.simulation import run_case

    ordinary_dir = tmp_path / "ordinary"
    retained_dir = tmp_path / "retained"
    ordinary_dir.mkdir()
    retained_dir.mkdir()
    ordinary_case = _with_reports(
        load_case(_write_inert_case(ordinary_dir)),
        ("gas_mole_fraction",),
    )
    retained_case = _with_reports(
        load_case(_write_inert_case(retained_dir)),
        ("gas_mole_fraction",),
    )

    ordinary = run_case(ordinary_case, property_registry=PROPERTY_REGISTRY)
    retained = run_case(
        retained_case,
        property_registry=PROPERTY_REGISTRY,
        retain_reporter=True,
    )

    assert ordinary.reporter is None
    assert retained.reporter is not None
    assert load_dataset(ordinary.results_path).identical(load_dataset(retained.results_path))


@pytest.mark.skipif(find_spec("daetools") is None, reason="DAETools is not installed")
def test_plot_failure_preserves_successful_simulation_and_netcdf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import packed_bed.plotting as plotting
    from packed_bed.reports import load_dataset
    from packed_bed.simulation import run_case

    case = _with_plots(
        _with_reports(load_case(_write_inert_case(tmp_path)), ("temperature", "pressure")),
        ("axial_profiles",),
    )

    def fail(_dataset, _path):
        raise RuntimeError("synthetic plot failure")

    registry = dict(plotting.PLOT_REGISTRY)
    registry["axial_profiles"] = replace(registry["axial_profiles"], render=fail)
    monkeypatch.setattr(plotting, "PLOT_REGISTRY", registry)

    result = run_case(case, property_registry=PROPERTY_REGISTRY)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.status == "success"
    assert result.results_path.is_file()
    assert {"temperature", "pressure"} <= set(load_dataset(result.results_path))
    assert result.plot_status == "failed"
    assert result.plot_errors == {"axial_profiles": "synthetic plot failure"}
    assert manifest["status"] == "success"
    assert manifest["plots"]["status"] == "failed"
    assert manifest["outputs"]["axial_profiles"] == {
        "status": "failed",
        "error": "synthetic plot failure",
    }


@pytest.mark.parametrize(
    ("reports", "row_count", "accounting_equations"),
    (
        ((), 57, ()),
        (("mass_balance",), 60, (
            "mass_in_total_accumulation",
            "mass_out_total_accumulation",
            "mass_bed_total_definition",
        )),
        (("heat_balance",), 61, (
            "heat_in_total_accumulation",
            "heat_out_total_accumulation",
            "heat_loss_total_accumulation",
            "heat_bed_total_definition",
        )),
        (("mass_balance", "heat_balance"), 64, (
            "mass_in_total_accumulation",
            "mass_out_total_accumulation",
            "mass_bed_total_definition",
            "heat_in_total_accumulation",
            "heat_out_total_accumulation",
            "heat_loss_total_accumulation",
            "heat_bed_total_definition",
        )),
    ),
)
@pytest.mark.skipif(find_spec("daetools") is None, reason="DAETools is not installed")
def test_requested_balances_control_accounting_dae_size(
    tmp_path: Path,
    reports,
    row_count: int,
    accounting_equations,
) -> None:
    from packed_bed.incidence_matrix import collect_solver_incidence_matrix
    from packed_bed.reports import create_dataset_reporter, load_dataset
    from packed_bed.simulation import PackedBedSimulation, execute_simulation

    case = _with_reports(load_case(_write_inert_case(tmp_path)), reports)
    simulation = PackedBedSimulation(case, PROPERTY_REGISTRY)
    reporter = create_dataset_reporter(case)
    observed = {}

    def inspect(initialized_simulation, _solver):
        model = initialized_simulation.model
        observed["rows"] = collect_solver_incidence_matrix(model).row_count
        observed["accounting"] = tuple(
            equation.Name
            for equation in model.Equations
            if equation.Name.startswith(("mass_", "heat_"))
        )
        observed["mass_variables"] = all(
            name in model.dictVariables
            for name in ("mass_in_total", "mass_out_total", "mass_bed_total")
        )
        observed["heat_variables"] = all(
            name in model.dictVariables
            for name in (
                "heat_in_total",
                "heat_out_total",
                "heat_loss_total",
                "heat_bed_total",
            )
        )

    execute_simulation(simulation, data_reporter=reporter, after_initialize=inspect)

    assert observed == {
        "rows": row_count,
        "accounting": accounting_equations,
        "mass_variables": "mass_balance" in reports,
        "heat_variables": "heat_balance" in reports,
    }
    dataset = load_dataset(reporter.results_path)
    assert ("mass_balance_error" in dataset) == ("mass_balance" in reports)
    assert ("heat_balance_error" in dataset) == ("heat_balance" in reports)


@pytest.mark.parametrize(
    ("mode", "nonzero_count"),
    (("forward_only", 203), ("reversible", 207)),
)
@pytest.mark.skipif(find_spec("daetools") is None, reason="DAETools is not installed")
def test_interior_flow_mode_controls_solver_incidence(
    tmp_path: Path,
    mode: str,
    nonzero_count: int,
) -> None:
    from packed_bed.incidence_matrix import collect_solver_incidence_matrix
    from packed_bed.simulation import PackedBedSimulation, execute_simulation

    case = _with_reports(load_case(_write_inert_case(tmp_path)), ("mass_balance", "heat_balance"))
    case = _with_interior_flow_mode(case, mode)
    simulation = PackedBedSimulation(case, PROPERTY_REGISTRY)
    observed = {}

    def inspect(initialized_simulation, _solver):
        matrix = collect_solver_incidence_matrix(initialized_simulation.model)
        observed.update(
            rows=matrix.row_count,
            columns=matrix.column_count,
            nonzeros=matrix.nonzero_count,
        )

    execute_simulation(simulation, after_initialize=inspect)

    assert observed == {"rows": 64, "columns": 64, "nonzeros": nonzero_count}
