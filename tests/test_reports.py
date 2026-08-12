from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from packed_bed.artifacts import generate_artifacts
from packed_bed.config import load_case
from packed_bed.plotting import PLOT_REGISTRY, _render_requested_plots
from packed_bed.reports import (
    REPORT_REGISTRY,
    RunResult,
    compute_balance_errors,
    extract_dataset,
    load_dataset,
    reporting_targets,
    write_dataset,
    write_run_manifest,
)
from test_config import _case_documents, _write_case


class FakeDomain:
    def __init__(self, points):
        self.Points = points


class FakeVariable:
    def __init__(self, time, values, domains=(), units=""):
        self.TimeValues = time
        self.Values = values
        self.Domains = tuple(FakeDomain(points) for points in domains)
        self.Units = units


class FakeProcess:
    def __init__(self, variables):
        self.dictVariables = {f"synthetic.{name}": value for name, value in variables.items()}


def _with_outputs(case, *, reports=None, plots=None):
    updates = {}
    if reports is not None:
        updates["requested_reports"] = tuple(reports)
    if plots is not None:
        updates["requested_plots"] = tuple(plots)
    outputs = case.run.outputs.model_copy(update=updates)
    return replace(case, run=case.run.model_copy(update={"outputs": outputs}))


def _synthetic_case_and_process(tmp_path: Path):
    documents = _case_documents()
    documents["chemistry.yaml"]["gas_species"] = ["N2", "H2"]
    documents["program.yaml"]["inlet_composition"]["initial"] = {"N2": 0.75, "H2": 0.25}
    documents["solids.yaml"]["solid_species"] = ["Ni", "NiO"]
    documents["solids.yaml"]["initial_profile"]["zones"][0]["values"] = {
        "Ni": 1.0,
        "NiO": 2.0,
    }
    case = load_case(_write_case(tmp_path, documents))
    case = replace(
        case,
        chemistry=case.chemistry.model_copy(update={"reaction_ids": ("reaction_1",)}),
    )

    time = np.array([0.0, 1.0, 1.0, 2.0])
    x_cell = np.array([1.0 / 6.0, 0.5, 5.0 / 6.0])
    x_face = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    gas_domain = np.arange(2)
    solid_domain = np.arange(2)
    reaction_domain = np.arange(1)
    gas_fraction = np.empty((4, 2, 3))
    gas_fraction[:, 0, :] = 0.6
    gas_fraction[:, 1, :] = 0.4
    gas_flux = np.ones((4, 2, 4))
    gas_flux[:, 0, :] = 2.0
    gas_flux[2, 0, -1] = 3.0
    gas_flux[3, :, -1] = 0.0
    solid_concentration = np.ones((4, 2, 3))
    solid_concentration[:, 1, :] = 3.0
    variables = {
        "T_in": FakeVariable(time, [690.0, 700.0, 710.0, 720.0], units="K"),
        "temp_bed": FakeVariable(time, 700.0 + np.arange(12).reshape(4, 3), (x_cell,), "K"),
        "P_in": FakeVariable(time, np.full(4, 100100.0), units="Pa"),
        "pres_bed": FakeVariable(time, 100000.0 + np.arange(12).reshape(4, 3), (x_cell,), "Pa"),
        "P_out": FakeVariable(time, np.full(4, 100000.0), units="Pa"),
        "u_s": FakeVariable(time, np.ones((4, 4)), (x_face,), "m/s"),
        "c_gas": FakeVariable(time, np.ones((4, 2, 3)), (gas_domain, x_cell), "mol/m^3"),
        "y_in": FakeVariable(time, np.tile([0.75, 0.25], (4, 1)), (gas_domain,), "1"),
        "y_gas": FakeVariable(time, gas_fraction, (gas_domain, x_cell), "1"),
        "c_sol": FakeVariable(time, solid_concentration, (solid_domain, x_cell), "mol/m^3"),
        "F_in": FakeVariable(time, [1.0, 1.1, 1.2, 1.3], units="mol/s"),
        "N_gas_face": FakeVariable(time, gas_flux, (gas_domain, x_face), "mol/(m^2 s)"),
        "R_rxn": FakeVariable(time, np.ones((4, 1, 3)), (reaction_domain, x_cell), "mol/(m^3 s)"),
        "J_gas_face": FakeVariable(time, np.ones((4, 2, 4)), (gas_domain, x_face), "W/m^2"),
        "heat_in_total": FakeVariable(time, [0.0, 10.0, 20.0, 30.0], units="J"),
        "heat_out_total": FakeVariable(time, [0.0, 1.0, 2.0, 3.0], units="J"),
        "heat_loss_total": FakeVariable(time, [0.0, 0.5, 1.0, 1.5], units="J"),
        "heat_bed_total": FakeVariable(time, [100.0, 108.5, 117.0, 125.5], units="J"),
        "mass_in_total": FakeVariable(time, [0.0, 1.0, 2.0, 3.0], units="kg"),
        "mass_out_total": FakeVariable(time, [0.0, 0.2, 0.4, 0.6], units="kg"),
        "mass_bed_total": FakeVariable(time, [10.0, 10.8, 11.6, 12.4], units="kg"),
    }
    return case, FakeProcess(variables)


EXPECTED_REPORTS = {
    "temperature": (
        ("T_in", "temp_bed"),
        {"inlet_temperature": ("time",), "temperature": ("time", "x_cell"), "outlet_temperature": ("time",)},
    ),
    "pressure": (
        ("P_in", "pres_bed", "P_out"),
        {"inlet_pressure": ("time",), "pressure": ("time", "x_cell"), "outlet_pressure": ("time",), "pressure_drop": ("time",)},
    ),
    "velocity": (("u_s",), {"velocity": ("time", "x_face")}),
    "gas_concentration": (("c_gas",), {"gas_concentration": ("time", "gas_species", "x_cell")}),
    "gas_mole_fraction": (
        ("y_in", "y_gas", "N_gas_face"),
        {"inlet_composition": ("time", "gas_species"), "gas_mole_fraction": ("time", "gas_species", "x_cell"), "outlet_composition": ("time", "gas_species")},
    ),
    "solid_concentration": (("c_sol",), {"solid_concentration": ("time", "solid_species", "x_cell")}),
    "solid_mole_fraction": (("c_sol",), {"solid_mole_fraction": ("time", "solid_species", "x_cell")}),
    "gas_flux": (
        ("F_in", "N_gas_face"),
        {"inlet_flow": ("time",), "gas_flux": ("time", "gas_species", "x_face"), "outlet_species_flow": ("time", "gas_species"), "outlet_flow": ("time",)},
    ),
    "reaction_rate": (("R_rxn",), {"reaction_rate": ("time", "reaction", "x_cell")}),
    "gas_enthalpy_flux": (("J_gas_face",), {"gas_enthalpy_flux": ("time", "gas_species", "x_face")}),
    "heat_balance": (
        ("heat_in_total", "heat_out_total", "heat_loss_total", "heat_bed_total"),
        {name: ("time",) for name in ("heat_in_total", "heat_out_total", "heat_loss_total", "heat_bed_total", "heat_balance_error")},
    ),
    "mass_balance": (
        ("mass_in_total", "mass_out_total", "mass_bed_total"),
        {name: ("time",) for name in ("mass_in_total", "mass_out_total", "mass_bed_total", "mass_balance_error")},
    ),
}


@pytest.mark.parametrize(("report_id", "expected"), EXPECTED_REPORTS.items())
def test_each_report_owns_exact_sources_outputs_dimensions_and_units(
    tmp_path: Path, report_id: str, expected
) -> None:
    case, process = _synthetic_case_and_process(tmp_path)
    case = _with_outputs(case, reports=(report_id,))
    expected_sources, expected_outputs = expected

    dataset = extract_dataset(process, case)

    assert reporting_targets((report_id,)) == expected_sources
    assert {field.name: field.dimensions for field in REPORT_REGISTRY[report_id].outputs} == expected_outputs
    assert {name: variable.dims for name, variable in dataset.data_vars.items()} == expected_outputs
    assert all("units" in variable.attrs for variable in dataset.data_vars.values())
    assert all(
        "source_variable" in variable.attrs or "derived_from" in variable.attrs
        for variable in dataset.data_vars.values()
    )
    assert set(dataset.coords) == {dimension for dimensions in expected_outputs.values() for dimension in dimensions}


def test_gas_mole_fraction_uses_model_boundaries_and_hides_flux_support(tmp_path: Path) -> None:
    case, process = _synthetic_case_and_process(tmp_path)
    case = _with_outputs(case, reports=("gas_mole_fraction",))

    dataset = extract_dataset(process, case)

    assert dataset.inlet_composition.values.tolist() == [[0.75, 0.25]] * 3
    assert np.allclose(
        dataset.outlet_composition.values,
        [[2.0 / 3.0, 1.0 / 3.0], [0.75, 0.25], [0.6, 0.4]],
    )
    assert "gas_flux" not in dataset
    assert "x_face" not in dataset.coords


def test_combined_dataset_normalizes_time_and_derives_boundaries(tmp_path: Path) -> None:
    case, process = _synthetic_case_and_process(tmp_path)
    case = _with_outputs(case, reports=EXPECTED_REPORTS)

    dataset = extract_dataset(process, case)

    assert dataset.time.values.tolist() == [0.0, 1.0, 2.0]
    assert dataset.gas_species.values.tolist() == ["N2", "H2"]
    assert dataset.temperature.sel(time=1.0, x_cell=0.5) == 707.0
    assert dataset.inlet_temperature.values.tolist() == [690.0, 710.0, 720.0]
    assert dataset.outlet_temperature.values.tolist() == [702.0, 708.0, 711.0]
    assert dataset.inlet_pressure.values.tolist() == [100100.0] * 3
    assert dataset.outlet_pressure.values.tolist() == [100000.0] * 3
    assert dataset.inlet_flow.values.tolist() == [1.0, 1.2, 1.3]
    assert np.allclose(dataset.pressure_drop, 100.0)
    xr.testing.assert_allclose(
        dataset.solid_mole_fraction.sum("solid_species"),
        xr.ones_like(dataset.solid_mole_fraction.isel(solid_species=0, drop=True)),
    )
    assert set(compute_balance_errors(dataset)) == {"heat", "mass"}
    assert reporting_targets(("gas_mole_fraction", "gas_flux")).count("N_gas_face") == 1


def test_axial_heatmap_time_edges_are_clamped_to_the_simulation_interval() -> None:
    from packed_bed.plotting.axial_profiles import _edges

    assert _edges([0.0, 1.0, 2.0], clamp=True).tolist() == [0.0, 0.5, 1.5, 2.0]


def test_plot_selection_cannot_change_the_dataset(tmp_path: Path) -> None:
    case, process = _synthetic_case_and_process(tmp_path)
    reports = ("temperature", "pressure", "gas_mole_fraction", "gas_flux")
    first = extract_dataset(process, _with_outputs(case, reports=reports, plots=()))
    second = extract_dataset(
        process,
        _with_outputs(case, reports=reports, plots=PLOT_REGISTRY),
    )

    assert first.identical(second)


def test_netcdf_only_plots_continue_after_one_failure(tmp_path: Path, monkeypatch) -> None:
    case, process = _synthetic_case_and_process(tmp_path)
    reports = ("temperature", "pressure", "gas_mole_fraction", "gas_flux")
    dataset = extract_dataset(process, _with_outputs(case, reports=reports))
    results_path = write_dataset(dataset, tmp_path / "output" / "results.nc")

    def fail(_dataset, _path):
        raise RuntimeError("synthetic plot failure")

    registry = dict(PLOT_REGISTRY)
    registry["outlet_composition"] = replace(registry["outlet_composition"], render=fail)
    monkeypatch.setattr("packed_bed.plotting.PLOT_REGISTRY", registry)
    plotted = _render_requested_plots(
        results_path,
        ("outlet_composition", "axial_profiles"),
        tmp_path / "artifacts",
    )

    assert plotted.errors == {"outlet_composition": "synthetic plot failure"}
    assert plotted.paths["axial_profiles"].is_file()
    assert load_dataset(results_path).identical(dataset)


def test_netcdf_round_trip_manifest_and_ml_conversion(tmp_path: Path) -> None:
    case, process = _synthetic_case_and_process(tmp_path)
    reports = ("temperature", "pressure", "gas_mole_fraction", "gas_flux")
    case = _with_outputs(case, reports=reports, plots=PLOT_REGISTRY)
    dataset = extract_dataset(process, case)
    results_path = write_dataset(dataset, tmp_path / "output" / "results.nc")
    loaded = load_dataset(results_path)
    xr.testing.assert_allclose(dataset, loaded)
    plots = _render_requested_plots(results_path, PLOT_REGISTRY, tmp_path / "artifacts")
    result = RunResult(
        case=case,
        output_directory=results_path.parent,
        results_path=results_path,
        runtime_s=1.25,
        balance_errors=compute_balance_errors(results_path),
        artifact_paths=plots.paths,
    )
    manifest_path = write_run_manifest(result)

    assert all(path.is_file() for path in plots.paths.values())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["outputs"]["results"]["sha256"]
    assert manifest["dataset"]["dimensions"]["gas_species"] == 2
    assert manifest["configuration"]["run"]["solver"]["suppress_algebraic_errors"] is False
    assert manifest["configuration"]["program"]["inlet_composition"]["initial"] == {
        "N2": 0.75,
        "H2": 0.25,
    }
    assert manifest["plots"]["status"] == "success"
    failed_path = write_run_manifest(
        replace(result, status="failed"),
        failure_stage="solver execution",
        traceback_text="synthetic traceback",
    )
    assert json.loads(failed_path.read_text(encoding="utf-8"))["failure"] == {
        "stage": "solver execution",
        "traceback": "synthetic traceback",
    }

    matrix_path = tmp_path / "matrix.csv"
    subprocess.run(
        [
            sys.executable,
            "tools/to_ml_matrix.py",
            str(results_path),
            str(matrix_path),
            "--variables",
            "temperature",
            "pressure",
        ],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert matrix_path.is_file()


def test_pre_run_artifacts_use_only_the_resolved_case(tmp_path: Path, monkeypatch) -> None:
    case, _process = _synthetic_case_and_process(tmp_path)
    monkeypatch.setattr("packed_bed.artifacts.find_spec", lambda _name: None)

    artifacts = generate_artifacts(case)

    assert set(artifacts) == {"initial_solid_profile_svg", "operating_program_svg"}
    assert all(path.is_file() for path in artifacts.values())


def test_empty_report_selection_writes_only_scheduled_time(tmp_path: Path) -> None:
    documents = _case_documents()
    documents["run.yaml"]["simulation"]["reporting_interval_s"] = 3.0
    case = load_case(_write_case(tmp_path, documents))

    dataset = extract_dataset(FakeProcess({}), case)

    assert dataset.time.values.tolist() == [0.0, 3.0, 6.0, 9.0, 10.0]
    assert not dataset.data_vars
    assert set(dataset.coords) == {"time"}
