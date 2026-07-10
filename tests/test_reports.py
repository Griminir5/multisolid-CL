from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import xarray as xr

from packed_bed.config import load_case
from packed_bed.plots import render_run_result_plots
from packed_bed.reports import (
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


def _synthetic_case_and_process(tmp_path: Path):
    documents = _case_documents()
    documents["chemistry.yaml"]["gas_species"] = ["N2", "H2"]
    documents["program.yaml"]["inlet_composition"]["initial"] = {"N2": 0.75, "H2": 0.25}
    documents["solids.yaml"]["solid_species"] = ["Ni", "NiO"]
    documents["solids.yaml"]["initial_profile"]["zones"][0]["values"] = {
        "Ni": 1.0,
        "NiO": 2.0,
    }
    documents["run.yaml"]["outputs"]["requested_reports"] = [
        "temperature",
        "pressure",
        "velocity",
        "gas_mole_fraction",
        "solid_mole_fraction",
        "gas_flux",
        "heat_balance",
        "mass_balance",
    ]
    case = load_case(_write_case(tmp_path, documents))

    time = np.array([0.0, 1.0, 1.0, 2.0])
    x_cell = np.array([1.0 / 6.0, 0.5, 5.0 / 6.0])
    x_face = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    gas_domain = np.arange(2)
    solid_domain = np.arange(2)
    temperature = 700.0 + np.arange(12).reshape(4, 3)
    pressure = 100000.0 + np.arange(12).reshape(4, 3)
    gas_fraction = np.empty((4, 2, 3))
    gas_fraction[:, 0, :] = 0.75
    gas_fraction[:, 1, :] = 0.25
    gas_flux = np.ones((4, 2, 4))
    gas_flux[:, 1, :] = 0.5
    solid_concentration = np.ones((4, 2, 3))
    solid_concentration[:, 1, :] = 3.0

    variables = {
        "temp_bed": FakeVariable(time, temperature, (x_cell,), "K"),
        "pres_bed": FakeVariable(time, pressure, (x_cell,), "Pa"),
        "u_s": FakeVariable(time, np.ones((4, 4)), (x_face,), "m/s"),
        "y_gas": FakeVariable(time, gas_fraction, (gas_domain, x_cell), ""),
        "c_sol": FakeVariable(
            time,
            solid_concentration,
            (solid_domain, x_cell),
            "mol/m**3",
        ),
        "N_gas_face": FakeVariable(
            time,
            gas_flux,
            (gas_domain, x_face),
            "mol/(m**2 * s)",
        ),
        "P_in": FakeVariable(time, np.full(4, 100100.0), units="Pa"),
        "P_out": FakeVariable(time, np.full(4, 100000.0), units="Pa"),
        "heat_in_total": FakeVariable(time, [0.0, 10.0, 20.0, 30.0], units="J"),
        "heat_out_total": FakeVariable(time, [0.0, 1.0, 2.0, 3.0], units="J"),
        "heat_loss_total": FakeVariable(time, [0.0, 0.5, 1.0, 1.5], units="J"),
        "heat_bed_total": FakeVariable(time, [100.0, 108.5, 117.0, 125.5], units="J"),
        "mass_in_total": FakeVariable(time, [0.0, 1.0, 2.0, 3.0], units="kg"),
        "mass_out_total": FakeVariable(time, [0.0, 0.2, 0.4, 0.6], units="kg"),
        "mass_bed_total": FakeVariable(time, [10.0, 10.8, 11.6, 12.4], units="kg"),
    }
    return case, FakeProcess(variables)


def test_one_dataset_extracts_labels_derivations_and_balances(tmp_path: Path) -> None:
    case, process = _synthetic_case_and_process(tmp_path)

    dataset = extract_dataset(process, case)

    assert dict(dataset.sizes) == {
        "time": 3,
        "x_cell": 3,
        "x_face": 4,
        "gas_species": 2,
        "solid_species": 2,
    }
    assert dataset.time.values.tolist() == [0.0, 1.0, 2.0]
    assert dataset.gas_species.values.tolist() == ["N2", "H2"]
    assert dataset.temperature.sel(time=1.0, x_cell=case.run.model.bed_length_m / 2) == 707.0
    assert "solid_concentration" not in dataset
    xr.testing.assert_allclose(
        dataset.solid_mole_fraction.sum("solid_species"),
        xr.ones_like(dataset.solid_mole_fraction.isel(solid_species=0, drop=True)),
    )
    assert np.allclose(dataset.pressure_drop, 100.0)
    assert set(compute_balance_errors(dataset)) == {"heat", "mass"}


def test_netcdf_round_trip_plots_manifest_and_ml_conversion(tmp_path: Path) -> None:
    case, process = _synthetic_case_and_process(tmp_path)
    dataset = extract_dataset(process, case)
    results_path = write_dataset(dataset, tmp_path / "output" / "results.nc")
    loaded = load_dataset(results_path)
    xr.testing.assert_allclose(dataset, loaded)

    result = RunResult(
        case=case,
        output_directory=results_path.parent,
        results_path=results_path,
        runtime_s=1.25,
        dataset=loaded,
        balance_errors=compute_balance_errors(loaded),
    )
    plots = render_run_result_plots(result, image_format="png")
    result = replace(result, artifact_paths=plots)
    manifest_path = write_run_manifest(result)

    assert all(path.is_file() for path in plots.values())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["outputs"]["results"]["sha256"]
    assert manifest["dataset"]["dimensions"]["gas_species"] == 2
    assert manifest["environment"]["packages"]["xarray"]

    failed_path = write_run_manifest(
        replace(result, status="failed"),
        failure_stage="solver execution",
        traceback_text="synthetic traceback",
    )
    failed_manifest = json.loads(failed_path.read_text(encoding="utf-8"))
    assert failed_manifest["failure"] == {
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


def test_solid_mole_fraction_reports_concentration_without_changing_the_model() -> None:
    variables = reporting_targets(("solid_mole_fraction",))

    assert variables == ("c_sol",)


def test_empty_report_selection_writes_only_program_data(tmp_path: Path) -> None:
    documents = _case_documents()
    documents["run.yaml"]["simulation"]["reporting_interval_s"] = 3.0
    case = load_case(_write_case(tmp_path, documents))

    dataset = extract_dataset(FakeProcess({}), case)

    assert dataset.time.values.tolist() == [0.0, 3.0, 6.0, 9.0, 10.0]
    assert set(dataset.data_vars) == {
        "inlet_flow",
        "inlet_temperature",
        "programmed_outlet_pressure",
        "inlet_composition",
    }


def test_solid_concentration_does_not_create_an_unrequested_fraction(tmp_path: Path) -> None:
    case, process = _synthetic_case_and_process(tmp_path)
    outputs = case.run.outputs.model_copy(
        update={"requested_reports": ("solid_concentration",)}
    )
    case = replace(case, run=case.run.model_copy(update={"outputs": outputs}))

    dataset = extract_dataset(process, case)

    assert "solid_concentration" in dataset
    assert "solid_mole_fraction" not in dataset
