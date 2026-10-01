import numpy as np
import gzip
import json
import os
import subprocess
import sys
import pytest
import xarray as xr
from types import SimpleNamespace

from tools.solver_study import WindowsJob, assess_errors, compare, compare_datasets, dataset_metrics, write_json


def example():
    return xr.Dataset({"temperature": (("time", "x_cell"), [[800., 801.], [805., 806.]]),
                       "outlet_species_flow": (("time", "gas_species"), [[1., 2.], [3., 4.]])},
                      coords={"time": [0., 2.], "x_cell": [.25, .75], "gas_species": ["H2", "N2"]})


def test_comparison_detects_local_error_and_rejects_nonfinite():
    reference = example()
    actual = reference.copy(deep=True)
    actual.temperature.values[1, 0] += 2.
    assert compare_datasets(reference, actual)["temperature"]["max_abs"] == 2.
    actual.temperature.values[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_datasets(reference, actual)


def test_comparison_checks_integrated_outlet_amount():
    reference = example()
    actual = reference.copy(deep=True)
    actual.outlet_species_flow.values *= 1.02
    error = compare_datasets(reference,actual)["net_outlet_amount"]
    assert error["normalized_error"] == pytest.approx(.02)
    assert error["total_and_quarter_max_absolute"][0] == pytest.approx(.12)
    assert error["mixed_tolerance_score"] > 1


def test_near_zero_product_uses_stream_scaled_absolute_allowance():
    reference = example()
    reference.outlet_species_flow.values[:, 0] = 0.
    actual = reference.copy(deep=True)
    actual.outlet_species_flow.values[:, 0] = 1e-7
    error = compare_datasets(reference, actual)["net_outlet_amount"]
    assert error["normalized_error"] > .001
    assert error["mixed_tolerance_score"] < 1
    actual.outlet_species_flow.values[:, 0] = .01
    assert compare_datasets(reference, actual)["net_outlet_amount"]["mixed_tolerance_score"] > 1


@pytest.mark.parametrize("coord,values", [("time", [0., 2.01]), ("gas_species", ["N2", "H2"])])
def test_comparison_rejects_misaligned_coordinates(coord, values):
    with pytest.raises(AssertionError):
        compare_datasets(example(), example().assign_coords({coord: values}))


def test_metrics_integrate_species_and_preserve_finiteness(tmp_path):
    path = tmp_path / "results.nc"
    example().to_netcdf(path, engine="scipy")
    metrics = dataset_metrics(path)
    assert metrics["finite"]
    assert metrics["time_end"] == 2.
    assert metrics["outlet_species_integrals"] == [4., 6.]


def test_pressure_drop_is_judged_on_its_own_scale_and_balance_is_checked():
    errors = {"pressure": dict(max_abs=10.,reference_scale=5e6),
              "pressure_drop": dict(max_abs=10.,reference_scale=100.)}
    assessment = assess_errors(errors,{"mass_balance":{"max_normalized":1e-3}})
    assert assessment["checks"]["pressure"]
    assert not assessment["checks"]["pressure_drop"]
    assert not assessment["checks"]["mass_balance"]
    assert not assessment["passed"]


def test_parallel_archive_comparison_matches_serial_and_invalidates_stale_cache(tmp_path):
    cases = [dict(id=f"case_{i}", tags=[]) for i in range(3)]
    write_json(tmp_path / 'study.json', dict(cases=cases))
    for i, case in enumerate(cases):
        for profile in ['reference', 'candidate_a', 'candidate_b']:
            folder = tmp_path / 'runs' / case['id'] / profile / '0'
            folder.mkdir(parents=True)
            dataset = example()
            if profile != 'reference':
                dataset.temperature.values[1, 0] += .2 * i
            plain = folder / 'results.nc'
            dataset.to_netcdf(plain, engine='scipy')
            archive = folder / 'results.nc.gz'
            archive.write_bytes(gzip.compress(plain.read_bytes()))
            write_json(folder / 'result.json', dict(case=case['id'], profile=profile, iteration=0,
                status='success', signature=case['id'] + profile, engine_sha256='fixture',
                dataset=str(archive), metrics=dataset_metrics(plain)))
    args = SimpleNamespace(root=tmp_path, reference='reference', workers=1)
    compare(args)
    destination = tmp_path / 'comparisons_reference.json'
    serial = json.loads(destination.read_text())
    destination.unlink()
    args.workers = 3
    compare(args)
    assert json.loads(destination.read_text()) == serial
    assert len(serial) == 6
    assert not serial[-1]['assessment']['passed']

    # A changed measurement must not inherit the earlier comparison result.
    folder = tmp_path / 'runs/case_2/candidate_b/0'
    row = json.loads((folder / 'result.json').read_text())
    plain = folder / 'results.nc'
    example().to_netcdf(plain, engine='scipy')
    (folder / 'results.nc.gz').write_bytes(gzip.compress(plain.read_bytes()))
    row.update(signature='replacement', metrics=dataset_metrics(plain))
    write_json(folder / 'result.json', row)
    compare(args)
    updated = json.loads(destination.read_text())
    assert updated[:-1] == serial[:-1]
    assert updated[-1]['assessment']['passed']


@pytest.mark.skipif(os.name != "nt",reason="Windows process containment")
def test_windows_job_terminates_a_worker():
    process = subprocess.Popen([sys._base_executable,"-c","import time; time.sleep(30)"])
    job = None
    try:
        job = WindowsJob(process)
        job.terminate()
        assert process.wait(timeout=5) == 124
    finally:
        if job: job.close()
        if process.poll() is None: process.kill()
