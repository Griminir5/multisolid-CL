from importlib.util import find_spec
from functools import partial
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from packed_bed.config import load_case
from packed_bed_ui.project import Project, input_hashes, read_json, write_json
from packed_bed_ui.worker import activate_snapshot, run_project_job, run_snapshot


def test_tampered_snapshot_fails_without_retry_in_place(tmp_path, source_case):
    project = Project.create(tmp_path / "project")
    case = project.add_case_from_files(source_case)
    job = read_json(project.prepare_execution([case]))
    folder = activate_snapshot(case.root / f".pending-{job['attempt_id']}")
    with (folder / "inputs/run.yaml").open("a") as stream:
        stream.write("\n# edited after snapshot\n")
    assert run_snapshot(folder) == 1
    status = read_json(folder / "status.json")
    assert status["state"] == "failed"
    assert "inputs have changed" in status["message"]
    assert run_snapshot(folder) == 1
    assert read_json(folder / "status.json") == status
    assert not (folder / "output").exists()


def fake_prepared_case(run_path, generate_artifacts_fn, run_case_fn, result_queue, *, barrier):
    from time import perf_counter, sleep
    folder = activate_snapshot(Path(run_path).parent.parent)
    snapshot = read_json(folder / "snapshot.json")
    started = perf_counter()
    write_json(folder / "status.json", {"state": "running", "started_at": started, "elapsed_s": 0.0})
    # Establish overlap deliberately; process startup can exceed the synthetic
    # work duration on a busy build host without violating the worker limit.
    barrier.wait(timeout=30)
    sleep(0.4)
    failed = snapshot["case_name"] == "First"
    write_json(folder / "status.json", {"state": "failed" if failed else "completed", "elapsed_s": 0.4})
    write_json(folder / "timing.json", {"start": started, "finish": perf_counter(),
                                        "pid": os.getpid(), "threads": os.environ["OMP_NUM_THREADS"]})
    result_queue.put("Synthetic failure" if failed else (folder / "output", {}, "not_requested", {}))


def slow_prepared_case(run_path, generate_artifacts_fn, run_case_fn, result_queue):
    from time import perf_counter, sleep
    folder = activate_snapshot(Path(run_path).parent.parent)
    write_json(folder / "status.json", {"state": "running", "started_at": perf_counter(), "elapsed_s": 0.0})
    write_json(folder / "pid.json", {"pid": os.getpid()})
    sleep(30)
    result_queue.put((folder / "output", {}, "not_requested", {}))


@pytest.mark.parametrize("workers", [1, 2])
def test_run_all_continues_after_failure_and_obeys_worker_and_thread_limits(tmp_path, source_case, workers):
    pytest.importorskip("PyQt6.QtCore")
    project = Project.create(tmp_path / "project")
    first = project.add_case_from_files(source_case, "First")
    for index in range(3):
        project.duplicate_case(first, f"Other {index}")
    for index, case in enumerate(project.cases):
        case.documents["run"]["solver"]["threads"] = 1 + index % 2
    path = project.prepare_execution(project.cases, max_workers=workers)
    worker = partial(fake_prepared_case, barrier=get_context("spawn").Barrier(workers))
    assert run_project_job(path, case_worker=worker) == 1
    assert [case.state()["state"] for case in project.cases] == ["failed", "completed", "completed", "completed"]
    assert read_json(path)["state"] == "failed"
    timing = [read_json(case.run_folder / "timing.json") for case in project.cases]
    assert [item["threads"] for item in timing] == ["1", "2", "1", "2"]
    concurrency = max(sum(other["start"] <= item["start"] < other["finish"] for other in timing) for item in timing)
    assert concurrency == workers


@pytest.mark.skipif(find_spec("daetools") is None, reason="DAETools is not installed")
@pytest.mark.parametrize("workers", [1, 2])
def test_real_project_worker_matches_engine_and_reruns_replace_results(tmp_path, source_case, workers):
    pytest.importorskip("PyQt6.QtCore")
    from packed_bed.reports import load_dataset
    from packed_bed.simulation import run_case

    stages = []
    expected = run_case(load_case(source_case), on_status=stages.append)
    assert stages == ["preparing", "initialising", "running", "writing_results"]
    project = Project.create(tmp_path / "project")
    first = project.add_case_from_files(source_case, "First")
    second = project.duplicate_case(first, "Second")
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    for attempt in range(2):
        job = project.prepare_execution(project.cases, max_workers=workers)
        completed = subprocess.run(
            [sys.executable, "-m", "packed_bed_ui", "--project-worker", str(job)],
            env=environment, capture_output=True, text=True, timeout=60,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr + job.read_text()
        assert read_json(job)["state"] == "completed"
        for case in project.cases:
            folder = case.run_folder
            assert case.state()["state"] == "completed"
            assert not case.state()["stale"]
            assert read_json(folder / "output/manifest.json")["status"] == "success"
            assert input_hashes(folder / "inputs") == read_json(folder / "snapshot.json")["input_hashes"]
            assert (folder / "worker.log").stat().st_size > 0
            assert not (folder / "old-result-marker").exists()
            (folder / "old-result-marker").write_text("remove on rerun")
            actual = load_dataset(next((folder / "output").glob("*.nc")))
            reference = load_dataset(expected.results_path)
            for name in reference.variables:
                if np.issubdtype(reference[name].dtype, np.number):
                    np.testing.assert_allclose(actual[name], reference[name], rtol=1e-10, atol=1e-12)
                else:
                    np.testing.assert_array_equal(actual[name], reference[name])
        assert not list(project.root.glob("cases/*/.pending-*"))
        assert not list(project.root.glob("cases/*/.previous-run"))


@pytest.mark.skipif(find_spec("daetools") is None, reason="DAETools is not installed")
def test_real_successful_baseline_generates_runs_and_rebuilds_study(tmp_path, source_case):
    pytest.importorskip("PyQt6.QtCore")
    from packed_bed_ui.studies import Factor
    from packed_bed_ui.study_store import baseline_eligibility

    project = Project.create(tmp_path / "project")
    baseline = project.add_case_from_files(source_case, "Benchmark")
    assert run_project_job(project.prepare_execution([baseline])) == 0
    assert baseline_eligibility(baseline).eligible
    study = project.study_store.create("Cell sweep", baseline)
    study.factors = [Factor("cells", "axial_cells", [3, 4])]
    project.study_store.save_study(study)
    cases = project.study_store.apply_preview(project.study_store.preview(study))
    assert run_project_job(project.prepare_execution(project.cases, max_workers=2)) == 0
    for case in cases:
        assert case.state()["state"] == "completed"
        assert not case.state()["stale"]
        assert (case.run_folder / "output/results.nc").exists()
    study.factors[0].values = [3, 5]
    project.study_store.save_study(study)
    replacements = project.study_store.apply_preview(project.study_store.preview(study))
    assert all(not case.root.exists() for case in cases)
    assert all(not case.run_folder.exists() for case in replacements)
    assert baseline.state()["state"] == "completed"
