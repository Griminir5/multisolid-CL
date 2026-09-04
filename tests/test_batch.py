from __future__ import annotations

import csv
from functools import partial
import json
from multiprocessing import active_children, get_context
import os
from pathlib import Path
from time import perf_counter, sleep

import pytest
import yaml

from packed_bed.batch import (
    BatchValidationError,
    expand_batch_cases,
    load_batch_spec,
    run_batch_file,
)
from packed_bed import batch
from packed_bed.config import resolve_case
from packed_bed.reports import RunResult


BASE_CASE = (
    Path(__file__).parents[1]
    / "packed_bed"
    / "examples"
    / "default_batch_case"
    / "base_case"
    / "run.yaml"
).resolve()


def _parallel_fake_run(case, *, barrier, **_kwargs):
    started_at = perf_counter()
    barrier.wait(timeout=15.0)
    case.output_directory.mkdir(parents=True, exist_ok=True)
    (case.output_directory / "worker.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": started_at,
                "finished_at": perf_counter(),
                "thread_limits": {
                    name: os.environ.get(name)
                    for name in (
                        "BLIS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    return RunResult(case=case, output_directory=case.output_directory)


def _slow_fake_run(case, **_kwargs):
    sleep(10.0)
    return RunResult(case=case, output_directory=case.output_directory)


def _large_payload_run(case, **_kwargs):
    if case.run.model.axial_cells == 4:
        raise RuntimeError("x" * 1_000_000)
    return RunResult(case=case, output_directory=case.output_directory,
                     plot_errors={"plot": "x" * 1_000_000})


def _crashing_run(case, **_kwargs):
    os._exit(7)


def _interruptible_run(case, **_kwargs):
    if case.run.model.axial_cells == 5:
        sleep(10.0)
    return RunResult(case=case, output_directory=case.output_directory)


def _write_batch(tmp_path: Path, axes: list[dict], **values) -> Path:
    document = {
        "base_case": str(BASE_CASE),
        "output_directory": "output",
        "axes": axes,
        **values,
    }
    path = tmp_path / "batch.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _patch_value(value_id: str, patch: dict) -> dict:
    return {"id": value_id, "patch": patch}


def _write_two_case_batch(tmp_path: Path, **values) -> Path:
    return _write_conditions(tmp_path, {
        "first": {"run": {"model": {"axial_cells": 4}}},
        "second": {"run": {"model": {"axial_cells": 5}}},
    }, **values)


def _write_conditions(tmp_path: Path, conditions: dict, **values) -> Path:
    return _write_batch(tmp_path, [{
        "id": "condition",
        "values": [_patch_value(name, patch) for name, patch in conditions.items()],
    }], **values)


def test_structured_patches_merge_recursively_in_axis_order(tmp_path: Path) -> None:
    batch_path = _write_batch(
        tmp_path,
        [
            {
                "id": "Condition / unsafe",
                "values": [
                    _patch_value(
                        "Flow Δ High",
                        {
                            "run": {
                                "model": {
                                    "axial_cells": 7,
                                    "ambient_temperature_k": 400.0,
                                }
                            },
                            "program": {"inlet_temperature": {"initial": 310.0}},
                        },
                    )
                ],
            },
            {
                "id": "refinement",
                "values": [
                    _patch_value(
                        "fine",
                        {"run": {"model": {"ambient_temperature_k": 450.0}}},
                    )
                ],
            },
        ],
    )

    (expanded,) = expand_batch_cases(load_batch_spec(batch_path))

    assert expanded.case_id == "condition-unsafe-flow-high__refinement-fine"
    assert expanded.run["model"]["axial_cells"] == 7
    assert expanded.run["model"]["ambient_temperature_k"] == 450.0
    assert expanded.run["model"]["bed_length_m"] == 6.0
    assert expanded.run["simulation"]["system_name"] == (
        "condition_unsafe_flow_high__refinement_fine"
    )
    assert expanded.program["inlet_temperature"]["initial"] == 310.0
    assert expanded.program["inlet_temperature"]["steps"]
    assert expanded.case_directory.parent == (tmp_path / "output" / "cases").resolve()


def test_ghsv_programs_use_each_resolved_batch_geometry(tmp_path: Path) -> None:
    program_path = (
        BASE_CASE.parents[1] / "programs" / "high_flow.yaml"
    ).resolve()
    batch_path = _write_batch(
        tmp_path,
        [
            {
                "id": "program",
                "values": [{"id": "ghsv", "program": "ghsv"}],
            },
            {
                "id": "geometry",
                "values": [
                    {"id": "narrow", "geometry": "narrow"},
                    {"id": "wide", "geometry": "wide"},
                ],
            },
        ],
        programs={"ghsv": str(program_path)},
        geometries={
            "narrow": {"model": {"bed_radius_m": 0.25}},
            "wide": {"model": {"bed_radius_m": 0.5}},
        },
    )

    expanded = expand_batch_cases(load_batch_spec(batch_path))
    resolved = [
        resolve_case(
            run_path=case.run_yaml_path,
            chemistry_path=case.case_directory / "chemistry.yaml",
            program_path=case.case_directory / "program.yaml",
            solids_path=case.case_directory / "solids.yaml",
            run_data=case.run,
            chemistry_data=case.chemistry,
            program_data=case.program,
            solids_data=case.solids,
        )
        for case in expanded
    ]

    assert [case.program["inlet_flow"]["basis"] for case in expanded] == [
        "ghsv_per_h",
        "ghsv_per_h",
    ]
    assert resolved[1].inlet_flow_program.initial_value == pytest.approx(
        4.0 * resolved[0].inlet_flow_program.initial_value
    )


def test_dotted_set_overrides_are_rejected(tmp_path: Path) -> None:
    batch_path = _write_batch(
        tmp_path,
        [
            {
                "id": "condition",
                "values": [
                    {
                        "id": "old-style",
                        "set": {"run.model.axial_cells": 4},
                    }
                ],
            }
        ],
    )

    with pytest.raises(BatchValidationError) as caught:
        load_batch_spec(batch_path)

    assert "axes.0.values.0.set" in str(caught.value)
    assert "Extra inputs are not permitted" in str(caught.value)


def test_workers_must_be_a_positive_integer(tmp_path: Path) -> None:
    batch_path = _write_conditions(tmp_path, {
        "value": {"run": {"model": {"axial_cells": 4}}},
    }, workers=0)

    with pytest.raises(BatchValidationError, match="workers.*positive integer"):
        load_batch_spec(batch_path)


def test_slug_collisions_are_rejected_before_materialization(tmp_path: Path) -> None:
    batch_path = _write_conditions(tmp_path, {
        "a b": {"run": {"model": {"axial_cells": 4}}},
        "a-b": {"run": {"model": {"axial_cells": 5}}},
    })

    with pytest.raises(BatchValidationError, match="slug collision"):
        expand_batch_cases(load_batch_spec(batch_path))

    assert not (tmp_path / "output").exists()


def test_case_root_must_resolve_inside_the_batch_output(tmp_path: Path) -> None:
    batch_path = _write_conditions(tmp_path, {
        "safe": {"run": {"model": {"axial_cells": 4}}},
    })
    outside = tmp_path / "outside"
    outside.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    if os.name == "nt":
        from _winapi import CreateJunction

        CreateJunction(str(outside), str(output / "cases"))
    else:
        (output / "cases").symlink_to(outside, target_is_directory=True)

    with pytest.raises(BatchValidationError, match="escapes batch output directory"):
        expand_batch_cases(load_batch_spec(batch_path))


def test_batch_validate_only_is_side_effect_free(tmp_path: Path) -> None:
    batch_path = _write_conditions(tmp_path, {
        "valid": {"run": {"model": {"axial_cells": 4}}},
    })
    paths_before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    result = run_batch_file(batch_path, validate_only=True)

    assert result.summary_path is None
    assert [record.status for record in result.records] == ["validation_passed"]
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == paths_before


def test_all_cases_are_validated_before_any_case_is_written_or_run(tmp_path: Path) -> None:
    batch_path = _write_conditions(tmp_path, {
        "valid": {"run": {"model": {"axial_cells": 4}}},
        "invalid": {"run": {"model": {"bed_length_m": 5.0}}},
    })
    calls = []

    def fake_run(case, **_kwargs):
        calls.append(case)
        return RunResult(case=case, output_directory=case.output_directory)

    result = run_batch_file(batch_path, run_case_fn=fake_run)

    assert calls == []
    assert [record.status for record in result.records] == [
        "validation_passed",
        "validation_failed",
    ]
    assert result.summary_path is None
    assert not (tmp_path / "output").exists()


def test_existing_case_output_is_rejected_before_a_run(tmp_path: Path) -> None:
    batch_path = _write_conditions(tmp_path, {
        "existing": {"run": {"model": {"axial_cells": 4}}},
    })
    (tmp_path / "output" / "cases" / "condition-existing").mkdir(parents=True)
    calls = []

    def fake_run(case, **_kwargs):
        calls.append(case)
        return RunResult(case=case, output_directory=case.output_directory)

    with pytest.raises(BatchValidationError, match="refusing to overwrite"):
        run_batch_file(batch_path, run_case_fn=fake_run)

    assert calls == []


def test_batch_execution_uses_resolved_case_plot_selection_and_artifacts(tmp_path: Path) -> None:
    batch_path = _write_batch(
        tmp_path,
        [
            {
                "id": "condition",
                "values": [
                    _patch_value(
                        "run",
                        {
                            "run": {
                                "model": {"axial_cells": 4},
                                "outputs": {
                                    "requested_reports": [
                                        "temperature",
                                        "pressure",
                                    ],
                                    "requested_plots": ["axial_profiles"],
                                },
                            }
                        },
                    )
                ],
            }
        ],
        artifacts=True,
    )
    artifact_calls = []
    run_calls = []

    def fake_artifacts(case):
        artifact_calls.append(case)
        return {}

    def fake_run(case, *, artifact_paths):
        run_calls.append((case, artifact_paths))
        return RunResult(
            case=case,
            output_directory=case.output_directory,
            plot_errors={"axial_profiles": "synthetic plot failure"},
        )

    result = run_batch_file(
        batch_path,
        generate_artifacts_fn=fake_artifacts,
        run_case_fn=fake_run,
    )

    assert len(artifact_calls) == 1
    assert run_calls == [(artifact_calls[0], {})]
    assert run_calls[0][0].run.outputs.requested_plots == ("axial_profiles",)
    assert result.records[0].status == "success"
    assert result.records[0].plot_status == "failed"
    assert result.plot_failed_count == 1
    assert "synthetic plot failure" in result.summary_path.read_text()
    assert result.summary_path == (tmp_path / "output" / "summary.csv").resolve()
    assert result.summary_path.is_file()
    assert not (tmp_path / "output" / "manifest.csv").exists()


def test_parallel_batch_runs_single_threaded_cases_in_overlapping_processes(
    tmp_path: Path,
) -> None:
    batch_path = _write_two_case_batch(tmp_path, workers=2)

    run = partial(_parallel_fake_run, barrier=get_context("spawn").Barrier(2))
    result = run_batch_file(batch_path, run_case_fn=run)

    assert result.workers == 2
    assert [record.status for record in result.records] == ["success", "success"]
    worker_details = [
        json.loads((record.output_directory / "worker.json").read_text(encoding="utf-8"))
        for record in result.records
    ]
    assert len({details["pid"] for details in worker_details}) == 2
    assert all(
        value == "1"
        for details in worker_details
        for value in details["thread_limits"].values()
    )
    assert max(details["started_at"] for details in worker_details) < min(
        details["finished_at"] for details in worker_details
    )
    for record in result.records:
        materialized_run = yaml.safe_load(record.run_yaml_path.read_text(encoding="utf-8"))
        assert materialized_run["solver"]["threads"] == 1


def test_parallel_case_timeouts_kill_each_worker_and_continue(tmp_path: Path) -> None:
    batch_path = _write_two_case_batch(tmp_path, workers=2, case_timeout_s=0.1)

    result = run_batch_file(batch_path, run_case_fn=_slow_fake_run)

    assert [record.status for record in result.records] == [
        "timeout_failed",
        "timeout_failed",
    ]
    assert all("Timed out after 0.1 seconds" in record.error for record in result.records)
    assert result.summary_path is not None
    summary = result.summary_path.read_text(encoding="utf-8")
    assert summary.count("timeout_failed") == 2


def test_large_worker_payloads_are_drained_before_process_exit(tmp_path: Path) -> None:
    result = run_batch_file(_write_two_case_batch(tmp_path, workers=2, case_timeout_s=5.0),
                            run_case_fn=_large_payload_run)
    assert [record.status for record in result.records] == ["simulation_failed", "success"]
    assert result.records[0].error == "x" * 1_000_000
    assert result.records[1].plot_errors == {"plot": "x" * 1_000_000}


def test_crashed_workers_are_reaped_and_reported(tmp_path: Path) -> None:
    children_before = active_children()
    result = run_batch_file(_write_two_case_batch(tmp_path, workers=2), run_case_fn=_crashing_run)
    assert [record.status for record in result.records] == ["simulation_failed"] * 2
    assert all("exited with code 7" in record.error for record in result.records)
    assert active_children() == children_before


@pytest.mark.parametrize("workers", [1, 2])
def test_interruption_preserves_completed_cases_and_reaps_workers(tmp_path, monkeypatch, workers):
    children_before = active_children()
    write_records = batch._write_records_csv
    interrupted = False

    def interrupt_after_first_completion(path, records, axis_ids):
        nonlocal interrupted
        write_records(path, records, axis_ids)
        if not interrupted and records[0].status == "success":
            with path.open(newline="", encoding="utf-8") as handle:
                assert next(csv.DictReader(handle))["status"] == "success"
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(batch, "_write_records_csv", interrupt_after_first_completion)
    with pytest.raises(KeyboardInterrupt):
        run_batch_file(_write_two_case_batch(tmp_path, workers=workers),
                       run_case_fn=_interruptible_run)
    with (tmp_path / "output" / "summary.csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert [record["status"] for record in records] == ["success", "interrupted_failed"]
    assert "KeyboardInterrupt" in records[1]["error"]
    assert active_children() == children_before


def test_interrupted_csv_write_preserves_previous_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "summary.csv"
    record = batch.BatchCaseRecord("case", {}, tmp_path, tmp_path / "run.yaml")
    batch._write_records_csv(path, (record,), ())
    previous = path.read_bytes()

    def interrupt_write(writer, row):
        writer.writer.writerow(["partial"])
        raise KeyboardInterrupt

    monkeypatch.setattr(csv.DictWriter, "writerow", interrupt_write)
    with pytest.raises(KeyboardInterrupt):
        batch._write_records_csv(path, (record,), ())
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
