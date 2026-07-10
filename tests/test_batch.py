from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from packed_bed.batch import (
    BatchValidationError,
    expand_batch_cases,
    load_batch_spec,
    run_batch_file,
)
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


def test_slug_collisions_are_rejected_before_materialization(tmp_path: Path) -> None:
    batch_path = _write_batch(
        tmp_path,
        [
            {
                "id": "condition",
                "values": [
                    _patch_value("a b", {"run": {"model": {"axial_cells": 4}}}),
                    _patch_value("a-b", {"run": {"model": {"axial_cells": 5}}}),
                ],
            }
        ],
    )

    with pytest.raises(BatchValidationError, match="slug collision"):
        expand_batch_cases(load_batch_spec(batch_path))

    assert not (tmp_path / "output").exists()


def test_case_root_must_resolve_inside_the_batch_output(tmp_path: Path) -> None:
    batch_path = _write_batch(
        tmp_path,
        [
            {
                "id": "condition",
                "values": [
                    _patch_value("safe", {"run": {"model": {"axial_cells": 4}}})
                ],
            }
        ],
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / "cases").symlink_to(outside, target_is_directory=True)

    with pytest.raises(BatchValidationError, match="escapes batch output directory"):
        expand_batch_cases(load_batch_spec(batch_path))


def test_batch_validate_only_is_side_effect_free(tmp_path: Path) -> None:
    batch_path = _write_batch(
        tmp_path,
        [
            {
                "id": "condition",
                "values": [
                    _patch_value("valid", {"run": {"model": {"axial_cells": 4}}})
                ],
            }
        ],
    )
    paths_before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    result = run_batch_file(batch_path, validate_only=True)

    assert result.summary_path is None
    assert [record.status for record in result.records] == ["validation_passed"]
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == paths_before


def test_all_cases_are_validated_before_any_case_is_written_or_run(tmp_path: Path) -> None:
    batch_path = _write_batch(
        tmp_path,
        [
            {
                "id": "condition",
                "values": [
                    _patch_value("valid", {"run": {"model": {"axial_cells": 4}}}),
                    _patch_value("invalid", {"run": {"model": {"bed_length_m": 5.0}}}),
                ],
            }
        ],
    )
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
    batch_path = _write_batch(
        tmp_path,
        [
            {
                "id": "condition",
                "values": [
                    _patch_value("existing", {"run": {"model": {"axial_cells": 4}}})
                ],
            }
        ],
    )
    (tmp_path / "output" / "cases" / "condition-existing").mkdir(parents=True)
    calls = []

    def fake_run(case, **_kwargs):
        calls.append(case)
        return RunResult(case=case, output_directory=case.output_directory)

    with pytest.raises(BatchValidationError, match="refusing to overwrite"):
        run_batch_file(batch_path, run_case_fn=fake_run)

    assert calls == []


def test_batch_execution_uses_resolved_cases_and_explicit_render_options(tmp_path: Path) -> None:
    batch_path = _write_batch(
        tmp_path,
        [
            {
                "id": "condition",
                "values": [
                    _patch_value("run", {"run": {"model": {"axial_cells": 4}}})
                ],
            }
        ],
        artifacts=True,
        plots=True,
    )
    artifact_calls = []
    run_calls = []

    def fake_artifacts(case):
        artifact_calls.append(case)
        return {}

    def fake_run(case, *, artifact_paths, render_plots):
        run_calls.append((case, artifact_paths, render_plots))
        return RunResult(case=case, output_directory=case.output_directory)

    result = run_batch_file(
        batch_path,
        generate_artifacts_fn=fake_artifacts,
        run_case_fn=fake_run,
    )

    assert len(artifact_calls) == 1
    assert run_calls == [(artifact_calls[0], {}, True)]
    assert result.records[0].status == "success"
    assert result.summary_path == (tmp_path / "output" / "summary.csv").resolve()
    assert result.summary_path.is_file()
    assert not (tmp_path / "output" / "manifest.csv").exists()
