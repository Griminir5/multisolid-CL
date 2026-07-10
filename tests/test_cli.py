from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from packed_bed import cli


BASE_CASE_DIRECTORY = (
    Path(__file__).parents[1]
    / "packed_bed"
    / "examples"
    / "default_batch_case"
    / "base_case"
).resolve()


def _copy_case(tmp_path: Path) -> Path:
    case_directory = tmp_path / "case"
    shutil.copytree(BASE_CASE_DIRECTORY, case_directory)
    return case_directory / "run.yaml"


def test_single_case_validate_only_creates_nothing_and_skips_runtime(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    run_path = _copy_case(tmp_path)
    paths_before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("runtime function called during validation")

    monkeypatch.setattr(cli, "generate_artifacts", forbidden)
    monkeypatch.setattr(cli, "run_simulation", forbidden)

    exit_code = cli.main([str(run_path), "--validate-only", "--artifacts", "--plots"])

    assert exit_code == 0
    assert "Validation passed:" in capsys.readouterr().out
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == paths_before
    assert not (run_path.parent / "output").exists()


def test_batch_validate_only_creates_no_cases_or_manifest(tmp_path: Path, capsys) -> None:
    batch_path = tmp_path / "batch.yaml"
    batch_path.write_text(
        yaml.safe_dump(
            {
                "base_case": str(BASE_CASE_DIRECTORY / "run.yaml"),
                "output_directory": "output",
                "axes": [
                    {
                        "id": "condition",
                        "values": [
                            {
                                "id": "valid",
                                "patch": {"run": {"model": {"axial_cells": 4}}},
                            }
                        ],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["batch", str(batch_path), "--validate-only"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "Batch validation complete: 1/1 cases passed.\n"
    assert "Manifest" not in captured.out
    assert not (tmp_path / "output").exists()


def test_cli_import_and_validation_do_not_import_daetools(tmp_path: Path) -> None:
    run_path = _copy_case(tmp_path)
    script = """
import builtins
import sys

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'daetools' or name.startswith('daetools.') or name == 'pyUnits':
        raise AssertionError(f'forbidden solver import: {name}')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from packed_bed.cli import main
assert main([sys.argv[1], '--validate-only']) == 0
assert not any(name == 'daetools' or name.startswith('daetools.') for name in sys.modules)
assert 'pyUnits' not in sys.modules
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]))

    completed = subprocess.run(
        [sys.executable, "-c", script, str(run_path)],
        check=True,
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert "Validation passed:" in completed.stdout


def test_validation_errors_are_concise_and_use_a_distinct_exit_code(
    tmp_path: Path,
    capsys,
) -> None:
    missing_path = tmp_path / "missing.yaml"

    exit_code = cli.main([str(missing_path), "--validate-only"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == f"error: run was not found: {missing_path.resolve()}\n"
