from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from packed_bed import cli
from packed_bed.config import load_case
from packed_bed.reports import RunResult


BASE_CASE_DIRECTORY = (
    Path(__file__).parents[1]
    / "packed_bed"
    / "examples"
    / "default_batch_case"
    / "base_case"
).resolve()


def _copy_case(tmp_path: Path, source: Path = BASE_CASE_DIRECTORY) -> Path:
    case_directory = tmp_path / "case"
    case_directory.mkdir()
    for path in source.glob("*.yaml"):
        shutil.copyfile(path, case_directory / path.name)
    return case_directory / "run.yaml"


@pytest.mark.parametrize("example", ["default_case", "default_batch_case/base_case"])
def test_single_case_validate_only_creates_nothing_and_skips_runtime(tmp_path, capsys, example):
    source = Path(__file__).parents[1] / "packed_bed" / "examples" / example
    run_path = _copy_case(tmp_path, source)
    paths_before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    exit_code = cli.main([str(run_path), "--validate-only", "--artifacts"])

    assert exit_code == 0
    assert "Validation passed:" in capsys.readouterr().out
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == paths_before
    assert not (run_path.parent / "output").exists()


def test_batch_validate_only_creates_no_cases_or_manifest(tmp_path: Path, capsys) -> None:
    batch_directory = tmp_path / "batch"
    shutil.copytree(BASE_CASE_DIRECTORY.parent, batch_directory, ignore=shutil.ignore_patterns("output"))
    paths_before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    exit_code = cli.main(["batch", str(batch_directory / "batch.yaml"), "--validate-only"])

    assert exit_code == 0
    assert capsys.readouterr().out == "Batch validation complete: 4/4 cases passed.\n"
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == paths_before
    assert not (batch_directory / "output").exists()


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
assert 'xarray' not in sys.modules
assert 'matplotlib' not in sys.modules
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


def test_plot_failure_returns_nonzero_without_changing_simulation_status(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    run_path = _copy_case(tmp_path)
    case = load_case(run_path)
    result = RunResult(
        case=case,
        output_directory=case.output_directory,
        status="success",
        runtime_s=0.1,
        plot_errors={"axial_profiles": "synthetic failure"},
    )
    monkeypatch.setitem(
        sys.modules,
        "packed_bed.simulation",
        SimpleNamespace(run_case=lambda *_args, **_kwargs: result),
    )

    exit_code = cli.main([str(run_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert result.status == "success"
    assert "plot 'axial_profiles' failed: synthetic failure" in captured.err


def test_dae_plotter_only_requests_reporter_retention(tmp_path: Path, monkeypatch) -> None:
    run_path = _copy_case(tmp_path)
    case = load_case(run_path)
    calls = []

    def fake_run(resolved_case, **kwargs):
        calls.append(kwargs)
        return RunResult(
            case=resolved_case,
            output_directory=resolved_case.output_directory,
            runtime_s=0.1,
            reporter=object(),
        )

    monkeypatch.setitem(
        sys.modules,
        "packed_bed.simulation",
        SimpleNamespace(run_case=fake_run),
    )
    monkeypatch.setattr(cli, "launch_daetools_plotter", lambda _result: 0)

    assert cli.main([str(run_path), "--dae-plotter"]) == 0
    assert calls == [{"artifact_paths": {}, "retain_reporter": True}]


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


def test_debug_exposes_unexpected_errors_and_help_lists_batch(monkeypatch):
    def fail(_path):
        raise RuntimeError("unexpected loader failure")

    monkeypatch.setattr(cli, "load_case", fail)
    assert cli.main(["case.yaml"]) == 1
    with pytest.raises(RuntimeError, match="unexpected loader failure"):
        cli.main(["case.yaml", "--debug"])
    assert "packed_bed batch --help" in cli.build_parser().format_help()
