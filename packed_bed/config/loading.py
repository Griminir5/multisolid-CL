from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from config import PackedBedValidationError
from .io import format_validation_error, read_yaml_mapping, validate_model
from .models import ChemistryConfig, ProgramConfig, RunBundle, RunConfig, SolidConfig
from .validation import validate_run_bundle


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def load_run_bundle(run_yaml_path: str | Path) -> RunBundle:
    run_path = Path(run_yaml_path).resolve()
    run = validate_model(RunConfig, read_yaml_mapping(run_path, "run.yaml"), "run.yaml", run_path)

    base_dir = run_path.parent
    chemistry_path = _resolve_path(base_dir, run.references.chemistry_file)
    program_path = _resolve_path(base_dir, run.references.program_file)
    solids_path = _resolve_path(base_dir, run.references.solids_file)

    for label, path in (
        ("run.references.chemistry_file", chemistry_path),
        ("run.references.program_file", program_path),
        ("run.references.solids_file", solids_path),
    ):
        if not path.exists():
            raise PackedBedValidationError(f"{label} does not exist: {path}")
        if not path.is_file():
            raise PackedBedValidationError(f"{label} must point to a file: {path}")

    chemistry = validate_model(
        ChemistryConfig,
        read_yaml_mapping(chemistry_path, "chemistry.yaml"),
        "chemistry.yaml",
        chemistry_path,
    )
    program = validate_model(
        ProgramConfig,
        read_yaml_mapping(program_path, "program.yaml"),
        "program.yaml",
        program_path,
    )
    solids = validate_model(
        SolidConfig,
        read_yaml_mapping(solids_path, "solids.yaml"),
        "solids.yaml",
        solids_path,
    )

    try:
        run_bundle = RunBundle(
            run_path=run_path,
            chemistry_path=chemistry_path,
            solids_path=solids_path,
            program_path=program_path,
            chemistry=chemistry,
            solids=solids,
            program=program,
            run=run,
        )
    except ValidationError as exc:
        raise format_validation_error("run bundle", run_path, exc) from exc

    return validate_run_bundle(run_bundle)