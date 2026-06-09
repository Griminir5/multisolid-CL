from __future__ import annotations

from pathlib import Path

from .errors import PackedBedValidationError
from .bundle import RunBundle
from .chemistry import ChemistryConfig
from .io import read_yaml_mapping, resolve_path
from .program import ProgramConfig
from .run import RunConfig
from .solids import SolidConfig
from .validation import validate_bundle_shape, validate_config_model, validate_run_bundle


def load_run_bundle(run_yaml_path: str | Path) -> RunBundle:
    run_path = Path(run_yaml_path).resolve()
    run = _load_config_model(RunConfig, run_path, "run.yaml")

    base_dir = run_path.parent
    chemistry_path = resolve_path(base_dir, run.references.chemistry_file)
    program_path = resolve_path(base_dir, run.references.program_file)
    solids_path = resolve_path(base_dir, run.references.solids_file)

    for label, path in (
        ("run.references.chemistry_file", chemistry_path),
        ("run.references.program_file", program_path),
        ("run.references.solids_file", solids_path),
    ):
        if not path.exists():
            raise PackedBedValidationError(f"{label} does not exist: {path}")
        if not path.is_file():
            raise PackedBedValidationError(f"{label} must point to a file: {path}")

    chemistry = _load_config_model(ChemistryConfig, chemistry_path, "chemistry.yaml")
    program = _load_config_model(ProgramConfig, program_path, "program.yaml")
    solids = _load_config_model(SolidConfig, solids_path, "solids.yaml")

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
    validate_bundle_shape(run_bundle)
    return validate_run_bundle(run_bundle)


def _load_config_model(model_type, path: Path, label: str):
    return validate_config_model(model_type, read_yaml_mapping(path, label), label, path)
