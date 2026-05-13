from __future__ import annotations

import csv
import itertools
import math
import multiprocessing as mp
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from time import perf_counter
from typing import Any, Callable

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .config import ConfigString, PackedBedValidationError, RunBundle, RunResult, _read_yaml_mapping, load_run_bundle


NORMAL_TEMPERATURE_K = 273.15
NORMAL_PRESSURE_PA = 100000.0
GAS_CONSTANT_J_PER_MOL_K = 8.31446
NORMAL_MOLAR_DENSITY_MOL_PER_M3 = NORMAL_PRESSURE_PA / (GAS_CONSTANT_J_PER_MOL_K * NORMAL_TEMPERATURE_K)

_PATH_PART_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?P<indexes>(?:\[\d+\])*)$")
_INDEX_RE = re.compile(r"\[(\d+)\]")
_CONFIG_ROOTS = frozenset({"run", "program", "solids", "chemistry"})
_PROCESS_TERMINATE_GRACE_S = 5.0


class BatchValidationError(PackedBedValidationError):
    pass


class BatchCaseTimeoutError(TimeoutError):
    pass


class BatchCaseWorkerError(RuntimeError):
    pass


class BatchConfigModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        populate_by_name=True,
    )


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("must be provided as a YAML sequence.")
    return tuple(value)


def _require_unique(values: tuple[str, ...], label: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"{label} contains duplicates: {', '.join(duplicates)}.")


def _format_validation_error(label: str, path: Path, exc: ValidationError) -> BatchValidationError:
    lines = [f"{label} is invalid: {path}"]
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"]) or "<root>"
        lines.append(f"- {location}: {error['msg']}")
    return BatchValidationError("\n".join(lines))


def _parse_config_path(raw_path: str) -> tuple[str | int, ...]:
    if raw_path == "":
        raise ValueError("set path must not be blank.")

    tokens: list[str | int] = []
    for raw_part in raw_path.split("."):
        match = _PATH_PART_RE.fullmatch(raw_part)
        if match is None:
            raise ValueError(f"invalid set path syntax: {raw_path!r}.")
        tokens.append(match.group("name"))
        tokens.extend(int(index) for index in _INDEX_RE.findall(match.group("indexes")))

    if not tokens or not isinstance(tokens[0], str) or tokens[0] not in _CONFIG_ROOTS:
        raise ValueError(f"set path must start with one of: {', '.join(sorted(_CONFIG_ROOTS))}.")
    return tuple(tokens)


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _set_config_path(root_mappings: dict[str, Any], raw_path: str, value: Any) -> None:
    tokens = _parse_config_path(raw_path)
    current = root_mappings[tokens[0]]
    traversed = str(tokens[0])

    for token in tokens[1:-1]:
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                raise BatchValidationError(f"Cannot set {raw_path!r}: {traversed}[{token}] does not exist.")
            current = current[token]
            traversed = f"{traversed}[{token}]"
            continue

        if not isinstance(current, dict) or token not in current:
            raise BatchValidationError(f"Cannot set {raw_path!r}: {traversed}.{token} does not exist.")
        current = current[token]
        traversed = f"{traversed}.{token}"

    final_token = tokens[-1]
    if isinstance(final_token, int):
        if not isinstance(current, list) or final_token >= len(current):
            raise BatchValidationError(f"Cannot set {raw_path!r}: {traversed}[{final_token}] does not exist.")
        current[final_token] = value
        return

    if not isinstance(current, dict) or final_token not in current:
        raise BatchValidationError(f"Cannot set {raw_path!r}: {traversed}.{final_token} does not exist.")
    current[final_token] = value


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BatchValidationError(f"{label} must be a mapping.")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise BatchValidationError(f"{label} must be numeric.")
    if not math.isfinite(float(value)):
        raise BatchValidationError(f"{label} must be finite.")
    return float(value)


def _coerce_case_timeout_s(value: Any, label: str = "case_timeout_s") -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be numeric seconds.")
    timeout_s = float(value)
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError(f"{label} must be a finite positive number of seconds.")
    return timeout_s


def _empty_bed_volume_m3(model_mapping: dict[str, Any]) -> float:
    bed_length = _require_number(model_mapping.get("bed_length_m"), "run.model.bed_length_m")
    bed_radius = _require_number(model_mapping.get("bed_radius_m"), "run.model.bed_radius_m")
    return math.pi * bed_radius**2 * bed_length


def _ghsv_to_mol_per_s(ghsv_per_h: Any, empty_bed_volume_m3: float, label: str) -> float:
    value = _require_number(ghsv_per_h, label)
    return value * empty_bed_volume_m3 * NORMAL_MOLAR_DENSITY_MOL_PER_M3 / 3600.0


def _materialize_inlet_flow_basis(program_mapping: dict[str, Any], run_mapping: dict[str, Any]) -> None:
    inlet_flow = _require_mapping(program_mapping.get("inlet_flow"), "program.inlet_flow")
    basis = inlet_flow.pop("basis", "mol_per_s")
    if basis == "mol_per_s":
        return
    if basis != "ghsv_per_h":
        raise BatchValidationError("program.inlet_flow.basis must be one of: mol_per_s, ghsv_per_h.")

    model_mapping = _require_mapping(run_mapping.get("model"), "run.model")
    volume_m3 = _empty_bed_volume_m3(model_mapping)
    inlet_flow["initial"] = _ghsv_to_mol_per_s(
        inlet_flow.get("initial"),
        volume_m3,
        "program.inlet_flow.initial",
    )
    steps = inlet_flow.get("steps", [])
    if steps is None:
        return
    if not isinstance(steps, list):
        raise BatchValidationError("program.inlet_flow.steps must be a sequence.")
    for step_index, step in enumerate(steps):
        step_mapping = _require_mapping(step, f"program.inlet_flow.steps[{step_index}]")
        if step_mapping.get("kind") == "ramp":
            step_mapping["target"] = _ghsv_to_mol_per_s(
                step_mapping.get("target"),
                volume_m3,
                f"program.inlet_flow.steps[{step_index}].target",
            )


class BatchAxisValue(BatchConfigModel):
    id: ConfigString
    program: ConfigString | None = None
    geometry: ConfigString | None = None
    set_values: dict[ConfigString, Any] = Field(default_factory=dict, alias="set")

    @field_validator("set_values")
    @classmethod
    def validate_set_paths(cls, value: dict[str, Any]) -> dict[str, Any]:
        for raw_path in value:
            _parse_config_path(raw_path)
        return value

    @model_validator(mode="after")
    def validate_has_effect(self) -> "BatchAxisValue":
        if self.program is None and self.geometry is None and not self.set_values:
            raise ValueError("must specify at least one of: program, geometry, set.")
        return self


class BatchAxis(BatchConfigModel):
    id: ConfigString
    values: tuple[BatchAxisValue, ...]

    @field_validator("values", mode="before")
    @classmethod
    def coerce_values(cls, value: Any) -> tuple[Any, ...]:
        return _as_tuple(value)

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: tuple[BatchAxisValue, ...]) -> tuple[BatchAxisValue, ...]:
        if not value:
            raise ValueError("must not be empty.")
        _require_unique(tuple(axis_value.id for axis_value in value), "axis values")
        return value


class GeometryPreset(BatchConfigModel):
    model: dict[ConfigString, Any] = Field(default_factory=dict)
    solids_file: ConfigString | None = None

    @field_validator("model")
    @classmethod
    def validate_model_patch(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("must not be empty.")
        return value


class BatchSpec(BatchConfigModel):
    base_case: ConfigString
    output_directory: ConfigString
    case_timeout_s: float | None = None
    programs: dict[ConfigString, ConfigString] = Field(default_factory=dict)
    geometries: dict[ConfigString, GeometryPreset] = Field(default_factory=dict)
    axes: tuple[BatchAxis, ...]

    @field_validator("case_timeout_s", mode="before")
    @classmethod
    def validate_case_timeout_s(cls, value: Any) -> float | None:
        return _coerce_case_timeout_s(value)

    @field_validator("axes", mode="before")
    @classmethod
    def coerce_axes(cls, value: Any) -> tuple[Any, ...]:
        return _as_tuple(value)

    @field_validator("axes")
    @classmethod
    def validate_axes(cls, value: tuple[BatchAxis, ...]) -> tuple[BatchAxis, ...]:
        if not value:
            raise ValueError("must not be empty.")
        _require_unique(tuple(axis.id for axis in value), "axes")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> "BatchSpec":
        for axis in self.axes:
            for axis_value in axis.values:
                if axis_value.program is not None and axis_value.program not in self.programs:
                    raise ValueError(
                        f"axis '{axis.id}' value '{axis_value.id}' references unknown program '{axis_value.program}'."
                    )
                if axis_value.geometry is not None and axis_value.geometry not in self.geometries:
                    raise ValueError(
                        f"axis '{axis.id}' value '{axis_value.id}' references unknown geometry '{axis_value.geometry}'."
                    )
        return self


@dataclass(frozen=True)
class BatchDocument:
    batch_path: Path
    spec: BatchSpec

    @property
    def base_dir(self) -> Path:
        return self.batch_path.parent

    @property
    def output_directory(self) -> Path:
        return _resolve_path(self.base_dir, self.spec.output_directory)


@dataclass(frozen=True)
class BaseCaseMappings:
    run_path: Path
    run: dict[str, Any]
    chemistry: dict[str, Any]
    program: dict[str, Any]
    solids: dict[str, Any]


@dataclass(frozen=True)
class MaterializedBatchCase:
    case_id: str
    selections: dict[str, str]
    case_directory: Path
    run_yaml_path: Path
    run: dict[str, Any]
    chemistry: dict[str, Any]
    program: dict[str, Any]
    solids: dict[str, Any]


@dataclass
class BatchCaseRecord:
    case_id: str
    selections: dict[str, str]
    case_directory: Path
    run_yaml_path: Path
    status: str = "pending"
    error: str = ""
    runtime_s: float | None = None
    output_directory: Path | None = None
    balance_errors: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchResult:
    batch_path: Path
    output_directory: Path
    manifest_path: Path
    summary_path: Path | None
    records: tuple[BatchCaseRecord, ...]

    @property
    def total_count(self) -> int:
        return len(self.records)

    @property
    def failed_count(self) -> int:
        return sum(1 for record in self.records if record.status.endswith("_failed"))


def load_batch_spec(batch_yaml_path: str | Path) -> BatchDocument:
    batch_path = Path(batch_yaml_path).resolve()
    data = _read_yaml_mapping(batch_path, "batch.yaml")
    try:
        spec = BatchSpec.model_validate(data)
    except ValidationError as exc:
        raise _format_validation_error("batch.yaml", batch_path, exc) from exc
    return BatchDocument(batch_path=batch_path, spec=spec)


def _load_base_case_mappings(document: BatchDocument) -> BaseCaseMappings:
    run_path = _resolve_path(document.base_dir, document.spec.base_case)
    run = _read_yaml_mapping(run_path, "run.yaml")
    references = _require_mapping(run.get("references"), "run.references")
    base_dir = run_path.parent
    chemistry = _read_yaml_mapping(_resolve_path(base_dir, references.get("chemistry_file", "")), "chemistry.yaml")
    program = _read_yaml_mapping(_resolve_path(base_dir, references.get("program_file", "")), "program.yaml")
    solids = _read_yaml_mapping(_resolve_path(base_dir, references.get("solids_file", "")), "solids.yaml")
    return BaseCaseMappings(
        run_path=run_path,
        run=run,
        chemistry=chemistry,
        program=program,
        solids=solids,
    )


def _load_program_preset(document: BatchDocument, program_name: str) -> dict[str, Any]:
    raw_path = document.spec.programs[program_name]
    return _read_yaml_mapping(_resolve_path(document.base_dir, raw_path), "program preset")


def _load_solids_preset(document: BatchDocument, raw_path: str) -> dict[str, Any]:
    return _read_yaml_mapping(_resolve_path(document.base_dir, raw_path), "geometry solids preset")


def _apply_geometry_preset(
    document: BatchDocument,
    geometry_name: str,
    run_mapping: dict[str, Any],
    current_solids: dict[str, Any],
) -> dict[str, Any]:
    preset = document.spec.geometries[geometry_name]
    model_mapping = _require_mapping(run_mapping.get("model"), "run.model")
    model_mapping.update(deepcopy(preset.model))
    if preset.solids_file is None:
        return current_solids
    return _load_solids_preset(document, preset.solids_file)


def _case_id_from_values(axis_values: tuple[tuple[BatchAxis, BatchAxisValue], ...]) -> str:
    return "__".join(f"{axis.id}_{axis_value.id}" for axis, axis_value in axis_values)


def _force_case_run_fields(run_mapping: dict[str, Any], case_id: str) -> None:
    references = _require_mapping(run_mapping.setdefault("references", {}), "run.references")
    references["chemistry_file"] = "chemistry.yaml"
    references["program_file"] = "program.yaml"
    references["solids_file"] = "solids.yaml"

    simulation = _require_mapping(run_mapping.setdefault("simulation", {}), "run.simulation")
    simulation["system_name"] = case_id

    outputs = _require_mapping(run_mapping.setdefault("outputs", {}), "run.outputs")
    outputs["directory"] = "output"
    outputs["artifacts_directory"] = "output/artifacts"


def expand_batch_cases(document: BatchDocument) -> tuple[MaterializedBatchCase, ...]:
    base = _load_base_case_mappings(document)
    cases: list[MaterializedBatchCase] = []
    axis_value_groups = [tuple((axis, axis_value) for axis_value in axis.values) for axis in document.spec.axes]
    case_root = document.output_directory / "cases"

    for axis_values in itertools.product(*axis_value_groups):
        case_id = _case_id_from_values(axis_values)
        run_mapping = deepcopy(base.run)
        chemistry_mapping = deepcopy(base.chemistry)
        program_mapping = deepcopy(base.program)
        solids_mapping = deepcopy(base.solids)
        selections = {axis.id: axis_value.id for axis, axis_value in axis_values}

        for _axis, axis_value in axis_values:
            if axis_value.program is not None:
                program_mapping = _load_program_preset(document, axis_value.program)
            if axis_value.geometry is not None:
                solids_mapping = _apply_geometry_preset(document, axis_value.geometry, run_mapping, solids_mapping)
            for raw_path, value in axis_value.set_values.items():
                _set_config_path(
                    {
                        "run": run_mapping,
                        "chemistry": chemistry_mapping,
                        "program": program_mapping,
                        "solids": solids_mapping,
                    },
                    raw_path,
                    deepcopy(value),
                )

        _force_case_run_fields(run_mapping, case_id)
        _materialize_inlet_flow_basis(program_mapping, run_mapping)
        case_directory = case_root / case_id
        cases.append(
            MaterializedBatchCase(
                case_id=case_id,
                selections=selections,
                case_directory=case_directory,
                run_yaml_path=case_directory / "run.yaml",
                run=run_mapping,
                chemistry=chemistry_mapping,
                program=program_mapping,
                solids=solids_mapping,
            )
        )

    return tuple(cases)


def _write_yaml_mapping(path: Path, mapping: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(mapping, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def materialize_batch_cases(document: BatchDocument) -> tuple[MaterializedBatchCase, ...]:
    cases = expand_batch_cases(document)
    for case in cases:
        case.case_directory.mkdir(parents=True, exist_ok=True)
        _write_yaml_mapping(case.case_directory / "run.yaml", case.run)
        _write_yaml_mapping(case.case_directory / "chemistry.yaml", case.chemistry)
        _write_yaml_mapping(case.case_directory / "program.yaml", case.program)
        _write_yaml_mapping(case.case_directory / "solids.yaml", case.solids)
    return cases


def _record_columns(axis_ids: tuple[str, ...], *, include_runtime: bool) -> list[str]:
    columns = [
        "case_id",
        *(f"axis_{axis_id}" for axis_id in axis_ids),
        "status",
        "error",
        "case_directory",
        "run_yaml",
    ]
    if include_runtime:
        columns.extend(
            [
                "runtime_s",
                "output_directory",
                "heat_balance_max_abs_error",
                "heat_balance_time_s",
                "heat_balance_unit",
                "mass_balance_max_abs_error",
                "mass_balance_time_s",
                "mass_balance_unit",
            ]
        )
    return columns


def _balance_field(record: BatchCaseRecord, key: str, attribute: str) -> str | float:
    error = record.balance_errors.get(key)
    if error is None:
        return ""
    return getattr(error, attribute)


def _record_row(record: BatchCaseRecord, axis_ids: tuple[str, ...], *, include_runtime: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": record.case_id,
        "status": record.status,
        "error": record.error,
        "case_directory": str(record.case_directory),
        "run_yaml": str(record.run_yaml_path),
    }
    for axis_id in axis_ids:
        row[f"axis_{axis_id}"] = record.selections.get(axis_id, "")
    if include_runtime:
        row.update(
            {
                "runtime_s": "" if record.runtime_s is None else record.runtime_s,
                "output_directory": "" if record.output_directory is None else str(record.output_directory),
                "heat_balance_max_abs_error": _balance_field(record, "heat", "max_abs_error"),
                "heat_balance_time_s": _balance_field(record, "heat", "time_s"),
                "heat_balance_unit": _balance_field(record, "heat", "unit"),
                "mass_balance_max_abs_error": _balance_field(record, "mass", "max_abs_error"),
                "mass_balance_time_s": _balance_field(record, "mass", "time_s"),
                "mass_balance_unit": _balance_field(record, "mass", "unit"),
            }
        )
    return row


def _write_records_csv(
    path: Path,
    records: tuple[BatchCaseRecord, ...],
    axis_ids: tuple[str, ...],
    *,
    include_runtime: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _record_columns(axis_ids, include_runtime=include_runtime)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(_record_row(record, axis_ids, include_runtime=include_runtime))


def _default_batch_functions():
    from .cli import generate_artifacts, run_simulation

    return generate_artifacts, run_simulation


def _run_case_direct(
    run_bundle: RunBundle,
    generate_artifacts_fn: Callable[[RunBundle], dict[str, Path]] | None,
    run_simulation_fn: Callable[..., RunResult],
) -> tuple[Path, dict[str, Any]]:
    artifact_paths = generate_artifacts_fn(run_bundle) if generate_artifacts_fn is not None else {}
    run_result = run_simulation_fn(run_bundle, artifact_paths=artifact_paths)
    return run_result.output_directory, dict(run_result.balance_errors)


def _run_case_worker(
    run_bundle: RunBundle,
    generate_artifacts_fn: Callable[[RunBundle], dict[str, Path]] | None,
    run_simulation_fn: Callable[..., RunResult],
    result_queue,
) -> None:
    try:
        output_directory, balance_errors = _run_case_direct(
            run_bundle,
            generate_artifacts_fn,
            run_simulation_fn,
        )
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})
        return

    result_queue.put(
        {
            "ok": True,
            "output_directory": output_directory,
            "balance_errors": balance_errors,
        }
    )


def _run_case_with_timeout(
    run_bundle: RunBundle,
    generate_artifacts_fn: Callable[[RunBundle], dict[str, Path]] | None,
    run_simulation_fn: Callable[..., RunResult],
    timeout_s: float,
) -> tuple[Path, dict[str, Any]]:
    coerced_timeout_s = _coerce_case_timeout_s(timeout_s)
    if coerced_timeout_s is None:
        raise ValueError("timeout_s must be provided.")
    timeout_s = coerced_timeout_s
    context = mp.get_context()
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_run_case_worker,
        args=(run_bundle, generate_artifacts_fn, run_simulation_fn, result_queue),
    )

    try:
        process.start()
        process.join(timeout_s)
        if process.is_alive():
            process.terminate()
            process.join(_PROCESS_TERMINATE_GRACE_S)
            if process.is_alive():
                kill = getattr(process, "kill", None)
                if kill is not None:
                    kill()
                    process.join()
            raise BatchCaseTimeoutError(f"Timed out after {timeout_s:g} seconds.")

        try:
            payload = result_queue.get(timeout=1.0)
        except Empty as exc:
            if process.exitcode == 0:
                message = "Batch case worker finished without returning a result."
            else:
                message = f"Batch case worker exited with code {process.exitcode}."
            raise BatchCaseWorkerError(message) from exc
    except Exception as exc:
        if isinstance(exc, (BatchCaseTimeoutError, BatchCaseWorkerError)):
            raise
        raise BatchCaseWorkerError(
            "Could not start the batch case worker process. "
            "When a timeout is set, custom batch functions must be picklable."
        ) from exc
    finally:
        result_queue.close()

    if payload.get("ok"):
        return Path(payload["output_directory"]), dict(payload["balance_errors"])
    raise BatchCaseWorkerError(str(payload.get("error", "Batch case worker failed.")))


def run_batch_file(
    batch_yaml_path: str | Path,
    *,
    validate_only: bool = False,
    case_timeout_s: float | None = None,
    generate_artifacts_fn: Callable[[RunBundle], dict[str, Path]] | None = None,
    run_simulation_fn: Callable[..., RunResult] | None = None,
) -> BatchResult:
    document = load_batch_spec(batch_yaml_path)
    effective_case_timeout_s = (
        _coerce_case_timeout_s(case_timeout_s) if case_timeout_s is not None else document.spec.case_timeout_s
    )
    cases = materialize_batch_cases(document)
    records = tuple(
        BatchCaseRecord(
            case_id=case.case_id,
            selections=case.selections,
            case_directory=case.case_directory,
            run_yaml_path=case.run_yaml_path,
        )
        for case in cases
    )
    axis_ids = tuple(axis.id for axis in document.spec.axes)
    manifest_path = document.output_directory / "manifest.csv"
    summary_path = document.output_directory / "summary.csv"

    default_generate_artifacts = None
    default_run_simulation = None
    if not validate_only and (generate_artifacts_fn is None or run_simulation_fn is None):
        default_generate_artifacts, default_run_simulation = _default_batch_functions()
    if generate_artifacts_fn is None:
        generate_artifacts_fn = default_generate_artifacts
    if run_simulation_fn is None:
        run_simulation_fn = default_run_simulation
    if not validate_only and run_simulation_fn is None:
        raise RuntimeError("No batch simulation function is configured.")

    for record in records:
        try:
            run_bundle = load_run_bundle(record.run_yaml_path)
        except Exception as exc:
            record.status = "validation_failed"
            record.error = str(exc)
            continue

        record.output_directory = run_bundle.output_directory
        if validate_only:
            record.status = "validation_passed"
            continue

        try:
            start = perf_counter()
            if effective_case_timeout_s is None:
                output_directory, balance_errors = _run_case_direct(
                    run_bundle,
                    generate_artifacts_fn,
                    run_simulation_fn,
                )
            else:
                output_directory, balance_errors = _run_case_with_timeout(
                    run_bundle,
                    generate_artifacts_fn,
                    run_simulation_fn,
                    effective_case_timeout_s,
                )
            record.runtime_s = perf_counter() - start
            record.output_directory = output_directory
            record.balance_errors = balance_errors
            record.status = "success"
        except BatchCaseTimeoutError as exc:
            record.runtime_s = perf_counter() - start
            record.status = "timeout_failed"
            record.error = str(exc)
        except Exception as exc:
            record.runtime_s = perf_counter() - start
            record.status = "simulation_failed"
            record.error = str(exc)

    _write_records_csv(manifest_path, records, axis_ids, include_runtime=False)
    if validate_only:
        return BatchResult(
            batch_path=document.batch_path,
            output_directory=document.output_directory,
            manifest_path=manifest_path,
            summary_path=None,
            records=records,
        )

    _write_records_csv(summary_path, records, axis_ids, include_runtime=True)
    return BatchResult(
        batch_path=document.batch_path,
        output_directory=document.output_directory,
        manifest_path=manifest_path,
        summary_path=summary_path,
        records=records,
    )


__all__ = [
    "BatchAxis",
    "BatchAxisValue",
    "BatchCaseTimeoutError",
    "BatchDocument",
    "BatchResult",
    "BatchSpec",
    "BatchValidationError",
    "GeometryPreset",
    "MaterializedBatchCase",
    "NORMAL_MOLAR_DENSITY_MOL_PER_M3",
    "expand_batch_cases",
    "load_batch_spec",
    "materialize_batch_cases",
    "run_batch_file",
]
