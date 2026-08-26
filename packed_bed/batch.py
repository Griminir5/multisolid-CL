from __future__ import annotations

import csv
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import re
from time import perf_counter
from typing import Any, Callable
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
import yaml

from .config import Case, PackedBedValidationError, resolve_case
from .config.load import read_yaml_mapping, resolve_path
from .config.models import ConfigString
from .reports import RunResult


_PROCESS_TERMINATE_GRACE_S = 5.0
_PROCESS_POLL_INTERVAL_S = 0.05
_SLUG_UNSAFE_RE = re.compile(r"[^a-z0-9]+")
_SINGLE_THREAD_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class BatchValidationError(PackedBedValidationError):
    pass


class BatchCaseTimeoutError(TimeoutError):
    pass


class BatchCaseWorkerError(RuntimeError):
    pass


class BatchConfigModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


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


def _coerce_case_timeout_s(value: Any, label: str = "case_timeout_s") -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be numeric seconds.")
    timeout_s = float(value)
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError(f"{label} must be a finite positive number of seconds.")
    return timeout_s


def _coerce_workers(value: Any, label: str = "workers") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return value


class BatchPatch(BatchConfigModel):
    run: dict[ConfigString, Any] = Field(default_factory=dict)
    chemistry: dict[ConfigString, Any] = Field(default_factory=dict)
    program: dict[ConfigString, Any] = Field(default_factory=dict)
    solids: dict[ConfigString, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_not_empty(self) -> "BatchPatch":
        if not any((self.run, self.chemistry, self.program, self.solids)):
            raise ValueError("must change at least one case document.")
        return self


class BatchAxisValue(BatchConfigModel):
    id: ConfigString
    program: ConfigString | None = None
    geometry: ConfigString | None = None
    patch: BatchPatch | None = None

    @model_validator(mode="after")
    def validate_has_effect(self) -> "BatchAxisValue":
        if self.program is None and self.geometry is None and self.patch is None:
            raise ValueError("must specify at least one of: program, geometry, patch.")
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

    @model_validator(mode="after")
    def validate_has_effect(self) -> "GeometryPreset":
        if not self.model and self.solids_file is None:
            raise ValueError("must specify model values or solids_file.")
        return self


class BatchSpec(BatchConfigModel):
    base_case: ConfigString
    output_directory: ConfigString
    case_timeout_s: float | None = None
    workers: int = 1
    artifacts: bool = False
    programs: dict[ConfigString, ConfigString] = Field(default_factory=dict)
    geometries: dict[ConfigString, GeometryPreset] = Field(default_factory=dict)
    axes: tuple[BatchAxis, ...]

    @field_validator("case_timeout_s", mode="before")
    @classmethod
    def validate_case_timeout_s(cls, value: Any) -> float | None:
        return _coerce_case_timeout_s(value)

    @field_validator("workers", mode="before")
    @classmethod
    def validate_workers(cls, value: Any) -> int:
        return _coerce_workers(value)

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
                        f"axis '{axis.id}' value '{axis_value.id}' references unknown "
                        f"program '{axis_value.program}'."
                    )
                if axis_value.geometry is not None and axis_value.geometry not in self.geometries:
                    raise ValueError(
                        f"axis '{axis.id}' value '{axis_value.id}' references unknown "
                        f"geometry '{axis_value.geometry}'."
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
        return resolve_path(self.base_dir, self.spec.output_directory)


@dataclass(frozen=True)
class BaseCaseMappings:
    run: dict[str, Any]
    chemistry: dict[str, Any]
    program: dict[str, Any]
    solids: dict[str, Any]


@dataclass(frozen=True)
class ExpandedBatchCase:
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
    plot_status: str = "not_requested"
    plot_errors: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchResult:
    batch_path: Path
    output_directory: Path
    summary_path: Path | None
    records: tuple[BatchCaseRecord, ...]
    workers: int = 1

    @property
    def total_count(self) -> int:
        return len(self.records)

    @property
    def failed_count(self) -> int:
        return sum(1 for record in self.records if record.status.endswith("_failed"))

    @property
    def plot_failed_count(self) -> int:
        return sum(1 for record in self.records if record.plot_errors)


def load_batch_spec(batch_yaml_path: str | Path) -> BatchDocument:
    batch_path = Path(batch_yaml_path).resolve()
    data = read_yaml_mapping(batch_path, "batch")
    try:
        spec = BatchSpec.model_validate(data)
    except ValidationError as exc:
        raise _format_validation_error("batch", batch_path, exc) from exc
    return BatchDocument(batch_path=batch_path, spec=spec)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BatchValidationError(f"{label} must be a mapping.")
    return value


def _load_base_case(document: BatchDocument) -> BaseCaseMappings:
    run_path = resolve_path(document.base_dir, document.spec.base_case)
    run = read_yaml_mapping(run_path, "run")
    references = _require_mapping(run.get("references"), "run.references")
    base_dir = run_path.parent
    return BaseCaseMappings(
        run=run,
        chemistry=read_yaml_mapping(
            resolve_path(base_dir, references.get("chemistry_file", "")), "chemistry"
        ),
        program=read_yaml_mapping(
            resolve_path(base_dir, references.get("program_file", "")), "program"
        ),
        solids=read_yaml_mapping(
            resolve_path(base_dir, references.get("solids_file", "")), "solids"
        ),
    )


def _merge_mapping(target: dict[str, Any], patch: dict[str, Any]) -> None:
    """Apply a deterministic recursive mapping patch; sequences replace wholesale."""

    for key, patch_value in patch.items():
        current_value = target.get(key)
        if isinstance(current_value, dict) and isinstance(patch_value, dict):
            _merge_mapping(current_value, patch_value)
        else:
            target[key] = deepcopy(patch_value)


def _apply_patch(documents: dict[str, dict[str, Any]], patch: BatchPatch) -> None:
    for name in ("run", "chemistry", "program", "solids"):
        values = getattr(patch, name)
        if values:
            _merge_mapping(documents[name], values)


def _safe_slug(value: str, *, max_length: int = 64) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = _SLUG_UNSAFE_RE.sub("-", ascii_value.lower()).strip("-")
    if not slug:
        slug = f"id-{hashlib.sha256(value.encode()).hexdigest()[:8]}"
    if len(slug) > max_length:
        digest = hashlib.sha256(value.encode()).hexdigest()[:8]
        slug = f"{slug[: max_length - 9].rstrip('-')}-{digest}"
    return slug


def _contained_path(root: Path, path: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise BatchValidationError(
            f"{label} escapes batch output directory {resolved_root}: {resolved_path}"
        ) from exc
    return resolved_path


def _force_case_run_fields(run: dict[str, Any], case_id: str) -> None:
    references = _require_mapping(run.setdefault("references", {}), "run.references")
    references.update(
        chemistry_file="chemistry.yaml",
        program_file="program.yaml",
        solids_file="solids.yaml",
    )
    simulation = _require_mapping(run.setdefault("simulation", {}), "run.simulation")
    system_name = case_id.replace("-", "_")
    if not system_name[0].isalpha():
        system_name = f"case_{system_name}"
    simulation["system_name"] = system_name
    outputs = _require_mapping(run.setdefault("outputs", {}), "run.outputs")
    outputs.update(directory="output", artifacts_directory="output/artifacts")


def expand_batch_cases(document: BatchDocument) -> tuple[ExpandedBatchCase, ...]:
    base = _load_base_case(document)
    program_presets = {
        name: read_yaml_mapping(resolve_path(document.base_dir, path), f"programs.{name}")
        for name, path in document.spec.programs.items()
    }
    geometry_solids = {
        name: read_yaml_mapping(
            resolve_path(document.base_dir, preset.solids_file),
            f"geometries.{name}.solids_file",
        )
        for name, preset in document.spec.geometries.items()
        if preset.solids_file is not None
    }
    axis_value_groups = [
        tuple((axis, axis_value) for axis_value in axis.values)
        for axis in document.spec.axes
    ]
    case_root = _contained_path(
        document.output_directory,
        document.output_directory / "cases",
        "batch case root",
    )
    cases: list[ExpandedBatchCase] = []

    for axis_values in itertools.product(*axis_value_groups):
        case_id = "__".join(
            f"{_safe_slug(axis.id)}-{_safe_slug(value.id)}" for axis, value in axis_values
        )
        if len(case_id) > 180:
            digest = hashlib.sha256(case_id.encode()).hexdigest()[:8]
            case_id = f"{case_id[:171].rstrip('-_')}-{digest}"
        documents = {
            "run": deepcopy(base.run),
            "chemistry": deepcopy(base.chemistry),
            "program": deepcopy(base.program),
            "solids": deepcopy(base.solids),
        }
        selections = {axis.id: axis_value.id for axis, axis_value in axis_values}

        for _axis, axis_value in axis_values:
            if axis_value.program is not None:
                documents["program"] = deepcopy(program_presets[axis_value.program])
            if axis_value.geometry is not None:
                preset = document.spec.geometries[axis_value.geometry]
                model = _require_mapping(documents["run"].get("model"), "run.model")
                _merge_mapping(model, preset.model)
                if axis_value.geometry in geometry_solids:
                    documents["solids"] = deepcopy(geometry_solids[axis_value.geometry])
            if axis_value.patch is not None:
                _apply_patch(documents, axis_value.patch)

        _force_case_run_fields(documents["run"], case_id)
        case_directory = _contained_path(
            document.output_directory,
            case_root / case_id,
            f"case '{case_id}' directory",
        )
        cases.append(
            ExpandedBatchCase(
                case_id=case_id,
                selections=selections,
                case_directory=case_directory,
                run_yaml_path=case_directory / "run.yaml",
                run=documents["run"],
                chemistry=documents["chemistry"],
                program=documents["program"],
                solids=documents["solids"],
            )
        )

    by_slug: dict[str, dict[str, str]] = {}
    for case in cases:
        if case.case_id in by_slug:
            raise BatchValidationError(
                f"Batch case slug collision for '{case.case_id}': "
                f"{by_slug[case.case_id]} and {case.selections}."
            )
        by_slug[case.case_id] = case.selections
    return tuple(cases)


def _resolve_expanded_case(expanded: ExpandedBatchCase) -> Case:
    return resolve_case(
        run_path=expanded.run_yaml_path,
        chemistry_path=expanded.case_directory / "chemistry.yaml",
        program_path=expanded.case_directory / "program.yaml",
        solids_path=expanded.case_directory / "solids.yaml",
        run_data=expanded.run,
        chemistry_data=expanded.chemistry,
        program_data=expanded.program,
        solids_data=expanded.solids,
    )


def _write_yaml(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def _write_case_files(cases: tuple[ExpandedBatchCase, ...]) -> None:
    for case in cases:
        _write_yaml(case.run_yaml_path, case.run)
        _write_yaml(case.case_directory / "chemistry.yaml", case.chemistry)
        _write_yaml(case.case_directory / "program.yaml", case.program)
        _write_yaml(case.case_directory / "solids.yaml", case.solids)


def _force_single_threaded_cases(cases: tuple[ExpandedBatchCase, ...]) -> None:
    for case in cases:
        solver = _require_mapping(case.run.get("solver"), "run.solver")
        solver["threads"] = 1


def _check_output_collisions(
    document: BatchDocument,
    cases: tuple[ExpandedBatchCase, ...],
    summary_path: Path,
) -> None:
    existing = [case.case_directory for case in cases if case.case_directory.exists()]
    if summary_path.exists():
        existing.append(summary_path)
    if document.output_directory.exists() and not document.output_directory.is_dir():
        existing.append(document.output_directory)
    if existing:
        rendered = "\n".join(f"- {path}" for path in existing)
        raise BatchValidationError(
            f"Batch outputs already exist; refusing to overwrite:\n{rendered}"
        )

    for case in cases:
        output = case.case_directory / "output"
        _contained_path(case.case_directory, output, f"case '{case.case_id}' output")
    _contained_path(document.output_directory, summary_path, "batch summary")


def _balance_field(record: BatchCaseRecord, key: str, attribute: str) -> str | float:
    error = record.balance_errors.get(key)
    return "" if error is None else getattr(error, attribute)


def _write_records_csv(
    path: Path,
    records: tuple[BatchCaseRecord, ...],
    axis_ids: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "case_id",
        *(f"axis_{axis_id}" for axis_id in axis_ids),
        "status",
        "error",
        "case_directory",
        "run_yaml",
        "runtime_s",
        "output_directory",
        "plot_status",
        "plot_errors",
        "heat_balance_max_abs_error",
        "heat_balance_time_s",
        "heat_balance_unit",
        "mass_balance_max_abs_error",
        "mass_balance_time_s",
        "mass_balance_unit",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "case_id": record.case_id,
                    "status": record.status,
                    "error": record.error,
                    "case_directory": str(record.case_directory),
                    "run_yaml": str(record.run_yaml_path),
                    **{
                        f"axis_{axis}": record.selections.get(axis, "")
                        for axis in axis_ids
                    },
                    "runtime_s": "" if record.runtime_s is None else record.runtime_s,
                    "output_directory": (
                        ""
                        if record.output_directory is None
                        else str(record.output_directory)
                    ),
                    "plot_status": record.plot_status,
                    "plot_errors": json.dumps(record.plot_errors, sort_keys=True),
                    "heat_balance_max_abs_error": _balance_field(
                        record, "heat", "max_abs_error"
                    ),
                    "heat_balance_time_s": _balance_field(record, "heat", "time_s"),
                    "heat_balance_unit": _balance_field(record, "heat", "unit"),
                    "mass_balance_max_abs_error": _balance_field(
                        record, "mass", "max_abs_error"
                    ),
                    "mass_balance_time_s": _balance_field(record, "mass", "time_s"),
                    "mass_balance_unit": _balance_field(record, "mass", "unit"),
                }
            )


def _run_case_direct(
    case: Case,
    generate_artifacts_fn: Callable[[Case], dict[str, Path]] | None,
    run_case_fn: Callable[..., RunResult] | None,
) -> tuple[Path, dict[str, Any], str, dict[str, str]]:
    if run_case_fn is None:
        from .simulation import run_case

        run_case_fn = run_case
    artifact_paths = generate_artifacts_fn(case) if generate_artifacts_fn is not None else {}
    result = run_case_fn(
        case,
        artifact_paths=artifact_paths,
    )
    return (
        result.output_directory,
        dict(result.balance_errors),
        result.plot_status,
        dict(result.plot_errors),
    )


def _run_case_worker(
    run_yaml_path: Path,
    single_threaded: bool,
    generate_artifacts_fn,
    run_case_fn,
    result_queue,
) -> None:
    if single_threaded:
        os.environ.update(_SINGLE_THREAD_ENVIRONMENT)
        os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    try:
        from .config import load_case

        case = load_case(run_yaml_path)
        output_directory, balance_errors, plot_status, plot_errors = _run_case_direct(
            case,
            generate_artifacts_fn,
            run_case_fn,
        )
        result_queue.put(
            {
                "ok": True,
                "output_directory": output_directory,
                "balance_errors": balance_errors,
                "plot_status": plot_status,
                "plot_errors": plot_errors,
            }
        )
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})


@contextmanager
def _single_threaded_worker_environment():
    previous = {name: os.environ.get(name) for name in _SINGLE_THREAD_ENVIRONMENT}
    threading_layer = os.environ.get("MKL_THREADING_LAYER")
    os.environ.update(_SINGLE_THREAD_ENVIRONMENT)
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if threading_layer is None:
            os.environ.pop("MKL_THREADING_LAYER", None)
        else:
            os.environ["MKL_THREADING_LAYER"] = threading_layer


@dataclass
class _ActiveCaseWorker:
    process: Any
    result_queue: Any
    record: BatchCaseRecord
    started_at: float


def _terminate_process(process) -> None:
    if not process.is_alive():
        process.join()
        return
    process.terminate()
    process.join(_PROCESS_TERMINATE_GRACE_S)
    if process.is_alive() and getattr(process, "kill", None) is not None:
        process.kill()
        process.join()


def _close_worker(worker: _ActiveCaseWorker) -> None:
    worker.result_queue.close()
    close_process = getattr(worker.process, "close", None)
    if close_process is not None:
        close_process()


def _apply_worker_payload(record: BatchCaseRecord, payload: dict[str, Any]) -> None:
    if not payload.get("ok"):
        raise BatchCaseWorkerError(str(payload.get("error", "Batch case worker failed.")))
    record.output_directory = Path(payload["output_directory"])
    record.balance_errors = dict(payload["balance_errors"])
    record.plot_status = str(payload["plot_status"])
    record.plot_errors = dict(payload["plot_errors"])
    record.status = "success"


def _run_cases_in_processes(
    cases: tuple[Case, ...],
    records: tuple[BatchCaseRecord, ...],
    *,
    workers: int,
    timeout_s: float | None,
    generate_artifacts_fn: Callable[[Case], dict[str, Path]] | None,
    run_case_fn: Callable[..., RunResult] | None,
) -> None:
    context = mp.get_context("spawn")
    pending = iter(zip(cases, records))
    active: list[_ActiveCaseWorker] = []
    exhausted = False
    try:
        while active or not exhausted:
            while len(active) < workers and not exhausted:
                try:
                    case, record = next(pending)
                except StopIteration:
                    exhausted = True
                    break

                record.status = "running"
                started_at = perf_counter()
                single_threaded = case.run.solver.threads == 1
                result_queue = context.Queue(maxsize=1)
                process = context.Process(
                    target=_run_case_worker,
                    args=(
                        case.run_path,
                        single_threaded,
                        generate_artifacts_fn,
                        run_case_fn,
                        result_queue,
                    ),
                )
                try:
                    # Apply limits before the child imports NumPy or DAETools.
                    if single_threaded:
                        with _single_threaded_worker_environment():
                            process.start()
                    else:
                        process.start()
                except Exception as exc:
                    if getattr(process, "pid", None) is not None:
                        _terminate_process(process)
                        close_process = getattr(process, "close", None)
                        if close_process is not None:
                            close_process()
                    result_queue.close()
                    record.status = "simulation_failed"
                    record.error = (
                        "Could not start the batch case worker; custom functions must be "
                        f"picklable. {exc}"
                    )
                    record.runtime_s = perf_counter() - started_at
                    continue
                active.append(
                    _ActiveCaseWorker(
                        process=process,
                        result_queue=result_queue,
                        record=record,
                        started_at=started_at,
                    )
                )

            made_progress = False
            now = perf_counter()
            for worker in tuple(active):
                elapsed = now - worker.started_at
                if (
                    timeout_s is not None
                    and elapsed >= timeout_s
                    and worker.process.is_alive()
                ):
                    _terminate_process(worker.process)
                    worker.record.status = "timeout_failed"
                    worker.record.error = f"Timed out after {timeout_s:g} seconds."
                elif worker.process.is_alive():
                    continue
                else:
                    worker.process.join()
                    try:
                        payload = worker.result_queue.get(timeout=1.0)
                        _apply_worker_payload(worker.record, payload)
                    except Empty:
                        worker.record.status = "simulation_failed"
                        worker.record.error = (
                            "Batch case worker finished without returning a result."
                            if worker.process.exitcode == 0
                            else (
                                "Batch case worker exited with code "
                                f"{worker.process.exitcode}."
                            )
                        )
                    except Exception as exc:
                        worker.record.status = "simulation_failed"
                        worker.record.error = str(exc)

                worker.record.runtime_s = perf_counter() - worker.started_at
                _close_worker(worker)
                active.remove(worker)
                made_progress = True

            if active and not made_progress:
                active[0].process.join(_PROCESS_POLL_INTERVAL_S)
    except BaseException:
        for worker in active:
            _terminate_process(worker.process)
            worker.record.runtime_s = perf_counter() - worker.started_at
            _close_worker(worker)
        raise


def run_batch_file(
    batch_yaml_path: str | Path,
    *,
    validate_only: bool = False,
    case_timeout_s: float | None = None,
    workers: int | None = None,
    generate_artifacts_fn: Callable[[Case], dict[str, Path]] | None = None,
    run_case_fn: Callable[..., RunResult] | None = None,
) -> BatchResult:
    document = load_batch_spec(batch_yaml_path)
    requested_workers = (
        _coerce_workers(workers) if workers is not None else document.spec.workers
    )
    timeout_s = (
        _coerce_case_timeout_s(case_timeout_s)
        if case_timeout_s is not None
        else document.spec.case_timeout_s
    )
    expanded_cases = expand_batch_cases(document)
    if requested_workers > 1:
        _force_single_threaded_cases(expanded_cases)
    effective_workers = min(requested_workers, len(expanded_cases))
    records = tuple(
        BatchCaseRecord(
            case_id=expanded.case_id,
            selections=expanded.selections,
            case_directory=expanded.case_directory,
            run_yaml_path=expanded.run_yaml_path,
        )
        for expanded in expanded_cases
    )
    resolved_cases: list[Case | None] = []
    for expanded, record in zip(expanded_cases, records):
        try:
            case = _resolve_expanded_case(expanded)
        except Exception as exc:
            resolved_cases.append(None)
            record.status = "validation_failed"
            record.error = str(exc)
        else:
            resolved_cases.append(case)
            record.status = "validation_passed"
            record.output_directory = case.output_directory

    result_without_files = BatchResult(
        batch_path=document.batch_path,
        output_directory=document.output_directory,
        summary_path=None,
        records=records,
        workers=effective_workers,
    )
    if validate_only or any(case is None for case in resolved_cases):
        return result_without_files

    summary_path = document.output_directory / "summary.csv"
    _check_output_collisions(document, expanded_cases, summary_path)
    _write_case_files(expanded_cases)

    if generate_artifacts_fn is None and document.spec.artifacts:
        from .artifacts import generate_artifacts

        generate_artifacts_fn = generate_artifacts

    cases = tuple(case for case in resolved_cases if case is not None)
    if effective_workers > 1 or timeout_s is not None:
        _run_cases_in_processes(
            cases,
            records,
            workers=effective_workers,
            timeout_s=timeout_s,
            generate_artifacts_fn=generate_artifacts_fn,
            run_case_fn=run_case_fn,
        )
    else:
        for case, record in zip(cases, records):
            record.status = "running"
            start = perf_counter()
            try:
                output_directory, balance_errors, plot_status, plot_errors = _run_case_direct(
                    case,
                    generate_artifacts_fn,
                    run_case_fn,
                )
                record.output_directory = output_directory
                record.balance_errors = balance_errors
                record.plot_status = plot_status
                record.plot_errors = plot_errors
                record.status = "success"
            except Exception as exc:
                record.status = "simulation_failed"
                record.error = str(exc)
            finally:
                record.runtime_s = perf_counter() - start

    axis_ids = tuple(axis.id for axis in document.spec.axes)
    _write_records_csv(summary_path, records, axis_ids)
    return BatchResult(
        batch_path=document.batch_path,
        output_directory=document.output_directory,
        summary_path=summary_path,
        records=records,
        workers=effective_workers,
    )


__all__ = (
    "BatchAxis",
    "BatchAxisValue",
    "BatchCaseTimeoutError",
    "BatchDocument",
    "BatchPatch",
    "BatchResult",
    "BatchSpec",
    "BatchValidationError",
    "ExpandedBatchCase",
    "GeometryPreset",
    "expand_batch_cases",
    "load_batch_spec",
    "run_batch_file",
)
