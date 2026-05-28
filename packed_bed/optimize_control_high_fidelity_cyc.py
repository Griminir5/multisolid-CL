#!/usr/bin/env python3
"""High-fidelity NSGA-II cyclic valve-control optimization.

This mirrors the nearby surrogates optimizer
``src/surrogates/optimization/optimize_control GEN_cyc.py`` but evaluates each
genome by running the packed-bed simulator directly.

The default run/chemistry/solids bundle is the latest
``packed_bed/examples/optim*_batch_case/base_case`` in this checkout. The
default context program is the 3000 s unoptimized program from the adjacent
``surrogates`` repo, if present, because that is the context used by the
surrogate optimizer. Population and generation defaults are intentionally small
because every objective evaluation is a full DAETools simulation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SURROGATES_ROOT = REPO_ROOT.parent / "surrogates"

SPECIES = ("Ar", "CH4", "CO", "CO2", "H2", "H2O", "He", "N2", "O2")
VALVE_NAMES = ("air", "steam", "fuel", "offgas")

S = 0
C = 3000
P = 4000
FINAL_TIME_S = 12000.0

N_STEPS = 5
VARIABLES_PER_STEP = 6
MIN_STEP_S = 10.0
MAX_STEP_S = 500.0
RAMP_DURATION_S = 5.0

FIXED_INLET_T_K = 600.0
FIXED_OUTLET_P_PA = 500000.0
FLOW_LIMITS_MOL_S = (0.0005, 0.003)

T_MAX_K = 1500.0
TEMP_EPS_K = 10.0
EPS_MIX = 1.0e-3
FAIL_OBJECTIVE = 1.0e9
FAIL_CONSTRAINT = 1.0e6
CV_TOL = 1.0e-6
DURATION_ATOL_S = 1.0e-6

IDX_CH4_IN = "CH4"
IDX_CO_IN = "CO"
IDX_CO2_IN = "CO2"
IDX_H2_IN = "H2"
IDX_O2_IN = "O2"

DEFAULT_CONTEXT_PROGRAM = (
    SURROGATES_ROOT / "src" / "surrogates" / "optimization" / "unoptimized.yaml"
)
DEFAULT_OUT_DIR = REPO_ROOT / "active_learning_optimization" / "high_fidelity_cyc_nsga2_round2"


def _valve_basis() -> np.ndarray:
    air = np.zeros(len(SPECIES), dtype=float)
    steam = np.zeros(len(SPECIES), dtype=float)
    fuel = np.zeros(len(SPECIES), dtype=float)
    offgas = np.zeros(len(SPECIES), dtype=float)

    air[SPECIES.index("Ar")] = 0.0093
    air[SPECIES.index("CO2")] = 0.0004
    air[SPECIES.index("O2")] = 0.2095
    air[SPECIES.index("N2")] = 0.7808

    steam[SPECIES.index("H2O")] = 1.0
    fuel[SPECIES.index("CH4")] = 1.0

    offgas[SPECIES.index("CO2")] = 0.85
    offgas[SPECIES.index("CO")] = 0.05
    offgas[SPECIES.index("CH4")] = 0.025
    offgas[SPECIES.index("H2")] = 0.075

    return np.stack([air, steam, fuel, offgas], axis=0)


VALVE_BASIS = _valve_basis()


@dataclass(frozen=True)
class DecodedGenome:
    durations_s: np.ndarray
    flows_mol_s: np.ndarray
    valve_weights: np.ndarray
    compositions: np.ndarray

    @property
    def cycle_length_s(self) -> float:
        return float(np.sum(self.durations_s))


@dataclass(frozen=True)
class EvaluationResult:
    genome_hash: str
    case_id: str
    status: str
    objectives: tuple[float, float]
    constraints: tuple[float, float, float]
    runtime_s: float
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "genome_hash": self.genome_hash,
            "case_id": self.case_id,
            "status": self.status,
            "objectives": list(self.objectives),
            "constraints": list(self.constraints),
            "runtime_s": self.runtime_s,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "EvaluationResult":
        return cls(
            genome_hash=str(data["genome_hash"]),
            case_id=str(data["case_id"]),
            status=str(data["status"]),
            objectives=tuple(float(v) for v in data["objectives"]),  # type: ignore[arg-type]
            constraints=tuple(float(v) for v in data["constraints"]),  # type: ignore[arg-type]
            runtime_s=float(data.get("runtime_s", 0.0)),
            error=str(data.get("error", "")),
        )


def _round_float(value: float, digits: int = 10) -> float:
    return float(round(float(value), digits))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=axis, keepdims=True)


def _normalize_composition(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).clip(min=0.0)
    total = float(values.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError(f"Invalid composition with sum {total}.")
    return values / total


def _composition_dict(values: np.ndarray, digits: int = 10) -> dict[str, float]:
    normalized = _normalize_composition(values)
    rounded = np.round(normalized, digits)
    rounded[rounded < 0.5 * 10.0**-digits] = 0.0
    correction_idx = int(np.argmax(rounded))
    rounded[correction_idx] = round(
        float(rounded[correction_idx] + 1.0 - float(rounded.sum())),
        digits,
    )
    return {species: float(rounded[index]) for index, species in enumerate(SPECIES)}


def _composition_array(mapping: dict[str, Any]) -> np.ndarray:
    return _normalize_composition(np.asarray([mapping[species] for species in SPECIES]))


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def _write_yaml_mapping(path: Path, mapping: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(mapping, handle, sort_keys=False, width=120)


def _step_duration_sum(steps: Sequence[dict[str, Any]]) -> float:
    return float(sum(float(step["duration_s"]) for step in steps))


def _current_scalar_value(initial: float, steps: Sequence[dict[str, Any]]) -> float:
    value = float(initial)
    for step in steps:
        if step["kind"] == "ramp":
            value = float(step["target"])
    return value


def _current_composition_value(
    initial: dict[str, Any], steps: Sequence[dict[str, Any]]
) -> np.ndarray:
    value = _composition_array(initial)
    for step in steps:
        if step["kind"] == "ramp":
            value = _composition_array(step["target"])
    return value


def _scalar_step_target(
    previous: float, step: dict[str, Any], active_duration: float
) -> float:
    if step["kind"] == "hold":
        return previous
    target = float(step["target"])
    fraction = active_duration / float(step["duration_s"])
    return float(previous + (target - previous) * fraction)


def _composition_step_target(
    previous: np.ndarray, step: dict[str, Any], active_duration: float
) -> np.ndarray:
    if step["kind"] == "hold":
        return previous
    target = _composition_array(step["target"])
    fraction = active_duration / float(step["duration_s"])
    return _normalize_composition(previous + (target - previous) * fraction)


def _append_hold(steps: list[dict[str, Any]], duration_s: float) -> None:
    if duration_s > DURATION_ATOL_S:
        steps.append({"kind": "hold", "duration_s": float(duration_s)})


def _append_ramp(
    steps: list[dict[str, Any]], duration_s: float, target: float | dict[str, float]
) -> None:
    if duration_s <= DURATION_ATOL_S:
        return
    clean_target = target if isinstance(target, dict) else _round_float(target, 12)
    steps.append(
        {
            "kind": "ramp",
            "duration_s": float(duration_s),
            "target": clean_target,
        }
    )


def _truncate_scalar_channel(channel: dict[str, Any], duration_s: float) -> dict[str, Any]:
    truncated = copy.deepcopy(channel)
    truncated["steps"] = []
    previous = float(channel["initial"])
    elapsed = 0.0

    for step in channel.get("steps", []):
        if elapsed >= duration_s - DURATION_ATOL_S:
            break
        step_duration = float(step["duration_s"])
        active_duration = min(step_duration, duration_s - elapsed)
        if active_duration <= DURATION_ATOL_S:
            break

        if step["kind"] == "hold":
            _append_hold(truncated["steps"], active_duration)
        elif step["kind"] == "ramp":
            target = _scalar_step_target(previous, step, active_duration)
            _append_ramp(truncated["steps"], active_duration, target)
            previous = target
        else:
            raise ValueError(f"Unknown scalar step kind: {step['kind']!r}")
        if step["kind"] == "hold":
            previous = _current_scalar_value(previous, [])
        elif active_duration >= step_duration - DURATION_ATOL_S:
            previous = float(step["target"])
        elapsed += active_duration

    _append_hold(truncated["steps"], duration_s - elapsed)
    return truncated


def _truncate_composition_channel(
    channel: dict[str, Any], duration_s: float
) -> dict[str, Any]:
    truncated = copy.deepcopy(channel)
    truncated["steps"] = []
    previous = _composition_array(channel["initial"])
    elapsed = 0.0

    for step in channel.get("steps", []):
        if elapsed >= duration_s - DURATION_ATOL_S:
            break
        step_duration = float(step["duration_s"])
        active_duration = min(step_duration, duration_s - elapsed)
        if active_duration <= DURATION_ATOL_S:
            break

        if step["kind"] == "hold":
            _append_hold(truncated["steps"], active_duration)
        elif step["kind"] == "ramp":
            target = _composition_step_target(previous, step, active_duration)
            _append_ramp(truncated["steps"], active_duration, _composition_dict(target))
            previous = target
        else:
            raise ValueError(f"Unknown composition step kind: {step['kind']!r}")
        if step["kind"] == "ramp" and active_duration >= step_duration - DURATION_ATOL_S:
            previous = _composition_array(step["target"])
        elapsed += active_duration

    _append_hold(truncated["steps"], duration_s - elapsed)
    return truncated


def _extend_scalar_channel_to_duration(
    channel: dict[str, Any], duration_s: float
) -> dict[str, Any]:
    extended = _truncate_scalar_channel(channel, min(duration_s, _step_duration_sum(channel.get("steps", []))))
    current_duration = _step_duration_sum(extended["steps"])
    _append_hold(extended["steps"], duration_s - current_duration)
    return extended


def _decode_genome(x: np.ndarray, n_steps: int) -> DecodedGenome:
    expected_vars = n_steps * VARIABLES_PER_STEP
    if x.shape != (expected_vars,):
        raise ValueError(f"Expected genome shape {(expected_vars,)}, got {x.shape}.")

    z = np.asarray(x, dtype=float).reshape(n_steps, VARIABLES_PER_STEP)
    durations = _sigmoid(z[:, 0]) * (MAX_STEP_S - MIN_STEP_S) + MIN_STEP_S
    flows = _sigmoid(z[:, 1]) * (FLOW_LIMITS_MOL_S[1] - FLOW_LIMITS_MOL_S[0]) + FLOW_LIMITS_MOL_S[0]
    valve_weights = _softmax(z[:, 2:6], axis=-1)
    compositions = valve_weights @ VALVE_BASIS
    compositions = compositions / np.clip(compositions.sum(axis=-1, keepdims=True), 1.0e-12, None)

    return DecodedGenome(
        durations_s=durations,
        flows_mol_s=flows,
        valve_weights=valve_weights,
        compositions=compositions,
    )


def _append_decoded_cycles(
    program: dict[str, Any],
    decoded: DecodedGenome,
    *,
    context_duration_s: float,
    final_time_s: float,
) -> dict[str, float]:
    flow_steps = program["inlet_flow"]["steps"]
    composition_steps = program["inlet_composition"]["steps"]

    flow_elapsed = _step_duration_sum(flow_steps)
    composition_elapsed = _step_duration_sum(composition_steps)
    if abs(flow_elapsed - context_duration_s) > DURATION_ATOL_S:
        raise ValueError(
            "Base inlet_flow duration must equal context before appending cycles: "
            f"got {flow_elapsed}, expected {context_duration_s}."
        )
    if abs(composition_elapsed - context_duration_s) > DURATION_ATOL_S:
        raise ValueError(
            "Base inlet_composition duration must equal context before appending cycles: "
            f"got {composition_elapsed}, expected {context_duration_s}."
        )

    elapsed = float(context_duration_s)
    previous_flow = _current_scalar_value(program["inlet_flow"]["initial"], flow_steps)
    previous_composition = _current_composition_value(
        program["inlet_composition"]["initial"], composition_steps
    )
    appended_steps = 0
    cycle_count = 0

    while elapsed < final_time_s - DURATION_ATOL_S:
        for step_index in range(decoded.durations_s.shape[0]):
            if elapsed >= final_time_s - DURATION_ATOL_S:
                break

            step_duration = float(decoded.durations_s[step_index])
            target_flow = float(decoded.flows_mol_s[step_index])
            target_composition = _normalize_composition(decoded.compositions[step_index])
            active_duration = min(step_duration, final_time_s - elapsed)
            ramp_duration = min(RAMP_DURATION_S, step_duration)

            if active_duration <= ramp_duration + DURATION_ATOL_S:
                completion = active_duration / ramp_duration
                effective_flow = previous_flow + (target_flow - previous_flow) * completion
                effective_composition = previous_composition + (
                    target_composition - previous_composition
                ) * completion
                _append_ramp(flow_steps, active_duration, effective_flow)
                _append_ramp(
                    composition_steps,
                    active_duration,
                    _composition_dict(effective_composition),
                )
                previous_flow = float(effective_flow)
                previous_composition = _normalize_composition(effective_composition)
                elapsed += active_duration
                appended_steps += 1
                break

            _append_ramp(flow_steps, ramp_duration, target_flow)
            _append_ramp(
                composition_steps,
                ramp_duration,
                _composition_dict(target_composition),
            )
            _append_hold(flow_steps, active_duration - ramp_duration)
            _append_hold(composition_steps, active_duration - ramp_duration)

            previous_flow = target_flow
            previous_composition = target_composition
            elapsed += active_duration
            appended_steps += 1

        cycle_count += 1

    return {
        "final_flow_duration_s": _step_duration_sum(flow_steps),
        "final_composition_duration_s": _step_duration_sum(composition_steps),
        "appended_step_count": appended_steps,
        "cycle_count_covered": cycle_count,
    }


def _validate_final_program(program: dict[str, Any], final_time_s: float) -> None:
    for channel_name, channel in program.items():
        duration = _step_duration_sum(channel.get("steps", []))
        if abs(duration - final_time_s) > DURATION_ATOL_S:
            raise ValueError(
                f"{channel_name}.steps must sum to {final_time_s}, got {duration}."
            )


def _build_program(
    context_program: dict[str, Any],
    decoded: DecodedGenome,
    *,
    context_duration_s: float,
    final_time_s: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    program = copy.deepcopy(context_program)
    program["inlet_flow"] = _truncate_scalar_channel(
        program["inlet_flow"], context_duration_s
    )
    program["inlet_composition"] = _truncate_composition_channel(
        program["inlet_composition"], context_duration_s
    )
    program["inlet_temperature"] = _extend_scalar_channel_to_duration(
        program["inlet_temperature"], final_time_s
    )
    program["outlet_pressure"] = _extend_scalar_channel_to_duration(
        program["outlet_pressure"], final_time_s
    )

    append_summary = _append_decoded_cycles(
        program,
        decoded,
        context_duration_s=context_duration_s,
        final_time_s=final_time_s,
    )
    _validate_final_program(program, final_time_s)
    return program, append_summary


def _latest_optim_batch_case() -> Path:
    examples = REPO_ROOT / "packed_bed" / "examples"
    matches: list[tuple[int, Path]] = []
    for path in examples.glob("optim*_batch_case"):
        match = re.fullmatch(r"optim(\d+)_batch_case", path.name)
        if match and (path / "base_case" / "run.yaml").exists():
            matches.append((int(match.group(1)), path))
    if not matches:
        raise FileNotFoundError("No packed_bed/examples/optim*_batch_case directory found.")
    return max(matches, key=lambda item: item[0])[1]


def _resolve_context_program(base_case_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    if DEFAULT_CONTEXT_PROGRAM.exists():
        return DEFAULT_CONTEXT_PROGRAM.resolve()
    return (base_case_dir / "base_case" / "program.yaml").resolve()


def _load_base_case(base_case_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_run_path = base_case_dir / "base_case" / "run.yaml"
    if not base_run_path.exists():
        raise FileNotFoundError(f"Expected base run YAML at {base_run_path}")
    run = _read_yaml_mapping(base_run_path)
    base_dir = base_run_path.parent
    references = run.get("references", {})
    chemistry = _read_yaml_mapping(base_dir / references.get("chemistry_file", "chemistry.yaml"))
    solids = _read_yaml_mapping(base_dir / references.get("solids_file", "solids.yaml"))
    return run, chemistry, solids


def _materialize_case(
    case_dir: Path,
    *,
    case_id: str,
    base_run: dict[str, Any],
    chemistry: dict[str, Any],
    solids: dict[str, Any],
    program: dict[str, Any],
    final_time_s: float,
) -> Path:
    run = copy.deepcopy(base_run)
    run.setdefault("references", {})
    run["references"]["chemistry_file"] = "chemistry.yaml"
    run["references"]["program_file"] = "program.yaml"
    run["references"]["solids_file"] = "solids.yaml"

    run.setdefault("simulation", {})
    run["simulation"]["system_name"] = case_id
    run["simulation"]["time_horizon_s"] = float(final_time_s)
    run["simulation"]["repeat_program"] = False

    run.setdefault("outputs", {})
    run["outputs"]["directory"] = "output"
    run["outputs"]["artifacts_directory"] = "output/artifacts"
    requested_reports = list(run["outputs"].get("requested_reports", []))
    for report_id in ("temperature", "pressure", "gas_mole_fraction", "heat_balance", "mass_balance"):
        if report_id not in requested_reports:
            requested_reports.append(report_id)
    run["outputs"]["requested_reports"] = requested_reports

    case_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml_mapping(case_dir / "run.yaml", run)
    _write_yaml_mapping(case_dir / "chemistry.yaml", chemistry)
    _write_yaml_mapping(case_dir / "solids.yaml", solids)
    _write_yaml_mapping(case_dir / "program.yaml", program)
    return case_dir / "run.yaml"


def _interp_rows(source_x: np.ndarray, values: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    return np.vstack([np.interp(target_x, source_x, row) for row in values])


def _extract_objectives_and_constraints(
    output_dir: Path,
    *,
    bed_length_m: float,
    start_row: int,
    window_length: int,
) -> tuple[tuple[float, float], tuple[float, float, float]]:
    reports = pd.read_pickle(output_dir / "reports.pkl").iloc[1:]
    gas = pd.read_pickle(output_dir / "gas_mole_fraction.pkl").iloc[1:]
    end_row = start_row + window_length
    if len(reports) < end_row or len(gas) < end_row:
        raise ValueError(
            f"Need at least {end_row} report rows after dropping t=0, got reports={len(reports)}, gas={len(gas)}."
        )

    reports_window = reports.iloc[start_row:end_row]
    gas_window = gas.iloc[start_row:end_row]

    outlet_flow = reports_window[("outlet_flowrate_mol_s", bed_length_m)].to_numpy(dtype=float)
    h2 = gas_window[("H2", bed_length_m)].to_numpy(dtype=float)
    co = gas_window[("CO", bed_length_m)].to_numpy(dtype=float)
    ch4 = gas_window[("CH4", bed_length_m)].to_numpy(dtype=float)
    co2 = gas_window[("CO2", bed_length_m)].to_numpy(dtype=float)
    o2 = gas_window[("O2", bed_length_m)].to_numpy(dtype=float)

    h2co_flow = outlet_flow * (h2 + co)
    syngas_objective = -float(np.mean(h2co_flow) * window_length)
    ch4_slip_objective = float(np.mean(ch4))

    temp = reports_window.xs("temperature_k", axis=1, level="feature")
    temp_x = temp.columns.to_numpy(dtype=float)
    temp_values = temp.to_numpy(dtype=float)
    sample_x = bed_length_m * np.arange(0.1, 1.0, 0.1)
    sampled_temp = np.column_stack(
        [
            _interp_rows(temp_x, temp_values, sample_x),
            _interp_rows(temp_x, temp_values, np.asarray([bed_length_m]))[:, 0],
        ]
    )
    temp_constraint = float(
        np.maximum(sampled_temp - T_MAX_K - TEMP_EPS_K, 0.0).mean()
    )

    outlet_combustibles = co + co2 + ch4 + h2
    outlet_o2_constraint = float(
        np.maximum(o2 * outlet_combustibles - EPS_MIX, 0.0).mean()
    )

    inlet_o2 = gas_window[(IDX_O2_IN, 0.0)].to_numpy(dtype=float)
    inlet_combustibles = (
        gas_window[(IDX_CO_IN, 0.0)].to_numpy(dtype=float)
        + gas_window[(IDX_CO2_IN, 0.0)].to_numpy(dtype=float)
        + gas_window[(IDX_CH4_IN, 0.0)].to_numpy(dtype=float)
        + gas_window[(IDX_H2_IN, 0.0)].to_numpy(dtype=float)
    )
    inlet_o2_constraint = float(
        np.maximum(inlet_o2 * inlet_combustibles - EPS_MIX, 0.0).mean()
    )

    return (
        (syngas_objective, ch4_slip_objective),
        (temp_constraint, outlet_o2_constraint, inlet_o2_constraint),
    )


def _hash_genome(x: np.ndarray) -> str:
    rounded = np.round(np.asarray(x, dtype=np.float64), 10)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def _read_cache(path: Path) -> dict[str, EvaluationResult]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return {key: EvaluationResult.from_json(value) for key, value in raw.items()}


def _write_cache(path: Path, cache: dict[str, EvaluationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value.to_json() for key, value in sorted(cache.items())}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    _write_yaml_mapping(path, metadata)


class HighFidelityValveReactor(Problem):
    def __init__(
        self,
        *,
        base_run: dict[str, Any],
        chemistry: dict[str, Any],
        solids: dict[str, Any],
        context_program: dict[str, Any],
        out_dir: Path,
        n_steps: int,
        context_duration_s: float,
        final_time_s: float,
        objective_window_start: int,
        objective_window_length: int,
        case_timeout_s: float | None,
        generate_artifacts: bool,
        dry_run: bool,
        rerun_failed: bool,
    ):
        self.base_run = base_run
        self.chemistry = chemistry
        self.solids = solids
        self.context_program = context_program
        self.out_dir = out_dir
        self.case_root = out_dir / "cases"
        self.n_steps = n_steps
        self.context_duration_s = context_duration_s
        self.final_time_s = final_time_s
        self.objective_window_start = objective_window_start
        self.objective_window_length = objective_window_length
        self.case_timeout_s = case_timeout_s
        self.generate_artifacts = generate_artifacts
        self.dry_run = dry_run
        self.rerun_failed = rerun_failed

        self.cache_path = out_dir / "evaluation_cache.json"
        self.eval_log_path = out_dir / "evaluation_log.csv"
        self.eval_cache = _read_cache(self.cache_path)
        self._case_counter = len(self.eval_cache)

        model = base_run.get("model", {})
        self.bed_length_m = float(model["bed_length_m"])

        super().__init__(
            n_var=n_steps * VARIABLES_PER_STEP,
            n_obj=2,
            n_ieq_constr=3,
            xl=-15.0 * np.ones(n_steps * VARIABLES_PER_STEP),
            xu=15.0 * np.ones(n_steps * VARIABLES_PER_STEP),
        )

    def _new_case_id(self, genome_hash: str) -> str:
        while True:
            case_id = f"eval_{self._case_counter:05d}_{genome_hash[:10]}"
            self._case_counter += 1
            if not (self.case_root / case_id).exists():
                return case_id

    def _record_eval(
        self,
        result: EvaluationResult,
        decoded: DecodedGenome,
        append_summary: dict[str, float] | None,
    ) -> None:
        fieldnames = [
            "genome_hash",
            "case_id",
            "status",
            "runtime_s",
            "syngas_objective",
            "ch4_slip_objective",
            "constraint_max_temperature",
            "constraint_outlet_o2_comb",
            "constraint_inlet_o2_comb",
            "cycle_length_s",
            "appended_step_count",
            "cycle_count_covered",
            "error",
        ]
        for step_index in range(self.n_steps):
            fieldnames.extend(
                [
                    f"step_{step_index}_duration_s",
                    f"step_{step_index}_flow_mol_s",
                    f"step_{step_index}_air_weight",
                    f"step_{step_index}_steam_weight",
                    f"step_{step_index}_fuel_weight",
                    f"step_{step_index}_offgas_weight",
                ]
            )

        write_header = not self.eval_log_path.exists()
        self.eval_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.eval_log_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            row: dict[str, Any] = {
                "genome_hash": result.genome_hash,
                "case_id": result.case_id,
                "status": result.status,
                "runtime_s": result.runtime_s,
                "syngas_objective": result.objectives[0],
                "ch4_slip_objective": result.objectives[1],
                "constraint_max_temperature": result.constraints[0],
                "constraint_outlet_o2_comb": result.constraints[1],
                "constraint_inlet_o2_comb": result.constraints[2],
                "cycle_length_s": decoded.cycle_length_s,
                "appended_step_count": "" if append_summary is None else append_summary["appended_step_count"],
                "cycle_count_covered": "" if append_summary is None else append_summary["cycle_count_covered"],
                "error": result.error,
            }
            for step_index in range(self.n_steps):
                row[f"step_{step_index}_duration_s"] = decoded.durations_s[step_index]
                row[f"step_{step_index}_flow_mol_s"] = decoded.flows_mol_s[step_index]
                for valve_index, valve_name in enumerate(VALVE_NAMES):
                    row[f"step_{step_index}_{valve_name}_weight"] = decoded.valve_weights[
                        step_index, valve_index
                    ]
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    def _run_high_fidelity_case(self, run_yaml_path: Path) -> Path:
        from packed_bed.batch import _run_case_direct, _run_case_with_timeout
        from packed_bed.cli import generate_artifacts, run_simulation
        from packed_bed.config import load_run_bundle

        run_bundle = load_run_bundle(run_yaml_path)
        artifact_fn = generate_artifacts if self.generate_artifacts else None
        if self.case_timeout_s is None:
            output_dir, _balance_errors = _run_case_direct(
                run_bundle,
                artifact_fn,
                run_simulation,
            )
        else:
            output_dir, _balance_errors = _run_case_with_timeout(
                run_bundle,
                artifact_fn,
                run_simulation,
                self.case_timeout_s,
            )
        return output_dir

    def evaluate_one(self, x: np.ndarray) -> EvaluationResult:
        genome_hash = _hash_genome(x)
        cached = self.eval_cache.get(genome_hash)
        cached_from_dry_run = cached is not None and cached.status == "dry_run"
        needs_real_run = cached_from_dry_run and not self.dry_run
        needs_failed_rerun = (
            cached is not None and self.rerun_failed and cached.status != "success"
        )
        if cached is not None and not needs_real_run and not needs_failed_rerun:
            return cached

        decoded = _decode_genome(x, self.n_steps)
        append_summary: dict[str, float] | None = None
        case_id = self._new_case_id(genome_hash)
        case_dir = self.case_root / case_id
        start = time.perf_counter()

        try:
            program, append_summary = _build_program(
                self.context_program,
                decoded,
                context_duration_s=self.context_duration_s,
                final_time_s=self.final_time_s,
            )
            run_yaml_path = _materialize_case(
                case_dir,
                case_id=case_id,
                base_run=self.base_run,
                chemistry=self.chemistry,
                solids=self.solids,
                program=program,
                final_time_s=self.final_time_s,
            )

            if self.dry_run:
                objectives = (0.0, 0.0)
                constraints = (0.0, 0.0, 0.0)
                status = "dry_run"
            else:
                output_dir = self._run_high_fidelity_case(run_yaml_path)
                objectives, constraints = _extract_objectives_and_constraints(
                    output_dir,
                    bed_length_m=self.bed_length_m,
                    start_row=self.objective_window_start,
                    window_length=self.objective_window_length,
                )
                status = "success"

            result = EvaluationResult(
                genome_hash=genome_hash,
                case_id=case_id,
                status=status,
                objectives=objectives,
                constraints=constraints,
                runtime_s=time.perf_counter() - start,
            )
        except Exception as exc:
            result = EvaluationResult(
                genome_hash=genome_hash,
                case_id=case_id,
                status="failed",
                objectives=(FAIL_OBJECTIVE, FAIL_OBJECTIVE),
                constraints=(FAIL_CONSTRAINT, FAIL_CONSTRAINT, FAIL_CONSTRAINT),
                runtime_s=time.perf_counter() - start,
                error=str(exc),
            )

        self.eval_cache[genome_hash] = result
        _write_cache(self.cache_path, self.eval_cache)
        self._record_eval(result, decoded, append_summary)
        return result

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        objectives: list[tuple[float, float]] = []
        constraints: list[tuple[float, float, float]] = []
        for row in np.asarray(x, dtype=float):
            result = self.evaluate_one(row)
            objectives.append(result.objectives)
            constraints.append(result.constraints)
        out["F"] = np.asarray(objectives, dtype=float)
        out["G"] = np.asarray(constraints, dtype=float)


def _as_2d_array(values: Any, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is not available.")
    array = np.asarray(values)
    if array.size == 0:
        return array.reshape(0, 0)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {array.shape}.")
    return array


def _population_front_indices(
    F: np.ndarray, G: np.ndarray | None, cv_tol: float = CV_TOL
) -> np.ndarray:
    keep = np.all(np.isfinite(F), axis=1)
    if G is not None and G.shape[0] == F.shape[0] and G.shape[1] > 0:
        feasible = np.maximum(G, 0.0).sum(axis=1) <= cv_tol
        if feasible.any():
            keep &= feasible
    candidate_indices = np.flatnonzero(keep)
    if candidate_indices.size == 0:
        return np.empty(0, dtype=int)
    sorter = NonDominatedSorting()
    front_local = sorter.do(F[candidate_indices], only_non_dominated_front=True)
    return candidate_indices[np.asarray(front_local, dtype=int)]


def _write_population_snapshot(path: Path, pop: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = _as_2d_array(pop.get("X"), "population X")
    F = _as_2d_array(pop.get("F"), "population F")
    try:
        G = _as_2d_array(pop.get("G"), "population G")
    except ValueError:
        G = np.empty((F.shape[0], 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, F=F, G=G)
    return X, F, G


def _write_front_csv(
    path: Path,
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray,
    problem: HighFidelityValveReactor,
) -> None:
    front_indices = _population_front_indices(F, G)
    fieldnames = [
        "front_rank",
        "population_index",
        "genome_hash",
        "case_id",
        "status",
        "syngas_objective",
        "ch4_slip_objective",
        "constraint_max_temperature",
        "constraint_outlet_o2_comb",
        "constraint_inlet_o2_comb",
        "cycle_length_s",
    ]
    for step_index in range(problem.n_steps):
        fieldnames.extend(
            [
                f"step_{step_index}_duration_s",
                f"step_{step_index}_flow_mol_s",
                f"step_{step_index}_air_weight",
                f"step_{step_index}_steam_weight",
                f"step_{step_index}_fuel_weight",
                f"step_{step_index}_offgas_weight",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for front_rank, population_index in enumerate(front_indices):
            genome_hash = _hash_genome(X[population_index])
            cached = problem.eval_cache.get(genome_hash)
            decoded = _decode_genome(X[population_index], problem.n_steps)
            row: dict[str, Any] = {
                "front_rank": front_rank,
                "population_index": int(population_index),
                "genome_hash": genome_hash,
                "case_id": "" if cached is None else cached.case_id,
                "status": "" if cached is None else cached.status,
                "syngas_objective": F[population_index, 0],
                "ch4_slip_objective": F[population_index, 1],
                "constraint_max_temperature": G[population_index, 0] if G.shape[1] > 0 else "",
                "constraint_outlet_o2_comb": G[population_index, 1] if G.shape[1] > 1 else "",
                "constraint_inlet_o2_comb": G[population_index, 2] if G.shape[1] > 2 else "",
                "cycle_length_s": decoded.cycle_length_s,
            }
            for step_index in range(problem.n_steps):
                row[f"step_{step_index}_duration_s"] = decoded.durations_s[step_index]
                row[f"step_{step_index}_flow_mol_s"] = decoded.flows_mol_s[step_index]
                for valve_index, valve_name in enumerate(VALVE_NAMES):
                    row[f"step_{step_index}_{valve_name}_weight"] = decoded.valve_weights[
                        step_index, valve_index
                    ]
            writer.writerow({field: row.get(field, "") for field in fieldnames})


class CheckpointCallback(Callback):
    def __init__(self, out_dir: Path, problem: HighFidelityValveReactor):
        super().__init__()
        self.out_dir = out_dir
        self.problem = problem

    def notify(self, algorithm: Any) -> None:
        pop = getattr(algorithm, "pop", None)
        if pop is None or len(pop) == 0:
            return
        generation = int(getattr(algorithm, "n_gen", 0))
        X, F, G = _write_population_snapshot(self.out_dir / "hf_nsga2_last_population.npz", pop)
        _write_front_csv(self.out_dir / "hf_nsga2_current_front.csv", X, F, G, self.problem)
        _write_population_snapshot(
            self.out_dir / f"checkpoints" / f"generation_{generation:04d}.npz",
            pop,
        )


def _load_resume_population(path: Path | None, n_var: int) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    with np.load(path) as data:
        if "X" not in data:
            raise ValueError(f"{path} does not contain X.")
        X = np.asarray(data["X"], dtype=float)
    if X.ndim != 2 or X.shape[1] != n_var:
        raise ValueError(f"Expected resume X shape [*, {n_var}], got {X.shape}.")
    return X


def _metadata(
    *,
    base_case_dir: Path,
    context_program_path: Path,
    out_dir: Path,
    pop_size: int,
    n_max_gen: int,
    n_steps: int,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "source": {
            "base_case_dir": str(base_case_dir),
            "context_program": str(context_program_path),
            "output_dir": str(out_dir),
        },
        "simulation": {
            "final_time_s": FINAL_TIME_S,
            "context_duration_s": C,
            "objective_window_rows_after_t0_drop": [C, C + P],
            "dry_run": dry_run,
        },
        "optimizer": {
            "algorithm": "NSGA-II",
            "population_size": pop_size,
            "n_max_gen": n_max_gen,
            "n_max_evals": pop_size * n_max_gen,
            "seed": 1,
        },
        "genome": {
            "n_steps": n_steps,
            "variables_per_step": VARIABLES_PER_STEP,
            "variable_order_per_step": [
                "duration_logit",
                "flow_logit",
                "air_logit",
                "steam_logit",
                "fuel_logit",
                "offgas_logit",
            ],
            "duration_limits_s": [MIN_STEP_S, MAX_STEP_S],
            "ramp_duration_s": RAMP_DURATION_S,
        },
        "controls": {
            "fixed_inlet_temperature_k": FIXED_INLET_T_K,
            "fixed_outlet_pressure_pa": FIXED_OUTLET_P_PA,
            "inlet_flow_limits_mol_s": list(FLOW_LIMITS_MOL_S),
            "valve_basis_species_order": list(SPECIES),
            "valve_basis": {
                valve_name: _composition_dict(VALVE_BASIS[index])
                for index, valve_name in enumerate(VALVE_NAMES)
            },
        },
        "objectives": {
            "F_order": [
                "negative_mean_outlet_H2_plus_CO_flow_times_P",
                "mean_outlet_CH4_fraction",
            ],
            "G_order": [
                "mean_positive_temperature_excess_over_1510_K",
                "mean_positive_outlet_O2_combustibles_minus_eps",
                "mean_positive_inlet_O2_combustibles_minus_eps",
            ],
            "optimization_is_minimization": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-case-dir",
        type=Path,
        default=None,
        help="Batch case directory whose base_case run/chemistry/solids should be used. Defaults to latest optim*_batch_case.",
    )
    parser.add_argument(
        "--context-program",
        type=Path,
        default=None,
        help="Program YAML to use for the first 3000 s context. Defaults to adjacent surrogates unoptimized.yaml if present.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--n-max-gen", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--case-timeout-s", type=float, default=None)
    parser.add_argument("--resume-population", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--with-artifacts", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize cases and run the optimizer plumbing without DAETools simulations.",
    )
    args = parser.parse_args(argv)
    if args.pop_size <= 0:
        parser.error("--pop-size must be positive")
    if args.n_max_gen <= 0:
        parser.error("--n-max-gen must be positive")
    if args.n_steps <= 0:
        parser.error("--n-steps must be positive")
    if args.case_timeout_s is not None and args.case_timeout_s <= 0.0:
        parser.error("--case-timeout-s must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_case_dir = (args.base_case_dir.resolve() if args.base_case_dir else _latest_optim_batch_case())
    context_program_path = _resolve_context_program(base_case_dir, args.context_program)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_run, chemistry, solids = _load_base_case(base_case_dir)
    context_program = _read_yaml_mapping(context_program_path)

    _write_metadata(
        out_dir / "run_metadata.yaml",
        _metadata(
            base_case_dir=base_case_dir,
            context_program_path=context_program_path,
            out_dir=out_dir,
            pop_size=args.pop_size,
            n_max_gen=args.n_max_gen,
            n_steps=args.n_steps,
            dry_run=bool(args.dry_run),
        ),
    )

    problem = HighFidelityValveReactor(
        base_run=base_run,
        chemistry=chemistry,
        solids=solids,
        context_program=context_program,
        out_dir=out_dir,
        n_steps=args.n_steps,
        context_duration_s=float(C),
        final_time_s=float(FINAL_TIME_S),
        objective_window_start=C,
        objective_window_length=P,
        case_timeout_s=args.case_timeout_s,
        generate_artifacts=bool(args.with_artifacts),
        dry_run=bool(args.dry_run),
        rerun_failed=bool(args.rerun_failed),
    )

    resume_path = args.resume_population
    if resume_path is None:
        candidate = out_dir / "hf_nsga2_last_population.npz"
        resume_path = candidate if candidate.exists() else None
    X0 = _load_resume_population(resume_path, problem.n_var)

    if X0 is not None:
        algorithm = NSGA2(
            pop_size=X0.shape[0],
            sampling=X0,
            crossover=SBX(eta=15, prob=0.75),
            mutation=PolynomialMutation(eta=15, prob=0.75),
            eliminate_duplicates=True,
        )
    else:
        algorithm = NSGA2(
            pop_size=args.pop_size,
            sampling=LHS(),
            crossover=SBX(eta=15, prob=0.75),
            mutation=PolynomialMutation(eta=15, prob=0.75),
            eliminate_duplicates=True,
        )

    termination = DefaultMultiObjectiveTermination(
        xtol=1.0e-8,
        cvtol=CV_TOL,
        ftol=1.0e-4,
        period=5,
        n_max_gen=args.n_max_gen,
        n_max_evals=args.pop_size * args.n_max_gen if X0 is None else X0.shape[0] * args.n_max_gen,
    )

    callback = CheckpointCallback(out_dir, problem)
    result = minimize(
        problem,
        algorithm,
        termination=termination,
        seed=args.seed,
        save_history=False,
        callback=callback,
        verbose=True,
    )

    last_pop = result.algorithm.pop
    X, F, G = _write_population_snapshot(out_dir / "hf_nsga2_last_population.npz", last_pop)
    _write_front_csv(out_dir / "hf_nsga2_final_front.csv", X, F, G, problem)
    final_cache = _read_cache(problem.cache_path)

    print(f"Base case: {base_case_dir}")
    print(f"Context program: {context_program_path}")
    print(f"Evaluations cached: {len(final_cache)}")
    print(f"Saved population: {out_dir / 'hf_nsga2_last_population.npz'}")
    print(f"Saved final front: {out_dir / 'hf_nsga2_final_front.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)
