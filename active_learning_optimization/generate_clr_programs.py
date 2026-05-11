#!/usr/bin/env python3
"""Generate small GHSV-basis program YAML files.

Each generated program has the same schema as the example program files:
inlet flow, inlet temperature, outlet pressure, and inlet composition. Flow is
always written as GHSV with ``basis: ghsv_per_h``. Composition targets are plain
Dirichlet samples over the configured species list.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

SPECIES = ("Ar", "CH4", "CO", "CO2", "H2", "H2O", "He", "N2", "O2")


def positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive finite number, got {raw!r}")
    return value


def nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError(f"expected a non-negative finite number, got {raw!r}")
    return value


def range_pair(values: Sequence[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} needs exactly two values")
    lo, hi = float(values[0]), float(values[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and lo > 0.0 and hi > lo):
        raise ValueError(f"{name} must be positive and increasing")
    return lo, hi


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def round_float(value: float, digits: int = 8) -> float:
    return float(round(float(value), digits))


def composition_dict(values: np.ndarray, digits: int = 8) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values / values.sum()
    rounded = np.round(values, digits)
    rounded[rounded <= 0.5 * 10.0**-digits] = 0.0
    idx = int(np.argmax(rounded))
    rounded[idx] = round(float(rounded[idx] + 1.0 - float(rounded.sum())), digits)
    return {species: float(rounded[i]) for i, species in enumerate(SPECIES)}


def pure_n2() -> dict[str, float]:
    return {species: 1.0 if species == "N2" else 0.0 for species in SPECIES}


def hold(duration_s: float) -> dict[str, Any]:
    return {"kind": "hold", "duration_s": round_float(duration_s, 6)}


def ramp(duration_s: float, target: Any) -> dict[str, Any]:
    return {"kind": "ramp", "duration_s": round_float(duration_s, 6), "target": target}


def add_step_group(program: dict[str, Any], duration_s: float, targets: dict[str, Any]) -> None:
    for channel, target in targets.items():
        program[channel]["steps"].append(ramp(duration_s, target))


def add_hold_group(program: dict[str, Any], duration_s: float) -> None:
    if duration_s <= 0.0:
        return
    for channel in program:
        program[channel]["steps"].append(hold(duration_s))


def split_hold_times(total_s: float, count: int, rng: np.random.Generator) -> list[float]:
    if count == 1:
        return [round_float(total_s, 6)]
    weights = rng.dirichlet(np.ones(count))
    holds = [round_float(float(total_s * weight), 6) for weight in weights[:-1]]
    holds.append(round_float(total_s - sum(holds), 6))
    return holds


def sample_program(rng: np.random.Generator, args: argparse.Namespace) -> dict[str, Any]:
    total_ramp_s = args.n_ramps * args.ramp_duration_s
    hold_budget_s = args.time_horizon_s - args.initial_hold_s - total_ramp_s
    if hold_budget_s < 0.0:
        raise ValueError("--time-horizon-s is too short for the requested ramps")

    ghsv_lo, ghsv_hi = args.ghsv_range_h_1
    temp_lo, temp_hi = args.temperature_range_k
    pressure_lo, pressure_hi = args.pressure_range_pa

    program: dict[str, Any] = {
        "inlet_flow": {
            "basis": "ghsv_per_h",
            "initial": round_float(log_uniform(rng, ghsv_lo, ghsv_hi)),
            "steps": [],
        },
        "inlet_temperature": {
            "initial": round_float(rng.uniform(temp_lo, temp_hi)),
            "steps": [],
        },
        "outlet_pressure": {
            "initial": round_float(log_uniform(rng, pressure_lo, pressure_hi)),
            "steps": [],
        },
        "inlet_composition": {
            "initial": pure_n2(),
            "steps": [],
        },
    }

    add_hold_group(program, args.initial_hold_s)
    hold_times = split_hold_times(hold_budget_s, args.n_ramps, rng)
    alpha = np.full(len(SPECIES), args.dirichlet_alpha, dtype=float)

    for hold_s in hold_times:
        add_step_group(
            program,
            args.ramp_duration_s,
            {
                "inlet_flow": round_float(log_uniform(rng, ghsv_lo, ghsv_hi)),
                "inlet_temperature": round_float(rng.uniform(temp_lo, temp_hi)),
                "outlet_pressure": round_float(log_uniform(rng, pressure_lo, pressure_hi)),
                "inlet_composition": composition_dict(rng.dirichlet(alpha), args.composition_digits),
            },
        )
        add_hold_group(program, float(hold_s))

    return program


def write_yaml(program: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(program, f, sort_keys=False, width=120)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GHSV-basis program YAML files.")
    parser.add_argument("--n-programs", type=int, default=56)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--out-dir", type=Path, default=Path("generated_programs"))
    parser.add_argument("--time-horizon-s", type=positive_float, default=3600.0)
    parser.add_argument("--n-ramps", type=int, default=8)
    parser.add_argument("--initial-hold-s", type=nonnegative_float, default=10.0)
    parser.add_argument("--ramp-duration-s", type=positive_float, default=5.0)
    parser.add_argument("--ghsv-range-h-1", nargs=2, type=positive_float, default=(25.0, 900.0))
    parser.add_argument("--temperature-range-k", nargs=2, type=positive_float, default=(623.15, 973.15))
    parser.add_argument("--pressure-range-pa", nargs=2, type=positive_float, default=(1.0e5, 3.5e6))
    parser.add_argument("--dirichlet-alpha", type=positive_float, default=0.5)
    parser.add_argument("--composition-digits", type=int, default=8)

    args = parser.parse_args(argv)
    if args.n_programs <= 0: 
        parser.error("--n-programs must be positive")
    if args.n_ramps <= 0:
        parser.error("--n-ramps must be positive")
    if args.composition_digits < 1:
        parser.error("--composition-digits must be at least 1")

    try:
        args.ghsv_range_h_1 = range_pair(args.ghsv_range_h_1, "--ghsv-range-h-1")
        args.temperature_range_k = range_pair(args.temperature_range_k, "--temperature-range-k")
        args.pressure_range_pa = range_pair(args.pressure_range_pa, "--pressure-range-pa")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(1, np.iinfo(np.int32).max, size=args.n_programs)

    for index, seed in enumerate(seeds):
        program = sample_program(np.random.default_rng(int(seed)), args)
        write_yaml(program, args.out_dir / f"program_{index:04d}.yaml")

    print(f"Wrote {args.n_programs} GHSV program YAML files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
