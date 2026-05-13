#!/usr/bin/env python3
"""Generate stratified GHSV-basis operating program YAML files."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

SPECIES = ("Ar", "CH4", "CO", "CO2", "H2", "H2O", "He", "N2", "O2")
SCALAR_CHANNELS = ("inlet_flow", "inlet_temperature", "outlet_pressure")

N_STEP_BINS = ((4, 10), (10, 20), (20, 100))
HOLD_DURATION_BINS_S = ((20.0, 100.0), (100.0, 750.0), (750.0, 1500.0))
RAMP_DURATION_BINS_S = ((5.0, 20.0), (20.0, 100.0), (100.0, 1000.0))
GHSV_RANGE_H_1 = (300.0, 1500.0)
TEMPERATURE_RANGE_K = (500.0, 900.0)
PRESSURE_RANGE_PA = (1.0e5, 35.0e5)
O2_RANGE = (0.10, 0.21)
SYNC_PROBABILITY = 0.5
PURGE_DURATION_S = 20.0
PURGE_RAMP_DURATION_S = 5.0
DIRICHLET_ALPHA = 0.5

COMPOSITION_CATEGORIES = ("oxidizing", "reducing", "inert")
COMPOSITION_WEIGHTS = np.array((0.30, 0.60, 0.10), dtype=float)
OXIDIZING_BALANCE_SPECIES = ("Ar", "CO2", "H2O", "He", "N2")
REDUCING_SPECIES = ("Ar", "CH4", "CO", "CO2", "H2", "H2O", "He", "N2")
INERT_SPECIES = ("Ar", "CO2", "H2O", "He", "N2")


@dataclass(frozen=True)
class ScalarSampler:
    lower: float
    upper: float
    log_scale: bool

    def sample(self, rng: np.random.Generator) -> float:
        value = float(rng.uniform(self.lower, self.upper))
        if self.log_scale:
            value = math.exp(value)
        return round_float(value)


def round_float(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def composition_dict(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values / values.sum()
    digits = 4
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


def sample_int_from_bins(rng: np.random.Generator, bins: Sequence[tuple[int, int]]) -> int:
    lo, hi = bins[int(rng.integers(len(bins)))]
    return int(rng.integers(lo, hi + 1))


def sample_float_from_bins(rng: np.random.Generator, bins: Sequence[tuple[float, float]]) -> float:
    lo, hi = bins[int(rng.integers(len(bins)))]
    return float(rng.uniform(lo, hi))


def sample_scalar_sampler(
    rng: np.random.Generator,
    value_range: tuple[float, float],
    *,
    log_scale: bool,
) -> ScalarSampler:
    lo, hi = value_range
    lo_t = math.log(lo) if log_scale else lo
    hi_t = math.log(hi) if log_scale else hi
    mean_t = rng.uniform(lo_t, hi_t)
    span_t = rng.uniform(0.0, 1.0) * (hi_t - lo_t)
    lower = max(lo_t, mean_t - 0.5 * span_t)
    upper = min(hi_t, mean_t + 0.5 * span_t)
    return ScalarSampler(lower=lower, upper=upper, log_scale=log_scale)


def stratified_categories(count: int, rng: np.random.Generator) -> list[str]:
    raw_counts = COMPOSITION_WEIGHTS * count
    counts = np.floor(raw_counts).astype(int)
    remaining = int(count - counts.sum())
    if remaining:
        fractional = raw_counts - counts
        scores = fractional + 1.0e-9 * rng.random(len(fractional))
        for idx in np.argsort(scores)[::-1][:remaining]:
            counts[int(idx)] += 1

    categories: list[str] = []
    for category, category_count in zip(COMPOSITION_CATEGORIES, counts):
        categories.extend([category] * int(category_count))
    rng.shuffle(categories)
    return categories


def sample_subset_composition(
    rng: np.random.Generator,
    species_subset: Sequence[str],
) -> dict[str, float]:
    values = np.zeros(len(SPECIES), dtype=float)
    draw = rng.dirichlet(np.full(len(species_subset), DIRICHLET_ALPHA, dtype=float))
    for species, fraction in zip(species_subset, draw):
        values[SPECIES.index(species)] = fraction
    return composition_dict(values)


def sample_oxidizing_composition(rng: np.random.Generator) -> dict[str, float]:
    o2_lo, o2_hi = O2_RANGE
    o2_fraction = float(rng.uniform(o2_lo, o2_hi))
    values = np.zeros(len(SPECIES), dtype=float)
    values[SPECIES.index("O2")] = o2_fraction

    balance = rng.dirichlet(np.full(len(OXIDIZING_BALANCE_SPECIES), DIRICHLET_ALPHA, dtype=float))
    for species, fraction in zip(OXIDIZING_BALANCE_SPECIES, balance):
        values[SPECIES.index(species)] = (1.0 - o2_fraction) * fraction
    return composition_dict(values)


def sample_composition(category: str, rng: np.random.Generator) -> dict[str, float]:
    if category == "oxidizing":
        return sample_oxidizing_composition(rng)
    if category == "reducing":
        return sample_subset_composition(rng, REDUCING_SPECIES)
    if category == "inert":
        return sample_subset_composition(rng, INERT_SPECIES)
    raise ValueError(f"unknown composition category: {category}")


def new_program(flow_initial: float, temperature_initial: float, pressure_initial: float) -> dict[str, Any]:
    return {
        "inlet_flow": {"basis": "ghsv_per_h", "initial": flow_initial, "steps": []},
        "inlet_temperature": {"initial": temperature_initial, "steps": []},
        "outlet_pressure": {"initial": pressure_initial, "steps": []},
        "inlet_composition": {"initial": pure_n2(), "steps": []},
    }


def append_hold_to_channels(program: dict[str, Any], channels: Sequence[str], duration_s: float) -> None:
    for channel in channels:
        program[channel]["steps"].append(hold(duration_s))


def append_synchronized_purge(program: dict[str, Any]) -> None:
    append_hold_to_channels(program, SCALAR_CHANNELS, PURGE_RAMP_DURATION_S)
    program["inlet_composition"]["steps"].append(ramp(PURGE_RAMP_DURATION_S, pure_n2()))
    append_hold_to_channels(program, (*SCALAR_CHANNELS, "inlet_composition"), PURGE_DURATION_S)


def append_composition_purge(steps: list[dict[str, Any]]) -> None:
    steps.append(ramp(PURGE_RAMP_DURATION_S, pure_n2()))
    steps.append(hold(PURGE_DURATION_S))


def sampled_durations(rng: np.random.Generator) -> tuple[float, float]:
    return (
        sample_float_from_bins(rng, RAMP_DURATION_BINS_S),
        sample_float_from_bins(rng, HOLD_DURATION_BINS_S),
    )


def build_synchronized_steps(
    program: dict[str, Any],
    categories: Sequence[str],
    samplers: dict[str, ScalarSampler],
    rng: np.random.Generator,
) -> None:
    for category in categories:
        if category == "oxidizing":
            append_synchronized_purge(program)

        ramp_s, hold_s = sampled_durations(rng)
        targets = {
            "inlet_flow": samplers["inlet_flow"].sample(rng),
            "inlet_temperature": samplers["inlet_temperature"].sample(rng),
            "outlet_pressure": samplers["outlet_pressure"].sample(rng),
            "inlet_composition": sample_composition(category, rng),
        }
        for channel, target in targets.items():
            program[channel]["steps"].append(ramp(ramp_s, target))
        append_hold_to_channels(program, (*SCALAR_CHANNELS, "inlet_composition"), hold_s)

        if category == "oxidizing":
            append_synchronized_purge(program)


def build_scalar_steps(
    sampler: ScalarSampler,
    count: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for _ in range(count):
        ramp_s, hold_s = sampled_durations(rng)
        steps.append(ramp(ramp_s, sampler.sample(rng)))
        steps.append(hold(hold_s))
    return steps


def build_composition_steps(
    categories: Sequence[str],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for category in categories:
        if category == "oxidizing":
            append_composition_purge(steps)

        ramp_s, hold_s = sampled_durations(rng)
        steps.append(ramp(ramp_s, sample_composition(category, rng)))
        steps.append(hold(hold_s))

        if category == "oxidizing":
            append_composition_purge(steps)
    return steps


def sample_program(rng: np.random.Generator) -> dict[str, Any]:
    n_steps = sample_int_from_bins(rng, N_STEP_BINS)
    categories = stratified_categories(n_steps, rng)
    synchronized = bool(rng.random() < SYNC_PROBABILITY)

    samplers = {
        "inlet_flow": sample_scalar_sampler(rng, GHSV_RANGE_H_1, log_scale=True),
        "inlet_temperature": sample_scalar_sampler(rng, TEMPERATURE_RANGE_K, log_scale=False),
        "outlet_pressure": sample_scalar_sampler(rng, PRESSURE_RANGE_PA, log_scale=True),
    }
    program = new_program(
        flow_initial=samplers["inlet_flow"].sample(rng),
        temperature_initial=samplers["inlet_temperature"].sample(rng),
        pressure_initial=samplers["outlet_pressure"].sample(rng),
    )

    if synchronized:
        build_synchronized_steps(program, categories, samplers, rng)
    else:
        for channel in SCALAR_CHANNELS:
            program[channel]["steps"] = build_scalar_steps(samplers[channel], n_steps, rng)
        program["inlet_composition"]["steps"] = build_composition_steps(categories, rng)

    return program


def write_yaml(program: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(program, f, sort_keys=False, width=120)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stratified GHSV-basis program YAML files.")
    parser.add_argument("--n-programs", type=int, default=66)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--out-dir", type=Path, default=Path("generated_programs"))
    args = parser.parse_args(argv)
    if args.n_programs <= 0:
        parser.error("--n-programs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(1, np.iinfo(np.int32).max, size=args.n_programs)

    for index, seed in enumerate(seeds):
        program = sample_program(np.random.default_rng(int(seed)))
        write_yaml(program, args.out_dir / f"program_{index:04d}.yaml")

    print(f"Wrote {args.n_programs} stratified GHSV program YAML files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
