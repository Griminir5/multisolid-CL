#!/usr/bin/env python3
"""
Generate YAML operating programs for dynamic packed-bed chemical-looping simulations.

The generator is deliberately not just a random Dirichlet sampler.  It mixes several
trajectory families:

  1. canonical_cycle: oxidizer -> purge -> reducer -> purge, with jittered timing;
  2. async_random: independently sampled composition/flow/temperature/pressure events;
  3. composition_sweep: frequent gas-composition changes, slower scalar controls;
  4. pulse_train: short flow/pressure/composition perturbations for dynamic excitation;
  5. boundary_sweep: sparse visits to high/low scalar-control bounds and gas extremes.

Flow can be sampled on either a molar-flow or GHSV basis.  The output can likewise be
written as mol/s or as GHSV h^-1.  If converting between GHSV and mol/s, provide either
bed_volume_m3 or bed_length_m + bed_radius_m, or a geometry CSV.

Example, one molar-flow YAML set sampled on GHSV basis for a single geometry:

  python generate_clr_programs.py \
      --n-programs 56 \
      --sample-flow-on ghsv \
      --write-flow-as molar \
      --bed-length-m 1.0 --bed-radius-m 0.05 \
      --out-dir generated_programs

Example, geometry-specific molar-flow YAMLs for 10 geometries:

  python generate_clr_programs.py \
      --n-programs 56 \
      --sample-flow-on ghsv \
      --write-flow-as molar \
      --geometry-csv geometries.csv \
      --out-dir generated_programs

Expected geometry CSV columns:

  geometry_id,bed_volume_m3

or:

  geometry_id,bed_length_m,bed_radius_m

The YAML schema emitted is compatible with the attached program.yaml style:

  inlet_flow:          initial + steps of hold/ramp
  inlet_temperature:   initial + steps of hold/ramp
  outlet_pressure:     initial + steps of hold/ramp
  inlet_composition:   initial species dict + steps of hold/ramp
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

R_GAS = 8.31446261815324  # J mol^-1 K^-1

SPECIES: Tuple[str, ...] = ("Ar", "CH4", "CO", "CO2", "H2", "H2O", "He", "N2", "O2")
IDX = {name: i for i, name in enumerate(SPECIES)}
COMBUSTIBLES = ("CH4", "CO", "H2")
OXIDIZER = "O2"


@dataclass(frozen=True)
class Event:
    start_s: float
    ramp_s: float
    target: Any


@dataclass(frozen=True)
class Geometry:
    geometry_id: str
    bed_volume_m3: Optional[float] = None


@dataclass(frozen=True)
class ProgramMeta:
    program_id: str
    family: str
    seed: int
    regimes: Tuple[str, ...]
    flow_basis_values: Tuple[float, ...]
    temperature_values_k: Tuple[float, ...]
    pressure_values_pa: Tuple[float, ...]


def _positive_float(value: str) -> float:
    x = float(value)
    if not math.isfinite(x) or x <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive finite float, got {value!r}")
    return x


def _nonnegative_float(value: str) -> float:
    x = float(value)
    if not math.isfinite(x) or x < 0.0:
        raise argparse.ArgumentTypeError(f"expected a non-negative finite float, got {value!r}")
    return x


def _pair_of_floats(values: Sequence[float], *, name: str) -> Tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} needs exactly two values")
    lo, hi = float(values[0]), float(values[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and lo > 0 and hi > lo):
        raise ValueError(f"{name} must be positive and increasing, got {values}")
    return lo, hi


def normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    y[y < 0.0] = 0.0
    total = float(y.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("cannot normalize an empty or invalid composition")
    return y / total


def zeros() -> np.ndarray:
    return np.zeros(len(SPECIES), dtype=float)


def as_composition_dict(y: np.ndarray, ndigits: int = 8) -> Dict[str, float]:
    """Round a composition while preserving an exact unit sum after rounding."""
    y = normalize(y)
    yr = np.round(y, ndigits)
    # Avoid harmless but ugly 1e-08 entries created by rounding drift.
    yr[yr <= 5.0 * 10.0 ** (-ndigits)] = 0.0
    drift = 1.0 - float(yr.sum())
    if abs(drift) > 0.0:
        idx = int(np.argmax(yr))
        yr[idx] = max(0.0, yr[idx] + drift)
    # One more normalization guard against pathological rounding after clipping.
    yr = normalize(yr)
    return {sp: float(yr[i]) for i, sp in enumerate(SPECIES)}


def composition_from_dict(d: Mapping[str, float]) -> np.ndarray:
    y = zeros()
    for sp, value in d.items():
        if sp not in IDX:
            raise ValueError(f"unknown species {sp!r}; expected one of {SPECIES}")
        y[IDX[sp]] = float(value)
    return normalize(y)


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def log_interp(lo: float, hi: float, q: float) -> float:
    return float(math.exp(math.log(lo) + float(q) * (math.log(hi) - math.log(lo))))


def lin_interp(lo: float, hi: float, q: float) -> float:
    return float(lo + float(q) * (hi - lo))


def clipped(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def steal_mass(y: np.ndarray, species: str, amount: float) -> np.ndarray:
    """Add amount to `species` by removing it from the currently largest other component."""
    y = normalize(y)
    amount = max(0.0, min(float(amount), 0.2))
    if amount == 0.0:
        return y
    j = IDX[species]
    candidates = y.copy()
    candidates[j] = -1.0
    donor = int(np.argmax(candidates))
    take = min(amount, y[donor] * 0.95)
    y[donor] -= take
    y[j] += take
    return normalize(y)


def jitter_nonzero(rng: np.random.Generator, y: np.ndarray, sigma: float = 0.12) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    nz = y > 1e-14
    if int(nz.sum()) <= 1:
        return normalize(y)
    y[nz] *= np.exp(rng.normal(0.0, sigma, int(nz.sum())))
    return normalize(y)


def fill_balance(
    rng: np.random.Generator,
    y: np.ndarray,
    species: Sequence[str],
    alpha: Optional[Sequence[float]] = None,
) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    current = float(y.sum())
    if current > 1.0 + 1e-12:
        return normalize(y)
    rem = max(0.0, 1.0 - current)
    if rem <= 1e-14:
        return normalize(y)
    if alpha is None:
        alpha = np.ones(len(species), dtype=float)
    weights = rng.dirichlet(np.asarray(alpha, dtype=float))
    for sp, w in zip(species, weights):
        y[IDX[sp]] += rem * float(w)
    return normalize(y)


def maybe_add_tracer(rng: np.random.Generator, y: np.ndarray) -> np.ndarray:
    if rng.random() < 0.20:
        tracer = "Ar" if rng.random() < 0.65 else "He"
        return steal_mass(y, tracer, rng.uniform(0.001, 0.025))
    return y


def air_composition() -> np.ndarray:
    y = zeros()
    y[IDX["Ar"]] = 0.0093
    y[IDX["CO2"]] = 0.0004
    y[IDX["N2"]] = 0.7808
    y[IDX["O2"]] = 0.2095
    return normalize(y)


def sample_oxidizer(rng: np.random.Generator) -> Tuple[np.ndarray, str]:
    choice = rng.choice(
        ["air", "lean_air", "oxygen_rich", "steam_air", "co2_o2"],
        p=[0.35, 0.25, 0.12, 0.16, 0.12],
    )
    y = zeros()
    if choice == "air":
        y = jitter_nonzero(rng, air_composition(), sigma=0.035)
    elif choice == "lean_air":
        o2 = log_uniform(rng, 0.015, 0.18)
        ar = rng.uniform(0.002, 0.014)
        co2 = rng.uniform(0.0, 0.015)
        y[IDX["O2"]] = o2
        y[IDX["Ar"]] = ar
        y[IDX["CO2"]] = co2
        y[IDX["N2"]] = max(0.0, 1.0 - y.sum())
        y = normalize(y)
    elif choice == "oxygen_rich":
        y[IDX["O2"]] = rng.uniform(0.22, 0.55)
        y = fill_balance(rng, y, ["N2", "Ar", "CO2"], alpha=[9.0, 0.4, 0.8])
    elif choice == "steam_air":
        y[IDX["O2"]] = rng.uniform(0.015, 0.18)
        y[IDX["H2O"]] = rng.uniform(0.05, min(0.70, 0.95 - y[IDX["O2"]]))
        y = fill_balance(rng, y, ["N2", "Ar", "CO2"], alpha=[8.0, 0.2, 0.6])
    elif choice == "co2_o2":
        y[IDX["O2"]] = rng.uniform(0.015, 0.35)
        y[IDX["CO2"]] = rng.uniform(0.10, min(0.85, 0.95 - y[IDX["O2"]]))
        y = fill_balance(rng, y, ["N2", "Ar"], alpha=[9.0, 0.3])
    # Avoid oxidizer + fuel feeds except in the explicit mixed regime.
    for sp in COMBUSTIBLES:
        y[IDX[sp]] = 0.0
    return normalize(y), f"oxidizer:{choice}"


def sample_purge(rng: np.random.Generator) -> Tuple[np.ndarray, str]:
    choice = rng.choice(
        ["n2", "steam", "co2", "n2_steam", "n2_co2", "ar_he"],
        p=[0.18, 0.36, 0.13, 0.08, 0.12, 0.13],
    )
    y = zeros()
    if choice == "n2":
        y[IDX["N2"]] = 1.0
    elif choice == "steam":
        y[IDX["H2O"]] = rng.uniform(0.60, 1.0)
        y = fill_balance(rng, y, ["N2", "Ar"], alpha=[8.0, 0.3])
    elif choice == "co2":
        y[IDX["CO2"]] = rng.uniform(0.60, 1.0)
        y = fill_balance(rng, y, ["N2", "Ar"], alpha=[8.0, 0.3])
    elif choice == "n2_steam":
        y[IDX["H2O"]] = rng.uniform(0.05, 0.80)
        y = fill_balance(rng, y, ["N2", "Ar"], alpha=[10.0, 0.3])
    elif choice == "n2_co2":
        y[IDX["CO2"]] = rng.uniform(0.05, 0.80)
        y = fill_balance(rng, y, ["N2", "Ar"], alpha=[10.0, 0.3])
    elif choice == "ar_he":
        y[IDX["Ar"]] = rng.uniform(0.05, 0.60)
        y[IDX["He"]] = rng.uniform(0.00, 0.20)
        y = fill_balance(rng, y, ["N2"], alpha=[1.0])
    for sp in (*COMBUSTIBLES, "O2"):
        y[IDX[sp]] = 0.0
    return normalize(y), f"purge:{choice}"


def sample_reducer(rng: np.random.Generator) -> Tuple[np.ndarray, str]:
    choice = rng.choice(
        ["h2_rich", "co_rich", "syngas", "ch4_steam", "wet_fuel_mix", "dry_fuel_mix"],
        p=[0.17, 0.13, 0.26, 0.22, 0.16, 0.06],
    )
    y = zeros()
    if choice == "h2_rich":
        y[IDX["H2"]] = rng.uniform(0.05, 0.75)
        if rng.random() < 0.55:
            y[IDX["H2O"]] = rng.uniform(0.00, min(0.55, 0.95 - y.sum()))
        y = fill_balance(rng, y, ["N2", "CO2", "H2O"], alpha=[6.0, 1.2, 2.0])
    elif choice == "co_rich":
        y[IDX["CO"]] = rng.uniform(0.05, 0.70)
        if rng.random() < 0.65:
            y[IDX["CO2"]] = rng.uniform(0.00, min(0.55, 0.95 - y.sum()))
        y = fill_balance(rng, y, ["N2", "CO2", "H2O"], alpha=[6.0, 2.0, 1.0])
    elif choice == "syngas":
        fuel = rng.uniform(0.05, 0.85)
        h2_fraction = rng.beta(1.4, 1.4)
        y[IDX["H2"]] = fuel * h2_fraction
        y[IDX["CO"]] = fuel * (1.0 - h2_fraction)
        product = rng.uniform(0.0, min(0.65, 0.95 - y.sum()))
        split = rng.beta(1.2, 1.2)
        y[IDX["H2O"]] = product * split
        y[IDX["CO2"]] = product * (1.0 - split)
        y = fill_balance(rng, y, ["N2", "Ar"], alpha=[10.0, 0.3])
    elif choice == "ch4_steam":
        y[IDX["CH4"]] = rng.uniform(0.015, 0.45)
        # Steam-to-carbon coverage from lean to strongly wet feeds.
        y[IDX["H2O"]] = rng.uniform(0.05, min(0.85, 0.98 - y.sum()))
        if rng.random() < 0.45:
            y[IDX["CO2"]] = rng.uniform(0.0, min(0.30, 0.98 - y.sum()))
        if rng.random() < 0.25:
            y[IDX["H2"]] = rng.uniform(0.0, min(0.20, 0.98 - y.sum()))
        y = fill_balance(rng, y, ["N2", "CO2"], alpha=[7.0, 1.0])
    elif choice == "wet_fuel_mix":
        # Similar spirit to a reformate/recycle gas, but with broader coverage.
        active = ["CH4", "CO", "CO2", "H2", "H2O", "N2"]
        alpha = np.array([1.0, 0.9, 1.3, 1.1, 2.0, 0.8]) * rng.uniform(0.7, 2.5)
        vals = rng.dirichlet(alpha)
        for sp, val in zip(active, vals):
            y[IDX[sp]] = val
    elif choice == "dry_fuel_mix":
        active = ["CH4", "CO", "H2", "N2", "CO2"]
        alpha = np.array([1.2, 1.2, 1.2, 0.9, 0.4]) * rng.uniform(0.7, 2.0)
        vals = rng.dirichlet(alpha)
        for sp, val in zip(active, vals):
            y[IDX[sp]] = val
    # Reducing feeds should not carry meaningful oxygen.
    if rng.random() < 0.12:
        y = steal_mass(y, "O2", rng.uniform(1e-5, 0.0025))
    else:
        y[IDX["O2"]] = 0.0
    y = maybe_add_tracer(rng, y)
    return normalize(y), f"reducer:{choice}"


def sample_mixed_low_o2(rng: np.random.Generator) -> Tuple[np.ndarray, str]:
    """Rare edge case: small O2 with fuel, to cover transition/leakage trajectories."""
    y = zeros()
    fuel_total = rng.uniform(0.015, 0.22)
    fuel_split = rng.dirichlet([0.8, 1.0, 1.0])
    for sp, val in zip(["CH4", "CO", "H2"], fuel_split):
        y[IDX[sp]] = fuel_total * float(val)
    y[IDX["O2"]] = rng.uniform(0.002, 0.025)
    product = rng.uniform(0.00, min(0.55, 0.92 - y.sum()))
    split = rng.beta(1.2, 1.2)
    y[IDX["CO2"]] = product * split
    y[IDX["H2O"]] = product * (1.0 - split)
    y = fill_balance(rng, y, ["N2", "Ar"], alpha=[10.0, 0.3])
    return normalize(y), "mixed_low_o2:fuel_oxidizer_transition"


def sample_composition(rng: np.random.Generator, regime: str) -> Tuple[np.ndarray, str]:
    if regime == "oxidizer":
        return sample_oxidizer(rng)
    if regime == "purge":
        return sample_purge(rng)
    if regime == "reducer":
        return sample_reducer(rng)
    if regime == "mixed_low_o2":
        return sample_mixed_low_o2(rng)
    raise ValueError(f"unknown composition regime {regime!r}")


def pure_n2_dict() -> Dict[str, float]:
    y = zeros()
    y[IDX["N2"]] = 1.0
    return as_composition_dict(y)


def events_to_steps(initial: Any, events: Sequence[Event], horizon_s: float) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    current = 0.0
    last_target = initial
    eps = 1e-9
    for ev in sorted(events, key=lambda e: e.start_s):
        start = clipped(ev.start_s, 0.0, horizon_s)
        if start < current:
            start = current
        if start >= horizon_s - eps:
            break
        if start > current + eps:
            steps.append({"kind": "hold", "duration_s": round(start - current, 6)})
            current = start
        ramp = clipped(ev.ramp_s, 0.0, horizon_s - current)
        if ramp > eps:
            steps.append({"kind": "ramp", "duration_s": round(ramp, 6), "target": ev.target})
            current += ramp
            last_target = ev.target
        if current >= horizon_s - eps:
            break
    if horizon_s > current + eps:
        steps.append({"kind": "hold", "duration_s": round(horizon_s - current, 6)})
    return steps


def random_times(
    rng: np.random.Generator,
    horizon_s: float,
    n: int,
    *,
    keep_away_from_end_s: float = 30.0,
) -> np.ndarray:
    if n <= 0:
        return np.empty(0)
    hi = max(0.0, horizon_s - keep_away_from_end_s)
    if n == 1:
        return np.array([rng.uniform(0.0, hi)])
    # Stratified random times reduce accidental clustering while retaining asynchrony.
    edges = np.linspace(0.0, hi, n + 1)
    return np.array([rng.uniform(edges[i], edges[i + 1]) for i in range(n)])


def sample_ramp_s(rng: np.random.Generator, channel: str, family: str) -> float:
    if channel == "temperature":
        if family == "pulse_train":
            return log_uniform(rng, 10.0, 180.0)
        if family == "boundary_sweep":
            return log_uniform(rng, 20.0, 500.0)
        return log_uniform(rng, 5.0, 240.0)
    if channel == "pressure":
        return log_uniform(rng, 2.0, 90.0 if family != "boundary_sweep" else 180.0)
    if channel == "flow":
        return log_uniform(rng, 1.0, 75.0 if family != "boundary_sweep" else 140.0)
    if channel == "composition":
        return log_uniform(rng, 1.0, 60.0 if family != "boundary_sweep" else 120.0)
    return log_uniform(rng, 1.0, 60.0)


def sample_flow_value(
    rng: np.random.Generator,
    base: float,
    bounds: Tuple[float, float],
    family: str,
    amplitude: float,
) -> float:
    lo, hi = bounds
    if family == "boundary_sweep" and rng.random() < 0.65:
        return log_interp(lo, hi, rng.choice([rng.uniform(0.0, 0.10), rng.uniform(0.90, 1.0)]))
    if family == "pulse_train":
        x = base * math.exp(rng.normal(0.0, 0.35 * amplitude))
    else:
        if rng.random() < 0.25:
            x = log_uniform(rng, lo, hi)
        else:
            x = base * math.exp(rng.normal(0.0, 0.60 * amplitude))
    return clipped(x, lo, hi)


def sample_temperature_value(
    rng: np.random.Generator,
    base: float,
    bounds: Tuple[float, float],
    family: str,
    amplitude: float,
) -> float:
    lo, hi = bounds
    if family == "boundary_sweep" and rng.random() < 0.65:
        return lin_interp(lo, hi, rng.choice([rng.uniform(0.0, 0.12), rng.uniform(0.88, 1.0)]))
    if rng.random() < 0.25:
        return rng.uniform(lo, hi)
    span = hi - lo
    x = base + rng.normal(0.0, 0.16 * amplitude * span)
    return clipped(x, lo, hi)


def sample_pressure_value(
    rng: np.random.Generator,
    base: float,
    bounds: Tuple[float, float],
    family: str,
    amplitude: float,
) -> float:
    lo, hi = bounds
    if family == "boundary_sweep" and rng.random() < 0.65:
        return log_interp(lo, hi, rng.choice([rng.uniform(0.0, 0.12), rng.uniform(0.88, 1.0)]))
    if rng.random() < 0.20:
        x = log_uniform(rng, lo, hi)
    else:
        x = base * math.exp(rng.normal(0.0, 0.22 * amplitude))
    return clipped(x, lo, hi)


def scalar_events(
    rng: np.random.Generator,
    *,
    channel: str,
    family: str,
    horizon_s: float,
    n_events: int,
    base_value: float,
    bounds: Tuple[float, float],
    amplitude: float,
    phase_times: Optional[Sequence[float]] = None,
) -> List[Event]:
    if phase_times is not None and len(phase_times) > 0:
        times: List[float] = []
        for t in phase_times:
            if rng.random() < {"flow": 0.70, "temperature": 0.55, "pressure": 0.38}.get(channel, 0.5):
                times.append(clipped(float(t) + rng.normal(0.0, 45.0), 0.0, max(0.0, horizon_s - 20.0)))
        # Add some independent perturbations.
        extra_n = max(0, n_events - len(times))
        times.extend(random_times(rng, horizon_s, extra_n).tolist())
        times = sorted(times)[: max(1, n_events)]
    else:
        times = random_times(rng, horizon_s, n_events).tolist()

    events: List[Event] = []
    for t in times:
        if channel == "flow":
            value = sample_flow_value(rng, base_value, bounds, family, amplitude)
        elif channel == "temperature":
            value = sample_temperature_value(rng, base_value, bounds, family, amplitude)
        elif channel == "pressure":
            value = sample_pressure_value(rng, base_value, bounds, family, amplitude)
        else:
            raise ValueError(channel)
        events.append(Event(start_s=float(t), ramp_s=sample_ramp_s(rng, channel, family), target=round(float(value), 8)))
    return events


def canonical_phase_times(rng: np.random.Generator, horizon_s: float) -> Tuple[List[float], List[str]]:
    seq = ["oxidizer", "purge", "reducer", "purge"]
    times: List[float] = []
    regimes: List[str] = []
    t = 0.0
    k = 0
    while t < horizon_s - 60.0:
        regime = seq[k % len(seq)]
        times.append(t)
        regimes.append(regime)
        if regime == "oxidizer":
            dt = log_uniform(rng, 120.0, 900.0)
        elif regime == "reducer":
            dt = log_uniform(rng, 180.0, 1400.0)
        else:
            dt = log_uniform(rng, 45.0, 450.0)
        t += dt
        k += 1
    return times, regimes


def composition_events(
    rng: np.random.Generator,
    *,
    family: str,
    horizon_s: float,
    event_density: float,
    phase_times: Optional[Sequence[float]] = None,
    phase_regimes: Optional[Sequence[str]] = None,
) -> Tuple[List[Event], Tuple[str, ...]]:
    events: List[Event] = []
    labels: List[str] = []

    if family == "canonical_cycle" and phase_times is not None and phase_regimes is not None:
        for t, regime in zip(phase_times, phase_regimes):
            y, label = sample_composition(rng, regime)
            events.append(Event(start_s=float(t), ramp_s=sample_ramp_s(rng, "composition", family), target=as_composition_dict(y)))
            labels.append(label)
        return events, tuple(labels)

    if family == "composition_sweep":
        n_events = int(round(rng.integers(8, 18) * event_density))
        regime_probs = [0.22, 0.30, 0.43, 0.05]  # oxidizer, purge, reducer, mixed
    elif family == "pulse_train":
        n_events = int(round(rng.integers(7, 16) * event_density))
        regime_probs = [0.24, 0.38, 0.34, 0.04]
    elif family == "boundary_sweep":
        n_events = int(round(rng.integers(5, 10) * event_density))
        regime_probs = [0.28, 0.22, 0.45, 0.05]
    else:  # async_random
        n_events = int(round(rng.integers(5, 14) * event_density))
        regime_probs = [0.25, 0.30, 0.40, 0.05]

    regimes = ["oxidizer", "purge", "reducer", "mixed_low_o2"]
    times = random_times(rng, horizon_s, max(1, n_events), keep_away_from_end_s=45.0)
    for t in times:
        regime = str(rng.choice(regimes, p=regime_probs))
        y, label = sample_composition(rng, regime)
        events.append(Event(start_s=float(t), ramp_s=sample_ramp_s(rng, "composition", family), target=as_composition_dict(y)))
        labels.append(label)
    return events, tuple(labels)


def latin_hypercube(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    h = np.empty((n, dim), dtype=float)
    for j in range(dim):
        order = rng.permutation(n)
        h[:, j] = (order + rng.random(n)) / n
    return h


def choose_families(n: int, requested: Sequence[str], rng: np.random.Generator) -> List[str]:
    if not requested:
        requested = ["canonical_cycle", "async_random", "composition_sweep", "pulse_train", "boundary_sweep"]
    valid = {"canonical_cycle", "async_random", "composition_sweep", "pulse_train", "boundary_sweep"}
    unknown = [f for f in requested if f not in valid]
    if unknown:
        raise ValueError(f"unknown families {unknown}; valid families are {sorted(valid)}")
    families = [requested[i % len(requested)] for i in range(n)]
    rng.shuffle(families)
    return families


def generate_program_in_sample_basis(
    *,
    program_index: int,
    seed: int,
    family: str,
    lhs_row: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], ProgramMeta]:
    rng = np.random.default_rng(seed)
    horizon_s = float(args.time_horizon_s)
    flow_bounds = tuple(args.ghsv_range_h_1 if args.sample_flow_on == "ghsv" else args.flow_range_mol_s)
    temp_bounds = tuple(args.temperature_range_k)
    pressure_bounds = tuple(args.pressure_range_pa)

    base_flow = log_interp(flow_bounds[0], flow_bounds[1], lhs_row[0])
    base_temp = lin_interp(temp_bounds[0], temp_bounds[1], lhs_row[1])
    base_pressure = log_interp(pressure_bounds[0], pressure_bounds[1], lhs_row[2])
    amplitude = lin_interp(0.45, 1.45, lhs_row[3])
    event_density = lin_interp(0.75, 1.35, lhs_row[4])

    initial_flow = round(base_flow, 8)
    initial_temp = round(base_temp, 8)
    initial_pressure = round(base_pressure, 8)
    initial_comp = pure_n2_dict()

    phase_times: Optional[List[float]] = None
    phase_regimes: Optional[List[str]] = None
    if family == "canonical_cycle":
        phase_times, phase_regimes = canonical_phase_times(rng, horizon_s)

    comp_events, comp_regime_labels = composition_events(
        rng,
        family=family,
        horizon_s=horizon_s,
        event_density=event_density,
        phase_times=phase_times,
        phase_regimes=phase_regimes,
    )

    if family == "composition_sweep":
        n_flow = int(round(rng.integers(3, 8) * event_density))
        n_temp = int(round(rng.integers(2, 6) * event_density))
        n_press = int(round(rng.integers(2, 6) * event_density))
    elif family == "pulse_train":
        n_flow = int(round(rng.integers(9, 22) * event_density))
        n_temp = int(round(rng.integers(3, 8) * event_density))
        n_press = int(round(rng.integers(8, 18) * event_density))
    elif family == "boundary_sweep":
        n_flow = int(round(rng.integers(4, 8) * event_density))
        n_temp = int(round(rng.integers(4, 8) * event_density))
        n_press = int(round(rng.integers(4, 8) * event_density))
    elif family == "canonical_cycle":
        phase_count = 0 if phase_times is None else len(phase_times)
        n_flow = max(3, int(round(0.9 * phase_count)))
        n_temp = max(2, int(round(0.6 * phase_count)))
        n_press = max(2, int(round(0.45 * phase_count)))
    else:  # async_random
        n_flow = int(round(rng.integers(4, 12) * event_density))
        n_temp = int(round(rng.integers(3, 9) * event_density))
        n_press = int(round(rng.integers(3, 9) * event_density))

    flow_events = scalar_events(
        rng,
        channel="flow",
        family=family,
        horizon_s=horizon_s,
        n_events=n_flow,
        base_value=base_flow,
        bounds=flow_bounds,
        amplitude=amplitude,
        phase_times=phase_times if family == "canonical_cycle" else None,
    )
    temp_events = scalar_events(
        rng,
        channel="temperature",
        family=family,
        horizon_s=horizon_s,
        n_events=n_temp,
        base_value=base_temp,
        bounds=temp_bounds,
        amplitude=amplitude,
        phase_times=phase_times if family == "canonical_cycle" else None,
    )
    pressure_events = scalar_events(
        rng,
        channel="pressure",
        family=family,
        horizon_s=horizon_s,
        n_events=n_press,
        base_value=base_pressure,
        bounds=pressure_bounds,
        amplitude=amplitude,
        phase_times=phase_times if family == "canonical_cycle" else None,
    )

    program = {
        "inlet_flow": {
            "initial": initial_flow,
            "steps": events_to_steps(initial_flow, flow_events, horizon_s),
        },
        "inlet_temperature": {
            "initial": initial_temp,
            "steps": events_to_steps(initial_temp, temp_events, horizon_s),
        },
        "outlet_pressure": {
            "initial": initial_pressure,
            "steps": events_to_steps(initial_pressure, pressure_events, horizon_s),
        },
        "inlet_composition": {
            "initial": initial_comp,
            "steps": events_to_steps(initial_comp, comp_events, horizon_s),
        },
    }

    flow_values = collect_scalar_values(program["inlet_flow"])
    temp_values = collect_scalar_values(program["inlet_temperature"])
    pressure_values = collect_scalar_values(program["outlet_pressure"])
    meta = ProgramMeta(
        program_id=f"program_{program_index:04d}",
        family=family,
        seed=seed,
        regimes=comp_regime_labels,
        flow_basis_values=tuple(float(x) for x in flow_values),
        temperature_values_k=tuple(float(x) for x in temp_values),
        pressure_values_pa=tuple(float(x) for x in pressure_values),
    )
    return program, meta


def collect_scalar_values(channel: Mapping[str, Any]) -> List[float]:
    values = [float(channel["initial"])]
    for step in channel.get("steps", []):
        if step.get("kind") == "ramp" and "target" in step:
            values.append(float(step["target"]))
    return values


def collect_composition_labels(meta: ProgramMeta, max_labels: int = 40) -> str:
    labels = list(meta.regimes)
    if len(labels) > max_labels:
        return "|".join(labels[:max_labels]) + f"|...(+{len(labels) - max_labels})"
    return "|".join(labels)


def bed_volume_from_length_radius(length_m: float, radius_m: float) -> float:
    return math.pi * float(radius_m) ** 2 * float(length_m)


def read_geometries(args: argparse.Namespace) -> List[Geometry]:
    if args.geometry_csv:
        geoms: List[Geometry] = []
        with open(args.geometry_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                gid = str(row.get("geometry_id") or row.get("id") or f"geom_{row_idx:03d}")
                if row.get("bed_volume_m3") not in (None, ""):
                    volume = float(row["bed_volume_m3"])
                else:
                    try:
                        length = float(row["bed_length_m"])
                        radius = float(row["bed_radius_m"])
                    except KeyError as exc:
                        raise ValueError(
                            "geometry CSV needs bed_volume_m3 or both bed_length_m and bed_radius_m"
                        ) from exc
                    volume = bed_volume_from_length_radius(length, radius)
                if volume <= 0.0 or not math.isfinite(volume):
                    raise ValueError(f"invalid bed volume for geometry {gid!r}: {volume}")
                geoms.append(Geometry(gid, volume))
        if not geoms:
            raise ValueError("geometry CSV did not contain any rows")
        return geoms

    if args.bed_volume_m3 is not None:
        return [Geometry("default", float(args.bed_volume_m3))]
    if args.bed_length_m is not None and args.bed_radius_m is not None:
        return [Geometry("default", bed_volume_from_length_radius(args.bed_length_m, args.bed_radius_m))]
    return [Geometry("default", None)]


def ghsv_to_molar_factor(geometry: Geometry, p_ref_pa: float, t_ref_k: float) -> float:
    if geometry.bed_volume_m3 is None:
        raise ValueError(
            "GHSV-to-molar-flow conversion needs bed volume. Provide --bed-volume-m3, "
            "--bed-length-m plus --bed-radius-m, or --geometry-csv."
        )
    # GHSV [h^-1] * V_bed [m^3] gives volumetric flow [m^3/h] at reference conditions.
    # Convert m^3/h to mol/s using ideal-gas c = P/(R T).
    return geometry.bed_volume_m3 * p_ref_pa / (R_GAS * t_ref_k) / 3600.0


def flow_value_to_molar(
    value: float,
    basis: str,
    geometry: Geometry,
    p_ref_pa: float,
    t_ref_k: float,
) -> float:
    if basis == "molar":
        return float(value)
    if basis == "ghsv":
        return float(value) * ghsv_to_molar_factor(geometry, p_ref_pa, t_ref_k)
    raise ValueError(basis)


def molar_to_flow_value(
    value_mol_s: float,
    basis: str,
    geometry: Geometry,
    p_ref_pa: float,
    t_ref_k: float,
) -> float:
    if basis == "molar":
        return float(value_mol_s)
    if basis == "ghsv":
        return float(value_mol_s) / ghsv_to_molar_factor(geometry, p_ref_pa, t_ref_k)
    raise ValueError(basis)


def convert_flow_channel(
    channel: Mapping[str, Any],
    *,
    sample_basis: str,
    output_basis: str,
    geometry: Geometry,
    p_ref_pa: float,
    t_ref_k: float,
) -> Dict[str, Any]:
    converted = deepcopy(dict(channel))

    def convert(v: float) -> float:
        molar = flow_value_to_molar(v, sample_basis, geometry, p_ref_pa, t_ref_k)
        out = molar_to_flow_value(molar, output_basis, geometry, p_ref_pa, t_ref_k)
        return round(float(out), 8)

    converted["initial"] = convert(float(converted["initial"]))
    for step in converted.get("steps", []):
        if step.get("kind") == "ramp" and "target" in step:
            step["target"] = convert(float(step["target"]))
    return converted


def convert_program_flow(
    program: Mapping[str, Any],
    *,
    sample_basis: str,
    output_basis: str,
    geometry: Geometry,
    p_ref_pa: float,
    t_ref_k: float,
) -> Dict[str, Any]:
    out = deepcopy(dict(program))
    out["inlet_flow"] = convert_flow_channel(
        out["inlet_flow"],
        sample_basis=sample_basis,
        output_basis=output_basis,
        geometry=geometry,
        p_ref_pa=p_ref_pa,
        t_ref_k=t_ref_k,
    )
    return out


def ensure_flow_conversion_is_possible(args: argparse.Namespace, geoms: Sequence[Geometry]) -> None:
    needs_geometry = args.sample_flow_on != args.write_flow_as
    if needs_geometry:
        missing = [g.geometry_id for g in geoms if g.bed_volume_m3 is None]
        if missing:
            raise ValueError(
                "Converting between GHSV and molar flow needs geometry, but these geometries "
                f"have no bed volume: {missing}."
            )


def dump_yaml(program: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(program, f, sort_keys=False, default_flow_style=False, width=120)


def manifest_row(
    *,
    meta: ProgramMeta,
    program: Mapping[str, Any],
    output_path: Path,
    output_dir: Path,
    geometry: Geometry,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    flow_output_values = collect_scalar_values(program["inlet_flow"])
    temp_values = collect_scalar_values(program["inlet_temperature"])
    pressure_values = collect_scalar_values(program["outlet_pressure"])
    row: Dict[str, Any] = {
        "program_id": meta.program_id,
        "family": meta.family,
        "seed": meta.seed,
        "geometry_id": geometry.geometry_id,
        "bed_volume_m3": "" if geometry.bed_volume_m3 is None else geometry.bed_volume_m3,
        "path": str(output_path.relative_to(output_dir)),
        "time_horizon_s": args.time_horizon_s,
        "sample_flow_on": args.sample_flow_on,
        "write_flow_as": args.write_flow_as,
        "flow_ref_p_pa": args.ghsv_ref_p_pa,
        "flow_ref_t_k": args.ghsv_ref_t_k,
        "initial_flow_written": flow_output_values[0],
        "min_flow_written": min(flow_output_values),
        "max_flow_written": max(flow_output_values),
        "initial_flow_sample_basis": meta.flow_basis_values[0],
        "min_flow_sample_basis": min(meta.flow_basis_values),
        "max_flow_sample_basis": max(meta.flow_basis_values),
        "initial_temperature_k": temp_values[0],
        "min_temperature_k": min(temp_values),
        "max_temperature_k": max(temp_values),
        "initial_outlet_pressure_pa": pressure_values[0],
        "min_outlet_pressure_pa": min(pressure_values),
        "max_outlet_pressure_pa": max(pressure_values),
        "composition_regimes": collect_composition_labels(meta),
    }
    if geometry.bed_volume_m3 is not None:
        row["ghsv_to_mol_s_factor"] = ghsv_to_molar_factor(
            geometry, args.ghsv_ref_p_pa, args.ghsv_ref_t_k
        )
    else:
        row["ghsv_to_mol_s_factor"] = ""
    return row


def write_manifest(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_generation_config(args: argparse.Namespace, path: Path) -> None:
    d = vars(args).copy()
    for key, value in list(d.items()):
        if isinstance(value, Path):
            d[key] = str(value)
    with open(path, "w") as f:
        json.dump(d, f, indent=2, sort_keys=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YAML operating programs for chemical-looping packed-bed simulations."
    )
    parser.add_argument("--n-programs", type=int, default=56, help="number of base programs to generate")
    parser.add_argument("--seed", type=int, default=20260508, help="master random seed")
    parser.add_argument("--out-dir", type=Path, default=Path("generated_programs"), help="output directory")
    parser.add_argument("--time-horizon-s", type=_positive_float, default=3600.0)
    parser.add_argument(
        "--families",
        nargs="*",
        default=["canonical_cycle", "async_random", "composition_sweep", "pulse_train", "boundary_sweep"],
        help="trajectory families to cycle through",
    )

    parser.add_argument("--sample-flow-on", choices=["ghsv", "molar"], default="ghsv")
    parser.add_argument("--write-flow-as", choices=["ghsv", "molar"], default="molar")
    parser.add_argument("--ghsv-range-h-1", nargs=2, type=_positive_float, default=(100.0, 20000.0))
    parser.add_argument("--flow-range-mol-s", nargs=2, type=_positive_float, default=(0.01, 40.0))
    parser.add_argument(
        "--ghsv-ref-p-pa",
        type=_positive_float,
        default=101325.0,
        help="reference pressure used for GHSV <-> molar-flow conversion",
    )
    parser.add_argument(
        "--ghsv-ref-t-k",
        type=_positive_float,
        default=273.15,
        help="reference temperature used for GHSV <-> molar-flow conversion",
    )

    parser.add_argument("--temperature-range-k", nargs=2, type=_positive_float, default=(623.15, 973.15))
    parser.add_argument("--pressure-range-pa", nargs=2, type=_positive_float, default=(1.0e5, 3.5e6))

    parser.add_argument("--geometry-csv", type=Path, default=None)
    parser.add_argument("--bed-volume-m3", type=_positive_float, default=None)
    parser.add_argument("--bed-length-m", type=_positive_float, default=None)
    parser.add_argument("--bed-radius-m", type=_positive_float, default=None)

    parser.add_argument(
        "--single-directory",
        action="store_true",
        help="write all files in one directory even when multiple geometries are supplied",
    )

    args = parser.parse_args(argv)
    if args.n_programs <= 0:
        parser.error("--n-programs must be positive")
    args.ghsv_range_h_1 = _pair_of_floats(args.ghsv_range_h_1, name="--ghsv-range-h-1")
    args.flow_range_mol_s = _pair_of_floats(args.flow_range_mol_s, name="--flow-range-mol-s")
    args.temperature_range_k = _pair_of_floats(args.temperature_range_k, name="--temperature-range-k")
    args.pressure_range_pa = _pair_of_floats(args.pressure_range_pa, name="--pressure-range-pa")
    if args.bed_volume_m3 is not None and (args.bed_length_m is not None or args.bed_radius_m is not None):
        parser.error("use either --bed-volume-m3 or --bed-length-m/--bed-radius-m, not both")
    if (args.bed_length_m is None) ^ (args.bed_radius_m is None):
        parser.error("--bed-length-m and --bed-radius-m must be supplied together")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    geoms = read_geometries(args)
    ensure_flow_conversion_is_possible(args, geoms)

    master_rng = np.random.default_rng(args.seed)
    families = choose_families(args.n_programs, args.families, master_rng)
    lhs = latin_hypercube(args.n_programs, 5, master_rng)
    child_seeds = master_rng.integers(1, np.iinfo(np.int32).max, size=args.n_programs, dtype=np.int64)

    manifest_rows: List[Dict[str, Any]] = []
    for i in range(args.n_programs):
        program, meta = generate_program_in_sample_basis(
            program_index=i,
            seed=int(child_seeds[i]),
            family=families[i],
            lhs_row=lhs[i],
            args=args,
        )
        for geom in geoms:
            program_out = convert_program_flow(
                program,
                sample_basis=args.sample_flow_on,
                output_basis=args.write_flow_as,
                geometry=geom,
                p_ref_pa=args.ghsv_ref_p_pa,
                t_ref_k=args.ghsv_ref_t_k,
            )
            if len(geoms) > 1 and not args.single_directory:
                rel = Path(f"geometry_{geom.geometry_id}") / f"{meta.program_id}.yaml"
            elif len(geoms) > 1:
                rel = Path(f"{meta.program_id}__{geom.geometry_id}.yaml")
            else:
                rel = Path(f"{meta.program_id}.yaml")
            path = out_dir / rel
            dump_yaml(program_out, path)
            manifest_rows.append(
                manifest_row(
                    meta=meta,
                    program=program_out,
                    output_path=path,
                    output_dir=out_dir,
                    geometry=geom,
                    args=args,
                )
            )

    write_manifest(manifest_rows, out_dir / "manifest.csv")
    write_generation_config(args, out_dir / "generation_config.json")
    print(f"Wrote {len(manifest_rows)} YAML files to {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
