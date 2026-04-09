from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np


@dataclass(frozen=True)
class ProgramStep:
    duration: float
    kind: Literal["hold", "ramp"]
    target: float | Mapping[str, float] | np.ndarray | None = None


@dataclass(frozen=True)
class ProgramSegment:
    start_time: float
    end_time: float
    start_value: float | np.ndarray
    end_value: float | np.ndarray


def default_inlet_composition(gas_species):
    inlet_y = np.zeros(len(gas_species), dtype=float)
    if inlet_y.size:
        inlet_y[0] = 1.0
    return inlet_y


def coerce_scalar(value, label):
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{label} must be finite.")
    return scalar


def coerce_composition_mapping(value, species_order, label="Composition"):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be provided as a species-keyed mapping.")

    expected = tuple(species_order)
    missing = [species_id for species_id in expected if species_id not in value]
    unexpected = [species_id for species_id in value if species_id not in expected]
    if missing:
        raise ValueError(f"{label} is missing entries for: {', '.join(missing)}.")
    if unexpected:
        raise ValueError(f"{label} contains unknown entries: {', '.join(unexpected)}.")

    composition = np.asarray([float(value[species_id]) for species_id in expected], dtype=float)
    if not np.all(np.isfinite(composition)):
        raise ValueError(f"{label} must contain only finite values.")
    if np.any(composition < -1e-12) or np.any(composition > 1.0 + 1e-12):
        raise ValueError(f"{label} entries must stay within [0, 1].")
    if not np.isclose(composition.sum(), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"{label} must sum to 1.")
    return composition


def coerce_vector(value, expected_size=None, label="Vector"):
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{label} must be provided as a 1D vector.")
    if expected_size is not None and vector.size != expected_size:
        raise ValueError(f"{label} must contain exactly {expected_size} entries.")
    if vector.size == 0:
        raise ValueError(f"{label} must contain at least one entry.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite values.")
    return vector.copy()


class ScalarProgram:
    """Finite hold/ramp program for scalar boundary conditions."""

    def __init__(self, initial_value):
        self.initial_value = coerce_scalar(initial_value, "Scalar program initial value")
        self.steps: list[ProgramStep] = []

    def hold(self, duration):
        self.steps.append(ProgramStep(duration=float(duration), kind="hold"))
        return self

    def ramp(self, duration, target):
        self.steps.append(
            ProgramStep(
                duration=float(duration),
                kind="ramp",
                target=coerce_scalar(target, "Scalar program target"),
            )
        )
        return self

    def build_segments(self, time_horizon=None):
        segments = []
        current_time = 0.0
        current_value = self.initial_value

        for step in self.steps:
            if step.duration <= 0.0:
                raise ValueError("Program step durations must be positive.")

            next_time = current_time + step.duration
            next_value = current_value if step.kind == "hold" else coerce_scalar(step.target, "Program target")
            segments.append(
                ProgramSegment(
                    start_time=current_time,
                    end_time=next_time,
                    start_value=current_value,
                    end_value=next_value,
                )
            )
            current_time = next_time
            current_value = next_value

        if time_horizon is not None and (not segments or segments[-1].end_time < time_horizon):
            segments.append(
                ProgramSegment(
                    start_time=current_time,
                    end_time=float(time_horizon),
                    start_value=current_value,
                    end_value=current_value,
                )
            )

        return segments


class VectorProgram:
    """Finite hold/ramp program for vector-valued boundary conditions."""

    def __init__(self, initial_value):
        self.initial_value = coerce_vector(initial_value, label="Vector program initial value")
        self.steps: list[ProgramStep] = []

    def hold(self, duration):
        self.steps.append(ProgramStep(duration=float(duration), kind="hold"))
        return self

    def ramp(self, duration, target):
        self.steps.append(
            ProgramStep(
                duration=float(duration),
                kind="ramp",
                target=coerce_vector(
                    target,
                    expected_size=self.initial_value.size,
                    label="Vector program target",
                ),
            )
        )
        return self

    def build_segments(self, time_horizon=None):
        segments = []
        current_time = 0.0
        current_value = self.initial_value.copy()

        for step in self.steps:
            if step.duration <= 0.0:
                raise ValueError("Program step durations must be positive.")

            next_time = current_time + step.duration
            next_value = current_value.copy() if step.kind == "hold" else coerce_vector(
                step.target,
                expected_size=self.initial_value.size,
                label="Vector program target",
            )
            segments.append(
                ProgramSegment(
                    start_time=current_time,
                    end_time=next_time,
                    start_value=current_value.copy(),
                    end_value=next_value.copy(),
                )
            )
            current_time = next_time
            current_value = next_value

        if time_horizon is not None and (not segments or segments[-1].end_time < time_horizon):
            segments.append(
                ProgramSegment(
                    start_time=current_time,
                    end_time=float(time_horizon),
                    start_value=current_value.copy(),
                    end_value=current_value.copy(),
                )
            )

        return segments
