from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from packed_bed.config.models import CompositionRampStep, HoldStep, ScalarRampStep


DEFAULT_SMOOTH_RAMP_WIDTH_S = 1.0


@dataclass(frozen=True)
class ProgramSegment:
    start_time: float
    end_time: float
    start_value: float | tuple[float, ...]
    end_value: float | tuple[float, ...]


@dataclass(frozen=True)
class ScalarProgram:
    initial_value: float
    segments: tuple[ProgramSegment, ...]

    def build_segments(self) -> tuple[ProgramSegment, ...]:
        return self.segments

    def smoothed_value_at(self, time_s: float, *, smooth_ramp_width_s: float) -> float:
        value = _evaluate_smoothed_program_value(
            self.initial_value,
            self.segments,
            time_s=time_s,
            smooth_ramp_width_s=smooth_ramp_width_s,
        )
        if isinstance(value, tuple):
            raise TypeError("Expected scalar-valued program.")
        return value


@dataclass(frozen=True)
class VectorProgram:
    initial_value: tuple[float, ...]
    segments: tuple[ProgramSegment, ...]

    def build_segments(self) -> tuple[ProgramSegment, ...]:
        return self.segments

    def smoothed_value_at(self, time_s: float, *, smooth_ramp_width_s: float) -> tuple[float, ...]:
        value = _evaluate_smoothed_program_value(
            self.initial_value,
            self.segments,
            time_s=time_s,
            smooth_ramp_width_s=smooth_ramp_width_s,
        )
        if not isinstance(value, tuple):
            raise TypeError("Expected tuple-valued program.")
        return value
    

def sum_step_durations(steps: tuple["HoldStep | ScalarRampStep | CompositionRampStep", ...]) -> float:
    return math.fsum(step.duration_s for step in steps)


def _interpolate_program_value(
    start_value: float | tuple[float, ...],
    end_value: float | tuple[float, ...],
    fraction: float,
) -> float | tuple[float, ...]:
    if isinstance(start_value, tuple):
        if not isinstance(end_value, tuple):
            raise TypeError("Expected tuple-valued program endpoints.")
        return tuple(
            start_component + (end_component - start_component) * fraction
            for start_component, end_component in zip(start_value, end_value)
        )

    if isinstance(end_value, tuple):
        raise TypeError("Expected scalar-valued program endpoints.")
    return start_value + (end_value - start_value) * fraction


def _smooth_positive_time_value(elapsed_time_s: float, smooth_ramp_width_s: float) -> float:
    if smooth_ramp_width_s <= 0.0:
        raise ValueError("smooth_ramp_width_s must be positive.")
    return 0.5 * (elapsed_time_s + math.sqrt(elapsed_time_s * elapsed_time_s + smooth_ramp_width_s**2))


def _smooth_ramp_fraction_value(segment: "ProgramSegment", time_s: float, smooth_ramp_width_s: float) -> float:
    duration_s = float(segment.end_time) - float(segment.start_time)
    if duration_s <= 0.0:
        raise ValueError("Program segments must have positive duration.")
    return (
        _smooth_positive_time_value(time_s - float(segment.start_time), smooth_ramp_width_s)
        - _smooth_positive_time_value(time_s - float(segment.end_time), smooth_ramp_width_s)
    ) / duration_s


def _evaluate_smoothed_program_value(
    initial_value: float | tuple[float, ...],
    segments: tuple["ProgramSegment", ...],
    *,
    time_s: float,
    smooth_ramp_width_s: float,
) -> float | tuple[float, ...]:
    if not segments:
        return initial_value

    if isinstance(segments[0].start_value, tuple):
        value = [float(component) for component in segments[0].start_value]
        for segment in segments:
            if not isinstance(segment.start_value, tuple) or not isinstance(segment.end_value, tuple):
                raise TypeError("Expected tuple-valued program endpoints.")
            fraction = _smooth_ramp_fraction_value(segment, time_s, smooth_ramp_width_s)
            for component_idx, (start_component, end_component) in enumerate(
                zip(segment.start_value, segment.end_value)
            ):
                value[component_idx] += (float(end_component) - float(start_component)) * fraction
        return tuple(value)

    value = float(segments[0].start_value)
    for segment in segments:
        if isinstance(segment.start_value, tuple) or isinstance(segment.end_value, tuple):
            raise TypeError("Expected scalar-valued program endpoints.")
        delta = float(segment.end_value) - float(segment.start_value)
        if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
            continue
        value += delta * _smooth_ramp_fraction_value(segment, time_s, smooth_ramp_width_s)
    return value


def compile_program_segments(
    initial_value: float | tuple[float, ...],
    steps: tuple["HoldStep | ScalarRampStep | CompositionRampStep", ...],
    *,
    repeat: bool,
    time_horizon: float | None,
    resolve_next_value,
) -> tuple["ProgramSegment", ...]:
    if repeat and time_horizon is None:
        raise ValueError("time_horizon must be provided when repeat=True.")

    current_time = 0.0
    current_value = initial_value
    segments: list[ProgramSegment] = []

    while True:
        for step_index, step in enumerate(steps):
            next_time = current_time + step.duration_s
            next_value = resolve_next_value(step_index, step, current_value)

            if time_horizon is not None and next_time > time_horizon:
                if current_time >= time_horizon:
                    return tuple(segments)

                fraction = (time_horizon - current_time) / step.duration_s
                segments.append(
                    ProgramSegment(
                        start_time=current_time,
                        end_time=time_horizon,
                        start_value=current_value,
                        end_value=_interpolate_program_value(current_value, next_value, fraction),
                    )
                )
                return tuple(segments)

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

        if not repeat or not steps or (time_horizon is not None and current_time >= time_horizon):
            return tuple(segments)
