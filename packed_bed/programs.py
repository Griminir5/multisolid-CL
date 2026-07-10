from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packed_bed.config.models import (
        CompositionChannelConfig,
        CompositionRampStep,
        HoldStep,
        ModelConfig,
        ProgramConfig,
        ScalarChannelConfig,
        ScalarRampStep,
    )


DEFAULT_SMOOTH_RAMP_WIDTH_S = 1.0
NORMAL_TEMPERATURE_K = 273.15
NORMAL_PRESSURE_PA = 100000.0
GAS_CONSTANT_J_PER_MOL_K = 8.31446
NORMAL_MOLAR_DENSITY_MOL_PER_M3 = (
    NORMAL_PRESSURE_PA / (GAS_CONSTANT_J_PER_MOL_K * NORMAL_TEMPERATURE_K)
)
ProgramValue = float | tuple[float, ...]


@dataclass(frozen=True)
class ProgramSegment:
    start_time: float
    end_time: float
    start_value: ProgramValue
    end_value: ProgramValue


@dataclass(frozen=True)
class CompiledProgram:
    initial_value: ProgramValue
    segments: tuple[ProgramSegment, ...]

    def value_at(self, time_s: float, *, smooth_ramp_width_s: float) -> ProgramValue:
        return _evaluate_smoothed_program_value(
            self.initial_value,
            self.segments,
            time_s=time_s,
            smooth_ramp_width_s=smooth_ramp_width_s,
        )

    @property
    def duration_s(self) -> float:
        return self.segments[-1].end_time if self.segments else 0.0


def sum_step_durations(steps: tuple["HoldStep | ScalarRampStep | CompositionRampStep", ...]) -> float:
    return math.fsum(step.duration_s for step in steps)


def _require_exact_keys(actual: set[str], expected: tuple[str, ...], label: str) -> None:
    expected_keys = set(expected)
    if actual == expected_keys:
        return
    missing = sorted(expected_keys - actual)
    extra = sorted(actual - expected_keys)
    differences = []
    if missing:
        differences.append(f"missing {', '.join(missing)}")
    if extra:
        differences.append(f"unexpected {', '.join(extra)}")
    raise ValueError(f"{label} species mismatch: {'; '.join(differences)}.")


def _interpolate_program_value(
    start_value: ProgramValue,
    end_value: ProgramValue,
    fraction: float,
) -> ProgramValue:
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
    initial_value: ProgramValue,
    segments: tuple["ProgramSegment", ...],
    *,
    time_s: float,
    smooth_ramp_width_s: float,
) -> ProgramValue:
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


def _compile_program_segments(
    initial_value: ProgramValue,
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


def compile_scalar_channel(
    channel: "ScalarChannelConfig",
    *,
    repeat: bool = False,
    time_horizon: float | None = None,
    value_scale: float = 1.0,
) -> CompiledProgram:
    initial_value = channel.initial * value_scale
    segments = _compile_program_segments(
        initial_value,
        channel.steps,
        repeat=repeat,
        time_horizon=time_horizon,
        resolve_next_value=lambda _step_index, step, current_value: (
            current_value if step.kind == "hold" else step.target * value_scale
        ),
    )
    return CompiledProgram(initial_value=initial_value, segments=segments)


def compile_composition_channel(
    channel: "CompositionChannelConfig",
    species_order: tuple[str, ...],
    *,
    repeat: bool = False,
    time_horizon: float | None = None,
) -> CompiledProgram:
    _require_exact_keys(set(channel.initial), species_order, "program.inlet_composition.initial")
    initial_value = tuple(channel.initial[species_id] for species_id in species_order)

    def resolve_next_value(step_index: int, step, current_value: ProgramValue) -> ProgramValue:
        if step.kind == "hold":
            return current_value

        _require_exact_keys(
            set(step.target),
            species_order,
            f"program.inlet_composition.steps[{step_index}].target",
        )
        return tuple(step.target[species_id] for species_id in species_order)

    segments = _compile_program_segments(
        initial_value,
        channel.steps,
        repeat=repeat,
        time_horizon=time_horizon,
        resolve_next_value=resolve_next_value,
    )
    return CompiledProgram(initial_value=initial_value, segments=segments)


def compile_program_channels(
    config: "ProgramConfig",
    gas_species: tuple[str, ...],
    model: "ModelConfig",
    *,
    repeat: bool,
    time_horizon: float,
) -> tuple[CompiledProgram, CompiledProgram, CompiledProgram, CompiledProgram]:
    """Compile all operating channels once, in their runtime field order."""

    inlet_flow_scale = 1.0
    if config.inlet_flow.basis == "ghsv_per_h":
        empty_bed_volume_m3 = math.pi * model.bed_radius_m**2 * model.bed_length_m
        inlet_flow_scale = empty_bed_volume_m3 * NORMAL_MOLAR_DENSITY_MOL_PER_M3 / 3600.0

    return (
        compile_scalar_channel(
            config.inlet_flow,
            repeat=repeat,
            time_horizon=time_horizon,
            value_scale=inlet_flow_scale,
        ),
        compile_composition_channel(
            config.inlet_composition,
            gas_species,
            repeat=repeat,
            time_horizon=time_horizon,
        ),
        compile_scalar_channel(
            config.inlet_temperature,
            repeat=repeat,
            time_horizon=time_horizon,
        ),
        compile_scalar_channel(
            config.outlet_pressure,
            repeat=repeat,
            time_horizon=time_horizon,
        ),
    )


__all__ = (
    "CompiledProgram",
    "DEFAULT_SMOOTH_RAMP_WIDTH_S",
    "NORMAL_MOLAR_DENSITY_MOL_PER_M3",
    "ProgramSegment",
    "compile_composition_channel",
    "compile_program_channels",
    "compile_scalar_channel",
    "sum_step_durations",
)
