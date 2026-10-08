from __future__ import annotations

import math
from dataclasses import dataclass
from bisect import bisect_right
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packed_bed.config.models import (
        CompositionChannelConfig,
        CompositionRampStep,
        FeedProgramConfig,
        FeedRampStep,
        FeedStreamConfig,
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
    """One startup cycle and its carried-forward repeating continuation.

    Smoothing has support only within one width of each ramp endpoint. Neither
    this representation nor its value at a time depends on the solver horizon.
    """
    initial_value: ProgramValue
    segments: tuple[ProgramSegment, ...]
    repeat_segments: tuple[ProgramSegment, ...] = ()

    @cached_property
    def _ends(self):
        return tuple(segment.end_time for segment in self.segments)

    @cached_property
    def _repeat_ends(self):
        return tuple(segment.end_time for segment in self.repeat_segments)

    def value_at(self, time_s: float, *, smooth_ramp_width_s: float) -> ProgramValue:
        width = float(smooth_ramp_width_s)
        if not math.isfinite(width) or width <= 0:
            raise ValueError("smooth_ramp_width_s must be finite and positive.")
        if not math.isfinite(time_s):
            raise ValueError("Program time must be finite.")
        if not self.segments:
            return self.initial_value
        period = self.duration_s
        first = max(0, math.floor((time_s - width) / period)) if self.repeat_segments else 0
        last = max(0, math.floor((time_s + width) / period)) if self.repeat_segments else 0
        value = None
        for cycle in range(first, last + 1):
            segments = self.repeat_segments if cycle else self.segments
            ends = self._repeat_ends if cycle else self._ends
            local = time_s - cycle * period
            index = bisect_right(ends, local - width)
            if value is None:
                value = segments[index].start_value if index < len(segments) else segments[-1].end_value
            for segment_index in range(index, len(segments)):
                segment = segments[segment_index]
                if segment.start_time >= local + width:
                    break
                fraction = _smooth_ramp_fraction_value(segment, local, width)
                if isinstance(value, tuple):
                    value = tuple(v + (b - a) * fraction for v, a, b in
                                  zip(value, segment.start_value, segment.end_value))
                else:
                    value += (segment.end_value - segment.start_value) * fraction
        return value

    def segments_until(self, stop_time):
        """Expand only for symbolic assembly or display, never truncate ramps."""
        cycle = 0
        while True:
            segments = self.repeat_segments if cycle else self.segments
            for segment in segments:
                shift = cycle * self.duration_s
                if segment.start_time > stop_time - shift:
                    return
                yield ProgramSegment(segment.start_time + shift, segment.end_time + shift,
                                     segment.start_value, segment.end_value)
            if not self.repeat_segments or not segments:
                return
            cycle += 1

    @property
    def duration_s(self) -> float:
        return self.segments[-1].end_time if self.segments else 0.0


@dataclass(frozen=True)
class RatioProgram:
    """Normalize a smoothed species-flow or flow-temperature program by flow.

    Division happens after smoothing so feed transitions conserve molar flow.
    The denominator is the shared, positive total molar flow program.
    """

    numerator: CompiledProgram
    denominator: CompiledProgram

    @staticmethod
    def _divide(value: ProgramValue, flow: ProgramValue) -> ProgramValue:
        if isinstance(flow, tuple) or not math.isfinite(flow) or flow <= 0:
            raise ValueError("RatioProgram requires a positive, finite scalar flow.")
        return tuple(v / flow for v in value) if isinstance(value, tuple) else value / flow

    @property
    def initial_value(self) -> ProgramValue:
        return self._divide(self.numerator.initial_value, self.denominator.initial_value)

    @property
    def duration_s(self) -> float:
        return self.numerator.duration_s

    def value_at(self, time_s: float, *, smooth_ramp_width_s: float) -> ProgramValue:
        return self._divide(
            self.numerator.value_at(time_s, smooth_ramp_width_s=smooth_ramp_width_s),
            self.denominator.value_at(time_s, smooth_ramp_width_s=smooth_ramp_width_s),
        )


def sum_step_durations(steps: tuple["HoldStep | ScalarRampStep | CompositionRampStep | FeedRampStep", ...]) -> float:
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


def smooth_positive_time(elapsed, width, *, minimum=min, maximum=max):
    """C1 compact-support smoothing of max(elapsed, 0), numeric or symbolic."""
    clipped = minimum(maximum(elapsed + width, 0 * width), 2 * width)
    return maximum(elapsed - width, 0 * width) + clipped * clipped / (4 * width)


def _smooth_ramp_fraction_value(segment: "ProgramSegment", time_s: float, smooth_ramp_width_s: float) -> float:
    duration_s = float(segment.end_time) - float(segment.start_time)
    if duration_s <= 0.0:
        raise ValueError("Program segments must have positive duration.")
    return (
        smooth_positive_time(time_s - float(segment.start_time), smooth_ramp_width_s)
        - smooth_positive_time(time_s - float(segment.end_time), smooth_ramp_width_s)
    ) / duration_s


def _compile_program_segments(initial_value, steps, *, repeat, time_horizon, resolve_next_value):
    # Targets are absolute, with omitted feed fields carried forward. After the
    # first traversal every later cycle is identical, including leading holds.
    current_value = initial_value
    cycles = []
    for _ in range(2 if repeat and steps else 1):
        current_time = 0.0
        segments = []
        for index, step in enumerate(steps):
            next_time = current_time + step.duration_s
            next_value = resolve_next_value(index, step, current_value)
            segments.append(ProgramSegment(current_time, next_time, current_value, next_value))
            current_time, current_value = next_time, next_value
        cycles.append(tuple(segments))
    return CompiledProgram(initial_value, cycles[0], cycles[1] if len(cycles) > 1 else ())


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
    return segments


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
    return segments


def compile_feed_stream(
    channel: "FeedStreamConfig",
    species_order: tuple[str, ...],
    *,
    repeat: bool = False,
    time_horizon: float | None = None,
    value_scale: float = 1.0,
) -> tuple[CompiledProgram, RatioProgram, RatioProgram]:
    """Ramp F, F*y and F*T together, then derive composition and temperature."""
    state = channel.initial
    _require_exact_keys(set(state.composition), species_order, "program.feed_stream.initial.composition")

    def flow_values(feed):
        flow = feed.flow * value_scale
        return (flow, *(flow * feed.composition[s] for s in species_order), flow * feed.temperature)

    initial = flow_values(state)

    def resolve_next_value(step_index, step, current_value):
        nonlocal state
        if step.kind == "hold":
            return current_value
        if step.target.composition is not None:
            _require_exact_keys(
                set(step.target.composition), species_order,
                f"program.feed_stream.steps[{step_index}].target.composition",
            )
        # Retain omitted fields from the last feed, including across repetitions.
        state = state.model_copy(update=step.target.model_dump(exclude_none=True))
        return flow_values(state)

    segments = _compile_program_segments(
        initial, channel.steps, repeat=repeat, time_horizon=time_horizon,
        resolve_next_value=resolve_next_value,
    )

    def project(select):
        def projected(values):
            return tuple(ProgramSegment(segment.start_time, segment.end_time,
                                        select(segment.start_value), select(segment.end_value))
                         for segment in values)
        return CompiledProgram(select(initial), projected(segments.segments), projected(segments.repeat_segments))

    flow = project(lambda value: value[0])
    species_flow = project(lambda value: value[1:-1])
    flow_temperature = project(lambda value: value[-1])
    return flow, RatioProgram(species_flow, flow), RatioProgram(flow_temperature, flow)


def compile_program_channels(
    config: "ProgramConfig | FeedProgramConfig",
    gas_species: tuple[str, ...],
    model: "ModelConfig",
    *,
    repeat: bool,
    time_horizon: float,
) -> tuple[CompiledProgram, CompiledProgram | RatioProgram, CompiledProgram | RatioProgram, CompiledProgram]:
    """Compile all operating channels once, in their runtime field order."""

    from .config.models import FeedProgramConfig

    flow_channel = config.feed_stream if isinstance(config, FeedProgramConfig) else config.inlet_flow
    inlet_flow_scale = 1.0
    if flow_channel.basis == "ghsv_per_h":
        empty_bed_volume_m3 = math.pi * model.bed_radius_m**2 * model.bed_length_m
        inlet_flow_scale = empty_bed_volume_m3 * NORMAL_MOLAR_DENSITY_MOL_PER_M3 / 3600.0

    if isinstance(config, FeedProgramConfig):
        return (
            *compile_feed_stream(
                config.feed_stream, gas_species, repeat=repeat,
                time_horizon=time_horizon, value_scale=inlet_flow_scale,
            ),
            compile_scalar_channel(config.outlet_pressure, repeat=repeat, time_horizon=time_horizon),
        )

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
    "RatioProgram",
    "compile_composition_channel",
    "compile_feed_stream",
    "compile_program_channels",
    "compile_scalar_channel",
    "sum_step_durations",
)
