"""Numerical previews shared by the CLI and desktop, with no rendering imports."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import Case
from .programs import DEFAULT_SMOOTH_RAMP_WIDTH_S, RatioProgram
from .solid_profiles import (
    build_cell_profiles,
    build_face_scalar_profile,
    build_uniform_axial_grid,
    zone_edges,
)


@dataclass(frozen=True)
class CasePreview:
    time_s: np.ndarray
    flow_mol_s: np.ndarray
    temperature_k: np.ndarray
    pressure_pa: np.ndarray
    mole_fractions: np.ndarray  # (time, gas species), in case species order
    cell_positions_m: np.ndarray
    face_positions_m: np.ndarray
    zone_edges_m: np.ndarray
    solid_concentrations_mol_m3_bed: np.ndarray  # (solid species, cell)
    interparticle_voidage: np.ndarray
    particle_voidage: np.ndarray
    particle_diameter_m: np.ndarray  # face domain


def preview_case(case: Case) -> CasePreview:
    """Sample the engine's compiled, smoothed programs and actual spatial grid."""
    programs = (
        case.inlet_flow_program, case.inlet_temperature_program,
        case.outlet_pressure_program, case.inlet_composition_program,
    )
    width = DEFAULT_SMOOTH_RAMP_WIDTH_S
    times = _smoothed_program_sample_times(
        programs, final_time=case.run.simulation.time_horizon_s, smooth_ramp_width_s=width,
    )
    values = [_series_from_smoothed_program(p, times, smooth_ramp_width_s=width) for p in programs]
    centers, faces = build_uniform_axial_grid(case.run.model.bed_length_m, case.run.model.axial_cells)
    e_b, e_p, concentrations = build_cell_profiles(case.solids, centers)
    return CasePreview(
        time_s=times,
        flow_mol_s=values[0],
        temperature_k=values[1],
        pressure_pa=values[2],
        mole_fractions=values[3],
        cell_positions_m=centers,
        face_positions_m=faces,
        zone_edges_m=zone_edges(case.solids),
        solid_concentrations_mol_m3_bed=concentrations,
        interparticle_voidage=e_b,
        particle_voidage=e_p,
        particle_diameter_m=build_face_scalar_profile(case.solids, faces, "d_p"),
    )


def _segment_changes_value(segment) -> bool:
    start_value = np.asarray(segment.start_value, dtype=float)
    end_value = np.asarray(segment.end_value, dtype=float)
    return not np.allclose(start_value, end_value, rtol=0.0, atol=1e-12)


def _smoothed_program_sample_times(programs, *, final_time: float, smooth_ramp_width_s: float) -> np.ndarray:
    final_time = float(final_time)
    if final_time <= 0.0:
        return np.array([0.0], dtype=float)

    programs = tuple(
        source
        for program in programs
        for source in ((program.numerator, program.denominator) if isinstance(program, RatioProgram) else (program,))
    )
    times = {0.0, final_time}
    total_segments = sum(len(program.segments) for program in programs)
    baseline_count = max(400, min(2500, 20 * total_segments + 400))
    times.update(float(value) for value in np.linspace(0.0, final_time, baseline_count))

    width = float(smooth_ramp_width_s)
    if width <= 0.0:
        raise ValueError("smooth_ramp_width_s must be positive.")
    edge_offsets = width * np.array(
        [-8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    )
    for program in programs:
        for segment in program.segments:
            start_time = float(segment.start_time)
            end_time = float(segment.end_time)
            for edge_time in (start_time, end_time):
                for sample_time in edge_time + edge_offsets:
                    if 0.0 <= sample_time <= final_time:
                        times.add(float(sample_time))

            if _segment_changes_value(segment):
                duration_s = max(end_time - start_time, 0.0)
                interior_count = max(
                    8,
                    min(80, int(math.ceil(duration_s / width)) * 4),
                )
                times.update(float(value) for value in np.linspace(start_time, end_time, interior_count))

    return np.asarray(sorted(times), dtype=float)


def _series_from_smoothed_program(
    program,
    times: np.ndarray,
    *,
    smooth_ramp_width_s: float,
) -> np.ndarray:
    return np.asarray(
        [
            program.value_at(float(time_s), smooth_ramp_width_s=smooth_ramp_width_s)
            for time_s in times
        ],
        dtype=float,
    )
