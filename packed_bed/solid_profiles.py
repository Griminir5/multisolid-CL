from __future__ import annotations

import numpy as np


_POSITION_TOL = 1e-12


def build_uniform_axial_grid(bed_length_m, axial_cells):
    face_positions = np.linspace(0.0, float(bed_length_m), int(axial_cells) + 1, dtype=float)
    cell_centers = 0.5 * (face_positions[:-1] + face_positions[1:])
    return cell_centers, face_positions


def zone_edges(solids_config):
    zones = solids_config.initial_profile.zones
    if not zones:
        return np.asarray([], dtype=float)
    return np.asarray(
        [float(zones[0].x_start_m), *[float(zone.x_end_m) for zone in zones]],
        dtype=float,
    )


def build_cell_profiles(solids_config, cell_centers_m):
    """Assign each cell's voidages and bed-volume solid concentrations together."""

    cell_centers_m = np.asarray(cell_centers_m, dtype=float)
    solid_species = solids_config.solid_species
    profile = np.zeros((len(solid_species), len(cell_centers_m)), dtype=float)
    e_b = np.zeros(len(cell_centers_m), dtype=float)
    e_p = np.zeros(len(cell_centers_m), dtype=float)
    assigned = np.zeros(len(cell_centers_m), dtype=bool)
    zones = solids_config.initial_profile.zones

    for zone_index, zone in enumerate(zones):
        is_last_zone = zone_index == len(zones) - 1
        if is_last_zone:
            mask = (cell_centers_m >= zone.x_start_m - _POSITION_TOL) & (
                cell_centers_m <= zone.x_end_m + _POSITION_TOL
            )
        else:
            mask = (cell_centers_m >= zone.x_start_m - _POSITION_TOL) & (
                cell_centers_m < zone.x_end_m - _POSITION_TOL
            )

        assigned |= mask
        e_b[mask] = zone.e_b
        e_p[mask] = zone.e_p
        for sol_idx, species_id in enumerate(solid_species):
            profile[sol_idx, mask] = float(zone.values[species_id])

    if cell_centers_m.size and not np.all(assigned):
        raise ValueError("Solid profile zones did not cover every cell center.")

    if solids_config.initial_profile.basis == "solid":
        profile *= solid_fraction_from_voidages(e_b, e_p)[np.newaxis, :]
    elif solids_config.initial_profile.basis != "bed":
        raise ValueError(f"Unsupported solid concentration basis '{solids_config.initial_profile.basis}'.")
    return e_b, e_p, profile


def build_face_scalar_profile(solids_config, face_positions_m, attribute_name):
    face_positions_m = np.asarray(face_positions_m, dtype=float)
    profile = np.zeros(len(face_positions_m), dtype=float)
    zones = solids_config.initial_profile.zones

    for face_index, position in enumerate(face_positions_m):
        assigned = False

        for zone_index in range(len(zones) - 1):
            boundary = float(zones[zone_index].x_end_m)
            if abs(position - boundary) <= _POSITION_TOL:
                left_value = float(getattr(zones[zone_index], attribute_name))
                right_value = float(getattr(zones[zone_index + 1], attribute_name))
                profile[face_index] = 0.5 * (left_value + right_value)
                assigned = True
                break

        if assigned:
            continue

        for zone_index, zone in enumerate(zones):
            is_last_zone = zone_index == len(zones) - 1
            upper_ok = (
                position <= zone.x_end_m + _POSITION_TOL
                if is_last_zone
                else position < zone.x_end_m - _POSITION_TOL
            )
            if position >= zone.x_start_m - _POSITION_TOL and upper_ok:
                profile[face_index] = float(getattr(zone, attribute_name))
                assigned = True
                break

        if not assigned:
            raise ValueError(
                f"Solid profile zones did not cover face position {position} for '{attribute_name}'."
            )

    return profile


def gas_fraction_from_voidages(e_b, e_p, *, mode="bed_and_particle"):
    """Gas storage volume per bed volume, with optional particle-pore storage."""

    e_b = np.asarray(e_b, dtype=float)
    e_p = np.asarray(e_p, dtype=float)
    if mode == "bed_only":
        return np.broadcast_arrays(e_b, e_p)[0]
    if mode == "bed_and_particle":
        return e_b + (1.0 - e_b) * e_p
    raise ValueError(f"Unsupported gas voidage mode '{mode}'.")


def solid_fraction_from_voidages(e_b, e_p):
    """Solid skeleton fraction, independent of the chosen gas storage model."""

    return (1.0 - np.asarray(e_b, dtype=float)) * (1.0 - np.asarray(e_p, dtype=float))
