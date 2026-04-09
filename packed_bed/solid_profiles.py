from __future__ import annotations

import numpy as np


_POSITION_TOL = 1e-12


def build_uniform_axial_grid(bed_length_m, axial_cells):
    face_positions = np.linspace(0.0, float(bed_length_m), int(axial_cells) + 1, dtype=float)
    cell_centers = 0.5 * (face_positions[:-1] + face_positions[1:])
    return cell_centers, face_positions


def zone_edges(solids_config):
    zones = solids_config.initial_profile_zones
    if not zones:
        return np.asarray([], dtype=float)
    return np.asarray(
        [float(zones[0].x_start_m), *[float(zone.x_end_m) for zone in zones]],
        dtype=float,
    )


def build_solid_profile_matrix(solids_config, cell_centers_m, solid_species):
    cell_centers_m = np.asarray(cell_centers_m, dtype=float)
    profile = np.zeros((len(solid_species), len(cell_centers_m)), dtype=float)
    assigned = np.zeros(len(cell_centers_m), dtype=bool)
    zones = solids_config.initial_profile_zones

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
        for sol_idx, species_id in enumerate(solid_species):
            profile[sol_idx, mask] = float(zone.values_mol_per_m3[species_id])

    if cell_centers_m.size and not np.all(assigned):
        raise ValueError("Solid profile zones did not cover every cell center.")

    return profile


def build_cell_scalar_profile(solids_config, cell_centers_m, attribute_name):
    cell_centers_m = np.asarray(cell_centers_m, dtype=float)
    profile = np.zeros(len(cell_centers_m), dtype=float)
    assigned = np.zeros(len(cell_centers_m), dtype=bool)
    zones = solids_config.initial_profile_zones

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

        profile[mask] = float(getattr(zone, attribute_name))
        assigned |= mask

    if cell_centers_m.size and not np.all(assigned):
        raise ValueError(f"Solid profile zones did not cover every cell center for '{attribute_name}'.")

    return profile


def build_face_scalar_profile(solids_config, face_positions_m, attribute_name):
    face_positions_m = np.asarray(face_positions_m, dtype=float)
    profile = np.zeros(len(face_positions_m), dtype=float)
    zones = solids_config.initial_profile_zones

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
            upper_ok = position <= zone.x_end_m + _POSITION_TOL if is_last_zone else position < zone.x_end_m - _POSITION_TOL
            if position >= zone.x_start_m - _POSITION_TOL and upper_ok:
                profile[face_index] = float(getattr(zone, attribute_name))
                assigned = True
                break

        if not assigned:
            raise ValueError(
                f"Solid profile zones did not cover face position {position} for '{attribute_name}'."
            )

    return profile


def gas_fraction_from_voidages(e_b, e_p):
    e_b = np.asarray(e_b, dtype=float)
    e_p = np.asarray(e_p, dtype=float)
    return e_b + (1.0 - e_b) * e_p


def solid_fraction_from_voidages(e_b, e_p):
    return 1.0 - gas_fraction_from_voidages(e_b, e_p)


def convert_solid_profile_to_bed_volume(solids_config, cell_centers_m, solid_fraction, solid_species):
    solid_profile_basis = build_solid_profile_matrix(solids_config, cell_centers_m, solid_species)
    if solids_config.concentration_unit == "mol_per_m3_solid":
        return solid_profile_basis * np.asarray(solid_fraction, dtype=float)[np.newaxis, :]
    if solids_config.concentration_unit == "mol_per_m3_bed":
        return solid_profile_basis
    raise ValueError(f"Unsupported solid concentration unit '{solids_config.concentration_unit}'.")
