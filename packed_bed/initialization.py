"""Pure initial-state calculation and its DAETools application boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Case
from .programs import DEFAULT_SMOOTH_RAMP_WIDTH_S, GAS_CONSTANT_J_PER_MOL_K
from .solid_profiles import (
    build_cell_scalar_profile,
    build_face_scalar_profile,
    convert_solid_profile_to_bed_volume,
    gas_fraction_from_voidages,
    solid_fraction_from_voidages,
)


CIRCLE_CONSTANT = 3.14159


@dataclass(frozen=True)
class InitialState:
    face_coordinates_m: np.ndarray
    interparticle_voidage: np.ndarray
    intraparticle_voidage: np.ndarray
    particle_diameter_m: np.ndarray
    gas_fraction: np.ndarray
    inlet_flow_mol_s: float
    inlet_molar_flux_mol_m2_s: float
    inlet_composition: np.ndarray
    inlet_temperature_k: float
    outlet_pressure_pa: float
    inlet_pressure_pa: float
    gas_concentration_mol_m3: np.ndarray
    solid_concentration_mol_m3: np.ndarray
    gas_enthalpy_j_mol: np.ndarray
    solid_enthalpy_j_mol: np.ndarray
    cell_enthalpy_j_m3: np.ndarray
    gas_density_kg_m3: np.ndarray
    gas_viscosity_pa_s: float
    face_velocity_m_s: np.ndarray
    bed_mass_kg: float
    bed_heat_j: float


def calculate_initial_state(
    case: Case,
    property_registry,
    *,
    smooth_ramp_width_s: float = DEFAULT_SMOOTH_RAMP_WIDTH_S,
) -> InitialState:
    """Calculate one consistently shaped initial state without DAETools objects."""

    cell_count = case.run.model.axial_cells
    face_coordinates = np.linspace(0.0, case.run.model.bed_length_m, cell_count + 1)
    cell_coordinates = 0.5 * (face_coordinates[:-1] + face_coordinates[1:])
    interparticle_voidage = build_cell_scalar_profile(case.solids, cell_coordinates, "e_b")
    intraparticle_voidage = build_cell_scalar_profile(case.solids, cell_coordinates, "e_p")
    particle_diameter = build_face_scalar_profile(case.solids, face_coordinates, "d_p")
    gas_fraction = gas_fraction_from_voidages(interparticle_voidage, intraparticle_voidage)
    solid_fraction = solid_fraction_from_voidages(interparticle_voidage, intraparticle_voidage)
    solid_concentration = convert_solid_profile_to_bed_volume(
        case.solids,
        cell_coordinates,
        solid_fraction,
        case.solids.solid_species,
    )

    def value_at_start(program):
        return program.value_at(0.0, smooth_ramp_width_s=smooth_ramp_width_s)

    inlet_flow = float(value_at_start(case.inlet_flow_program))
    if inlet_flow <= 0.0:
        raise ValueError("Initialization currently assumes a positive inlet molar flow.")
    inlet_composition = np.asarray(value_at_start(case.inlet_composition_program), dtype=float)
    inlet_temperature = float(value_at_start(case.inlet_temperature_program))
    outlet_pressure = float(value_at_start(case.outlet_pressure_program))

    gas_molecular_weights = np.asarray(
        [property_registry.get_record(name).mw for name in case.chemistry.gas_species],
        dtype=float,
    )
    solid_molecular_weights = np.asarray(
        [property_registry.get_record(name).mw for name in case.solids.solid_species],
        dtype=float,
    )
    gas_viscosities = np.asarray([
        property_registry.viscosity_value(name, inlet_temperature)
        for name in case.chemistry.gas_species
    ])
    gas_enthalpy = np.asarray([
        property_registry.enthalpy_value(name, inlet_temperature)
        for name in case.chemistry.gas_species
    ])
    solid_enthalpy = np.asarray([
        property_registry.enthalpy_value(name, inlet_temperature)
        for name in case.solids.solid_species
    ])

    mixture_molecular_weight = float(inlet_composition @ gas_molecular_weights)
    mixture_viscosity = float(inlet_composition @ gas_viscosities)
    area_m2 = CIRCLE_CONSTANT * case.run.model.bed_radius_m**2
    inlet_molar_flux = inlet_flow / area_m2

    def ergun_terms(voidage, particle_diameter_m, density_weight):
        alpha = 150.0 * mixture_viscosity * (1.0 - voidage) ** 2 / (
            voidage**3 * particle_diameter_m**2
        )
        beta = 1.75 * (1.0 - voidage) / (voidage**3 * particle_diameter_m)
        pressure_term = alpha * inlet_molar_flux * GAS_CONSTANT_J_PER_MOL_K * inlet_temperature
        density_term = (
            density_weight
            * beta
            * mixture_molecular_weight
            * inlet_molar_flux**2
            * GAS_CONSTANT_J_PER_MOL_K
            * inlet_temperature
        )
        return pressure_term, density_term

    inlet_half_cell_width = cell_coordinates[0] - face_coordinates[0]
    outlet_half_cell_width = face_coordinates[-1] - cell_coordinates[-1]

    def pressure_profile(inlet_pressure):
        pressures = np.zeros(cell_count, dtype=float)
        pressure_term, density_term = ergun_terms(interparticle_voidage[0], particle_diameter[0], 1.0)
        pressures[0] = (
            inlet_pressure - inlet_half_cell_width * pressure_term / inlet_pressure
        ) / (1.0 + inlet_half_cell_width * density_term / inlet_pressure**2)
        if pressures[0] <= 0.0:
            raise ValueError("Computed a non-positive first-cell pressure during initialization.")

        for face_index in range(1, cell_count):
            left_index = face_index - 1
            right_index = face_index
            face_width = cell_coordinates[right_index] - cell_coordinates[left_index]
            face_voidage = 0.5 * (
                interparticle_voidage[left_index] + interparticle_voidage[right_index]
            )
            pressure_term, density_term = ergun_terms(face_voidage, particle_diameter[face_index], 0.5)
            left_pressure = pressures[left_index]
            pressures[right_index] = (
                left_pressure - face_width * (pressure_term + density_term) / left_pressure
            ) / (1.0 + face_width * density_term / left_pressure**2)
            if pressures[right_index] <= 0.0:
                raise ValueError(
                    "Computed a non-positive interior pressure during initialization."
                )
        return pressures

    def outlet_residual(inlet_pressure):
        pressures = pressure_profile(inlet_pressure)
        pressure_term, density_term = ergun_terms(interparticle_voidage[-1], particle_diameter[-1], 1.0)
        predicted_outlet = pressures[-1] - outlet_half_cell_width * (
            pressure_term + density_term
        ) / pressures[-1]
        return predicted_outlet - outlet_pressure

    lower_inlet_pressure = outlet_pressure * (1.0 + 1.0e-8)
    upper_inlet_pressure = max(lower_inlet_pressure * 1.05, lower_inlet_pressure + 100.0)
    for _ in range(80):
        upper_residual = outlet_residual(upper_inlet_pressure)
        if not np.isfinite(upper_residual):
            raise ValueError("Unable to bracket the inlet pressure during initialization.")
        if upper_residual > 0.0:
            break
        upper_inlet_pressure *= 1.5
    else:
        raise ValueError("Unable to bracket the inlet pressure during initialization.")

    for _ in range(80):
        midpoint = 0.5 * (lower_inlet_pressure + upper_inlet_pressure)
        midpoint_residual = outlet_residual(midpoint)
        if not np.isfinite(midpoint_residual):
            raise ValueError("Unable to solve for the inlet pressure during initialization.")
        if midpoint_residual > 0.0:
            upper_inlet_pressure = midpoint
        else:
            lower_inlet_pressure = midpoint

    inlet_pressure = 0.5 * (lower_inlet_pressure + upper_inlet_pressure)
    pressure = pressure_profile(inlet_pressure)
    gas_total_concentration = gas_fraction * pressure / (
        GAS_CONSTANT_J_PER_MOL_K * inlet_temperature
    )
    gas_concentration = inlet_composition[:, None] * gas_total_concentration[None, :]
    gas_density = pressure * mixture_molecular_weight / (
        GAS_CONSTANT_J_PER_MOL_K * inlet_temperature
    )
    cell_enthalpy = gas_concentration.T @ gas_enthalpy + solid_concentration.T @ solid_enthalpy
    cell_widths = np.diff(face_coordinates)
    bed_mass = area_m2 * np.sum(
        (gas_concentration.T @ gas_molecular_weights + solid_concentration.T @ solid_molecular_weights)
        * cell_widths
    )
    bed_heat = area_m2 * np.sum(cell_enthalpy * cell_widths)

    face_velocity = np.empty(cell_count + 1, dtype=float)
    flow_velocity_numerator = inlet_molar_flux * GAS_CONSTANT_J_PER_MOL_K * inlet_temperature
    face_velocity[0] = flow_velocity_numerator / inlet_pressure
    face_velocity[1:] = flow_velocity_numerator / pressure
    return InitialState(
        face_coordinates_m=face_coordinates,
        interparticle_voidage=interparticle_voidage,
        intraparticle_voidage=intraparticle_voidage,
        particle_diameter_m=particle_diameter,
        gas_fraction=gas_fraction,
        inlet_flow_mol_s=inlet_flow,
        inlet_molar_flux_mol_m2_s=inlet_molar_flux,
        inlet_composition=inlet_composition,
        inlet_temperature_k=inlet_temperature,
        outlet_pressure_pa=outlet_pressure,
        inlet_pressure_pa=inlet_pressure,
        gas_concentration_mol_m3=gas_concentration,
        solid_concentration_mol_m3=solid_concentration,
        gas_enthalpy_j_mol=gas_enthalpy,
        solid_enthalpy_j_mol=solid_enthalpy,
        cell_enthalpy_j_m3=cell_enthalpy,
        gas_density_kg_m3=gas_density,
        gas_viscosity_pa_s=mixture_viscosity,
        face_velocity_m_s=face_velocity,
        bed_mass_kg=float(bed_mass),
        bed_heat_j=float(bed_heat),
    )


def configure_model(model, case: Case, state: InitialState) -> None:
    """Apply fixed parameters and domains calculated for an initial state."""

    from pyUnits import J, K, Pa, m, mol, s

    model.R_gas.SetValue(GAS_CONSTANT_J_PER_MOL_K * (Pa * m**3) / (mol * K))
    model.R_bed.SetValue(case.run.model.bed_radius_m * m)
    model.T_env.SetValue(case.run.model.ambient_temperature_k * K)
    model.U_eff.SetValue(
        case.run.model.heat_transfer_coefficient_w_per_m2_k * J / (K * s * m**2)
    )
    model.T_in_const.SetValue(float(case.inlet_temperature_program.initial_value) * K)
    model.P_out_const.SetValue(float(case.outlet_pressure_program.initial_value) * Pa)
    model.F_in_const.SetValue(float(case.inlet_flow_program.initial_value) * mol / s)
    model.y_in_const.SetValues(np.asarray(case.inlet_composition_program.initial_value))
    model.set_axial_grid_from_faces(state.face_coordinates_m)
    model.e_b.SetValues(state.interparticle_voidage)
    model.e_p.SetValues(state.intraparticle_voidage)
    model.d_p.SetValues(state.particle_diameter_m)
    model.gasfrac.SetValues(state.gas_fraction)


def apply_initial_state(model, state: InitialState) -> None:
    """Apply one calculated state to the DAETools variables."""

    from pyUnits import J, K, Pa, kg, m, mol, s

    gas_count, cell_count = state.gas_concentration_mol_m3.shape
    solid_count = state.solid_concentration_mol_m3.shape[0]
    gas_total = state.gas_concentration_mol_m3.sum(axis=0)
    solid_total = state.solid_concentration_mol_m3.sum(axis=0)
    pressure = (
        gas_total * GAS_CONSTANT_J_PER_MOL_K * state.inlet_temperature_k / state.gas_fraction
    )

    for cell_index in range(cell_count):
        model.T.SetInitialGuess(cell_index, state.inlet_temperature_k * K)
        model.P.SetInitialGuess(cell_index, pressure[cell_index] * Pa)
        model.mu_g.SetInitialGuess(cell_index, state.gas_viscosity_pa_s * Pa * s)
        model.rho_g.SetInitialGuess(
            cell_index, state.gas_density_kg_m3[cell_index] * (Pa * s**2) / m**2
        )
        model.ct_gas.SetInitialGuess(cell_index, gas_total[cell_index] * mol / m**3)
        model.ct_sol.SetInitialGuess(cell_index, solid_total[cell_index] * mol / m**3)
        model.h_cell.SetInitialCondition(cell_index, state.cell_enthalpy_j_m3[cell_index] * J / m**3)

    for gas_index in range(gas_count):
        model.y_in.SetInitialGuess(gas_index, state.inlet_composition[gas_index])
        for cell_index in range(cell_count):
            model.c_gas.SetInitialCondition(
                gas_index, cell_index, state.gas_concentration_mol_m3[gas_index, cell_index] * mol / m**3
            )
            model.y_gas.SetInitialGuess(gas_index, cell_index, state.inlet_composition[gas_index])
            model.h_gas.SetInitialGuess(
                gas_index, cell_index, state.gas_enthalpy_j_mol[gas_index] * J / mol
            )

    for solid_index in range(solid_count):
        for cell_index in range(cell_count):
            concentration = state.solid_concentration_mol_m3[solid_index, cell_index]
            model.c_sol.SetInitialCondition(
                solid_index, cell_index, concentration * mol / m**3
            )
            model.h_sol.SetInitialGuess(
                solid_index, cell_index, state.solid_enthalpy_j_mol[solid_index] * J / mol
            )

    model.F_in.SetInitialGuess(state.inlet_flow_mol_s * mol / s)
    model.T_in.SetInitialGuess(state.inlet_temperature_k * K)
    model.P_in.SetInitialGuess(state.inlet_pressure_pa * Pa)
    model.P_out.SetInitialGuess(state.outlet_pressure_pa * Pa)
    if model.mass_in_total is not None:
        model.mass_in_total.SetInitialCondition(0.0 * kg)
        model.mass_out_total.SetInitialCondition(0.0 * kg)
        model.mass_bed_total.SetInitialGuess(state.bed_mass_kg * kg)
    if model.heat_in_total is not None:
        model.heat_in_total.SetInitialCondition(0.0 * J)
        model.heat_out_total.SetInitialCondition(0.0 * J)
        model.heat_loss_total.SetInitialCondition(0.0 * J)
        model.heat_bed_total.SetInitialGuess(state.bed_heat_j * J)

    axial_dispersion = 0.5 * np.abs(state.face_velocity_m_s) * state.particle_diameter_m
    species_flux = state.inlet_composition * state.inlet_molar_flux_mol_m2_s
    for face_index, velocity in enumerate(state.face_velocity_m_s):
        model.u_s.SetInitialGuess(face_index, velocity * m / s)
        model.Dax.SetInitialGuess(face_index, axial_dispersion[face_index] * m**2 / s)
        for gas_index in range(gas_count):
            model.N_gas_face.SetInitialGuess(
                gas_index, face_index, species_flux[gas_index] * mol / (s * m**2)
            )
            model.J_gas_face.SetInitialGuess(
                gas_index,
                face_index,
                species_flux[gas_index]
                * state.gas_enthalpy_j_mol[gas_index]
                * J
                / (s * m**2),
            )

    if model.R_rxn is not None:
        for reaction_index in range(model.N_rxn.NumberOfPoints):
            for cell_index in range(cell_count):
                model.R_rxn.SetInitialGuess(
                    reaction_index, cell_index, 0.0 * mol / (m**3 * s)
                )


__all__ = (
    "InitialState",
    "apply_initial_state",
    "calculate_initial_state",
    "configure_model",
)
