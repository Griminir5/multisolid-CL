__doc__ = """
This file is means as the very first step towards a model for chemical looping in packed-bed reactors.
This file implements a gas/solid bed skeleton with EOS/Ergun pressure closure and a basic heat balance.
This variant wires inlet flow, inlet composition, inlet temperature, and outlet pressure through a native DAETOOLS operation program.
"""

import sys

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from daetools.pyDAE import *

from .axial_schemes import reconstruct_face_states, validate_scheme_name
from .config import ModelConfig, RunBundle, SolidConfig, SolidZoneConfig
from .properties import DEFAULT_PROPERTY_REGISTRY
from .reactions import DEFAULT_REACTION_CATALOG
from .solid_profiles import (
    build_cell_scalar_profile,
    build_face_scalar_profile,
    build_solid_profile_matrix as _shared_build_solid_profile_matrix,
    convert_solid_profile_to_bed_volume as _shared_convert_solid_profile_to_bed_volume,
    gas_fraction_from_voidages,
    solid_fraction_from_voidages,
)
from pyUnits import kg, J, K, Pa, m, mol, s  # this will not show up because pylance cannot get to .pyd files


molar_flux_type =       daeVariableType(name="molar_flux_type", units=mol / (s * m**2), 
                                        lowerBound=-100000, upperBound=100000, initialGuess=0, absTolerance=1e-5,)
molar_flow_type =       daeVariableType(name="molar_flow_type", units=mol / s,
                                        lowerBound=-1000, upperBound=1000, initialGuess=0, absTolerance=1e-5,)
molar_conc_type =       daeVariableType( name="molar_conc_type", units=mol / m**3,
                                        lowerBound=0, upperBound=100000, initialGuess=0, absTolerance=1e-5,)
molar_conc_sol_type =   daeVariableType(name="molar_conc_sol_type", units=mol / m**3,
                                        lowerBound=0, upperBound=1000000, initialGuess=0, absTolerance=1e-5,)
molar_frac_type =       daeVariableType(name="molar_frac_type", units=dimless,
                                        lowerBound=-0.1, upperBound=1.1, initialGuess=0, absTolerance=1e-5,)
dispersion_type =       daeVariableType(name="dispersion_type", units=m**2 / s,
                                        lowerBound=0, upperBound=100, initialGuess=0, absTolerance=1e-5,)
molar_source_type =     daeVariableType(name="molar_source_type", units=mol / (m**3 * s),
                                        lowerBound=-1000000, upperBound=1000000, initialGuess=0, absTolerance=1e-5,)

temp_type =             daeVariableType(name="temp_type", units=K,
                                        lowerBound=100, upperBound=2000, initialGuess=500, absTolerance=1e-5,)
molar_enthalpy_type =   daeVariableType(name="molar_enthalpy_type", units=J / mol,
                                        lowerBound=-1e12, upperBound=1e12, initialGuess=0, absTolerance=1e-5,)
volum_enthaply_type =   daeVariableType(name="volum_enthaply_type", units= J / m**3, 
                                        lowerBound=-1e12, upperBound=1e12, initialGuess=0, absTolerance=1e-5,)
heat_flux_type =        daeVariableType(name="heat_flux_type", units=J / (s * m**2),
                                        lowerBound=-1e12, upperBound=1e12, initialGuess=0, absTolerance=1e-5,)
molar_inventory_type =  daeVariableType(name="molar_inventory_type", units=mol,
                                        lowerBound=-1e12, upperBound=1e12, initialGuess=0, absTolerance=1e-5,)
energy_inventory_type = daeVariableType(name="energy_inventory_type", units=J,
                                        lowerBound=-1e20, upperBound=1e20, initialGuess=0, absTolerance=1e-2,)
viscosity_type =        daeVariableType(name="viscosity_type", units=Pa * s,
                                        lowerBound=0, upperBound=1, initialGuess=1e-5, absTolerance=1e-8,)
density_type =          daeVariableType(name="density_type", units=kg / m**3,
                                        lowerBound=0, upperBound=1e4, initialGuess=1, absTolerance=1e-5,)
molecular_weight_type = daeVariableType(name="molecular_weight_type", units=kg / mol,
                                        lowerBound=0, upperBound=1, initialGuess=0.03, absTolerance=1e-8,)


pres_type =             daeVariableType(name="pres_type", units=Pa,
                                        lowerBound=1e-3, upperBound=1e7, initialGuess=1e5, absTolerance=1e-5,)
velocity_type =         daeVariableType(name="velocity_type", units=m / s,
                                        lowerBound=-100, upperBound=100, initialGuess=1, absTolerance=1e-5,)

fraction_type =         daeVariableType(name="fraction_type", units=dimless,
                                        lowerBound=-0.1, upperBound=1.1, initialGuess=0, absTolerance=1e-5,)


def _default_model_config():
    return ModelConfig()


def _default_solid_config(solid_species, bed_length_m):
    initial_solids = {species_id: 0.0 for species_id in solid_species}
    if solid_species:
        initial_solids[solid_species[0]] = 100000.0
    return SolidConfig(
        solid_species=tuple(solid_species),
        concentration_unit="mol_per_m3_bed",
        initial_profile_zones=(
            SolidZoneConfig(
                x_start_m=0.0,
                x_end_m=float(bed_length_m),
                values_mol_per_m3=initial_solids,
                e_b=0.5,
                e_p=0.5,
                d_p=0.01,
            ),
        ),
    )


def _build_solid_profile_matrix(solids_config: SolidConfig, cell_centers_m, solid_species):
    return _shared_build_solid_profile_matrix(solids_config, cell_centers_m, solid_species)


def _convert_solid_profile_to_bed_volume(solids_config: SolidConfig, cell_centers_m, solid_fraction, solid_species):
    return _shared_convert_solid_profile_to_bed_volume(
        solids_config,
        cell_centers_m,
        solid_fraction,
        solid_species,
    )


@dataclass(frozen=True)
class ProgramStep:
    duration: float
    kind: Literal["hold", "ramp"]
    target: float | np.ndarray | None = None


@dataclass(frozen=True)
class ProgramSegment:
    start_time: float
    end_time: float
    start_value: float | np.ndarray
    end_value: float | np.ndarray


def _coerce_inlet_composition(value, expected_size=None, label="Inlet composition"):
    composition = np.asarray(value, dtype=float)

    if composition.ndim != 1:
        raise ValueError(f"{label} must be provided as a 1D vector.")
    if expected_size is not None and composition.size != expected_size:
        raise ValueError(f"{label} must contain exactly {expected_size} entries.")
    if composition.size == 0:
        raise ValueError(f"{label} must contain at least one entry.")
    if not np.all(np.isfinite(composition)):
        raise ValueError(f"{label} must contain only finite values.")
    if np.any(composition < -1e-12) or np.any(composition > 1.0 + 1e-12):
        raise ValueError(f"{label} entries must stay within [0, 1].")
    if not np.isclose(composition.sum(), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"{label} must sum to 1.")

    return composition.copy()


def _default_inlet_composition(gas_species):
    inlet_y = np.zeros(len(gas_species), dtype=float)
    inlet_y[0] = 1.0
    return inlet_y


class ScalarProgram:
    """Simple finite hold/ramp program compiled into DAETOOLS piecewise segments."""

    def __init__(self, initial_value):
        self.initial_value = float(initial_value)
        self.steps: list[ProgramStep] = []

    def hold(self, duration):
        self.steps.append(ProgramStep(duration=float(duration), kind="hold"))
        return self

    def ramp(self, duration, target):
        self.steps.append(ProgramStep(duration=float(duration), kind="ramp", target=float(target)))
        return self

    def build_segments(self, time_horizon=None):
        segments = []
        current_time = 0.0
        current_value = self.initial_value

        for step in self.steps:
            if step.duration <= 0.0:
                raise ValueError("Program step durations must be positive.")

            next_time = current_time + step.duration
            next_value = current_value if step.kind == "hold" else step.target
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
    """Finite hold/ramp program for vector-valued inlet conditions such as composition."""

    def __init__(self, initial_value):
        self.initial_value = _coerce_inlet_composition(initial_value)
        self.steps: list[ProgramStep] = []

    def hold(self, duration):
        self.steps.append(ProgramStep(duration=float(duration), kind="hold"))
        return self

    def ramp(self, duration, target):
        self.steps.append(
            ProgramStep(
                duration=float(duration),
                kind="ramp",
                target=_coerce_inlet_composition(
                    target,
                    expected_size=self.initial_value.size,
                    label="Inlet composition program target",
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
            next_value = current_value.copy() if step.kind == "hold" else np.asarray(step.target, dtype=float).copy()
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


# Wired operating program. The initial values match the steady-state
# initialization point; later segments are native DAETOOLS IF/ELSE branches.
INLET_FLOW_PROGRAM = (
    ScalarProgram(initial_value=0.785)
    .hold(100.0)
    .ramp(100.0, 1.785)
    .hold(2000.0)

)
INLET_TEMPERATURE_PROGRAM = (
    ScalarProgram(initial_value=500.0)
    .hold(250.0)
    .ramp(250.0, 1500)

)
OUTLET_PRESSURE_PROGRAM = (
    ScalarProgram(initial_value=1.01325e5)
    .hold(500)
    .ramp(1500.0, 90e5)
)


class CLBed_mass(daeModel):
    def __init__(
        self,
        Name,
        gas_species,
        solid_species,
        property_registry=DEFAULT_PROPERTY_REGISTRY,
        mass_scheme="weno3",
        heat_scheme="weno3",
        Parent=None,
        Description="Gas/solid mass balance-only bed",
    ):
        daeModel.__init__(self, Name, Parent, Description)

        self.gas_species = list(gas_species)
        self.solid_species = list(solid_species)
        self.property_registry = property_registry
        self.mass_scheme = validate_scheme_name(mass_scheme)
        self.heat_scheme = validate_scheme_name(mass_scheme if heat_scheme is None else heat_scheme)
        self.inlet_flow_segments = []
        self.inlet_composition_segments = []
        self.inlet_temperature_segments = []
        self.outlet_pressure_segments = []

        self.R_gas = daeParameter("R_gas", (Pa * m**3) / (mol * K), self, "Gas constant")
        self.pi = daeParameter("&pi;", dimless, self, "Circle constant")

        self.L_bed = daeParameter("Bed_length", m, self, "Length of the reactor bed")
        self.R_bed = daeParameter("Bed_radius", m, self, "Radius of the reactor bed")

        self.x_centers = daeDomain("Cell_centers", self, m, "Axial cell centers domain over the packed bed")
        self.x_faces = daeDomain("Cell_faces", self, m, "Axial cell faces domain over the packed bed")
        self.N_gas = daeDomain("Gas_comps", self, dimless, "Number of gaseous components")
        self.N_sol = daeDomain("Solid_comps", self, dimless, "Number of solid components")
        self.N_gas.CreateArray(len(self.gas_species))
        self.N_sol.CreateArray(len(self.solid_species))

        self.d_p = daeParameter("Particle_length", m, self, "Characteristic length of the solid particles", [self.x_faces])
        self.e_b = daeParameter("Interparticle_voidage", dimless, self, "Interparticle (between particles) voidage", [self.x_centers])
        self.e_p = daeParameter("Intraparticle_voidage", dimless, self, "Intraparticle (within particles) voidage", [self.x_centers])

        self.xval_cells = daeParameter("xval_cells", m, self, "Coordinate of cell centers")
        self.xval_faces = daeParameter("xval_faces", m, self, "Coordinate of cell faces")

        self.xval_cells.DistributeOnDomain(self.x_centers)
        self.xval_faces.DistributeOnDomain(self.x_faces)

        self.gasfrac = daeVariable("gasfrac", fraction_type, self, "Fraction of total bed volume occupied by gas", [self.x_centers])
        self.solfrac = daeVariable("solfrac", fraction_type, self, "Fraction of total bed volume occupied by solid", [self.x_centers])

        self.c_gas = daeVariable("c_gas", molar_conc_type, self, "Concentration of gaseous component i per total bed volume", [self.N_gas, self.x_centers])
        self.c_sol = daeVariable("c_sol", molar_conc_sol_type, self, "Concentration of solid component i per total bed volume", [self.N_sol, self.x_centers])
        self.ct_gas = daeVariable("c_gas_tot", molar_conc_type, self, "Total concentration of gas per total bed volume", [self.x_centers])
        self.ct_sol = daeVariable("c_sol_tot", molar_conc_sol_type, self, "Total concentration of solid per total bed volume", [self.x_centers])
        self.y_gas = daeVariable("y_gas", molar_frac_type, self, "Molar fraction of gaseous component i", [self.N_gas, self.x_centers])
        self.y_sol = daeVariable("y_sol", molar_frac_type, self, "Molar fraction of solid component i", [self.N_sol, self.x_centers])
        self.N_gas_face = daeVariable("N_gas_face", molar_flux_type, self, "Species i molar flux at cell faces", [self.N_gas, self.x_faces])

        self.S_sol = daeVariable("S_sol", molar_source_type, self, "Net source of solid component i per total bed volume", [self.N_sol, self.x_centers])

        self.T = daeVariable("temp_bed", temp_type, self, "Temperature inside a cell", [self.x_centers])
        self.h_cell = daeVariable("h_cell", volum_enthaply_type, self, "Enthalpy per total bed volume", [self.x_centers])
        self.h_gas = daeVariable("h_gas", molar_enthalpy_type, self, "Moalr enthalpy of gas i in a cell", [self.N_gas, self.x_centers])
        self.h_sol = daeVariable("h_sol", molar_enthalpy_type, self, "Moalr enthalpy of solid i in a cell", [self.N_sol, self.x_centers])
        self.J_gas_face = daeVariable("J_gas_face", heat_flux_type, self, "Enthalpy flow at cell faces attributable to component i", [self.N_gas, self.x_faces])
        
        self.material_in_total = daeVariable("material_in_total", molar_inventory_type, self, "Cumulative gas-phase material that has entered the bed")
        self.material_out_total = daeVariable("material_out_total", molar_inventory_type, self, "Cumulative gas-phase material that has left the bed")
        self.material_bed_total = daeVariable("material_bed_total", molar_inventory_type, self, "Gas plus solid material currently residing in the bed")
        
        self.heat_in_total = daeVariable("heat_in_total", energy_inventory_type, self, "Cumulative gas-phase enthalpy that has entered the bed")
        self.heat_out_total = daeVariable("heat_out_total", energy_inventory_type, self, "Cumulative gas-phase enthalpy that has left the bed")
        self.heat_bed_total = daeVariable("heat_bed_total", energy_inventory_type, self, "Gas plus solid enthalpy currently residing in the bed")
        
        self.Dax = daeVariable("Dax", dispersion_type, self, "Face axial dispersion coefficient", [self.x_faces])
        self.u_s = daeVariable("u_s", velocity_type, self, "Face superficial velocity", [self.x_faces])
        self.P = daeVariable("pres_bed", pres_type, self, "Pressure inside a cell", [self.x_centers])
        self.mu_g = daeVariable("mu_g", viscosity_type, self, "Mole-averaged gas viscosity in a cell", [self.x_centers])
        self.rho_g = daeVariable("rho_g", density_type, self, "Gas density in a cell", [self.x_centers])
        
        self.F_in_const = daeParameter("F_in_const", molar_flow_type.Units, self, "Fixed total molar flow at the inlet")
        self.y_in_const = daeParameter("y_in_const", molar_frac_type.Units, self, "Fixed molar fraction of component i at the inlet", [self.N_gas])
        self.T_in_const = daeParameter("T_in_const", K, self, "Fixed temperature at the inlet")
        self.P_out_const = daeParameter("P_out_const", Pa, self, "Fixed pressure at the outlet")

        self.F_in = daeVariable("F_in", molar_flow_type, self, "Total molar flow at the inlet")
        self.y_in = daeVariable("y_in", molar_frac_type, self, "Molar fraction of component i at the inlet", [self.N_gas])
        self.T_in = daeVariable("T_in", temp_type, self, "Temperature at the inlet")
        self.P_in = daeVariable("P_in", pres_type, self, "Pressure at the inlet boundary")
        self.P_out = daeVariable("P_out", pres_type, self, "Pressure at the outlet boundary")

        self.mw_mix = daeVariable("mw_mix", molecular_weight_type, self, "Gas mixture molecular weight in a cell", [self.x_centers])

    def SetAxialGridFromFaces(self, face_locations):
        face_locations = np.asarray(face_locations, dtype=float)

        if face_locations.ndim != 1:
            raise ValueError("Face locations must be provided as a 1D sequence.")
        if face_locations.size < 2:
            raise ValueError("At least two face locations are required.")
        if not np.all(np.diff(face_locations) > 0.0):
            raise ValueError("Face locations must be strictly increasing.")
        if not np.isclose(face_locations[0], 0.0):
            raise ValueError("The first face must be located at x = 0.")

        bed_length = self.L_bed.GetValue()
        if bed_length <= 0.0:
            raise ValueError("Set Bed_length before constructing the axial grid.")
        if not np.isclose(face_locations[-1], bed_length):
            raise ValueError("The last face must be located at x = L_bed.")

        center_locations = 0.5 * (face_locations[:-1] + face_locations[1:])

        self.x_faces.CreateStructuredGrid(face_locations.size - 1, 0, 1)
        self.x_centers.CreateStructuredGrid(center_locations.size - 1, 0, 1)

        self.x_faces.Points = face_locations.tolist()
        self.x_centers.Points = center_locations.tolist()
        self.xval_faces.SetValues(face_locations)
        self.xval_cells.SetValues(center_locations)

    def SetUniformAxialGrid(self, n_cells):
        if n_cells < 1:
            raise ValueError("The bed must contain at least one cell.")

        face_locations = np.linspace(0.0, self.L_bed.GetValue(), n_cells + 1)
        self.SetAxialGridFromFaces(face_locations)

    def SetOperationProgram(
        self,
        inlet_flow_program=None,
        inlet_composition_program=None,
        inlet_temperature_program=None,
        outlet_pressure_program=None,
        time_horizon=None,
    ):
        self.inlet_flow_segments = (
            [] if inlet_flow_program is None else inlet_flow_program.build_segments(time_horizon=time_horizon)
        )
        if inlet_composition_program is None:
            self.inlet_composition_segments = []
        else:
            inlet_composition = _coerce_inlet_composition(
                inlet_composition_program.initial_value,
                expected_size=len(self.gas_species),
                label="Inlet composition program initial value",
            )
            vector_segments = inlet_composition_program.build_segments(time_horizon=time_horizon)
            self.inlet_composition_segments = [
                [
                    ProgramSegment(
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        start_value=float(segment.start_value[gas_idx]),
                        end_value=float(segment.end_value[gas_idx]),
                    )
                    for segment in vector_segments
                ]
                for gas_idx in range(inlet_composition.size)
            ]
        self.inlet_temperature_segments = (
            [] if inlet_temperature_program is None else inlet_temperature_program.build_segments(time_horizon=time_horizon)
        )
        self.outlet_pressure_segments = (
            [] if outlet_pressure_program is None else outlet_pressure_program.build_segments(time_horizon=time_horizon)
        )

    def _segment_expression(self, segment, units):
        start_value = Constant(segment.start_value * units)
        end_value = Constant(segment.end_value * units)

        if math.isclose(segment.start_value, segment.end_value, rel_tol=0.0, abs_tol=1e-12):
            return start_value

        return start_value + (end_value - start_value) * (
            (Time() - Constant(segment.start_time * s)) / Constant((segment.end_time - segment.start_time) * s)
        )

    def _declare_program_equations(self, variable, default_expression, segments, units, equation_prefix):
        if not segments:
            eq = self.CreateEquation(f"{equation_prefix}_default")
            eq.Residual = variable() - default_expression
            return

        first_segment = segments[0]
        self.IF(Time() < Constant(first_segment.end_time * s))
        eq = self.CreateEquation(f"{equation_prefix}_000")
        eq.Residual = variable() - self._segment_expression(first_segment, units)

        for index, segment in enumerate(segments[1:], start=1):
            self.ELSE_IF(Time() < Constant(segment.end_time * s))
            eq = self.CreateEquation(f"{equation_prefix}_{index:03d}")
            eq.Residual = variable() - self._segment_expression(segment, units)

        self.ELSE("Hold the final operation-program value after the last breakpoint.")
        eq = self.CreateEquation(f"{equation_prefix}_{len(segments):03d}")
        eq.Residual = variable() - Constant(segments[-1].end_value * units)
        self.END_IF()

    def _declare_indexed_program_equations(self, variable, default_accessor, indexed_segments, units, equation_prefix):
        if not indexed_segments:
            eq = self.CreateEquation(f"{equation_prefix}_default")
            idx = eq.DistributeOnDomain(self.N_gas, eClosedClosed, "i")
            eq.Residual = variable(idx) - default_accessor(idx)
            return

        for gas_idx, segments in enumerate(indexed_segments):
            if not segments:
                eq = self.CreateEquation(f"{equation_prefix}_{gas_idx:03d}_default")
                eq.Residual = variable(gas_idx) - default_accessor(gas_idx)
                continue

            first_segment = segments[0]
            self.IF(Time() < Constant(first_segment.end_time * s))
            eq = self.CreateEquation(f"{equation_prefix}_{gas_idx:03d}_000")
            eq.Residual = variable(gas_idx) - self._segment_expression(first_segment, units)

            for index, segment in enumerate(segments[1:], start=1):
                self.ELSE_IF(Time() < Constant(segment.end_time * s))
                eq = self.CreateEquation(f"{equation_prefix}_{gas_idx:03d}_{index:03d}")
                eq.Residual = variable(gas_idx) - self._segment_expression(segment, units)

            self.ELSE("Hold the final operation-program value after the last breakpoint.")
            eq = self.CreateEquation(f"{equation_prefix}_{gas_idx:03d}_{len(segments):03d}")
            eq.Residual = variable(gas_idx) - Constant(segments[-1].end_value * units)
            self.END_IF()

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        Nc = self.x_centers.NumberOfPoints
        Nf = self.x_faces.NumberOfPoints
        Ng = self.N_gas.NumberOfPoints
        Ns = self.N_sol.NumberOfPoints

        if Ng != len(self.gas_species):
            raise RuntimeError("Gas component domain size must match gas_species.")
        if Ns != len(self.solid_species):
            raise RuntimeError("Solid component domain size must match solid_species.")
        if Nf != Nc + 1:
            raise RuntimeError("The axial grid must have exactly one more face than cell center.")

        center_coords = [self.xval_cells(idx_cell) for idx_cell in range(Nc)]
        face_coords = [self.xval_faces(idx_face) for idx_face in range(Nf)]
        cross_section_area = self.pi() * self.R_bed() ** 2
        conc_eps = Constant(1e-8 * mol / m**3)
        enthalpy_eps = Constant(1e-8 * J / mol)

        self._declare_program_equations(
            self.F_in,
            self.F_in_const(),
            self.inlet_flow_segments,
            molar_flow_type.Units,
            "Active_inlet_flow",
        )

        self._declare_indexed_program_equations(
            self.y_in,
            self.y_in_const,
            self.inlet_composition_segments,
            molar_frac_type.Units,
            "Active_inlet_composition",
        )

        self._declare_program_equations(
            self.T_in,
            self.T_in_const(),
            self.inlet_temperature_segments,
            K,
            "Active_inlet_temperature",
        )

        self._declare_program_equations(
            self.P_out,
            self.P_out_const(),
            self.outlet_pressure_segments,
            Pa,
            "Active_outlet_pressure",
        )



        eq = self.CreateEquation("gas_bed_fraction")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.gasfrac(idx_cell) - self.e_b(idx_cell) - (1 - self.e_b(idx_cell)) * self.e_p(idx_cell)

        eq = self.CreateEquation("solid_bed_fraction")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = 1 - self.gasfrac(idx_cell) - self.solfrac(idx_cell)



        eq = self.CreateEquation("total_concentration_closure")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.ct_gas(idx_cell) - Sum(self.c_gas.array("*", idx_cell))

        eq = self.CreateEquation("molar_fraction_calc")
        idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, "i")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.y_gas(idx_gas, idx_cell) * self.ct_gas(idx_cell) - self.c_gas(idx_gas, idx_cell)

        eq = self.CreateEquation("gas_mixture_molecular_weight")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        mw_mix_expr = Constant(0 * (Pa * m * s**2) / mol)
        for gas_idx, species_name in enumerate(self.gas_species):
            mw_mix_expr = mw_mix_expr + self.y_gas(gas_idx, idx_cell) * Constant(
                self.property_registry.get_record(species_name).mw * (Pa * m * s**2) / mol
            )
        eq.Residual = self.mw_mix(idx_cell) - mw_mix_expr

        eq = self.CreateEquation("gas_mixture_viscosity")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        mu_mix_expr = Constant(0 * Pa * s)
        for gas_idx, species_name in enumerate(self.gas_species):
            mu_mix_expr = mu_mix_expr + self.y_gas(gas_idx, idx_cell) * self.property_registry.viscosity_expression(
                species_name,
                self.T(idx_cell),
            )
        eq.Residual = self.mu_g(idx_cell) - mu_mix_expr

        eq = self.CreateEquation("gas_equation_of_state")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.P(idx_cell) * self.gasfrac(idx_cell) - self.ct_gas(idx_cell) * self.R_gas() * self.T(idx_cell)

        eq = self.CreateEquation("gas_density_closure")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.rho_g(idx_cell) - self.P(idx_cell) * self.mw_mix(idx_cell) / (self.R_gas() * self.T(idx_cell))



        eq = self.CreateEquation("solid_total_concentration_closure")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.ct_sol(idx_cell) - Sum(self.c_sol.array("*", idx_cell))

        eq = self.CreateEquation("solid_molar_fraction_calc")
        idx_sol = eq.DistributeOnDomain(self.N_sol, eClosedClosed, "j")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.y_sol(idx_sol, idx_cell) * self.ct_sol(idx_cell) - self.c_sol(idx_sol, idx_cell)



        eq = self.CreateEquation("axial_dispersion_face")
        idx_face = eq.DistributeOnDomain(self.x_faces, eClosedClosed, "x_f")
        eq.Residual = self.Dax(idx_face) - Abs(self.u_s(idx_face)) * 0.5 * self.d_p(idx_face)

        zero_velocity = Constant(0 * m / s)
        zero_molar_flux = Constant(0 * mol / (s * m**2))
        for face_index in range(1, Nf - 1):
            idx_cell_L = face_index - 1
            idx_cell_R = face_index
            dx = center_coords[idx_cell_R] - center_coords[idx_cell_L]
            ct_L = self.ct_gas(idx_cell_L)
            ct_R = self.ct_gas(idx_cell_R)
            ct_face = 0.5 * (ct_L + ct_R)

            eq = self.CreateEquation(f"face_flux_{face_index}")
            idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, "i")

            uplus = Max(self.u_s(face_index), zero_velocity)
            uminus = Min(self.u_s(face_index), zero_velocity)
            c_face_L, c_face_R = reconstruct_face_states(
                lambda idx_cell: self.c_gas(idx_gas, idx_cell) / self.gasfrac(idx_cell),
                face_index,
                Nc,
                self.mass_scheme,
                conc_eps,
            )
            eq.Residual = (
                self.N_gas_face(idx_gas, face_index)
                - uplus * c_face_L
                - uminus * c_face_R
                + self.Dax(face_index)
                * ct_face
                * (self.y_gas(idx_gas, idx_cell_R) - self.y_gas(idx_gas, idx_cell_L))
                / dx
            )

            eq = self.CreateEquation(f"face_enthalpy_flux_{face_index}")
            idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, "i")
            nplus = Max(self.N_gas_face(idx_gas, face_index), zero_molar_flux)
            nminus = Min(self.N_gas_face(idx_gas, face_index), zero_molar_flux)
            h_face_L, h_face_R = reconstruct_face_states(
                lambda idx_cell: self.h_gas(idx_gas, idx_cell),
                face_index,
                Nc,
                self.heat_scheme,
                enthalpy_eps,
            )
            eq.Residual = (
                self.J_gas_face(idx_gas, face_index)
                - nplus * h_face_L
                - nminus * h_face_R
            )

        for idx_cell in range(Nc):
            dx = face_coords[idx_cell + 1] - face_coords[idx_cell]

            eq = self.CreateEquation(f"species_balance_cell_{idx_cell}")
            idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, "i")
            eq.Residual = dt(self.c_gas(idx_gas, idx_cell)) + (
                self.N_gas_face(idx_gas, idx_cell + 1) - self.N_gas_face(idx_gas, idx_cell)
            ) / dx

            eq = self.CreateEquation(f"solid_species_balance_cell_{idx_cell}")
            idx_sol = eq.DistributeOnDomain(self.N_sol, eClosedClosed, "j")
            eq.Residual = dt(self.c_sol(idx_sol, idx_cell)) - self.S_sol(idx_sol, idx_cell)

        eq = self.CreateEquation("rhs_boundary_flux")
        idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, "i")
        idx_face = eq.DistributeOnDomain(self.x_faces, eUpperBound, "x_L")
        eq.Residual = self.N_gas_face(idx_gas, idx_face) - self.u_s(idx_face) * self.c_gas(idx_gas, Nc - 1) / self.gasfrac(Nc - 1)

        eq = self.CreateEquation("lhs_boundary_flux")
        idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, "i")
        eq.Residual = self.y_in(idx_gas) * self.F_in() / cross_section_area - self.N_gas_face(idx_gas, 0)
        eq = self.CreateEquation("inlet_pressure_from_flow")
        eq.Residual = self.F_in() / cross_section_area - self.u_s(0) * self.P_in() / (self.R_gas() * self.T_in())

        dx = center_coords[0] - face_coords[0]
        e_b_face = self.e_b(0)
        ergun_drag = (
            150 * self.mu_g(0) * (1 - e_b_face) ** 2 / (e_b_face ** 3 * self.d_p(0) ** 2) * self.u_s(0)
            + 1.75 * self.rho_g(0) * (1 - e_b_face) / (e_b_face ** 3 * self.d_p(0)) * Abs(self.u_s(0)) * self.u_s(0)
        )
        eq = self.CreateEquation("ergun_face_0")
        eq.Residual = (self.P_in() - self.P(0)) / dx - ergun_drag

        for face_index in range(1, Nf - 1):
            idx_cell_L = face_index - 1
            idx_cell_R = face_index
            dx = center_coords[idx_cell_R] - center_coords[idx_cell_L]
            mu_face = 0.5 * (self.mu_g(idx_cell_L) + self.mu_g(idx_cell_R))
            rho_face = 0.5 * (self.rho_g(idx_cell_L) + self.rho_g(idx_cell_R))
            e_b_face = 0.5 * (self.e_b(idx_cell_L) + self.e_b(idx_cell_R))
            ergun_drag = (
                150 * mu_face * (1 - e_b_face) ** 2 / (e_b_face ** 3 * self.d_p(face_index) ** 2) * self.u_s(face_index)
                + 1.75 * rho_face * (1 - e_b_face) / (e_b_face ** 3 * self.d_p(face_index)) * Abs(self.u_s(face_index)) * self.u_s(face_index)
            )

            eq = self.CreateEquation(f"ergun_face_{face_index}")
            eq.Residual = (self.P(idx_cell_L) - self.P(idx_cell_R)) / dx - ergun_drag

        dx = face_coords[Nf - 1] - center_coords[Nc - 1]
        e_b_face = self.e_b(Nc - 1)
        ergun_drag = (
            150 * self.mu_g(Nc - 1) * (1 - e_b_face) ** 2 / (e_b_face ** 3 * self.d_p(Nf - 1) ** 2) * self.u_s(Nf - 1)
            + 1.75 * self.rho_g(Nc - 1) * (1 - e_b_face) / (e_b_face ** 3 * self.d_p(Nf - 1)) * Abs(self.u_s(Nf - 1)) * self.u_s(Nf - 1)
        )
        eq = self.CreateEquation(f"ergun_face_{Nf - 1}")
        eq.Residual = (self.P(Nc - 1) - self.P_out()) / dx - ergun_drag



        for gas_idx, species_name in enumerate(self.gas_species):
            eq = self.CreateEquation(f"gas_component_enthalpy_{species_name}")
            idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
            eq.Residual = self.h_gas(gas_idx, idx_cell) - self.property_registry.enthalpy_expression(
                species_name,
                self.T(idx_cell),
            )

        for sol_idx, species_name in enumerate(self.solid_species):
            eq = self.CreateEquation(f"solid_component_enthalpy_{species_name}")
            idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
            eq.Residual = self.h_sol(sol_idx, idx_cell) - self.property_registry.enthalpy_expression(
                species_name,
                self.T(idx_cell),
            )

        for gas_idx, species_name in enumerate(self.gas_species):
            eq = self.CreateEquation(f"lhs_boundary_enthalpy_flux_{species_name}")
            eq.Residual = (
                self.J_gas_face(gas_idx, 0)
                - self.N_gas_face(gas_idx, 0)
                * self.property_registry.enthalpy_expression(species_name, self.T_in())
            )

            eq = self.CreateEquation(f"rhs_boundary_enthalpy_flux_{species_name}")
            eq.Residual = (
                self.J_gas_face(gas_idx, Nf - 1)
                - self.N_gas_face(gas_idx, Nf - 1) * self.h_gas(gas_idx, Nc - 1)
            )



        eq = self.CreateEquation("total_cell_enthalpy")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.h_cell(idx_cell) - Sum(self.c_gas.array("*", idx_cell)*self.h_gas.array("*", idx_cell)) - Sum(self.c_sol.array("*", idx_cell)*self.h_sol.array("*", idx_cell))

        material_bed_total = Constant(0.0 * mol)
        heat_bed_total = Constant(0.0 * J)
        for idx_cell in range(Nc):
            dx = face_coords[idx_cell + 1] - face_coords[idx_cell]

            eq = self.CreateEquation(f"energy_balance_cell_{idx_cell}")
            eq.Residual = dt(self.h_cell(idx_cell)) + (
                Sum(self.J_gas_face.array("*", idx_cell + 1)) - Sum(self.J_gas_face.array("*", idx_cell))
            ) / dx

            material_bed_total = material_bed_total + cross_section_area * (self.ct_gas(idx_cell) + self.ct_sol(idx_cell)) * dx
            heat_bed_total = heat_bed_total + cross_section_area * self.h_cell(idx_cell) * dx

        eq = self.CreateEquation("material_in_total_accumulation")
        eq.Residual = dt(self.material_in_total()) - cross_section_area * Sum(self.N_gas_face.array("*", 0))

        eq = self.CreateEquation("material_out_total_accumulation")
        eq.Residual = dt(self.material_out_total()) - cross_section_area * Sum(self.N_gas_face.array("*", Nf - 1))

        eq = self.CreateEquation("material_bed_total_definition")
        eq.Residual = self.material_bed_total() - material_bed_total

        eq = self.CreateEquation("heat_in_total_accumulation")
        eq.Residual = dt(self.heat_in_total()) - cross_section_area * Sum(self.J_gas_face.array("*", 0))

        eq = self.CreateEquation("heat_out_total_accumulation")
        eq.Residual = dt(self.heat_out_total()) - cross_section_area * Sum(self.J_gas_face.array("*", Nf - 1))

        eq = self.CreateEquation("heat_bed_total_definition")
        eq.Residual = self.heat_bed_total() - heat_bed_total

        eq = self.CreateEquation("solid_source_term_placeholder")
        idx_sol = eq.DistributeOnDomain(self.N_sol, eClosedClosed, "j")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.S_sol(idx_sol, idx_cell)


class simBed(daeSimulation):
    def __init__(
        self,
        gas_species=None,
        solid_species=None,
        solid_config: SolidConfig | None = None,
        property_registry=DEFAULT_PROPERTY_REGISTRY,
        mass_scheme="weno3",
        heat_scheme="weno3",
        inlet_flow_program=INLET_FLOW_PROGRAM,
        inlet_composition_program=None,
        inlet_temperature_program=INLET_TEMPERATURE_PROGRAM,
        outlet_pressure_program=OUTLET_PRESSURE_PROGRAM,
        operation_time_horizon=30000.0,
        model_config: ModelConfig | None = None,
        system_name="MassTrsf",
    ):
        daeSimulation.__init__(self)

        self.property_registry = property_registry
        self.gas_species = list(self.property_registry.species_ids("gas") if gas_species is None else gas_species)
        self.solid_species = list(self.property_registry.species_ids("solid") if solid_species is None else solid_species)
        self.model_config = model_config if model_config is not None else _default_model_config()
        self.solid_config = (
            solid_config
            if solid_config is not None
            else _default_solid_config(self.solid_species, self.model_config.bed_length_m)
        )
        self.mass_scheme = validate_scheme_name(mass_scheme)
        self.heat_scheme = validate_scheme_name("central" if heat_scheme is None else heat_scheme)
        self.inlet_flow_program = inlet_flow_program
        self.inlet_composition_program = inlet_composition_program
        self.inlet_temperature_program = inlet_temperature_program
        self.outlet_pressure_program = outlet_pressure_program
        self.operation_time_horizon = operation_time_horizon
        self.system_name = system_name

        self.model = CLBed_mass(
            self.system_name,
            self.gas_species,
            self.solid_species,
            property_registry=self.property_registry,
            mass_scheme=self.mass_scheme,
            heat_scheme=self.heat_scheme,
        )
        self.model.SetOperationProgram(
            inlet_flow_program=self.inlet_flow_program,
            inlet_composition_program=self.inlet_composition_program,
            inlet_temperature_program=self.inlet_temperature_program,
            outlet_pressure_program=self.outlet_pressure_program,
            time_horizon=self.operation_time_horizon,
        )

    def SetUpParametersAndDomains(self):
        self.model.R_gas.SetValue(self.model_config.gas_constant * (Pa * m**3) / (mol * K))
        self.model.pi.SetValue(self.model_config.pi_value)
        self.model.L_bed.SetValue(self.model_config.bed_length_m * m)
        self.model.R_bed.SetValue(self.model_config.bed_radius_m * m)
        inlet_temperature = 500.0 if self.inlet_temperature_program is None else self.inlet_temperature_program.initial_value
        outlet_pressure = 1.01325e5 if self.outlet_pressure_program is None else self.outlet_pressure_program.initial_value
        inlet_flow = 0.785 if self.inlet_flow_program is None else self.inlet_flow_program.initial_value
        inlet_y = (
            _default_inlet_composition(self.gas_species)
            if self.inlet_composition_program is None
            else _coerce_inlet_composition(
                self.inlet_composition_program.initial_value,
                expected_size=len(self.gas_species),
                label="Inlet composition program initial value",
            )
        )

        self.model.T_in_const.SetValue(inlet_temperature * K)
        self.model.P_out_const.SetValue(outlet_pressure * Pa)

        self.model.F_in_const.SetValue(inlet_flow * mol / s)
        self.model.SetUniformAxialGrid(self.model_config.axial_cells)
        self.model.y_in_const.SetValues(inlet_y)

        center_coords = np.asarray(self.model.xval_cells.npyValues, dtype=float)
        face_coords = np.asarray(self.model.xval_faces.npyValues, dtype=float)
        self.model.e_b.SetValues(build_cell_scalar_profile(self.solid_config, center_coords, "e_b"))
        self.model.e_p.SetValues(build_cell_scalar_profile(self.solid_config, center_coords, "e_p"))
        self.model.d_p.SetValues(build_face_scalar_profile(self.solid_config, face_coords, "d_p"))

    def SetUpVariables(self):
        ng = self.model.N_gas.NumberOfPoints
        ns = self.model.N_sol.NumberOfPoints
        nc = self.model.x_centers.NumberOfPoints
        nf = self.model.x_faces.NumberOfPoints

        inlet_y = np.asarray(self.model.y_in_const.npyValues, dtype=float)
        area = self.model.pi.GetValue() * self.model.R_bed.GetValue() ** 2

        fin = self.model.F_in_const.GetValue()
        if fin <= 0.0:
            raise ValueError("Steady-state initialization currently assumes a positive inlet molar flow.")

        r_gas = self.model.R_gas.GetValue()
        outlet_pressure = self.model.P_out_const.GetValue()
        inlet_temperature = self.model.T_in_const.GetValue()

        center_coords = np.asarray(self.model.xval_cells.npyValues, dtype=float)
        face_coords = np.asarray(self.model.xval_faces.npyValues, dtype=float)
        e_b0 = np.asarray(self.model.e_b.npyValues, dtype=float)
        e_p0 = np.asarray(self.model.e_p.npyValues, dtype=float)
        d_p0 = np.asarray(self.model.d_p.npyValues, dtype=float)
        gasfrac0 = gas_fraction_from_voidages(e_b0, e_p0)
        solfrac0 = solid_fraction_from_voidages(e_b0, e_p0)
        c0_sol = _convert_solid_profile_to_bed_volume(
            self.solid_config,
            center_coords,
            solfrac0,
            self.solid_species,
        )
        ct0_sol = c0_sol.sum(axis=0)

        gas_mw = np.asarray(
            [self.model.property_registry.get_record(gas_name).mw for gas_name in self.gas_species],
            dtype=float,
        )
        gas_mu = np.asarray(
            [self.model.property_registry.viscosity_value(gas_name, inlet_temperature) for gas_name in self.gas_species],
            dtype=float,
        )
        gas_h0 = np.asarray(
            [self.model.property_registry.enthalpy_value(gas_name, inlet_temperature) for gas_name in self.gas_species],
            dtype=float,
        )
        solid_h0 = np.asarray(
            [self.model.property_registry.enthalpy_value(sol_name, inlet_temperature) for sol_name in self.solid_species],
            dtype=float,
        )

        mw_mix_scalar = float(inlet_y @ gas_mw)
        mu_mix_scalar = float(inlet_y @ gas_mu)
        molar_flux_in = fin / area

        def ergun_terms(e_b_value, d_p_value, rho_weight):
            alpha = 150.0 * mu_mix_scalar * (1.0 - e_b_value) ** 2 / (e_b_value ** 3 * d_p_value ** 2)
            beta = 1.75 * (1.0 - e_b_value) / (e_b_value ** 3 * d_p_value)
            a_term = alpha * molar_flux_in * r_gas * inlet_temperature
            b_term = rho_weight * beta * mw_mix_scalar * molar_flux_in ** 2 * r_gas * inlet_temperature
            return a_term, b_term

        dx_in = center_coords[0] - face_coords[0]
        dx_out = face_coords[-1] - center_coords[-1]

        def pressure_profile_from_inlet(pin):
            pressures = np.zeros(nc, dtype=float)

            a_term, b_term = ergun_terms(e_b0[0], d_p0[0], 1.0)
            pressures[0] = (pin - dx_in * a_term / pin) / (1.0 + dx_in * b_term / pin ** 2)
            if pressures[0] <= 0.0:
                raise ValueError("Computed a non-positive first-cell pressure during steady-state initialization.")

            for face_idx in range(1, nf - 1):
                idx_left = face_idx - 1
                idx_right = face_idx
                dx_face = center_coords[idx_right] - center_coords[idx_left]
                e_b_face = 0.5 * (e_b0[idx_left] + e_b0[idx_right])
                a_term, b_term = ergun_terms(e_b_face, d_p0[face_idx], 0.5)
                p_left = pressures[idx_left]
                pressures[idx_right] = (p_left - dx_face * (a_term + b_term) / p_left) / (1.0 + dx_face * b_term / p_left ** 2)
                if pressures[idx_right] <= 0.0:
                    raise ValueError("Computed a non-positive interior pressure during steady-state initialization.")

            return pressures

        def outlet_pressure_residual(pin):
            pressures = pressure_profile_from_inlet(pin)
            a_term, b_term = ergun_terms(e_b0[-1], d_p0[-1], 1.0)
            predicted_outlet = pressures[-1] - dx_out * (a_term + b_term) / pressures[-1]
            return predicted_outlet - outlet_pressure

        lower_pin = outlet_pressure * (1.0 + 1e-8)
        upper_pin = max(lower_pin * 1.05, lower_pin + 100.0)
        upper_residual = outlet_pressure_residual(upper_pin)

        while upper_residual <= 0.0:
            upper_pin *= 1.5
            upper_residual = outlet_pressure_residual(upper_pin)

        for _ in range(80):
            mid_pin = 0.5 * (lower_pin + upper_pin)
            if outlet_pressure_residual(mid_pin) > 0.0:
                upper_pin = mid_pin
            else:
                lower_pin = mid_pin

        p_in0 = 0.5 * (lower_pin + upper_pin)
        p0 = pressure_profile_from_inlet(p_in0)
        ct0 = gasfrac0 * p0 / (r_gas * inlet_temperature)
        c0 = inlet_y[:, np.newaxis] * ct0[np.newaxis, :]
        mw_mix0 = np.zeros(nc, dtype=float) + mw_mix_scalar
        mu_mix0 = np.zeros(nc, dtype=float) + mu_mix_scalar
        rho0 = p0 * mw_mix0 / (r_gas * inlet_temperature)
        h_cell0 = c0.T @ gas_h0 + c0_sol.T @ solid_h0
        cell_widths = np.diff(face_coords)
        material_bed_total0 = area * np.sum((ct0 + ct0_sol) * cell_widths)
        heat_bed_total0 = area * np.sum(h_cell0 * cell_widths)

        face_velocity0 = np.empty(nf, dtype=float)
        face_velocity0[0] = molar_flux_in * r_gas * inlet_temperature / p_in0
        for face_idx in range(1, nf):
            face_velocity0[face_idx] = molar_flux_in * r_gas * inlet_temperature / p0[face_idx - 1]
        dax0 = 0.5 * np.abs(face_velocity0) * d_p0
        face_flux = inlet_y * molar_flux_in

        for cell_idx in range(nc):
            self.model.gasfrac.SetInitialGuess(cell_idx, gasfrac0[cell_idx])
            self.model.solfrac.SetInitialGuess(cell_idx, 1.0 - gasfrac0[cell_idx])
            self.model.T.SetInitialGuess(cell_idx, inlet_temperature * K)
            self.model.P.SetInitialGuess(cell_idx, p0[cell_idx] * Pa)
            self.model.mu_g.SetInitialGuess(cell_idx, mu_mix0[cell_idx] * Pa * s)
            self.model.rho_g.SetInitialGuess(cell_idx, rho0[cell_idx] * (Pa * s**2) / m**2)
            self.model.mw_mix.SetInitialGuess(cell_idx, mw_mix0[cell_idx] * (Pa * m * s**2) / mol)
            self.model.ct_gas.SetInitialGuess(cell_idx, ct0[cell_idx] * mol / m**3)
            self.model.ct_sol.SetInitialGuess(cell_idx, ct0_sol[cell_idx] * mol / m**3)
            self.model.h_cell.SetInitialCondition(cell_idx, h_cell0[cell_idx] * J / m**3)

        for gas_idx in range(ng):
            self.model.y_in.SetInitialGuess(gas_idx, inlet_y[gas_idx])
            for cell_idx in range(nc):
                self.model.c_gas.SetInitialCondition(gas_idx, cell_idx, c0[gas_idx, cell_idx] * mol / m**3)
                self.model.y_gas.SetInitialGuess(gas_idx, cell_idx, inlet_y[gas_idx])
                self.model.h_gas.SetInitialGuess(gas_idx, cell_idx, gas_h0[gas_idx] * J / mol)

        for sol_idx in range(ns):
            for cell_idx in range(nc):
                self.model.c_sol.SetInitialCondition(sol_idx, cell_idx, c0_sol[sol_idx, cell_idx] * mol / m**3)
                y0_sol = 0.0 if ct0_sol[cell_idx] <= 0.0 else c0_sol[sol_idx, cell_idx] / ct0_sol[cell_idx]
                self.model.y_sol.SetInitialGuess(sol_idx, cell_idx, y0_sol)
                self.model.S_sol.SetInitialGuess(sol_idx, cell_idx, 0.0 * mol / (m**3 * s))
                self.model.h_sol.SetInitialGuess(sol_idx, cell_idx, solid_h0[sol_idx] * J / mol)

        self.model.F_in.SetInitialGuess(fin * mol / s)
        self.model.T_in.SetInitialGuess(inlet_temperature * K)
        self.model.P_in.SetInitialGuess(p_in0 * Pa)
        self.model.P_out.SetInitialGuess(outlet_pressure * Pa)
        self.model.material_in_total.SetInitialCondition(0.0 * mol)
        self.model.material_out_total.SetInitialCondition(0.0 * mol)
        self.model.material_bed_total.SetInitialGuess(material_bed_total0 * mol)
        self.model.heat_in_total.SetInitialCondition(0.0 * J)
        self.model.heat_out_total.SetInitialCondition(0.0 * J)
        self.model.heat_bed_total.SetInitialGuess(heat_bed_total0 * J)

        for face_idx in range(nf):
            self.model.u_s.SetInitialGuess(face_idx, face_velocity0[face_idx] * m / s)
            self.model.Dax.SetInitialGuess(face_idx, dax0[face_idx] * m**2 / s)

            for gas_idx in range(ng):
                self.model.N_gas_face.SetInitialGuess(gas_idx, face_idx, face_flux[gas_idx] * mol / (s * m**2))
                self.model.J_gas_face.SetInitialGuess(gas_idx, face_idx, face_flux[gas_idx] * gas_h0[gas_idx] * J / (s * m**2))


@dataclass(frozen=True)
class SimulationAssembly:
    run_bundle: RunBundle
    simulation: simBed


def configure_evaluation_mode():
    cfg = daeGetConfig()
    cfg.SetString("daetools.core.equations.evaluationMode", "computeStack_OpenMP")


def build_idas_solver(relative_tolerance=1e-6):
    solver = daeIDAS()
    solver.RelativeTolerance = relative_tolerance
    return solver


def assemble_simulation(
    run_bundle: RunBundle,
    *,
    property_registry=DEFAULT_PROPERTY_REGISTRY,
    reaction_catalog=DEFAULT_REACTION_CATALOG,
) -> SimulationAssembly:
    unimplemented = [
        reaction_id
        for reaction_id in run_bundle.chemistry.reaction_ids
        if reaction_catalog[reaction_id].kinetics_hook is None
    ]
    if unimplemented:
        raise NotImplementedError(
            "Selected reactions do not have kinetics implementations: " + ", ".join(unimplemented)
        )

    inlet_flow_program = None if run_bundle.program.inlet_flow is None else run_bundle.program.inlet_flow.compile_program()
    inlet_composition_program = (
        None
        if run_bundle.program.inlet_composition is None
        else run_bundle.program.inlet_composition.compile_program(run_bundle.chemistry.gas_species)
    )
    inlet_temperature_program = (
        None if run_bundle.program.inlet_temperature is None else run_bundle.program.inlet_temperature.compile_program()
    )
    outlet_pressure_program = (
        None if run_bundle.program.outlet_pressure is None else run_bundle.program.outlet_pressure.compile_program()
    )

    simulation = simBed(
        gas_species=run_bundle.chemistry.gas_species,
        solid_species=run_bundle.solids.solid_species,
        solid_config=run_bundle.solids,
        property_registry=property_registry,
        mass_scheme=run_bundle.run.mass_scheme,
        heat_scheme=run_bundle.run.heat_scheme,
        inlet_flow_program=inlet_flow_program,
        inlet_composition_program=inlet_composition_program,
        inlet_temperature_program=inlet_temperature_program,
        outlet_pressure_program=outlet_pressure_program,
        operation_time_horizon=run_bundle.run.time_horizon_s,
        model_config=run_bundle.run.model,
        system_name=run_bundle.run.system_name,
    )
    return SimulationAssembly(run_bundle=run_bundle, simulation=simulation)


def _set_reporting_on(simulation):
    simulation.model.SetReportingOn(True)


def run_assembled_simulation(assembly: SimulationAssembly):
    configure_evaluation_mode()
    simulation = assembly.simulation
    _set_reporting_on(simulation)
    simulation.ReportTimeDerivatives = assembly.run_bundle.run.report_time_derivatives
    simulation.ReportingInterval = assembly.run_bundle.run.reporting_interval_s
    simulation.TimeHorizon = assembly.run_bundle.run.time_horizon_s

    solver = build_idas_solver(relative_tolerance=assembly.run_bundle.run.solver.relative_tolerance)
    reporter = daeNoOpDataReporter()
    log = daePythonStdOutLog()
    log.PrintProgress = False

    simulation.Initialize(solver, reporter, log)
    simulation.SolveInitial()
    simulation.Run()
    return reporter


def guiRun(qtApp, mass_scheme="weno3", heat_scheme="weno3"):
    configure_evaluation_mode()

    simulation = simBed(mass_scheme=mass_scheme, heat_scheme=heat_scheme)
    simulation.model.SetReportingOn(True)
    simulation.ReportTimeDerivatives = False
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 10000
    simulator = daeSimulator(qtApp, simulation=simulation, daesolver=build_idas_solver())
    simulator.exec()


if __name__ == "__main__":
    qtApp = daeCreateQtApplication(sys.argv)
    guiRun(qtApp)
