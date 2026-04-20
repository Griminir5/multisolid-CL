__doc__ = """
This file is a general working model of a packed-bed chemical looping reactor. This defines the basic equations
and program control profiles. Reactions are added in external submodules and are accessed via registered hooks.
"""

import sys

import math
from dataclasses import dataclass

import numpy as np
from daetools.pyDAE import *

from .axial_schemes import reconstruct_face_states
from .config import ModelConfig, ProgramSegment, RunBundle, ScalarProgram, SolidConfig, VectorProgram
from .kinetics import KineticsContext, resolve_kinetics_hooks
from .reporting import reporting_targets
from .reactions import ReactionNetwork, build_reaction_network
from .solid_profiles import (
    build_cell_scalar_profile,
    build_face_scalar_profile,
    build_solid_profile_matrix,
    convert_solid_profile_to_bed_volume,
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

SMOOTH_RAMP_WIDTH_S = 1.0


class CLBed_mass(daeModel):
    def __init__(
        self,
        Name,
        gas_species,
        solid_species,
        reaction_network: ReactionNetwork,
        reaction_rate_hooks,
        property_registry,
        mass_scheme,
        heat_scheme,
        materialize_solid_mole_fractions=False,
        smooth_ramp_width_s=SMOOTH_RAMP_WIDTH_S,
        Description="",
        Parent=None,
    ):
        daeModel.__init__(self, Name, Parent, Description)

        self.gas_species = list(gas_species)
        self.solid_species = list(solid_species)
        self.reaction_coupling = ReactionCoupling.from_network(
            reaction_network=reaction_network,
            reaction_rate_hooks=reaction_rate_hooks,
            gas_species=self.gas_species,
            solid_species=self.solid_species,
        )
        self.reaction_network = self.reaction_coupling.network
        self.reaction_rate_hooks = self.reaction_coupling.rate_hooks
        self.property_registry = property_registry
        self.mass_scheme = mass_scheme
        self.heat_scheme = heat_scheme
        self.materialize_solid_mole_fractions = bool(materialize_solid_mole_fractions)
        if smooth_ramp_width_s <= 0.0:
            raise ValueError("smooth_ramp_width_s must be positive.")
        self.smooth_ramp_width_s = float(smooth_ramp_width_s)
        self.inlet_flow_segments = []
        self.inlet_composition_segments = []
        self.inlet_temperature_segments = []
        self.outlet_pressure_segments = []
        self.gas_species_index = self.reaction_coupling.gas_species_index
        self.solid_species_index = self.reaction_coupling.solid_species_index
        self.reaction_index = self.reaction_coupling.reaction_index

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
        if self.reaction_coupling.has_reactions:
            self.N_rxn = daeDomain("Reactions", self, dimless, "Number of reactions")
            self.N_rxn.CreateArray(self.reaction_coupling.reaction_count)
        else:
            self.N_rxn = None

        self.d_p = daeParameter("Particle_length", m, self, "Characteristic length of the solid particles", [self.x_faces])
        self.e_b = daeParameter("Interparticle_voidage", dimless, self, "Interparticle (between particles) voidage", [self.x_centers])
        self.e_p = daeParameter("Intraparticle_voidage", dimless, self, "Intraparticle (within particles) voidage", [self.x_centers])

        self.xval_cells = daeParameter("xval_cells", m, self, "Coordinate of cell centers")
        self.xval_faces = daeParameter("xval_faces", m, self, "Coordinate of cell faces")

        self.xval_cells.DistributeOnDomain(self.x_centers)
        self.xval_faces.DistributeOnDomain(self.x_faces)

        self.gasfrac = daeParameter("gasfrac", dimless, self, "Fraction of total bed volume occupied by gas", [self.x_centers])
        self.solfrac = daeParameter("solfrac", dimless, self, "Fraction of total bed volume occupied by solid", [self.x_centers])

        self.c_gas = daeVariable("c_gas", molar_conc_type, self, "Concentration of gaseous component i per total bed volume", [self.N_gas, self.x_centers])
        self.c_sol = daeVariable("c_sol", molar_conc_sol_type, self, "Concentration of solid component i per total bed volume", [self.N_sol, self.x_centers])
        self.ct_gas = daeVariable("c_gas_tot", molar_conc_type, self, "Total concentration of gas per total bed volume", [self.x_centers])
        self.ct_sol = daeVariable("c_sol_tot", molar_conc_sol_type, self, "Total concentration of solid per total bed volume", [self.x_centers])
        self.y_gas = daeVariable("y_gas", molar_frac_type, self, "Molar fraction of gaseous component i", [self.N_gas, self.x_centers])
        self.y_sol = None
        if self.materialize_solid_mole_fractions:
            self.y_sol = daeVariable("y_sol", molar_frac_type, self, "Molar fraction of solid component i", [self.N_sol, self.x_centers])
        self.N_gas_face = daeVariable("N_gas_face", molar_flux_type, self, "Species i molar flux at cell faces", [self.N_gas, self.x_faces])

        self.R_rxn = None
        if self.reaction_coupling.has_reactions:
            self.R_rxn = daeVariable("R_rxn", molar_source_type, self, "Reaction rate of reaction k per total bed volume", [self.N_rxn, self.x_centers])
            

        self.T = daeVariable("temp_bed", temp_type, self, "Temperature inside a cell", [self.x_centers])
        self.h_cell = daeVariable("h_cell", volum_enthaply_type, self, "Enthalpy per total bed volume", [self.x_centers])
        self.h_gas = daeVariable("h_gas", molar_enthalpy_type, self, "Molar enthalpy of gas i in a cell", [self.N_gas, self.x_centers])
        self.h_sol = daeVariable("h_sol", molar_enthalpy_type, self, "Molar enthalpy of solid i in a cell", [self.N_sol, self.x_centers])
        self.J_gas_face = daeVariable("J_gas_face", heat_flux_type, self, "Enthalpy flow at cell faces attributable to component i", [self.N_gas, self.x_faces])
        
        self.heat_in_total = daeVariable("heat_in_total", energy_inventory_type, self, "Cumulative gas-phase enthalpy that has entered the bed")
        self.heat_out_total = daeVariable("heat_out_total", energy_inventory_type, self, "Cumulative gas-phase enthalpy that has left the bed")
        self.heat_bed_total = daeVariable("heat_bed_total", energy_inventory_type, self, "Gas plus solid enthalpy currently residing in the bed")
        
        self.Dax = daeVariable("Dax", dispersion_type, self, "Face axial dispersion coefficient", [self.x_faces])
        self.u_s = daeVariable("u_s", velocity_type, self, "Face superficial velocity", [self.x_faces])
        self.P = daeVariable("pres_bed", pres_type, self, "Pressure inside a cell", [self.x_centers])
        self.mu_g = daeVariable("mu_g", viscosity_type, self, "Mole-averaged gas viscosity in a cell", [self.x_centers])
        self.rho_g = daeVariable("rho_g", density_type, self, "Gas density in a cell", [self.x_centers])
        
        self.F_in_const = daeParameter("F_in_const", molar_flow_type.Units, self, "Default fixed total molar flow at the inlet")
        self.y_in_const = daeParameter("y_in_const", molar_frac_type.Units, self, "Default fixed molar fraction of component i at the inlet", [self.N_gas])
        self.T_in_const = daeParameter("T_in_const", K, self, "Default fixed temperature at the inlet")
        self.P_out_const = daeParameter("P_out_const", Pa, self, "Default fixed pressure at the outlet")

        self.F_in = daeVariable("F_in", molar_flow_type, self, "Total molar flow at the inlet")
        self.y_in = daeVariable("y_in", molar_frac_type, self, "Molar fraction of component i at the inlet", [self.N_gas])
        self.T_in = daeVariable("T_in", temp_type, self, "Temperature at the inlet")
        self.P_in = daeVariable("P_in", pres_type, self, "Pressure at the inlet boundary")
        self.P_out = daeVariable("P_out", pres_type, self, "Pressure at the outlet boundary")

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

    def _smooth_positive_time(self, elapsed_time):
        width = Constant(self.smooth_ramp_width_s * s)
        return Constant(0.5) * (elapsed_time + Sqrt(elapsed_time * elapsed_time + width * width))

    def _smooth_ramp_fraction(self, segment):
        duration_s = float(segment.end_time) - float(segment.start_time)

        start_time = Constant(float(segment.start_time) * s)
        end_time = Constant(float(segment.end_time) * s)
        return (
            self._smooth_positive_time(Time() - start_time)
            - self._smooth_positive_time(Time() - end_time)
        ) / Constant(duration_s * s)

    def _smooth_program_expression(self, default_expression, segments, units):
        if not segments:
            return default_expression

        expression = Constant(float(segments[0].start_value) * units)
        for segment in segments:
            delta = float(segment.end_value) - float(segment.start_value)
            if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
                continue
            expression = expression + Constant(delta * units) * self._smooth_ramp_fraction(segment)
        return expression

    def _declare_program_equations(self, variable, default_expression, segments, units, equation_prefix):
        eq = self.CreateEquation(f"{equation_prefix}_smooth")
        eq.Residual = variable() - self._smooth_program_expression(default_expression, segments, units)

    def _declare_indexed_program_equations(self, variable, default_accessor, indexed_segments, units, equation_prefix):
        for gas_idx, segments in enumerate(indexed_segments):
            eq = self.CreateEquation(f"{equation_prefix}_{gas_idx}_smooth")
            eq.Residual = variable(gas_idx) - self._smooth_program_expression(
                default_accessor(gas_idx),
                segments,
                units,
            )
    
    def _gas_mixture_molecular_weight_expression(self, idx_cell):
        mw_mix_expr = Constant(0.0 * kg / mol)
        for gas_idx, species_name in enumerate(self.gas_species):
            mw_mix_expr = mw_mix_expr + self.y_gas(gas_idx, idx_cell) * Constant(
                self.property_registry.get_record(species_name).mw * kg / mol
            )
        return mw_mix_expr
    
    def build_kinetics_context(self, idx_cell):
        return KineticsContext(
            model=self,
            idx_cell=idx_cell,
            gas_species_index=self.gas_species_index,
            solid_species_index=self.solid_species_index,
            reaction_index=self.reaction_index,
        )
    
    def _source_expression(self, coefficients, idx_cell):
        source = Constant(0.0 * mol / (m**3 * s))
        if self.R_rxn is None:
            return source

        for reaction_idx, coefficient in enumerate(coefficients):
            if coefficient != 0.0:
                source = source + Constant(coefficient) * self.R_rxn(reaction_idx, idx_cell)
        return source

    def _gas_source_expression(self, gas_idx, idx_cell):
        return self._source_expression(
            self.reaction_network.gas_source_matrix[gas_idx],
            idx_cell,
        )

    def _solid_source_expression(self, sol_idx, idx_cell):
        return self._source_expression(
            self.reaction_network.solid_source_matrix[sol_idx],
            idx_cell,
        )
    
    
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



        self._declare_program_equations(self.F_in, self.F_in_const(), self.inlet_flow_segments, molar_flow_type.Units, "Active_inlet_flow")

        self._declare_indexed_program_equations(self.y_in, self.y_in_const, self.inlet_composition_segments, molar_frac_type.Units, "Active_inlet_composition")

        self._declare_program_equations(self.T_in, self.T_in_const(), self.inlet_temperature_segments, K, "Active_inlet_temperature")

        self._declare_program_equations(self.P_out, self.P_out_const(), self.outlet_pressure_segments, Pa, "Active_outlet_pressure")



        eq = self.CreateEquation("total_concentration_closure")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.ct_gas(idx_cell) - Sum(self.c_gas.array("*", idx_cell))

        eq = self.CreateEquation("molar_fraction_calc")
        idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, "i")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.y_gas(idx_gas, idx_cell) * self.ct_gas(idx_cell) - self.c_gas(idx_gas, idx_cell)

        eq = self.CreateEquation("gas_equation_of_state")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.P(idx_cell) * self.gasfrac(idx_cell) - self.ct_gas(idx_cell) * self.R_gas() * self.T(idx_cell)



        eq = self.CreateEquation("gas_mixture_viscosity")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        mu_mix_expr = Constant(0 * Pa * s)
        for gas_idx, species_name in enumerate(self.gas_species):
            mu_mix_expr = mu_mix_expr + self.y_gas(gas_idx, idx_cell) * self.property_registry.viscosity_expression(species_name, self.T(idx_cell))
        eq.Residual = self.mu_g(idx_cell) - mu_mix_expr

        eq = self.CreateEquation("gas_density_closure")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.rho_g(idx_cell) - self.P(idx_cell) * self._gas_mixture_molecular_weight_expression(idx_cell) / (self.R_gas() * self.T(idx_cell))



        eq = self.CreateEquation("solid_total_concentration_closure")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.ct_sol(idx_cell) - Sum(self.c_sol.array("*", idx_cell))

        if self.y_sol is not None:
            eq = self.CreateEquation("solid_molar_fraction_calc")
            idx_sol = eq.DistributeOnDomain(self.N_sol, eClosedClosed, "j")
            idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
            eq.Residual = self.y_sol(idx_sol, idx_cell) * self.ct_sol(idx_cell) - self.c_sol(idx_sol, idx_cell)



        eq = self.CreateEquation("axial_dispersion_face")
        idx_face = eq.DistributeOnDomain(self.x_faces, eClosedClosed, "x_f")
        eq.Residual = self.Dax(idx_face) - Abs(self.u_s(idx_face)) * 0.5 * self.d_p(idx_face)


        for face_index in range(1, Nf - 1): # face enthalpy and material flux equations
            idx_cell_L = face_index - 1
            idx_cell_R = face_index
            dx = center_coords[idx_cell_R] - center_coords[idx_cell_L]
            ct_L = self.ct_gas(idx_cell_L)
            ct_R = self.ct_gas(idx_cell_R)
            ct_face = 0.5 * ((ct_L + ct_R)/self.gasfrac(idx_cell_L))

            eq = self.CreateEquation(f"face_flux_{face_index}")
            idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, "i")

            c_face_L, c_face_R = reconstruct_face_states(
                lambda idx_cell: self.c_gas(idx_gas, idx_cell) / self.gasfrac(idx_cell),
                face_index,
                Nc,
                self.mass_scheme,
                conc_eps,
            )
            eq.Residual = (
                self.N_gas_face(idx_gas, face_index)
                - self.u_s(face_index) * c_face_L
                + self.Dax(face_index)
                * ct_face
                * (self.y_gas(idx_gas, idx_cell_R) - self.y_gas(idx_gas, idx_cell_L))
                / dx
            )

            eq = self.CreateEquation(f"face_enthalpy_flux_{face_index}")
            idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, "i")

            h_face_L, h_face_R = reconstruct_face_states(
                lambda idx_cell: self.h_gas(idx_gas, idx_cell),
                face_index,
                Nc,
                self.heat_scheme,
                enthalpy_eps,
            )
            eq.Residual = (
                self.J_gas_face(idx_gas, face_index)
                - self.N_gas_face(idx_gas, face_index) * h_face_L
            )

        for idx_cell in range(Nc): # cell enthalpy and material balances
            dx = face_coords[idx_cell + 1] - face_coords[idx_cell]

            for gas_idx in range(Ng):
                eq = self.CreateEquation(f"species_balance_cell_{idx_cell}_{self.gas_species[gas_idx]}")
                eq.Residual = dt(self.c_gas(gas_idx, idx_cell)) + (
                    self.N_gas_face(gas_idx, idx_cell + 1) - self.N_gas_face(gas_idx, idx_cell)
                ) / dx - self._gas_source_expression(gas_idx, idx_cell)

            for sol_idx in range(Ns):
                eq = self.CreateEquation(f"solid_species_balance_cell_{idx_cell}_{self.solid_species[sol_idx]}")
                eq.Residual = dt(self.c_sol(sol_idx, idx_cell)) - self._solid_source_expression(sol_idx, idx_cell)


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


        heat_bed_total = Constant(0.0 * J)
        for idx_cell in range(Nc):
            dx = face_coords[idx_cell + 1] - face_coords[idx_cell]

            eq = self.CreateEquation(f"energy_balance_cell_{idx_cell}")
            eq.Residual = dt(self.h_cell(idx_cell)) + (
                Sum(self.J_gas_face.array("*", idx_cell + 1)) - Sum(self.J_gas_face.array("*", idx_cell))
            ) / dx

            heat_bed_total = heat_bed_total + cross_section_area * self.h_cell(idx_cell) * dx

        eq = self.CreateEquation("heat_in_total_accumulation")
        eq.Residual = dt(self.heat_in_total()) - cross_section_area * Sum(self.J_gas_face.array("*", 0))

        eq = self.CreateEquation("heat_out_total_accumulation")
        eq.Residual = dt(self.heat_out_total()) - cross_section_area * Sum(self.J_gas_face.array("*", Nf - 1))

        eq = self.CreateEquation("heat_bed_total_definition")
        eq.Residual = self.heat_bed_total() - heat_bed_total

        if self.reaction_network.has_reactions:
            for reaction_idx, reaction in enumerate(self.reaction_network.reactions):
                eq = self.CreateEquation(f"reaction_rate_{reaction.id}")
                idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
                kinetics_context = self.build_kinetics_context(idx_cell)
                eq.Residual = self.R_rxn(reaction_idx, idx_cell) - self.reaction_rate_hooks[reaction_idx](
                    kinetics_context
                )
