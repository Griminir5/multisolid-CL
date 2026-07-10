"""DAETools variables, parameters, domains, ports, and packed-bed equations."""

import math

import numpy as np
from daetools.pyDAE import (
    Abs, Constant, Max, Min, Sqrt, Sum, Time,
    daeDomain, daeModel, daeParameter, daeVariable, daeVariableType,
    dimless, dt, eClosedClosed,
)

from .axial_schemes import reconstruct_face_states, split_face_flux
from .config import Case
from .initialization import CIRCLE_CONSTANT
from .kinetics import KineticsContext
from .programs import DEFAULT_SMOOTH_RAMP_WIDTH_S
from .reactions import ReactionNetwork
from pyUnits import J, K, Pa, kg, m, mol, s

def _variable_type(name, units, lower, upper, initial=0.0, tolerance=1.0e-5):
    return daeVariableType(
        name=name,
        units=units,
        lowerBound=lower,
        upperBound=upper,
        initialGuess=initial,
        absTolerance=tolerance,
    )


molar_flux_type = _variable_type("molar_flux_type", mol / (s * m**2), -1.0e5, 1.0e5)
molar_flow_type = _variable_type("molar_flow_type", mol / s, -1.0e3, 1.0e3)
molar_conc_type = _variable_type("molar_conc_type", mol / m**3, 0.0, 1.0e5)
molar_conc_sol_type = _variable_type("molar_conc_sol_type", mol / m**3, 0.0, 1.0e6)
molar_frac_type = _variable_type("molar_frac_type", dimless, -0.1, 1.1)
dispersion_type = _variable_type("dispersion_type", m**2 / s, 0.0, 100.0)
molar_source_type = _variable_type("molar_source_type", mol / (m**3 * s), -1.0e6, 1.0e6)
temp_type = _variable_type("temp_type", K, 100.0, 2000.0, 500.0)
molar_enthalpy_type = _variable_type("molar_enthalpy_type", J / mol, -1.0e12, 1.0e12)
volume_enthalpy_type = _variable_type("volume_enthalpy_type", J / m**3, -1.0e12, 1.0e12)
heat_flux_type = _variable_type("heat_flux_type", J / (s * m**2), -1.0e12, 1.0e12)
mass_inventory_type = _variable_type("mass_inventory_type", kg, -1.0e12, 1.0e12, tolerance=1.0e-8)
energy_inventory_type = _variable_type("energy_inventory_type", J, -1.0e20, 1.0e20, tolerance=1.0e-2)
viscosity_type = _variable_type("viscosity_type", Pa * s, 0.0, 1.0, 1.0e-5, 1.0e-8)
density_type = _variable_type("density_type", kg / m**3, 0.0, 1.0e4, 1.0)
pres_type = _variable_type("pres_type", Pa, 1.0e-3, 1.0e7, 1.0e5)
velocity_type = _variable_type("velocity_type", m / s, -100.0, 100.0, 1.0)

class PackedBedModel(daeModel):
    def __init__(
        self,
        name,
        case: Case,
        reaction_network: ReactionNetwork,
        reaction_rate_hooks,
        property_registry,
        smooth_ramp_width_s=DEFAULT_SMOOTH_RAMP_WIDTH_S,
        description="",
        parent=None,
    ):
        daeModel.__init__(self, name, parent, description)

        self.gas_species = list(case.chemistry.gas_species)
        self.solid_species = list(case.solids.solid_species)
        self.reaction_network = reaction_network
        self.reaction_rate_hooks = tuple(reaction_rate_hooks)
        self.property_registry = property_registry
        self.mass_scheme = case.run.simulation.mass_scheme
        self.heat_scheme = case.run.simulation.heat_scheme
        requested_reports = set(case.run.outputs.requested_reports)
        if smooth_ramp_width_s <= 0.0:
            raise ValueError("smooth_ramp_width_s must be positive.")
        self.smooth_ramp_width_s = float(smooth_ramp_width_s)
        self.inlet_flow_program = case.inlet_flow_program
        self.inlet_composition_program = case.inlet_composition_program
        self.inlet_temperature_program = case.inlet_temperature_program
        self.outlet_pressure_program = case.outlet_pressure_program
        self.initial_inlet_composition = np.asarray(case.inlet_composition_program.initial_value)
        self.gas_species_index = {species_id: idx for idx, species_id in enumerate(self.gas_species)}
        self.solid_species_index = {species_id: idx for idx, species_id in enumerate(self.solid_species)}
        if self.reaction_network.has_reactions and len(self.reaction_rate_hooks) != self.reaction_network.reaction_count:
            raise ValueError("Reaction rate hooks must align one-to-one with the selected reaction network.")
        if not self.reaction_network.has_reactions and self.reaction_rate_hooks:
            raise ValueError("Reaction rate hooks were provided for a non-reactive simulation.")

        self.R_gas = daeParameter("R_gas", (Pa * m**3) / (mol * K), self, "Gas constant")
        self.R_bed = daeParameter("Bed_radius", m, self, "Radius of the reactor bed")

        self.x_centers = daeDomain("Cell_centers", self, m, "Axial cell centers domain over the packed bed")
        self.x_faces = daeDomain("Cell_faces", self, m, "Axial cell faces domain over the packed bed")
        self.N_gas = daeDomain("Gas_comps", self, dimless, "Number of gaseous components")
        self.N_sol = daeDomain("Solid_comps", self, dimless, "Number of solid components")
        self.N_gas.CreateArray(len(self.gas_species))
        self.N_sol.CreateArray(len(self.solid_species))
        if self.reaction_network.has_reactions:
            self.N_rxn = daeDomain("Reactions", self, dimless, "Number of reactions")
            self.N_rxn.CreateArray(self.reaction_network.reaction_count)
        else:
            self.N_rxn = None

        self.d_p = daeParameter("Particle_length", m, self, "Characteristic length of the solid particles", [self.x_faces])
        self.e_b = daeParameter("Interparticle_voidage", dimless, self, "Interparticle (between particles) voidage", [self.x_centers])
        self.e_p = daeParameter("Intraparticle_voidage", dimless, self, "Intraparticle (within particles) voidage", [self.x_centers])

        self.xval_cells = daeParameter("xval_cells", m, self, "Coordinate of cell centers")
        self.xval_cells.DistributeOnDomain(self.x_centers)

        self.gasfrac = daeParameter("gasfrac", dimless, self, "Fraction of total bed volume occupied by gas", [self.x_centers])

        self.N_gas_face = daeVariable("N_gas_face", molar_flux_type, self, "Species i molar flux at cell faces", [self.N_gas, self.x_faces])
        self.c_sol = daeVariable("c_sol", molar_conc_sol_type, self, "Concentration of solid component i per total bed volume", [self.N_sol, self.x_centers])
        self.ct_gas = daeVariable("c_gas_tot", molar_conc_type, self, "Total concentration of gas per total bed volume", [self.x_centers])
        self.ct_sol = daeVariable("c_sol_tot", molar_conc_sol_type, self, "Total concentration of solid per total bed volume", [self.x_centers])
        self.y_gas = daeVariable("y_gas", molar_frac_type, self, "Molar fraction of gaseous component i", [self.N_gas, self.x_centers])
        self.c_gas = daeVariable("c_gas", molar_conc_type, self, "Concentration of gaseous component i per total bed volume", [self.N_gas, self.x_centers])

        self.R_rxn = None
        if self.reaction_network.has_reactions:
            self.R_rxn = daeVariable("R_rxn", molar_source_type, self, "Reaction rate of reaction k per total bed volume", [self.N_rxn, self.x_centers])


        self.T = daeVariable("temp_bed", temp_type, self, "Temperature inside a cell", [self.x_centers])
        self.h_cell = daeVariable("h_cell", volume_enthalpy_type, self, "Enthalpy per total bed volume", [self.x_centers])
        self.h_gas = daeVariable("h_gas", molar_enthalpy_type, self, "Molar enthalpy of gas i in a cell", [self.N_gas, self.x_centers])
        self.h_sol = daeVariable("h_sol", molar_enthalpy_type, self, "Molar enthalpy of solid i in a cell", [self.N_sol, self.x_centers])
        self.J_gas_face = daeVariable("J_gas_face", heat_flux_type, self, "Enthalpy flow at cell faces attributable to component i", [self.N_gas, self.x_faces])

        self.Dax = daeVariable("Dax", dispersion_type, self, "Face axial dispersion coefficient", [self.x_faces])
        self.u_s = daeVariable("u_s", velocity_type, self, "Face superficial velocity", [self.x_faces])
        self.P = daeVariable("pres_bed", pres_type, self, "Pressure inside a cell", [self.x_centers])
        self.mu_g = daeVariable("mu_g", viscosity_type, self, "Mole-averaged gas viscosity in a cell", [self.x_centers])
        self.rho_g = daeVariable("rho_g", density_type, self, "Gas density in a cell", [self.x_centers])

        self.mass_in_total = self.mass_out_total = self.mass_bed_total = None
        if "mass_balance" in requested_reports:
            self.mass_in_total = daeVariable("mass_in_total", mass_inventory_type, self, "Cumulative gas-phase mass that has entered the bed")
            self.mass_out_total = daeVariable("mass_out_total", mass_inventory_type, self, "Cumulative gas-phase mass that has left the bed")
            self.mass_bed_total = daeVariable("mass_bed_total", mass_inventory_type, self, "Gas plus solid mass currently residing in the bed")

        self.heat_in_total = self.heat_out_total = None
        self.heat_loss_total = self.heat_bed_total = None
        if "heat_balance" in requested_reports:
            self.heat_in_total = daeVariable("heat_in_total", energy_inventory_type, self, "Cumulative gas-phase enthalpy that has entered the bed")
            self.heat_out_total = daeVariable("heat_out_total", energy_inventory_type, self, "Cumulative gas-phase enthalpy that has left the bed")
            self.heat_loss_total = daeVariable("heat_loss_total", energy_inventory_type, self, "Cumulative heat transferred from the bed to the environment")
            self.heat_bed_total = daeVariable("heat_bed_total", energy_inventory_type, self, "Gas plus solid enthalpy currently residing in the bed")

        self.F_in_const = daeParameter("F_in_const", molar_flow_type.Units, self, "Default fixed total molar flow at the inlet")
        self.y_in_const = daeParameter("y_in_const", molar_frac_type.Units, self, "Default fixed molar fraction of component i at the inlet", [self.N_gas])
        self.T_in_const = daeParameter("T_in_const", K, self, "Default fixed temperature at the inlet")
        self.P_out_const = daeParameter("P_out_const", Pa, self, "Default fixed pressure at the outlet")

        self.F_in = daeVariable("F_in", molar_flow_type, self, "Total molar flow at the inlet")
        self.y_in = daeVariable("y_in", molar_frac_type, self, "Molar fraction of component i at the inlet", [self.N_gas])
        self.T_in = daeVariable("T_in", temp_type, self, "Temperature at the inlet")
        self.P_in = daeVariable("P_in", pres_type, self, "Pressure at the inlet boundary")
        self.P_out = daeVariable("P_out", pres_type, self, "Pressure at the outlet boundary")

        self.T_env = daeParameter("T_env", K, self, "Temperature of the ambient environment outisde the reactor")
        self.U_eff = daeParameter("U_eff", J / (K * s * m**2), self, "Heat transfer coefficient from inside of the bed to the environment")

    def _source_expression(self, coefficients, idx_cell):
        source = Constant(0.0 * mol / (m**3 * s))
        if self.R_rxn is None:
            return source

        for reaction_idx, coefficient in enumerate(coefficients):
            if coefficient != 0.0:
                source = source + Constant(coefficient) * self.R_rxn(reaction_idx, idx_cell)
        return source

    def set_axial_grid_from_faces(self, face_locations):
        face_locations = np.asarray(face_locations, dtype=float)
        center_locations = 0.5 * (face_locations[:-1] + face_locations[1:])
        self.x_faces.CreateStructuredGrid(face_locations.size - 1, 0, 1)
        self.x_centers.CreateStructuredGrid(center_locations.size - 1, 0, 1)
        self.x_faces.Points = face_locations.tolist()
        self.x_centers.Points = center_locations.tolist()
        self.xval_cells.SetValues(center_locations)
        self.face_locations_m = face_locations

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
        face_coords = [Constant(value * m) for value in self.face_locations_m]
        cross_section_area = Constant(CIRCLE_CONSTANT) * self.R_bed() ** 2
        conc_eps = Constant(1e-8 * mol / m**3)
        enthalpy_eps = Constant(1e-8 * J / mol)
        gas_molecular_weights = [
            Constant(self.property_registry.get_record(species_name).mw * kg / mol)
            for species_name in self.gas_species
        ]
        solid_molecular_weights = [
            Constant(self.property_registry.get_record(species_name).mw * kg / mol)
            for species_name in self.solid_species
        ]

        # Shared mathematical expressions. Equation creation remains below in solver order.
        def smooth_program_expression(default_expression, program, units, component=None):
            segments = program.segments
            if not segments:
                return default_expression
            width = Constant(self.smooth_ramp_width_s * s)

            def scalar(value):
                return float(value if component is None else value[component])

            def smooth_positive_time(elapsed_time):
                return Constant(0.5) * (
                    elapsed_time + Sqrt(elapsed_time * elapsed_time + width * width)
                )

            expression = Constant(scalar(segments[0].start_value) * units)
            for segment in segments:
                delta = scalar(segment.end_value) - scalar(segment.start_value)
                if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
                    continue
                start_time = Constant(float(segment.start_time) * s)
                end_time = Constant(float(segment.end_time) * s)
                duration = Constant((float(segment.end_time) - float(segment.start_time)) * s)
                ramp_fraction = (
                    smooth_positive_time(Time() - start_time)
                    - smooth_positive_time(Time() - end_time)
                ) / duration
                expression = expression + Constant(delta * units) * ramp_fraction
            return expression

        def heat_loss_density(idx_cell):
            return (self.T(idx_cell) - self.T_env()) * (2.0 * self.U_eff()) / self.R_bed()

        def gas_face_flux_residual(gas_idx, face_index):
            if face_index == 0:
                return self.y_in(gas_idx) * self.F_in() / cross_section_area - self.N_gas_face(gas_idx, 0)
            if face_index == Nf - 1:
                return (
                    self.N_gas_face(gas_idx, face_index)
                    - self.u_s(face_index) * self.c_gas(gas_idx, Nc - 1) / self.gasfrac(Nc - 1)
                )

            idx_cell_L = face_index - 1
            idx_cell_R = face_index
            dx = center_coords[idx_cell_R] - center_coords[idx_cell_L]
            ct_L = self.ct_gas(idx_cell_L)
            ct_R = self.ct_gas(idx_cell_R)
            ct_face = Constant(0.5) * (
                ct_L / self.gasfrac(idx_cell_L)
                + ct_R / self.gasfrac(idx_cell_R)
            )
            c_face_left, c_face_right = reconstruct_face_states(
                lambda idx_cell: self.c_gas(gas_idx, idx_cell) / self.gasfrac(idx_cell),
                face_index,
                Nc,
                self.mass_scheme,
                conc_eps,
                minimum=Min,
                maximum=Max,
            )
            convective_flux = split_face_flux(
                self.u_s(face_index),
                c_face_left,
                c_face_right,
                absolute=Abs,
            )
            return (
                self.N_gas_face(gas_idx, face_index)
                - convective_flux
                + self.Dax(face_index)
                * ct_face
                * (self.y_gas(gas_idx, idx_cell_R) - self.y_gas(gas_idx, idx_cell_L))
                / dx
            )

        def gas_face_enthalpy_residual(gas_idx, face_index):
            species_name = self.gas_species[gas_idx]
            if face_index == 0:
                return (
                    self.J_gas_face(gas_idx, 0)
                    - self.N_gas_face(gas_idx, 0)
                    * self.property_registry.enthalpy_expression(species_name, self.T_in())
                )
            if face_index == Nf - 1:
                return (
                    self.J_gas_face(gas_idx, face_index)
                    - self.N_gas_face(gas_idx, face_index) * self.h_gas(gas_idx, Nc - 1)
                )

            h_face_left, h_face_right = reconstruct_face_states(
                lambda idx_cell: self.h_gas(gas_idx, idx_cell),
                face_index,
                Nc,
                self.heat_scheme,
                enthalpy_eps,
                minimum=Min,
                maximum=Max,
            )
            transported_enthalpy_flux = split_face_flux(
                self.N_gas_face(gas_idx, face_index),
                h_face_left,
                h_face_right,
                absolute=Abs,
            )
            return self.J_gas_face(gas_idx, face_index) - transported_enthalpy_flux

        def ergun_drag(viscosity, density, voidage, particle_diameter, velocity):
            return (
                150 * viscosity * (1 - voidage) ** 2 / (voidage**3 * particle_diameter**2)
                * velocity
                + 1.75 * density * (1 - voidage) / (voidage**3 * particle_diameter)
                * Abs(velocity) * velocity
            )

        def ergun_residual(face_index):
            if face_index == 0:
                dx = center_coords[0] - face_coords[0]
                e_b_face = self.e_b(0)
                drag = ergun_drag(
                    self.mu_g(0), self.rho_g(0), e_b_face, self.d_p(0), self.u_s(0)
                )
                return (self.P_in() - self.P(0)) / dx - drag

            if face_index == Nf - 1:
                dx = face_coords[Nf - 1] - center_coords[Nc - 1]
                e_b_face = self.e_b(Nc - 1)
                drag = ergun_drag(
                    self.mu_g(Nc - 1),
                    self.rho_g(Nc - 1),
                    e_b_face,
                    self.d_p(face_index),
                    self.u_s(face_index),
                )
                return (self.P(Nc - 1) - self.P_out()) / dx - drag

            idx_cell_L = face_index - 1
            idx_cell_R = face_index
            dx = center_coords[idx_cell_R] - center_coords[idx_cell_L]
            mu_face = 0.5 * (self.mu_g(idx_cell_L) + self.mu_g(idx_cell_R))
            rho_face = 0.5 * (self.rho_g(idx_cell_L) + self.rho_g(idx_cell_R))
            e_b_face = 0.5 * (self.e_b(idx_cell_L) + self.e_b(idx_cell_R))
            drag = ergun_drag(
                mu_face,
                rho_face,
                e_b_face,
                self.d_p(face_index),
                self.u_s(face_index),
            )
            return (self.P(idx_cell_L) - self.P(idx_cell_R)) / dx - drag

        def bed_mass_expression():
            mass_bed_total = Constant(0.0 * kg)
            for idx_cell in range(Nc):
                dx = face_coords[idx_cell + 1] - face_coords[idx_cell]
                gas_mass_density = Constant(0.0 * kg / m**3)
                for gas_idx in range(Ng):
                    gas_mass_density = gas_mass_density + self.c_gas(gas_idx, idx_cell) * gas_molecular_weights[gas_idx]

                solid_mass_density = Constant(0.0 * kg / m**3)
                for sol_idx in range(Ns):
                    solid_mass_density = solid_mass_density + self.c_sol(sol_idx, idx_cell) * solid_molecular_weights[sol_idx]

                mass_bed_total = mass_bed_total + cross_section_area * (gas_mass_density + solid_mass_density) * dx
            return mass_bed_total

        def bed_heat_expressions():
            heat_bed_total = Constant(0.0 * J)
            heat_loss_rate_total = Constant(0.0 * J / s)
            for idx_cell in range(Nc):
                dx = face_coords[idx_cell + 1] - face_coords[idx_cell]
                heat_bed_total = heat_bed_total + cross_section_area * self.h_cell(idx_cell) * dx
                heat_loss_rate_total = heat_loss_rate_total + cross_section_area * heat_loss_density(idx_cell) * dx
            return heat_bed_total, heat_loss_rate_total

        def mass_flux_expressions():
            mass_flux_in = Constant(0.0 * kg / (s * m**2))
            mass_flux_out = Constant(0.0 * kg / (s * m**2))
            for gas_idx in range(Ng):
                mass_flux_in = mass_flux_in + self.N_gas_face(gas_idx, 0) * gas_molecular_weights[gas_idx]
                mass_flux_out = mass_flux_out + self.N_gas_face(gas_idx, Nf - 1) * gas_molecular_weights[gas_idx]
            return mass_flux_in, mass_flux_out

        def gas_component_enthalpy_residual(gas_idx, idx_cell):
            species_name = self.gas_species[gas_idx]
            return self.h_gas(gas_idx, idx_cell) - self.property_registry.enthalpy_expression(
                species_name,
                self.T(idx_cell),
            )

        temperature_anchor_gas_idx = int(np.argmax(self.initial_inlet_composition))

        # Species and solid balances. Keep DAETools' flattened variable order for ILU diagonals.
        for gas_idx in range(Ng):
            for idx_cell in range(Nc):
                eq = self.CreateEquation(f"species_balance_cell_{idx_cell}_{self.gas_species[gas_idx]}")
                source = self._source_expression(
                    self.reaction_network.gas_source_matrix[gas_idx],
                    idx_cell,
                )
                dx = face_coords[idx_cell + 1] - face_coords[idx_cell]
                eq.Residual = dt(self.c_gas(gas_idx, idx_cell)) + (
                    self.N_gas_face(gas_idx, idx_cell + 1) - self.N_gas_face(gas_idx, idx_cell)
                ) / dx - source

        for sol_idx in range(Ns):
            for idx_cell in range(Nc):
                eq = self.CreateEquation(f"solid_species_balance_cell_{idx_cell}_{self.solid_species[sol_idx]}")
                source = self._source_expression(
                    self.reaction_network.solid_source_matrix[sol_idx],
                    idx_cell,
                )
                eq.Residual = dt(self.c_sol(sol_idx, idx_cell)) - source

        eq = self.CreateEquation("total_concentration_closure")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.ct_gas(idx_cell) - Sum(self.c_gas.array("*", idx_cell))

        eq = self.CreateEquation("solid_total_concentration_closure")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.ct_sol(idx_cell) - Sum(self.c_sol.array("*", idx_cell))

        eq = self.CreateEquation("molar_fraction_calc")
        idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, "i")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.y_gas(idx_gas, idx_cell) * self.ct_gas(idx_cell) - self.c_gas(idx_gas, idx_cell)

        # Gas face fluxes and reaction rates.
        for gas_idx, species_name in enumerate(self.gas_species):
            for face_index in range(Nf):
                if face_index == 0:
                    equation_name = f"lhs_boundary_flux_{species_name}"
                elif face_index == Nf - 1:
                    equation_name = f"rhs_boundary_flux_{species_name}"
                else:
                    equation_name = f"face_flux_{face_index}_{species_name}"
                eq = self.CreateEquation(equation_name)
                eq.Residual = gas_face_flux_residual(gas_idx, face_index)

        if self.reaction_network.has_reactions:
            for reaction_idx, reaction in enumerate(self.reaction_network.reactions):
                eq = self.CreateEquation(f"reaction_rate_{reaction.id}")
                idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
                kinetics_context = KineticsContext(
                    model=self,
                    idx_cell=idx_cell,
                    gas_species_index=self.gas_species_index,
                    solid_species_index=self.solid_species_index,
                )
                eq.Residual = self.R_rxn(reaction_idx, idx_cell) - self.reaction_rate_hooks[reaction_idx](
                    kinetics_context
                )

        # Energy balances and component enthalpy closures.
        temperature_anchor_species = self.gas_species[temperature_anchor_gas_idx]
        eq = self.CreateEquation(f"gas_component_enthalpy_{temperature_anchor_species}")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = gas_component_enthalpy_residual(temperature_anchor_gas_idx, idx_cell)

        for idx_cell in range(Nc):
            dx = face_coords[idx_cell + 1] - face_coords[idx_cell]
            eq = self.CreateEquation(f"energy_balance_cell_{idx_cell}")
            eq.Residual = dt(self.h_cell(idx_cell)) + (
                Sum(self.J_gas_face.array("*", idx_cell + 1)) - Sum(self.J_gas_face.array("*", idx_cell))
            ) / dx + heat_loss_density(idx_cell)

        for gas_idx, species_name in enumerate(self.gas_species):
            if gas_idx == temperature_anchor_gas_idx:
                eq = self.CreateEquation("total_cell_enthalpy")
                idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
                eq.Residual = self.h_cell(idx_cell) - Sum(self.c_gas.array("*", idx_cell)*self.h_gas.array("*", idx_cell)) - Sum(self.c_sol.array("*", idx_cell)*self.h_sol.array("*", idx_cell))
            else:
                eq = self.CreateEquation(f"gas_component_enthalpy_{species_name}")
                idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
                eq.Residual = gas_component_enthalpy_residual(gas_idx, idx_cell)

        for sol_idx, species_name in enumerate(self.solid_species):
            eq = self.CreateEquation(f"solid_component_enthalpy_{species_name}")
            idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
            eq.Residual = self.h_sol(sol_idx, idx_cell) - self.property_registry.enthalpy_expression(
                species_name,
                self.T(idx_cell),
            )

        for gas_idx, species_name in enumerate(self.gas_species):
            for face_index in range(Nf):
                if face_index == 0:
                    equation_name = f"lhs_boundary_enthalpy_flux_{species_name}"
                elif face_index == Nf - 1:
                    equation_name = f"rhs_boundary_enthalpy_flux_{species_name}"
                else:
                    equation_name = f"face_enthalpy_flux_{face_index}_{species_name}"
                eq = self.CreateEquation(equation_name)
                eq.Residual = gas_face_enthalpy_residual(gas_idx, face_index)

        # Dispersion, momentum, pressure, and gas-property closures.
        eq = self.CreateEquation("axial_dispersion_face")
        idx_face = eq.DistributeOnDomain(self.x_faces, eClosedClosed, "x_f")
        eq.Residual = self.Dax(idx_face) - Abs(self.u_s(idx_face)) * 0.5 * self.d_p(idx_face)

        for face_index in range(Nf):
            eq = self.CreateEquation(f"ergun_face_{face_index}")
            eq.Residual = ergun_residual(face_index)

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
        mixture_molecular_weight = Constant(0.0 * kg / mol)
        for gas_idx, species_name in enumerate(self.gas_species):
            mixture_molecular_weight = (
                mixture_molecular_weight
                + self.y_gas(gas_idx, idx_cell)
                * Constant(self.property_registry.get_record(species_name).mw * kg / mol)
            )
        eq.Residual = self.rho_g(idx_cell) - self.P(idx_cell) * mixture_molecular_weight / (
            self.R_gas() * self.T(idx_cell)
        )

        # Integral mass and heat accounting.
        if self.mass_in_total is not None:
            mass_flux_in, mass_flux_out = mass_flux_expressions()
            eq = self.CreateEquation("mass_in_total_accumulation")
            eq.Residual = dt(self.mass_in_total()) - cross_section_area * mass_flux_in

            eq = self.CreateEquation("mass_out_total_accumulation")
            eq.Residual = dt(self.mass_out_total()) - cross_section_area * mass_flux_out

            eq = self.CreateEquation("mass_bed_total_definition")
            eq.Residual = self.mass_bed_total() - bed_mass_expression()

        if self.heat_in_total is not None:
            heat_bed_total, heat_loss_rate_total = bed_heat_expressions()
            eq = self.CreateEquation("heat_in_total_accumulation")
            eq.Residual = dt(self.heat_in_total()) - cross_section_area * Sum(self.J_gas_face.array("*", 0))

            eq = self.CreateEquation("heat_out_total_accumulation")
            eq.Residual = dt(self.heat_out_total()) - cross_section_area * Sum(self.J_gas_face.array("*", Nf - 1))

            eq = self.CreateEquation("heat_loss_total_accumulation")
            eq.Residual = dt(self.heat_loss_total()) - heat_loss_rate_total

            eq = self.CreateEquation("heat_bed_total_definition")
            eq.Residual = self.heat_bed_total() - heat_bed_total

        # Smooth operating programs and the existing oriented boundary equations.
        eq = self.CreateEquation("Active_inlet_flow_smooth")
        eq.Residual = self.F_in() - smooth_program_expression(
            self.F_in_const(),
            self.inlet_flow_program,
            molar_flow_type.Units,
        )

        for gas_idx in range(Ng):
            eq = self.CreateEquation(f"Active_inlet_composition_{gas_idx}_smooth")
            eq.Residual = self.y_in(gas_idx) - smooth_program_expression(
                self.y_in_const(gas_idx),
                self.inlet_composition_program,
                molar_frac_type.Units,
                component=gas_idx,
            )

        eq = self.CreateEquation("Active_inlet_temperature_smooth")
        eq.Residual = self.T_in() - smooth_program_expression(
            self.T_in_const(),
            self.inlet_temperature_program,
            K,
        )

        eq = self.CreateEquation("inlet_pressure_from_flow")
        eq.Residual = self.F_in() / cross_section_area - self.u_s(0) * self.P_in() / (self.R_gas() * self.T_in())

        eq = self.CreateEquation("Active_outlet_pressure_smooth")
        eq.Residual = self.P_out() - smooth_program_expression(
            self.P_out_const(),
            self.outlet_pressure_program,
            Pa,
        )
__all__ = ("PackedBedModel",)
