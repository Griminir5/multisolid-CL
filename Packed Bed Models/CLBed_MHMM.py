__doc__ = """
This file is means as the very first step towards a model for chemical looping in packed-bed reactors.
This file implements a gas/solid mass-balance skeleton with a temporary plug-flow velocity closure.
"""

import sys

import numpy as np
from daetools.pyDAE import *

from packed_bed_properties import DEFAULT_PROPERTY_REGISTRY, PropertyRegistry
from pyUnits import GW, J, K, Pa, m, mol, s  # this will not show up because pylance cannot get to .pyd files


molar_flux_type =       daeVariableType(name="molar_flux_type", units=mol / (s * m**2), 
                                        lowerBound=-100000, upperBound=100000, initialGuess=0, absTolerance=1e-5,)
molar_flow_type =       daeVariableType(name="molar_flow_type", units=mol / s,
                                        lowerBound=-1000, upperBound=1000, initialGuess=0, absTolerance=1e-5,)
molar_conc_type =       daeVariableType( name="molar_conc_type", units=mol / m**3,
                                        lowerBound=0, upperBound=100000, initialGuess=0, absTolerance=1e-5,)
molar_conc_sol_type =   daeVariableType(name="molar_conc_sol_type", units=mol / m**3,
                                        lowerBound=0, upperBound=1000000, initialGuess=0, absTolerance=1e-5, valueConstraint=eValueGTEQ)
molar_frac_type =       daeVariableType(name="molar_frac_type", units=dimless,
                                        lowerBound=-0.1, upperBound=1.1, initialGuess=0, absTolerance=1e-5,)
dispersion_type =       daeVariableType(name="dispersion_type", units=m**2 / s,
                                        lowerBound=0, upperBound=100, initialGuess=0, absTolerance=1e-5,)
molar_source_type =     daeVariableType(name="molar_source_type", units=mol / (m**3 * s),
                                        lowerBound=-1000000, upperBound=1000000, initialGuess=0, absTolerance=1e-5,)

length_type =           daeVariableType(name="length_type", units=m,
                                        lowerBound=0, upperBound=20, initialGuess=0, absTolerance=1e-5,)

temp_type =             daeVariableType(name="temp_type", units=K,
                                        lowerBound=100, upperBound=2000, initialGuess=500, absTolerance=1e-5,)
molar_enthalpy_type =   daeVariableType(name="molar_enthalpy_type", units=J / mol,
                                        lowerBound=-1e7, upperBound=1e7, initialGuess=0, absTolerance=1e-5,)

pres_type =             daeVariableType(name="pres_type", units=Pa,
                                        lowerBound=1e-3, upperBound=1e7, initialGuess=1e5, absTolerance=1e-5,)
velocity_type =         daeVariableType(name="velocity_type", units=m / s,
                                        lowerBound=-100, upperBound=100, initialGuess=1, absTolerance=1e-5,)

fraction_type =         daeVariableType(name="fraction_type", units=dimless,
                                        lowerBound=-0.1, upperBound=1.1, initialGuess=0, absTolerance=1e-5,)


VALID_GAS_SPECIES = ["AR", "CH4", "CO", "CO2", "H2", "H2O", "HE", "N2", "O2"]
VALID_SOLID_SPECIES = ["Ni", "NiO", "CaAl2O4"]
# Example real solid lists can replace the placeholder above, e.g.
# ["CaAl:A-01", "Ni", "NiO", "Fe", "FeO", "Fe3O4", "Fe2O3", "Ca", "CaO", "CaCO3", "CaSO4"]


class CLBed_mass(daeModel):
    def __init__(
        self,
        Name,
        gas_species,
        solid_species,
        property_registry=DEFAULT_PROPERTY_REGISTRY,
        Parent=None,
        Description="Gas/solid mass balance-only bed",
    ):
        daeModel.__init__(self, Name, Parent, Description)

        self.gas_species = list(gas_species)
        self.solid_species = list(solid_species)
        self.property_registry = property_registry

        self.R_gas = daeParameter("R_gas", (Pa * m**3) / (mol * K), self, "Gas constant")
        self.pi = daeParameter("&pi;", dimless, self, "Circle constant")

        self.L_bed = daeParameter("Bed_length", m, self, "Length of the reactor bed")
        self.R_bed = daeParameter("Bed_radius", m, self, "Radius of the reactor bed")

        self.d_p = daeParameter("Particle_length", m, self, "Characteristic length of the solid particles")
        self.c_in = daeParameter("conc_inlet", mol / (m**3), self, "Temporary total molar concentration used in the plug-flow closure")
        self.T_const = daeParameter("T_const", K, self, "Temporary uniform bed temperature used until the energy balance is implemented")

        self.x_centers = daeDomain("Cell_centers", self, m, "Axial cell centers domain over the packed bed")
        self.x_faces = daeDomain("Cell_faces", self, m, "Axial cell faces domain over the packed bed")
        self.N_gas = daeDomain("Gas_comps", self, dimless, "Number of gaseous components")
        self.N_sol = daeDomain("Solid_comps", self, dimless, "Number of solid components")
        self.N_gas.CreateArray(len(self.gas_species))
        self.N_sol.CreateArray(len(self.solid_species))

        self.xval_cells = daeParameter("xval_cells", m, self, "Coordinate of cell centers")
        self.xval_faces = daeParameter("xval_faces", m, self, "Coordinate of cell faces")

        self.xval_cells.DistributeOnDomain(self.x_centers)
        self.xval_faces.DistributeOnDomain(self.x_faces)

        self.c_gas = daeVariable("c_gas", molar_conc_type, self, "Concentration of gaseous component i per total bed volume", [self.N_gas, self.x_centers])
        self.c_sol = daeVariable("c_sol", molar_conc_sol_type, self, "Concentration of solid component i per total bed volume", [self.N_sol, self.x_centers])

        self.ct_gas = daeVariable("c_gas_tot", molar_conc_type, self, "Total concentration of gas per total bed volume", [self.x_centers])
        self.ct_sol = daeVariable("c_sol_tot", molar_conc_sol_type, self, "Total concentration of solid per total bed volume", [self.x_centers])

        self.y_gas = daeVariable("y_gas", molar_frac_type, self, "Molar fraction of gaseous component i", [self.N_gas, self.x_centers])
        self.y_sol = daeVariable("y_sol", molar_frac_type, self, "Molar fraction of solid component i", [self.N_sol, self.x_centers])
        self.S_sol = daeVariable("S_sol", molar_source_type, self, "Net source of solid component i per total bed volume", [self.N_sol, self.x_centers])

        self.T = daeVariable("temp_bed", temp_type, self, "Temporary LTE cell temperature used by the enthalpy correlations", [self.x_centers])
        self.h_gas = daeVariable("h_gas", molar_enthalpy_type, self, "Gas-component molar enthalpy at the cell temperature", [self.N_gas, self.x_centers])
        self.h_sol = daeVariable("h_sol", molar_enthalpy_type, self, "Solid-component molar enthalpy at the cell temperature", [self.N_sol, self.x_centers])

        self.N_gas_face = daeVariable("N_gas_face", molar_flux_type, self, "Species i molar flux at cell faces", [self.N_gas, self.x_faces])
        self.Dax = daeVariable("Dax", dispersion_type, self, "Face axial dispersion coefficient", [self.x_faces])
        self.u_s = daeVariable("u_s", velocity_type, self, "Face superficial velocity", [self.x_faces])

        self.F_in_const = daeParameter("F_in_const", molar_flow_type.Units, self, "Fixed total molar flow at the inlet")
        self.y_in_const = daeParameter("y_in_const", molar_frac_type.Units, self, "Fixed molar fraction of component i at the inlet", [self.N_gas])

        self.F_in = daeVariable("F_in", molar_flow_type, self, "Total molar flow at the inlet")
        self.y_in = daeVariable("y_in", molar_frac_type, self, "Molar fraction of component i at the inlet", [self.N_gas])

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

        eq = self.CreateEquation("Active_inlet_flow")
        eq.Residual = self.F_in() - self.F_in_const()

        eq = self.CreateEquation("Active_inlet_composition")
        idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, "i")
        eq.Residual = self.y_in(idx_gas) - self.y_in_const(idx_gas)

        eq = self.CreateEquation("total_concentration_closure")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.ct_gas(idx_cell) - Sum(self.c_gas.array("*", idx_cell))

        eq = self.CreateEquation("molar_fraction_calc")
        idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, "i")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.y_gas(idx_gas, idx_cell) * self.ct_gas(idx_cell) - self.c_gas(idx_gas, idx_cell)

        eq = self.CreateEquation("solid_total_concentration_closure")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.ct_sol(idx_cell) - Sum(self.c_sol.array("*", idx_cell))

        eq = self.CreateEquation("solid_molar_fraction_calc")
        idx_sol = eq.DistributeOnDomain(self.N_sol, eClosedClosed, "j")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.y_sol(idx_sol, idx_cell) * self.ct_sol(idx_cell) - self.c_sol(idx_sol, idx_cell)

        eq = self.CreateEquation("temperature_closure")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.T(idx_cell) - self.T_const()

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

        eq = self.CreateEquation("solid_source_term_placeholder")
        idx_sol = eq.DistributeOnDomain(self.N_sol, eClosedClosed, "j")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, "x")
        eq.Residual = self.S_sol(idx_sol, idx_cell)

        eq = self.CreateEquation("lhs_boundary_flux")
        idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, "i")
        eq.Residual = self.y_in(idx_gas) * self.F_in() / (self.pi() * self.R_bed() ** 2) - self.N_gas_face(idx_gas, 0)

        eq = self.CreateEquation("axial_dispersion_face")
        idx_face = eq.DistributeOnDomain(self.x_faces, eClosedClosed, "x_f")
        eq.Residual = self.Dax(idx_face) - Abs(self.u_s(idx_face)) * 0.5 * self.d_p()

        zero_velocity = Constant(0 * m / s)
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
            eq.Residual = (
                self.N_gas_face(idx_gas, face_index)
                - uplus * self.c_gas(idx_gas, idx_cell_L)
                - uminus * self.c_gas(idx_gas, idx_cell_R)
                + self.Dax(face_index)
                * ct_face
                * (self.y_gas(idx_gas, idx_cell_R) - self.y_gas(idx_gas, idx_cell_L))
                / dx
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
        eq.Residual = self.N_gas_face(idx_gas, idx_face) - self.u_s(idx_face) * self.c_gas(idx_gas, Nc - 1)

        eq = self.CreateEquation("plug_flow_spec_face")
        idx_face = eq.DistributeOnDomain(self.x_faces, eClosedClosed, "x_f")
        eq.Residual = self.u_s(idx_face) - self.F_in() / (self.c_in() * self.pi() * self.R_bed() ** 2)


class simBed(daeSimulation):
    def __init__(
        self,
        gas_species=None,
        solid_species=None,
        property_registry=DEFAULT_PROPERTY_REGISTRY,
    ):
        daeSimulation.__init__(self)

        self.gas_species = list(VALID_GAS_SPECIES if gas_species is None else gas_species)
        self.solid_species = list(VALID_SOLID_SPECIES if solid_species is None else solid_species)
        self.property_registry = property_registry

        self.model = CLBed_mass(
            "MassTrsf",
            self.gas_species,
            self.solid_species,
            property_registry=self.property_registry,
        )

    def SetUpParametersAndDomains(self):
        self.model.R_gas.SetValue(8.314462 * (Pa * m**3) / (mol * K))
        self.model.pi.SetValue(3.14)

        self.model.L_bed.SetValue(2.5 * m)
        self.model.R_bed.SetValue(0.1 * m)
        self.model.d_p.SetValue(0.01 * m)
        self.model.T_const.SetValue(600 * K)

        self.model.c_in.SetValue(50.0 * mol / m**3)
        self.model.F_in_const.SetValue(0.785 * mol / s)
        self.model.SetUniformAxialGrid(10)

        inlet_y = np.zeros(len(self.gas_species), dtype=float)
        if "AR" in self.gas_species:
            inlet_y[self.gas_species.index("AR")] = 1.0
        else:
            inlet_y[0] = 1.0
        self.model.y_in_const.SetValues(inlet_y)

    def SetUpVariables(self):
        ng = self.model.N_gas.NumberOfPoints
        ns = self.model.N_sol.NumberOfPoints
        nc = self.model.x_centers.NumberOfPoints
        nf = self.model.x_faces.NumberOfPoints

        c0 = np.zeros((ng, nc), dtype=float)
        if "N2" in self.gas_species:
            c0[self.gas_species.index("N2"), :] = 25.0
        else:
            c0[0, :] = 25.0
        ct0 = c0.sum(axis=0)

        c0_sol = np.zeros((ns, nc), dtype=float)
        c0_sol[0, :] = 100000.0
        ct0_sol = c0_sol.sum(axis=0)

        inlet_y = np.asarray(self.model.y_in_const.npyValues, dtype=float)
        area = self.model.pi.GetValue() * self.model.R_bed.GetValue() ** 2

        fin = self.model.F_in_const.GetValue()
        u0 = fin / (self.model.c_in.GetValue() * area)
        dax0 = 0.5 * abs(u0) * self.model.d_p.GetValue()

        initial_temperature = self.model.T_const.GetValue()

        gas_h0 = np.asarray([self.model.property_registry.enthalpy_value(gas_name, initial_temperature) for gas_name in self.gas_species], dtype=float)
        solid_h0 = np.asarray([self.model.property_registry.enthalpy_value(sol_name, initial_temperature) for sol_name in self.solid_species], dtype=float)

        for cell_idx in range(nc):
            self.model.T.SetInitialGuess(cell_idx, initial_temperature * K)
            self.model.ct_gas.SetInitialGuess(cell_idx, ct0[cell_idx] * mol / m**3)
            self.model.ct_sol.SetInitialGuess(cell_idx, ct0_sol[cell_idx] * mol / m**3)

        for gas_idx in range(ng):
            self.model.y_in.SetInitialGuess(gas_idx, inlet_y[gas_idx])
            for cell_idx in range(nc):
                self.model.c_gas.SetInitialCondition(gas_idx, cell_idx, c0[gas_idx, cell_idx] * mol / m**3)
                y0 = 0.0 if ct0[cell_idx] <= 0.0 else c0[gas_idx, cell_idx] / ct0[cell_idx]
                self.model.y_gas.SetInitialGuess(gas_idx, cell_idx, y0)
                self.model.h_gas.SetInitialGuess(gas_idx, cell_idx, gas_h0[gas_idx] * J / mol)

        for sol_idx in range(ns):
            for cell_idx in range(nc):
                self.model.c_sol.SetInitialCondition(sol_idx, cell_idx, c0_sol[sol_idx, cell_idx] * mol / m**3)
                y0_sol = 0.0 if ct0_sol[cell_idx] <= 0.0 else c0_sol[sol_idx, cell_idx] / ct0_sol[cell_idx]
                self.model.y_sol.SetInitialGuess(sol_idx, cell_idx, y0_sol)
                self.model.S_sol.SetInitialGuess(sol_idx, cell_idx, 0.0 * mol / (m**3 * s))
                self.model.h_sol.SetInitialGuess(sol_idx, cell_idx, solid_h0[sol_idx] * J / mol)

        self.model.F_in.SetInitialGuess(fin * mol / s)

        for face_idx in range(nf):
            self.model.u_s.SetInitialGuess(face_idx, u0 * m / s)
            self.model.Dax.SetInitialGuess(face_idx, dax0 * m**2 / s)

            if face_idx == 0:
                face_flux = inlet_y * fin / area
            elif u0 >= 0.0:
                face_flux = u0 * c0[:, face_idx - 1]
            else:
                face_flux = u0 * c0[:, min(face_idx, nc - 1)]

            for gas_idx in range(ng):
                self.model.N_gas_face.SetInitialGuess(gas_idx, face_idx, face_flux[gas_idx] * mol / (s * m**2))


def guiRun(qtApp):

    simulation = simBed()
    simulation.model.SetReportingOn(True)
    simulation.ReportTimeDerivatives = True
    simulation.ReportingInterval = 0.1
    simulation.TimeHorizon = 10
    simulator = daeSimulator(qtApp, simulation=simulation)
    simulator.exec()


if __name__ == "__main__":
    qtApp = daeCreateQtApplication(sys.argv)
    guiRun(qtApp)
