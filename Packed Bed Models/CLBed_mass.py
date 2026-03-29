__doc__ = """
This file is means as the very first step towards a model for chemical looping in packed-bed reactors.
This file implements a simple gas-phase mass balance with a temporary plug-flow velocity closure.
"""

# 1. Import the modules
import sys
from typing import Dict
from dataclasses import dataclass
import numpy as np
from daetools.pyDAE import *
from cycle_program import *

from pyUnits import m, s, K, kmol, Pa, GW # this will not show up because pylance cannot get to .pyd files

#
molar_flux_type = daeVariableType(name="molar_flux_type", units=kmol/m**2/s,
                                  lowerBound=-100, upperBound=100, initialGuess=0, absTolerance=1e-5)
molar_flow_type = daeVariableType(name="molar_flow_type", units=kmol/s,
                                  lowerBound=-10, upperBound=10, initialGuess=0, absTolerance=1e-5)
molar_conc_type = daeVariableType(name="molar_conc_type", units=kmol/m**3,
                                  lowerBound=0, upperBound=10, initialGuess=0, absTolerance=1e-5)
molar_conc_sol_type = daeVariableType(name="molar_conc_sol_type", units=kmol/m**3,
                                  lowerBound=0, upperBound=10, initialGuess=0, absTolerance=1e-5, valueConstraint=eValueGTEQ)
molar_frac_type = daeVariableType(name="molar_frac_type", units=dimless,
                                  lowerBound=-0.1, upperBound=1.1, initialGuess=0, absTolerance=1e-5)
dispersion_type = daeVariableType(name="dispersion_type", units=m**2/s,
                                  lowerBound=0, upperBound=100, initialGuess=0, absTolerance=1e-5)

length_type = daeVariableType(name="length_type", units=m,
                              lowerBound=0, upperBound=100, initialGuess=0, absTolerance=1e-5)

temp_type = daeVariableType(name="temp_type", units=K,
                            lowerBound=100, upperBound=2000, initialGuess=500, absTolerance=1e-5)

pres_type = daeVariableType(name="pres_type", units=Pa,
                            lowerBound=1e-3, upperBound=1e7, initialGuess=1e5, absTolerance=1e-5)
velocity_type = daeVariableType(name="velocity_type", units=m/s,
                                lowerBound=-100, upperBound=100, initialGuess=1, absTolerance=1e-5)

fraction_type = daeVariableType(name="fraction_type", units=dimless,
                                lowerBound=0, upperBound=1, initialGuess=0, absTolerance=1e-5)

VALID_GAS_SPECIES = ["AR", "CH4", "CO", "CO2", "H2", "H2O", "HE", "N2", "O2"]
#VALID_SOLID_SPECIES = ["CaAl:A-01", "Ni", "NiO"]

class CLBed_mass(daeModel):
    def __init__(self, Name, gas_species, Parent = None, Description = "Simple gas mass balance-only bed"):
        daeModel.__init__(self, Name, Parent, Description)
        # Fixed Parameters; some will not be necessary for this simple model, but keeping them for posterity
        self.gas_species = gas_species
        self._inlet_program = None
        self._compiled_inlet_program = None
        self._inlet_program_functions = {}

        self.R_gas = daeParameter("R_gas", (Pa*m**3)/(kmol* K), self, "Gas constant")
        self.pi = daeParameter("&pi;", dimless, self, "Circle constant")

        self.L_bed = daeParameter("Bed_length", m, self, "Length of the reactor bed")
        self.R_bed = daeParameter("Bed_radius", m, self, "Radius of the reactor bed")

        self.d_p = daeParameter("Particle_length", m, self, "Characteristic length of the solid particles")
        self.c_in = daeParameter("conc_inlet", kmol/(m**3), self, "Temporary total molar concentration used in the plug-flow closure")
        #self.e_b = daeParameter("Bed_voidage", dimless, self, "Interparticle voidage") # V_empty * V_vessel^-1
        #self.e_p = daeParameter("Particle_voidage", dimless, self, "Intraparticle voidage") # V_pores * V_pellet^-1

        # self.E_i = daeParameter("Ergun_inertial", dimless, self, "Inertial parameter for the Ergun equation")
        # self.E_v = daeParameter("Ergun_viscous", dimless, self, "Viscous parameter for the Ergun equation")

        # self.T_amb = daeParameter("Ambient_temperature", K, self, "Temperature of the ambient environment where heat losses go")
        # self.U = daeParameter("Ambient_HTC", GW/(K*m**2), self, "Heat transfer coefficient between the bed and ambient environment")



        # Domains over which distributed variables are indexed; this includes not only spatial, but also abstract domains such as species and reaction sets
        self.x_centers = daeDomain("Cell_centers",  self, m, "Axial cell centers domain over the packed bed")
        self.x_faces = daeDomain("Cell_faces",  self, m, "Axial cell faces domain over the packed bed")
        self.N_gas = daeDomain("Gas_comps",  self, dimless, "Number of gaseous components")
        # self.N_sol = daeDomain("Solid_comps",  self, dimless, "Number of solid components")
        self.N_gas.CreateArray(len(self.gas_species))

        self.xval_cells = daeParameter("xval_cells", m, self, "Coordinate of cell centers")
        self.xval_faces = daeParameter("xval_faces", m, self, "Coordinate of cell faces")

        self.xval_cells.DistributeOnDomain(self.x_centers)
        self.xval_faces.DistributeOnDomain(self.x_faces)

        # Variables; the actual pieces that participate in integration and solution
        ## Variables distributed over cell centers
        ### State cell variables
        self.c_gas = daeVariable("c_gas", molar_conc_type, self, "Concentration of gaseous component i per total bed volume", [self.N_gas, self.x_centers]) # kmol_i / m_bed^3
        #self.c_sol = daeVariable("c_sol", molar_conc_sol_type, self, "Concentration of solid component i per total bed volume", [self.N_sol, self.x_centers]) # kmol_i / m_bed^3

        ### Algebraic cell variables
        self.ct_gas = daeVariable("c_gas_tot", molar_conc_type, self, "Total concentration of gas per total bed volume", [self.x_centers]) # kmol_mix / m_bed^3
        #self.ct_sol = daeVariable("c_sol_tot", molar_conc_sol_type, self, "Total concentration of solid per total bed volume", [self.x_centers]) # kmol_mix / m_bed^3

        self.y_gas = daeVariable("y_gas", molar_frac_type, self, "Molar fraction of gaseous component i", [self.N_gas, self.x_centers])
        #self.y_sol = daeVariable("y_sol", molar_frac_type, self, "Molar fraction of solid component i", [self.N_sol, self.x_centers])

        #self.T = daeVariable("temp_bed", temp_type, self, "Temperature of a cell (gas and solid are in LTE)", [self.x_centers])
        #self.P = daeVariable("pres_bed", pres_type, self, "Pressure of a cell", [self.x_centers])

        #self.gasfrac = daeVariable("gasfrac", fraction_type, self, "Total bed volume occupied by gas") # m_void^3 / m_bed^3
        #self.solfrac = daeVariable("solfrac", fraction_type, self, "Total bed volume occupied by solid") # m_solid^3 / m_bed^3

        ## Variables distributed over cell faces
        self.N_gas_face = daeVariable("N_gas_face", molar_flux_type, self, "Species i molar flux at cell faces", [self.N_gas, self.x_faces])
        self.Dax = daeVariable("Dax", dispersion_type, self, "Face axial dispersion coefficient", [self.x_faces])
        self.u_s = daeVariable("u_s", velocity_type, self, "Face superficial velocity", [self.x_faces])

        ## Input data at the inlet
        self.F_in = daeParameter("F_in", molar_flow_type.Units, self, "Total molar flow at the inlet")
        self.y_in = daeParameter("y_in", molar_frac_type.Units, self, "Molar fraction of component i at the inlet", [self.N_gas])

    def SetInletProgram(self, inlet_program, repeat=False, time_horizon=None):
        if set(inlet_program.gas_species) != set(self.gas_species):
            raise ValueError(
                f"InletProgram gas species {inlet_program.gas_species} do not match model gas species {tuple(self.gas_species)}."
            )

        self._inlet_program = inlet_program
        self._compiled_inlet_program = inlet_program.build(repeat=repeat, time_horizon=time_horizon)
        return self._compiled_inlet_program

    def ClearInletProgram(self):
        self._inlet_program = None
        self._compiled_inlet_program = None
        self._inlet_program_functions = {}

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

        # Helpers
        Nc = self.x_centers.NumberOfPoints
        Nf = self.x_faces.NumberOfPoints
        Ng = self.N_gas.NumberOfPoints

        if Ng != len(self.gas_species):
            raise RuntimeError("Gas component domain size must match gas_species.")
        if Nf != Nc + 1:
            raise RuntimeError("The axial grid must have exactly one more face than cell center.")

        center_coords = [self.xval_cells(idx_cell) for idx_cell in range(Nc)]
        face_coords = [self.xval_faces(idx_face) for idx_face in range(Nf)]

        inlet_flow_expr = self.F_in()
        inlet_y_expr = None

        if self._compiled_inlet_program is not None:
            schedule = self._compiled_inlet_program
            time_points = list(schedule["times"])
            self._inlet_program_functions = {}

            self._inlet_program_functions["F_in"] = daeLinearInterpolationFunction(
                "F_in_schedule",
                self,
                molar_flow_type.Units,
                time_points,
                list(schedule["F_in"]),
                Time(),
            )
            inlet_flow_expr = self._inlet_program_functions["F_in"]()

            self._inlet_program_functions["y_in"] = {}
            for idx_gas, species in enumerate(self.gas_species):
                function_name = f"y_in_schedule_{species}"
                schedule_values = list(schedule["y_in"][species])
                schedule_fn = daeLinearInterpolationFunction(
                    function_name,
                    self,
                    molar_frac_type.Units,
                    time_points,
                    schedule_values,
                    Time(),
                )
                self._inlet_program_functions["y_in"][species] = schedule_fn
        else:
            inlet_y_expr = None


        # Cell closure and mole fraction (gas) calculations
        eq = self.CreateEquation("total_concentration_closure")
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, 'x')
        eq.Residual = self.ct_gas(idx_cell) - Sum(self.c_gas.array('*', idx_cell))

        eq = self.CreateEquation("molar_fraction_calc")
        idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, 'i')
        idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, 'x')
        eq.Residual = self.y_gas(idx_gas, idx_cell) * self.ct_gas(idx_cell) - self.c_gas(idx_gas, idx_cell)
        
        # Mass Transfer LHS boundary
        eq = self.CreateEquation("lhs_boundary_flux")
        idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, 'i')
        if self._compiled_inlet_program is not None:
            inlet_y_expr = Constant(0 * dimless)
            for gas_index, species in enumerate(self.gas_species):
                selector = 1 - (idx_gas() - gas_index) / (idx_gas() - gas_index + 1E-15)
                inlet_y_expr += selector * self._inlet_program_functions["y_in"][species]()
        else:
            inlet_y_expr = self.y_in(idx_gas)
        eq.Residual = inlet_y_expr * inlet_flow_expr / (self.pi() * self.R_bed()**2) - self.N_gas_face(idx_gas, 0)

        # Axial Dispersion coefficient calculation
        eq = self.CreateEquation("axial_dispersion_face")
        idx_face = eq.DistributeOnDomain(self.x_faces, eClosedClosed, 'x_f')
        eq.Residual = self.Dax(idx_face) - Abs(self.u_s(idx_face)) * 0.5 * self.d_p()

        # Flux calculation on interior faces
        eq = self.CreateEquation("face_flux")
        idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, 'i')
        idx_face = eq.DistributeOnDomain(self.x_faces, eOpenOpen, 'x_f')

        cL = Constant(0 * kmol / m**3)
        cR = Constant(0 * kmol / m**3)
        yL = Constant(0 * dimless)
        yR = Constant(0 * dimless)
        ct_face = Constant(0 * kmol / m**3)
        inv_dx = Constant(0 * m**(-1))

        for face_index in range(1, Nf-1):
            cell_L = face_index - 1
            cell_R = face_index
            dx = center_coords[cell_R] - center_coords[cell_L]
            selector = 1 - (idx_face() - face_index) / (idx_face() - face_index + 1E-15)
            ct_L = Sum(self.c_gas.array('*', cell_L))
            ct_R = Sum(self.c_gas.array('*', cell_R))

            cL += selector * self.c_gas(idx_gas, cell_L)
            cR += selector * self.c_gas(idx_gas, cell_R)
            yL += selector * self.y_gas(idx_gas, cell_L)
            yR += selector * self.y_gas(idx_gas, cell_R)
            ct_face += selector * 0.5 * (ct_L + ct_R)
            inv_dx += selector / dx

        uplus  = Max(self.u_s(idx_face), 0) # this is for switching which face is "upwind" when velocity goes negative
        uminus = Min(self.u_s(idx_face), 0) # same as above, maybe should be replaced with an approximation
        eq.Residual = self.N_gas_face(idx_gas, idx_face) - uplus*cL - uminus*cR + self.Dax(idx_face)*ct_face*(yR-yL)*inv_dx
        
        # Cell species balances

        for idx_cell in range(Nc):
            dx = face_coords[idx_cell+1] - face_coords[idx_cell]

            eq = self.CreateEquation(f"species_balance_cell_{idx_cell}")
            idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, 'i')
            eq.Residual = dt(self.c_gas(idx_gas, idx_cell)) + (self.N_gas_face(idx_gas, idx_cell+1) - self.N_gas_face(idx_gas, idx_cell))/dx

        # Mass transfer RHS boundary

        eq = self.CreateEquation("rhs_boundary_flux")
        idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, 'i')
        idx_face = eq.DistributeOnDomain(self.x_faces, eUpperBound, 'x_L')
        eq.Residual = self.N_gas_face(idx_gas, idx_face) - self.u_s(idx_face) * self.c_gas(idx_gas, Nc-1)

        # Equation that determines velocity; later on will be replaced with an EOS and momentum equation

        eq = self.CreateEquation("plug_flow_spec_face")
        idx_face = eq.DistributeOnDomain(self.x_faces, eClosedClosed, 'x_f')
        eq.Residual = self.u_s(idx_face) - inlet_flow_expr / (self.c_in() * self.pi() * self.R_bed()**2)
