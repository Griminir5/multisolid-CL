__doc__ = """
This file is means as the very first step towards a model for chemical looping in packed-bed reactors.
This file implements a very simple mass balance, with pressure and temperature considered fixed.
"""

# 1. Import the modules
import math
import sys
from dataclasses import dataclass
from time import localtime, strftime
import numpy as np
from daetools.pyDAE import *

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

gas_list = ["AR", "CH4", "CO", "CO2", "H2", "H2O", "HE", "N2", "O2"]
#solid_list = ["CaAl:A-01", "Ni", "NiO"]


def _as_float(value, field_name):
    if hasattr(value, "value"):
        value = value.value

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a float-like value or quantity, got {value!r}") from exc


def _validate_inlet_composition(gas_species, composition, tolerance=1e-9, context="composition"):
    validated = {}

    for species in gas_species:
        if species not in composition:
            raise ValueError(f"Missing inlet composition for species '{species}' in {context}.")

        value = _as_float(composition[species], f"{context} for species '{species}'")
        if value < -tolerance or value > 1.0 + tolerance:
            raise ValueError(
                f"Inlet composition for species '{species}' in {context} must lie in [0, 1]; received {value}."
            )
        validated[species] = value

    total = sum(validated[species] for species in gas_species)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"Inlet composition in {context} must sum to 1.0 within {tolerance:g}; received {total:.12g}."
        )

    return validated


def _coerce_composition(gas_species, composition, current=None, tolerance=1e-9, context="composition"):
    if composition is None:
        if current is None:
            raise ValueError(f"No inlet composition was supplied for {context}.")
        return dict(current)

    if isinstance(composition, dict):
        validated = dict(current) if current is not None else {}
        unknown_species = sorted(set(composition) - set(gas_species))
        if unknown_species:
            raise ValueError(f"Unknown gas species in {context}: {unknown_species}")

        for species, value in composition.items():
            validated[species] = _as_float(value, f"{context} for species '{species}'")
    else:
        if isinstance(composition, (str, bytes)):
            raise TypeError(f"{context} must be a mapping or an ordered sequence, not a string.")

        try:
            values = list(composition)
        except TypeError as exc:
            raise TypeError(f"{context} must be a mapping or an ordered sequence.") from exc

        if len(values) != len(gas_species):
            raise ValueError(
                f"{context} must contain {len(gas_species)} entries; received {len(values)}."
            )

        validated = {
            species: _as_float(value, f"{context} for species '{species}'")
            for species, value in zip(gas_species, values)
        }

    return _validate_inlet_composition(gas_species, validated, tolerance=tolerance, context=context)


@dataclass(frozen=True)
class InletProgramStep:
    duration: float
    kind: str = "ramp"
    F_target: object = None
    y_target: object = None


class InletProgram:
    """Piecewise-linear inlet program made of ramps and holds."""

    def __init__(self, gas_species, initial_F, initial_y, composition_tolerance=1e-9):
        self.gas_species = tuple(gas_species)
        self.composition_tolerance = float(composition_tolerance)
        self.initial_F = _as_float(initial_F, "initial_F")
        self.initial_y = _coerce_composition(
            self.gas_species,
            initial_y,
            current=None,
            tolerance=self.composition_tolerance,
            context="initial inlet composition",
        )
        self.steps = []

    def add_ramp(self, duration, F=None, y=None):
        duration_value = _as_float(duration, "ramp duration")
        if duration_value <= 0.0:
            raise ValueError("Ramp duration must be strictly positive.")

        self.steps.append(InletProgramStep(duration=duration_value, kind="ramp", F_target=F, y_target=y))
        return self

    def add_hold(self, duration):
        duration_value = _as_float(duration, "hold duration")
        if duration_value <= 0.0:
            raise ValueError("Hold duration must be strictly positive.")

        self.steps.append(InletProgramStep(duration=duration_value, kind="hold"))
        return self

    def build(self, repeats=1, time_horizon=None):
        try:
            repeats = int(repeats)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"repeats must be an integer-like value, got {repeats!r}") from exc

        if repeats < 1:
            raise ValueError("repeats must be at least 1.")

        horizon = None if time_horizon is None else _as_float(time_horizon, "time_horizon")
        if horizon is not None and horizon < 0.0:
            raise ValueError("time_horizon must be non-negative.")

        times = [0.0]
        F_profile = [self.initial_F]
        y_profiles = {species: [self.initial_y[species]] for species in self.gas_species}

        current_time = 0.0
        current_F = self.initial_F
        current_y = dict(self.initial_y)

        for repeat_idx in range(repeats):
            for step_idx, step in enumerate(self.steps):
                current_time += step.duration
                next_F = current_F if step.F_target is None else _as_float(
                    step.F_target, f"F target in step {step_idx + 1} of repeat {repeat_idx + 1}"
                )
                next_y = _coerce_composition(
                    self.gas_species,
                    step.y_target,
                    current=current_y,
                    tolerance=self.composition_tolerance,
                    context=f"step {step_idx + 1} of repeat {repeat_idx + 1}",
                )

                times.append(current_time)
                F_profile.append(next_F)
                for species in self.gas_species:
                    y_profiles[species].append(next_y[species])

                current_F = next_F
                current_y = next_y

        if horizon is not None:
            if horizon < current_time - 1e-12:
                raise ValueError(
                    f"time_horizon ({horizon}) is shorter than the inlet program duration ({current_time})."
                )
            if horizon > current_time + 1e-12:
                times.append(horizon)
                F_profile.append(current_F)
                for species in self.gas_species:
                    y_profiles[species].append(current_y[species])

        if len(times) < 2:
            raise ValueError(
                "An inlet program must span at least two time points. Add a ramp/hold step or provide time_horizon."
            )

        return {
            "times": times,
            "F_in": F_profile,
            "y_in": y_profiles,
            "end_time": times[-1],
        }

    def Build(self, repeats=1, time_horizon=None):
        return self.build(repeats=repeats, time_horizon=time_horizon)

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

        self.T = daeParameter("Bed_temperature_param", K, self, "Fixed temperature of over the whole bed") # for now a parameter
        self.P = daeParameter("Bed_pressure_param", Pa, self, "Fixed pressure over the whole bed") # for now a parameter

        self.d_p = daeParameter("Particle_length", m, self, "Characteristic length of the solid particles")
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

        self.xval_cells = daeParameter("xval_cells", m, self, "Coordinate of cell centers")
        self.xval_faces = daeParameter("xval_faces", m, self, "Coordinate of cell faces")

        self.T.DistributeOnDomain(self.x_centers)
        self.P.DistributeOnDomain(self.x_centers)
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

        ##Variables at the inlet
        self.F_in = daeVariable("F_in", molar_flow_type, self, "Total molar flow at the inlet")
        self.y_in = daeVariable("y_in", molar_frac_type, self, "Molar fraction of component i at the inlet", [self.N_gas])

    def ValidateInletComposition(self, composition, tolerance=1e-9, context="inlet composition"):
        return _coerce_composition(
            self.gas_species,
            composition,
            current=None,
            tolerance=tolerance,
            context=context,
        )

    def SetInletProgram(self, inlet_program, repeats=1, time_horizon=None):
        if not isinstance(inlet_program, InletProgram):
            raise TypeError("inlet_program must be an InletProgram instance.")
        if tuple(inlet_program.gas_species) != tuple(self.gas_species):
            raise ValueError(
                f"InletProgram gas species {inlet_program.gas_species} do not match model gas species {tuple(self.gas_species)}."
            )

        self._inlet_program = inlet_program
        self._compiled_inlet_program = inlet_program.build(repeats=repeats, time_horizon=time_horizon)
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

        if Nf != Nc + 1:
            raise RuntimeError("The axial grid must have exactly one more face than cell center.")

        center_coords = [self.xval_cells(idx_cell) for idx_cell in range(Nc)]
        face_coords = [self.xval_faces(idx_face) for idx_face in range(Nf)]

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

            eq = self.CreateEquation("scheduled_inlet_total_molar_flow")
            eq.Residual = self.F_in() - self._inlet_program_functions["F_in"]()

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

                eq = self.CreateEquation(f"scheduled_inlet_mole_fraction_{species}")
                eq.Residual = self.y_in(idx_gas) - schedule_fn()


        # Cell closure anbd mole fraction (gas) calculations
        for idx_cell in range(Nc):
            eq = self.CreateEquation(f"total_concentration_closure_cell_{idx_cell}")
            rhs = 0
            for idx_gas in range(Ng):
                rhs += self.c_gas(idx_gas, idx_cell)
            eq.Residual = self.ct_gas(idx_cell) - rhs

            for idx_gas in range(Ng):
                eq = self.CreateEquation(f"molar_fraction_calc_{self.gas_species[idx_gas]}_cell_{idx_cell}")
                eq.Residual = self.y_gas(idx_gas, idx_cell) * self.ct_gas(idx_cell) - self.c_gas(idx_gas, idx_cell)
        
        # Mass Transfer LHS boundary
        for idx_gas in range(Ng):
            eq = self.CreateEquation(f"lhs_boundary_flux_{self.gas_species[idx_gas]}")
            eq.Residual = self.y_in(idx_gas)*self.F_in/(self.pi() * self.R_bed()**2) - self.N_gas_face(idx_gas,0)

        # Axial Dispersion coefficient calculation
        for idx_face in range(Nf):
            eq = self.CreateEquation(f"axial_dispersion_face_{idx_face}")
            eq.Residual = self.Dax(idx_face) - Abs(self.u_s(idx_face))*0.5*self.d_p()

        # Flux calculation on interior faces
        for idx_face in range(1, Nf-1):
            cell_L = idx_face - 1
            cell_R = idx_face

            dx = center_coords[cell_R] - center_coords[cell_L]

            uplus  = Max(self.u_s(idx_face), 0) # this is for switching which face is "upwind" when velocity goes negative
            uminus = Min(self.u_s(idx_face), 0) # same as above, maybe should be replaced with an approximation

            ct_face = 0.5 * (self.ct_gas(cell_L) + self.ct_gas(cell_R)) # simple reconstruction, will need to be replaced with weno/higher order schemes with potentially flux limiter

            for idx_gas in range(Ng):
                eq = self.CreateEquation(f"face_flux_face_{idx_face}_{self.gas_species[idx_gas]}")

                cL = self.c_gas(idx_gas, cell_L)
                cR = self.c_gas(idx_gas, cell_R)

                yL = self.y_gas(idx_gas, cell_L)
                yR = self.y_gas(idx_gas, cell_R)

                eq.Residual = self.N_gas_face(idx_gas, idx_face) - uplus*cL - uminus*cR + self.Dax(idx_face)*ct_face*(yR-yL)/dx
        
        # Cell species balances

        for idx_cell in range(Nc):
            dx = face_coords[idx_cell+1] - face_coords[idx_cell]

            for idx_gas in range(Ng):
                eq = self.CreateEquation(f"species_balance_cell_{idx_cell}_{self.gas_species[idx_gas]}")
                eq.Residual = dt(self.c_gas(idx_gas, idx_cell)) + (self.N_gas_face(idx_gas, idx_cell+1) - self.N_gas_face(idx_gas, idx_cell))/dx

        # Mass transfer LHS boundary

        for idx_gas in range(Ng):
            eq = self.CreateEquation(f"rhs_boundary_flux_{self.gas_species[idx_gas]}")
            eq.Residual = self.N_gas_face(idx_gas, Nf-1) - self.u_s(Nf-1)*self.c_gas(idx_gas, Nc-1)

        # Equation that determines velocity on all faces except first; later on will be replaced with an EOS and momentum equation

        for idx_face in range(1, Nf):
            cell_L = idx_face - 1
            cell_R = idx_face

            cL = self.ct_gas(cell_L)
            cR = self.ct_gas(cell_R)

            eq = self.CreateEquation(f"plug_flow_velocity_face_{idx_face}")
            eq.Residual = self.u_s(idx_face)*0.5*(cL + cR)
