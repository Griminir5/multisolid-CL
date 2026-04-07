__doc__ = """
Single-component 1D transport test-bed for axial discretization schemes.

The purpose of this file is to isolate spatial discretization effects from the
full packed-bed model. The balances remain fixed while the face reconstruction
used in the convective flux can be swapped by name.

Model scope:
- single scalar concentration balance
- single scalar temperature balance
- fixed positive superficial velocity
- optional axial mass and heat diffusion
- cumulative inlet/outlet and bed-inventory variables for conservation checks
- no EOS, no Ergun closure, no reactions

Recommended first comparisons:
- mass_scheme="upwind1", heat_scheme="upwind1"
- mass_scheme="upwind1", heat_scheme="central"
- mass_scheme="muscl_minmod", heat_scheme="muscl_minmod"
- mass_scheme="weno3", heat_scheme="weno3"
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from daetools.pyDAE import *
from pyUnits import J, K, m, mol, s
from axial_schemes import SUPPORTED_SCHEMES, reconstruct_face_left_value, validate_scheme_name


scalar_conc_type = daeVariableType(
    name="scalar_conc_type",
    units=mol / m**3,
    lowerBound=-1e6,
    upperBound=1e6,
    initialGuess=0,
    absTolerance=1e-6,
)
molar_flux_type = daeVariableType(
    name="molar_flux_type",
    units=mol / (s * m**2),
    lowerBound=-1e6,
    upperBound=1e6,
    initialGuess=0,
    absTolerance=1e-6,
)
temperature_test_type = daeVariableType(
    name="temperature_test_type",
    units=K,
    lowerBound=-2000,
    upperBound=5000,
    initialGuess=500,
    absTolerance=1e-6,
)
heat_flux_type = daeVariableType(
    name="heat_flux_type",
    units=J / (s * m**2),
    lowerBound=-1e12,
    upperBound=1e12,
    initialGuess=0,
    absTolerance=1e-4,
)
molar_inventory_type = daeVariableType(
    name="molar_inventory_type",
    units=mol / m**2,
    lowerBound=-1e12,
    upperBound=1e12,
    initialGuess=0,
    absTolerance=1e-6,
)
heat_inventory_type = daeVariableType(
    name="heat_inventory_type",
    units=J / m**2,
    lowerBound=-1e20,
    upperBound=1e20,
    initialGuess=0,
    absTolerance=1e-2,
)


class SimpleAxialTransportBed(daeModel):
    def __init__(
        self,
        Name,
        mass_scheme="upwind1",
        heat_scheme=None,
        Parent=None,
        Description="Single-component axial transport scheme test-bed",
    ):
        daeModel.__init__(self, Name, Parent, Description)

        self.mass_scheme = validate_scheme_name(mass_scheme)
        self.heat_scheme = validate_scheme_name(
            self.mass_scheme if heat_scheme is None else heat_scheme
        )

        self.L_bed = daeParameter("Bed_length", m, self, "Length of the 1D test domain")
        self.u_const = daeParameter("u_const", m / s, self, "Fixed positive superficial velocity")
        self.Dax_const = daeParameter("Dax_const", m**2 / s, self, "Fixed axial mass dispersion coefficient")
        self.lambda_ax = daeParameter(
            "lambda_ax",
            J / (s * m * K),
            self,
            "Fixed axial thermal conductivity / heat-diffusion coefficient",
        )
        self.rhoCp_flow = daeParameter(
            "rhoCp_flow",
            J / (m**3 * K),
            self,
            "Volumetric heat capacity attached to convective heat transport",
        )
        self.rhoCp_store = daeParameter(
            "rhoCp_store",
            J / (m**3 * K),
            self,
            "Volumetric heat capacity used in the accumulation term",
        )

        self.c_in_const = daeParameter("c_in_const", mol / m**3, self, "Dirichlet inlet concentration")
        self.T_in_const = daeParameter("T_in_const", K, self, "Dirichlet inlet temperature")

        self.x_centers = daeDomain("Cell_centers", self, m, "Axial cell centers")
        self.x_faces = daeDomain("Cell_faces", self, m, "Axial cell faces")

        self.xval_cells = daeParameter("xval_cells", m, self, "Cell-center coordinates")
        self.xval_faces = daeParameter("xval_faces", m, self, "Face coordinates")
        self.xval_cells.DistributeOnDomain(self.x_centers)
        self.xval_faces.DistributeOnDomain(self.x_faces)

        self.c = daeVariable("c", scalar_conc_type, self, "Single-component concentration", [self.x_centers])
        self.T = daeVariable("T", temperature_test_type, self, "Temperature", [self.x_centers])
        self.N_face = daeVariable("N_face", molar_flux_type, self, "Mass flux at cell faces", [self.x_faces])
        self.Q_face = daeVariable("Q_face", heat_flux_type, self, "Heat flux at cell faces", [self.x_faces])
        self.material_in_total = daeVariable(
            "material_in_total",
            molar_inventory_type,
            self,
            "Cumulative material that has entered the bed per unit cross-sectional area",
        )
        self.material_out_total = daeVariable(
            "material_out_total",
            molar_inventory_type,
            self,
            "Cumulative material that has left the bed per unit cross-sectional area",
        )
        self.material_bed_total = daeVariable(
            "material_bed_total",
            molar_inventory_type,
            self,
            "Material currently residing in the bed per unit cross-sectional area",
        )
        self.heat_in_total = daeVariable(
            "heat_in_total",
            heat_inventory_type,
            self,
            "Cumulative heat that has entered the bed per unit cross-sectional area",
        )
        self.heat_out_total = daeVariable(
            "heat_out_total",
            heat_inventory_type,
            self,
            "Cumulative heat that has left the bed per unit cross-sectional area",
        )
        self.heat_bed_total = daeVariable(
            "heat_bed_total",
            heat_inventory_type,
            self,
            "Heat currently residing in the bed per unit cross-sectional area",
        )

    def SetAxialGridFromFaces(self, face_locations):
        face_locations = np.asarray(face_locations, dtype=float)

        if face_locations.ndim != 1:
            raise ValueError("Face locations must be a 1D sequence.")
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
            raise ValueError("The test bed must contain at least one cell.")

        face_locations = np.linspace(0.0, self.L_bed.GetValue(), n_cells + 1)
        self.SetAxialGridFromFaces(face_locations)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        Nc = self.x_centers.NumberOfPoints
        Nf = self.x_faces.NumberOfPoints

        if Nf != Nc + 1:
            raise RuntimeError("The axial grid must have exactly one more face than cell center.")

        center_coords = [self.xval_cells(idx_cell) for idx_cell in range(Nc)]
        face_coords = [self.xval_faces(idx_face) for idx_face in range(Nf)]

        conc_eps = Constant(1e-8 * mol / m**3)
        temp_eps = Constant(1e-8 * K)

        dx_in = center_coords[0] - face_coords[0]
        eq = self.CreateEquation("lhs_boundary_mass_flux")
        eq.Residual = (
            self.N_face(0)
            - self.u_const() * self.c_in_const()
            + self.Dax_const() * (self.c(0) - self.c_in_const()) / dx_in
        )

        eq = self.CreateEquation("lhs_boundary_heat_flux")
        eq.Residual = (
            self.Q_face(0)
            - self.rhoCp_flow() * self.u_const() * self.T_in_const()
            + self.lambda_ax() * (self.T(0) - self.T_in_const()) / dx_in
        )

        for face_index in range(1, Nf - 1):
            idx_cell_L = face_index - 1
            idx_cell_R = face_index
            dx = center_coords[idx_cell_R] - center_coords[idx_cell_L]

            c_face = reconstruct_face_left_value(self.c, face_index, self.mass_scheme, conc_eps)
            T_face = reconstruct_face_left_value(self.T, face_index, self.heat_scheme, temp_eps)

            eq = self.CreateEquation(f"mass_flux_face_{face_index}")
            eq.Residual = (
                self.N_face(face_index)
                - self.u_const() * c_face
                + self.Dax_const() * (self.c(idx_cell_R) - self.c(idx_cell_L)) / dx
            )

            eq = self.CreateEquation(f"heat_flux_face_{face_index}")
            eq.Residual = (
                self.Q_face(face_index)
                - self.rhoCp_flow() * self.u_const() * T_face
                + self.lambda_ax() * (self.T(idx_cell_R) - self.T(idx_cell_L)) / dx
            )

        eq = self.CreateEquation("rhs_boundary_mass_flux")
        eq.Residual = self.N_face(Nf - 1) - self.u_const() * self.c(Nc - 1)

        eq = self.CreateEquation("rhs_boundary_heat_flux")
        eq.Residual = self.Q_face(Nf - 1) - self.rhoCp_flow() * self.u_const() * self.T(Nc - 1)

        material_bed_total = Constant(0.0 * mol / m**2)
        heat_bed_total = Constant(0.0 * J / m**2)
        for idx_cell in range(Nc):
            dx = face_coords[idx_cell + 1] - face_coords[idx_cell]

            eq = self.CreateEquation(f"mass_balance_cell_{idx_cell}")
            eq.Residual = dt(self.c(idx_cell)) + (self.N_face(idx_cell + 1) - self.N_face(idx_cell)) / dx

            eq = self.CreateEquation(f"heat_balance_cell_{idx_cell}")
            eq.Residual = self.rhoCp_store() * dt(self.T(idx_cell)) + (
                self.Q_face(idx_cell + 1) - self.Q_face(idx_cell)
            ) / dx

            material_bed_total = material_bed_total + self.c(idx_cell) * dx
            heat_bed_total = heat_bed_total + self.rhoCp_store() * self.T(idx_cell) * dx

        eq = self.CreateEquation("material_in_total_accumulation")
        eq.Residual = dt(self.material_in_total()) - self.N_face(0)

        eq = self.CreateEquation("material_out_total_accumulation")
        eq.Residual = dt(self.material_out_total()) - self.N_face(Nf - 1)

        eq = self.CreateEquation("material_bed_total_definition")
        eq.Residual = self.material_bed_total() - material_bed_total

        eq = self.CreateEquation("heat_in_total_accumulation")
        eq.Residual = dt(self.heat_in_total()) - self.Q_face(0)

        eq = self.CreateEquation("heat_out_total_accumulation")
        eq.Residual = dt(self.heat_out_total()) - self.Q_face(Nf - 1)

        eq = self.CreateEquation("heat_bed_total_definition")
        eq.Residual = self.heat_bed_total() - heat_bed_total


def _format_case_name(mass_scheme, heat_scheme):
    if mass_scheme == heat_scheme:
        return f"Case_{mass_scheme}_both"
    return f"Case_mass_{mass_scheme}_heat_{heat_scheme}"


class AxialSchemeBatch(daeModel):
    def __init__(self, Name, case_specs, Parent=None, Description="Batch axial-scheme transport test-bed"):
        daeModel.__init__(self, Name, Parent, Description)

        self.case_specs = []
        self.case_models = []

        for case_spec in case_specs:
            mass_scheme = validate_scheme_name(case_spec["mass_scheme"])
            heat_scheme = validate_scheme_name(case_spec["heat_scheme"])
            case_name = case_spec.get("name", _format_case_name(mass_scheme, heat_scheme))

            self.case_specs.append(
                {
                    "name": case_name,
                    "mass_scheme": mass_scheme,
                    "heat_scheme": heat_scheme,
                }
            )
            self.case_models.append(
                SimpleAxialTransportBed(
                    case_name,
                    mass_scheme=mass_scheme,
                    heat_scheme=heat_scheme,
                    Parent=self,
                )
            )

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)


class simSimpleAxialTransport(daeSimulation):
    def __init__(self, mass_scheme="upwind1", heat_scheme=None):
        daeSimulation.__init__(self)

        self.mass_scheme = mass_scheme
        self.heat_scheme = mass_scheme if heat_scheme is None else heat_scheme
        self.model = SimpleAxialTransportBed(
            "AxialSchemeTest",
            mass_scheme=self.mass_scheme,
            heat_scheme=self.heat_scheme,
        )

        self.c_init = 0.0
        self.c_step = 25.0
        self.T_init = 500.0
        self.T_step = 1500.0

    def SetUpParametersAndDomains(self):
        self.model.L_bed.SetValue(2.5 * m)
        self.model.u_const.SetValue(1.0 * m / s)
        self.model.Dax_const.SetValue(0.0 * m**2 / s)
        self.model.lambda_ax.SetValue(0.0 * J / (s * m * K))
        self.model.rhoCp_flow.SetValue(400.0 * J / (m**3 * K))
        self.model.rhoCp_store.SetValue(2.0e6 * J / (m**3 * K))
        self.model.c_in_const.SetValue(self.c_init * mol / m**3)
        self.model.T_in_const.SetValue(self.T_init * K)
        self.model.SetUniformAxialGrid(10)

    def SetUpVariables(self):
        nc = self.model.x_centers.NumberOfPoints
        nf = self.model.x_faces.NumberOfPoints

        dx_in = float(self.model.xval_cells.npyValues[0] - self.model.xval_faces.npyValues[0])
        u0 = self.model.u_const.GetValue()
        D0 = self.model.Dax_const.GetValue()
        rhoCp_flow = self.model.rhoCp_flow.GetValue()
        rhoCp_store = self.model.rhoCp_store.GetValue()
        lambda_ax = self.model.lambda_ax.GetValue()
        bed_length = self.model.L_bed.GetValue()

        if u0 <= 0.0:
            raise ValueError("This scheme test harness currently assumes a positive superficial velocity.")

        inlet_mass_flux0 = u0 * self.c_init - D0 * (self.c_init - self.c_init) / dx_in
        inlet_heat_flux0 = rhoCp_flow * u0 * self.T_init - lambda_ax * (self.T_init - self.T_init) / dx_in

        for idx_cell in range(nc):
            self.model.c.SetInitialCondition(idx_cell, self.c_init * mol / m**3)
            self.model.T.SetInitialCondition(idx_cell, self.T_init * K)

        for idx_face in range(nf):
            self.model.N_face.SetInitialGuess(idx_face, inlet_mass_flux0 * mol / (s * m**2))
            self.model.Q_face.SetInitialGuess(idx_face, inlet_heat_flux0 * J / (s * m**2))

        self.model.material_in_total.SetInitialCondition(0.0 * mol / m**2)
        self.model.material_out_total.SetInitialCondition(0.0 * mol / m**2)
        self.model.material_bed_total.SetInitialGuess(self.c_init * bed_length * mol / m**2)
        self.model.heat_in_total.SetInitialCondition(0.0 * J / m**2)
        self.model.heat_out_total.SetInitialCondition(0.0 * J / m**2)
        self.model.heat_bed_total.SetInitialGuess(rhoCp_store * self.T_init * bed_length * J / m**2)

        # Trigger simultaneous concentration and temperature fronts after consistent initialization.
        self.model.c_in_const.SetValue(self.c_step * mol / m**3)
        self.model.T_in_const.SetValue(self.T_step * K)


class simAxialTransportBatch(daeSimulation):
    def __init__(self, case_specs=None):
        daeSimulation.__init__(self)

        if case_specs is None:
            case_specs = [
                {"name": _format_case_name(scheme_name, scheme_name), "mass_scheme": scheme_name, "heat_scheme": scheme_name}
                for scheme_name in SUPPORTED_SCHEMES
            ]

        self.case_specs = list(case_specs)
        self.model = AxialSchemeBatch("AxialSchemeBatch", self.case_specs)

        self.c_init = 0.0
        self.c_step = 25.0
        self.T_init = 500.0
        self.T_step = 1500.0

    def SetUpParametersAndDomains(self):
        for case_model in self.model.case_models:
            case_model.L_bed.SetValue(2.5 * m)
            case_model.u_const.SetValue(1.0 * m / s)
            case_model.Dax_const.SetValue(0.0 * m**2 / s)
            case_model.lambda_ax.SetValue(0.0 * J / (s * m * K))
            case_model.rhoCp_flow.SetValue(400.0 * J / (m**3 * K))
            case_model.rhoCp_store.SetValue(2.0e6 * J / (m**3 * K))
            case_model.c_in_const.SetValue(self.c_init * mol / m**3)
            case_model.T_in_const.SetValue(self.T_init * K)
            case_model.SetUniformAxialGrid(10)

    def SetUpVariables(self):
        for case_model in self.model.case_models:
            nc = case_model.x_centers.NumberOfPoints
            nf = case_model.x_faces.NumberOfPoints

            dx_in = float(case_model.xval_cells.npyValues[0] - case_model.xval_faces.npyValues[0])
            u0 = case_model.u_const.GetValue()
            D0 = case_model.Dax_const.GetValue()
            rhoCp_flow = case_model.rhoCp_flow.GetValue()
            rhoCp_store = case_model.rhoCp_store.GetValue()
            lambda_ax = case_model.lambda_ax.GetValue()
            bed_length = case_model.L_bed.GetValue()

            if u0 <= 0.0:
                raise ValueError("This scheme test harness currently assumes a positive superficial velocity.")

            inlet_mass_flux0 = u0 * self.c_init - D0 * (self.c_init - self.c_init) / dx_in
            inlet_heat_flux0 = rhoCp_flow * u0 * self.T_init - lambda_ax * (self.T_init - self.T_init) / dx_in

            for idx_cell in range(nc):
                case_model.c.SetInitialCondition(idx_cell, self.c_init * mol / m**3)
                case_model.T.SetInitialCondition(idx_cell, self.T_init * K)

            for idx_face in range(nf):
                case_model.N_face.SetInitialGuess(idx_face, inlet_mass_flux0 * mol / (s * m**2))
                case_model.Q_face.SetInitialGuess(idx_face, inlet_heat_flux0 * J / (s * m**2))

            case_model.material_in_total.SetInitialCondition(0.0 * mol / m**2)
            case_model.material_out_total.SetInitialCondition(0.0 * mol / m**2)
            case_model.material_bed_total.SetInitialGuess(self.c_init * bed_length * mol / m**2)
            case_model.heat_in_total.SetInitialCondition(0.0 * J / m**2)
            case_model.heat_out_total.SetInitialCondition(0.0 * J / m**2)
            case_model.heat_bed_total.SetInitialGuess(rhoCp_store * self.T_init * bed_length * J / m**2)

            case_model.c_in_const.SetValue(self.c_step * mol / m**3)
            case_model.T_in_const.SetValue(self.T_step * K)


def _set_reporting_on(simulation):
    simulation.model.SetReportingOn(True)
    for case_model in getattr(simulation.model, "case_models", []):
        case_model.SetReportingOn(True)


def _run_with_noop_reporter(simulation, reporting_interval, time_horizon, report_time_derivatives=True):
    _set_reporting_on(simulation)
    simulation.ReportTimeDerivatives = report_time_derivatives
    simulation.ReportingInterval = reporting_interval
    simulation.TimeHorizon = time_horizon

    solver = daeIDAS()
    reporter = daeNoOpDataReporter()
    log = daePythonStdOutLog()
    log.PrintProgress = False

    simulation.Initialize(solver, reporter, log)
    simulation.SolveInitial()
    simulation.Run()
    return reporter


def _sanitize_name(name):
    safe_chars = []
    for char in name:
        if char.isalnum() or char in ("-", "_"):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars).strip("_")
    return sanitized or "case"


def _get_reported_variable(reporter, variable_path):
    try:
        return reporter.Process.dictVariables[variable_path]
    except KeyError as exc:
        available = sorted(reporter.Process.dictVariables.keys())
        raise KeyError(
            f"Reported variable '{variable_path}' was not found. "
            f"Available variables: {available}"
        ) from exc


def _extract_case_results(
    reporter,
    variable_prefix,
    case_name,
    mass_scheme,
    heat_scheme,
    rhoCp_store_value,
):
    c_var = _get_reported_variable(reporter, f"{variable_prefix}.c")
    t_var = _get_reported_variable(reporter, f"{variable_prefix}.T")
    n_var = _get_reported_variable(reporter, f"{variable_prefix}.N_face")
    q_var = _get_reported_variable(reporter, f"{variable_prefix}.Q_face")
    material_in_var = _get_reported_variable(reporter, f"{variable_prefix}.material_in_total")
    material_out_var = _get_reported_variable(reporter, f"{variable_prefix}.material_out_total")
    material_bed_var = _get_reported_variable(reporter, f"{variable_prefix}.material_bed_total")
    heat_in_var = _get_reported_variable(reporter, f"{variable_prefix}.heat_in_total")
    heat_out_var = _get_reported_variable(reporter, f"{variable_prefix}.heat_out_total")
    heat_bed_var = _get_reported_variable(reporter, f"{variable_prefix}.heat_bed_total")

    time_values = np.asarray(c_var.TimeValues, dtype=float)
    x_cell_values = np.asarray(c_var.Domains[0].Points, dtype=float)
    x_face_values = np.asarray(n_var.Domains[0].Points, dtype=float)

    c_values = np.asarray(c_var.Values, dtype=float)
    t_values = np.asarray(t_var.Values, dtype=float)
    n_values = np.asarray(n_var.Values, dtype=float)
    q_values = np.asarray(q_var.Values, dtype=float)
    material_in_total = np.asarray(material_in_var.Values, dtype=float).reshape(-1)
    material_out_total = np.asarray(material_out_var.Values, dtype=float).reshape(-1)
    material_bed_total = np.asarray(material_bed_var.Values, dtype=float).reshape(-1)
    heat_in_total = np.asarray(heat_in_var.Values, dtype=float).reshape(-1)
    heat_out_total = np.asarray(heat_out_var.Values, dtype=float).reshape(-1)
    heat_bed_total = np.asarray(heat_bed_var.Values, dtype=float).reshape(-1)

    net_mass_boundary_flux = n_values[:, -1] - n_values[:, 0]
    net_energy_boundary_flux = q_values[:, -1] - q_values[:, 0]

    cumulative_mass_boundary = material_out_total - material_in_total
    cumulative_energy_boundary = heat_out_total - heat_in_total

    material_balance_error = (material_bed_total - material_bed_total[0]) + cumulative_mass_boundary
    heat_balance_error = (heat_bed_total - heat_bed_total[0]) + cumulative_energy_boundary

    return {
        "case_name": case_name,
        "mass_scheme": mass_scheme,
        "heat_scheme": heat_scheme,
        "time_s": time_values,
        "x_cell_m": x_cell_values,
        "x_face_m": x_face_values,
        "c_mol_per_m3": c_values,
        "T_K": t_values,
        "N_face_mol_per_m2_s": n_values,
        "Q_face_J_per_m2_s": q_values,
        "material_in_total_mol_per_m2": material_in_total,
        "material_out_total_mol_per_m2": material_out_total,
        "material_bed_total_mol_per_m2": material_bed_total,
        "heat_in_total_J_per_m2": heat_in_total,
        "heat_out_total_J_per_m2": heat_out_total,
        "heat_bed_total_J_per_m2": heat_bed_total,
        "net_mass_boundary_flux_mol_per_m2_s": net_mass_boundary_flux,
        "net_energy_boundary_flux_J_per_m2_s": net_energy_boundary_flux,
        "cumulative_mass_boundary_mol_per_m2": cumulative_mass_boundary,
        "cumulative_energy_boundary_J_per_m2": cumulative_energy_boundary,
        "material_balance_error_mol_per_m2": material_balance_error,
        "heat_balance_error_J_per_m2": heat_balance_error,
        "mass_inventory_mol_per_m2": material_bed_total,
        "energy_inventory_J_per_m2": heat_bed_total,
        "mass_balance_error_mol_per_m2": material_balance_error,
        "energy_balance_error_J_per_m2": heat_balance_error,
        "rhoCp_store_J_per_m3_K": rhoCp_store_value,
    }


def _build_profile_table(time_values, x_values, field_a, field_b):
    n_time = time_values.size
    n_space = x_values.size
    return np.column_stack(
        (
            np.repeat(time_values, n_space),
            np.tile(x_values, n_time),
            field_a.reshape(-1),
            field_b.reshape(-1),
        )
    )


def _build_conservation_table(case_data):
    return np.column_stack(
        (
            case_data["time_s"],
            case_data["material_in_total_mol_per_m2"],
            case_data["material_out_total_mol_per_m2"],
            case_data["material_bed_total_mol_per_m2"],
            case_data["material_balance_error_mol_per_m2"],
            case_data["heat_in_total_J_per_m2"],
            case_data["heat_out_total_J_per_m2"],
            case_data["heat_bed_total_J_per_m2"],
            case_data["heat_balance_error_J_per_m2"],
        )
    )


def _write_numeric_csv(path, data, header):
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def export_axial_scheme_results(case_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "scheme_comparison.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_name",
                "mass_scheme",
                "heat_scheme",
                "max_abs_material_balance_error_mol_per_m2",
                "max_abs_heat_balance_error_J_per_m2",
            ]
        )

        for case_data in case_results:
            case_stem = _sanitize_name(case_data["case_name"])

            cell_profiles = _build_profile_table(
                case_data["time_s"],
                case_data["x_cell_m"],
                case_data["c_mol_per_m3"],
                case_data["T_K"],
            )
            _write_numeric_csv(
                output_dir / f"{case_stem}_cell_profiles.csv",
                cell_profiles,
                "time_s,x_cell_m,c_mol_per_m3,T_K",
            )

            face_fluxes = _build_profile_table(
                case_data["time_s"],
                case_data["x_face_m"],
                case_data["N_face_mol_per_m2_s"],
                case_data["Q_face_J_per_m2_s"],
            )
            _write_numeric_csv(
                output_dir / f"{case_stem}_face_fluxes.csv",
                face_fluxes,
                "time_s,x_face_m,N_face_mol_per_m2_s,Q_face_J_per_m2_s",
            )

            conservation = _build_conservation_table(case_data)
            _write_numeric_csv(
                output_dir / f"{case_stem}_conservation.csv",
                conservation,
                (
                    "time_s,"
                    "material_in_total_mol_per_m2,"
                    "material_out_total_mol_per_m2,"
                    "material_bed_total_mol_per_m2,"
                    "material_balance_error_mol_per_m2,"
                    "heat_in_total_J_per_m2,"
                    "heat_out_total_J_per_m2,"
                    "heat_bed_total_J_per_m2,"
                    "heat_balance_error_J_per_m2"
                ),
            )

            writer.writerow(
                [
                    case_data["case_name"],
                    case_data["mass_scheme"],
                    case_data["heat_scheme"],
                    float(np.max(np.abs(case_data["material_balance_error_mol_per_m2"]))),
                    float(np.max(np.abs(case_data["heat_balance_error_J_per_m2"]))),
                ]
            )


def consoleRun(
    mass_scheme="upwind1",
    heat_scheme=None,
    output_dir=None,
    reporting_interval=0.01,
    time_horizon=10.0,
):
    simulation = simSimpleAxialTransport(mass_scheme=mass_scheme, heat_scheme=heat_scheme)

    try:
        reporter = _run_with_noop_reporter(
            simulation,
            reporting_interval=reporting_interval,
            time_horizon=time_horizon,
        )
        case_name = _format_case_name(simulation.mass_scheme, simulation.heat_scheme)
        case_results = [
            _extract_case_results(
                reporter,
                simulation.model.Name,
                case_name,
                simulation.mass_scheme,
                simulation.heat_scheme,
                float(simulation.model.rhoCp_store.GetValue()),
            )
        ]
        if output_dir is not None:
            export_axial_scheme_results(case_results, output_dir)
        return case_results[0]
    finally:
        try:
            simulation.Finalize()
        except Exception:
            pass


def consoleRunAll(case_specs=None, output_dir=None, reporting_interval=0.01, time_horizon=10.0):
    simulation = simAxialTransportBatch(case_specs=case_specs)

    try:
        reporter = _run_with_noop_reporter(
            simulation,
            reporting_interval=reporting_interval,
            time_horizon=time_horizon,
        )
        case_results = []
        variable_prefix_root = simulation.model.Name
        for case_model, case_spec in zip(simulation.model.case_models, simulation.model.case_specs):
            case_results.append(
                _extract_case_results(
                    reporter,
                    f"{variable_prefix_root}.{case_model.Name}",
                    case_spec["name"],
                    case_spec["mass_scheme"],
                    case_spec["heat_scheme"],
                    float(case_model.rhoCp_store.GetValue()),
                )
            )

        if output_dir is not None:
            export_axial_scheme_results(case_results, output_dir)
        return case_results
    finally:
        try:
            simulation.Finalize()
        except Exception:
            pass


def _parse_export_args(argv):
    parser = argparse.ArgumentParser(
        description="Run the axial-scheme test case without the GUI and save CSV results."
    )
    parser.add_argument("--export-dir", required=True, help="Directory where the CSV files will be written.")
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single mass/heat scheme pair instead of the default batch run.",
    )
    parser.add_argument("--mass-scheme", default="upwind1", help="Mass-flux reconstruction scheme.")
    parser.add_argument(
        "--heat-scheme",
        default=None,
        help="Heat-flux reconstruction scheme. Defaults to the mass scheme when omitted.",
    )
    parser.add_argument("--reporting-interval", type=float, default=0.01, help="Result reporting interval in seconds.")
    parser.add_argument("--time-horizon", type=float, default=10.0, help="Simulation end time in seconds.")
    return parser.parse_args(argv)


def guiRun(qtApp, mass_scheme="upwind1", heat_scheme=None):
    simulation = simSimpleAxialTransport(mass_scheme=mass_scheme, heat_scheme=heat_scheme)
    _set_reporting_on(simulation)
    simulation.ReportTimeDerivatives = True
    simulation.ReportingInterval = 100
    simulation.TimeHorizon = 100000
    simulator = daeSimulator(qtApp, simulation=simulation)
    simulator.exec()


def guiRunAll(qtApp, case_specs=None):
    simulation = simAxialTransportBatch(case_specs=case_specs)
    _set_reporting_on(simulation)
    simulation.ReportTimeDerivatives = True
    simulation.ReportingInterval = 1
    simulation.TimeHorizon = 10000
    simulator = daeSimulator(qtApp, simulation=simulation)
    simulator.exec()


if __name__ == "__main__":
    if "--export-dir" in sys.argv[1:]:
        args = _parse_export_args(sys.argv[1:])
        if args.single:
            consoleRun(
                mass_scheme=args.mass_scheme,
                heat_scheme=args.heat_scheme,
                output_dir=args.export_dir,
                reporting_interval=args.reporting_interval,
                time_horizon=args.time_horizon,
            )
        else:
            consoleRunAll(
                output_dir=args.export_dir,
                reporting_interval=args.reporting_interval,
                time_horizon=args.time_horizon,
            )
    else:
        qtApp = daeCreateQtApplication(sys.argv)

        # Useful first comparisons:
        # guiRun(qtApp, mass_scheme="central", heat_scheme="central")
        # guiRun(qtApp, mass_scheme="central", heat_scheme="upwind1")
        guiRunAll(qtApp)
        # guiRun(qtApp, mass_scheme="upwind1", heat_scheme="upwind1")
        # guiRun(qtApp, mass_scheme="upwind1", heat_scheme="central")
        # guiRun(qtApp, mass_scheme="muscl_minmod", heat_scheme="muscl_minmod")
        # guiRun(qtApp, mass_scheme="weno3", heat_scheme="weno3")
