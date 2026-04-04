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
- no EOS, no Ergun closure, no reactions

Recommended first comparisons:
- mass_scheme="upwind1", heat_scheme="upwind1"
- mass_scheme="upwind1", heat_scheme="central"
- mass_scheme="muscl_minmod", heat_scheme="muscl_minmod"
- mass_scheme="weno3", heat_scheme="weno3"
"""

import sys

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

        for idx_cell in range(Nc):
            dx = face_coords[idx_cell + 1] - face_coords[idx_cell]

            eq = self.CreateEquation(f"mass_balance_cell_{idx_cell}")
            eq.Residual = dt(self.c(idx_cell)) + (self.N_face(idx_cell + 1) - self.N_face(idx_cell)) / dx

            eq = self.CreateEquation(f"heat_balance_cell_{idx_cell}")
            eq.Residual = self.rhoCp_store() * dt(self.T(idx_cell)) + (
                self.Q_face(idx_cell + 1) - self.Q_face(idx_cell)
            ) / dx


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
        lambda_ax = self.model.lambda_ax.GetValue()

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
            lambda_ax = case_model.lambda_ax.GetValue()

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

            case_model.c_in_const.SetValue(self.c_step * mol / m**3)
            case_model.T_in_const.SetValue(self.T_step * K)


def guiRun(qtApp, mass_scheme="upwind1", heat_scheme=None):
    simulation = simSimpleAxialTransport(mass_scheme=mass_scheme, heat_scheme=heat_scheme)
    simulation.model.SetReportingOn(True)
    simulation.ReportTimeDerivatives = True
    simulation.ReportingInterval = 100
    simulation.TimeHorizon = 100000
    simulator = daeSimulator(qtApp, simulation=simulation)
    simulator.exec()


def guiRunAll(qtApp, case_specs=None):
    simulation = simAxialTransportBatch(case_specs=case_specs)
    simulation.model.SetReportingOn(True)
    for case_model in simulation.model.case_models:
        case_model.SetReportingOn(True)
    simulation.ReportTimeDerivatives = True
    simulation.ReportingInterval = 100
    simulation.TimeHorizon = 100000
    simulator = daeSimulator(qtApp, simulation=simulation)
    simulator.exec()


if __name__ == "__main__":
    qtApp = daeCreateQtApplication(sys.argv)

    # Useful first comparisons:
    # guiRun(qtApp, mass_scheme="central", heat_scheme="central")
    # guiRun(qtApp, mass_scheme="central", heat_scheme="upwind1")
    guiRunAll(qtApp)
    # guiRun(qtApp, mass_scheme="upwind1", heat_scheme="upwind1")
    # guiRun(qtApp, mass_scheme="upwind1", heat_scheme="central")
    # guiRun(qtApp, mass_scheme="muscl_minmod", heat_scheme="muscl_minmod")
    # guiRun(qtApp, mass_scheme="weno3", heat_scheme="weno3")
