"""DAETools simulation construction, solver selection, execution, and cleanup."""

from __future__ import annotations

from importlib import import_module
import os

from daetools.pyDAE import (
    daeGetConfig,
    daeIDAS,
    daeNoOpDataReporter,
    daePythonStdOutLog,
    daeSimulation,
)

from .config import Case
from .initialization import apply_initial_state, calculate_initial_state, configure_model
from .kinetics import resolve_kinetics_hooks
from .model import PackedBedModel
from .programs import DEFAULT_SMOOTH_RAMP_WIDTH_S
from .reactions import build_reaction_network
from .reports import reporting_targets


_SOLVER_REGISTRY = {
    "trilinos_klu": ("trilinos", "pyTrilinos", "daeCreateTrilinosSolver", ("Amesos_Klu", "")),
    "trilinos_umfpack": ("trilinos", "pyTrilinos", "daeCreateTrilinosSolver", ("Amesos_Umfpack", "")),
    "trilinos_lapack": ("trilinos", "pyTrilinos", "daeCreateTrilinosSolver", ("Amesos_Lapack", "")),
    "trilinos_aztecoo": ("trilinos", "pyTrilinos", "daeCreateTrilinosSolver", ("AztecOO", "ILUT")),
    "trilinos_aztecoo_ifpack": ("trilinos", "pyTrilinos", "daeCreateTrilinosSolver", ("AztecOO_Ifpack", "ILU")),
    "trilinos_aztecoo_ml": ("trilinos", "pyTrilinos", "daeCreateTrilinosSolver", ("AztecOO_ML", "DD-ML")),
    "superlu": ("superlu", "pySuperLU", "daeCreateSuperLUSolver", ()),
    "superlu_mt": ("superlu_mt", "pySuperLU_MT", "daeCreateSuperLUSolver", ()),
    "intel_pardiso": ("intel_pardiso", "pyIntelPardiso", "daeCreateIntelPardisoSolver", ()),
}


class PackedBedSimulation(daeSimulation):
    def __init__(
        self,
        case: Case,
        property_registry,
        *,
        smooth_ramp_width_s: float = DEFAULT_SMOOTH_RAMP_WIDTH_S,
    ):
        daeSimulation.__init__(self)
        self.case = case
        self.property_registry = property_registry
        self.smooth_ramp_width_s = float(smooth_ramp_width_s)
        reaction_network = build_reaction_network(
            case.chemistry.reaction_ids,
            case.chemistry.gas_species,
            case.solids.solid_species,
            families=case.reaction_families,
        )
        reaction_rate_hooks = resolve_kinetics_hooks(
            reaction_network,
            case.reaction_families,
        )
        self.model = PackedBedModel(
            case.run.simulation.system_name,
            case,
            reaction_network,
            reaction_rate_hooks,
            property_registry,
            smooth_ramp_width_s=self.smooth_ramp_width_s,
        )
        self.initial_state = None

    def SetUpParametersAndDomains(self):
        self.initial_state = calculate_initial_state(
            self.case,
            self.property_registry,
            smooth_ramp_width_s=self.smooth_ramp_width_s,
        )
        configure_model(self.model, self.case, self.initial_state)

    def SetUpVariables(self):
        if self.initial_state is None:
            raise RuntimeError("Initial state must be calculated before variables are configured.")
        apply_initial_state(self.model, self.initial_state)


def configure_threads(threads: int) -> None:
    """Configure execution threads; zero keeps environment limits and uses DAETools' default."""

    if threads < 0:
        raise ValueError("threads must not be negative.")
    if threads > 0:
        value = str(threads)
        os.environ.update(
            MKL_NUM_THREADS=value,
            OMP_NUM_THREADS=value,
            MKL_DYNAMIC="FALSE",
            OMP_DYNAMIC="FALSE",
        )
        os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

    daetools_config = daeGetConfig()
    daetools_config.SetString(
        "daetools.core.equations.evaluationMode",
        "computeStack_OpenMP",
    )
    daetools_config.SetInteger(
        "daetools.core.equations.computeStack_OpenMP.numThreads",
        threads,
    )


def _configure_aztecoo_ifpack(linear_solver):
    from daetools.solvers.aztecoo_options import daeAztecOptions

    linear_solver.NumIters = 1000
    linear_solver.Tolerance = 1.0e-8
    parameters = linear_solver.ParameterList
    parameters.set_int("AZ_solver", daeAztecOptions.AZ_gmres)
    parameters.set_int("AZ_kspace", 100)
    parameters.set_int("AZ_scaling", daeAztecOptions.AZ_none)
    parameters.set_int("AZ_reorder", 0)
    parameters.set_int("AZ_conv", daeAztecOptions.AZ_r0)
    parameters.set_int("AZ_keep_info", 1)
    parameters.set_int("AZ_output", daeAztecOptions.AZ_none)
    parameters.set_int("AZ_diagnostics", daeAztecOptions.AZ_none)
    parameters.set_int("fact: level-of-fill", 3)
    parameters.set_float("fact: absolute threshold", 1.0e-5)
    parameters.set_float("fact: relative threshold", 1.0)
    return linear_solver


def create_linear_solver(name: str):
    """Create one solver from the explicit supported registry."""

    try:
        module_name, backend_name, factory_name, arguments = _SOLVER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported linear solver: {name}") from exc
    module = import_module(f"daetools.solvers.{module_name}")
    factory = getattr(getattr(module, backend_name), factory_name)
    solver = factory(*arguments)
    return _configure_aztecoo_ifpack(solver) if name.endswith("_ifpack") else solver


def _configure_reporting(
    simulation: PackedBedSimulation,
    *,
    include_plot_variables: bool,
) -> None:
    variable_names = reporting_targets(
        simulation.case.run.outputs.requested_reports,
        include_plot_variables=include_plot_variables,
    )
    simulation.model.SetReportingOn(False)
    missing = [name for name in variable_names if name not in simulation.model.dictVariables]
    if missing:
        raise ValueError(
            "Cannot enable reporting for unknown variables: "
            f"{', '.join(missing)}. Available entries: "
            f"{', '.join(sorted(simulation.model.dictVariables))}."
        )
    for name in variable_names:
        simulation.model.dictVariables[name].ReportingOn = True


def execute_simulation(
    simulation: PackedBedSimulation,
    *,
    include_plot_variables: bool = False,
    data_reporter=None,
    after_initialize=None,
):
    """Initialize, run, finalize, and flush reports through one execution path."""

    case = simulation.case
    configure_threads(case.run.solver.threads)
    _configure_reporting(
        simulation,
        include_plot_variables=include_plot_variables,
    )
    simulation.ReportTimeDerivatives = case.run.simulation.report_time_derivatives
    simulation.ReportingInterval = case.run.simulation.reporting_interval_s
    simulation.TimeHorizon = case.run.simulation.time_horizon_s

    solver = daeIDAS()
    solver.RelativeTolerance = case.run.solver.relative_tolerance
    solver.SetLASolver(create_linear_solver(case.run.solver.name))
    reporter = data_reporter if data_reporter is not None else daeNoOpDataReporter()
    if data_reporter is not None and hasattr(reporter, "IsConnected") and not reporter.IsConnected():
        process_name = case.run.simulation.system_name
        if not reporter.Connect(str(case.output_directory), process_name):
            raise RuntimeError(f"Cannot connect data reporter for process '{process_name}'.")

    log = daePythonStdOutLog()
    log.PrintProgress = False
    initialized = False
    try:
        simulation.Initialize(solver, reporter, log)
        initialized = True
        if after_initialize is not None:
            after_initialize(simulation, solver)
        simulation.SolveInitial()
        simulation.Run()
    finally:
        if initialized:
            simulation.Finalize()

    if (
        data_reporter is not None
        and hasattr(reporter, "write_outputs")
        and not getattr(reporter, "_written", False)
        and getattr(reporter, "write_error", None) is None
    ):
        try:
            reporter.write_outputs()
        except Exception as exc:
            raise RuntimeError("Data reporter failed while writing simulation reports.") from exc

    write_error = getattr(reporter, "write_error", None)
    if write_error is not None:
        raise RuntimeError("Data reporter failed while writing simulation reports.") from write_error
    return reporter


__all__ = (
    "PackedBedSimulation",
    "configure_threads",
    "create_linear_solver",
    "execute_simulation",
)
