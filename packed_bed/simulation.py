"""DAETools simulation construction, solver selection, execution, and cleanup."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
import os
from pathlib import Path
from time import perf_counter
import traceback

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
from .properties import PROPERTY_REGISTRY
from .reactions import build_reaction_network
from .reports import (
    RunResult,
    compute_balance_errors,
    create_dataset_reporter,
    reporting_targets,
    write_run_manifest,
)


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
            BLIS_NUM_THREADS=value,
            MKL_NUM_THREADS=value,
            NUMEXPR_NUM_THREADS=value,
            OMP_NUM_THREADS=value,
            OPENBLAS_NUM_THREADS=value,
            VECLIB_MAXIMUM_THREADS=value,
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


def configure_idas(solver_config) -> None:
    """Apply per-case IDAS nonlinear and algebraic-error controls."""

    daetools_config = daeGetConfig()
    daetools_config.SetBoolean(
        "daetools.IDAS.SuppressAlg",
        solver_config.suppress_algebraic_errors,
    )
    daetools_config.SetInteger(
        "daetools.IDAS.MaxNonlinIters",
        solver_config.max_nonlinear_iterations,
    )
    daetools_config.SetFloat(
        "daetools.IDAS.NonlinConvCoef",
        solver_config.nonlinear_convergence_coefficient,
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
) -> None:
    variable_names = reporting_targets(simulation.case.run.outputs.requested_reports)
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
    data_reporter=None,
    after_initialize=None,
):
    """Initialize, run, finalize, and flush reports through one execution path."""

    case = simulation.case
    configure_threads(case.run.solver.threads)
    configure_idas(case.run.solver)
    _configure_reporting(simulation)
    simulation.ReportTimeDerivatives = case.run.simulation.report_time_derivatives
    simulation.ReportingInterval = case.run.simulation.reporting_interval_s
    simulation.TimeHorizon = case.run.simulation.time_horizon_s

    solver = daeIDAS()
    solver.RelativeTolerance = case.run.solver.relative_tolerance
    solver.SetLASolver(create_linear_solver(case.run.solver.name))
    reporter = data_reporter if data_reporter is not None else daeNoOpDataReporter()
    if data_reporter is not None and not reporter.IsConnected():
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

    finish = getattr(reporter, "finish", None)
    if finish is not None:
        finish()
    return reporter


def run_case(
    case: Case,
    property_registry=None,
    artifact_paths: dict[str, Path] | None = None,
    *,
    retain_reporter: bool = False,
) -> RunResult:
    """Run one resolved case and write its dataset and manifest."""

    if property_registry is None:
        property_registry = PROPERTY_REGISTRY

    case.output_directory.mkdir(parents=True, exist_ok=True)
    solver_artifacts: dict[str, Path] = {}
    stage = "model construction"
    dataset_reporter = None
    result = RunResult(case=case, output_directory=case.output_directory)
    started_at = perf_counter()
    try:
        simulation = PackedBedSimulation(case, property_registry)
        dataset_reporter = create_dataset_reporter(case)
        after_initialize = None
        if case.run.outputs.solver_incidence_matrix:
            from .incidence_matrix import write_solver_incidence_artifacts

            def after_initialize(initialized_simulation, _solver):
                solver_artifacts.update(
                    write_solver_incidence_artifacts(
                        model=initialized_simulation.model,
                        output_dir=case.artifacts_directory,
                    )
                )

        stage = "solver execution"
        reporter = execute_simulation(
            simulation,
            data_reporter=dataset_reporter,
            after_initialize=after_initialize,
        )
        result = replace(
            result,
            results_path=dataset_reporter.results_path,
            runtime_s=perf_counter() - started_at,
            artifact_paths={**dict(artifact_paths or {}), **solver_artifacts},
            reporter=reporter if retain_reporter else None,
            balance_errors=compute_balance_errors(dataset_reporter.results_path),
        )
        if case.run.outputs.requested_plots:
            from .plotting import _render_requested_plots

            try:
                plot_result = _render_requested_plots(
                    dataset_reporter.results_path,
                    case.run.outputs.requested_plots,
                    case.artifacts_directory,
                )
            except Exception as exc:
                plot_paths = {}
                plot_errors = {
                    plot_id: str(exc) for plot_id in case.run.outputs.requested_plots
                }
            else:
                plot_paths = plot_result.paths
                plot_errors = plot_result.errors
            result = replace(
                result,
                artifact_paths={**result.artifact_paths, **plot_paths},
                plot_errors=plot_errors,
            )
        stage = "manifest writing"
        return replace(result, manifest_path=write_run_manifest(result))
    except Exception as exc:
        if stage != "manifest writing":
            failed_result = replace(
                result, status="failed", runtime_s=perf_counter() - started_at,
                results_path=getattr(dataset_reporter, "results_path", None),
                artifact_paths={**dict(artifact_paths or {}), **solver_artifacts, **result.artifact_paths},
            )
            try:
                write_run_manifest(
                    failed_result,
                    failure_stage=stage,
                    traceback_text=traceback.format_exc(),
                )
            except Exception as manifest_error:
                exc.add_note(f"Could not write failure manifest: {manifest_error}")
        raise


__all__ = (
    "PackedBedSimulation",
    "configure_idas",
    "configure_threads",
    "create_linear_solver",
    "execute_simulation",
    "run_case",
)
