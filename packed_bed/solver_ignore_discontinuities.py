from __future__ import annotations

import math

from daetools.pyDAE import daeNoOpDataReporter, daePythonStdOutLog, eDoNotStopAtDiscontinuity

from . import solver as _base


CLBed_mass = _base.CLBed_mass
SimulationAssembly = _base.SimulationAssembly
assemble_simulation = _base.assemble_simulation
build_idas_solver = _base.build_idas_solver
configure_evaluation_mode = _base.configure_evaluation_mode
guiRun = _base.guiRun
simBed = _base.simBed


def _integrate_until_time(simulation, target_time, *, max_step_s=None, tolerance=1e-12):
    current_time = float(simulation.CurrentTime)

    if target_time < current_time - tolerance:
        raise ValueError(
            f"Cannot integrate backwards from t={current_time:.16g} to t={target_time:.16g}."
        )
    if max_step_s is not None and max_step_s <= 0.0:
        raise ValueError("The maximum step size must be positive when substepping is enabled.")

    while current_time + tolerance < target_time:
        next_time = target_time
        if max_step_s is not None:
            next_time = min(next_time, current_time + max_step_s)
        if math.isclose(next_time, target_time, rel_tol=0.0, abs_tol=tolerance):
            next_time = target_time

        simulation.IntegrateUntilTime(next_time, eDoNotStopAtDiscontinuity)
        updated_time = float(simulation.CurrentTime)
        if updated_time <= current_time + tolerance:
            raise RuntimeError(
                f"Failed to advance the simulation from t={current_time:.16g} toward t={target_time:.16g}."
            )
        current_time = updated_time


def _reporting_times(simulation, *, tolerance=1e-12):
    reporting_interval = float(simulation.ReportingInterval)
    time_horizon = float(simulation.TimeHorizon)
    current_time = float(simulation.CurrentTime)

    if reporting_interval <= 0.0:
        raise ValueError("The reporting interval must be positive.")

    report_times = []
    full_steps = int(math.floor(time_horizon / reporting_interval + tolerance))
    for step_index in range(1, full_steps + 1):
        report_time = step_index * reporting_interval
        if report_time > current_time + tolerance:
            report_times.append(report_time)

    if time_horizon > current_time + tolerance:
        report_times.append(time_horizon)

    return _base._sorted_unique_times(report_times, tolerance=tolerance)


def _run_with_reporting_times(simulation, *, tolerance=1e-12):
    for report_time in _reporting_times(simulation, tolerance=tolerance):
        _integrate_until_time(simulation, report_time, tolerance=tolerance)
        simulation.ReportData(float(simulation.CurrentTime))


def _warm_start_first_reporting_interval(simulation, *, max_step_s=0.1):
    first_report_time = min(float(simulation.ReportingInterval), float(simulation.TimeHorizon))
    current_time = float(simulation.CurrentTime)
    tolerance = 1e-12

    if first_report_time <= current_time + tolerance:
        return
    if first_report_time <= max_step_s + tolerance:
        return

    warm_start_time = min(first_report_time, current_time + max(1.0, max_step_s))
    _integrate_until_time(
        simulation,
        warm_start_time,
        max_step_s=max_step_s,
        tolerance=tolerance,
    )
    if math.isclose(float(simulation.CurrentTime), first_report_time, rel_tol=0.0, abs_tol=tolerance):
        simulation.ReportData(float(simulation.CurrentTime))


def run_assembled_simulation(
    assembly: SimulationAssembly,
    *,
    report_ids=None,
    include_plot_variables=False,
    include_benchmark_snapshot=False,
):
    configure_evaluation_mode()
    simulation = assembly.simulation
    if report_ids is None:
        report_ids = _base._requested_report_ids(assembly)
    _base._set_reporting_on(
        simulation,
        report_ids,
        include_plot_variables=include_plot_variables,
        include_benchmark_snapshot=include_benchmark_snapshot,
    )
    simulation.ReportTimeDerivatives = assembly.run_bundle.run.report_time_derivatives
    simulation.ReportingInterval = assembly.run_bundle.run.reporting_interval_s
    simulation.TimeHorizon = assembly.run_bundle.run.time_horizon_s

    solver = build_idas_solver(relative_tolerance=assembly.run_bundle.run.solver.relative_tolerance)
    reporter = daeNoOpDataReporter()
    log = daePythonStdOutLog()
    log.PrintProgress = False

    simulation.Initialize(solver, reporter, log)
    simulation.SolveInitial()
    _warm_start_first_reporting_interval(simulation)
    _run_with_reporting_times(simulation)
    return reporter


__all__ = [
    "CLBed_mass",
    "SimulationAssembly",
    "assemble_simulation",
    "build_idas_solver",
    "configure_evaluation_mode",
    "guiRun",
    "run_assembled_simulation",
    "simBed",
]
