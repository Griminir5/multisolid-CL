from __future__ import annotations

import math

from daetools.pyDAE import *

from pyUnits import s

from . import solver as _base
from .config import ModelConfig, RunBundle, ScalarProgram, SolidConfig, VectorProgram
from .kinetics import resolve_kinetics_hooks
from .reactions import ReactionNetwork, build_reaction_network


SMOOTH_RAMP_WIDTH_S = 1.0


class CLBed_mass(_base.CLBed_mass):
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
        *,
        smooth_ramp_width_s=SMOOTH_RAMP_WIDTH_S,
        materialize_source_terms=False,
        materialize_solid_mole_fractions=False,
        Description="",
        Parent=None,
    ):
        super().__init__(
            Name,
            gas_species,
            solid_species,
            reaction_network=reaction_network,
            reaction_rate_hooks=reaction_rate_hooks,
            property_registry=property_registry,
            mass_scheme=mass_scheme,
            heat_scheme=heat_scheme,
            materialize_source_terms=materialize_source_terms,
            materialize_solid_mole_fractions=materialize_solid_mole_fractions,
            Description=Description,
            Parent=Parent,
        )
        if smooth_ramp_width_s <= 0.0:
            raise ValueError("smooth_ramp_width_s must be positive.")
        self.smooth_ramp_width_s = float(smooth_ramp_width_s)

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
            expression = expression + Constant(delta * units) * self._smooth_ramp_fraction(segment)
        return expression

    def _declare_program_equations(self, variable, default_expression, segments, units, equation_prefix):
        eq = self.CreateEquation(f"{equation_prefix}_smooth")
        eq.Residual = variable() - self._smooth_program_expression(default_expression, segments, units)

    def _declare_indexed_program_equations(self, variable, default_accessor, indexed_segments, units, equation_prefix):
        for gas_idx, segments in enumerate(indexed_segments):
            eq = self.CreateEquation(f"{equation_prefix}_{gas_idx:03d}_smooth")
            eq.Residual = variable(gas_idx) - self._smooth_program_expression(
                default_accessor(gas_idx),
                segments,
                units,
            )


class simBed(_base.simBed):
    def __init__(
        self,
        gas_species,
        solid_species,
        reaction_network: ReactionNetwork,
        reaction_rate_hooks,
        solid_config: SolidConfig,
        property_registry,
        mass_scheme,
        heat_scheme,
        inlet_flow_program: ScalarProgram,
        inlet_composition_program: VectorProgram,
        inlet_temperature_program: ScalarProgram,
        outlet_pressure_program: ScalarProgram,
        operation_time_horizon,
        model_config: ModelConfig,
        system_name,
        *,
        smooth_ramp_width_s=SMOOTH_RAMP_WIDTH_S,
        materialize_source_terms=False,
        materialize_solid_mole_fractions=False,
    ):
        daeSimulation.__init__(self)

        self.property_registry = property_registry
        self.gas_species = list(gas_species)
        self.solid_species = list(solid_species)
        self.reaction_network = reaction_network
        self.reaction_rate_hooks = tuple(reaction_rate_hooks)
        self.model_config = model_config
        self.solid_config = solid_config
        self.mass_scheme = mass_scheme
        self.heat_scheme = heat_scheme
        self.inlet_flow_program = inlet_flow_program
        self.inlet_composition_program = inlet_composition_program
        self.inlet_temperature_program = inlet_temperature_program
        self.outlet_pressure_program = outlet_pressure_program
        self.operation_time_horizon = operation_time_horizon
        self.system_name = system_name
        self.materialize_source_terms = bool(materialize_source_terms)
        self.materialize_solid_mole_fractions = bool(materialize_solid_mole_fractions)
        self.smooth_ramp_width_s = float(smooth_ramp_width_s)

        self.model = CLBed_mass(
            self.system_name,
            self.gas_species,
            self.solid_species,
            reaction_network=self.reaction_network,
            reaction_rate_hooks=self.reaction_rate_hooks,
            property_registry=self.property_registry,
            mass_scheme=self.mass_scheme,
            heat_scheme=self.heat_scheme,
            smooth_ramp_width_s=self.smooth_ramp_width_s,
            materialize_source_terms=self.materialize_source_terms,
            materialize_solid_mole_fractions=self.materialize_solid_mole_fractions,
        )
        self.model.SetOperationProgram(
            inlet_flow_program=self.inlet_flow_program,
            inlet_composition_program=self.inlet_composition_program,
            inlet_temperature_program=self.inlet_temperature_program,
            outlet_pressure_program=self.outlet_pressure_program,
        )


def assemble_simulation(
    run_bundle: RunBundle,
    *,
    property_registry,
    reaction_catalog,
    smooth_ramp_width_s=SMOOTH_RAMP_WIDTH_S,
) -> _base.SimulationAssembly:
    reaction_network = build_reaction_network(
        run_bundle.chemistry.reaction_ids,
        run_bundle.chemistry.gas_species,
        run_bundle.solids.solid_species,
        reaction_catalog=reaction_catalog,
    )
    reaction_rate_hooks = resolve_kinetics_hooks(reaction_network)

    inlet_flow_program = run_bundle.program.inlet_flow.compile_program(
        repeat=run_bundle.run.repeat_program,
        time_horizon=run_bundle.run.time_horizon_s,
    )
    inlet_composition_program = run_bundle.program.inlet_composition.compile_program(
        run_bundle.chemistry.gas_species,
        repeat=run_bundle.run.repeat_program,
        time_horizon=run_bundle.run.time_horizon_s,
    )
    inlet_temperature_program = run_bundle.program.inlet_temperature.compile_program(
        repeat=run_bundle.run.repeat_program,
        time_horizon=run_bundle.run.time_horizon_s,
    )
    outlet_pressure_program = run_bundle.program.outlet_pressure.compile_program(
        repeat=run_bundle.run.repeat_program,
        time_horizon=run_bundle.run.time_horizon_s,
    )
    requested_reports = set(run_bundle.run.outputs.requested_reports)
    materialize_source_terms = bool(requested_reports & {"gas_source", "solid_source"})
    materialize_solid_mole_fractions = "solid_mole_fraction" in requested_reports

    simulation = simBed(
        gas_species=run_bundle.chemistry.gas_species,
        solid_species=run_bundle.solids.solid_species,
        reaction_network=reaction_network,
        reaction_rate_hooks=reaction_rate_hooks,
        solid_config=run_bundle.solids,
        property_registry=property_registry,
        mass_scheme=run_bundle.run.mass_scheme,
        heat_scheme=run_bundle.run.heat_scheme,
        inlet_flow_program=inlet_flow_program,
        inlet_composition_program=inlet_composition_program,
        inlet_temperature_program=inlet_temperature_program,
        outlet_pressure_program=outlet_pressure_program,
        operation_time_horizon=run_bundle.run.time_horizon_s,
        model_config=run_bundle.run.model,
        system_name=run_bundle.run.system_name,
        smooth_ramp_width_s=smooth_ramp_width_s,
        materialize_source_terms=materialize_source_terms,
        materialize_solid_mole_fractions=materialize_solid_mole_fractions,
    )
    return _base.SimulationAssembly(run_bundle=run_bundle, simulation=simulation)


def run_assembled_simulation(
    assembly: _base.SimulationAssembly,
    *,
    report_ids=None,
    include_plot_variables=False,
    include_benchmark_snapshot=False,
):
    _base.configure_evaluation_mode()
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

    solver = _base.build_idas_solver(relative_tolerance=assembly.run_bundle.run.solver.relative_tolerance)
    reporter = daeNoOpDataReporter()
    log = daePythonStdOutLog()
    log.PrintProgress = False

    simulation.Initialize(solver, reporter, log)
    simulation.SolveInitial()
    simulation.Run()
    return reporter


CLBed_mass_base = _base.CLBed_mass
SimulationAssembly = _base.SimulationAssembly
build_idas_solver = _base.build_idas_solver
configure_evaluation_mode = _base.configure_evaluation_mode
guiRun = _base.guiRun


__all__ = [
    "CLBed_mass",
    "CLBed_mass_base",
    "SMOOTH_RAMP_WIDTH_S",
    "SimulationAssembly",
    "assemble_simulation",
    "build_idas_solver",
    "configure_evaluation_mode",
    "guiRun",
    "run_assembled_simulation",
    "simBed",
]
