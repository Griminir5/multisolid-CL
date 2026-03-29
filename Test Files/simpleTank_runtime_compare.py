import math
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from daetools.pyDAE import *
from pyUnits import m, s  # this will not show up because pylance cannot get to .pyd files

volumetric_flow_type = daeVariableType(
    name="volumetric_flow_type",
    units=m**3 / s,
    lowerBound=-100000,
    upperBound=100000,
    initialGuess=10,
    absTolerance=1e-5,
)
length_type = daeVariableType(
    name="length_type",
    units=m,
    lowerBound=-100000,
    upperBound=100000,
    initialGuess=10,
    absTolerance=1e-5,
)
volume_type = daeVariableType(
    name="volume_type",
    units=m**3,
    lowerBound=-100000,
    upperBound=100000,
    initialGuess=10,
    absTolerance=1e-5,
)


@dataclass(frozen=True)
class ProgramStep:
    duration: float
    kind: Literal["hold", "ramp"]
    target: float | None = None


@dataclass(frozen=True)
class ProgramSegment:
    start_time: float
    end_time: float
    start_value: float
    end_value: float


class ScalarProgram:
    """Piecewise-linear schedule for a single scalar input."""

    def __init__(self, initial_value: float):
        self.initial_value = initial_value
        self.steps: list[ProgramStep] = []

    def add_ramp(self, duration, target):
        self.steps.append(ProgramStep(duration=duration, kind="ramp", target=target))

    def add_hold(self, duration):
        self.steps.append(ProgramStep(duration=duration, kind="hold"))

    def build(self, repeat=False, time_horizon=None):
        times = [0.0]
        values = [self.initial_value]

        current_time = 0.0
        current_value = self.initial_value

        while True:
            for step in self.steps:
                current_time += step.duration

                if step.kind == "hold":
                    next_value = current_value
                else:
                    next_value = step.target

                times.append(current_time)
                values.append(next_value)

                current_value = next_value

                if repeat and current_time >= time_horizon:
                    break

            if not repeat or current_time >= time_horizon or not self.steps:
                break

        if time_horizon is not None and times[-1] < time_horizon:
            times.append(time_horizon)
            values.append(current_value)

        return {
            "times": times,
            "values": values,
            "end_time": times[-1],
        }


inlet_flow_program = ScalarProgram(initial_value=10.0)
inlet_flow_program.add_hold(duration=10)
inlet_flow_program.add_ramp(100, 1.0)
inlet_flow_program.add_ramp(100, 2.0)
inlet_flow_program.add_ramp(100, 1.0)
inlet_flow_program.add_hold(duration=200)
inlet_flow_program.add_ramp(10, 3.0)
inlet_flow_program.add_hold(duration=20)


class SimpleTankBase(daeModel):
    def __init__(self, Name, Parent=None, Description="Simple cylindrical isothermal tank"):
        daeModel.__init__(self, Name, Parent, Description)

        self._inlet_program = None
        self._compiled_inlet_program = None

        self.Radius = daeParameter("Radius", m, self, "Cylindrical tank radius")
        self.Flow_in_default = daeParameter(
            "Flowrate_in",
            m**3 / s,
            self,
            "Default inlet volumetric flow used when no inlet program is attached",
        )
        self.Kflow = daeParameter("Flow_constant", m**2.5 / s, self, "FlowConstant")
        self.pi = daeParameter("&pi;", dimless, self, "Circle constant")

        self.Volume = daeVariable("Volume", volume_type, self, "Volume of liquid")
        self.Level = daeVariable("Level", length_type, self, "Liquid level")
        self.Flow_out = daeVariable("Flowrate_out", volumetric_flow_type, self, "Outlet volumetric flow")
        self.Flow_in_active = daeVariable(
            "Flowrate_in_active",
            volumetric_flow_type,
            self,
            "Inlet volumetric flow actually used by the tank equations",
        )

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        self._declare_active_inlet_flow_equations()

        eq = self.CreateEquation("Volume_level", "Equation which relates the liquid level and volume in a tank")
        eq.Residual = self.Volume() - self.Radius() * self.Radius() * self.pi() * self.Level()

        eq = self.CreateEquation("Volume_balance", "Differential equation that relates volume change to inflow/outflow")
        eq.Residual = dt(self.Volume()) - self.Flow_in_active() + self.Flow_out()

        eq = self.CreateEquation("Outflow_calc", "Determination of outflow amount from level")
        eq.Residual = self.Flow_out() - self.Kflow() * Sqrt(self.Level())

    def SetInletProgram(self, inlet_program, repeat=False, time_horizon=None):
        self._inlet_program = inlet_program
        self._compiled_inlet_program = inlet_program.build(repeat=repeat, time_horizon=time_horizon)
        self._after_setting_inlet_program()
        return self._compiled_inlet_program

    def ClearInletProgram(self):
        self._inlet_program = None
        self._compiled_inlet_program = None
        self._clear_inlet_program_cache()

    def _after_setting_inlet_program(self):
        pass

    def _clear_inlet_program_cache(self):
        pass

    def _declare_active_inlet_flow_equations(self):
        raise NotImplementedError


class SimpleTankInterpolation(SimpleTankBase):
    def __init__(self, Name, Parent=None, Description="Simple tank using interpolation-based inlet schedule"):
        SimpleTankBase.__init__(self, Name, Parent, Description)
        self._inlet_program_functions = {}

    def _clear_inlet_program_cache(self):
        self._inlet_program_functions = {}

    def _resolve_inlet_flow_input(self):
        if self._compiled_inlet_program is None:
            return self.Flow_in_default()

        schedule = self._compiled_inlet_program
        self._inlet_program_functions = {}
        self._inlet_program_functions["F_in"] = daeLinearInterpolationFunction(
            "F_in_schedule",
            self,
            volumetric_flow_type.Units,
            list(schedule["times"]),
            list(schedule["values"]),
            Time(),
        )
        return self._inlet_program_functions["F_in"]()

    def _declare_active_inlet_flow_equations(self):
        eq = self.CreateEquation(
            "active_inlet_flow",
            "Map the active inlet flow to either the constant default or the interpolated schedule.",
        )
        eq.Residual = self.Flow_in_active() - self._resolve_inlet_flow_input()


class SimpleTankPiecewise(SimpleTankBase):
    def __init__(self, Name, Parent=None, Description="Simple tank using piecewise native equations for the inlet schedule"):
        SimpleTankBase.__init__(self, Name, Parent, Description)
        self._compiled_inlet_segments: list[ProgramSegment] = []

    def _after_setting_inlet_program(self):
        if self._compiled_inlet_program is None:
            self._compiled_inlet_segments = []
            return

        times = list(self._compiled_inlet_program["times"])
        values = list(self._compiled_inlet_program["values"])
        self._compiled_inlet_segments = [
            ProgramSegment(
                start_time=times[idx],
                end_time=times[idx + 1],
                start_value=values[idx],
                end_value=values[idx + 1],
            )
            for idx in range(len(times) - 1)
        ]

    def _clear_inlet_program_cache(self):
        self._compiled_inlet_segments = []

    def _segment_expression(self, segment: ProgramSegment):
        units = volumetric_flow_type.Units
        start_value = Constant(segment.start_value * units)
        end_value = Constant(segment.end_value * units)

        if math.isclose(segment.start_value, segment.end_value, rel_tol=0.0, abs_tol=1e-12):
            return start_value

        duration = segment.end_time - segment.start_time
        if duration <= 0:
            raise RuntimeError("Invalid inlet-program segment with non-positive duration")

        return start_value + (end_value - start_value) * (
            (Time() - Constant(segment.start_time * s)) / Constant(duration * s)
        )

    def _declare_constant_inlet_flow_equation(self, value_expression, description):
        eq = self.CreateEquation("active_inlet_flow", description)
        eq.Residual = self.Flow_in_active() - value_expression

    def _declare_active_inlet_flow_equations(self):
        if self._compiled_inlet_program is None:
            self._declare_constant_inlet_flow_equation(
                self.Flow_in_default(),
                "Map the active inlet flow to the constant default value.",
            )
            return

        schedule = self._compiled_inlet_program
        if not self._compiled_inlet_segments:
            self._declare_constant_inlet_flow_equation(
                Constant(schedule["values"][0] * volumetric_flow_type.Units),
                "Map the active inlet flow to the degenerate single-point schedule value.",
            )
            return

        first_segment = self._compiled_inlet_segments[0]
        self.IF(Time() < Constant(first_segment.end_time * s))
        eq = self.CreateEquation("active_inlet_flow_000", "First piecewise inlet-flow segment.")
        eq.Residual = self.Flow_in_active() - self._segment_expression(first_segment)

        for index, segment in enumerate(self._compiled_inlet_segments[1:], start=1):
            self.ELSE_IF(Time() < Constant(segment.end_time * s))
            eq = self.CreateEquation(
                f"active_inlet_flow_{index:03d}",
                f"Piecewise inlet-flow segment {index}.",
            )
            eq.Residual = self.Flow_in_active() - self._segment_expression(segment)

        final_value = Constant(schedule["values"][-1] * volumetric_flow_type.Units)
        self.ELSE("Hold the final inlet-program value after the last breakpoint.")
        eq = self.CreateEquation(
            f"active_inlet_flow_{len(self._compiled_inlet_segments):03d}",
            "Final post-program inlet-flow hold segment.",
        )
        eq.Residual = self.Flow_in_active() - final_value
        self.END_IF()


class simTank(daeSimulation):
    def __init__(self, backend_mode, program, program_repeats, time_horizon):
        daeSimulation.__init__(self)

        if backend_mode == "piecewise":
            self.model = SimpleTankPiecewise("SimpleTank")
        elif backend_mode == "interpolation":
            self.model = SimpleTankInterpolation("SimpleTank")
        else:
            raise ValueError(f"Unsupported backend mode: {backend_mode}")

        self.backend_mode = backend_mode
        self.program = program
        self.TimeHorizon = time_horizon

        if self.program is not None:
            self.model.SetInletProgram(self.program, repeat=program_repeats, time_horizon=time_horizon)

    def SetUpParametersAndDomains(self):
        initial_flow = 10 if self.program is None else self.program.initial_value
        self.model.Flow_in_default.SetValue(initial_flow)
        self.model.Kflow.SetValue(1)
        self.model.Radius.SetValue(1)
        self.model.pi.SetValue(3.14)

    def SetUpVariables(self):
        self.model.Volume.SetInitialCondition(1 * (m**3))


def configure_evaluation_mode(backend_mode):
    cfg = daeGetConfig()

    if backend_mode == "piecewise":
        cfg.SetString("daetools.core.equations.evaluationMode", "computeStack_OpenMP")
    elif backend_mode == "interpolation":
        cfg.SetString("daetools.core.equations.evaluationMode", "evaluationTree_OpenMP")
    else:
        raise ValueError(f"Unsupported backend mode: {backend_mode}")


def build_simulation(backend_mode, time_horizon=1000, reporting_interval=5):
    configure_evaluation_mode(backend_mode)

    simulation = simTank(
        backend_mode=backend_mode,
        program=inlet_flow_program,
        program_repeats=False,
        time_horizon=time_horizon,
    )
    simulation.model.SetReportingOn(True)
    simulation.ReportTimeDerivatives = True
    simulation.ReportingInterval = reporting_interval
    return simulation


def run_once(backend_mode="piecewise", gui_run=False, time_horizon=1000, reporting_interval=5):
    simulation = build_simulation(
        backend_mode=backend_mode,
        time_horizon=time_horizon,
        reporting_interval=reporting_interval,
    )

    start = perf_counter()

    if gui_run:
        qt_app = daeCreateQtApplication(sys.argv)
        simulator = daeSimulator(qt_app, simulation=simulation)
        simulator.exec()
        elapsed = perf_counter() - start
        print(f"{backend_mode} backend completed in {elapsed:.3f} s (GUI mode)")
        return simulator, elapsed

    result = daeActivity.simulate(
        simulation,
        reportingInterval=reporting_interval,
        timeHorizon=time_horizon,
        reportTimeDerivatives=True,
        guiRun=False,
    )
    elapsed = perf_counter() - start
    print(f"{backend_mode} backend completed in {elapsed:.3f} s (console mode)")
    return result, elapsed


def run(backend_mode="piecewise", gui_run=False, time_horizon=1000, reporting_interval=5, runs=1):
    if gui_run or runs == 1:
        result, _ = run_once(
            backend_mode=backend_mode,
            gui_run=gui_run,
            time_horizon=time_horizon,
            reporting_interval=reporting_interval,
        )
        return result

    elapsed_times = []
    result = None
    for index in range(runs):
        print(f"Starting benchmark run {index + 1}/{runs} for {backend_mode}...")
        result, elapsed = run_once(
            backend_mode=backend_mode,
            gui_run=False,
            time_horizon=time_horizon,
            reporting_interval=reporting_interval,
        )
        elapsed_times.append(elapsed)

    average_time = sum(elapsed_times) / len(elapsed_times)
    summary = (
        f"{backend_mode} benchmark summary: runs={runs}, "
        f"avg={average_time:.3f} s, min={min(elapsed_times):.3f} s, max={max(elapsed_times):.3f} s"
    )
    if runs > 1:
        warm_run_times = elapsed_times[1:]
        warm_average_time = sum(warm_run_times) / len(warm_run_times)
        summary += f", warm_avg={warm_average_time:.3f} s"
    print(summary)
    return result


def parse_cli(argv):
    gui_run = True
    backend_mode = "piecewise"
    runs = 1
    time_horizon = 1000.0
    reporting_interval = 5.0

    for arg in argv[1:]:
        if arg == "console":
            gui_run = False
        elif arg in {"piecewise", "interpolation"}:
            backend_mode = arg
        elif arg.startswith("runs="):
            runs = int(arg.split("=", 1)[1])
        elif arg.startswith("time_horizon="):
            time_horizon = float(arg.split("=", 1)[1])
        elif arg.startswith("reporting_interval="):
            reporting_interval = float(arg.split("=", 1)[1])
        else:
            raise ValueError(
                "Usage: python simpleTank_runtime_compare.py "
                "[console] [piecewise|interpolation] [runs=N] "
                "[time_horizon=SECONDS] [reporting_interval=SECONDS]"
            )

    return gui_run, backend_mode, runs, time_horizon, reporting_interval


if __name__ == "__main__":
    gui_run, backend_mode, runs, time_horizon, reporting_interval = parse_cli(sys.argv)
    run(
        backend_mode=backend_mode,
        gui_run=gui_run,
        runs=runs,
        time_horizon=time_horizon,
        reporting_interval=reporting_interval,
    )
