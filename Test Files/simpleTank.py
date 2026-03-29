import sys
from dataclasses import dataclass
from typing import Literal

from daetools.pyDAE import *
from pyUnits import m, s # this will not show up because pylance cannot get to .pyd files

volumetric_flow_type = daeVariableType(name="volumetric_flow_type", units=m**3/s,
                                  lowerBound=-100000, upperBound=100000, initialGuess=10, absTolerance=1e-5)
length_type = daeVariableType(name="length_type", units=m,
                                  lowerBound=-100000, upperBound=100000, initialGuess=10, absTolerance=1e-5)
volume_type = daeVariableType(name="volume_type", units=m**3,
                                  lowerBound=-100000, upperBound=100000, initialGuess=10, absTolerance=1e-5)

@dataclass(frozen=True)
class ProgramStep:
    duration: float
    kind: Literal["hold", "ramp"]
    target: float | None = None


class ScalarProgram:
    """Piecewise-linear schedule for a single scalar input.

    This class is plain Python. It does not know anything about DAETOOLS.
    The model later converts the `times`/`values` breakpoint lists returned by
    `build()` into a `daeLinearInterpolationFunction`.
    """

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

class SimpleTank(daeModel):
    def __init__(self, Name, Parent = None, Description = "Simple cylindrical isothermal tank"):
        daeModel.__init__(self, Name, Parent, Description)

        self._inlet_program = None
        self._compiled_inlet_program = None
        self._inlet_program_functions = {}

        self.Radius = daeParameter("Radius", m, self, "Cylindrical tank radius")
        self.Flow_in_default = daeParameter(
            "Flowrate_in",
            m**3/s,
            self,
            "Default inlet volumetric flow used when no inlet program is attached",
        )
        self.Kflow = daeParameter("Flow_constant", m**2.5/s, self, "FlowConstant")
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

    def _resolve_inlet_flow_input(self):
        """Return the symbolic inlet-flow expression to feed into the model.

        If no program is attached this is just the constant default parameter.
        If a program is attached this becomes the time-interpolated schedule.
        """
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

    def DeclareEquations(self):
        """
        3.2 Declare equations and state transition networks
            All models must implement DeclareEquations function and all equations must be specified here.

            Tipically the user-defined DeclareEquations function should first call the DeclareEquations
            function from the parent class (here daeModel). This is important when a model contains instances
            of other models to allow DeclareEquations calls from those (child-)models. If a model is simple
            (like in this example) there is no need for daeModel.DeclareEquations(self) call since it does nothing.
            However, it is a good idea to call it always (to avoid mistakes).
        """
        daeModel.DeclareEquations(self)

        # Resolve the inlet-flow input once, then use the active inlet-flow
        # variable throughout the model equations.
        resolved_inlet_flow = self._resolve_inlet_flow_input()

        eq = self.CreateEquation(
            "active_inlet_flow",
            "Map the active inlet flow to either the constant default or the scheduled program.",
        )
        eq.Residual = self.Flow_in_active() - resolved_inlet_flow

        eq = self.CreateEquation("Volume_level", "Equation which relates the lqiuid level and volume in a tank")
        eq.Residual = self.Volume() - self.Radius()*self.Radius()*self.pi()*self.Level()

        eq = self.CreateEquation("Volume_balance", "Differential equation that relates volume change to inflow/outflow")
        eq.Residual = dt(self.Volume()) - self.Flow_in_active() + self.Flow_out()

        eq = self.CreateEquation("Outflow_calc", "Determination of outflow amount from level")
        eq.Residual = self.Flow_out() - self.Kflow()*Sqrt(self.Level())

    def SetInletProgram(self, inlet_program, repeat=False, time_horizon=None):
        self._inlet_program = inlet_program
        self._compiled_inlet_program = inlet_program.build(repeat=repeat, time_horizon=time_horizon)
        return self._compiled_inlet_program

    def ClearInletProgram(self):
        self._inlet_program = None
        self._compiled_inlet_program = None
        self._inlet_program_functions = {}

class simTank(daeSimulation):
    def __init__(self, program, program_repeats, time_horizon):
        """
        4.1 First, the base class constructor has to be called, and then the model for simulation instantiated.
            daeSimulation class has three properties used to store the model: 'Model', 'model' and 'm'.
            They are absolutely equivalent, and user can choose which one to use.
            For clarity, here the shortest one will be used: m.
        """
        daeSimulation.__init__(self)

        self.model = SimpleTank("SimpleTank")
        self.program = program
        self.TimeHorizon = time_horizon

        if self.program is not None:
            self.model.SetInletProgram(self.program, repeat=program_repeats, time_horizon=time_horizon)

    def SetUpParametersAndDomains(self):
        """
        4.2 Initialize domains and parameters
            Every simulation class must implement SetUpParametersAndDomains method, even if it is empty.
            It is used to set the values of the parameters, initialize domains etc.
            In this example nothing has to be done.
        """
        initial_flow = 10 if self.program is None else self.program.initial_value
        self.model.Flow_in_default.SetValue(initial_flow)
        self.model.Kflow.SetValue(1)
        self.model.Radius.SetValue(1)
        self.model.pi.SetValue(3.14)

    
    def SetUpVariables(self):
        """
        4.3 Set initial conditions, initial guesses, fix degreees of freedom, etc.
            Every simulation class must implement SetUpVariables method, even if it is empty.
            In this example the only thing needed to be done is to set the initial condition for the variable tau.
            That can be done using the SetInitialCondition function.
        """
        self.model.Volume.SetInitialCondition(1*(m**3))

def guiRun(qtApp):
    # Interpolation functions are runtime/external nodes in DAETOOLS.
    # They are not supported by the default compute-stack evaluation mode.
    daeGetConfig().SetString("daetools.core.equations.evaluationMode", "evaluationTree_OpenMP")

    simulation = simTank(program=inlet_flow_program, program_repeats=False, time_horizon=1000)
    simulation.model.SetReportingOn(True)
    simulation.ReportTimeDerivatives = True
    simulation.ReportingInterval = 5
    simulator  = daeSimulator(qtApp, simulation = simulation)
    simulator.exec()

if __name__ == "__main__":
    qtApp = daeCreateQtApplication(sys.argv)
    guiRun(qtApp)
