# Compute-Stack Scheduling Guide

This note explains how to implement time programs such as holds and ramps in DAETOOLS while staying on `computeStack_OpenMP`.

It is based on the example in [Test Files/simpleTank_runtime_compare.py](Test Files/simpleTank_runtime_compare.py), which contains two backends:

- `interpolation`: schedule expressed with `daeLinearInterpolationFunction(...)`
- `piecewise`: schedule expressed with native DAETOOLS `IF/ELSE_IF/ELSE` equations

The second approach is the one to use when you want to remain on the compute stack.

## 1. Why this is needed

`daeLinearInterpolationFunction(...)` behaves like a runtime/external node from the evaluator's point of view.
That is why interpolation-based schedules require `evaluationTree_OpenMP`.

If you want to stay on `computeStack_OpenMP`, the schedule must be written using native DAETOOLS symbolic constructs only, for example:

- `Time()`
- `Constant(...)`
- arithmetic expressions
- `IF/ELSE_IF/ELSE/END_IF`
- `STN/STATE/ON_CONDITION` if the schedule is state-driven rather than simple time segmentation

The key rule is:

- Do not inject interpolation/external-function nodes into the residual expression
- Do express the schedule as a finite set of DAETOOLS equations over time

## 2. Core idea

Instead of saying:

```python
F_in(t) = interpolate(times, values, t)
```

you say:

```text
if t < t1:
    F_in = segment_0_expression
elif t < t2:
    F_in = segment_1_expression
...
else:
    F_in = final_value
```

Each segment expression is written in ordinary DAETOOLS algebra.

For a hold:

```text
F(t) = v0
```

For a linear ramp from `(t0, v0)` to `(t1, v1)`:

```text
F(t) = v0 + (v1 - v0) * (t - t0) / (t1 - t0)
```

In DAETOOLS form:

```python
Constant(v0 * units) + (Constant(v1 * units) - Constant(v0 * units)) * (
    (Time() - Constant(t0 * s)) / Constant((t1 - t0) * s)
)
```

## 3. Structure of the implementation

The example separates the implementation into three layers.

### 3.1 Program definition layer

This is plain Python.

It describes the schedule in a solver-independent way:

- initial value
- hold steps
- ramp steps

In the example this is done by:

- `ProgramStep`
- `ProgramSegment`
- `ScalarProgram`

This layer does not depend on DAETOOLS equations.

Its job is only to compile user-friendly steps into breakpoints and segments.

## 3.2 Model layer

The model keeps a normal model variable such as:

```python
self.Flow_in_active = daeVariable(...)
```

and then drives it from the compiled schedule.

This is important because the rest of the model can use a normal variable:

```python
eq.Residual = dt(self.Volume()) - self.Flow_in_active() + self.Flow_out()
```

That keeps the schedule logic isolated in one place.

## 3.3 Backend layer

The example contains two different schedule backends:

- `SimpleTankInterpolation`
- `SimpleTankPiecewise`

The interpolation backend resolves the active input by creating a `daeLinearInterpolationFunction`.

The piecewise backend resolves it by creating DAETOOLS `IF/ELSE_IF/ELSE` equations over `Time()`.

That second pattern is the reusable one.

## 4. How the piecewise backend works

### 4.1 Compile the schedule into segments

After calling:

```python
program.build(...)
```

the code converts breakpoint arrays into segments:

- `start_time`
- `end_time`
- `start_value`
- `end_value`

Each segment represents either:

- a hold, where `start_value == end_value`
- a ramp, where `start_value != end_value`

### 4.2 Convert each segment into a DAETOOLS expression

The helper `_segment_expression(...)` does this.

For a hold it returns:

```python
Constant(segment.start_value * units)
```

For a ramp it returns:

```python
start_value + (end_value - start_value) * (
    (Time() - Constant(segment.start_time * s)) / Constant(duration * s)
)
```

### 4.3 Create a piecewise equation block

The backend then emits:

```python
self.IF(Time() < Constant(first_segment.end_time * s))
...
self.ELSE_IF(Time() < Constant(next_segment.end_time * s))
...
self.ELSE()
...
self.END_IF()
```

Each branch defines an equation for the same active input variable:

```python
eq.Residual = self.Flow_in_active() - segment_expression
```

After the last breakpoint, the schedule is held at its final value.

## 5. How to use this pattern in another model

The pattern is general.

Assume your model has some scheduled input `u(t)` such as:

- inlet flow
- heat duty
- pressure setpoint
- feed composition
- valve opening
- current or voltage program

The recommended steps are:

### Step 1: Introduce an "active" model variable

Instead of directly putting the schedule logic everywhere, create one variable:

```python
self.U_active = daeVariable("U_active", some_type, self, "Scheduled input actually used by the equations")
```

Then use `self.U_active()` in the real model equations.

### Step 2: Keep the user schedule outside the equations

Define your schedule in plain Python, for example:

- initial value
- list of holds
- list of ramps

This makes schedule authoring easy and model logic clean.

### Step 3: Compile the schedule before initialization

Before `DeclareEquations()` runs, convert the user steps into a finite list of time segments.

This is important: the piecewise branch structure must be known when the equations are declared.

### Step 4: Emit DAETOOLS piecewise equations

Inside `DeclareEquations()`:

- if there is no schedule, assign the default parameter/value
- otherwise, emit a piecewise `IF/ELSE_IF/ELSE` block that maps the active variable to the segment expression

### Step 5: Keep the model equations unaware of the schedule implementation

All real physics/chemistry/process equations should just consume `self.U_active()`.

That way you can switch scheduling backends later without rewriting the whole model.

## 6. Minimal template for a general model

```python
class MyModel(daeModel):
    def __init__(self, Name, Parent=None, Description=""):
        daeModel.__init__(self, Name, Parent, Description)

        self.U_default = daeParameter("U_default", units, self, "Default input")
        self.U_active  = daeVariable("U_active", variable_type, self, "Scheduled input used by the model")

        self._compiled_schedule = None
        self._compiled_segments = []

    def SetProgram(self, program, repeat=False, time_horizon=None):
        self._compiled_schedule = program.build(repeat=repeat, time_horizon=time_horizon)
        self._compiled_segments = build_segments(self._compiled_schedule)

    def _segment_expression(self, segment):
        if is_hold(segment):
            return Constant(segment.start_value * units)

        return Constant(segment.start_value * units) + (
            Constant(segment.end_value * units) - Constant(segment.start_value * units)
        ) * ((Time() - Constant(segment.start_time * s)) / Constant((segment.end_time - segment.start_time) * s))

    def _declare_program_equations(self):
        if self._compiled_schedule is None:
            eq = self.CreateEquation("U_active_default", "")
            eq.Residual = self.U_active() - self.U_default()
            return

        first = self._compiled_segments[0]
        self.IF(Time() < Constant(first.end_time * s))
        eq = self.CreateEquation("U_active_000", "")
        eq.Residual = self.U_active() - self._segment_expression(first)

        for i, segment in enumerate(self._compiled_segments[1:], start=1):
            self.ELSE_IF(Time() < Constant(segment.end_time * s))
            eq = self.CreateEquation(f"U_active_{i:03d}", "")
            eq.Residual = self.U_active() - self._segment_expression(segment)

        self.ELSE()
        eq = self.CreateEquation("U_active_final", "")
        eq.Residual = self.U_active() - Constant(self._compiled_schedule["values"][-1] * units)
        self.END_IF()

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        self._declare_program_equations()

        eq = self.CreateEquation("MainBalance", "")
        eq.Residual = ... self.U_active() ...
```

## 7. When to use `IF/ELSE_IF` and when to use `STN`

Use `IF/ELSE_IF/ELSE` when:

- the schedule is purely a function of time
- the branches are just piecewise mathematical definitions
- the system can switch freely based on time windows

Use `STN/STATE/ON_CONDITION` when:

- the input depends on stateful mode logic
- the transition depends on model conditions, not just fixed time intervals
- the system has hysteresis, latching, or event-driven mode changes

Examples:

- Time program for inlet flow: `IF/ELSE_IF` is usually enough
- Controller modes such as startup/ramp/hold/shutdown with conditional transitions: `STN` is usually better

## 8. How to extend it

### 8.1 More scheduled variables

If your model needs multiple scheduled inputs:

- create one active variable per input
- compile one segment list per input
- emit one piecewise block per input

Examples:

- `Flow_in_active`
- `Temperature_setpoint_active`
- `Pressure_setpoint_active`

### 8.2 More schedule shapes

The same idea can be extended beyond linear ramps.

As long as the segment expression is written using native DAETOOLS math, it can stay on the compute stack.

Examples:

- exponential approach:

```python
v0 + (v1 - v0) * (1 - Exp(-(Time() - t0) / tau))
```

- sinusoidal segment
- polynomial segment
- custom algebraic segment

Avoid:

- `daeLinearInterpolationFunction`
- `daeScalarExternalFunction`
- Python callbacks inside residual evaluation

if the goal is to remain on the compute stack.

### 8.3 Repeating programs

If the schedule must repeat, there are two common options:

- compile enough repeated segments up to the chosen time horizon before initialization
- rewrite the segment logic in terms of a wrapped local time, if the wrapped-time algebra can be expressed natively

The first option is simpler and safer.

### 8.4 Event-based modifications during a run

If you need to change the schedule dynamically during simulation, a fixed piecewise block may not be enough.

In that case you may need:

- `STN/ON_CONDITION`
- controlled `ReAssignValue(...)`
- reinitialization after state/input changes

That is a different pattern from the precompiled time schedule described in this guide.

## 9. Practical guidance

### Keep the schedule isolated

Do not scatter time conditions across every equation.

Prefer:

- one active variable
- one scheduling block
- all physics use the active variable

### Keep the compiled schedule finite

The piecewise structure is declared once.
If your schedule is known ahead of time, compile it fully before simulation startup.

### Expect tradeoffs

A piecewise compute-stack schedule is not guaranteed to beat interpolation on every tiny benchmark.

Why:

- the piecewise version introduces discontinuity handling
- the interpolation version may be structurally simpler for small models
- reporting and framework overhead can dominate in small tests

The reason to prefer the piecewise approach is that for large expensive models you avoid runtime/external evaluator nodes and keep the whole symbolic model in the compute-stack path.

## 10. Files to look at

- [Test Files/simpleTank_runtime_compare.py](Test Files/simpleTank_runtime_compare.py)
  - complete reference implementation
- [Test Files/simpleTank.py](Test Files/simpleTank.py)
  - original interpolation-based version

## 11. Command examples

Run the compute-stack piecewise version:

```powershell
python "Test Files/simpleTank_runtime_compare.py" console piecewise
```

Run the interpolation evaluation-tree version:

```powershell
python "Test Files/simpleTank_runtime_compare.py" console interpolation
```

Run repeated timing tests:

```powershell
python "Test Files/simpleTank_runtime_compare.py" console piecewise runs=6 time_horizon=10000 reporting_interval=50
python "Test Files/simpleTank_runtime_compare.py" console interpolation runs=6 time_horizon=10000 reporting_interval=50
```

## 12. Summary

To stay on `computeStack_OpenMP`, do not schedule with interpolation nodes.

Instead:

1. define the schedule in plain Python
2. compile it into finite time segments
3. map those segments to native DAETOOLS `IF/ELSE_IF/ELSE` equations
4. feed the rest of the model through an active scheduled-input variable

That pattern is general and can be reused in any model where the schedule is known before equation declaration.
