# DAETools Examples Reference and Packed-Bed Modeling Guide

## Scope

This guide is based on the Python sources under `daetools-2.6.0-win64/daetools/examples`.

- The folder contains runnable tutorials, helper models, UI files, post-processing scripts, and generated data/assets.
- The center of gravity here is the Python source, because that is where the reusable DAETools modeling patterns live.
- The `.html` files are still useful as companion narrative docs, but they mostly mirror the examples rather than add new implementation patterns.

The examples folder does **not** contain a direct dynamic packed-bed example with reacting gas and solid phases implemented in finite volume form. The best path is to combine patterns from several examples. That recommendation is an inference from the example set, not something DAETools states explicitly.

## `whats_the_time.py`: the baseline mental model

Start with `whats_the_time.py`. It is the smallest complete DAETools example and defines the mental model that the larger examples never abandon:

1. Import `pyDAE`, units, and variable types.
2. Define reusable `daeVariableType` objects.
3. Build a `daeModel` subclass.
4. Declare domains, parameters, variables, and submodels in `__init__`.
5. Declare residual equations in `DeclareEquations`.
6. Build a `daeSimulation` subclass.
7. Set parameters/domains in `SetUpParametersAndDomains`.
8. Set initial conditions and guesses in `SetUpVariables`.
9. Configure solver, reporters, logging, reporting interval, and time horizon.
10. Initialize, solve initial conditions, run, and finalize.

For a packed-bed model, the same structure still applies. The only real difference is that the model body becomes much larger and is likely split into submodels.

The most reusable lessons from `whats_the_time.py` are:

- Keep variable types explicit and unit-aware.
- Treat every equation as an implicit residual.
- Use `SaveModelReport` and `SaveRuntimeModelReport` early while building a new model.
- Keep model definition and simulation setup separate.

## Cross-cutting patterns from the example set

These are the patterns that appear repeatedly and matter most when building real models.

### Domains and distributed variables

- `tutorial1.py` is the cleanest introduction to domains, distributed variables, domain subsets, and boundary equations.
- `tutorial3.py` shows that `CreateStructuredGrid(...)` can be followed by replacing `domain.Points` with a non-uniform grid. This is useful for inlet layers, ignition fronts, or narrow reaction zones.
- `tutorial10.py` shows domain bounds driven by parameter values and the use of integrals over distributed quantities.

### Equation assembly style

- Small models use `DistributeOnDomain(...)` directly inside equations, as in `tutorial1.py`.
- Large models often switch to NumPy arrays of `adouble` objects to make stencil assembly clearer and to control sparsity, as in `tutorial11.py`, `tutorial16.py`, `tutorial18.py`, and especially `tutorial_che_6.py`.
- For packed-bed work, NumPy-backed equation assembly is often the better long-term choice once you move beyond a toy PDE.

### Finite-volume transport

- `tutorial_cv_6.py`, `tutorial_cv_7.py`, `tutorial_cv_8.py`, and `tutorial_cv_11.py` are the clearest examples of one-dimensional high-resolution finite-volume transport in DAETools via `daeHRUpwindSchemeEquation`.
- `tutorial_che_4.py` and `tutorial_che_5.py` show the same idea in a chemical-engineering context and compare flux limiters.
- `tutorial_adv_2.py` shows the same transport logic built manually, without the helper class.
- `tutorial_che_6.py` is the strongest example of **manual cell-centered finite-volume assembly** with separate cell centers, cell faces, ghost cells, harmonic means, fluxes at faces, and divergences in cells.

### Initialization and solver staging

- `tutorial7.py`, `tutorial13.py`, `tutorial15.py`, and `tutorial_che_8.py` show staged logic and state transitions.
- `tutorial_che_6.py` linearizes a nonlinear reaction law at `t = 0` to help initialization.
- `tutorial_che_8.py` progressively activates more nonlinear physics and calls `Reinitialize()` between stages.
- This is directly relevant to a packed bed with stiff coupling between gas transport, solid conversion, heat transfer, and kinetics.

### External properties and correlations

- `tutorial14.py` shows Python and shared-library external functions.
- `tutorial19.py` shows thermo-property packages and wrappers.
- For a real packed bed, these patterns matter if you want property correlations outside the equation file or want Cape-Open / CoolProp support.

### Constraints, reporting, and performance

- `tutorial20.py` shows IDA variable constraints. This is useful if concentrations, mass fractions, or porosity-like quantities must stay nonnegative.
- `tutorial8.py` and `tutorial17.py` are the main reporting/logging examples.
- `tutorial9.py`, `tutorial11.py`, `tutorial12.py`, and `tutorial21.py` cover direct solvers, iterative solvers, preconditioners, and parallel residual/Jacobian evaluation.

## Working with multiple gaseous components

This is shown most clearly in `compartment.py`, `support.py`, `membrane.py`, `membrane_unit.py`, and `tutorial_che_8.py`.

### Main idea

The tutorials do **not** use a special built-in “chemical component set” object. Instead, they represent components with an ordinary discrete DAETools domain, usually named `Nc`.

Typical pattern:

```python
self.Nc = daeDomain("Nc", self, unit(), "Number of components")
self.z  = daeDomain("z",  self, m,     "Axial domain")

self.C = daeVariable("C", molar_concentration_t, self, "", [self.Nc, self.z])
self.X = daeVariable("X", fraction_t,            self, "", [self.Nc, self.z])
self.Cin = daeVariable("C_in", molar_concentration_t, self, "", [self.Nc])
self.Xin = daeVariable("X_in", fraction_t,            self, "", [self.Nc])
```

Then, in `SetUpParametersAndDomains`, the component set is created with:

```python
self.m.Nc.CreateArray(3)   # e.g. N2, O2, He
```

For your case, the natural mapping is:

```python
components = ["Nitrogen", "Oxygen", "Helium"]
N2, O2, He = 0, 1, 2
```

The examples imply that **you manage the meaning of component index yourself**. DAETools stores the array shape, not the chemistry semantics. In practice, keep a Python list or dict that maps indexes to names and reuse that ordering everywhere.

### Variables can be distributed over components only, space only, or both

The examples show all three patterns:

- over components only: `Xin(i)`, `Xout(i)`, `Cin(i)`, `Cout(i)` in `compartment.py`,
- over space only: `P(z)`, `T(z)` or `T()` depending on the model,
- over components and space: `C(i,z)`, `X(i,z)`, `Flux(i,z)` in `compartment.py`,
- over components and two spatial coordinates: `X(i,z,r)` in `support.py` and `membrane.py`.

So yes: if you want a composition vector that is **not** distributed over the axial domain, you can define it just over the component domain:

```python
self.x_gas = daeVariable("x_gas", fraction_t, self, "Bulk gas composition", [self.Nc])
```

If instead you want a different composition in every axial cell:

```python
self.x_gas = daeVariable("x_gas", fraction_t, self, "Gas composition", [self.Nc, self.z])
```

### How composition closure is handled in the examples

The tutorials do not automatically normalize composition for you. You usually add one of these closures yourself:

- define concentrations `C_i` and compute fractions `X_i = C_i / sum_j C_j`,
- define fractions directly and impose `sum_i X_i = 1`,
- define `Nc-1` independent fractions and compute the last one as `1 - sum(others)`.

Examples:

- `compartment.py` uses concentrations as primary variables and computes fractions with
  `X(i,z) * Sum(C(*,z)) = C(i,z)`.
- `compartment.py` also enforces `P/(R T) = sum_i C_i`.
- `support.py` and `membrane.py` use expressions like `1 - Sum(self.X.array('*', z, r))` and `1 - Sum(self.Theta.array('*', z, r))`, which again shows that the sum-to-one closure remains explicit.

### How to get the composition vector at a cell or point

This is the core pattern you need for a packed bed.

If `X` is defined over `[Nc, z]`, then at axial point `z`:

```python
x_cell = self.X.array('*', z)
```

This returns an `adouble_array` containing the composition across all components at that location. The same pattern works for concentrations:

```python
c_cell = self.C.array('*', z)
```

Examples of this slicing pattern:

- `compartment.py`: `self.C.array('*', z)`
- `support.py`: `self.X.array('*', z, r)`
- `membrane.py`: `self.Xinlet.array('*', z)`, `self.Xoutlet.array('*', z)`, `self.Theta.array('*', z, r)`

If you prefer to be fully explicit, or if a third-party API is picky, you can always build the vector yourself:

```python
x_cell = [self.X(0, z), self.X(1, z), self.X(2, z)]
```

That explicit list-of-components pattern matches the style used in `tutorial19.py`, where `x = [0.60, 0.40]`.

### Practical recommendation for N2, O2, He

For a 1D packed bed, I would use:

```python
self.Nc = daeDomain("Nc", self, unit(), "Gas components")
self.zc = daeDomain("zc", self, m, "Axial cell centers")

self.Cg = daeVariable("Cg", molar_concentration_t, self, "Gas concentration", [self.Nc, self.zc])
self.Xg = daeVariable("Xg", fraction_t,            self, "Gas mole fraction", [self.Nc, self.zc])
```

and keep the component ordering fixed:

```python
components = ["Nitrogen", "Oxygen", "Helium"]
```

That same order must be used in:

- your `Nc` indexing,
- your parameter arrays,
- any stoichiometric vectors,
- any thermo-package compound list.

## Thermo-property packages and EoS inputs

This is covered mainly by `tutorial19.py` and the helper wrapper `daetools/pyDAE/thermo_packages.py`.

### What exists in the examples

DAETools exposes thermo packages through `daeThermoPhysicalPropertyPackage` and the higher-level `daeThermoPackage` wrapper.

The examples show two backends:

- Cape-Open thermo packages on Windows,
- CoolProp via a Cape-Open-style wrapper on all platforms.

### How a thermo package is loaded

The tutorial shows these patterns:

```python
self.tpp = daeThermoPackage("TPP", self, "")

self.tpp.LoadCapeOpen(
    "ChemSep Property Package Manager",
    "Water+Ethanol",
    ["Water", "Ethanol"],
    [],
    {"Liquid": eLiquid},
    eMole,
    {}
)
```

or

```python
self.tpp.LoadCoolProp(
    ["Water", "Ethanol"],
    [],
    {"Liquid": eLiquid},
    eMole,
    {}
)
```

For your gas example, the same structure would become something like:

```python
self.tpp.LoadCoolProp(
    ["Nitrogen", "Oxygen", "Helium"],
    [],
    {"Vapor": eVapor},
    eMole,
    {"backend": "HEOS"}
)
```

The order of compound IDs is critical. It must match the order used in your model’s component domain.

### Low-level vs high-level thermo calls

`tutorial19.py` makes an important distinction.

- Non-underscore calls such as `CalcSinglePhaseScalarProperty(...)` and wrapper calls such as `self.tpp.cp(...)` return DAETools expression objects and are meant to be used in equations.
- Underscore calls such as `_CalcSinglePhaseScalarProperty(...)` return plain floats and are meant for direct calculations outside the equation system.

That means:

- use `self.tpp.rho(...)`, `self.tpp.cp(...)`, `self.tpp.mu(...)`, `self.tpp.kappa(...)`, `self.tpp.phi(...)`, etc. inside residual equations,
- use `_CalcSinglePhaseScalarProperty(...)` only when you want a number right now, outside the model equations.

### What arguments the thermo package expects

From `tutorial19.py`:

- `P`, `T`, and `x` can be floats or DAETools expressions,
- `x` is the composition vector,
- `phase` and `basis` can be supplied explicitly,
- results are returned in SI units.

The wrapper in `thermo_packages.py` shows the standard calling shape:

```python
self.tpp.cp(P, T, x, phase='Vapor', basis=eMole)
self.tpp.rho(P, T, x, phase='Vapor', basis=eMass)
self.tpp.mu(P, T, x, phase='Vapor', basis=eUndefinedBasis)
self.tpp.kappa(P, T, x, phase='Vapor', basis=eUndefinedBasis)
self.tpp.phi(P, T, x, phase='Vapor')
```

If only one phase was loaded, the wrapper sets that phase as the default. If multiple phases were loaded, specify `phase=...` explicitly.

### How to pass composition from a cell in a domain to an EoS

This is the pattern you want in a packed-bed model.

Assume:

```python
self.Xg = daeVariable("Xg", fraction_t, self, "", [self.Nc, self.zc])
self.Pg = daeVariable("Pg", pressure_t, self, "", [self.zc])
self.Tg = daeVariable("Tg", temperature_t, self, "", [self.zc])
self.rhog = daeVariable("rho_g", density_t, self, "", [self.zc])
```

Then, in an equation distributed over `zc`:

```python
eq = self.CreateEquation("rho_g")
z = eq.DistributeOnDomain(self.zc, eClosedClosed)
x_cell = self.Xg.array('*', z)
eq.Residual = self.rhog(z) - self.tpp.rho(self.Pg(z), self.Tg(z), x_cell,
                                          phase='Vapor', basis=eMass)
```

That is the most tutorial-consistent way to call a thermo package at a cell.

An explicit alternative is:

```python
x_cell = [self.Xg(0, z), self.Xg(1, z), self.Xg(2, z)]
eq.Residual = self.rhog(z) - self.tpp.rho(self.Pg(z), self.Tg(z), x_cell,
                                          phase='Vapor', basis=eMass)
```

The tutorials do not show this exact distributed thermo call, but it follows directly from:

- `tutorial19.py`, which says `P`, `T`, and `x` can be floats or DAETools objects,
- `tutorial3.py`, `compartment.py`, `support.py`, and `membrane.py`, which show how to extract a component slice using `.array('*', ...)`.

### What kind of properties you can request

From `tutorial19.py` and `thermo_packages.py`, the high-level wrapper exposes at least:

- transport properties: `cp`, `kappa`, `mu`, `Dab`,
- density and caloric functions: `rho`, `h`, `s`, `G`, `H`, `I`,
- excess functions: `h_E`, `s_E`, `G_E`, `H_E`, `I_E`, `V_E`,
- phase-equilibrium-style properties: `f`, `phi`, `a`, `gamma`, `z`,
- two-phase properties: `K`, `logK`, `surfaceTension`.

Notes:

- some of these return scalars,
- some return vectors,
- `Dab` is flattened and accessed as `Dab(i*Nc + j)`,
- support varies by backend and package.

### Important behavior and limitations from the tutorials

From `tutorial19.py`:

- all properties are returned in SI units,
- not all Cape-Open properties are supported by all property packages,
- the calls are not thread safe,
- code generation is not supported for models using thermo packages,
- thermo packages are not supported by the Compute Stack evaluation mode,
- for those models `tutorial19.py` switches evaluation mode to `evaluationTree_OpenMP`.

In practice, if your packed-bed model uses thermo packages heavily, expect to:

- keep component ordering very disciplined,
- specify phase and basis explicitly when in doubt,
- avoid assuming thread safety,
- and test the property package separately with numeric underscore calls before embedding it into residual equations.

## Recommended implementation strategy for a dynamic packed bed

This is the practical synthesis I would use for a new packed-bed model.

### Recommendation in one sentence

Use `whats_the_time.py` for structure, `tutorial_che_6.py` for conservative cell-centered finite-volume assembly and nested solid submodels, `tutorial_cv_6.py` to `tutorial_cv_8.py` for high-resolution axial flux handling, `tutorial_che_1.py` and `tutorial_che_7.py` for kinetics and energy-balance closures, and `compartment.py` plus `membrane_unit.py` for multi-unit coupling style.

### Gas phase

For the axial gas phase, there are two realistic implementation choices:

- **Preferred for a real packed bed:** manual cell-centered finite volume, following `tutorial_che_6.py`.
- **Good for a fast first prototype:** `daeHRUpwindSchemeEquation`, following `tutorial_cv_6.py` to `tutorial_cv_8.py`.

Why I prefer the first option:

- It is explicitly conservative.
- It separates cell-center states from face fluxes.
- It makes interphase source terms and pressure-drop terms easier to place correctly.
- It scales better to multicomponent transport, separate gas/solid energy balances, and custom closures.

### Solid phase

Two common cases are supported by patterns in the examples:

- **Lumped solid conversion at each axial cell:** store solid states directly on the axial bed grid.
- **Intraparticle diffusion or reaction:** create a particle submodel per axial cell, following `tutorial_che_6.py`.

The hybrid pattern from `tutorial_che_6.py` is especially attractive:

- gas/electrolyte domain: manual finite volume,
- particle domain: radial distributed model using built-in derivatives.

For a packed bed, that translates naturally into:

- gas phase: axial FV,
- solid pellet: radial FDM or FV inside each axial cell.

### Interphase coupling

Use the same sign-consistent source pattern everywhere:

```text
gas accumulation + net gas flux = interphase transfer + homogeneous reaction
solid accumulation              = -interphase transfer + heterogeneous reaction
gas energy balance              = ... + h_gs a_s (T_s - T_g) + reaction heat terms
solid energy balance            = ... - h_gs a_s (T_s - T_g) + reaction heat terms
```

Relevant examples:

- `compartment.py`: axial transport plus interface source `aV * Flux`.
- `tutorial_che_6.py`: divergence of face fluxes plus source from submodels.
- `membrane_unit.py`: explicit coupling equations between child models.

### Convective fluxes and source terms

The high-resolution FV tutorials show an important distinction:

- `tutorial_cv_7.py`: if you know a consistent integral of the source term, you can discretize convection and source together with higher fidelity.
- `tutorial_cv_8.py`: if you do not, a cell-average source treatment is still workable.

For a packed bed, my default would be:

- use a high-resolution limiter for axial convection,
- use face-based fluxes for diffusion/dispersion,
- use cell-average reaction source terms unless you have a strong reason to do a more coupled source discretization.

### Boundary conditions

You have three reusable styles in the examples:

- direct boundary equations via `eLowerBound` and `eUpperBound`, as in `tutorial1.py`,
- a dedicated algebraic boundary cell, as in `tutorial_cv_6.py`,
- ghost points, as in `tutorial_che_6.py`.

For a packed bed with manual FV, ghost points are usually the cleanest option for:

- zero-gradient outlet conditions,
- symmetry,
- no-flux walls,
- known inlet states.

### Non-uniform meshes

Useful examples:

- `tutorial3.py`: overwrite `domain.Points` after grid creation,
- `tutorial_che_4.py` and `tutorial_che_5.py`: explicit custom point arrays,
- `tutorial_che_6.py`: separate center and face coordinates.

For a packed bed, non-uniform axial meshes are worth considering near:

- the inlet,
- sharp reaction fronts,
- thermal fronts,
- regions with rapid porosity or source-term changes.

### Initialization strategy

For a stiff reactive packed bed, assume you will need staged initialization.

Good patterns to reuse:

- give strong initial guesses for difficult algebraic variables, as in `tutorial_che_8.py`,
- ramp the forcing term instead of switching it instantly, as in `tutorial_che_6.py`,
- start from simpler physics and then activate the full model, as in `tutorial_che_8.py`,
- store and reload initialization states once you find a robust start, as in `tutorial10.py`.

### Practical DAETools skeleton

This is the structure I would start from:

```python
class PelletCell(daeModel):
    def __init__(self, Name, gas_state_refs, Parent=None):
        daeModel.__init__(self, Name, Parent)
        self.r = daeDomain("r", self, m, "Pellet radius")
        # solid variables, parameters, kinetics, intraparticle transport

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        # radial diffusion / reaction / surface flux equations


class PackedBed1D(daeModel):
    def __init__(self, Name, Parent=None):
        daeModel.__init__(self, Name, Parent)
        self.z_centers = daeDomain("z_centers", self, m, "Axial cell centers")
        self.z_faces   = daeDomain("z_faces",   self, m, "Axial cell faces")
        # gas species, pressure, gas/solid temperatures, conversion, etc.
        # child pellet models, one per axial cell if needed

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        # build adouble arrays for cell-center states
        # build ghost points
        # compute face fluxes
        # compute divergences
        # add reaction and transfer source terms
        # add algebraic closures and pressure drop


class simPackedBed(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = PackedBed1D("packed_bed")

    def SetUpParametersAndDomains(self):
        # explicit center/face coordinates
        # parameter values
        pass

    def SetUpVariables(self):
        # initial conditions and initial guesses
        pass
```

## Highest-value examples for this packed-bed project

If the goal is specifically a **dynamic packed bed with reacting solids and gases using finite volume**, these are the examples I would revisit first.

### Tier 1: direct building blocks

- `whats_the_time.py`: minimal DAETools lifecycle.
- `tutorial_che_6.py`: manual cell-centered FV, face fluxes, ghost cells, nested submodels, staged initialization.
- `tutorial_cv_6.py`: transient convection-diffusion with `daeHRUpwindSchemeEquation`.
- `tutorial_cv_7.py`: steady convection-diffusion with a consistent source integral.
- `tutorial_cv_8.py`: transient convection-reaction with source handling.
- `tutorial_cv_11.py`: reversed-flow check for the high-resolution scheme.
- `tutorial3.py`: non-uniform grids and NumPy/adouble interoperability.

### Tier 2: coupling and reactor-specific structure

- `compartment.py`: multicomponent axial transport plus interface source terms.
- `tutorial_che_1.py`: dynamic reaction and energy balances.
- `tutorial_che_7.py`: plug-flow reactor kinetics and energy closure along the axial coordinate.
- `membrane_unit.py`: wrapper-model architecture with child models and interface constraints.
- `tutorial_che_8.py`: coupled transport units plus staged nonlinear initialization.

### Tier 3: useful extensions

- `tutorial10.py`: saved initialization files.
- `tutorial11.py`: sparse equation layout and iterative solver setup.
- `tutorial14.py`: external functions for correlations.
- `tutorial19.py`: thermo-property packages.
- `tutorial20.py`: positivity/sign constraints.
- `tutorial21.py`: parallel equation evaluation.
- `tutorial_adv_2.py`, `tutorial_che_4.py`, `tutorial_che_5.py`: flux limiter behavior on sharp fronts.

## Suggested reading order

This is the reading order I would recommend for you and for future reuse:

1. `whats_the_time.py`
2. `tutorial1.py`
3. `tutorial3.py`
4. `tutorial10.py`
5. `tutorial11.py`
6. `tutorial_cv_6.py`
7. `tutorial_cv_7.py`
8. `tutorial_cv_8.py`
9. `tutorial_che_6.py`
10. `compartment.py`
11. `tutorial_che_1.py`
12. `tutorial_che_7.py`
13. `membrane.py`, `support.py`, `membrane_unit.py`
14. `tutorial_che_8.py`
15. `tutorial14.py`, `tutorial19.py`, `tutorial20.py`, `tutorial21.py`

## Example atlas

This section is the compact reference inventory for the Python example set.

### Getting started

- `whats_the_time.py`: minimal DAETools model/simulation lifecycle; the first file to clone when starting a new model.

### Core tutorials

- `tutorial1.py`: distributed domains, distributed variables/equations, and Dirichlet/Neumann boundary conditions.
- `tutorial2.py`: discrete arrays, distributed parameters, degrees of freedom, initial guesses, and solver statistics.
- `tutorial3.py`: array operations on variables, `Constant`/`Array`, NumPy interoperability, and non-uniform grids.
- `tutorial4.py`: discontinuous equations via `daeIF`; useful when equations change by regime.
- `tutorial5.py`: discontinuous equations via `daeSTN`; explicit state-machine style.
- `tutorial6.py`: ports, port connections, and composition of units.
- `tutorial7.py`: quasi-steady-state initialization, operating procedures, and resetting DOFs/initial conditions.
- `tutorial8.py`: data reporters and exporting to MAT, XLS, JSON, VTK, XML, HDF5, Pandas, and user-defined reporters.
- `tutorial9.py`: direct linear solvers.
- `tutorial10.py`: initialization files, parameter-driven domain bounds, and integrals.
- `tutorial11.py`: iterative linear solvers, preconditioners, sparse assembly patterns, and NumPy-backed equation construction.
- `tutorial12.py`: `superLU` and `superLU_MT` direct solver usage.
- `tutorial13.py`: event ports, `ON_CONDITION`, `ON_EVENT`, and user actions.
- `tutorial14.py`: external functions in Python and shared libraries; useful for correlations and lookup tables.
- `tutorial15.py`: nested state-transition networks.
- `tutorial16.py`: manual finite-element assembly with NumPy arrays of DAETools objects.
- `tutorial17.py`: TCP/IP log server for debugging and monitoring.
- `tutorial18.py`: another NumPy/adouble example on a matrix ODE system.
- `tutorial19.py`: thermo-physical property packages and Cape-Open / CoolProp access patterns.
- `tutorial20.py`: Sundials IDA variable constraints.
- `tutorial21.py`: parallel equation evaluation modes.

### Advanced tutorials

- `tutorial_adv_1.py`: interactive GUI-driven simulation run.
- `tutorial_adv_2.py`: manual high-resolution upwind/flux-limiter discretization for a population balance.
- `tutorial_adv_3.py`: code generators, model exchange, FMI export, and other integration paths.
- `tutorial_adv_4.py`: OpenCS code generation from a DAETools model.

### Chemical engineering tutorials

- `tutorial_che_1.py`: dynamic CSTR with multiple reactions and energy balance; useful for reaction/thermal source-term structure.
- `tutorial_che_2.py`: binary distillation column model.
- `tutorial_che_3.py`: seeded crystallization with the method of moments.
- `tutorial_che_4.py`: FV population-balance benchmark with many flux limiters, case 1.
- `tutorial_che_5.py`: FV population-balance benchmark with many flux limiters, case 2.
- `tutorial_che_6.py`: lithium-ion porous-electrode model; the strongest packed-bed-adjacent example in the folder.
- `tutorial_che_7.py`: steady nonisothermal plug-flow reactor with Arrhenius kinetics.
- `tutorial_che_8.py`: gas separation on a porous membrane with support; strong example of coupled transport units and staged initialization.
- `tutorial_che_9.py`: larger chemical reaction network / DAE benchmark problem.

### Chemical engineering optimization tutorials

- `tutorial_che_opt_1.py`: optimization of the `tutorial_che_1.py` CSTR.
- `tutorial_che_opt_2.py`: parameter estimation for alpha-pinene isomerization.
- `tutorial_che_opt_3.py`: marine population dynamic-system estimation problem.
- `tutorial_che_opt_4.py`: catalytic cracking parameter estimation.
- `tutorial_che_opt_5.py`: methanol-to-hydrocarbon parameter estimation.
- `tutorial_che_opt_6.py`: catalyst mixing optimization in a plug-flow reactor.

### Code verification tutorials

- `tutorial_cv_1.py`: exact-solution verification for first-order equations and sensitivities.
- `tutorial_cv_2.py`: 1D transient convection-diffusion MMS benchmark.
- `tutorial_cv_3.py`: 1D transient convection-diffusion MMS benchmark with a different boundary-condition setup.
- `tutorial_cv_4.py`: 2D transient convection-diffusion MMS benchmark.
- `tutorial_cv_5.py`: 1D heat-conduction FE verification problem.
- `tutorial_cv_6.py`: 1D transient convection-diffusion solved with the high-resolution FV scheme.
- `tutorial_cv_7.py`: 1D steady convection-diffusion with source-integral treatment.
- `tutorial_cv_8.py`: 1D transient convection-reaction with the high-resolution FV scheme.
- `tutorial_cv_9.py`: solid-body rotation verification problem.
- `tutorial_cv_10.py`: rotating Gaussian hill convection-diffusion verification problem.
- `tutorial_cv_11.py`: reversed-flow version of `tutorial_cv_6.py` to check limiter behavior.

### deal.II finite-element tutorials

- `tutorial_dealii_1.py`: deal.II finite-element introduction and DAE form of FE systems.
- `tutorial_dealii_2.py`: transient heat convection-diffusion FE problem.
- `tutorial_dealii_3.py`: Cahn-Hilliard FE example.
- `tutorial_dealii_4.py`: transient heat conduction on a nontrivial geometry.
- `tutorial_dealii_5.py`: flow through porous media.
- `tutorial_dealii_6.py`: diffusion-reaction in an irregular catalyst shape.
- `tutorial_dealii_7.py`: buoyancy-driven Stokes flow with temperature coupling.
- `tutorial_dealii_8.py`: parallel-plate reactor with bulk convection-diffusion and surface diffusion-reaction.
- `tutorial_dealii_9.py`: lid-driven cavity flow.

### OpenCS tutorials

- `tutorial_opencs_daetools_1.py`: DAETools version of the Akzo Nobel DAE benchmark.
- `tutorial_opencs_dae_1.py`: Akzo Nobel stiff chemical-kinetics DAE benchmark.
- `tutorial_opencs_dae_2.py`: OpenCS reimplementation of `tutorial1.py`.
- `tutorial_opencs_dae_3.py`, `tutorial_opencs_dae_3_groups.py`, `tutorial_opencs_dae_3_kernels.py`, `tutorial_opencs_dae_3_vector_kernels.py`, `tutorial_opencs_dae_3_fpga.py`, `tutorial_opencs_dae_3_kernels_fpga.py`, `tutorial_opencs_dae_3_single_source.py`: Brusselator PDE across grouping, kernel, vector, and accelerator variants.
- `tutorial_opencs_dae_5_cv.py`, `tutorial_opencs_dae_6_cv.py`, `tutorial_opencs_dae_7_cv.py`: OpenCS reimplementations of the DAETools code-verification cases.
- `tutorial_opencs_dae_8.py`, `tutorial_opencs_dae_8_kernels.py`, `tutorial_opencs_dae_8_vector_kernels.py`: Cahn-Hilliard finite-difference example and its kernel variants.
- `tutorial_opencs_dae_9.py`: 2D heat-equation DAE benchmark.
- `tutorial_opencs_dae_10.py`, `tutorial_opencs_dae_10_kernels.py`: 2D advection-diffusion semi-discrete DAE benchmark.
- `tutorial_opencs_ode_1.py`: Roberts stiff ODE kinetics.
- `tutorial_opencs_ode_2.py`: 2D advection-diffusion ODE benchmark.
- `tutorial_opencs_ode_3.py`, `tutorial_opencs_ode_3_groups.py`, `tutorial_opencs_ode_3_kernels.py`: 2-species diurnal-kinetics advection-diffusion problem and grouped/kernelized variants.
- `tutorial_opencs_aux.py`: result-comparison helper.
- `tutorial_opencs_dae_vec_tests.py`: vectorization testbed and comparison helper.

### Optimization tutorials

- `opt_tutorial1.py`: IPOPT setup and options.
- `opt_tutorial2.py`: Bonmin setup and options.
- `opt_tutorial3.py`: NLOPT setup and options.
- `opt_tutorial4.py`: using SciPy minimization with a DAETools simulation for objective/gradient evaluation.
- `opt_tutorial5.py`: least-squares fitting with SciPy and a DAETools simulation.
- `opt_tutorial6.py`: `daeMinpackLeastSq` example.
- `opt_tutorial7.py`: monitoring optimization progress.

### Sensitivity analysis tutorials

- `tutorial_sa_1.py`: forward sensitivity analysis for a small dynamic system.
- `tutorial_sa_2.py`: local derivative-based sensitivity analysis on a reversible reaction.
- `tutorial_sa_3.py`: global variance-based sensitivity analysis via SALib.

### Support and utility files

- `compartment.py`: multicomponent axial dispersed-flow compartment with interface flux source terms; very relevant to packed-bed gas-phase structure.
- `support.py`: porous support transport with state switching between no resistance, Fick, and Maxwell-Stefan models.
- `membrane.py`: generalized Maxwell-Stefan membrane transport model.
- `membrane_unit.py`: wrapper model composing feed, membrane, support, and permeate regions and coupling them explicitly.
- `membrane_variable_types.py`: reusable variable-type definitions for the membrane examples.
- `fl_analytical.py`: analytical solution helper using an external interpolation function for flux-limiter comparisons.
- `generate_fmus.py`: FMU export script over selected examples.
- `run_examples.py`, `RunExamples_ui.py`, `RunExamples_ui_webengine.py`: example browser/launcher.
- `tutorial_adv_1_ui.py`, `tutorial17_ui.py`: UI support for interactive examples.
- `dae_example_4_plots.py`, `dae_example_4_plots-old-working.py`, `dae_example_4_plots_dt_data.py`: post-processing and stored data for example-4 style comparisons.
- `__init__.py`: package entry point.
- `test.py`: tiny local test script.

## What I would actually copy into a packed-bed model

If I were implementing the packed bed immediately, I would directly reuse these ideas:

- From `whats_the_time.py`: the overall class structure and solver/reporting lifecycle.
- From `tutorial3.py`: non-uniform axial mesh support.
- From `tutorial10.py`: saved initialization values.
- From `tutorial11.py`: NumPy arrays of `adouble` objects for explicit stencil assembly.
- From `tutorial_che_6.py`: manual FV bookkeeping with centers, faces, ghost cells, harmonic means, divergences, and per-cell submodels.
- From `tutorial_cv_6.py` to `tutorial_cv_8.py`: high-resolution upwind fluxes and source-term handling.
- From `tutorial_che_1.py` and `tutorial_che_7.py`: Arrhenius kinetics and energy-balance source structures.
- From `compartment.py`: the pattern `transport + interfacial source`.
- From `membrane_unit.py`: explicit coupling equations between child regions/models.
- From `tutorial_che_8.py`: staged activation of nonlinear physics and reinitialization.
- From `tutorial14.py` and `tutorial19.py`: external property/thermo closures.
- From `tutorial20.py`: positivity constraints on concentrations if the solver tolerates them well.
- From `tutorial21.py`: parallel equation evaluation if the model grows large.

## Bottom line

The examples point to a clear implementation path:

- build the packed-bed model as a normal `daeModel` / `daeSimulation` pair,
- use manual cell-centered finite volume for the axial gas phase,
- use either lumped solids or nested pellet submodels for the solid phase,
- couple phases through explicit source terms and interface laws,
- initialize in stages,
- and treat the example set as a toolbox rather than searching for a single perfect starting file.

If I come back to this later, the first files to reopen should be:

- `whats_the_time.py`
- `tutorial3.py`
- `tutorial10.py`
- `tutorial11.py`
- `tutorial_cv_6.py`
- `tutorial_cv_7.py`
- `tutorial_cv_8.py`
- `tutorial_che_6.py`
- `compartment.py`
- `membrane_unit.py`
- `tutorial_che_1.py`
- `tutorial_che_7.py`
- `tutorial_che_8.py`
