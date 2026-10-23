# General tab

Set how the case runs and which data it records. This tab also opens the
predefined plots available from the case's retained run.

## Simulation fields

| Field or option | What it does |
| --- | --- |
| Time horizon (s) | Total simulated duration. Editable for repeating programs; otherwise derived from Program's timed steps. |
| Reporting interval (s) | Spacing of saved samples, independent of the solver's internal step sizes. The final horizon is also recorded. |
| Number of cells | Number of axial cells used to represent the bed. More cells resolve finer spatial detail and increase the numerical work. |
| Mass scheme | Chooses how species concentrations are reconstructed at interior cell faces. |
| Heat scheme | Chooses how species enthalpies are reconstructed for convective heat transport. |

Both scheme selectors offer:

| Option | Meaning |
| --- | --- |
| `upwind1` | First-order upstream value. |
| `central` | Average of the two adjacent cell values. |
| `linear_upwind2` | Second-order upstream linear reconstruction. |
| `muscl_minmod` | Linear reconstruction with a minmod slope limiter. |
| `weno3` | Third-order weighted reconstruction using local smoothness. |
| `weno5` | Fifth-order weighted reconstruction using a wider stencil. |

The nominal order applies where the required stencil and smoothness are
available. Check grid and tolerance sensitivity for your particular problem.

## Solver fields

| Field or option | What it does |
| --- | --- |
| Backend — Standard | Constructs and integrates the model through DAETools. |
| Backend — Compiled | Uses DAETools for construction and initial conditions, then generates and integrates a native model. Compatible compiled models are cached. |
| Linear solver | Selects the method used to solve the integration system. Unavailable choices explain their missing runtime requirement. |
| Number of threads (0 = environment) | Numerical threads within this case. Zero retains environment limits; KLU factorization remains serial. |
| Relative tolerance | Relative contribution to integration error control; smaller values ask the solver for tighter local accuracy. |
| Advanced solver settings… | Opens the additional integration controls listed below. |
| Cache status | Shows compiled-model cache information as execution progresses. A first build can take longer than later cached runs. |

| Linear solver option | Meaning and availability |
| --- | --- |
| SuperLU | Sparse direct factorization; Standard and Compiled. |
| KLU | Sparse direct factorization with serial factorization; Standard and Compiled. |
| Band | Direct factorization using the matrix's band structure; Compiled only. |
| LAPACK (dense) | Dense direct factorization; Standard only. |
| AztecOO / ILUT | Iterative solver with an ILUT preconditioner; Standard only. |
| AztecOO / Ifpack ILU | Iterative solver with an Ifpack incomplete-LU preconditioner; Standard only. |
| AztecOO / ML | Iterative solver with an ML multilevel preconditioner; Standard only. |
| SUNDIALS GMRES / Ifpack ILU | GMRES iteration with an Ifpack incomplete-LU preconditioner; Standard only. |

## Advanced solver settings

| Field or option | What it does |
| --- | --- |
| Suppress algebraic errors | Excludes algebraic variables from the local integration error test; their equations still have to be solved. |
| Concentration absolute tolerance (mol/m³) | Absolute tolerance for gas and solid concentration states, on a bed-volume basis; particularly relevant near zero concentration. |
| Maximum nonlinear iterations | Maximum nonlinear iterations attempted in a solver step. |
| Nonlinear convergence coefficient | Sets the nonlinear solver's convergence-test coefficient. |
| Maximum integration order | Caps the variable-order backward-differentiation method at an order from 1 to 5. |
| Scale residuals (compiled) | Scales equations for numerical conditioning without changing their solutions. |
| Step growth threshold (compiled) | Threshold for increasing the integration step size; a smaller value permits smaller proposed increases. |
| Nonlinear refresh interval (compiled) | Controls forced Jacobian refresh after repeated nonlinear corrections; zero uses the default nonlinear solver. |
| Vector exponentials (compiled) | Enables the vector exponential implementation in generated numerical code. |
| Band reciprocals (compiled) | Enables reciprocal-based arithmetic in the compiled Band solver. Available only with Band. |

## Requested reports and plots

**Requested reports** chooses what the simulation records. **Requested plots**
chooses predefined plots; **Show** opens an available plot from the retained run.
If results are stale, that plot still belongs to the earlier run.
Plots open in a separate window while they are prepared in the background, so
you can continue using the application. Closing a loading plot cancels its
preparation. Saved plots are reused when they match the retained results.

| Report option | What it records |
| --- | --- |
| Temperature | Inlet, cell-centre, and outlet temperature. |
| Pressure | Inlet, cell-centre, and outlet pressure, plus pressure drop. |
| Velocity | Superficial gas velocity at cell faces. |
| Gas concentration | Gas concentration by species and cell. |
| Gas mole fraction | Inlet, cell-centre, and outlet gas composition. |
| Solid concentration | Solid concentration by species and cell, per bed volume. |
| Solid mole fraction | Derived solid mole fractions by species and cell. |
| Gas flux | Inlet molar flow, species fluxes at faces, and total/species outlet flows. |
| Reaction rate | Rate of each reaction in each cell. |
| Gas enthalpy flux | Species enthalpy fluxes at faces. |
| Heat balance | Stored internal energy, cumulative boundary enthalpy transfer, heat loss, and balance error. |
| Mass balance | Stored mass, cumulative boundary mass transfer, and balance error. |

| Plot option | What it shows |
| --- | --- |
| Outlet composition | Inlet and outlet gas mole fractions over time; requires Gas mole fraction. |
| Outlet conditions | Outlet temperature, pressure drop, and total outlet flow; requires Temperature, Pressure, and Gas flux. |
| Axial profiles | Temperature and pressure over time and bed position; requires Temperature and Pressure. |
| Average solid composition | One mole-fraction curve per solid species, averaged over all bed cells; requires Solid mole fraction. |
| PCA trajectory | A two-dimensional trajectory using all selected reports, with points colored by time and Start/End markers; requires at least one report. |

**Average solid composition** takes the arithmetic mean of each species' local
mole fraction across the uniform bed cells at each reporting time. Cells with
no solids contribute zero, matching the Solid mole fraction report. This is a
spatial average of local fractions, rather than the fraction of total bed moles.

**PCA trajectory** uses each numeric report output, including every species and
cell or face, as separate inputs. Each input is centered and scaled to unit
standard deviation across the saved times before fitting a single PCA basis for
the run. Time and position coordinates are not inputs. Constant inputs are
ignored; reports with more species or spatial points contribute more inputs.
The axes show the first two principal-component scores and their percentages of
explained variance. Points are connected in reporting-time order. If the data
vary in only one direction, PC2 is zero; a constant run sits at the origin.
Nonfinite report values prevent PCA rendering instead of silently omitting times.
The PCA basis is fitted separately for each run, so axes can differ between runs.

Choose the required quantities before running. Editing a Report worksheet later
cannot produce data that the simulation did not record.

Read more: [General settings](../GUI-GUIDE.md#general-numerics-and-recorded-outputs) ·
[Backend and solver choices](../GUI-GUIDE.md#choose-a-backend-and-solver) ·
[Numerical treatment](../EQUATIONS.md#time-integration).
