# multisolid-CL

`packed_bed` simulates gas flow, heat transfer, and reactions in a packed-bed reactor.
The model uses DAETools. YAML files specify the species, solids, operating program, geometry, solver, and outputs.
Each run writes an xarray dataset in NetCDF format and a JSON manifest.

A small PyQt6 desktop client now lives in [desktop/](desktop/README.md).
It supports projects with multiple cases, draft run settings, shared previews,
and Run case / Run all execution. Each case keeps one latest run; rerunning
replaces its results, and input edits mark retained results stale. See its README for setup and the remaining work
toward [PLAN.md](PLAN.md).

**Install the package.** Use Python 3.11 or 3.12. The commands below use PowerShell from the repository directory.

1. Create an environment.

   ```powershell
   python -m venv .venv
   ```

2. Activate the environment.

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the package and test dependencies.

   ```powershell
   python -m pip install -e ".[dev]"
   ```

4. Install [DAETools 2.6.0](https://daetools.sourceforge.io/downloads.html) and its dependencies in the same environment.
   Use the DAETools archive for your operating system and Python version.
   From the extracted DAETools directory, run `python -m pip install .`.

Configuration validation does not need DAETools. Simulation needs a working DAETools installation and a supported solver.
This repository does not distribute DAETools, OpenCS, or their binary dependencies.

For the DAETools plotter, install the GUI dependencies with `python -m pip install -e ".[plotter]"`.
For the system graph, install Graphviz.
Then run `python -m pip install -e ".[graph]"`.
Graphviz must be available to `pygraphviz`.

**Validate a case.** Start with the [default case](packed_bed/examples/default_case/run.yaml).

```powershell
python -m packed_bed packed_bed/examples/default_case/run.yaml --validate-only
```

Validation detects duplicate YAML keys, invalid fields, missing species, unresolved references, and incompatible report or plot selections.
It does not create output files. It does not establish the scientific validity of a reaction or property correlation.

**Run a case.** Remove `--validate-only` to start the solver.

```powershell
python -m packed_bed packed_bed/examples/default_case/run.yaml
```

Use `--artifacts` to create the operating-program diagram and initial solid profile before the run.
If `pygraphviz` is available, this option also creates the system graph.
Use `--dae-plotter` to open the DAETools plotter after the run.
Use `--debug` to show the traceback for an unexpected error.
The installed `packed-bed` command accepts the same arguments as `python -m packed_bed`.

A single run can replace files in its output directory. Use a different output directory to keep an earlier result.

The default case includes three solver configurations with the same physical inputs,
1,000-second horizon, reports and tolerances. Each uses one numerical thread.

| Run file in `packed_bed/examples/default_case` | Backend / solver | Output directory |
| --- | --- | --- |
| `run.yaml` | DAETools / SuperLU baseline | `output` |
| `run_compiled.yaml` | Compiled / SuperLU_MT | `output_compiled` |
| `run_band.yaml` | Compiled / band LU | `output_band` |

For example, run `python -m packed_bed packed_bed/examples/default_case/run_band.yaml`.
The compiled variants require the optional runtime and compiler described below.
The band configuration uses `step_growth_threshold: 1.25` and
`nonlinear_refresh_interval: 4` to handle sharp reaction transients. Its tolerances
match the baseline. Optional vector exponentials and reciprocal diagonals remain disabled.

**Edit the input files.** The run file refers to three other files.
All relative paths use the directory that contains the referring file.

| File | Contents |
| --- | --- |
| `run.yaml` | File references, geometry, solver settings, output paths, reports, and plots |
| `chemistry.yaml` | Gas species, reaction families, and reaction IDs |
| `program.yaml` | Separate inlet channels or a feed stream, plus outlet pressure |
| `solids.yaml` | Solid species, initial concentrations, voidages, and particle diameters |

Use the complete YAML files in the example directory as templates.
Use kelvin, pascals, metres, seconds, and mol/s unless a field specifies another unit.
Each composition must include every selected gas species. The mole fractions must sum to 1.
Solid zones must cover the bed without gaps or overlaps.

`model.gas_voidage_mode` selects the gas storage volume used for concentrations
in mol/m³ of bed:

| Value | Gas volume / bed volume |
| --- | --- |
| `bed_and_particle` (default) | `e_b + (1 - e_b) * e_p` |
| `bed_only` | `e_b` |

The selected fraction is used consistently in gas initialization, the equation of
state, transport and gas energy storage. Solid volume always uses
`(1 - e_b) * (1 - e_p)`. The gas-phase density remains `P * MW_mix / (R * T)`;
Ergun still uses interparticle voidage and the solver calculates flow and velocity
variations across the bed. For example, add `gas_voidage_mode: bed_only` under
`model:` in `run.yaml` to exclude particle-pore gas storage.

A program channel has an `initial` value and an optional list of `hold` or `ramp` steps.
Each step has a `duration_s`. Each ramp also has a `target`.
A channel without steps stays constant.

With `repeat_program: true`, the next cycle starts from the previous cycle's final value.
The program does not reset to its initial value between cycles.

The solver smooths ramps with a one-second width. Thus, the value at time zero can differ from the declared initial value.
The simulation horizon limits the compiled program.

For a flow in gas hourly space velocity, set `inlet_flow.basis: ghsv_per_h`.
The compiler uses the bed volume, 273.15 K, and 100,000 Pa to convert GHSV to mol/s.
The manifest retains the declared basis and the compiled values.

**Ramp between feeds.** Set `simulation.program_mode: feed_stream` in `run.yaml`
to specify one inlet feed program plus the independent outlet-pressure channel.
The default, `separate_channels`, keeps the existing flow, temperature, and
composition channels. Each mode requires its own program format; mixing the
formats is rejected.

The default case includes a complete [feed program](packed_bed/examples/default_case/program_feed_stream.yaml)
and matching [run configuration](packed_bed/examples/default_case/run_feed_stream.yaml):

```powershell
python -m packed_bed packed_bed/examples/default_case/run_feed_stream.yaml
```

This variant uses the original feed compositions, flowrates, temperatures, and
flow/composition stage durations. Temperature changes with each feed, moving
the final cooldown five seconds earlier; the independent pressure schedule is
unchanged. The run keeps the baseline's 1,000-second horizon and writes to
`output_feed_stream`.

```yaml
# run.yaml (excerpt)
simulation:
  program_mode: feed_stream
```

For a case selecting `N2` and `H2O`, the corresponding `program.yaml` could be:

```yaml
feed_stream:
  basis: mol_per_s
  initial:
    flow: 1000.0
    temperature: 773.15
    composition: {N2: 1.0, H2O: 0.0}
  steps:
    - kind: ramp
      duration_s: 5.0
      target:
        flow: 333.05
        temperature: 673.15
        composition: {N2: 0.0, H2O: 1.0}
    - kind: hold
      duration_s: 225.0
outlet_pressure:
  initial: 3000000.0
  steps: []
```

Each ramp target may contain any combination of `flow`, `temperature`, and
`composition`. Omitted fields retain their previous values, including across
repeated cycles. A supplied composition must include every selected gas species
and sum to one. Use a `hold` step to retain the entire feed. Flow and temperature
must be positive; zero-flow stages are not supported. Set
`feed_stream.basis: ghsv_per_h` to use the same GHSV conversion as the
separate-channel mode.

Feed ramps interpolate total flow **F**, species flows **F × y**, and **F × T**.
The existing one-second smoothing applies to these quantities before dividing
by total flow to obtain composition and temperature. This avoids species-flow
overshoot caused by independently ramping total flow and mole fractions. For
example, halfway from 2736.77 mol/s air to 333.05 mol/s steam, steam contributes
166.53 mol/s and about 10.85% of the mixture. Temperature is weighted by molar
flow; this approximates mixing without enforcing an enthalpy balance. Smoothing
still rounds the transition edges and can change the inlet conditions at time
zero. Plots, initialization, and both solver backends use these derived conditions;
the manifest records the underlying flow programs and their ratios.

**Select numerical settings.** The mass and heat schemes are independent.
Available schemes are `upwind1`, `central`, `linear_upwind2`, `muscl_minmod`, `weno3`, and `weno5`.
The default interior flow mode is `forward_only`.
The `reversible` mode permits flow reversal at interior faces. It does not change the inlet and outlet boundary conditions.

`solver.name` selects a backend from the registry in [simulation.py](packed_bed/simulation.py).
The default is `trilinos_klu`. The default example selects `superlu`.
Use a backend included in your DAETools installation.
`threads: 0` retains environment limits and uses the DAETools default thread count.
A positive value sets the thread count.

The IDAS defaults are `suppress_algebraic_errors: false`, `max_nonlinear_iterations: 4`, and `nonlinear_convergence_coefficient: 0.33`.
The default example changes these values and uses `relative_tolerance: 1.0e-3`.
Compare important results with a stricter tolerance before you use a new solver configuration.

**Use the compiled CPU backend.** Install the optional runtime with
`python -m pip install -e ".[compiled]"` and a native C++17 compiler:

| Platform | Compiler |
| --- | --- |
| Linux x64 | GCC (`sudo apt install g++` on Debian/Ubuntu) or Clang |
| Windows x64 | Visual Studio Build Tools with the C++ x64 component |
| macOS Intel / Apple Silicon | Xcode Command Line Tools (`xcode-select --install`) |

On Linux/macOS, activate the Python environment with `source .venv/bin/activate`.
The compiler is detected on `PATH`; set `CXX` to a compiler executable name or full path to override it.
The adapter uses the bundled libraries from the pinned `scikit-sundae==1.1.3` wheel
(SUNDIALS 7.5.0, double precision, 32-bit indices). Arbitrary system or Conda SUNDIALS
installations are not interchangeable with this ABI. Wheel availability limits the supported
platforms; see [the runtime's platform support](https://scikit-sundae.readthedocs.io/en/stable/user_guide/installation.html#platform-support).
DAETools must also be installed for your Python version and platform.

The compiled backend derives its equations from the DAETools model, eliminates explicit algebraic definitions,
and runs native residual and analytic-Jacobian functions with SUNDIALS IDA. DAETools still solves the initial conditions.
Set these entries under `solver` in the case's `run.yaml` (retain any other solver settings):

```yaml
solver:
  backend: compiled
  name: superlu  # Also supports superlu_mt and band.
```

Then run the case normally:

```sh
python -m packed_bed packed_bed/examples/default_case/run.yaml
```

The first run compiles a model-specific kernel; later runs reuse `.packed_bed_cache` beside the case.
Kernels use `.dll`, `.so` or `.dylib` as appropriate; cache keys include the platform, CPU architecture,
compiler and build flags. Set `PACKED_BED_COMPILED_CACHE` to share a cache directory.
AVX2 acceleration is detected at runtime; other CPUs use scalar kernels. The optional
`vector_exponentials: true` also requires FMA and falls back to scalar math when unavailable.
The backend supports the usual datasets and plots;
derivative reports, custom DAETools reporters and incidence-matrix output require `solver.backend: daetools`.
Run `python -m pytest tests/compiled` to check native kernels, solver callbacks and reactor integration.

**Run a batch.** A batch expands the combinations of named axis values into separate cases.

```powershell
python -m packed_bed batch packed_bed/examples/default_batch_case/batch.yaml --validate-only
python -m packed_bed batch packed_bed/examples/default_batch_case/batch.yaml --workers 4 --case-timeout-s 600
```

The first command validates all twelve example cases without writes: two programs ×
two geometries × three solvers (SuperLU baseline, compiled SuperLU, compiled band).
Running all twelve cases requires the compiled-backend dependencies above.
The second command runs up to four cases at the same time.
Each case keeps its own `solver.threads` setting, including in concurrent batches.
The shipped examples use one numerical thread per case.
Keep only the desired entries under the `solver` axis in `batch.yaml` to select solvers.
The batch retains its 7,200-second horizon and `1e-5` relative tolerance for every solver.
All three use `concentration_absolute_tolerance: 1e-11` to resolve trace concentrations
through repeated reaction cycles; the band solver also uses the Newton refresh controls above.

A timeout stops that case. The batch continues with the other cases.

The command-line values override `workers` and `case_timeout_s` in the batch file.
Without these settings, the batch uses one worker and no timeout.
The worker count cannot exceed the case count. It is not limited automatically by the CPU count.

The batch validates every case before it creates case files or starts simulations.
Nested patches merge mappings in axis order. Later axes take precedence; lists replace earlier lists.
Program and geometry presets use the same case validation as a single run.
Each case receives its own four YAML files and output directory.

`summary.csv` records case selections, status, runtime, output paths, plot errors, and available balance errors.
The batch updates this file after each case completes and on a handled interruption.
A batch refuses to overwrite existing case directories or its summary.
To run the batch again, select a new output directory. Automatic resume is not available.
Use `python -m packed_bed batch --help` for the batch options.

**Compare solver timings.** From the repository, run:

```sh
python -m tools.benchmark_solvers --output untracked/solver_timings --repeats 3
```

This times the default case and all four batch conditions sequentially, using one numerical
thread and a fresh Python process for each run. Each solver gets a first run and three repeats;
compiled repeats must hit the kernel cache. Physics, tolerances and reports stay the same;
plot rendering is disabled. `--scope default` or `--scope batch` selects one example.
Use a new output directory for each benchmark. The output contains `summary.csv`, raw
measurements, environment details, report differences, copied cases and per-run logs.
The main timing includes initialization, compilation, integration and output writing;
process startup and imports are recorded separately. Batch timing is the sum of sequential
case times, so it does not measure the speedup from multiple batch workers.
For a stricter cross-check, add `--rtol 1e-6 --atol 1e-9`; these overrides apply equally
to all three solvers. Peak report differences are recorded in `comparisons.json`.
Measured example timings and numerical differences are in the
[solver benchmark report](docs/solver_benchmarks.md).

**Select reports and plots.** Reports determine the contents of `results.nc`.
The [report registry](packed_bed/reports.py) defines the variables, dimensions, and units.

Available reports are `temperature`, `pressure`, `velocity`, `gas_concentration`, `gas_mole_fraction`, `solid_concentration`, `solid_mole_fraction`,
`gas_flux`, `reaction_rate`, `gas_enthalpy_flux`, `heat_balance`, and `mass_balance`.
The reaction-rate report requires at least one selected reaction.
An empty report list produces a dataset with the scheduled time coordinate and run metadata.

| Plot | Required reports |
| --- | --- |
| `axial_profiles` | `temperature`, `pressure` |
| `outlet_composition` | `gas_mole_fraction` |
| `outlet_conditions` | `temperature`, `pressure`, `gas_flux` |

Set `outputs.requested_plots: []` to disable automatic plots.
Plots read the completed NetCDF file. Plot selection does not change the dataset.
A plot failure does not discard a successful simulation result.

Solid mole fraction is a derived output. It does not add a solver variable.
The mass and heat balance reports add three and four accounting equations, respectively.
The outlet composition uses the last-face species fluxes. At zero total flow, it uses the final cell's composition.
Set `outputs.solver_incidence_matrix: true` for a labelled solver-incidence CSV and PNG.

**Read the results.** Each run writes these files under `outputs.directory`.

| File | Contents |
| --- | --- |
| `results.nc` | Selected variables with labelled time, species, cell, face, and reaction dimensions |
| `manifest.json` | Input configuration, source hashes, code commit, package versions, output inventory, and status |

File-based runs capture input hashes from the bytes parsed by the loader.
The manifest preserves those hashes if the source files change later.
An in-memory case has no source-file hashes.
The code commit refers to the package repository, independently of the case directory.
An installation without a Git checkout records no code commit.
A failed run records its failure stage and traceback when it can write the manifest.

```python
from packed_bed.reports import load_dataset

results = load_dataset("packed_bed/examples/default_case/output/results.nc")
outlet = results.outlet_composition.sel(gas_species="H2")
```

For a machine-learning table, select the variables explicitly.

```python
matrix = results[["outlet_temperature", "outlet_flow"]].to_stacked_array(
    "feature", sample_dims=("time",)
)
matrix.to_pandas().to_csv("features.csv")
```

**Use the Python interface.** Load the case before you import the simulation runtime.

```python
from packed_bed.config import load_case

case = load_case("packed_bed/examples/default_case/run.yaml")
from packed_bed.simulation import run_case

result = run_case(case)
```

Read the source in this order:

| Source | Responsibility |
| --- | --- |
| [config/models.py](packed_bed/config/models.py), [config/load.py](packed_bed/config/load.py) | Parse, validate, and resolve one `Case` |
| [programs.py](packed_bed/programs.py) | Compile operating channels |
| [properties.py](packed_bed/properties.py), [reactions.py](packed_bed/reactions.py), [kinetics/](packed_bed/kinetics/) | Define properties, stoichiometry, and rate expressions |
| [model.py](packed_bed/model.py) | Declare variables and equations |
| [initialization.py](packed_bed/initialization.py) | Calculate and apply the initial state |
| [simulation.py](packed_bed/simulation.py) | Configure the solver, execute the run, and finalize outputs |
| [reports.py](packed_bed/reports.py), [plotting/](packed_bed/plotting/) | Extract results and render selected plots |
| [batch.py](packed_bed/batch.py), [cli.py](packed_bed/cli.py) | Execute batches and process command-line arguments |

**Add a species or reaction family.** Keep scientific definitions with their source references.

1. Add a `SpeciesProperties` record to `PROPERTY_REGISTRY` in `properties.py`.
2. Specify the phase, molecular weight in kg/mol, and enthalpy correlation in J/mol.
3. For a gas species, also specify a viscosity correlation in Pa s.
4. Add the species to the applicable chemistry, composition, or solid-profile maps.

A property correlation supplies `value(temperature)` for numeric values and `dae_expression(temperature)` for symbolic values.
`PolynomialHeatCapacity` coefficients use ascending powers of `T - t_ref`.
`ShomateHeatCapacity` and `QuadraticViscosity` remain separate correlations.

For a new family, create one module under `kinetics/`.
Import `KineticsContext`, `ReactionDefinition`, and `ReactionFamily` from `packed_bed.reactions`.
Define the reactions and their rate functions in that module. Export one `FAMILY` object.
Add that object to `FAMILY_REGISTRY` in `kinetics/__init__.py`.

Use negative stoichiometric coefficients for reactants and positive coefficients for products.
List catalysts separately. Include all required species in the reaction and family declarations.

Each rate function receives a context with the model, cell index, and gas/solid index methods.
Return rates in mol/(m^3 s) per total bed volume.
Keep DAETools imports lazy so that configuration validation can run without the solver.
The [kinetics source notes](packed_bed/kinetics/KINETICS_SOURCES.md) describe the supplied families.

Keep `DeclareEquations` contiguous. Preserve equation names, order, and solver incidence during mechanical changes.

**Run the tests.** Use the same environment that contains the editable package.

```powershell
python -m pytest
```

Tests use temporary output directories. Solver tests need DAETools; they skip when the package is absent.
To run only the checks that do not need the solver, use this command:

```powershell
python -m pytest --ignore=tests/test_solver_infrastructure.py
```

The tests cover configuration, examples, transport, inert initialization, worker failures, provenance, and output structure.
They do not establish the scientific validity of the supplied reaction mechanisms.

**Use the research tools.** These commands operate from the repository directory.

```powershell
python tools/generate_clr_programs.py --help
python -m Property_Estimation.hcap_linear_fit fe3o4 --h-ref -1118380 --output heat_capacity.png
python -m Property_Estimation.visc_fit o2 --output viscosity.png
```

The fitting tools accept species, data paths, and polynomial order as command-line inputs.
Without `--output`, they open a figure window.
The local `alex_repro/` comparison script reads NetCDF outputs.
The local `active_learning_optimization/` directory contains result extraction and comparison tools.
These local research directories remain ignored by Git. The obsolete relaunch helper is retained as text under `active_learning_optimization/archive/`.

**License.** This project uses GPL-3.0-only. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Third-party dependencies retain their own licenses.

The language reference for this README is [ASD-STE100, Issue 9](https://www.asd-ste100.org/).
