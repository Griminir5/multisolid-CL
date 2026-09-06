# multisolid-CL

`packed_bed` simulates gas flow, heat transfer, and reactions in a packed-bed reactor.
The model uses DAETools. YAML files specify the species, solids, operating program, geometry, solver, and outputs.
Each run writes an xarray dataset in NetCDF format and a JSON manifest.

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

**Edit the input files.** The run file refers to three other files.
All relative paths use the directory that contains the referring file.

| File | Contents |
| --- | --- |
| `run.yaml` | File references, geometry, solver settings, output paths, reports, and plots |
| `chemistry.yaml` | Gas species, reaction families, and reaction IDs |
| `program.yaml` | Inlet flow, inlet temperature, outlet pressure, and inlet composition |
| `solids.yaml` | Solid species, initial concentrations, voidages, and particle diameters |

Use the complete YAML files in the example directory as templates.
Use kelvin, pascals, metres, seconds, and mol/s unless a field specifies another unit.
Each composition must include every selected gas species. The mole fractions must sum to 1.
Solid zones must cover the bed without gaps or overlaps.

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

**Use the compiled CPU backend on Windows x64.** Install the optional runtime with
`python -m pip install -e ".[compiled]"` and the C++ x64 component of Visual Studio Build Tools.
The compiled backend derives its equations from the DAETools model, eliminates explicit algebraic definitions,
and runs native residual and analytic-Jacobian functions with SUNDIALS IDA. DAETools still solves the initial conditions.

```powershell
python -m packed_bed packed_bed/examples/default_case/run_compiled.yaml
```

The first run compiles a model-specific kernel; later runs reuse `.packed_bed_cache` beside the case.
Set `PACKED_BED_COMPILED_CACHE` to share a cache directory. The backend supports the usual datasets and plots;
derivative reports, custom DAETools reporters and incidence-matrix output require `solver.backend: daetools`.
See [the performance guide](docs/performance.md) for supported settings, validation and measured tradeoffs.

**Run a batch.** A batch expands the combinations of named axis values into separate cases.

```powershell
python -m packed_bed batch packed_bed/examples/default_batch_case/batch.yaml --validate-only
python -m packed_bed batch packed_bed/examples/default_batch_case/batch.yaml --workers 4 --case-timeout-s 600
```

The first command validates all four example cases without writes.
The second command runs up to four cases at the same time.
Each worker uses one numerical thread when the requested worker count exceeds one.

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
