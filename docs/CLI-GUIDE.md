# CLI guide

The command-line interface runs cases and parameter sweeps from YAML files. It
writes NetCDF data, a provenance manifest, and any requested plots. Use it from
the Python environment described in the
[installation guide](../README.md#install-the-full-repository-from-source).

You can work in any folder containing your input files. The examples below use
a folder called `my-case`; only the optional
[repository examples](#repository-examples) use paths inside a source checkout.
Use `python` from your activated environment. The installed `packed-bed`
command accepts the same arguments as `python -m packed_bed`.

| Task | Go to |
| --- | --- |
| Validate or run existing inputs | [Run a case](#run-a-case) |
| Create inputs from scratch | [A complete first case](#a-complete-first-case) |
| Change a program, bed, or reaction selection | [Edit the input files](#edit-the-input-files) |
| Choose a solver or compiled execution | [Solver settings](#solver-settings) |
| Generate input diagrams | [Input diagrams and the DAETools plotter](#input-diagrams-and-the-daetools-plotter) |
| Run parameter combinations | [Run a batch](#run-a-batch) |
| Read results or export data | [Outputs and analysis](#outputs-and-analysis) |
| Automate runs from Python | [Python API](#python-api) |
| Interpret failures | [Exit codes and troubleshooting](#exit-codes-and-troubleshooting) |

## Run a case

A case normally has four YAML files:

```text
my-case/
├── run.yaml
├── chemistry.yaml
├── program.yaml
└── solids.yaml
```

`run.yaml` points to the other three. If someone supplies a case, keep the
referenced files in their original relative layout.

Validate before running:

```sh
python -m packed_bed my-case/run.yaml --validate-only
```

Success prints `Validation passed:` followed by the resolved input path.
Then run the same case:

```sh
python -m packed_bed my-case/run.yaml
```

The terminal shows solver progress, elapsed time, and any requested balance
diagnostics. Outputs go to the paths specified in `run.yaml`. For the complete
example below, look in `my-case/output/`.

A single-case run can replace files at those output paths. Use a separate case
folder or change both output directories when you want to keep an earlier result.
The CLI does not manage the desktop's run snapshots and status files.

### Command reference

```sh
python -m packed_bed --help
python -m packed_bed batch --help
```

| Single-case argument | Meaning |
| --- | --- |
| `run.yaml` | Required positional path to the main case file |
| `--validate-only` | Load and validate inputs without simulating or creating output files |
| `--artifacts` | Generate input diagrams before executing the simulation |
| `--dae-plotter` | Retain the in-memory DAETools process and open its plotter after the run |
| `--debug` | Show a traceback for unexpected errors |
| `-h`, `--help` | Show command help |

Simulation settings belong in YAML. There are no single-case `--solver`,
`--output`, `--workers`, or `--case-timeout-s` flags. Batch-specific options
are described [below](#batch-options).

### What validation checks

Validation checks YAML structure, supported fields, species and reaction
selections, composition keys and totals, zone coverage, program timing, and
report/plot requirements. It resolves the selected scientific definitions.

For built-in definitions, validation does not need DAETools or a compiler. It
does not prove that a native solver can load, a compiled runtime is available,
or a model will integrate successfully. Python plugins require local approval
and their factories can execute during CLI validation; `--validate-only` is
not a metadata-only plugin browser.

`--validate-only` returns before `--artifacts` or `--dae-plotter` takes effect.
It creates no simulation outputs.

## A complete first case

Create the folder `my-case` and save these four UTF-8 files inside it. This is
a small nonreacting example: nitrogen flows through an inert alumina bed for
10 seconds. It is intended to demonstrate the input format and workflow.

### 1. run.yaml

```yaml
references:
  chemistry_file: chemistry.yaml
  program_file: program.yaml
  solids_file: solids.yaml

simulation:
  system_name: FirstCase
  time_horizon_s: 10.0
  reporting_interval_s: 1.0
  repeat_program: false
  program_mode: separate_channels
  interior_flow_mode: forward_only
  mass_scheme: upwind1
  heat_scheme: upwind1
  report_time_derivatives: false

model:
  bed_length_m: 0.1
  bed_radius_m: 0.01
  axial_cells: 5
  gas_voidage_mode: bed_and_particle
  ambient_temperature_k: 700.0
  heat_transfer_coefficient_w_per_m2_k: 0.0
  axial_heat_dispersion_w_per_m_k: 0.0

solver:
  backend: daetools
  name: superlu
  threads: 1
  relative_tolerance: 1.0e-6

outputs:
  directory: output
  artifacts_directory: output/artifacts
  requested_reports:
    - temperature
    - pressure
    - gas_mole_fraction
    - gas_flux
    - heat_balance
    - mass_balance
  requested_plots:
    - outlet_composition
    - outlet_conditions
```

### 2. chemistry.yaml

```yaml
gas_species: [N2]
reaction_families: []
reaction_ids: []
```

Empty reaction lists are valid for an inert case. A reacting case selects both
families and the individual reactions to enable.

### 3. solids.yaml

```yaml
solid_species: [Al2O3]
initial_profile:
  basis: bed
  concentration_unit: mol_per_m3
  zones:
    - x_start_m: 0.0
      x_end_m: 0.1
      e_b: 0.4
      e_p: 0.2
      d_p: 0.003
      values:
        Al2O3: 25000.0
```

Here `e_b` is interparticle voidage, `e_p` is intraparticle voidage, and `d_p`
is particle diameter in metres. The solid concentration is per total bed volume
because `basis` is `bed`.

### 4. program.yaml

```yaml
inlet_flow:
  basis: mol_per_s
  initial: 0.001
  steps: []
inlet_temperature:
  initial: 700.0
  steps: []
outlet_pressure:
  initial: 101325.0
  steps: []
inlet_composition:
  initial: {N2: 1.0}
  steps: []
```

Empty step lists hold each initial value constant. In hand-authored CLI inputs,
the explicit `time_horizon_s` defines the duration even when all channels are
constant.

Run the validation and execution commands in [Run a case](#run-a-case).
**Checkpoint:** a successful run creates `results.nc`, `manifest.json`, and the
two requested SVG plots. Nitrogen's mole fraction remains one because this case
contains a single gas species and no reactions.

## Edit the input files

### Paths and YAML conventions

All paths in `run.references`, `outputs.directory`, and
`outputs.artifacts_directory` are relative to the folder containing
`run.yaml`, unless absolute. The artifacts path is **not** relative to the
output directory: use `output/artifacts` to put it inside `output`.
The shell's working directory only determines how your command finds the
initial `run.yaml`.

Use spaces for YAML indentation, ordinary numeric values without unit suffixes,
and `true` / `false` for booleans. For scientific notation, write `1.0e-7`;
forms such as `1e-7` may be read as strings by the YAML reader. Unknown fields
and duplicate mapping keys are rejected. Quote chemical keys such as `'NO'`
wherever they occur, since unquoted `NO` is interpreted as a boolean.

### Main run settings

| Section | Main settings |
| --- | --- |
| `references` | Paths to chemistry, program, and solids files |
| `simulation` | System name, horizon, reporting interval, program mode/repetition, flow mode, mass/heat schemes |
| `model` | Bed length/radius, number of cells, voidage treatment, ambient temperature, wall heat transfer, axial heat dispersion |
| `solver` | Backend, linear solver, thread count, tolerances, and integration controls |
| `outputs` | Output/artifact directories and requested reports/plots |

Times are in seconds, lengths in metres, temperatures in K, and pressures in Pa.
Inlet flows, temperatures, pressures, and step durations must be positive.
The reporting interval sets saved samples, separately from the solver's adaptive
internal time steps; it must not exceed the simulation horizon. The grid needs
at least three axial cells.

Supported mass and heat schemes are `upwind1`, `central`, `linear_upwind2`,
`muscl_minmod`, `weno3`, and `weno5`. Set mass and heat schemes independently.
See the [equations guide](EQUATIONS.md) for their reconstruction rules and
boundary treatment.

`gas_voidage_mode` is `bed_only` or `bed_and_particle`.
`interior_flow_mode` is `forward_only` or `reversible`; reversing interior
flow does not turn the configured inlet into a negative-flow boundary.
`heat_transfer_coefficient_w_per_m2_k` describes wall heat transfer and
`axial_heat_dispersion_w_per_m_k` describes axial heat spreading. Zero disables
the corresponding term.

The complete field definitions and defaults are in
[`config/models.py`](../packed_bed/config/models.py). Write settings explicitly
when they matter for reproducibility: omitted YAML settings retain their schema
defaults, which differ from the desktop's new-case preset.

### Chemistry and solids

`gas_species` and `solid_species` identify the case components. Their order
also defines the reported species order. Every initial and ramp composition
must name exactly the selected gas components, with non-negative fractions
summing to one within `1e-12`. Include zero fractions explicitly.

Every solid zone must name every selected solid in `values`. Zones start at
zero, meet without gaps or overlaps, and end at the bed length. Changing
`bed_length_m` in YAML does not automatically rescale zone boundaries.
Voidages must be strictly between zero and one, particle diameter must be
positive, and solid loadings must be non-negative.

Solid `basis` may be `bed` or `solid`; `concentration_unit` may be
`mol_per_m3` or `kg_per_m3`. Changing either field changes the interpretation
of the numbers, not the numbers themselves. The engine converts the chosen
basis and units into its internal molar bed-volume concentrations.

For reacting cases, `reaction_families` makes models available and
`reaction_ids` selects the reactions. Include all required species. Custom
species references, family instances, and role bindings are documented in the
[plugin guide](PLUGINS.md#case-yaml-and-engine-integration). The
[built-in kinetics sources](../packed_bed/kinetics/KINETICS_SOURCES.md) describe
the scientific basis of the supplied reaction families.

### Holds, ramps, and repetition

A hold has `kind: hold` and positive `duration_s`. A ramp also has a `target`.
For example, replace the first case's `inlet_temperature` entry with:

```yaml
inlet_temperature:
  initial: 700.0
  steps:
    - kind: ramp
      duration_s: 2.0
      target: 750.0
    - kind: hold
      duration_s: 8.0
```

This defines a temperature change followed by a hold over the 10-second horizon.
Ramps use the engine's smoothing; see [model equations](EQUATIONS.md) for the
exact time interpolation.

With `repeat_program: false`, every channel that has steps must total the
explicit horizon within `1e-9` seconds. A channel with no steps stays constant.
With `repeat_program: true`, schedules repeat and the horizon can span part
of a cycle or several cycles. Each channel repeats its own step list, carrying
its final value into the next cycle. It does not reset to `initial`: add a ramp
back to the starting value when that is part of the intended cycle.

`inlet_flow.basis` is `mol_per_s` or `ghsv_per_h`. GHSV uses the total bed
volume and reference conditions **273.15 K and 100,000 Pa**. Editing the basis
in YAML does not convert the entered values for you.

### Combined feed mode

Set `simulation.program_mode: feed_stream` to specify flow, temperature, and
composition in one feed channel. Replace the entire `program.yaml` with this
alternative for the first case:

```yaml
feed_stream:
  basis: mol_per_s
  initial:
    flow: 0.001
    temperature: 700.0
    composition: {N2: 1.0}
  steps:
    - kind: ramp
      duration_s: 2.0
      target:
        temperature: 750.0
    - kind: hold
      duration_s: 8.0
outlet_pressure:
  initial: 101325.0
  steps: []
```

A feed ramp may change flow, temperature, composition, or a combination.
Unspecified target fields carry forward their preceding values. The engine
interpolates and smooths total molar flow `F`, species molar flows `F*y`, and
`F*T`, then divides the latter two by `F` to obtain composition and temperature.
This gives molar-flow-weighted temperature, without a feed enthalpy balance;
see [Operating programs](EQUATIONS.md#operating-programs). Outlet pressure remains
separate. Do not mix `feed_stream` with the independent inlet-channel keys in one
program file.

## Solver settings

Set `solver.backend` to `daetools` (the desktop's Standard backend) or
`compiled`. Both use DAETools for model construction and consistent initial
conditions; Compiled then integrates the generated native model.

| Backend | Accepted `solver.name` values |
| --- | --- |
| `daetools` | `superlu`, `klu`, `trilinos_lapack`, `trilinos_aztecoo`, `trilinos_aztecoo_ifpack`, `trilinos_aztecoo_ml`, `sundials_gmres_ifpack` |
| `compiled` | `superlu`, `klu`, `band` |

Availability depends on the installed solver adapters, native libraries, and
compiler. Compiled KLU needs the managed runtime with KLU support. See the
[installation guide](../README.md#install-the-full-repository-from-source) and
[native runtime setup](../desktop/BUILD_WINDOWS.md#native-bundles).

To try Compiled with the first case, change its existing solver section to:

```yaml
solver:
  backend: compiled
  name: band
  threads: 1
  relative_tolerance: 1.0e-6
```

The first run compiles automatically; compatible later runs reuse the cache.
For a standalone case the default is `.packed_bed_cache/` beside `run.yaml`.
CLI batches share a cache under their batch output directory. Set
`PACKED_BED_COMPILED_CACHE` to choose another location; an explicit setting
takes precedence over both defaults. Relative environment-variable paths are
resolved from the process's working directory. Remove a cache only when no run
is using it.

`solver.threads` sets numerical threads per case. A positive value sets limits
for DAETools and the applicable numerical libraries; `0` keeps environment
limits and uses DAETools' default. KLU factorization is serial.

Other solver fields include `concentration_absolute_tolerance` in mol/m³ bed,
`suppress_algebraic_errors`, `max_nonlinear_iterations`,
`nonlinear_convergence_coefficient`, and `maximum_order` (1–5).
`scale_residuals`, non-default `step_growth_threshold`,
`nonlinear_refresh_interval`, and `vector_exponentials` require Compiled.
`band_reciprocals` additionally requires `name: band`.
Compiled rejects `report_time_derivatives: true` and
`outputs.solver_incidence_matrix: true`.

## Input diagrams and the DAETools plotter

To create diagrams and then run:

```sh
python -m packed_bed my-case/run.yaml --artifacts
```

This writes `system_graph.svg`, `operating_program.svg`, and
`initial_solid_profile.svg` in the artifacts directory. The graph needs Graphviz
`neato`, discovered on `PATH` or through `MULTISOLID_GRAPHVIZ`. A graph-rendering
failure warns and skips that graph while allowing the other diagrams and
simulation to continue.

To generate only input diagrams, save this as `preview_case.py` beside
`my-case` and run `python preview_case.py`:

```python
from packed_bed.config import load_case
from packed_bed.artifacts import generate_artifacts

case = load_case("my-case/run.yaml")
for name, path in generate_artifacts(case).items():
    print(name, path)
```

This does not integrate the model. Numerical previews do not need DAETools for
the built-in definitions. Setting `outputs.solver_incidence_matrix: true`
requests a different artifact: it is generated after solver initialization,
requires execution, and is supported only by the Standard backend.

To open the DAETools plotter after a run:

```sh
python -m packed_bed my-case/run.yaml --dae-plotter
```

It needs DAETools' plotter dependencies and a graphical session. Close the window
to finish the command. Normal CLI runs and SVG generation do not require opening
a GUI. This flag opens the just-completed process, not an arbitrary existing
NetCDF file; use [Outputs and analysis](#outputs-and-analysis) for saved results.

## Run a batch

A batch starts from one base case, applies named variations, and runs every
combination. It does not change the base files.

For this example, keep the first case in its original `separate_channels` mode.
Save the following as `batch.yaml` beside the `my-case` folder:

```yaml
base_case: my-case/run.yaml
output_directory: sweep-output
workers: 2
case_timeout_s: 120.0
artifacts: false

axes:
  - id: flow
    values:
      - id: low
        patch:
          program:
            inlet_flow:
              initial: 0.001
      - id: high
        patch:
          program:
            inlet_flow:
              initial: 0.002
  - id: cells
    values:
      - id: coarse
        patch:
          run:
            model:
              axial_cells: 5
      - id: fine
        patch:
          run:
            model:
              axial_cells: 10
```

Two flows × two grids produce four cases. Validate them all:

```sh
python -m packed_bed batch batch.yaml --validate-only
```

Expect `Batch validation complete: 4/4 cases passed.`. Then execute:

```sh
python -m packed_bed batch batch.yaml
```

All cases are validated before any generated case files or simulations are
created. If one fails validation, the batch stops without starting simulations.
After execution starts, individual simulation failures and timeouts are recorded
and the remaining cases continue.

### Batch options

| Argument / YAML field | Meaning |
| --- | --- |
| `batch.yaml` | Required positional path |
| `--validate-only` | Expand and validate every combination without writing outputs |
| `--workers N` / `workers` | Maximum concurrent cases; positive integer, default 1 |
| `--case-timeout-s S` / `case_timeout_s` | Positive wall-clock limit per case; default is no timeout |
| `--debug` | Show tracebacks for unexpected errors escaping batch execution |
| YAML `artifacts: true` | Generate input diagrams for each case before execution |

Command-line workers and timeout override their YAML values:

```sh
python -m packed_bed batch batch.yaml --workers 2 --case-timeout-s 300
```

Run that command as an alternative to the preceding execution, or first choose
a fresh `output_directory`. There is no batch `--artifacts` or
`--dae-plotter` flag.

With more than one effective worker, or any timeout, cases run in separate
processes. Each retains its own `solver.threads` setting; workers are capped by
the number of cases. With one effective worker and no timeout, cases execute
sequentially in the current process. The timeout counts from worker startup and
includes diagram generation, compilation, initialization, and output processing,
as well as integration. It does not count time spent waiting in the unstarted queue.

### Presets and patch order

Paths in `base_case`, `output_directory`, `programs`, and geometry
`solids_file` entries are relative to `batch.yaml`. The base case's referenced
input files remain relative to its own `run.yaml`.

In addition to a `patch`, an axis value can select a named `program` or
`geometry`. For example, this fragment defines presets:

```yaml
programs:
  warm_feed: programs/warm.yaml
geometries:
  short_bed:
    model:
      bed_length_m: 0.05
    solids_file: geometries/short_solids.yaml
```

An axis value can then use `program: warm_feed` or `geometry: short_bed`.
Supply the referenced files before validating. A program preset replaces the
whole program document. A geometry merges its model fields and, if provided,
replaces the solids document. Adjust zone boundaries when changing length;
CLI batch geometry edits do not rescale them automatically.

Axes apply in their listed order. Within one axis value, the order is program,
geometry, then patch. Mappings merge recursively; lists and scalar values replace
their previous values. For example, a patched `steps` list replaces the entire
list. Later changes win when axes touch the same field.

Axis IDs must be unique, and value IDs must be unique within their axis. Every
value must specify a program, geometry, or nonempty patch. Each Cartesian
combination becomes one case. To list particular pairings instead of a full
cross-product, use one axis whose values each carry the complete paired changes.

### Batch outputs and reruns

The example creates:

```text
sweep-output/
├── summary.csv
├── cases/
│   ├── flow-low__cells-coarse/
│   │   ├── run.yaml
│   │   ├── chemistry.yaml
│   │   ├── program.yaml
│   │   ├── solids.yaml
│   │   ├── definitions.json
│   │   └── output/
│   │       ├── results.nc
│   │       └── manifest.json
│   └── ... three more case folders
└── .packed_bed_cache/       # Created if compiled execution needs it
```

Case folder names are generated from the axis/value IDs. Each generated run
uses local references, a generated `system_name`, `output/`, and
`output/artifacts/`; patches cannot override these generated fields.

Selected plugin files are copied into the generated case and identified by
`definitions.json`, so those inputs can travel together. Keep the descriptor
and any `plugins/` folder when moving a generated case.

`summary.csv` records case IDs, axis selections, status, errors, elapsed
wall-clock time, paths, plot status/errors, and requested heat/mass balance
diagnostics. It is updated during execution. Common statuses are
`validation_passed`, `running`, `success`, `simulation_failed`,
`timeout_failed`, and `interrupted_failed`. A successful simulation can still
have plot errors, recorded separately.

**Batch execution refuses to overwrite an existing summary or generated case
folder.** Use a new `output_directory` for another batch. Validation-only does
not check those output collisions. There is no automatic resume command; a
generated case can be run separately using its `run.yaml`, but that does not
update the original batch summary.

## Outputs and analysis

For the first case, a successful run produces:

```text
my-case/output/
├── results.nc
├── manifest.json
└── artifacts/
    ├── outlet_composition_vs_time.svg
    └── outlet_conditions_vs_time.svg
```

`manifest.json` records resolved configuration and scientific definitions,
input/output hashes, software/environment information, runtime and solver
statistics, dataset variables/dimensions, balance errors, and plot status.
A simulation failure normally attempts to write a failure manifest with a stage
and traceback. Early import failures, forced termination, or failures writing
the manifest itself may leave none.

A failed run may leave samples recorded before failure. Those are reporting-time
samples, not rejected solver iterations or necessarily the exact failure instant.
Use a fresh output location to avoid confusing old files with a later failed
attempt. A single-case run does not clear every unrelated file in its output
directory.

### Reports and plots

Reports select what is saved; plots consume those quantities. An empty
`requested_plots: []` disables post-run plots.

| Report ID | Main data |
| --- | --- |
| `temperature` | Inlet, cell, and outlet temperature |
| `pressure` | Inlet, cell, outlet, and pressure drop |
| `velocity` | Superficial velocity at faces |
| `gas_concentration` | Gas species concentration per total bed volume |
| `gas_mole_fraction` | Inlet, cell, and flux-derived outlet composition |
| `solid_concentration` | Solid species concentration per total bed volume |
| `solid_mole_fraction` | Solid mole fractions |
| `gas_flux` | Inlet flow, species face fluxes, and outlet flows |
| `reaction_rate` | Extent rates by reaction and cell; requires selected reactions |
| `gas_enthalpy_flux` | Species enthalpy fluxes at faces |
| `heat_balance` | Stored internal energy, cumulative boundary enthalpy/heat loss, and balance error |
| `mass_balance` | Stored mass, cumulative inlet/outlet mass, and balance error |

| Plot ID | Required reports | SVG filename |
| --- | --- | --- |
| `outlet_composition` | `gas_mole_fraction` | `outlet_composition_vs_time.svg` |
| `outlet_conditions` | `temperature`, `pressure`, `gas_flux` | `outlet_conditions_vs_time.svg` |
| `axial_profiles` | `temperature`, `pressure` | `axial_profiles_vs_time.svg` |

Validation rejects missing plot dependencies. Changing a report selection after
a run does not add data to the existing NetCDF file. A plot failure leaves the
simulation results available, records the error, and makes the CLI return a
nonzero exit code.

### Read NetCDF and export CSV

Save this as `inspect_results.py` beside `my-case`, then run
`python inspect_results.py`:

```python
import json
from pathlib import Path
from packed_bed.reports import load_dataset, compute_balance_errors

output = Path("my-case/output")
dataset = load_dataset(output / "results.nc")
print(dataset)
print(dataset["outlet_temperature"].isel(time=-1).item())
print(compute_balance_errors(dataset))

nitrogen = dataset["outlet_composition"].sel(gas_species="N2")
nitrogen.to_series().to_csv("nitrogen-outlet.csv")

manifest = json.loads((output / "manifest.json").read_text())
print(manifest["status"], manifest["plots"]["status"])
```

The dataset uses named coordinates such as `time`, `x_cell`, `x_face`,
`gas_species`, `solid_species`, and `reaction`, where relevant to the requested
reports. Physical units are stored in variable attributes. Cell-centre and face
positions are distinct grids. Time is stored as elapsed numeric seconds.

To recreate a supported plot from saved data without rerunning the solver:

```python
from pathlib import Path
from packed_bed.reports import load_dataset
from packed_bed.plotting import PLOT_REGISTRY

dataset = load_dataset("my-case/output/results.nc")
plot = PLOT_REGISTRY["outlet_composition"]
plot.render(dataset, Path("outlet-composition.svg"))
```

Save this as `plot_results.py` and run `python plot_results.py` from the same
working folder. The saved data must contain the plot's required variables.

There is no CLI Excel-export subcommand. Use Python for your own data processing
or the desktop's [Report and Results workflows](GUI-GUIDE.md#export-a-case-to-excel)
for configured workbooks. Balance diagnostics quantify conservation error; they
do not independently establish scientific accuracy or grid convergence.

## Plugins and desktop projects

The CLI discovers plugins from a containing desktop project, or from a
`definitions.json` beside a standalone/generated `run.yaml`. Locked inputs
must match the recorded built-in fingerprint and plugin hashes. Use the matching
application/plugin contents or explicitly regenerate the inputs.

Python code needs approval on this computer before loading. Open the project in
the desktop application and use **Plugins → Enable / Allow code…** as appropriate.
Copying a project or setting an enabled flag does not convey approval.
See the [plugin guide](PLUGINS.md) for definitions, packaging, and exact locks.

CLI `batch.yaml` files and desktop parameter studies have different workflows.
The desktop can import an individual `run.yaml`, but does not import a CLI batch
as a new study. Running desktop-owned input files directly through the CLI also
does not update the desktop's run status or snapshots; use the desktop's Run
action to keep those records together.

## Python API

For a script that loads and runs a case:

```python
from packed_bed.config import load_case
from packed_bed.simulation import run_case

case = load_case("my-case/run.yaml")
result = run_case(case)
print(result.status)
print(result.results_path)
print(result.manifest_path)
print(result.plot_errors)
```

Save as `run_case.py` beside `my-case` and run `python run_case.py`. It writes
the same configured outputs as the CLI. A nonempty `result.plot_errors` indicates
post-processing failure even when `result.status` is `"success"`.

`load_case` resolves executable scientific definitions. For metadata-only input
inspection without importing plugin implementations, use
`packed_bed.config.load.inspect_case_file`; its `CaseInputs` result is for
inspection and is not an executable `Case`.

`packed_bed.batch.run_batch_file(path, validate_only=True)` validates a batch
and returns its records. Without `validate_only`, it executes and writes outputs;
`workers=` and `case_timeout_s=` provide overrides. Put process-based batch
execution inside `if __name__ == "__main__":` in your script, so spawned workers
can import it without starting the batch again.

## Repository examples

With a source checkout, these commands run from the repository root:

```sh
python -m packed_bed examples/simulations/default_case/run.yaml --validate-only
python -m packed_bed batch examples/simulations/default_batch_case/batch.yaml --validate-only
```

The [single-case example](../examples/simulations/default_case/) demonstrates
reacting chemistry and timed operating channels. Remove `--validate-only` to
execute it.

The [batch example](../examples/simulations/default_batch_case/batch.yaml) combines
two programs, two bed geometries, and three solver configurations, for **12
cases**. The solver axis includes Standard SuperLU, Compiled SuperLU, and Compiled
Band; executing every combination needs their runtimes. Keep only the solver
axis values you intend to use. The example has `workers: 4` and a much longer
simulation horizon than the small first case in this guide.

## Exit codes and troubleshooting

| Exit code | Meaning |
| --- | --- |
| `0` | Validation passed, or all simulations and requested plots completed successfully |
| `2` | Invalid command syntax, caught input/batch validation error, or a batch refused because of output collisions |
| `1` | Other caught execution/setup errors, failed/timed-out batch cases, or requested plot failures |

These are the normal CLI return codes. Interrupts and external termination may
produce platform-specific statuses. `--debug` re-raises unexpected top-level
exceptions; ordinary validation errors remain concise. Batch per-case failures
are captured in the summary, so `--debug` does not turn every worker failure into
a terminal traceback.

For a terminal log, redirect both standard output and standard error:

```sh
python -m packed_bed my-case/run.yaml --debug > run.log 2>&1
```

That command executes the case. `run.log` is placed in the shell's current
directory and overwritten by this redirection. The CLI does not automatically
create the desktop's `worker.log`.

| Problem | What to check |
| --- | --- |
| A referenced YAML file cannot be found | Resolve its path from the folder containing `run.yaml`; batch preset paths start at `batch.yaml`. |
| A number is rejected as a string | Use plain numbers with no unit suffix and scientific notation such as `1.0e-7`. |
| Unknown field or duplicate key | Check spelling and nesting. Configuration is strict; there are no ignored extra fields. |
| Species/composition mismatch | Include every selected gas in every composition and every selected solid in every zone. |
| Program duration mismatch | Match each non-empty channel to the explicit horizon for non-repeating programs. |
| Zone coverage fails after a length change | Update zone boundaries or supply a matching geometry solids file. |
| Validation passes but execution cannot start | Check DAETools, the selected adapter, and the compiled runtime/compiler if used. Validation does not load all native dependencies. |
| Plugin approval or fingerprint mismatch | Follow the [plugin guide](PLUGINS.md#troubleshooting); preserve the selected package files and descriptor. |
| Batch outputs already exist | Choose a fresh batch output directory. Validation-only does not check collisions. |
| A case times out while compiling | The timeout includes compilation and initialization; choose a limit that allows them. |
| Results exist but the command returns 1 | Inspect plot errors and `manifest.json`; post-processing can fail after successful integration. |
| Final batch summary cannot be written | Close readers locking `summary.csv` and check write access. Completed case outputs remain in their directories. |
| No data after a forced stop | A worker may have stopped before publishing samples or a manifest. Inspect terminal logs and avoid treating old files as the new result. |

The implementation references are [CLI arguments](../packed_bed/cli.py),
[configuration models](../packed_bed/config/models.py),
[batch expansion/execution](../packed_bed/batch.py), and
[reports and provenance](../packed_bed/reports.py).

[Documentation index](../README.md#documentation)
