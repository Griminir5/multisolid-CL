# `packed_bed` Package Guide

This guide explains what each file in [`packed_bed`](/c:/MyRepos/multisolid-CL/packed_bed) is for, what the important functions and classes inside it do, and why the file exists in the package.

## Big Picture

The package is organized as a pipeline:

1. YAML files define the system and the run.
2. The config layer loads those files into typed Python objects.
3. The validation layer checks that the requested system makes sense.
4. The visualization layer produces static artifacts before the solver runs.
5. The solver layer assembles and runs the DAETOOLS model.
6. The export layer converts reported variables into CSV outputs.

The intended top-level flow is:

`load_run_bundle(...) -> validate_run_bundle(...) -> render_* -> assemble_simulation(...) -> run_assembled_simulation(...) -> export_run_outputs(...)`

The convenience wrapper for that whole flow is `run_simulation(...)`.

## File Map

### [`__init__.py`](/c:/MyRepos/multisolid-CL/packed_bed/__init__.py)

Purpose:
- Defines the public package surface.
- Re-exports the functions and types that outside code is supposed to use.

Why it exists:
- So callers can write `from packed_bed import run_simulation, load_run_bundle, ...` instead of importing from many internal modules.

Important contents:
- Re-exports high-level API functions such as `run_simulation`, `load_run_bundle`, `validate_run_bundle`, `assemble_simulation`, `build_system_graph`, `render_operating_program`, and `render_initial_solid_profile`.
- Re-exports important dataclasses like `ChemistryConfig`, `SolidConfig`, `RunBundle`, `RunResult`, `ReactionDefinition`, and `SpeciesPropertyRecord`.

### [`__main__.py`](/c:/MyRepos/multisolid-CL/packed_bed/__main__.py)

Purpose:
- Makes `python -m packed_bed ...` work.

Why it exists:
- Python packages need `__main__.py` to behave as module-level CLI entry points.

Important contents:
- Calls `main()` from [`cli.py`](/c:/MyRepos/multisolid-CL/packed_bed/cli.py).

### [`api.py`](/c:/MyRepos/multisolid-CL/packed_bed/api.py)

Purpose:
- Provides the cleanest top-level library entry point.

Why it exists:
- The lower-level modules are intentionally separated, but most callers want one function that runs the whole workflow.

Important contents:
- `RunResult`: immutable summary of a completed run, including output directory, artifact paths, report paths, and key CSV locations.
- `_coerce_run_bundle(...)`: accepts either a `RunBundle` or a path to `run.yaml` and normalizes to a `RunBundle`.
- `run_simulation(...)`: the high-level orchestration function. It:
  - loads or accepts a run bundle,
  - validates it,
  - renders the graph/program/solid-profile artifacts,
  - assembles the DAETOOLS simulation,
  - executes the run,
  - exports CSV outputs,
  - returns a `RunResult`.

### [`cli.py`](/c:/MyRepos/multisolid-CL/packed_bed/cli.py)

Purpose:
- Implements the non-GUI command-line interface.

Why it exists:
- This is the first usable interface for people who do not want to write Python code.

Important contents:
- `build_parser()`: defines CLI arguments.
  - positional `run_yaml`
  - `--output-dir`
  - `--artifacts-dir`
  - `--validate-only`
- `main(argv=None)`: CLI driver. It loads the run bundle, applies output overrides, validates the input, and either:
  - exits after validation, or
  - executes the full run and prints the key output locations.

Use:

```bash
python -m packed_bed
python -m packed_bed packed_bed/examples/default_case/run.yaml
python -m packed_bed packed_bed/examples/default_case/run.yaml --validate-only
```

### [`config.py`](/c:/MyRepos/multisolid-CL/packed_bed/config.py)

Purpose:
- Defines the typed configuration objects used everywhere else.
- Loads YAML into those objects.

Why it exists:
- The rest of the package should work with structured data, not raw YAML dictionaries.

Important dataclasses:
- `ChemistryConfig`
  - Ordered gas species list.
  - Selected reaction ids.
- `SolidZoneConfig`
  - One axial zone in the initial solid profile.
  - Holds `x_start_m`, `x_end_m`, species concentration values, and the zone-specific `e_b`, `e_p`, and `d_p` values.
- `SolidConfig`
  - Ordered solid species list.
  - Unit basis for the authored solid concentrations.
  - Full axial initial-profile definition.
- `ScalarChannelConfig`
  - One scalar operating-program channel such as inlet flow or outlet pressure.
  - Method: `compile_program()` converts config into a runtime `ScalarProgram`.
- `CompositionChannelConfig`
  - Inlet gas-composition program.
  - Method: `compile_program(species_order)` converts species-keyed mappings into a runtime `VectorProgram`.
- `ProgramConfig`
  - Groups the four supported program channels.
- `ModelConfig`
  - Bed geometry, grid size, legacy default voidages/particle size, and constants used to parameterize the DAETOOLS model.
- `SolverConfig`
  - Solver-level numeric options, currently just relative tolerance.
- `OutputConfig`
  - Output directory, artifact directory, and requested report ids.
- `RunConfig`
  - Everything needed to actually execute a run, including references to the config files used.
- `RunBundle`
  - The complete loaded input set: paths plus parsed `chemistry`, `solids`, `program`, and `run`.

Important helper functions:
- `_read_yaml(path)`
  - Reads a YAML file and requires a top-level mapping.
- `_resolve_path(base_dir, value)`
  - Resolves relative config-file paths against the `run.yaml` directory.
- `_coerce_string_tuple(...)`
  - Validates list-shaped string inputs.
- `_coerce_float_mapping(...)`
  - Validates mapping-shaped numeric inputs.
- `_parse_solid_zones(...)`
  - Parses the zone list from `solids.yaml`, including concentrations plus `e_b`, `e_p`, and `d_p`.
- `_parse_solid_config(...)`
  - Parses the full solid configuration file.
- `_build_legacy_solid_config(...)`
  - Compatibility path for older runs that still specify solid initialization in `run.yaml`.
  - It also maps the old global `interparticle_voidage`, `intraparticle_voidage`, and `particle_diameter_m` settings into a single full-bed solid zone so old inputs do not break immediately.
- `_parse_steps(...)`
  - Parses hold/ramp program step lists.
- `_parse_scalar_channel(...)`
  - Parses scalar channels like flow, temperature, and pressure.
- `_parse_composition_channel(...)`
  - Parses inlet composition as species-keyed mappings.
- `load_run_bundle(run_yaml_path)`
  - Main loader for the package.
  - Reads `run.yaml`, resolves references, reads the other YAML files, builds all typed config objects, and returns a `RunBundle`.

### [`programs.py`](/c:/MyRepos/multisolid-CL/packed_bed/programs.py)

Purpose:
- Holds the general runtime representation of operating programs.

Why it exists:
- The config layer needs a clean, solver-independent way to represent piecewise hold/ramp schedules.

Important contents:
- `ProgramStep`
  - One hold or ramp segment in authored form.
- `ProgramSegment`
  - One compiled segment with explicit start/end times and values.
- `default_inlet_composition(gas_species)`
  - Produces a simple default composition with the first gas species set to `1.0`.
- `coerce_scalar(...)`
  - Validates scalar values.
- `coerce_composition_mapping(...)`
  - Validates species-keyed mole-fraction mappings.
- `coerce_vector(...)`
  - Validates raw vector inputs.
- `ScalarProgram`
  - Runtime builder for scalar schedules.
  - Methods:
    - `hold(duration)`
    - `ramp(duration, target)`
    - `build_segments(time_horizon=None)`
- `VectorProgram`
  - Same idea, but for vector-valued channels such as inlet composition.

Note:
- This file is the clean program abstraction for the new modular package.
- There is still a similar copy inside [`solver.py`](/c:/MyRepos/multisolid-CL/packed_bed/solver.py) because that file was ported directly from the legacy monolith with minimal numerical edits. That duplication is technical debt, not a design goal.

### [`properties.py`](/c:/MyRepos/multisolid-CL/packed_bed/properties.py)

Purpose:
- Defines the species property library and property-correlation classes.

Why it exists:
- The solver should not hard-code thermodynamic and transport correlations inline.
- This is the first extension point for adding new species.

Important contents:
- `_as_float_array(...)`
  - Utility for numeric evaluation of correlations.
- `MolarEnthalpyCorrelation`
  - Abstract base class for molar enthalpy correlations.
- `GasViscosityCorrelation`
  - Abstract base class for gas viscosity correlations.
- Correlation implementations:
  - `CpZerothMolar`
  - `CpQuadraticMolar`
  - `CpCubicMolar`
  - `CpQuarticMolar`
  - `CpShomateMolar`
  - `ViscosityQuadratic`

Each correlation typically provides:
- `dae_expression(temperature)`
  - Builds a DAETOOLS symbolic expression for use inside residual equations.
- `value(temperature)`
  - Evaluates the correlation numerically for initialization or preprocessing.

Additional helper methods like `cp_dae_expression(...)` and `cp_value(...)` exist on the heat-capacity-based enthalpy classes because heat capacity is useful separately from integrated enthalpy.

Registry layer:
- `SpeciesPropertyRecord`
  - One species entry with phase, molecular weight, enthalpy, viscosity, and optional future properties.
  - `__post_init__()` enforces phase/property consistency.
- `PropertyRegistry`
  - Lookup and validation wrapper around the species dictionary.
  - Methods:
    - `has_species(...)`
    - `species_ids(...)`
    - `get_record(...)`
    - `require_species(...)`
    - `enthalpy_expression(...)`
    - `enthalpy_value(...)`
    - `viscosity_expression(...)`
    - `viscosity_value(...)`

Data store:
- `DEFAULT_PROPERTY_REGISTRY`
  - The actual built-in species library shipped with the package.

### [`reactions.py`](/c:/MyRepos/multisolid-CL/packed_bed/reactions.py)

Purpose:
- Defines the machine-readable reaction catalog.

Why it exists:
- `KINETICS_SOURCES.md` is human-readable reference material.
- The code needs structured reaction metadata to validate selections, build graphs, and later attach executable kinetics.

Important contents:
- `ReactionDefinition`
  - One catalogued reaction.
  - Holds id, name, phase, stoichiometry, required species, source reference, optional `kinetics_hook`, and notes.
- `DEFAULT_REACTION_CATALOG`
  - Current built-in reaction metadata.

Current design intent:
- Reactions can be selected now.
- Metadata-only reactions validate and visualize correctly.
- If `kinetics_hook` is still `None`, solver assembly fails before execution.

### [`reporting.py`](/c:/MyRepos/multisolid-CL/packed_bed/reporting.py)

Purpose:
- Defines the stable report ids users can request in `run.yaml`.

Why it exists:
- The output layer should not depend on raw DAETOOLS variable names in user configs.

Important contents:
- `ReportDefinition`
  - Report id, human description, and optional DAETOOLS variable name.
- `REPORT_VARIABLE_REGISTRY`
  - Maps stable external ids like `temperature`, `pressure`, `gas_flux`, `material_balance`, and `heat_balance` to internal meaning.

### [`axial_schemes.py`](/c:/MyRepos/multisolid-CL/packed_bed/axial_schemes.py)

Purpose:
- Holds reusable face-reconstruction formulas for finite-volume transport.

Why it exists:
- The bed solver needs interchangeable reconstruction schemes without embedding all of them directly inside the model equations.

Important contents:
- `SUPPORTED_SCHEMES`
  - The currently supported scheme ids.
- `validate_scheme_name(scheme_name)`
  - Rejects unsupported scheme names early.
- `_minmod(...)`
  - Slope limiter helper for MUSCL/minmod.
- `reconstruct_face_left_value(...)`
  - Builds the left-biased state at a face.
- `reconstruct_face_right_value(...)`
  - Builds the right-biased state at a face.
- `reconstruct_face_states(...)`
  - Convenience wrapper returning both face states.

### [`solid_profiles.py`](/c:/MyRepos/multisolid-CL/packed_bed/solid_profiles.py)

Purpose:
- Holds pure-Python helpers for mapping zone-based solid input data onto the model grids.

Why it exists:
- The solver and the visualization layer both need the same interpretation of `solids.yaml`.
- Keeping that logic out of [`solver.py`](/c:/MyRepos/multisolid-CL/packed_bed/solver.py) avoids duplicating zone-to-grid rules and keeps the plotting code independent of DAETOOLS.

Important contents:
- `build_uniform_axial_grid(...)`
  - Builds the default uniform face and cell-center coordinates from `bed_length_m` and `axial_cells`.
- `zone_edges(...)`
  - Returns the authored zone-edge array for plotting zone-wise data.
- `build_solid_profile_matrix(...)`
  - Expands the species concentration input onto the cell-center grid.
- `build_cell_scalar_profile(...)`
  - Expands zone-wise scalar properties like `e_b` and `e_p` onto the cell-center grid.
- `build_face_scalar_profile(...)`
  - Expands zone-wise scalar properties like `d_p` onto the face grid.
  - If a face falls exactly on a zone boundary, it averages the left and right zone values.
- `gas_fraction_from_voidages(...)`
  - Computes the total gas-filled bed fraction from `e_b` and `e_p`.
- `solid_fraction_from_voidages(...)`
  - Computes the complementary solid bed fraction.
- `convert_solid_profile_to_bed_volume(...)`
  - Converts authored solid concentrations to the solver's bed-volume basis when the input is authored on a solid-volume basis.

### [`validation.py`](/c:/MyRepos/multisolid-CL/packed_bed/validation.py)

Purpose:
- Checks that the requested simulation is internally consistent before any DAETOOLS objects are created.

Why it exists:
- Failures should happen at the configuration boundary, not deep inside solver initialization.

Important contents:
- `PackedBedValidationError`
  - Package-specific validation exception.
- `_validate_unique(...)`
  - Reused duplicate-check helper.
- `_validate_scalar_channel(...)`
  - Checks scalar program steps.
- `_validate_composition_channel(...)`
  - Checks composition schedules.
- `_validate_solid_profile(...)`
  - Checks:
    - solid species list
    - concentration basis
    - zone coverage
    - continuity
    - species completeness per zone
    - nonnegative concentrations
    - valid `e_b`, `e_p`, and `d_p` values in every zone
- `validate_run_bundle(...)`
  - Main validator.
  - Checks:
    - gas and solid species are valid,
    - required correlations exist,
    - selected reactions are compatible,
    - program inputs are valid,
    - model/run parameters are sane,
    - requested reports are known.

### [`visualization.py`](/c:/MyRepos/multisolid-CL/packed_bed/visualization.py)

Purpose:
- Produces static pre-run visual artifacts.

Why it exists:
- Users should be able to inspect the chemistry and operating program before trusting the simulation run.

Important contents:
- `GraphNode`, `GraphEdge`, `SystemGraph`
  - Dataclasses describing the system graph in a package-neutral form.
- `build_system_graph(...)`
  - Creates the bipartite species/reaction graph from the selected chemistry.
- `_series_from_segments(...)`
  - Small helper for converting compiled program segments into plot-ready series.
- `render_system_graph(...)`
  - Renders the chemistry graph to PNG and SVG using `networkx` and `matplotlib`.
- `render_operating_program(...)`
  - Renders the four supported operating-program channels to PNG and SVG.
- `render_initial_solid_profile(...)`
  - Renders a four-panel initialization artifact from `solids.yaml` to PNG and SVG.
  - The figure includes:
    - the authored solid concentration input,
    - the initialized bed-basis solid concentration profile,
    - the voidage and derived volume-fraction profiles,
    - the face-domain particle characteristic length profile.

### [`solver.py`](/c:/MyRepos/multisolid-CL/packed_bed/solver.py)

Purpose:
- Holds the actual DAETOOLS model, simulation wrapper, initialization logic, and execution helpers.

Why it exists:
- This is where the numerical packed-bed model lives.

This file is the most important file in the package and currently also the messiest one.

Key top-level helpers:
- `_default_model_config()`
  - Builds default geometry/constant settings when nothing is supplied.
- `_default_solid_config(...)`
  - Builds a default solid profile for quick starts and compatibility.
- `_build_solid_profile_matrix(...)`
  - Compatibility wrapper around the shared solid-profile helper.
- `_convert_solid_profile_to_bed_volume(...)`
  - Converts authored solid concentrations from `mol/m^3 solid` to the solver’s `mol/m^3 bed` basis when needed.

Legacy-compatible program classes:
- `ProgramStep`
- `ProgramSegment`
- `_coerce_inlet_composition(...)`
- `_default_inlet_composition(...)`
- `ScalarProgram`
- `VectorProgram`

Why these are here:
- `CLBed_mass` and `simBed` were ported from the old monolithic script with minimal behavioral change.
- These program utilities were kept inside the solver during that extraction.
- The package also has the cleaner equivalents in [`programs.py`](/c:/MyRepos/multisolid-CL/packed_bed/programs.py).
- Long-term, these two implementations should be unified.

Main model class:
- `CLBed_mass(daeModel)`
  - Defines domains, variables, parameters, and residual equations for the packed bed.

Important methods on `CLBed_mass`:
- `SetAxialGridFromFaces(face_locations)`
  - Sets a custom axial grid.
- `SetUniformAxialGrid(n_cells)`
  - Convenience wrapper for uniform grids.
- `SetOperationProgram(...)`
  - Compiles and stores the active boundary-condition schedules.
- `_segment_expression(...)`
  - Converts one program segment into a DAETOOLS expression.
- `_declare_program_equations(...)`
  - Emits scalar IF/ELSE program equations.
- `_declare_indexed_program_equations(...)`
  - Emits indexed program equations for vector channels like inlet composition.
- `DeclareEquations()`
  - Builds the full model:
    - phase-volume closures
    - species balances
    - EOS and density
    - mixture molecular weight and viscosity
    - flux reconstruction
    - Ergun pressure closure
    - enthalpy closures
    - energy balance
    - cumulative mass and heat inventories
    - placeholder solid source term

Simulation wrapper:
- `simBed(daeSimulation)`
  - Packages the DAETOOLS model into a runnable simulation.

Important methods on `simBed`:
- `__init__(...)`
  - Captures runtime choices and creates the `CLBed_mass` model.
- `SetUpParametersAndDomains()`
  - Pushes geometry, constants, grid, zone-derived `e_b` and `e_p` cell profiles, face-domain `d_p`, and boundary constants into the DAETOOLS model.
- `SetUpVariables()`
  - Computes the steady initial guess / initial condition state, including:
    - solid initialization on the model's bed-volume basis,
    - inlet-state initialization,
    - pressure profile guess,
    - transport and enthalpy variables,
    - inventory variables.
  - This method now uses the face-distributed `d_p` profile during Ergun and axial-dispersion initialization.

Nested helper functions inside `SetUpVariables()`:
- `ergun_terms(...)`
  - Computes coefficients used in the pressure initialization.
- `pressure_profile_from_inlet(pin)`
  - Computes the cell pressure profile from a candidate inlet pressure.
- `outlet_pressure_residual(pin)`
  - Residual used to solve for the inlet pressure that matches the imposed outlet pressure.

Assembly and run helpers:
- `SimulationAssembly`
  - Small immutable wrapper bundling `RunBundle` and the constructed `simBed`.
- `configure_evaluation_mode()`
  - Configures DAETOOLS equation evaluation mode.
- `build_idas_solver(...)`
  - Creates and configures the IDAS solver.
- `assemble_simulation(...)`
  - Converts a validated `RunBundle` into a runnable `SimulationAssembly`.
  - Also enforces the current reaction policy: metadata-only reactions are rejected here.
- `_set_reporting_on(...)`
  - Enables model reporting before execution.
- `run_assembled_simulation(...)`
  - Actually initializes and runs the DAETOOLS simulation using a no-op reporter.
- `guiRun(...)`
  - Legacy GUI-oriented entry point retained for manual DAETOOLS UI use.

### [`export.py`](/c:/MyRepos/multisolid-CL/packed_bed/export.py)

Purpose:
- Converts DAETOOLS reporter output into CSV files.

Why it exists:
- DAETOOLS exposes reported variables in its own internal data structures.
- The package needs stable, simple outputs for non-Python users.

Important contents:
- `_get_reported_variable(...)`
  - Looks up a DAETOOLS variable by path and gives a useful error if it is missing.
- `_flatten_variable(...)`
  - Converts DAETOOLS arrays into long-form row dictionaries.
- `_write_rows(...)`
  - CSV writer helper.
- `_extract_balance_data(...)`
  - Computes the material and heat balance tables and balance errors from reported totals.
- `export_run_outputs(...)`
  - Writes:
    - `run_summary.csv`
    - `balances.csv`
    - one CSV per requested report id

### [`examples/default_case/chemistry.yaml`](/c:/MyRepos/multisolid-CL/packed_bed/examples/default_case/chemistry.yaml)

Purpose:
- Minimal chemistry selection example.

Why it exists:
- Shows how to choose gas species and reaction ids.

### [`examples/default_case/solids.yaml`](/c:/MyRepos/multisolid-CL/packed_bed/examples/default_case/solids.yaml)

Purpose:
- Example initial solid profile.

Why it exists:
- Demonstrates the new solid-profile schema, including:
  - ordered solid species,
  - concentration basis,
  - zone-based axial profile,
  - zone-wise `e_b`,
  - zone-wise `e_p`,
  - zone-wise `d_p`.

### [`examples/default_case/program.yaml`](/c:/MyRepos/multisolid-CL/packed_bed/examples/default_case/program.yaml)

Purpose:
- Example operating program.

Why it exists:
- Shows the current four supported boundary channels and the hold/ramp syntax.

### [`examples/default_case/run.yaml`](/c:/MyRepos/multisolid-CL/packed_bed/examples/default_case/run.yaml)

Purpose:
- Top-level run definition.

Why it exists:
- This is the file the CLI points at.
- It wires together the chemistry, solids, and program files and adds solver/output settings.

### [`examples/default_case/output/`](/c:/MyRepos/multisolid-CL/packed_bed/examples/default_case/output)

Purpose:
- Example generated outputs from a successful run.

Why it exists:
- Mainly for inspection and sanity-checking during development.

Contents:
- CSV report files such as `temperature.csv`, `pressure.csv`, `solid_mole_fraction.csv`, `material_balance.csv`, and `heat_balance.csv`.
- `run_summary.csv`
- `balances.csv`
- `artifacts/` with the generated plot/image files.

These are generated files, not source code.

## How The Files Fit Together

If you run:

```bash
python -m packed_bed packed_bed/examples/default_case/run.yaml
```

the flow is:

1. [`__main__.py`](/c:/MyRepos/multisolid-CL/packed_bed/__main__.py) calls [`cli.py`](/c:/MyRepos/multisolid-CL/packed_bed/cli.py).
2. [`cli.py`](/c:/MyRepos/multisolid-CL/packed_bed/cli.py) calls `load_run_bundle(...)` from [`config.py`](/c:/MyRepos/multisolid-CL/packed_bed/config.py).
3. [`validation.py`](/c:/MyRepos/multisolid-CL/packed_bed/validation.py) checks the resulting `RunBundle`.
4. [`api.py`](/c:/MyRepos/multisolid-CL/packed_bed/api.py) calls the visualization functions in [`visualization.py`](/c:/MyRepos/multisolid-CL/packed_bed/visualization.py).
5. [`solver.py`](/c:/MyRepos/multisolid-CL/packed_bed/solver.py) assembles the DAETOOLS model and runs it.
6. [`export.py`](/c:/MyRepos/multisolid-CL/packed_bed/export.py) writes the output CSVs.

## Current Design Notes

### Good separations already in place

- Config loading is separate from validation.
- Validation is separate from solver assembly.
- Visualization is separate from simulation.
- Export is separate from execution.
- Property and reaction libraries are separate from the numerical model.

### Known rough edges

- [`solver.py`](/c:/MyRepos/multisolid-CL/packed_bed/solver.py) still contains duplicated program classes also present in [`programs.py`](/c:/MyRepos/multisolid-CL/packed_bed/programs.py).
- The numerical model is still concentrated in a single large file.
- Reactions are metadata-only right now; executable kinetics are still the next step.
- The example `output/` directory is useful during development, but it is generated content, not package source.

## Practical Editing Guidance

If you want to:

- Add a new species:
  edit [`properties.py`](/c:/MyRepos/multisolid-CL/packed_bed/properties.py)

- Add a new predefined reaction:
  edit [`reactions.py`](/c:/MyRepos/multisolid-CL/packed_bed/reactions.py)

- Change YAML schema:
  edit [`config.py`](/c:/MyRepos/multisolid-CL/packed_bed/config.py) and [`validation.py`](/c:/MyRepos/multisolid-CL/packed_bed/validation.py)

- Add a new requested report id:
  edit [`reporting.py`](/c:/MyRepos/multisolid-CL/packed_bed/reporting.py) and usually [`export.py`](/c:/MyRepos/multisolid-CL/packed_bed/export.py)

- Add a new visualization artifact:
  edit [`visualization.py`](/c:/MyRepos/multisolid-CL/packed_bed/visualization.py) and wire it in from [`api.py`](/c:/MyRepos/multisolid-CL/packed_bed/api.py)

- Change the actual packed-bed equations:
  edit [`solver.py`](/c:/MyRepos/multisolid-CL/packed_bed/solver.py)

- Change CLI behavior:
  edit [`cli.py`](/c:/MyRepos/multisolid-CL/packed_bed/cli.py)
