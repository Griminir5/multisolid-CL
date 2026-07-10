# multisolid-CL Refactor Completion Plan

## Purpose

Finish the refactor so that the program is substantially easier to read, understand, modify, and trust. The intended result is a smaller codebase with fewer layers, fewer private helper functions, fewer duplicate representations, and a clear path from configuration to model construction to simulation to results.

This is an architectural refactor of the overall program. Reaction kinetics remain modular, swappable scientific inputs whose correctness is ultimately the responsibility of the person supplying or selecting them.

## Decisions and constraints

These decisions are fixed for this refactor:

- Organize reaction kinetics by **reaction family**, including support-specific families where their component requirements differ.
- Split the current copper implementation into at least:
  - copper on SiO2;
  - copper on Al2O3.
- A family may only require the components it actually uses. Selecting copper on SiO2 must not require aluminium oxides or spinels that cannot exist in that family.
- Keep regularization constants as clearly named module-level values inside each kinetics file. Do not turn them into general model parameters.
- Do not preserve obsolete kinetics versions or IDs merely for compatibility. Remove inferior variants, make the preferred implementation the default, and update tracked examples and documentation in the same change.
- Do not repair the iron kinetics as part of this refactor.
- Do not add a kinetics testing suite, rate-grid regression tests, scientific reference-value tests, or a reactive smoke test.
- Do not test whether the derivative of enthalpy is heat capacity.
- Do not enforce the stated validity temperature range of a property correlation. Scientific applicability remains the user's responsibility.
- Keep `DeclareEquations` as one contiguous block. Its internal ordering must remain visible and directly controllable for future iterative solver work.
- Retain the central, linear-upwind, and MUSCL schemes, alongside the useful upwind and WENO schemes listed below.
- Permit flow reversal inside the bed through conservative flux splitting. Keep the existing boundary-condition assumptions unchanged.
- Evaluate xarray as the internal result representation before committing to it.
- Keep conversion to machine-learning data as a separate operation so its feature set can evolve independently.

## Refactoring principles

### Prefer visible flow over indirection

The normal execution path should be easy to follow:

1. load and validate configuration;
2. construct a case;
3. compile operating programs;
4. select properties and reaction families;
5. build the model;
6. initialize and run it;
7. extract one results dataset;
8. produce requested reports and plots.

Avoid chains of wrappers that merely rename arguments, forward calls, or conceal mutations. A helper is worthwhile when it captures a real concept, removes substantial repetition, or isolates a difficult rule. It is not worthwhile merely because a block is several lines long.

### Reduce concepts, not just underscores

The goal is not to turn private functions into public functions. The goal is to remove unnecessary functions and intermediate representations. Closely related logic should stay together when reading it in sequence is clearer than jumping between small helpers.

### Keep modules substantial and cohesive

Prefer a small number of understandable modules over a large tree of tiny files. Split a module only when its responsibilities are independently explainable and commonly changed or tested in isolation.

### Separate infrastructure validation from scientific validation

The program should validate structure: required fields, dimensions, names, component availability, compatible shapes, and resolvable references. It should not claim that a user-selected property correlation or kinetics family is scientifically correct for a particular experiment.

### Preserve behaviour deliberately

Before changing a subsystem, record its current externally useful behaviour. Preserve that behaviour unless this plan explicitly removes it. Prefer small, reviewable changes over a second wholesale rewrite.

## Target package structure

The exact package name can follow the repository's existing convention, but the production code should converge on roughly this structure:

```text
multisolid_cl/
    config/
        models.py
        load.py
    kinetics/
        __init__.py
        nickel_medrano.py
        reforming_xu_froment.py
        reforming_numaguchi.py
        copper_sio2.py
        copper_al2o3.py
        iron_he.py
    programs.py
    properties.py
    reactions.py
    axial_schemes.py
    solid_profiles.py
    model.py
    initialization.py
    simulation.py
    batch.py
    reports.py
    plots.py
    diagnostics.py
    cli.py
tools/
    to_ml_matrix.py
tests/
```

This is a destination, not a requirement to rename everything at once. Each move should delete the superseded path in the same change so there is only one authoritative implementation.

## Detailed design

### 1. Configuration: two modules and one resolved case

Consolidate the current fragmented configuration package into:

- `config/models.py`: declarative, data-only configuration models;
- `config/load.py`: file parsing, duplicate-key detection, defaults, reference resolution, and structural validation.

Use one resolved `Case` object as the handoff to the runtime. Remove `RunBundle` and other overlapping containers if they represent the same information. Do not retain flat compatibility aliases for nested fields unless an active caller genuinely needs them.

`load_case(...)` should return a complete, structurally valid case or one useful error report. Validation must:

- have no file-writing side effects;
- avoid importing DAETools;
- report errors using configuration paths meaningful to the user;
- detect duplicate keys rather than accepting the last one silently;
- resolve program, property, component, reaction-family, scheme, and output references;
- validate shapes, required fields, unique names, zone coverage, and compatible dimensions;
- avoid enforcing scientific validity ranges or scientific correctness.

Pure configuration imports and `--validate-only` must work on a machine without DAETools.

### 2. Operating programs: one compiled representation

Represent runtime programs with a small pair of types such as:

- `ProgramSegment`;
- `CompiledProgram`.

Configuration models describe user input; compiled programs perform interpolation and lookup. Remove additional wrapper types and duplicated flattened arrays if the compiled object already supplies the operation.

Compile and validate all programs before constructing the solver model. Put GHSV-basis support in ordinary program configuration instead of implementing it as a batch-only mutation.

### 3. Properties: consolidate correlation mechanics

Replace the several polynomial heat-capacity implementations with one `PolynomialHeatCapacity` that accepts coefficients and a documented coefficient convention. Retain genuinely different correlations as separate concepts, for example:

- `PolynomialHeatCapacity`;
- `ShomateHeatCapacity`;
- `QuadraticViscosity`;
- `SpeciesProperties`;
- `PropertyRegistry`.

Keep the distinction between enthalpy and viscosity explicit. Remove boilerplate subclasses that only provide coefficients.

Property validation should check that the selected correlation exists, has the required interface, receives structurally valid parameters, and returns compatible values in ordinary use. It should not:

- test `dH/dT == Cp`;
- enforce a correlation's advertised temperature range;
- judge whether a correlation is appropriate for the user's material or experiment.

### 4. Reactions: family-owned definitions and dependencies

Keep the generic reaction framework in `reactions.py`. It should contain only the concepts shared by the whole program, such as:

- `ReactionDefinition`;
- `ReactionNetwork`;
- a compact `ReactionFamily` data object or protocol;
- the explicit family-loading mechanism.

Put each actual kinetics family in one readable module under `kinetics/`. Each module owns:

- its reaction definitions;
- its rate expressions and hooks;
- its module-level regularization constants;
- its required gas and solid components;
- one exported `FAMILY` object that describes the family to the core.

Use support-specific modules when support chemistry changes the available species. In particular:

```text
kinetics/copper_sio2.py
kinetics/copper_al2o3.py
```

The SiO2 family must not mention or require `Al2O3`, `CuAl2O4`, `CuAlO2`, or other spinel-related components. The Al2O3 family may declare those components when its mechanism uses them.

Avoid decorator registration and import-time side effects. An explicit registry in `kinetics/__init__.py` or a small loader makes the available families visible and searchable. Keep DAETools imports lazy where practical so metadata and configuration can be inspected without loading the solver stack.

Do not introduce a formal kinetics versioning scheme. When a better implementation replaces a worse one:

- remove the worse implementation;
- use the clear canonical family/reaction name for the preferred implementation;
- update repository-owned configurations and documentation atomically;
- rely on Git history and the run manifest for provenance.

Testing here is limited to program infrastructure: the family registry loads, references resolve, and family-local component requirements remain isolated. It must not test reaction rates, scientific parameters, reactive trajectories, or the correctness of a mechanism.

### 5. Axial schemes and reversible interior flow

Keep these schemes for now:

- first-order upwind;
- central;
- second-order linear upwind;
- MUSCL with minmod limiter;
- WENO3;
- WENO5.

Remove the broken WENO7 implementation, the old fifth-order linear-upwind implementation, and helpers used only by those schemes.

Give every retained reconstruction one clear interface that can produce left and right face states. Avoid scheme-specific call paths in the model equations.

Permit velocity reversal at any interior face using conservative flux splitting. For a transported concentration, use the equivalent of:

```text
u_plus  = (u + abs(u)) / 2
u_minus = (u - abs(u)) / 2
convective_flux = u_plus * c_left + u_minus * c_right
```

Apply the same directional split to the advective enthalpy flux, using the appropriate left and right reconstructed states. Keep dispersion terms unchanged. At zero velocity the convective contribution should vanish cleanly.

Do not redesign inlet and outlet boundary conditions in this refactor. Document that internal faces support reversal while the existing boundary orientation and conditions remain in force.

Infrastructure tests should cover constant and linear reconstruction where applicable, boundary-safe stencil selection, mirrored left/right behaviour, and positive, negative, and zero-velocity flux splitting. These are transport tests, not reaction tests.

### 6. Model, initialization, and execution

Split the current solver module into exactly three cohesive files:

- `model.py`: DAETools variables, parameters, domains, ports, and equations;
- `initialization.py`: calculation and application of initial state;
- `simulation.py`: solver selection, assembly, execution, finalization, and runtime options.

Rename opaque legacy classes to direct names such as `PackedBedModel` and `PackedBedSimulation`.

#### Keep `DeclareEquations` contiguous

Keep `DeclareEquations` as one method and one ordered block. Do not extract its sections into private declaration methods. Use strong section comments and local variables to make the sequence readable while leaving the exact declaration order apparent:

```text
DeclareEquations
    base declarations
    program/interpolation equations
    gas-phase balances
    solid-phase balances
    energy balances
    momentum/pressure equations
    boundary equations
    reporting/derived equations
```

During cleanup, preserve equation names, declaration order, variable order, and the solver incidence structure unless a separately justified change requires otherwise. Add a small infrastructure regression that records those names and ordering. This is especially important before iterative solver support is introduced.

Use local, descriptive calculations inside the block rather than creating many one-use private functions. Extract only a repeated mathematical concept that is easier to understand independently, such as the shared face-flux calculation.

#### Initialization

Separate pure initial-state calculation from applying values to DAETools objects. Keep initialization data in one clearly shaped structure rather than several parallel dictionaries.

Do not add a reactive smoke case. A very small inert/nonreactive execution case may exercise grid construction, initialization, transport, pressure, solver assembly, and output extraction without making claims about any kinetics family.

#### Simulation

Use an explicit solver registry instead of wildcard imports or string-to-global lookup. Define and document the meaning of thread counts, including `threads=0`, in one place. Ensure cleanup and final result extraction occur through one execution path.

### 7. Batch execution and CLI

Make the CLI a thin adapter over ordinary Python functions. It should parse arguments, invoke the same load/run/report functions used by library callers, and translate exceptions into concise messages and exit codes.

`--validate-only` must not create directories, plots, manifests, or other files.

Keep batch orchestration in one coherent module rather than spreading it through a small package. Replace dotted indexed mutations with structured recursive patches or typed override data. Validate all generated cases before starting any expensive runs.

Batch execution must provide:

- safe, readable case slugs;
- output path containment;
- pre-run collision detection;
- deterministic override application;
- explicit handling of failed cases;
- no batch-specific scientific configuration rules.

Plots and expensive diagnostics should be explicit options, not side effects of every run.

### 8. Results: one extraction path and an xarray spike

First implement a focused xarray spike using one representative inert result and one realistic result shape. Compare it with the current pandas/MultiIndex path on:

- extraction code size and readability;
- labelled dimensions and coordinate safety;
- gas-cell versus face data;
- units and metadata;
- NetCDF round trips;
- plotting ergonomics;
- balance calculations;
- straightforward downstream ML conversion;
- dependency and runtime cost.

Adopt xarray only if it materially removes custom axis, reshaping, and MultiIndex code. The likely representation is a single `xarray.Dataset` with relevant dimensions such as:

```text
time
x_cell
x_face
gas_species
solid_species
reaction
```

If the spike succeeds, create exactly one central extraction function, for example `extract_dataset(process, case)`. Reports, plots, balances, and durable output must consume this dataset rather than re-reading solver objects independently.

A run may then write:

- `results.nc` for labelled numerical data;
- `manifest.json` for reproducibility and file inventory;
- only the explicitly requested reports or plots.

Keep a lean `RunResult` containing paths, summary status, and optionally the in-memory dataset. Do not store redundant success flags, duplicate balance paths, several forms of the same metadata, or the full simulation object by default.

Derived solid mole fractions belong in extraction/post-processing unless the solver itself requires them. Changing output selection must not change the DAE system.

#### ML conversion remains separate

Do not embed a fixed feature matrix or a permanent list of ML features into core result extraction. Put conversion in a separate tool such as `tools/to_ml_matrix.py`, or in a downstream project. It should consume the labelled dataset and explicitly select/derive the features wanted at that time.

### 9. Reports, plots, diagnostics, and provenance

Merge overlapping result metadata and report writers into one `reports.py` with one authoritative mapping from report names to fields. Consolidate ordinary static plotting in `plots.py`. Avoid one module per plot type unless a plot becomes genuinely complex.

Retain useful incidence diagnostics in a compact, reproducible form, such as a CSV plus a static image. Remove the very large interactive HTML output if it is not actively used.

Write a run manifest that records infrastructure provenance without formal kinetics versioning. Useful fields include:

- Git commit and dirty-worktree state;
- Python, package, DAETools, and solver versions;
- resolved configuration and input hashes;
- selected family, reaction, property, and scheme names;
- grid, tolerance, and runtime settings;
- output filenames, dimensions, units, and hashes;
- runtime and balance summaries;
- failure stage and traceback when a run fails.

The selected family name and Git commit are enough to identify the kinetics implementation used. Do not duplicate each regularization constant as a model parameter merely for the manifest.

### 10. Research and one-off scripts

Remove research scripts from importable production modules. Delete obsolete scripts. Move useful ones under `tools/`, give them a main guard and explicit CLI arguments, and remove hard-coded local paths and import-time execution.

Likely candidates include the task-text generator, hard-coded result extractor, optimizer experiments, and import-time property fitting. They should not affect package import or runtime behaviour.

## Testing strategy

The suite should protect program structure and numerical plumbing without pretending to validate user science.

### Fast pure-Python suite

Run without DAETools and cover:

- configuration parsing, defaults, duplicate-key errors, and useful paths in errors;
- structural validation and reference resolution;
- zone and program compilation;
- family discovery and family-local component requirements;
- batch patching, safe paths, and collision detection;
- all retained axial reconstruction schemes;
- positive, negative, and zero-velocity flux splitting;
- xarray extraction and durable round trips if the spike is adopted;
- report schemas and manifest generation;
- import safety and absence of file-writing side effects during validation.

### Solver infrastructure checks

Use a tiny inert/nonreactive case to cover only:

- model construction;
- equation names and declaration order;
- initial-state application;
- transport and pressure execution;
- result extraction;
- basic mass/energy accounting at the program-infrastructure level.

Keep this local or manual if DAETools is not readily available in CI. Ordinary CI should run the pure suite on supported Python versions, initially 3.11 and 3.12.

### Explicitly excluded tests

Do not add:

- reaction-rate reference tests;
- kinetics parameter regression grids;
- reactive smoke cases;
- nickel-only cases presented as general reaction validation;
- iron kinetics repair or validation;
- enthalpy-derivative versus heat-capacity tests;
- property validity-temperature enforcement;
- assertions that a user-selected scientific model is suitable for a real system.

## Implementation sequence

### Phase 0 — preserve the baseline

1. Preserve the current refactor branch as a reviewable checkpoint.
2. Start the completion work from an up-to-date master branch, carrying over only deliberate refactor changes.
3. Record baseline production line count, import time, test status, sample inert run outputs, equation names/order, and output schemas.
4. Add lightweight formatting, linting, and pure import checks if they are not already reliable.

**Exit condition:** there is a reproducible baseline and every later deletion can be compared against it.

### Phase 1 — establish the pure core

1. Consolidate configuration into `models.py` and `load.py`.
2. Introduce the single resolved `Case` handoff.
3. Consolidate operating programs.
4. Make validation side-effect-free and independent of DAETools.
5. Add structural tests for configuration, programs, and references.

**Exit condition:** representative cases can be loaded, resolved, and validated without DAETools or filesystem output.

### Phase 2 — simplify properties and define reaction families

1. Consolidate property correlation classes.
2. Reduce the generic reaction layer to its shared data structures and loader.
3. Define the family interface and explicit registry.
4. Migrate one simple family as a proof of the interface.
5. Verify component requirements are family-local without testing scientific rate behaviour.

**Exit condition:** configuration selects an explicit family whose metadata and required components resolve through pure code.

### Phase 3 — simplify CLI and batch workflow

1. Make the CLI call the pure load/validate/run functions.
2. Make `--validate-only` side-effect-free.
3. Replace dotted batch mutations with structured patches.
4. Move GHSV-basis handling into normal program configuration.
5. Add safe output naming, containment, and collision checks.

**Exit condition:** a single case and a batch use the same configuration and execution path.

### Phase 4 — reorganize the solver and implement flow splitting

1. Create `model.py`, `initialization.py`, and `simulation.py`.
2. Rename the model and simulation classes.
3. Keep `DeclareEquations` as one contiguous, commented, ordered method.
4. Unify the reconstruction interface for retained axial schemes.
5. Remove WENO7 and old fifth-order linear upwind.
6. Implement species and enthalpy flux splitting at interior faces.
7. Keep boundary conditions unchanged and document the limitation.
8. Add transport-scheme and equation-order infrastructure checks.

**Exit condition:** the inert case runs through the new structure, internal reversed velocity selects the correct face state, and the equation sequence remains deliberate and visible.

### Phase 5 — unify results and reporting

1. Run and document the xarray spike.
2. If successful, adopt one labelled dataset and one extraction path; otherwise retain the simpler existing representation and still centralize extraction.
3. Merge report metadata and writers.
4. Consolidate plotting.
5. Add the compact manifest and durable results output.
6. Move ML conversion into its own tool consuming the stable results artifact.

**Exit condition:** every report, plot, balance, and downstream conversion starts from the same extracted result.

### Phase 6 — migrate and clean up kinetics families

1. Move remaining families into one-family modules.
2. Split copper on SiO2 from copper on Al2O3.
3. Keep each family's regularization values as local module globals.
4. Remove inferior duplicate kinetics implementations and make the preferred form canonical.
5. Update owned configurations and documentation to the canonical names.
6. Leave iron scientific behaviour unchanged.
7. Do not add kinetics correctness tests.

**Exit condition:** all bundled kinetics are swappable family modules, and no selected family drags in irrelevant solid components.

### Phase 7 — delete, document, and harden

1. Delete superseded modules, aliases, wrappers, dead schemes, duplicate reports, and obsolete scripts.
2. Search for orphan imports, stale names, import-time work, and hard-coded paths.
3. Update the README with one quick-start path and one clear configuration example.
4. Add a short architecture document explaining the runtime flow and extension points.
5. Correct and simplify kinetics-source documentation without adding a versioning system.
6. Run the full pure suite and the local inert solver check.
7. Compare final line count, module count, helper count, and execution outputs with the baseline.

**Exit condition:** there is one obvious way to perform each core operation, and deleted compatibility code has no live callers.

## Suggested review and merge units

Keep the work reviewable in four larger, coherent changes rather than dozens of file-move-only commits:

1. **Pure core:** configuration, programs, properties, and initial family interface.
2. **Workflow:** CLI, batch execution, validation behaviour, and path safety.
3. **Solver and results:** model split, contiguous equations, reversible interior flux, xarray decision, reports, and provenance.
4. **Family migration and cleanup:** support-specific copper families, remaining family moves, canonical naming, deletions, and documentation.

Within each change, move code and delete its old implementation together. Avoid long periods with two supported paths.

## Quantitative guardrails

Line count is a useful pressure toward simplicity, not a target to game. Production code should plausibly fall from roughly 10,000 lines to about 6,800–7,500 lines while tests become more focused. Directional subsystem targets are:

| Area | Approximate target |
| --- | ---: |
| Configuration | 450–550 lines |
| Model, initialization, simulation | 1,050–1,150 lines |
| Batch execution | 400–500 lines |
| Results and reports | 400–550 lines |
| Plots and visualization | 450–600 lines |
| Properties | 300–400 lines |
| Reactions and bundled family modules | 1,900–2,200 lines |
| Diagnostics | 200–350 lines |

Also track:

- number of production modules;
- number of top-level private helpers;
- number of duplicated runtime representations;
- number of import-time side effects;
- number of separate paths that extract or reshape results.

A reasonable goal is to reduce the roughly 200 top-level private helpers on the refactor branch toward 100–120, but only by inlining one-use plumbing, merging overlapping concepts, and deleting dead code—not by renaming helpers or creating large unstructured functions.

## Definition of done

The refactor is finished when:

- a new reader can follow the normal run path through a short list of modules;
- pure configuration and validation work without DAETools and write no files;
- there is one resolved case, one compiled-program representation, and one results extraction path;
- reaction kinetics are explicit, swappable families with family-local component dependencies;
- copper on SiO2 does not require Al2O3 or spinel components;
- regularization values remain readable globals within their family files;
- obsolete kinetics variants can be removed without compatibility scaffolding;
- `DeclareEquations` remains one readable, deliberately ordered block;
- central, first-order upwind, second-order linear-upwind, MUSCL-minmod, WENO3, and WENO5 are supported through one reconstruction interface;
- interior species and enthalpy transport handle positive, negative, and zero velocity through flux splitting;
- existing boundary-condition assumptions remain unchanged and documented;
- the xarray decision is supported by a small measured spike rather than preference alone;
- ML feature conversion is separate from core result extraction;
- scientific property and kinetics validity are not falsely guaranteed by infrastructure tests;
- no reactive smoke case, iron repair, or kinetics test suite has entered scope;
- obsolete wrappers, duplicate representations, broken schemes, and research leftovers are gone;
- the production code is materially smaller and its remaining abstractions correspond to concepts a human reader needs.

