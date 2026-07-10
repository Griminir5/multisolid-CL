# Architecture

The normal runtime has one resolved configuration, one model-construction path, and one
result-extraction path:

```text
YAML files
  -> config.load.load_case
  -> resolved Case
  -> simulation.run_case
  -> PackedBedSimulation
  -> model.PackedBedModel + initialization.InitialState
  -> simulation.execute_simulation
  -> reports.extract_dataset
  -> results.nc + manifest.json
```

## Runtime flow

1. **Load and resolve configuration.**
   [`config/load.py`](../packed_bed/config/load.py) parses the run, chemistry, program,
   and solids YAML files into the data-only models in
   [`config/models.py`](../packed_bed/config/models.py). It resolves reaction-family
   names through `kinetics.FAMILY_REGISTRY`, compiles operating programs, and validates
   species properties, reactions, report names, shapes, and references. The result is a
   frozen `Case`, which is the only handoff to the runtime.

2. **Assemble the runtime.**
   `simulation.run_case` is the library entry point used by the single-case CLI and batch
   runner. It supplies the default `PROPERTY_REGISTRY` unless a registry is passed
   explicitly. `PackedBedSimulation` builds a `ReactionNetwork` from the reactions
   selected in the case, resolves the corresponding kinetics hooks, and constructs one
   `PackedBedModel`.

3. **Initialize and execute.**
   [`initialization.py`](../packed_bed/initialization.py) first calculates a numerical
   `InitialState`, then applies its domains, parameters, and variable values at the
   DAETools boundary. [`model.py`](../packed_bed/model.py) owns the DAETools variables
   and the deliberately ordered `DeclareEquations` block. `execute_simulation` selects
   the linear solver, enables only the requested reporting targets, initializes, solves,
   runs, and finalizes through one cleanup path.

4. **Extract one labelled result.**
   The local reporter created by `reports.create_dataset_reporter` receives the DAETools
   process. [`reports.extract_dataset`](../packed_bed/reports.py) is the sole conversion
   to an `xarray.Dataset`; it assigns labelled `time`, cell, face, species, and reaction
   coordinates and adds requested derived values. The dataset is written as `results.nc`
   with the SciPy NetCDF engine.

5. **Summarize and record the run.**
   `RunResult` carries the dataset and output paths. Balance summaries and standard plots
   consume that dataset rather than re-reading solver objects. `write_run_manifest`
   writes `manifest.json` with configuration choices, environment and Git provenance,
   input/output hashes, dataset inventory, runtime, and any requested balance summaries.
   Failures before manifest writing produce the same manifest with the failing stage and
   traceback when possible.

The batch path in [`batch.py`](../packed_bed/batch.py) expands and resolves every case
before writing case files or starting expensive work, then invokes the same
`simulation.run_case` function for each case.

## Extension points

| Concept | Interface | Where to extend |
|---|---|---|
| Kinetics family | A module exports one `ReactionFamily` named `FAMILY`, containing `ReactionDefinition` objects, family-local species requirements, and one callable hook per reaction. Hooks receive a `KineticsContext`. | Add the family module under [`kinetics/`](../packed_bed/kinetics/) and add its `FAMILY` explicitly to `FAMILY_REGISTRY` in [`kinetics/__init__.py`](../packed_bed/kinetics/__init__.py). |
| Species properties | A `SpeciesProperties` record supplies phase, molecular weight, enthalpy correlation, and gas viscosity where required. Correlations implement numeric `value` and symbolic `dae_expression`. | Add records or correlation types in [`properties.py`](../packed_bed/properties.py). `validate_case` and `run_case` also accept an explicit `PropertyRegistry`. |
| Reports | `REPORT_SPECS` maps a public report id to model variables; `MODEL_VARIABLES` maps reported DAETools variables to dataset names and dimensions. Derived results are added after central extraction. | Update both mappings as applicable in [`reports.py`](../packed_bed/reports.py), and extend `extract_dataset` only for a genuinely derived value. |
| Axial schemes | `reconstruct_face_states` returns left and right states for one interior face; `split_face_flux` applies the transport direction. | Add the public id to `SUPPORTED_SCHEMES` and its boundary-safe reconstruction in [`axial_schemes.py`](../packed_bed/axial_schemes.py). Both configuration validation and the model use this interface. |

Report selection normally changes recording and dataset contents, not the physical model.
The current mass- and heat-balance reports are the explicit exception: their observer
variables and equations are constructed only when requested. Reaction-rate variables are
essential whenever reactions are selected and are merely recorded when their report is
requested.

## Import-safety boundary

Configuration and validation must work without DAETools and must not write files. The
safe side of this boundary includes configuration models and loading, compiled programs,
reaction metadata and family discovery, numerical property evaluation, axial schemes,
and report metadata.

Solver-backed imports are deliberately deferred:

- `model.py` and `simulation.py` are imported by the CLI only when a run
  starts;
- kinetics families use the lazy symbols in `kinetics/runtime.py`, so inspecting family
  metadata does not load DAETools;
- property correlations import DAETools units only when building symbolic expressions;
- `reports.py` imports xarray only for extraction/loading and DAETools only when creating
  the reporter;
- plotting, optional graph, and GUI dependencies are imported only when their explicit
  options are used.

Keep dependencies pointing from runtime code toward the safe modules. In particular,
configuration, registry metadata, and validation must not import `model.py` or
`simulation.py`; this preserves `--validate-only` as a solver-free, side-effect-free
operation.
