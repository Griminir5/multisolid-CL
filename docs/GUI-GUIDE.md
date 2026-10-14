# GUI guide

This guide covers the MultiSolid desktop application: creating cases, running
simulations, comparing results, and managing parameter studies. Installation and
development setup are in the [repository README](../README.md). For scientific
definitions, see the [model equations](EQUATIONS.md) and [plugin guide](PLUGINS.md).

A **project** contains cases, studies, reusable definitions, and plugins. A
**case** contains its inputs and **one latest run**. Editing scientific inputs
keeps its existing results and marks them **stale**. Starting another run replaces that
case's previous results, including when the new run later fails or is cancelled.

| Task | Go to |
| --- | --- |
| Define your own case | [Create and edit a case](#create-and-edit-a-case) |
| Run a case for the first time | [First run](#first-run) |
| Choose a solver or run several cases | [Run simulations](#run-simulations) |
| View plots and logs | [Inspect results](#inspect-results) |
| Export one case to Excel | [Case Report](#export-a-case-to-excel) |
| Compare several cases in Excel | [Project Results](#compare-cases-in-project-results) |
| Vary inputs systematically | [Parameter studies](#parameter-studies) |
| Transfer a project | [Share project inputs](#share-project-inputs) |
| Recover unfinished edits | [Saving and recovery](#saving-and-recovery) |
| Resolve a problem | [Troubleshooting](#troubleshooting) |

## Projects and navigation

The welcome screen offers **Create new project**, **Open existing project**,
**Import archived project**, and recent projects. Opening a project resumes it
in place. Creating or importing requires a new folder; an existing folder is
never merged or overwritten.

On Windows, the default parent is `%USERPROFILE%\Documents\MultiSolid`, independent
of a Documents location redirected by OneDrive. On other systems it uses the OS
Documents location, falling back to `~/Documents/MultiSolid`. **Browse…** selects
another parent, and the application remembers a custom creation location.
Use a local, writable folder outside the application installation. Project
archives can be transferred to synced storage when you want to share inputs.

The **Project** menu provides creation, opening, saving, closing, folder access,
archives, recent projects, and reusable definitions. **Plugins** opens the
project's scientific definitions. The **Help** menu opens bundled HTML in your
browser, including Getting started, the User guide, reference guides, and
Troubleshooting. **Help for this screen** (**F1**) opens a dedicated short page
for the welcome screen, project screen, each case tab, Project Results, or
parameter studies. **Search help…** opens the help home with search across the
bundled pages. Help works offline, including the equations; links marked
**online** require an internet connection.

**About MultiSolid** shows application and component versions and lets you
preview and copy diagnostic information. **Licences and notices** shows the
application's notices.

Each case row shows input readiness and the latest run status separately:

| Input status | Meaning |
| --- | --- |
| Ready | Inputs and the required runtime pass the execution checks. |
| Underdefined | Some required values are unfinished. |
| Invalid | Inputs conflict or contain unsupported values. |
| Runtime unavailable | The selected backend or solver is missing a required component. |
| Needs update | The case belongs to a study that must be rebuilt. |

**Not run**, **Queued**, **Running**, **Succeeded**, **Failed**, **Cancelled**, and
**Interrupted** describe the latest execution. A **stale** suffix means that run
does not match the current scientific inputs or selected plugin contents. It can
still be inspected when its files are available.

Use the row's **Run**, **Duplicate**, **Edit**, and **Delete** icons. Double-click
to edit; press **F2** on a case name to rename it. Deleting asks before removing
the case and its latest results. Study groups can be collapsed in the case list.

Recent projects also appear under **Project → Recent projects**, with their
locations and last-interaction times. Opening, editing, and starting execution
update recency; background result writes do not. Unavailable folders are omitted.
If you move or copy a project, open it once with **Open existing project** to add
its new location. Opening a project elsewhere does not change the preferred
parent for newly created projects.

## Create and edit a case

Choose **New Case** and give it a name. It starts as an incomplete draft, so
select the chemistry and supply the physical inputs before running it. Drafts
save automatically, including unfinished values.

New cases start in **Feed** mode with **Repeat program** enabled, 20 axial cells,
WENO3 mass and heat schemes, and **Compiled / Band** execution with one thread
and relative tolerance `1e-7`. **Outlet composition** and **Outlet conditions**
plots and their required reports are selected. These defaults do not fill in the
bed, species, feed, or a useful time horizon for you.

The editor has five tabs:

| Tab | What you define |
| --- | --- |
| General | Simulation duration, reporting interval, spatial discretization, solver, and recorded quantities/plots |
| Chemistry | Gas and solid species, reaction families, selected reactions, and role bindings |
| Bed | Geometry, heat transfer, material zones, voidages, and solid loadings |
| Program | Initial inlet conditions, outlet pressure, and timed hold/ramp steps |
| Report | An Excel workbook layout using recorded data |

For a new case, it is often easiest to complete **Chemistry → Bed → Program**,
then review **General**. Red field borders and fills identify missing or invalid
values; hover for an explanation. A red tab marker points to a tab with errors.
Rules involving several values, such as composition totals, highlight the group.
**Details…** explains errors that cannot be attached to a single field.

### General: numerics and recorded outputs

Set the simulation horizon, reporting interval, number of cells, and mass/heat
schemes. The reporting interval controls saved samples; it is separate from the
solver's internal time steps. For a non-repeating program, the horizon is derived
from the timed steps and cannot be edited here.

Choose **Backend**, **Linear solver**, **Number of threads**, and **Relative
tolerance**. **Advanced solver settings…** exposes additional tolerances and
integration controls. See [Run simulations](#run-simulations) for availability and
compiled execution.

**Requested reports** selects the quantities to record. **Requested plots**
selects plots and identifies the reports they require. Request the quantities you
will need before running: changing an Excel layout later cannot create data that
were never recorded. The **Report** tab arranges those data into worksheets; it
does not control what the solver records.

### Chemistry: species and reactions

1. Add gas and solid species from their respective pickers. Each definition
   shows its source.
2. Choose **+ Add reaction families…** and add the required models.
3. Select the reactions to include. Use a family's **+ Species** action to add
   missing required components.
4. Inspect **Bindings…** when a role could refer to more than one selected
   component, such as two property variants of the same chemical.
5. Set the new species' gas fractions in **Program** and solid loadings in
   **Bed** as needed. Added values default to `0`; removal deletes the
   species' entries from compositions and zones.

Selecting a family makes its reactions available; inspect the reaction
checkboxes to choose what will run. Rates from overlapping selected models add.
If you intend one model to replace another, remove the unwanted selection.

The reaction graph updates in the background. Click a node to highlight its
direct neighbours and connecting links; click again or click empty space to
clear the highlight. Scroll or use **+ / −** to zoom, drag to pan, and choose
**Fit** to show the whole graph. The layout refits when the pane resizes.

The graph uses Graphviz `neato`. Source installations can use it on `PATH` or
set `MULTISOLID_GRAPHVIZ` to its executable. See the
[source installation instructions](../README.md#4-install-the-external-tools)
if it is missing. The packaged desktop application includes Graphviz.

### Bed: geometry and material zones

Set the radius and length, ambient temperature, heat-transfer coefficient,
gas-occupied bed fraction, and flow reversibility. **Heat dispersion (W/m/K)**
sets effective axial heat spreading; `0` disables it. The implemented
[boundary conditions and transport equations](EQUATIONS.md) explain these choices.

Each material-zone row contains its start/end positions, interparticle and
intraparticle voidages, particle diameter, and each selected solid's loading.
Choose the **Solid density basis** (bed or solid volume) and **Concentration
units** (mol/m³ or kg/m³) that match your data. Changing these selectors changes
how the entered values are interpreted; it does not convert those numbers.
The displayed **wt%** is calculated from the zone's initial loadings and selected
molecular weights, not an additional input.

The first zone starts at zero and the last ends at the bed length. **+ Add zone**
splits the final zone, defaults its solid loadings to `0`, and leaves its voidages
and particle diameter blank. Complete the zones so they cover the bed without
gaps or overlaps. Changing a multi-zone bed's length offers fixed or
proportionally scaled internal boundaries.

The preview below the settings displays the sampled bed profiles used by the
engine. Use it to inspect your zones after changing boundaries or cell count.

### Program: operating conditions over time

Set the initial conditions first, then add timed steps.

| Mode | How you specify the inlet |
| --- | --- |
| Feed | One feed state combines flow, temperature, and composition. A ramp can change selected parts together. Unticked values carry over from the preceding feed. |
| Independent channels | Flow, temperature, and composition have separate step lists. |

Outlet pressure has its own channel in both modes. Choose **Hold** to retain the
current state for a duration, or **Ramp** to reach a target over that duration.
Targets display their units. Switching modes preserves each mode's inlet steps;
the first visit to another mode copies the initial feed values and starts with
no inlet steps. Outlet pressure is shared.

In a composition dialog, new species values default to `0`. Set non-negative
mole fractions for the gases present so the total is one.
**Normalize** scales a positive total to one while preserving ratios. It becomes
available once all entries are valid and their total is positive.

**Repeat program** repeats each channel's step list, with an editable overall
horizon in **General**. Each cycle carries forward the previous cycle's final
values; add a ramp back to the starting state if the cycle should return there.
For a non-repeating program, the longest channel sets the horizon, and every
non-empty channel must have the same duration. A non-repeating program with no
timed steps needs at least a hold or ramp to define its horizon. Hand-authored
[CLI inputs](CLI-GUIDE.md#holds-ramps-and-repetition) can instead specify an
explicit horizon with all channels constant.

**Flow basis** switches between mol/s and GHSV (h⁻¹). This conversion uses the
bed geometry and reference conditions of **273.15 K and 100,000 Pa**. Set a
positive radius and length before converting existing numeric flows.

The preview shows the engine's smoothed schedules, repetition, and GHSV
conversion. Feed transitions interpolate molar flow, species molar flows, and
flow × temperature; see [Operating programs](EQUATIONS.md#operating-programs).
Use a channel's arrow to collapse it, or **Focus** to give it more room;
**Show all** restores the channels. Preview toolbars support zoom, pan, and image
saving.

### First run

Once you have completed the case inputs in the tabs above:

1. Choose **← Back to project** and check that the case shows **Ready**. If it
   does not, open the case and inspect the highlighted inputs or **Details…**.
2. Choose the case's **Run** icon. Hover over the icons for their labels.
3. After **Succeeded**, open the case, go to **General**, and choose **Show**
   beside an available requested plot. Use **Report** to prepare an Excel export.

**Checkpoint:** the case list shows a successful run and the case can display its
retained results. Use **Open latest run log** if execution fails.

To compare a changed case with this result, duplicate the original and edit the
copy. The original keeps its result; the duplicate starts with copied inputs and
report layout but no results.

If you prefer to start with existing case files, choose **Import Case** and select
their `run.yaml`. Keep its referenced YAML files alongside it in their original
layout. The application copies the inputs into your project and leaves the
originals unchanged. Review the imported case's tabs before running it.

For users who have the source repository, the
[default case](../examples/simulations/default_case/run.yaml) is a complete example.
It uses **Standard / SuperLU**, which importing preserves.

## Run simulations

Return to the project case list and use a case's **Run** icon. For a batch,
check the cases to include and choose **Run all included cases**. All included
cases are validated before any execution begins; uncheck unfinished drafts.
A failure in one running case does not stop the remaining cases.

**Maximum workers** controls simultaneous cases and is saved with the project.
Each case keeps its own thread setting. The two controls are independent:
raising the worker limit does not edit a case's numerical settings.
Elapsed time updates while a case is active and freezes when it finishes.
**Cancel execution** stops active workers and their children and cancels the queue.

### Choose a backend and solver

| Backend | Offered linear solvers | What to expect |
| --- | --- | --- |
| Standard | SuperLU, KLU, LAPACK (dense), AztecOO with ILUT / Ifpack ILU / ML, SUNDIALS GMRES with Ifpack ILU | Uses the installed DAETools solver adapters. |
| Compiled | SuperLU, KLU, Band | Builds a native model and reuses compatible compiled models. Availability depends on the installed runtime and compiler. |

The dropdown explains unavailable choices. Switching to an incompatible backend
or solver asks before updating affected settings; **Cancel** keeps the original
inputs. Imported advanced settings are preserved. An unsupported selection
blocks execution instead of silently substituting another solver.

Both backends need DAETools; Compiled uses it for model construction and initial
conditions. Source installations need the matching adapters and native runtime
described in the [installation guide](../README.md#install-the-full-repository-from-source).
The Windows distributions include their compiler, Graphviz, and solver libraries.
KLU factorization is serial on both backends.

### Compiled progress and caching

A first compiled run may spend several minutes generating and compiling its model.
Progress distinguishes cache checks, waiting for compilation, generation, and
compilation from actual integration. Later compatible cases reuse
`project/.packed_bed_cache/`, including cases in studies. Different operating
schedules can share a compiled model while keeping their own program data.

The cache survives reruns and is excluded from project archives. It can be deleted
while the project is idle and will be rebuilt when needed. The project must be
writable. For source builds, see the
[compiled runtime guide](../desktop/BUILD_WINDOWS.md#compiled-runtime).

### What a rerun replaces

Once a case starts, the new attempt replaces its previous input snapshot, logs,
status, and outputs. A later failure or cancellation does not restore the old
run. A failed preflight, or cancellation while a case is still queued, leaves
its previous results intact.

Duplicate a case before editing or running a comparison if you want the original
result to remain available. Duplicates copy inputs and the Report definition,
then start with **Not run**. Prepared runs use fixed input and plugin copies, so
later edits cannot change the inputs associated with a retained result.

## Inspect results

Open the case and use **General → Requested plots → Show** for a separate plot
window. Available retained NetCDF data can be plotted even when stale. Plot
toolbars provide pan, zoom, and image saving.

**Open case folder** opens the case's files; its `run/` folder contains the
latest snapshot, solver log, and outputs. **Open latest run log** opens
`worker.log`. Outputs include NetCDF data, the engine provenance manifest, and
any generated plots.

Choose **Report** for one case's Excel workbook, or **Results…** in the project
case list to compare successful runs. A stale result still describes its retained
run's inputs, not your current draft. Reports include those retained inputs and
provenance.

## Export a case to Excel

**Report** opens on the first user worksheet, starting with a blank sheet for a new
configuration. Name it and choose what its rows represent
(time by default, or any available coordinate axis). **Add columns…** selects a
quantity and values for its other axes. Three species at three positions create
nine columns; remove, reorder, or rename individual columns after adding them.
Every column heading includes the value unit. Hover over a preview heading to
read its full definition.

Coordinates and available quantities always follow the current case inputs.
Row selections support All, First, Last, and Selected values. **All** columns
regenerate when the grid changes; surviving columns keep their labels and order,
and manually removed positions stay excluded. First/Last follows the endpoint.
Manually selected coordinates that no longer exist are removed with a notice;
unfinished inputs preserve selections until coordinates can be calculated again.
Missing quantities remain visible for repair. When results exist, the preview
shows recorded values and coordinates from the retained run, limited to 20 rows
and 12 columns; export contains the full selection. Case information also shows
the retained run's inputs and provenance. Before a run, the preview shows expected
coordinates and headers and explains why value cells are empty. Editing controls
continue to follow current inputs, including when retained results are stale.

Older saved reports stored every generated coordinate as an exact selection.
Re-add those columns with **All** to enable automatic grid updates; the old format
does not record whether they were originally selected individually or with All.

**Export workbook…** writes native Excel Tables, with the header row and first
column frozen. The fixed first sheet, **Case information**, contains the retained
run's settings, provenance, and an exact dictionary of the exported columns.
Stale or partial results can be exported when the selected data and their input
snapshot are available. No data are interpolated or silently dropped. Export
can be cancelled and only replaces an existing workbook after success.

Reports autosave with the project independently of scientific inputs. Editing a
report does not make results stale, and reports remain editable on generated
study cases. Duplicating a case copies its report definition without results.
**Save as template…** writes a portable JSON layout. **Apply template…** previews
its sheets, sizes, and any compatibility issues before replacing the current
configuration. Each applied template is an independent editable copy. Templates
contain no results, case identity, or run provenance.

## Compare cases in Project Results

Click **Results…** beside **Run all included cases**. A dialog first asks which
cases to make available in Results. Only cases whose latest run succeeded and
whose result files are present can be selected. Select cases individually or
check a study group; filtering also matches factor selections. **Select all** and
**Clear all** affect every eligible case, including cases hidden by the filter.
These checkboxes are independent of Run all inclusion. Cancel keeps the project
case list open. **Select cases…** inside Results reopens the same dialog.

The worksheet controls work like the case Report tab, with **Case** as another
axis. Choose what rows represent, then **Add columns…**:

- **Time** rows with five cases and three outlet species give 15 value columns.
- **Position** rows compare profiles at explicitly selected recorded times.
- **Case** rows compare individual recorded values. Time defaults to **Last**;
  First/Last resolves separately for each case. This does not calculate an average.

Shared rows must have the same recorded coordinates. For different grids, use
**Split by case** to replace the current sheet with a separate, editable worksheet
for each case, or select row coordinates recorded in all referenced cases. The
export never interpolates or silently matches nearby samples. Cell positions and
face positions remain separate axes.

All expands to explicit columns when added, using each case's own grid. Saved
columns keep their case identities and coordinate selections. Missing cases or
coordinates remain visible for repair after a rerun or study rebuild. First/Last
follows the retained run's endpoints; add columns again to include new cases or
coordinates. **Refresh results** reloads the current retained data.

Preview shows up to 20 rows and 12 columns; **Export workbook…** exports the full
selection as native Excel Tables. The fixed **Results information** sheet includes
run identities, study selections, retained inputs, status, staleness, available
balance diagnostics and exact per-case coordinates for every value column.
Successful but stale results can be exported if their selected data and snapshot
are readable. Missing quantities or samples block the affected worksheet with a reason.

**Save as template…** saves the worksheet layout as JSON, without run data or case
identities. **Apply template…** previews the sheets, sizes, compatibility and case
mapping before replacing the layout. Select the same number of cases as the
template; its case positions map to the selected cases in their displayed order.
Applied templates are independent copies, and edits do not change the template file.

For a failed run, open the individual case's **Report** tab to export any saved
partial data. A normal solver failure can retain samples recorded before the
failure. These show the state at reporting times, not rejected solver iterations
or necessarily the exact failure instant. A hard crash, forced stop, or failure
before recording may leave no exportable data; consult **Open latest run log** too.

Use Excel for your own averages, integrals and plots. Project Results layouts
autosave and support draft recovery independently of case Report definitions and
scientific inputs. Export can be cancelled; an existing workbook is replaced only
after success. Existing individual-case plot windows remain available.

## Parameter studies

Run the intended baseline case successfully first. **New Parameter Study** accepts
only a case whose latest run succeeded, whose current inputs are ready, and whose
successful snapshot still matches those inputs. The study copies those inputs as
a fixed baseline, records the successful attempt, and does not copy results.
**Inspect baseline** is read-only; **Replace baseline** selects another successful
case. Subsequent changes to the source case do not change the saved baseline.
Selected plugins still use the project's current contents for future runs; see
[Project plugins](#project-plugins).
The baseline also retains a copy of its report configuration. Every generation
or rebuild copies that layout independently into each fresh case, with All axes
resolved against that case's inputs. Older studies without a saved report layout
capture their source report on their next rebuild, if the source is still available.
Study rows have edit and delete icons. Deleting a study asks for confirmation,
then removes its generated cases and results while keeping independent cases
and reusable definitions.

The study workspace contains variations beside a live case preview:

- **+ Add variation** selects a named parameter with units, or a reusable program or
  bed configuration. Numerical variations accept value lists or linearly spaced
  ranges. Program fields include initial values, selected step durations, and ramp
  targets. Step references survive earlier-step deletion and never move silently
  to another step.
- **All combinations** produces the Cartesian product. **Explicit case rows**
  specifies individual pairings, with definition dropdowns and numeric cells.
  The row table supports spreadsheet copy/paste, duplicate, move, remove, and
  undo/redo. Blank cells remain unfinished.
- Length sweeps scale all solid-zone boundaries proportionally. Program and bed
  replacements cannot overlap numerical variations inside the replaced inputs.
  Timing, species, compositions, geometry, and solver choices use engine validation.
- Inspect a preview row for the same program and bed plots available in the case
  editor. Invalid scientific inputs can become draft cases; ambiguous or unfinished
  variation rules must be repaired before generation. Previews are cancellable.
- **Create N cases** writes cases without running them. Generated inputs are
  read-only; use **Edit study** or **Duplicate as independent case**.

For example, three programs × three bed configurations produce nine cases. With
independent Benchmark and Exploratory cases, the project has 11 cases. Benchmark
must have succeeded before it is selected as the study baseline. Definitions and
the copied baseline do not add simulations to the count.

### Rebuilding a study

Autosave stores the study rule without changing its generated cases. Changing the
baseline, variations, or a referenced definition marks the study **Needs rebuild**
and all its generated cases **Needs update**. Those cases cannot run until rebuilt;
existing results remain inspectable. Reverting the changes clears this requirement.
Renaming a study or definition does not require a rebuild.

**Replace X cases with Y** deletes **every** previous generated case in that study,
including its inputs, snapshots, logs, and results, then creates fresh cases with
new identities, generated names, and default inclusion. Even unchanged combinations
are replaced. The preview states the case and result counts before this action.
There are no archived cases or restoration. Rebuilds preserve independent cases,
other studies, the fixed baseline, and the reusable-definition library.

### Reusable programs and beds

Use **Project → Reusable definitions…**, or create/edit definitions while choosing
a study variation. Programs contain the operating program, its mode and repetition
setting. Bed configurations contain length, radius, ambient temperature,
heat-transfer coefficient (U), axial heat dispersion, gas voidage mode, flow
reversibility, and solids. The existing input controls edit these definitions
with explicit **Save / Cancel**. Inherited inputs
provide preview context. Unused definitions remain in the library; deleting a
referenced definition is blocked until its study references are removed.

Older bed definitions continue inheriting any settings they do not yet contain.
Editing and saving them captures these settings from the displayed inputs. A bed
variation cannot be combined with a numerical sweep of a setting that bed owns.

Create desktop studies with **New Parameter Study** using a successful baseline
case. **Run all included cases** executes included independent and generated cases
together, using the project's **Maximum workers** setting and each case's thread count.

### CLI batches and older studies

Engine `batch.yaml` files are used through the CLI (`python -m packed_bed batch --help`),
and cannot be imported as desktop studies. The desktop still supports importing an
individual case's `run.yaml` and importing complete project archives.

Studies imported by earlier versions remain readable and retain their saved rules.
For those existing advanced studies, only settings overridden by the original batch
preset apply; other settings still come from the baseline. A pending imported study
can still use **Add baseline as independent case** and **Replace baseline** to
establish a successful baseline. No saved cases or results are removed by retiring
the batch-import action.

## Share project inputs

Choose **Project → Export project…** to save a `.msproject` archive. This includes
every case (including unfinished drafts), reusable definitions, parameter studies,
Report/Results layouts and registered plugins. Results, run snapshots, logs and
local settings are excluded. Save the archive outside the project folder; a
cancelled or failed export leaves an existing archive unchanged.

Choose **Project → Import project archive…**, available even before opening a
project. Select the archive, then choose a project name and parent location using
the normal creation form. The destination must be a new folder. A successful
import opens the project and adds it to Recent projects. Imported cases start at
**Not run**; their saved report layouts can be used again after running the cases.
Established study baselines and studies awaiting rebuild retain their state.

Archives carry the current plugin catalogue, including unused/disabled plugins.
Importing and browsing never execute supplied Python code or grant local approval.
Use **Plugins → Allow code… / Enable** when needed. Missing/incompatible packages,
damaged archives and unsafe file paths block import with an explanation. Transfers
run in a cancellable dialog while the project is idle.

## Project plugins

Open **Plugins** beside **Project**. Register a `.msplugin` archive or a plugin
folder's `manifest.yaml`, enable it, and select its species or families in
**Chemistry**. Registration alone does not change case selections.

Plugins in an imported project archive are already registered. Enabled data-only
plugins work immediately. For a Python plugin marked **Needs local approval**,
choose **Allow code…** and wait for **Checks passed.** Disabled plugins remain
disabled until enabled.

Importing and browsing do not execute plugin Python. Approval permits code to
run with your user permissions during checks and simulations; workers are not a
security sandbox. Approval stays on this computer. Code/resource changes require
approval again; manifest parameter edits do not.

Cases and studies use the current contents of their selected project plugins for
future runs, including studies with a fixed baseline. Retained run snapshots
keep their original plugin copies. The [plugin guide](PLUGINS.md) explains
variants, replacement, updates, and sharing; the
[example walkthrough](../examples/plugins/README.md) teaches plugin creation.

## Saving and recovery

Case inputs, Report definitions, Project Results layouts, and study rules save
after a short pause, including unfinished values. **Project → Save Project**,
switching projects, and closing flush pending edits, including the active table
cell. If saving fails, the draft stays in the editor and navigation is blocked;
read the reason in the status bar and resolve the write problem before leaving.

Edits create a recovery checkpoint in `.drafts/` before the autosave delay.
After an interruption, opening the project offers:

| Choice | Effect |
| --- | --- |
| Recover drafts | Restore the unfinished edits and reassess readiness and staleness. |
| Discard drafts | Keep the saved inputs and discard the recovery edits. |
| Cancel | Leave the recovery decision pending without opening the project. |

Recovery preserves retained run snapshots and results. It does not resume a
solver. Unfinished retained runs are marked **Interrupted** after reopening.
Project and execution locks prevent concurrent access.

An input archive excludes run data. To retain a full project with its latest
results, close it and copy the entire project folder. Each case still has only
one latest run; the copied folder does not provide a history of earlier attempts.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Run is unavailable | Check input readiness, highlighted tabs, and **Details…**. Complete the draft or resolve its runtime requirement. |
| Run all refuses to start | Every included case must pass validation. Uncheck unfinished cases or rebuild cases marked **Needs update**. |
| First compiled run seems slow | Check whether progress says generating or compiling. The first build can take several minutes; later compatible runs use the cache. |
| Solver is unavailable | Read the dropdown's explanation. For a source installation, check DAETools adapters, compiler, and native runtime setup. |
| Chemistry graph is missing | Check Graphviz `neato` or `MULTISOLID_GRAPHVIZ`, then use **Retry** in the graph pane. |
| A composition will not normalize | Enter every species value, keep them non-negative, and ensure the total is positive. |
| Program horizon cannot be edited | Non-repeating programs derive it from timed steps. Match all non-empty channel durations or enable repetition and set the horizon. |
| A new species makes the case incomplete | Gas fractions and solid-zone loadings default to `0`. Check that gas fractions sum to one and inspect reaction bindings. |
| A plugin has no effect | Select its definitions in Chemistry or use the plugin replacement action. See the [plugin guide](PLUGINS.md). |
| Results are stale | Inputs or selected plugin contents have changed. Inspect/export the retained run, or run the case again for current inputs. |
| A plot or workbook lacks a quantity | Check whether that quantity was recorded in the retained run. Request it in General and rerun if needed. |
| Cases cannot share a Results worksheet | Select exactly shared recorded row coordinates or use **Split by case**. Export does not interpolate. |
| A failed run is missing from Project Results | Use that case's **Report** tab for any saved partial samples and inspect its log. Project Results selects successful runs. |
| A study baseline cannot be selected | Run a ready case successfully and keep its inputs unchanged before selecting it. |
| A study case cannot be edited | Edit the study or duplicate the generated case as an independent case. Report layouts remain editable. |
| Saving or opening is blocked | Read the status message, check write access, and close another process using the project. Preserve the open draft until saving succeeds. |
| A moved project is missing from Recent projects | Open its new location with **Open existing project**. |

## Appendix: project files and developer reference

```text
project/
  project.json                   # name, cases, studies, max_workers, registered plugins
  plugins/<plugin-id>/current/    # current registered plugin files
  .drafts/                       # recovery checkpoints for unfinished edits
  .packed_bed_cache/             # reusable compiled models; excluded from archives
  cases/<case-id>/
    inputs/                      # run/program/solids/chemistry.yaml for this case
    run/                         # one latest execution, replaced on rerun
      inputs/                    # fixed inputs associated with the latest result
      snapshot.json              # input hashes, fingerprint, case/attempt identity
      status.json
      worker.log
      output/                    # NetCDF, engine manifest, and plots
  studies/<study-id>/
    study.json                   # rule, fixed-baseline provenance, stable step targets
    baseline/                    # four copied scientific documents
    batch.yaml                   # compatibility: rule saved by an older batch import
  definitions/<definition-id>/   # metadata plus program.yaml or solids.yaml
  execution.json                 # latest Run case / Run all queue status
  execution.log                  # worker startup diagnostics
```

Prepared snapshots briefly live in `.pending-<attempt>` beside a case's `run/`.
Once validated, a snapshot replaces that case's run folder; `.previous-run` is a
short-lived backup during the directory swap. Recovery completes an interrupted
swap. Attempt identities distinguish worker events; they are not run history.
Input/metadata files use atomic replacement. Study and definition writes use a
staged directory transaction. Rebuilding commits project metadata only after
staging the complete replacement case set. Before that
commit, recovery restores old cases; after it, recovery deletes temporary backups.
Recovery runs before loading case inputs. Project archives transfer saved
definitions and exclude recovery checkpoints and run data.

File publication and temporary-directory cleanup share the engine's
[`file_io.py`](../packed_bed/file_io.py) policy. Status files, results, manifests,
cache records, plugin files and exports use unique temporary files before atomic
replacement. Brief Windows reader locks are retried for up to 0.75 seconds;
permanent access errors still surface. Run/study folder swaps use the same retry
policy while retaining their existing rollback order. Export cancellation is
checked again before each replacement attempt. Temporary-file and scratch-directory
cleanup preserve the original failure if cleanup also fails.

The application reads the current project format (3) and run-snapshot format (1).
Older project formats are rejected rather than migrated. Project and solver locks
prevent concurrent access. Reopening after an interrupted session marks unfinished
retained runs as Interrupted.

| File | Responsibility |
| --- | --- |
| [window.py](../desktop/packed_bed_ui/window.py) | Welcome screen, project case list, navigation, and actions |
| [case_list.py](../desktop/packed_bed_ui/case_list.py) | Collapsible study groups, case status, inclusion, and icon actions |
| [editor.py](../desktop/packed_bed_ui/editor.py) | Shared input controls, read-only inspection, case autosave, and retained plots |
| [studies.py](../desktop/packed_bed_ui/studies.py) | Parameter catalogue, pure expansion, validation, and generation signatures |
| [study_store.py](../desktop/packed_bed_ui/study_store.py) | Baseline eligibility, persistence/imports, and transactional complete rebuilds |
| [study_editor.py](../desktop/packed_bed_ui/study_editor.py) | Study workspace, variation dialogs, and cancellable previews |
| [study_tables.py](../desktop/packed_bed_ui/study_tables.py) | Preview and explicit-row models, spreadsheet paste, and definition dropdowns |
| [definition_editor.py](../desktop/packed_bed_ui/definition_editor.py) | Reusable-definition library and shared inspection dialogs |
| [inputs.py](../desktop/packed_bed_ui/inputs.py) | Explicit draft initialization, input validation, timing, and zone scaling |
| [general.py](../desktop/packed_bed_ui/general.py) | Numerical/solver settings and report/plot selections |
| [chemistry.py](../desktop/packed_bed_ui/chemistry.py) | Species lists and reaction families |
| [reaction_graph.py](../desktop/packed_bed_ui/reaction_graph.py) | Asynchronous Graphviz SVG preview, node highlighting, zoom and pan |
| [bed.py](../desktop/packed_bed_ui/bed.py) | Geometry, material zones, and reactor boundary rules |
| [program_editor.py](../desktop/packed_bed_ui/program_editor.py) | Program modes, hold/ramp tables, feed targets, timing, and flow conversion |
| [editor_widgets.py](../desktop/packed_bed_ui/editor_widgets.py) | Shared selection lists, tables, and preview canvases |
| [project.py](../desktop/packed_bed_ui/project.py) | Project/case storage, imports, readiness, and run preparation |
| [execution.py](../desktop/packed_bed_ui/execution.py) | Start, monitor, and stop the project's worker with Qt |
| [worker.py](../desktop/packed_bed_ui/worker.py) | Verify/activate snapshots, collect logs, publish status, call the shared engine loop |
| [results.py](../desktop/packed_bed_ui/results.py) | Project Results, using the shared Report worksheet controls |
| [project_results.py](../desktop/packed_bed_ui/project_results.py) | Case-axis selections, native-grid validation and project Excel export |
| [batch.py](../packed_bed/batch.py) | Shared case scheduling, concurrency, thread limits, and worker cleanup |
| [preview.py](../packed_bed/preview.py) | Scientific preview arrays without Qt or plotting imports |

Study selectors and the project library share `DefinitionList` and its editing actions.
Both the workspace and storage use `preview_study()`; its lazy mode lets Qt consume
one captured preview incrementally. Shared document access, scientific fingerprint
inputs, timing, and scaling live in `inputs.py`.

`ProjectCase.resolve()` uses the engine's metadata-only `inspect_case()`.
Workers load the snapshots and call `run_case()` with its optional status callback. The UI and
batch CLI both use `run_cases_in_processes()` for concurrent scheduling and
worker cleanup. The CLI retains its in-process path for one worker with no timeout.

Each project stores one current copy of each plugin; Save replaces that copy.
Code/resource approval belongs to the local computer and survives parameter edits.
Workers reconstruct authoritative definitions from plugin copies in their input
snapshots. See the [plugin guide](PLUGINS.md) for workflows and the API.

### Source setup and packaging checks

Use the [repository installation and development instructions](../README.md#development).
Opening projects and generating numerical previews with built-in definitions do
not require DAETools. Python plugins may require it when their implementations
load; both execution backends require it. From an activated source environment,
`packed-bed-ui` and `python -m packed_bed_ui` launch the same application.
An optional argument may be a project folder or its `project.json`.

For relevant engine or packaging changes, run these from the repository root:

```sh
python -m packed_bed examples/simulations/default_case/run.yaml --validate-only
python -m packed_bed.compiled.smoke
python -m pip wheel --no-deps --no-build-isolation --wheel-dir build/wheels ./desktop
```

The compiled smoke check requires a working compiler and a managed native runtime
including KLU; the scikit-SUNDAE wheel alone does not supply KLU.
Wheel building checks desktop packaging separately. See
[Building the Windows distribution](../desktop/BUILD_WINDOWS.md) for release
isolation, runtime staging, and installer/portable builds.

The interface follows system light/dark appearance and uses a shared numeric
font. Maratype supplies the wordmark, project title, and M/S ribbon; its
[font notice](../desktop/packed_bed_ui/assets/fonts/Maratype-LICENSE.txt) stays
beside the asset. The launch splash appears before the workspace loads;
solver workers run without it.

[Documentation index](../README.md#documentation)
