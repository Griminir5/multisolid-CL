# MultiSolid desktop starter

A project contains simulation cases. Each case owns its inputs and **one latest
run**. Running it again replaces its previous results. Editing inputs leaves
those results available and marks them **stale** until the case runs again.

## Start developing

Use Python 3.11/3.12 and the DAETools setup in the [repository README](../README.md).
From the repository root:

```sh
python -m pip install -e ".[dev]" -e ./desktop
packed-bed-ui
```

`python -m packed_bed_ui` starts the same application. Supply a project folder
as an optional argument to open it directly. Opening projects and generating
previews do not need DAETools. Execution needs DAETools with SuperLU.
The Chemistry graph uses a separate Graphviz `neato` runtime. See
[Graphviz setup and bundling](GRAPHVIZ.md); source development can also use
`neato` on `PATH` or the `MULTISOLID_GRAPHVIZ` executable override.

## Use the starter

1. At launch, choose **Create new project**, **Open existing project**, or a recent
   project. Creation asks for a name and parent location and shows the resulting
   folder. It starts with `MultiSolid` in the OS Documents location (home if that
   location is unavailable), and remembers a custom parent chosen for creation.
   The folder is created on submission; existing folders cannot be overwritten
   or merged. Opening a project resumes it in place.
2. Inside the project, use **New Case** to start from an empty draft in **Feed** mode with a repeating
   program and relative tolerance `1e-5`. **Import Case** copies a `run.yaml` and its referenced inputs
   into this project, preserving the original files.
3. Project actions are in the top **Project** menu. Each case row shows its name,
   input readiness, latest result, and icon actions: **Run**, **Duplicate**, **Edit**,
   **Delete**. Hover for tooltips. Double-click a case to edit it; press F2 on its
   name to rename it. **← Back to project** returns to the list. Delete asks before removing
   the case and its latest results.
4. The case editor has five tabs: **General**, **Chemistry**, **Bed**, **Program**,
   and **Report**. General contains numerical and solver settings and requested
   reports/plots. Chemistry selects species and collapsible reaction families
   beside a fitted reaction graph. Click a node to highlight its direct neighbours
   and connecting links; click it again or click empty space to clear. Scroll or
   use **+/−** to zoom, drag to pan, and use **Fit** to reset the view. Layouts
   update in the background after edits and fit again when the pane resizes.
   Bed places settings on the left and material zones on the right above
   the numerical preview. Program edits initial values and timed hold/ramp steps
   beside its preview, ordered as flow, temperature, composition, and pressure.
   Target headers show the units. Click a channel's arrow to collapse its settings
   and give the other channels more room. In a composition dialog, **Normalize**
   scales the entered mole fractions to sum to 1 while preserving their ratios.
   Fill every species with a non-negative number and use a positive total to enable it.
   Drafts save after a short pause, including incomplete
   values. Hover over a validation message for its full details.
5. **Run** in the project case list executes one case. **Run all included cases** runs the checked
   cases as one batch, continuing after individual failures. **Maximum workers**
   sets the number of simultaneous cases and is saved with the project. Each case
   keeps its own thread setting; increasing workers never changes case inputs. All included cases
   are validated before any execution starts. Uncheck drafts you want to leave
   out. Elapsed time updates while a case is active and freezes when it finishes.
   **Cancel execution** stops and reaps active workers and cancels the queue.
6. Open a case's folder for its latest NetCDF, manifest, and existing CLI plots.
   **Open latest run log** opens its solver log. General's plot rows have **Show**
   buttons that open separate windows when retained NetCDF results are available,
   including stale results. Report configures and exports Excel workbooks.

A rerun replaces the previous snapshot, logs, status, and outputs when that
case starts, even if the new attempt fails or is cancelled. Invalid preflight
and cases cancelled before starting keep their previous results. **Duplicate
case** first if you want to retain a separate comparison; the copy starts with
no results.

Recent projects appear on the launch screen and in **Project → Recent projects**,
with names, locations, and last-interaction times. The small index lives in the
user's system settings, independently of project folders. Opening, editing, or
starting execution updates recency; result writes and status polling do not.
Unavailable folders are omitted, with availability checked in the background.
Use **Open existing project** once to register a moved or externally copied project.
Opening elsewhere does not change the preferred parent for new projects.

Case inputs, Report definitions, and study edits save after a short pause, including
unfinished values. Saving, switching projects, and closing flush pending edits,
including active table cells. If saving fails, the draft stays in the editor and
navigation is blocked with the failure reason in the status bar.

Edits also write a small recovery checkpoint in the project's `.drafts/` folder
before the autosave delay. After an interruption, opening the project offers
**Recover drafts**, **Discard drafts** (keep the saved inputs), or **Cancel**.
Recovery reassesses input readiness and staleness. It preserves retained snapshots
and results, and never resumes a solver. Case saves reuse the study transaction
mechanism so an interrupted save cannot leave a mixture of old and new YAML files.

The Bed and Program previews use the engine's actual smoothed programs, feed mixing,
repetition, GHSV conversion, and sampled solid profiles. Plot toolbars support
zoom, pan, and image saving. Advanced scientific settings survive import/save;
an unsupported solver blocks execution rather than being silently replaced.
Choose **Standard** or **Compiled** in General. Standard offers SuperLU,
SuperLU_MT, KLU, UMFPACK, LAPACK, AztecOO with ILUT/Ifpack/ML, and SUNDIALS
GMRES with Ifpack. These reuse the bundled DAE Tools adapters; Pardiso remains
unavailable in the desktop. Compiled offers SuperLU, SuperLU_MT, KLU, and Band
when the managed runtime is installed. The dropdown explains unavailable choices. Switching to an incompatible
backend or solver asks once before updating the affected settings; Cancel keeps
the original inputs. Advanced solver settings open in a separate dialog.

Compiled runs build automatically and reuse `project/.packed_bed_cache/`, shared
by the project's cases and studies. The first build can take several minutes;
later compatible runs reuse it. Progress includes cache checking, waiting,
generation, and compilation. Cancel stops workers and compiler children. The cache
survives reruns, is excluded from project archives, and can be deleted while idle.
A writable project is required. KLU factorization is serial on both backends.
The packaged application supplies its compiler and libraries; source developers
can stage the same bundle using the [compiled runtime guide](COMPILED.md).

For a non-repeating program, the horizon is calculated from the longest channel
and is disabled in General. Every non-empty channel must have the same duration;
a program with no timed steps needs a hold or ramp to define its horizon. Cyclic
programs allow an editable horizon. Switching program modes preserves each mode's
inlet steps in case metadata; a mode opened for the first time copies the initial
feed values and starts with no inlet steps. Outlet pressure is shared and stays
in the same pane. Switching flow basis converts existing numeric flows using the
bed geometry and the engine's GHSV reference conditions.

Adding species leaves new composition fractions and solid loadings blank for the
user to fill. Removing species removes their entries from compositions and zones.
Reaction families expose an **+ Species** action for their declared requirements.
A new zone splits the final zone's extent and starts with blank material values.
The first zone starts at zero and the final zone ends at the bed length. Changing
a multi-zone bed's length offers fixed or proportionally scaled internal boundaries.

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

## Excel reports

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

## Project Results

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
The baseline also retains a copy of its report configuration. Every generation
or rebuild copies that layout independently into each fresh case, with All axes
resolved against that case's inputs. Existing studies capture their source report
on their first rebuild after this feature is added, if the source is still available.
Study rows have edit and delete icons. Deleting a study asks for confirmation,
then removes its generated cases and results while keeping independent cases
and reusable definitions.

The study workspace contains variations beside a live case preview:

- **Add variation** selects a named parameter with units, or a reusable program or
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

### Reusable definitions and imports

Use **Project → Reusable definitions…**, or create/edit definitions while choosing
a study variation. Programs contain the operating program, its mode and repetition
setting. Bed configurations contain length, radius, ambient temperature, heat-transfer
coefficient (U), gas voidage mode, flow reversibility, and solids. The existing input
controls edit these definitions with explicit **Save / Cancel**. Inherited inputs
provide preview context. Unused definitions remain in the library; deleting a
referenced definition is blocked until its study references are removed.

Older bed definitions continue inheriting any settings they do not yet contain.
Editing and saving them captures these settings from the displayed inputs. A bed
variation cannot be combined with a numerical sweep of a setting that bed owns.

**Import Parameter Study** copies an engine `batch.yaml` and its referenced inputs
into a pending study. Use **Add baseline as independent case**, run that case
successfully, then select it with **Replace baseline** before generating cases.
The original source files are preserved. Exactly representable rules become native
editable studies; other rules remain advanced imported studies with their original
axis order and patch behavior, available for inspection and generation. Advanced
rules retain the original preset ownership of companion settings.

Batch worker/time-limit settings remain in the portable original rule. Desktop
execution uses the project's Maximum workers setting and each case's thread count.

## Storage and code

```text
project/
  project.json                   # name, cases, studies, max_workers, extension requirements
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
    batch.yaml                   # portable original rule, for imported studies
  definitions/<definition-id>/   # metadata plus program.yaml or solids.yaml
  execution.json                 # latest Run case / Run all queue status
  execution.log                  # worker startup diagnostics
```

Prepared snapshots briefly live in `.pending-<attempt>` beside a case's `run/`.
Once validated, a snapshot replaces that case's run folder; `.previous-run` is a
short-lived backup during the directory swap. Recovery completes an interrupted
swap. Attempt identities distinguish worker events; they are not run history.
Input/metadata files use atomic replacement. Study and definition writes use a staged directory transaction. Rebuilding commits
project metadata only after staging the complete replacement case set. Before that
commit, recovery restores old cases; after it, recovery deletes temporary backups.
Recovery runs before loading case inputs. Project archives transfer saved
definitions and exclude recovery checkpoints and run data.

The application reads the current project format (3) and run-snapshot format (1).
Older project formats are rejected rather than migrated. Project and solver locks
prevent concurrent access. Reopening after an interrupted session marks unfinished
retained runs as Interrupted.

| File | Responsibility |
| --- | --- |
| `packed_bed_ui/window.py` | Welcome screen, project case list, navigation, and actions |
| `packed_bed_ui/case_list.py` | Collapsible study groups, case status, inclusion, and icon actions |
| `packed_bed_ui/editor.py` | Shared input controls, read-only inspection, case autosave, and retained plots |
| `packed_bed_ui/studies.py` | Parameter catalogue, pure expansion, validation, and generation signatures |
| `packed_bed_ui/study_store.py` | Baseline eligibility, persistence/imports, and transactional complete rebuilds |
| `packed_bed_ui/study_editor.py` | Study workspace, variation dialogs, and cancellable previews |
| `packed_bed_ui/study_tables.py` | Preview and explicit-row models, spreadsheet paste, and definition dropdowns |
| `packed_bed_ui/definition_editor.py` | Reusable-definition library and shared inspection dialogs |
| `packed_bed_ui/inputs.py` | Explicit draft initialization, input validation, timing, and zone scaling |
| `packed_bed_ui/general.py` | Numerical/solver settings and report/plot selections |
| `packed_bed_ui/chemistry.py` | Species lists and reaction families |
| `packed_bed_ui/reaction_graph.py` | Asynchronous Graphviz SVG preview, node highlighting, zoom and pan |
| `packed_bed_ui/bed.py` | Geometry, material zones, and reactor boundary rules |
| `packed_bed_ui/program_editor.py` | Program modes, hold/ramp tables, feed targets, timing, and flow conversion |
| `packed_bed_ui/editor_widgets.py` | Shared selection lists, tables, and preview canvases |
| `packed_bed_ui/project.py` | Project/case storage, imports, readiness, and run preparation |
| `packed_bed_ui/execution.py` | Start, monitor, and stop the project's worker with Qt |
| `packed_bed_ui/worker.py` | Verify/activate snapshots, collect logs, publish status, call the shared engine loop |
| `packed_bed_ui/results.py` | Project Results, using the shared Report worksheet controls |
| `packed_bed_ui/project_results.py` | Case-axis selections, native-grid validation and project Excel export |
| `../packed_bed/batch.py` | Shared case scheduling, concurrency, thread limits, and worker cleanup |
| `../packed_bed/preview.py` | Scientific preview arrays without Qt or plotting imports |

Study selectors and the project library share `DefinitionList` and its editing actions.
Both the workspace and storage use `preview_study()`; its lazy mode lets Qt consume
one captured preview incrementally. Shared document access, scientific fingerprint
inputs, timing, and scaling live in `inputs.py`.

`ProjectCase.resolve()` uses the engine's metadata-only `inspect_case()`. Workers load the
snapshots and call `run_case()` with its optional status callback. The UI and
batch CLI both use `run_cases_in_processes()` for concurrent scheduling and
worker cleanup. The CLI retains its in-process path for one worker with no timeout. There is no second scheduler inside a widget.

Each project stores one current copy of each plugin; Save replaces that copy.
Code/resource approval belongs to the local computer and survives parameter edits.
Workers reconstruct authoritative definitions from plugin copies in their input snapshots. See [PLUGINS.md](../PLUGINS.md) for the plugin workflow and API.

## Checks

```sh
python -m pytest desktop/tests -q
python -m pytest tests/test_batch.py tests/test_cli.py tests/test_solver_infrastructure.py -q
python -m pip wheel --no-deps --no-build-isolation ./desktop
```

Tests cover multi-case projects, the mixed 11-case example, portable imports,
invalid drafts, successful-baseline eligibility, complete study rebuilds, reusable
definitions, spreadsheet editing, staleness, transaction recovery, interrupted execution,
project locking, and real DAETools runs compared with direct engine results.
Qt checks run offscreen; real solver checks skip when DAETools is absent.

## Project plugins

Open **Plugins** beside **Project** in the menu bar. To add a standalone plugin,
choose **Register plugin…** and select its `.msplugin` file or `manifest.yaml`.
Select **Enable** for a disabled plugin, then choose its definitions in the case's
**Chemistry** tab. Species and reaction pickers display their sources.

Plugins included in an imported `.msproject` archive are already installed in
that project. Enabled data-only plugins work immediately. For a Python plugin
marked **Needs local approval**, select it, click **Allow code…**, and approve
only if you trust the source. Wait for **Checks passed.**, then run the cases;
their existing plugin selections are preserved. Disabled plugins remain disabled
until you select **Enable**. No separate plugin download or pip install is needed.

Importing and browsing do not execute supplied code. Approval allows Python code
to run during checks and simulations with your user permissions; the worker
processes are **not a security sandbox**. Checks do not establish that a plugin is
safe or scientifically correct. Approval stays on your computer. Changes to code
or resources require approval again; parameter-only edits do not.

See [plugin setup, sharing and safety](../README.md#scientific-plugins) for the
full workflow and [the plugin guide](../PLUGINS.md) for authoring examples.
