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

## Use the starter

1. At launch, choose **Create new project** or **Open existing project**.
   Creating a project creates a new folder and an empty case list. Opening one
   resumes it in place.
2. Inside the project, use **New Case** to start from an example or an empty
   draft. **Import Case** copies a `run.yaml` and its referenced inputs
   into this project, preserving the original files.
3. Project actions are in the top **Project** menu. Each case row shows its name,
   input readiness, latest result, and icon actions: **Run**, **Duplicate**, **Edit**,
   **Delete**. Hover for tooltips. Double-click a case to edit it; press F2 on its
   name to rename it. **← Back to project** returns to the list. Delete asks before removing
   the case and its latest results.
4. The case editor has five tabs: **General**, **Chemistry**, **Bed**, **Program**,
   and **Results**. General contains numerical and solver settings and requested
   reports/plots. Chemistry selects species and collapsible reaction families
   beside a fitted reaction graph. Bed edits geometry and material zones above
   the numerical preview. Program edits initial values and timed hold/ramp steps
   beside its preview. Drafts save after a short pause, including incomplete
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
   including stale results. The Results tab is a placeholder pending its design.

A rerun replaces the previous snapshot, logs, status, and outputs when that
case starts, even if the new attempt fails or is cancelled. Invalid preflight
and cases cancelled before starting keep their previous results. **Duplicate
case** first if you want to retain a separate comparison; the copy starts with
no results.

The Bed and Program previews use the engine's actual smoothed programs, feed mixing,
repetition, GHSV conversion, and sampled solid profiles. Plot toolbars support
zoom, pan, and image saving. Advanced scientific settings survive import/save;
an unsupported solver blocks execution rather than being silently replaced.
The backend dropdown disables Compiled, and the solver dropdown identifies
choices unavailable in the current desktop runtime. Advanced solver settings
open in a separate dialog.

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

## Parameter studies

Run the intended baseline case successfully first. **New Parameter Study** accepts
only a case whose latest run succeeded, whose current inputs are ready, and whose
successful snapshot still matches those inputs. The study copies those inputs as
a fixed baseline, records the successful attempt, and does not copy results.
**Inspect baseline** is read-only; **Replace baseline** selects another successful
case. Subsequent changes to the source case do not change the saved baseline.

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
Recovery runs before loading case inputs. General case-draft recovery and project
archives remain pending.

The project format is 3; the run-snapshot format remains 1. Format-2 projects retain
all existing cases/results and gain study metadata, with a `project-v2.json` backup.
Legacy studies require successful-baseline verification before their next generation.
Their materialized cases remain usable until a study edit requires rebuilding.

Original format-1 starter projects migrate on opening: their inputs become one
independent case, and their latest run is copied into its run slot. The original
`inputs/`, `runs/`, and a `project-v1.json` copy remain as a one-time migration
backup. They are not shown as run history or modified by later reruns. The
application uses project and solver locks to prevent concurrent access. Reopening
after an interrupted session marks unfinished retained runs as Interrupted.

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
| `packed_bed_ui/chemistry.py` | Species lists, reaction families, and a native Qt network graph |
| `packed_bed_ui/bed.py` | Geometry, material zones, and reactor boundary rules |
| `packed_bed_ui/program_editor.py` | Program modes, hold/ramp tables, feed targets, timing, and flow conversion |
| `packed_bed_ui/editor_widgets.py` | Shared selection lists, tables, and preview canvases |
| `packed_bed_ui/project.py` | Project/case storage, imports, migration, readiness, and run preparation |
| `packed_bed_ui/execution.py` | Start, monitor, and stop the project's worker with Qt |
| `packed_bed_ui/worker.py` | Verify/activate snapshots, collect logs, publish status, call the shared engine loop |
| `../packed_bed/batch.py` | Shared case scheduling, concurrency, thread limits, and worker cleanup |
| `../packed_bed/preview.py` | Scientific preview arrays without Qt or plotting imports |

Study selectors and the project library share `DefinitionList` and its editing actions.
Both the workspace and storage use `preview_study()`; its lazy mode lets Qt consume
one captured preview incrementally. Shared document access, scientific fingerprint
inputs, timing, and scaling live in `inputs.py`.

`ProjectCase.resolve()` uses the engine's `resolve_case()`. Workers load the
snapshots and call `run_case()` with its optional status callback. The UI and
batch CLI both use `run_cases_in_processes()` for concurrent scheduling and
worker cleanup. The CLI retains its in-process path for one worker with no timeout. There is no second scheduler inside a widget.

Extension requirements belong to the project. Code extensions are not loaded by
this starter; a project requiring them cannot run yet. See [PLAN.md](../PLAN.md)
for the full definition/extension design and remaining release work.

## Checks

```sh
python -m pytest desktop/tests -q
python -m pytest tests/test_batch.py tests/test_cli.py tests/test_solver_infrastructure.py -q
python -m pip wheel --no-deps --no-build-isolation ./desktop
```

Tests cover multi-case projects, the mixed 11-case example, portable imports,
invalid drafts, successful-baseline eligibility, complete study rebuilds, reusable
definitions, spreadsheet editing, staleness, transaction recovery, migration, interrupted execution,
project locking, and real DAETools runs compared with direct engine results.
Qt checks run offscreen; real solver checks skip when DAETools is absent.
