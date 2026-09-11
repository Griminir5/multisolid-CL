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

## Cases from a batch

**Import Parameter Study** expands an existing engine `batch.yaml` into this
project's case list. It copies the base inputs, referenced programs and solid
configurations, and the batch rule into `studies/`, keeping references portable.
No simulations run during import, and the source files remain untouched. Study
cases appear under a collapsible group with a summary of their status. The group
checkbox includes or excludes all its cases. Collapsing it preserves selection.

A batch specifying three programs and three solid configurations produces nine
cases. Alongside independently authored Benchmark and Exploratory cases, the
project has 11 cases that can be run together. The saved rule and reusable input
files do not add extra simulations.

The creation/import buttons sit below the case list. **New Parameter Study** is
disabled: its builder is deferred for separate design. Reusable input editors and
regeneration are also pending. Imported generated cases are
editable materialized copies with origin metadata; changes to those copies do
not change the retained batch rule. Importing another batch adds new cases.
Batch worker/time-limit settings are retained in the copied rule; the starter's
Run all uses the project’s Maximum workers setting and has no timeout control.

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
  studies/<study-id>/             # imported batch rule and referenced inputs
  execution.json                 # latest Run case / Run all queue status
  execution.log                  # worker startup diagnostics
```

Prepared snapshots briefly live in `.pending-<attempt>` beside a case's `run/`.
Once validated, a snapshot replaces that case's run folder; `.previous-run` is a
short-lived backup during the directory swap. Recovery completes an interrupted
swap. Attempt identities distinguish worker events; they are not run history.
Input/metadata files use atomic replacement. Full multi-file draft recovery and
project archives remain pending.

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
| `packed_bed_ui/editor.py` | Five-tab editor, autosave, numerical previews, and retained plot windows |
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
invalid drafts, staleness, replacement, migration, interrupted execution,
project locking, and real DAETools runs compared with direct engine results.
Qt checks run offscreen; real solver checks skip when DAETools is absent.
