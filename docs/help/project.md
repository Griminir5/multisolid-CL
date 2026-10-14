# Project screen

Use this screen to organise cases, start simulations, and open comparisons.
Each case keeps its inputs and one latest run. Double-click a case to open its
five editing tabs; use **Back to project** to return here.

## Case list and actions

| Field or option | What it does |
| --- | --- |
| Include checkbox | Includes this case in Run all; a study group's checkbox controls its cases together. |
| Inputs | Shows input readiness, separately from execution status. |
| Latest result | Shows the latest execution status, elapsed time when available, and whether results are stale. |
| Duplicate icon | Copies the case's inputs and report layout into a new case without results. |
| Delete icon | Deletes the case and its latest results after confirmation. On a study, deletes its generated cases too. |
| New Case | Creates an incomplete draft with numerical defaults. Complete Chemistry, Bed, Program, and General before running. |
| Import Case | Copies a YAML case into this project. Select `run.yaml` with its referenced files alongside it. |
| New Parameter Study | Opens the study-name and baseline-case dialog. The baseline must have succeeded and still match its current ready inputs. |
| Maximum workers | Maximum number of simulations running at once. Each case's General tab sets its numerical threads separately. |
| Run all included cases | Validates and runs the checked cases; uncheck unfinished cases first. |
| Cancel execution | Stops active workers and cancels the remaining queue. |
| Results… | Selects successful cases for a comparison workbook. Its selection is separate from Run all inclusion. |

## Project and application menus

| Control | What it does |
| --- | --- |
| Create New Project… | Starts a new project in a new folder. |
| Open Existing Project… | Opens an existing project in place. |
| Save Project | Commits current editor changes; normal edits also save automatically. |
| Close Project | Closes the project and returns to the welcome screen after handling pending changes. |
| Open Project Folder | Opens the project directory in the system file manager. |
| Reusable definitions… | Opens the library of saved bed and operating-program configurations. |
| Export project… | Writes a transferable archive of project inputs, definitions, and layouts, without simulation results. |
| Import project archive… | Opens a shared archive in a new project folder. |
| Recent projects | Opens an available recent project. |
| Plugins | Manages the project's species, property, and reaction definitions and their sources. |
| Help | Opens the bundled guides, screen references, search, troubleshooting, About, and licences. F1 opens this screen's reference. |

## Read the two statuses

Input readiness tells you whether the case can run: **Ready**, **Underdefined**,
**Invalid**, **Runtime unavailable**, or **Needs update**. Execution status
describes the latest attempt, such as **Not run**, **Running**, or **Succeeded**.
**Stale** means the retained result belongs to earlier inputs or plugin contents.

## Run and compare

Check the cases to include, then use **Run all included cases**. All included
cases must pass validation before execution starts. **Maximum workers** controls
simultaneous cases; each case has its own numerical thread setting in General.
**Cancel execution** stops the active workers and cancels queued cases.

Starting a run replaces that case's previous results, even if the new attempt
fails. Duplicate a case and run the copy when you want to keep a comparison.

**Results…** opens a case-selection dialog for comparing successful runs and
exporting an Excel workbook. Its selection is separate from Run all inclusion.
The **Project** menu provides saving, folder access, archives, and reusable
definitions; **Plugins** manages the project's scientific definitions.

Read more: [Run simulations](../GUI-GUIDE.md#run-simulations) ·
[Parameter studies](../GUI-GUIDE.md#parameter-studies) ·
[Saving and recovery](../GUI-GUIDE.md#saving-and-recovery).
