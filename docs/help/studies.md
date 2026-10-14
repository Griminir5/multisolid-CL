# Parameter study screen

Define variations of a saved baseline, review the resulting cases, and create
them for execution from the project screen.

## Study and baseline controls

| Field or option | What it does |
| --- | --- |
| Study name | Renames this study without requiring a rebuild. |
| Baseline label | Identifies the fixed set of inputs from which variations are made. |
| Inspect baseline | Opens those saved inputs read-only. |
| Replace baseline | Selects a different successful, unchanged, ready case as the fixed baseline. |
| Add baseline as independent case | When offered for an imported study, copies its saved baseline into a separate case. |
| All combinations | Generates every combination of the selected variation values. |
| Explicit case rows | Uses the particular combinations entered in the row table. |
| Reusable definitions… | Opens the saved bed and operating-program library. |

## Variation fields and dialogs

| Field or option | What it does |
| --- | --- |
| Parameter / Values table | Lists the varied parameters and their values; double-click a row to edit it. |
| + Add variation | Opens the parameter picker, including reusable program and bed replacements. |
| Parameter entry | Chooses the input to vary; its baseline value and units identify the field. Unavailable entries explain why. |
| Edit variation | Edits the selected parameter's values or definition choices. |
| Values | Enters an explicit list, separated by commas, spaces, or new lines. |
| Range | Generates a linearly spaced set from Start to End. |
| Start / End | First and last range values in the parameter's units. |
| Number of values | Number of values in the range. |
| Expanded values | Read-only preview of the resulting list, or a message explaining an unfinished rule. |
| Reusable-definition checkboxes | Selects the saved program or bed definitions used for a replacement variation. |

## Explicit rows and preview

| Field or option | What it does |
| --- | --- |
| Explicit-row numeric cell | Enters that case's value for the corresponding variation. Blank cells remain unfinished. |
| Explicit-row definition selector | Selects a reusable program or bed for that case. |
| Candidate table | Previews the cases produced by the rules; selecting a row shows its validation details. |
| Inspect selected case | Opens that candidate's inputs, bed, and program for inspection. |
| Refresh preview | Recomputes the candidate list from the current rules. |
| Case count / replacement note | Shows how many cases will be created and what an existing study rebuild will replace. |
| Create N cases | Creates the displayed cases without running them. |
| Replace X cases with Y | After confirmation, deletes the study's old generated cases and results and creates fresh cases. |

## Set the variations

The baseline is a fixed copy of a successful case's inputs. **Inspect baseline**
opens it read-only; **Replace baseline** selects another eligible successful
case. Later edits to the source case do not change the saved baseline.

Use **+ Add variation** to choose a numerical parameter, reusable program, or
reusable bed. Numerical variations accept explicit values or a spaced range.
**All combinations** combines every variation value; **Explicit case rows**
lets you enter particular pairings.

Inspect the live case preview, including the bed and program. Complete unfinished
variation rules before generation. Invalid scientific inputs can be generated
as drafts, but must be resolved before execution.

## Create and rebuild cases

**Create N cases** creates the inputs without running them. Return to the project
screen to run the generated cases. Their scientific inputs are read-only; edit
the study or duplicate a generated case as an independent case. Report layouts
remain editable.

Changing study rules marks its generated cases **Needs update**. Rebuilding
replaces generated cases and their results after confirmation. Keep any run you
need by exporting its results before rebuilding.

Read more: [Parameter studies](../GUI-GUIDE.md#parameter-studies) ·
[Rebuilding a study](../GUI-GUIDE.md#rebuilding-a-study).
