# Project Results screen

Compare successful case runs and export their recorded values together in an
Excel workbook. The selected cases can include independent cases and study cases.

## Case selection and navigation

| Field or option | What it does |
| --- | --- |
| Select cases… | Opens the selector for successful retained runs. |
| Latest run | Shows execution status; cases without readable successful results cannot be selected. |
| Select all / Clear all | Changes every eligible case, including cases hidden by the filter. |
| Refresh results | Reloads retained results after runs or files change. |

## Workbook fields and options

| Field or option | What it does |
| --- | --- |
| Worksheets list | Selects the fixed Results information sheet or a user worksheet. |
| Name | Renames the selected worksheet. |
| Rows represent | Selects the axis down rows: Time, Cell position, Face position, Gas species, Solid species, Reaction, or Case as applicable. Remove columns before changing it. |
| Row values — All | Includes the available shared row coordinates. |
| Row values — First / Last | Uses the first or last row coordinate; for case-specific selections, endpoints resolve separately for each run. |
| Row values — Selected values | Uses explicitly chosen row coordinates. |
| Choose values… / Filter coordinates… | Selects particular coordinates with Ctrl/Shift and filters the list. OK accepts; Cancel discards changes. |
| Add columns… — Quantity | Chooses a recorded output and shows its units. The [quantity list](report.md#quantity-names) explains each built-in option. |
| Case in Add columns | Chooses the cases represented by value columns when Case is not the row axis. |
| Other coordinate selectors | Choose times, cell/face positions, species, or reactions as required by the quantity. All/First/Last/Selected values work as above. |
| Split by case | Replaces the current sheet with one editable sheet per referenced case, preserving each recorded grid. |
| Results information | Shows run identities, study selections, retained inputs, and available balance diagnostics. |
| Column dictionary | Shows quantities, units, and exact per-case coordinates behind the exported columns. |
| Preview table | Shows up to 20 rows and 12 columns from the selected data. |
| Counts / validation message | Shows full sheet size or explains an incompatible or unavailable selection. |
| Save as template… | Saves a reusable layout without run data or case identities. |
| Apply template… | Previews sheets and maps template case positions to the selected cases in displayed order. Apply replaces the layout; Cancel keeps it. |
| Export workbook… | Writes the full workbook and Results information sheet. Cancel stops export; existing files are replaced only after success. |

## Select cases and arrange sheets

**Select cases…** changes the cases available to the workbook. This selection
is independent of the project screen's Run all checkboxes. Existing columns keep
their case identities; add columns to include newly selected cases.

Choose a row axis, then use **Add columns…**:

- **Time** compares histories at selected positions or for outlet quantities.
- **Position** compares profiles at selected recorded times.
- **Case** compares values across cases. First/Last resolves separately for each
  run; it does not calculate an average.

## Compare compatible samples

Shared rows require exactly shared recorded coordinates. If cases use different
grids or reporting times, select common coordinates or use **Split by case** to
make separate editable worksheets. Export does not interpolate between samples.
Cell positions and face positions remain separate axes.

Use **Refresh results** after runs change. Missing cases or coordinates stay
visible for repair. Successful stale results can be exported when their retained
data and snapshot remain readable. To export partial data from a failed run, use
that case's **Report** tab instead.

## Export and reuse layouts

The preview shows up to 20 rows and 12 columns. **Export workbook…** exports the
full selection, with a **Results information** sheet identifying the runs and
coordinates behind the values. Layouts save automatically.

**Save as template…** saves the layout without data or case identities.
**Apply template…** previews compatibility and maps template case positions to
your selected cases in their displayed order.

Read more: [Comparing cases](../GUI-GUIDE.md#compare-cases-in-project-results).
