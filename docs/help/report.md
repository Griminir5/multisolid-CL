# Report tab

Arrange one case's recorded results into an Excel workbook. A report layout
controls the export; the **General** tab controls what the simulation records.

## Worksheet and column controls

| Field or option | What it does |
| --- | --- |
| Worksheets list | Selects the fixed Case information sheet or a user worksheet to edit. |
| Name | Renames the selected worksheet. |
| Rows represent | Chooses the coordinate that varies down the sheet. Remove its columns before changing the row axis. |
| Row values | Selects which coordinates become rows, using the modes below. |
| Add columns… | Chooses a recorded quantity and its remaining coordinates. |
| Case information | Shows retained inputs, run settings, status, and provenance for the fixed information sheet. |
| Column dictionary | Shows the exact quantity, units, and coordinate selections behind exported columns. |
| Preview table | Displays up to 20 rows and 12 columns; before a run it can show expected coordinates with empty values. |
| Counts and validation message | Shows the full worksheet size or explains why selected data cannot be exported. |
| Save as template… | Saves the layout as JSON, without run data or case identity. |
| Apply template… | Reviews worksheet sizes and compatibility, then replaces the current layout on Apply. Cancel keeps it. |
| Export workbook… | Writes the complete selected data and information sheet to Excel. Export's Cancel stops writing; an existing file is replaced only after success. |

## Coordinate and quantity options

| Field or option | What it does |
| --- | --- |
| Time (s) | Varies the saved reporting time down rows or chooses a time for each value column. |
| Cell position (m) | Selects cell-centre coordinates for cell quantities. |
| Face position (m) | Selects cell-boundary coordinates for fluxes and velocity. |
| Gas species / Solid species / Reaction | Selects the relevant component or reaction axis when the quantity has it. |
| All | Includes all available coordinates. In this case report, All column rules regenerate with input grid changes. |
| First / Last | Chooses the first or last coordinate on that axis; it follows the endpoint when coordinates change. |
| Selected values | Uses an explicit list of coordinates. |
| Choose values… | Opens the coordinate list; Ctrl/Shift selects multiple entries. OK accepts the list; Cancel discards changes. |
| Filter coordinates… | Narrows the coordinate list without clearing selections hidden by the filter. |
| Quantity in Add columns | Chooses the output variable and shows its units. Options depend on requested reports; quantities without the chosen row axis are disabled. |
| Remaining coordinate selectors | Choose each other axis of that quantity, producing one column for each combination. |

## Quantity names

The available quantities depend on the reports selected in General. Names below
use the same words as the picker; each option displays its units.

| Quantity options | What they contain |
| --- | --- |
| Inlet temperature / Temperature / Outlet temperature | Temperature at the inlet, each cell centre, or final cell (K). |
| Inlet pressure / Pressure / Outlet pressure / Pressure drop | Boundary or cell pressure, or inlet minus outlet pressure (Pa). |
| Velocity | Superficial gas velocity at faces (m/s). |
| Gas concentration | Gas species inventory per bed volume (mol/m³). |
| Inlet composition / Gas mole fraction / Outlet composition | Gas mole fractions at the inlet, cells, or outlet; outlet composition uses fluxes with a zero-flow fallback. |
| Solid concentration | Solid species concentration per bed volume (mol/m³ bed). |
| Solid mole fraction | Each solid species' share of the local total solid molar concentration. |
| Inlet flow / Outlet flow | Total inlet or outlet molar flow (mol/s). |
| Gas flux | Species molar flux at faces (mol/(m² s)). |
| Outlet species flow | Outlet molar flow of each gas species (mol/s). |
| Reaction rate | Reaction rate by reaction and cell (mol/(m³ bed s)). |
| Gas enthalpy flux | Species enthalpy flux at faces (J/(m² s)). |
| Heat in total / Heat out total / Heat loss total | Cumulative boundary enthalpy transfer and wall heat loss (J). |
| Heat bed total | Current stored internal energy in the bed (J). |
| Heat balance error | Change in stored energy minus net transferred energy (J). |
| Mass in total / Mass out total | Cumulative boundary mass transfer (kg). |
| Mass bed total | Current mass stored in the bed (kg). |
| Mass balance error | Change in stored mass minus net transferred mass (kg). |

## Inputs and retained data

Cell positions and face positions are different axes. The editing controls
follow the current inputs: All columns regenerates when the grid changes, and
First/Last follows the endpoint. The preview shows the retained run's values
when results exist. Review selections after changing inputs, especially if the
retained results are stale.

You can prepare a layout before a run, but exporting needs the selected recorded
data. Missing quantities or coordinates must be resolved before the affected
worksheet can export. A stale result still describes its retained run, not the
case's newly edited scientific inputs.

A failed run can sometimes provide saved partial samples. These describe saved
reporting times, not necessarily the instant of failure. Use **Open latest run
log** to inspect the execution.

Layouts save automatically and remain editable for study-generated cases.
**Save as template…** and **Apply template…** reuse layouts without copying run
data. To compare several cases together, return to the project and open
**Results…**.

Read more: [Export a case to Excel](../GUI-GUIDE.md#export-a-case-to-excel) ·
[Project Results](results.md).
