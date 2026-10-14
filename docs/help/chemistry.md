# Chemistry tab

Choose the gas species, solid species, and reactions that belong to this case.
Definitions can come from the built-in catalogue or the project's plugins.

## Species and reaction controls

| Field or option | What it does |
| --- | --- |
| Gas species / Solid species | Lists the species definitions included in this case, with their sources. |
| + Add gas species… / + Add solid species… | Opens the picker for that phase. Added species start at zero in compositions or solid loadings. |
| + Add reaction families… | Makes the chosen reaction families available in the case. |
| Reaction checkbox | Includes or excludes that individual reaction from execution. |
| + Species | Adds a family's missing required species, with zero initial fractions/loadings. |
| Bindings… | Opens a selector for each reaction role so you can choose the matching case component. |
| Each role selector in Bindings | Chooses among compatible selected species; useful when several variants describe the same chemical. |

Available species, families, reactions, and binding roles depend on the selected
definitions and installed project plugins. Their names and descriptions appear
in the picker and hover details.

## Reaction graph controls

| Control or symbol | What it does or means |
| --- | --- |
| Click a node | Highlights its immediate neighbours and connections; click again or on empty space to clear. |
| Green / copper / neutral nodes | Gas species / solid species / reactions. |
| Red nodes | Species required by the selected chemistry but missing from the case. |
| Dashed links | Catalysts or rate dependencies. |

## Build the chemistry

1. Add the gas and solid species you need. Check each definition's source when
   several variants describe the same chemical.
2. Use **+ Add reaction families…** to make the required models available.
3. Check the individual reactions that should run. A family's **+ Species**
   action adds missing components required by its reactions.
4. Inspect **Bindings…** when a reaction role could refer to more than one
   selected component.
5. Set gas fractions in **Program** and solid loadings in **Bed**. Newly added
   species start at zero in those inputs.

Selecting a family and selecting its reactions are separate steps. Rates from
overlapping selected models add together, so remove an unwanted model when
replacing it with another.

## Inspect the reaction graph

Click a node to highlight its neighbours and connections. Click it again, or
click empty space, to clear the highlight. Scroll or use **+ / −** to zoom,
drag to pan, and use **Fit** to show the whole graph.

Removing a species also removes its entries from compositions and bed zones.
Review those tabs after changing the species list.

Read more: [Chemistry workflow](../GUI-GUIDE.md#chemistry-species-and-reactions) ·
[Plugin guide](../PLUGINS.md) ·
[Kinetics sources](../../packed_bed/kinetics/KINETICS_SOURCES.md).
