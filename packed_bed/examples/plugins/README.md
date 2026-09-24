# Example scientific plugins

Each folder is a complete, independent plugin. Only a normal MultiSolid installation
is needed. The numbers and the ammonia rate law are illustrative, not validated
property datasets or an industrial ammonia synthesis model.

| Plugin | Contents |
| --- | --- |
| [ammonia](ammonia/manifest.yaml) | NH3, a new inverse-square Cp/enthalpy correlation, and N2 + 3 H2 → 2 NH3 kinetics. |
| [nitrogen_oxides](nitrogen_oxides/manifest.yaml) | NO and NO2, plus changed nitrogen Cp coefficients. Data only: no Python code. |
| [enthalpy_overrides](enthalpy_overrides/manifest.yaml) | One new logarithmic enthalpy correlation, used by CO2 and H2O variants. |

## Try them in the application

1. Open a project, choose **Plugins → Register plugin**, and select a folder's
   `manifest.yaml` (or its `.msplugin` archive). Registration copies the whole
   plugin into the project; later edits to the source folder do not change that copy.
2. For the two Python plugins, select the plugin and **Enable**, then allow its
   code. The nitrogen oxides plugin is enabled immediately.
3. In the case's Chemistry tab, add the desired species from the plugin's source.
   For ammonia, add built-in N2 and H2, plugin NH3 and the **Ammonia synthesis**
   family. Select its reaction and set a feed, for example 0.25 N2 / 0.75 H2.
   Keep an inert solid bed and use a temperature in 298.15–1200 K for this example.

To apply an override to existing cases, select the plugin's **Nitrogen**,
**Carbon Dioxide** or **Water** entry and choose **Use this as replacement…**.
Select the existing definition it should replace (for example, its built-in
counterpart) and review the affected cases. Apply this once for nitrogen, or once
each for CO2 and H2O. Component IDs and compositions stay
the same. Registration and enablement alone never change case selections.
You can instead add both definitions as distinct components in one case.

Double-click a plugin species to edit its values. Its custom enthalpy correlation
and declared parameters are available in the species editor. **Add species** also
offers the correlations already in that plugin. Changes affect only that species;
CO2 and H2O share code but have separate coefficients. Select the ammonia reaction
and **Edit** to change its declared rate coefficient `k`.

## What the examples implement

All temperatures are in K, molar enthalpies in J/mol, Cp in J/(mol·K), molecular
weights in kg/mol and viscosities in Pa·s. Both custom correlations integrate Cp
exactly and anchor enthalpy at `h_ref` at `t_ref`.

- Ammonia: `Cp(T) = cp_inf − deficit·(t_ref/T)²`. See
  [enthalpy.py](ammonia/enthalpy.py) for numerical and symbolic forms.
- Ammonia reaction: `r = k·c_N2·c_H2`, with concentrations per **total bed volume**
  and `r` in mol/(m³ bed·s). The source terms are `−r`, `−3r`, `+2r`. Negative
  trial concentrations are clamped to zero. The rate is first order in each
  reactant; stoichiometric coefficients do not specify reaction orders. See
  [kinetics.py](ammonia/kinetics.py). This deliberately simple, irreversible rate
  has no catalyst, Arrhenius law or equilibrium model.
- Nitrogen oxides: built-in polynomial Cp and quadratic viscosity, with all
  coefficients in [manifest.yaml](nitrogen_oxides/manifest.yaml). The N2 variant
  changes just the Cp coefficients to `[30, 0.004, 0.000001]`. **Quote `'NO'` in
  YAML**, including dictionary keys: an unquoted `NO` is interpreted as a boolean.
- CO2 and H2O: `Cp(T) = cp_inf − deficit·t_ref/T`, whose integral contains
  `log(T/t_ref)`. See [correlation.py](enthalpy_overrides/correlation.py).
  Their molecular weights, viscosities and reference enthalpies are copied from
  the built-ins; their Cp curves are illustrative replacements.

## Modify and share

Work on a copy of a plugin folder. Edit species and coefficients in `manifest.yaml`;
edit advanced correlations or kinetics in its Python files. Module-level
`PARAMETERS` declares the editable values, units and bounds. Python and YAML use
the same factory contract; no registration decorators or dependencies are needed.

After changing Python declarations, regenerate their browsing metadata. This is
an explicit authoring operation that executes your local factories. It fills in
reaction roles, stoichiometry and parameter declarations, so these need not be
maintained by hand. Register the updated manifest in the app to replace its copy.

To regenerate and package all three examples, run this from the repository root
using the application's Python environment:

```python
from pathlib import Path
from packed_bed.plugins.check import generate_metadata
from packed_bed.plugins.storage import pack_plugin

for name in ("ammonia", "nitrogen_oxides", "enthalpy_overrides"):
    folder = Path("packed_bed/examples/plugins") / name
    generate_metadata(folder)
    pack_plugin(folder, Path("dist/example-plugins") / f"{name}.msplugin")
```

Alternatively, **Export** in the plugin browser shares the current project copy.
Each archive contains everything its plugin needs; the examples do not depend
on each other. The application's **Check** checks implementation contracts and
parameters, not the scientific accuracy of the supplied data.
