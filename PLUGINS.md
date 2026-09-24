# Scientific plugins

Open a project and choose **Plugins** beside **Project** in the menu bar. Built-in definitions are always available. **Register plugin** imports a `.msplugin` archive or a folder selected through its `manifest.yaml` into the project; **Export** shares one plugin independently. Copying the project folder carries its plugins with it.

Enabling a plugin makes its definitions available in the Chemistry pickers. It does not change a case's component list. The pickers, graphs and recorded reports show each definition's source. Bed and Program labels, including their preview legends, show only the formula.

## Species and parameter variants

Choose **Create plugin**, enter its name, then use **Add species** for each species you want in it. Saving returns to the same plugin and selects the new species. IDs are generated automatically. **Register plugin** imports an existing manifest or archive; **Export** shares the selected plugin.

Select a built-in species or reaction and choose **Create variant** to make an independent plugin copy. Name the plugin, edit the values and save. A species variant can change its name, molecular weight, polynomial or Shomate heat capacity/enthalpy and quadratic gas viscosity. Custom enthalpy correlations declared in the same plugin also appear in the species editor, with their declared parameters and units. Double-click a definition in your plugin to edit it; double-click a built-in definition to create a variant.

**Save** checks the values and saves the plugin. Invalid values stay in the open editor with an explanation. **Cancel**, Escape and closing the editor discard changes without creating files or drafts. Saving unchanged values does nothing.

All property values use SI. Polynomial Cp coefficients multiply successive powers of `T − Tref`: three coefficients specify degree-two Cp and degree-three enthalpy. Reference enthalpy is in J/mol; molecular weight is in kg/mol. Shomate coefficients use `τ = T / (1000 K)`, with an explicitly supplied reference enthalpy.

Reaction forms expose only the implementation's declared parameters. For example, Medrano exposes `R0_M`, `K0_M_PER_S` and activation energies, while reaction orders and fitted approximations remain implementation constants. Reset restores implementation defaults. Prefactors retain the existing implementations' numerical values; their displayed basis explains pressure/concentration normalization. No edits mutate the built-in module globals.

To use both original water and altered water, add both in Chemistry. They have separate component IDs, compositions and balance equations. When you add families or change species, the editor chooses matching available components and saves those bindings. **Bindings** lets you choose a different available component for each role. **+ Species** adds missing components; compositions and loadings still need values. Add a different reaction variant through the family picker. Overlapping reactions have additive rates and produce an applicability caution.

In the plugin browser, selecting a family shows its complete definition; selecting a reaction shows only that reaction. The tree groups definitions under their plugin and reactions under their family. Plugin actions are above the tree; definition actions are beside the details. Enablement checks run in the browser without opening another window; failures appear there with details, and **Cancel check** stops a pending check.

Reaction details include declared editable parameters, current values, defaults, units and bounds. Built-in reactions and their parameter variants show the parameters used by that reaction; shared parameters are marked. External families without per-reaction grouping show their family parameter list with an explicit label. Choose **Edit** on the selected reaction to change its parameters; values belonging only to other reactions are preserved. Select the family to edit all its parameters. **Create variant** makes a separate copy.

To switch existing ordinary cases, select the species or reaction family you want to use and choose **Use this as replacement…**. Then select the existing definition to replace; the list contains only compatible definitions currently used in the project. The confirmation explicitly names the old and new definitions and lists the affected cases and reusable definitions. Applying it preserves component/instance IDs, compositions and bindings. A replacement species must preserve chemical key and phase; a replacement mechanism must preserve reaction IDs and species roles. This action leaves study selections and saved runs unchanged. Editing the contents of a plugin already selected by a study does affect its future runs.

## Saving and execution

There is one current copy of each project plugin, stored at `plugins/<plugin-id>/current/`. Saving replaces that copy atomically. There are no revision choices or version increments. Registering a plugin with the same plugin ID asks before replacing the project's copy. Creating a variant gives it a separate identity.

Cases and studies use the current contents of their selected plugins. Editing a selected plugin marks old results stale; no new baseline or study rebuild is required just for a plugin edit. Unrelated plugins do not affect results. Removing or disabling a plugin still used by inputs is blocked with a list of its users.

Preparing a simulation copies its selected plugins alongside its input snapshot. Later plugin edits or removal cannot change that prepared run or its results. CLI batches also carry their plugin files with their generated inputs. Content hashes are internal integrity checks, not user-managed revisions.

Python plugins require local approval before execution. Approval covers the code and included resources, so editing parameters does not ask again; changing code or resources does. A copied project's enabled flag cannot grant approval on another computer. Browsing never imports plugin code. Checking runs inline in the browser or editor, with errors shown there and a **Cancel check** button. Python checks and simulations run in separate processes, which are **not a security sandbox**.

## Python authoring

Copy one of the example folders from `packed_bed/examples/plugins/`:

- `ammonia`: NH3, a custom enthalpy correlation and N2 + 3 H2 → 2 NH3 kinetics.
- `nitrogen_oxides`: NO, NO2 and a built-in nitrogen parameter variant, without Python.
- `enthalpy_overrides`: CO2 and H2O variants using one custom enthalpy correlation.

See the [example walkthrough](packed_bed/examples/plugins/README.md) for registration, replacement, formulas and archive creation. Each plugin works independently.

The [Waste iron plugin](plugins/waste_iron/README.md) transfers the fitted oxidation/reduction kinetics and SiO2/Al2O3 property definitions from `waste_iron_kinetics` at `9660e5d`. Its folder can be registered directly or exported as an independent archive.

The example coefficients demonstrate the API; they are not validated scientific datasets. Work on your own copy of the folder. For Python development, regenerate catalogue metadata from the implementation with:

```python
from packed_bed.plugins.check import generate_metadata
from packed_bed.plugins.storage import pack_plugin

generate_metadata("my-plugin")  # Executes your local Python factories.
pack_plugin("my-plugin", "my-plugin.msplugin")
```

Register the folder’s `manifest.yaml` directly, or register the archive, then use **Check** or **Enable**. The GUI browses the manifest without importing Python code. Metadata generation fills reaction and parameter declarations from Python and omits unused defaults. A disagreement between the manifest and Python declarations blocks execution. There are no public `extension check/pack` commands.

A plugin contains `manifest.yaml`, Python modules and any local resources. Use relative imports between your modules, and locate included resources relative to `__file__`. Use only Python's standard library and the application's bundled libraries (`numpy`, `scipy`, `yaml`, `pydantic`, `xarray`, `matplotlib`, `daetools`, `pyUnits`, and the public `packed_bed.plugins` API). There are no plugin dependencies, pip installs, downloads or extra runtimes. Include any custom species/correlations needed by your mechanisms. Cloning a custom plugin copies the entire package under a new identity, preserving attribution without a live dependency on the original.

The public API exports `ParameterSpec`, `BaseCorrelation`, the built-in correlation classes, `ReactionDefinition`, `ReactionFamily` and `KineticsContext`. An implementation reference such as `kinetics:create` names a local module and factory. The module declares:

```python
PARAMETERS = {
    "k0": ParameterSpec(1e-5, "m^3/(mol*s)", "Rate coefficient", minimum=0),
}

def create(parameters):
    # Return a ReactionFamily or BaseCorrelation implementation.
    ...
```

Parameters are immutable. A rate hook receives the same values through `context.parameters`; use `context.gas_index(role)`, `context.solid_index(role)` and `context.molecular_weight(role)` to access the selected component. Never capture molecular weights from a global property registry. Declare every species accessed by rate helpers in the reaction's `required_species`, including non-stoichiometric dependencies. Return symbolic rates in mol/(m³ bed·s); account for the concentration basis when using the model state.

Enthalpy correlations implement `value(T)`, `dae_expression(T)`, `cp_value(T)` and `cp_dae_expression(T)`. Viscosity correlations implement the first two methods. Numerical methods accept scalar and array temperatures. Package checks use the normal property and reaction constructors and verify declared parameters and interfaces. The additional numerical/symbolic self-test harness and in-app property previews are omitted. Authors must verify scientific validity and agreement between numerical and symbolic implementations.

Current project and snapshot formats are supported. Obsolete project migrations, saved plugin drafts and hash-named plugin revisions are not supported.

## Input format and engine API

Existing built-in inputs retain their identifiers. Optional mappings in `chemistry.yaml` make custom choices explicit:

```yaml
gas_species: [N2, H2, NH3]
species_definitions:
  NH3: example_ammonia:NH3
reaction_families: [ammonia]
mechanisms:
  ammonia:
    definition: example_ammonia:synthesis
reaction_ids: [ammonia/synthesis]
```

The program composition maps use these component IDs. Add explicit `bindings` in a mechanism selection when a role should use a different component ID. Solid selections remain in `solids.yaml`; their optional definition references use the same `species_definitions` map. Existing built-in mechanism instance names retain their legacy reaction IDs even after project replacement. Other instances use `<instance>/<local-reaction-id>`.

`load_case` discovers a containing project and uses its current plugins; run snapshots use their saved copies. Standalone inputs can put a `definitions.json` next to `run.yaml` with `{"root": "relative/path/to/project", "lock": {"builtin": "<fingerprint>", "plugins": {"lab": "<hash>"}}}`. CLI batches copy the selected plugins into each generated case folder and write this descriptor with `root: "."`, so the case folder can be moved independently. For manually prepared descriptors, the referenced plugin folders must travel with the inputs; there is no machine-global plugin dependency.

For Python callers, `select_definitions(chemistry, solids, catalogue)` returns immutable metadata. `materialize_definitions(selection, catalogue)` returns the passive `DefinitionEnvironment` containing exactly `properties`, `reaction_network`, `rate_hooks` and `selection`. Pass it to `load_case(..., definitions=environment)` or `resolve_case(..., definitions=environment)`. `Case.definitions` is authoritative for validation, initialization and simulation. Independent property-registry/model-hook overrides have been removed; a supplied environment never fills missing definitions from global built-ins. `inspect_case` and `inspect_case_file` return metadata-only `CaseInputs` for editors, not partially executable cases.
