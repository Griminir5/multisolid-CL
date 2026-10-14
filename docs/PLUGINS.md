# Plugin guide

Plugins add species, property equations, and reaction models to MultiSolid. Start
with the smallest change you need: changing coefficients usually needs no Python;
writing a new equation does.

| What you want to do | Start here |
| --- | --- |
| Change an existing species or reaction parameter | [Create a variant in the application](#create-a-variant-in-the-application) |
| Define a new species using the existing property equations | [Create a species plugin](#create-a-species-plugin) |
| Write a new property equation or reaction rate | [Develop-a-plugin walkthrough](../examples/plugins/README.md#lesson-2-write-a-new-property-equation) |
| Understand the files and Python API | [Authoring reference](#authoring-reference) |
| Use a plugin someone sent you | [Install and use a plugin](#install-and-use-a-plugin) |
| Send your plugin to someone else | [Share and update plugins](#share-and-update-plugins) |
| Fix a problem | [Troubleshooting](#troubleshooting) |

The [example walkthrough](../examples/plugins/README.md) builds a complete plugin
one file at a time. It starts with a species in YAML, then adds a Python property
equation and a reaction. The examples use illustrative coefficients and kinetics;
they are teaching material, not validated scientific models.

## Choose what to define

A plugin is a folder containing `manifest.yaml` and, when needed, Python files.
The manifest gives the plugin a name and describes what it provides. An exported
`.msplugin` file packages that folder for sharing.

| Part | Meaning | Example |
| --- | --- | --- |
| Species | A chemical identity, phase, molecular weight, and property definitions | NH3 gas |
| Correlation | An equation that calculates a property from temperature | Molar enthalpy and heat capacity |
| Reaction family | One or more reactions, with their stoichiometry and rate equations | Ammonia synthesis |
| Parameter | A value that users can change without editing an equation | A rate coefficient `k` |

**You do not need to write Python for every species.** Polynomial or Shomate heat
capacity and quadratic gas viscosity are already available. Use the application
forms or a manifest to supply their coefficients. Use Python when those equations
cannot express your property model, or when you need a new rate law.

## Create a variant in the application

This is the quickest way to try different coefficients for an existing model.
You need only the desktop application and an open project.

For example, to give nitrogen a different heat capacity:

1. Open **Plugins**, beside **Project** in the menu bar.
2. Expand the built-in definitions and select **Nitrogen**. Use the filter to
   find it if necessary.
3. Choose **Create variant…** and enter a recognizable **New plugin name**, such
   as `My nitrogen properties`.
4. Choose **Polynomial Cp** under **Heat capacity / enthalpy**. Set **Tref (K)**
   to `298.15`, **H at Tref (J/mol)** to `0`, and **Cp coefficients a0, a1, …**
   to `30, 0.004, 0.000001`. These are illustrative values. Keep the chemical
   key `N2` and phase `gas` to make this a nitrogen replacement.
5. Save. The application checks the definition and creates an enabled plugin.
6. Select the new plugin's **Nitrogen** definition and choose
   **Use this as replacement…**. Choose the existing nitrogen definition and
   review the affected cases before applying the change.

The last step connects your definition to cases that already use nitrogen.
If the project has no such cases, add the new definition in a case's **Chemistry**
tab instead. Creating or enabling a plugin alone does not change any case.

The same workflow works for reaction parameters: select a built-in reaction or
family, choose **Create variant…**, edit its exposed values, and save. Select the
family entry when replacing an existing family in cases. A reaction's editor
exposes only its declared parameters; changing stoichiometry or adding a new rate
equation requires Python authoring.

Double-click a definition in your own plugin to edit it. Double-click a built-in
definition to create a variant. Selecting a family shows all its parameters;
selecting a reaction shows its parameter group when one is available. Shared
parameters are marked. **Reset shown values** restores implementation defaults.

**Save** checks and commits the edit. Invalid values remain in the editor with
an explanation. **Cancel**, Escape, or closing the editor discards unsaved edits.
Saving unchanged values does not create another copy.

## Create a species plugin

For a new species, choose **Plugins → Create plugin…**, name it, then select the
plugin and choose **Add species…**. The application generates the internal IDs.
Fill in these fields:

| Field | What to enter |
| --- | --- |
| Name | A readable label, such as `Ammonia` |
| Chemical key | The chemical identity used to match reaction roles, such as `NH3` |
| Phase | `gas` or `solid` |
| Molecular weight | **kg/mol**, so 17.03 g/mol becomes 0.01703 kg/mol |
| Minimum / maximum T | The temperature interval in K for which your definition is intended |
| Heat capacity / enthalpy | An equation type, its coefficients, and reference enthalpy |
| Gas viscosity | Quadratic coefficients in the displayed SI units; required for gas species |
| Scientific source / Notes | Data source, fit assumptions, and limitations |

Save, then add another species if needed. A plugin may contain several species.
Custom enthalpy correlations already declared in that plugin also appear in the
species form. Their parameters are editable there. Custom viscosity code and its
configuration are edited in the source package.

Prefer text files? [Lesson 1](../examples/plugins/README.md#lesson-1-create-a-plugin-with-one-file)
provides a complete `manifest.yaml` to copy. There is no Python file, build step,
or metadata-generation step for that example.

### Property units and coefficient conventions

Temperatures are in K, molecular weights in kg/mol, molar enthalpies in J/mol,
heat capacities in J/(mol·K), and viscosities in Pa·s.

| Built-in equation | Manifest model | Parameters |
| --- | --- | --- |
| Polynomial heat capacity | `builtin:polynomial` | `coefficients`, `t_ref`, `h_form_ref` |
| Shomate heat capacity | `builtin:shomate` | `a0` through `a4`, `t_ref`, `h_form_ref` |
| Quadratic gas viscosity | `builtin:quadratic_viscosity` | `a0`, `a1`, `a2`, `t_ref` |

For polynomial Cp, with `ΔT = T − t_ref`:

```text
Cp(T) = a0 + a1 ΔT + a2 ΔT² + …
h(T)  = h_form_ref + a0 ΔT + (a1/2) ΔT² + (a2/3) ΔT³ + …
```

Coefficients are in ascending order. Coefficient `ai` has units
J/(mol·K^(i+1)); three coefficients give quadratic Cp. These are powers of
`T − t_ref`, so coefficients fitted to powers of `T` need conversion first.

For Shomate Cp, `τ = T / (1000 K)` and
`Cp = a0 + a1 τ + a2 τ² + a3 τ³ + a4/τ²`. Coefficients use J/(mol·K).
MultiSolid integrates this expression and anchors enthalpy at `h_form_ref` at
`t_ref`; do not paste a different source's full enthalpy coefficient set into
these fields.

For quadratic viscosity, `μ = a0 + a1 ΔT + a2 ΔT²`. The coefficients have units
Pa·s, Pa·s/K, and Pa·s/K² respectively. Its reference temperature may differ from
the enthalpy reference temperature.

Use consistent enthalpy references across reacting species. See the
[model equations](EQUATIONS.md) for how properties and reaction source terms enter
the balances.

## Install and use a plugin

1. Open the destination project and choose **Plugins → Register plugin…**.
2. Select the supplied `.msplugin` archive, or select `manifest.yaml` inside the
   plugin folder. Keep its Python files and resources beside it.
3. A plugin using only built-in equations is checked and enabled immediately.
   For a Python plugin, select it and choose **Enable**, then allow its code if
   you trust its source. Wait for **Checks passed.**
4. In your case's **Chemistry** tab, add the plugin's species or reaction family
   from the appropriate picker. Select the reactions you intend to run.
5. Check the family **Bindings…** and supply the corresponding gas compositions
   in **Program** and solid loadings in **Bed**. **+ Species** can add missing
   components; it does not supply their compositions or loadings.

Registration copies the folder into the project. Later edits to the original
folder do not update the project copy; register it again after editing.
Normal desktop use does not require a separate Python installation or a pip
command for the plugin.

### Replace a definition or add a separate component

Use **Use this as replacement…** when you want an existing case component to use
new properties. Select the **new** species or family in Plugins, then choose the
existing definition it should replace. Review the listed ordinary cases and
reusable definitions. Component IDs, family instance IDs, compositions, and role
bindings are preserved.

A replacement species must have the same chemical key and phase. A replacement
family must have the same ordered gas/solid roles and reaction IDs. The action
does not change study selections or saved runs.

Adding both original water and a water variant in Chemistry creates two distinct
components, each with its own ID, composition, and balance equation. Use
**Bindings…** to choose which component serves each reaction role. Likewise,
selecting two reaction variants makes their selected rates add; remove a family
you do not want to run. The editor shows an applicability caution for overlapping
reactions.

### Python approval and checks

The browser reads the manifest without importing plugin Python. **Check** and
**Enable** execute it in a separate process; the application asks for local code
approval first. An enabled plugin copied from another computer may show
**Needs local approval**: choose **Allow code…** and wait for the check to finish.

Approved code runs with your user permissions. The worker is not a security
sandbox, and passing checks does not establish safety or scientific accuracy.
Approval covers code and included resources. Changing manifest parameter values
does not request approval again; changing code or resources does.

Check failures and details appear in the browser or editor. **Cancel check** stops
a pending check. Checks validate package structure, parameter declarations,
factory return types, reaction metadata, and basic property construction. They do
not compare every numerical and symbolic expression or validate a reactor model.

## Share and update plugins

To share the version you edited in the application, select its plugin entry and
choose **Export…**. Send the resulting `.msplugin` file. The recipient registers
it in their project. You can also share the entire project folder; its plugins
travel with it, while Python approval stays on each computer.

For updates, keep the same plugin `id` and register the new manifest or archive.
The application asks before replacing changed contents. To keep two alternatives,
use **Create variant…**, or give your source copy a new `id` before registering it.
Changing just the display name does not create an independent plugin.

Each project stores one current copy at `plugins/<plugin-id>/current/`. There is
no revision selector and no required version increment. Edit your authoring
folder and re-register, or use the application forms and export their result.
Avoid treating the project copy as a live link to your source folder.

Cases and studies use the current contents of their selected plugins. Changing a
selected plugin makes previous results stale; a plugin edit alone does not
require a new study baseline or a study rebuild. An unrelated plugin edit does
not affect those results. Removing or disabling a plugin still used by inputs is
blocked, with its users listed.

Prepared runs keep copies of their selected plugins beside their input snapshot.
Later edits cannot change that snapshot. Generated CLI batch cases also carry
their selected plugin files. Include a source citation, applicable temperature
range, assumptions, and any redistribution terms with a plugin you distribute.

## Authoring reference

Follow the [worked tutorial](../examples/plugins/README.md) for complete files and
commands. This section explains the contract when you adapt them to your own
model. Python authoring commands use the
[source environment](../README.md#install-the-full-repository-from-source), with
the repository root as the working directory.

### Files and names

```text
my_plugin/
├── manifest.yaml       # Identity, species, and implementation references
├── enthalpy.py         # Optional custom property equation
├── kinetics.py         # Optional reaction family and rates
├── data/               # Optional local resources
└── README.md           # Scientific basis and usage notes
```

`manifest.yaml` must be at the root of the folder or archive. Reference a Python
factory as `module:function`: `kinetics:create` calls `create(parameters)` in
`kinetics.py`. A nested module can use `models.kinetics:create`.

| Manifest field | Meaning |
| --- | --- |
| `id` | Stable plugin identity, such as `my_ammonia` |
| `name` | Human-readable plugin name |
| `description`, `source` | Purpose and provenance |
| `species` | Map from local species IDs to property definitions |
| `correlations` | Map from local correlation IDs to custom implementations |
| `mechanisms` | Map from local family IDs to kinetics implementations |
| `derived_from` | Optional attribution to an original definition |

`format_version` and `api_version` currently default to `1`; `version` defaults
to the string `"1"`. The current workflow does not use version fields to select
historical plugin contents.

Plugin and local definition IDs start with a letter and use letters, digits,
underscores, dots, or hyphens. Choose portable names, avoiding Windows device
names and trailing dots. `builtin` is reserved as a plugin ID.

Keep these three names distinct:

| Name | Example | Purpose |
| --- | --- | --- |
| Definition reference | `my_ammonia:NH3` | Plugin ID plus local species ID |
| Chemical key / reaction role | `NH3` | Matches a species to a chemical role |
| Case component ID | `NH3` or `NH3_variant` | Identifies the component in case compositions and balances |

Two definitions can have the same chemical key but different references. Case
bindings choose which selected component supplies that role. Quote YAML strings
such as `'NO'`, including map keys, because the YAML reader interprets unquoted
`NO` as a boolean.

### Write declarations once, generate browsing metadata

The manifest lets the application show a Python plugin's parameters and reactions
without importing its code. Authors write those declarations in Python, then
generate their manifest copies. You do not need to maintain both by hand.

1. Write `PARAMETERS` and `create(parameters)` in your Python module.
2. Put its `implementation` reference in the manifest. For a correlation, also
   set `kind` to `enthalpy` or `viscosity`.
3. Run `generate_metadata(folder)`. It fills parameter declarations and, for a
   reaction family, the gas roles, solid roles, and reactions.
4. Check the package, then register its completed manifest or exported archive.

The [tutorial build script](../examples/plugins/README.md#prepare-the-python-authoring-tools)
performs metadata generation, checking, and packaging with one command.
Generation executes your local factories and rewrites `manifest.yaml`; use it
on your own authoring files. It is not needed for a species plugin that uses only
built-in property equations. Changing Python parameter declarations or reaction
metadata requires regeneration; changing a species' parameter values does not.

### Parameters and factories

Import the supported types from `packed_bed.plugins`. Its exports include
`ParameterSpec`, `parameter_group`, `BaseCorrelation`, `PolynomialHeatCapacity`,
`ShomateHeatCapacity`, `QuadraticViscosity`, `ReactionDefinition`, `ReactionFamily`,
and `KineticsContext`.

Declare editable scalar parameters in a module-level dictionary:

```python
from packed_bed.plugins import ParameterSpec

PARAMETERS = {
    "k": ParameterSpec(
        0.001, "m^3 bed/(mol*s)",
        "Rate coefficient using concentrations per bed volume", minimum=0,
    ),
}
```

Every parameter needs a default and unit or explicit dimensionless basis. Bounds
are optional and inclusive. The factory receives validated, immutable values.
Use factory arguments and `context.parameters`, rather than modifying global
constants. Only declared parameters appear in the application editor.

The three places parameters appear have different jobs:

| Location | What it stores |
| --- | --- |
| Python `PARAMETERS` and generated manifest declarations | Defaults, units, descriptions, and bounds |
| `species.<id>.enthalpy.parameters` or `.viscosity.parameters` | Values for that species' correlation |
| `mechanisms.<id>.values` | Overrides for that plugin reaction family's declared defaults |

Two species can share correlation code and have different parameter values.
The [CO2/H2O example](../examples/plugins/enthalpy_overrides/manifest.yaml)
demonstrates this. Dotted declarations created with `parameter_group` resolve
into nested parameter mappings; for example, `oxidation.k` becomes
`parameters["oxidation"]["k"]`.

### Property equations

A property factory returns a `BaseCorrelation` subclass. Implement these methods:

| Method | Enthalpy correlation returns | Viscosity correlation returns |
| --- | --- | --- |
| `value(T)` | Numerical h in J/mol | Numerical μ in Pa·s |
| `dae_expression(T)` | Symbolic h with units | Symbolic μ with units |
| `cp_value(T)` | Numerical Cp in J/(mol·K) | Not used |
| `cp_dae_expression(T)` | Symbolic Cp with units | Not used |

Numerical methods accept scalar temperatures and NumPy arrays, preserving array
shape. Symbolic methods accept DAETools expressions; use DAETools functions and
unit-bearing constants there. Keep solver imports inside symbolic methods so
metadata generation and ordinary numerical checks can run without the solver.

Require `dh/dT = Cp` and `h(t_ref) = h_ref`. Check finite values over your intended
range, positive Cp and viscosity where applicable, and agreement between the
numerical and symbolic implementations. Registration is not a full scientific
self-test. [Lesson 2](../examples/plugins/README.md#lesson-2-write-a-new-property-equation)
includes a numerical consistency check.

### Reaction equations

A kinetics factory returns a `ReactionFamily` with a name, reaction definitions,
required gas and solid roles, and a map from each reaction ID to its rate function.
The map needs exactly one callable for every reaction.

A `ReactionDefinition` provides `id`, `name`, `phase` (`gas_gas`, `gas_solid`, or
`solid_solid`), `stoichiometry`, `required_species`, and `source_reference`.
Negative stoichiometric coefficients consume a species; positive ones produce
it. Declare all species read by a rate in `required_species`, including catalysts
or other non-stoichiometric dependencies. List catalysts in `catalyst_species`
without including them in stoichiometry.

The rate receives a `KineticsContext`. Use:

- `context.model` and `context.idx_cell` for the local model state;
- `context.gas_index(role)` and `context.solid_index(role)` for bound indices;
- `context.molecular_weight(role)` for the selected component's molecular weight;
- `context.parameters` for the immutable parameter values.

Return a symbolic **reaction-extent rate in mol/(m³ bed·s)**. The engine multiplies
it by stoichiometric coefficients to form species sources. Do not return a
species-consumption rate without accounting for that coefficient. State
variables `model.c_gas` and `model.c_sol` are per total bed volume. If your rate
law uses concentration per gas volume, convert using `model.gasfrac`. Rate orders
come from your equation, independently of stoichiometry.

`reversible` is descriptive metadata; the rate function must implement any
reverse or equilibrium term. Document guards for negative trial concentrations
and zero inventories. Use the case's resolved properties rather than a global
property registry. See [Lesson 3](../examples/plugins/README.md#lesson-3-add-a-reaction)
for a complete factory and rate function, and the [equations guide](EQUATIONS.md)
for the balance and energy conventions.

### Dependencies and resources

Keep a plugin self-contained. Include any custom species and correlations its
mechanisms need. The three supplied examples work independently; one plugin
cannot rely on another plugin's implementation files.

Use relative imports between included modules, such as
`from .helpers import rate_factor`. Locate local data relative to `__file__`.
Supported libraries are Python's standard library, `numpy`, `scipy`, `yaml`,
`pydantic`, `xarray`, `matplotlib`, `daetools`, `pyUnits`, and the public
`packed_bed.plugins` API. Users should not need plugin-specific pip installs or
runtime downloads.

Packages cannot include symlinks or additional native libraries/runtimes
(`.so`, `.dll`, `.pyd`, `.exe`, `.whl`). Paths must be portable across supported
platforms. The package limit is 100 MiB and 5,000 files. Write archives outside
the source plugin folder so they do not become part of the package.

## Case YAML and engine integration

The application writes these mappings for you. For hand-authored inputs, this
`chemistry.yaml` fragment selects the supplied ammonia example:

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

Use `my_ammonia:NH3` and `my_ammonia:synthesis` for the tutorial's own plugin.
These are chemistry selections, not a complete reactor case. Supply the other
case files and make the plugin available through the project or a descriptor.

Program composition maps use case component IDs. When the role and component
names differ, add a `bindings` map inside that mechanism selection, for example
`bindings: {N2: N2_variant, H2: H2, NH3: NH3}` with `N2_variant` selected in
`gas_species`. Solid components are selected in `solids.yaml`; their definition
references also belong in `chemistry.yaml`'s `species_definitions` map.

Reaction IDs normally take the form `<instance>/<local-reaction-id>`. Existing
built-in family instance names retain their legacy reaction IDs, including after
a compatible replacement.

`load_case` discovers a containing project and its current plugins. Saved run
snapshots use their frozen copies. Standalone inputs can place `definitions.json`
beside `run.yaml`:

```json
{
  "root": "relative/path/to/project",
  "lock": {
    "builtin": "<built-in SHA-256 fingerprint>",
    "plugins": {"my_ammonia": "<plugin SHA-256 content hash>"}
  }
}
```

Here `root` is relative to the descriptor; the selected package is under
`plugins/my_ammonia/current/` within that root. The strings in angle brackets are
placeholders, not valid hashes. Desktop snapshots and CLI batch generation write
these descriptors for you. Batch cases use `root: "."` and carry their plugins.

An exact lock checks both built-in definitions and selected plugin contents.
Mismatches are rejected: use the matching application and plugin copies, or
explicitly regenerate inputs against the current definitions. Inputs without an
exact lock use current definitions. A descriptor does not grant local Python
approval.

For Python integration, `select_definitions(chemistry, solids, catalogue)` and
`materialize_definitions(selection, catalogue)` in
[`packed_bed.definitions`](../packed_bed/definitions.py) separate metadata
selection from implementation loading. Materialization returns a
`DefinitionEnvironment` with `properties`, `reaction_network`, `rate_hooks`, and
`selection`. Pass it through `load_case(..., definitions=environment)` or
`resolve_case(..., definitions=environment)`. `Case.definitions` is authoritative;
a supplied environment does not fill missing definitions from global built-ins.
`inspect_case` and `inspect_case_file` return metadata-only `CaseInputs` for editors.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| Plugin is enabled, but the case still uses the old properties | Select the new definition in Chemistry or use **Use this as replacement…**. |
| No compatible replacement appears | Select a species or family entry. Check that an ordinary case or reusable definition uses the old definition and that chemical key/phase or family roles/reaction IDs match. |
| Editing the source files has no effect | Register the updated `manifest.yaml` again; the project holds a copy. |
| A new copy replaces my previous plugin | Give it a different top-level `id`; changing only `name` keeps its identity. |
| A parameter is missing from the editor | Declare it in Python `PARAMETERS`, regenerate metadata, then re-register. For existing declared values, edit species `parameters` or mechanism `values`. |
| Python declarations or reaction metadata differ from the manifest | Run the tutorial build script after editing declarations, then register the generated package. |
| `NO` is rejected as a species key | Quote `'NO'` wherever it occurs in YAML. |
| `Needs local approval` | Select the plugin and choose **Allow code…** on this computer. |
| Missing implementation or unsupported import | Include the referenced `.py` file, use relative imports for your own modules, and use supported libraries. |
| Property check rejects returned values | Ensure numerical methods accept arrays and return finite arrays of the same shape. Basic checks sample 300 K and 1000 K; also test your declared range yourself. |
| Missing species or ambiguous roles | Add the needed components, then set **Bindings…** and their compositions/loadings. |
| Remove or Disable is blocked | Replace or remove its uses in the cases, reusable definitions, and study baselines listed by the application. |
| Saved input fingerprint or hash mismatch | Use the recorded application/plugin contents or explicitly regenerate inputs; do not alter the lock to conceal the mismatch. |

For a larger scientific package with provenance and fitted kinetics, see the
[waste iron plugin](../plugins/waste_iron/README.md). The manifest contract is in
[`schema.py`](../packed_bed/plugins/schema.py), checking and metadata generation
in [`check.py`](../packed_bed/plugins/check.py), and packaging in
[`storage.py`](../packed_bed/plugins/storage.py).

[Documentation index](../README.md#documentation)
