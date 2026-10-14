# Create your first plugin

This walkthrough builds plugins from small, complete files. Start with Lesson 1
if you only need new property coefficients. Continue to Lessons 2 and 3 when you
need to write an equation.

| Lesson | What you create | Files you write |
| --- | --- | --- |
| [1. A species using built-in equations](#lesson-1-create-a-plugin-with-one-file) | Your own nitrogen properties | One YAML file |
| [2. A new property equation](#lesson-2-write-a-new-property-equation) | NH3 with a custom heat-capacity/enthalpy equation | A manifest and one Python module |
| [3. A reaction](#lesson-3-add-a-reaction) | N2 + 3 H2 → 2 NH3 with an editable rate coefficient | One more Python module and a short manifest entry |

Each lesson ends with something you can register in the application. The
coefficients and rate law are illustrative, not validated property datasets or
an industrial ammonia model. General usage, units, API details, and
troubleshooting are in the [plugin guide](../../docs/PLUGINS.md).

## Before you start

For Lesson 1, you need a text editor and the desktop application. Alternatively,
you can create a species entirely in the application using
[Create plugin / Add species](../../docs/PLUGINS.md#create-a-species-plugin).

For Lessons 2 and 3, use the
[source installation](../../README.md#install-the-full-repository-from-source)
and activate its Python environment. Run the terminal commands below from the
repository root, the folder containing `pyproject.toml`. The supplied factories
keep DAETools imports inside symbolic methods, so generating metadata and running
their numerical checks does not require loading the solver. Simulations do.

Create a workspace folder called `build/plugin-workshop` inside the repository.
Use your file manager or editor to create folders and files as instructed.
`build/` is ignored by Git, so keep a separate copy of work you want to retain
or share. Save files as UTF-8 plain text with their exact extensions:
`manifest.yaml`, not `manifest.yaml.txt`.

## Lesson 1: create a plugin with one file

### 1. Create the folder and manifest

Inside `build/plugin-workshop`, create a folder called `my_nitrogen`.
Create `my_nitrogen/manifest.yaml` with this complete content:

```yaml
id: my_nitrogen
name: My nitrogen properties
source: Tutorial example; illustrative Cp coefficients.
species:
  N2:
    name: Nitrogen
    chemical_key: N2
    phase: gas
    mw: 0.0280134
    temperature_range: [298.15, 1200.0]
    enthalpy:
      model: builtin:polynomial
      parameters:
        coefficients: [30.0, 0.004, 0.000001]
        t_ref: 298.15
        h_form_ref: 0.0
    viscosity:
      model: builtin:quadratic_viscosity
      parameters:
        t_ref: 1000.0
        a0: 4.15082118e-5
        a1: 2.78501267e-8
        a2: -6.09391431e-12
```

That is the entire plugin. The molecular weight and viscosity are taken from the
built-in nitrogen definition; the heat-capacity coefficients are illustrative.

The fields you are most likely to change are:

| Field | Purpose |
| --- | --- |
| `id: my_nitrogen` | Identifies this plugin. Use a new ID for a separate plugin, and keep it for later updates. |
| `name` | The label shown in the plugin browser. |
| `species: N2:` | The species' local ID; its full reference is `my_nitrogen:N2`. |
| `chemical_key: N2` | Identifies this as nitrogen, so it can fill an N2 reaction role or replace built-in nitrogen. |
| `mw` | Molecular weight in **kg/mol**. |
| `enthalpy.parameters` | This species' property coefficients, reference temperature, and reference enthalpy. |
| `source` | Where the values came from; replace the tutorial note with your scientific source for real data. |

Here `Cp = 30 + 0.004 (T − 298.15) + 0.000001 (T − 298.15)²`, in J/(mol·K).
The application integrates Cp to obtain molar enthalpy, with `h(298.15 K) = 0`.
The declared temperature interval is in K. Gas species also need viscosity;
solid species omit the `viscosity` entry.

### 2. Register it and use it

1. Open a project and choose **Plugins → Register plugin…**.
2. Select `build/plugin-workshop/my_nitrogen/manifest.yaml`.
3. Confirm that **My nitrogen properties** appears as **Enabled**. Expand it and
   select **Nitrogen** to inspect your definition.
4. To change an existing case, select this new nitrogen and choose
   **Use this as replacement…**, choose the existing nitrogen definition, and
   review the affected cases. If no case uses nitrogen yet, add your nitrogen
   from the species picker in a case's **Chemistry** tab.

**Checkpoint:** the case's nitrogen definition comes from **My nitrogen
properties**. Merely registering the plugin does not select it in a case.

### 3. Edit and share it

Try changing `30.0` to `31.0` in the coefficients. Register the same manifest
again and accept replacement of the project's copy. The top-level `id` stays
`my_nitrogen`. You can also double-click the plugin species in the application
and edit its coefficients there.

To share the current project copy, select the plugin and choose **Export…**.
This produces a `.msplugin` archive that another user can register.

You can stop here if built-in property equations meet your needs. The supplied
[nitrogen oxides manifest](nitrogen_oxides/manifest.yaml) follows this same
pattern, adding NO and NO2 alongside the N2 variant. Quote `'NO'` in YAML:
unquoted `NO` is read as a boolean.

## Prepare the Python authoring tools

The remaining lessons define equations in Python. You will write their
parameters and reactions once in Python and let the application library generate
the manifest's browsing metadata.

Save the following complete script as
`build/plugin-workshop/build_plugin.py`, **outside** the plugin folders:

```python
import argparse
from pathlib import Path

from packed_bed.plugins.check import check_package, generate_metadata
from packed_bed.plugins.storage import package_hash, pack_plugin

parser = argparse.ArgumentParser(
    description="Generate metadata, check, and package a plugin you are authoring."
)
parser.add_argument("folder", type=Path, help="Folder containing manifest.yaml")
parser.add_argument("archive", type=Path, help="Output .msplugin path outside that folder")
args = parser.parse_args()

generate_metadata(args.folder)
# Execute this particular local package for the authoring check.
# This does not save desktop approval or grant approval to recipients.
check_package(args.folder, approved=(package_hash(args.folder),))
archive = pack_plugin(args.folder, args.archive)
print("Checks passed.")
print(f"Created {archive}")
```

This is a helper you create for the tutorial, not a preinstalled command.
It executes the package's Python factories, rewrites `manifest.yaml` with their
declarations, checks the package, and creates the archive. Run it on your own
authoring files. The first argument is a **folder**, not its manifest file.

The same script also packages Lesson 1:

```sh
python build/plugin-workshop/build_plugin.py build/plugin-workshop/my_nitrogen dist/plugin-workshop/my_nitrogen.msplugin
```

It should print `Checks passed.` and the archive path. The output directory is
created automatically. Keep the source folder for editing; the archive is what
you distribute.

## Lesson 2: write a new property equation

Now create NH3 with a heat capacity that approaches a high-temperature limit:

```text
Cp(T) = cp_inf − deficit (t_ref/T)²
h(T)  = h_ref + cp_inf (T − t_ref)
        + deficit t_ref² (1/T − 1/t_ref)
```

The second equation is the exact integral of the first, anchored at `h_ref`.
This is the same model used in the supplied [ammonia example](ammonia/).

### 1. Describe the species and point to the equation

Create `build/plugin-workshop/my_ammonia/manifest.yaml`:

```yaml
id: my_ammonia
name: My ammonia model
source: Tutorial example; properties and kinetics are illustrative.
species:
  NH3:
    name: Ammonia
    chemical_key: NH3
    phase: gas
    mw: 0.01703052
    temperature_range: [298.15, 1200.0]
    enthalpy:
      model: inverse_square
      parameters:
        cp_inf: 70.0
        deficit: 35.0
        t_ref: 298.15
        h_ref: -46000.0
    viscosity:
      model: builtin:quadratic_viscosity
      parameters:
        t_ref: 298.15
        a0: 1.0e-5
        a1: 3.0e-8
        a2: 0.0
correlations:
  inverse_square:
    name: Inverse-square Cp
    kind: enthalpy
    implementation: enthalpy:create
```

The connection is:

```text
NH3.enthalpy.model: inverse_square
    → correlations.inverse_square
    → enthalpy:create
    → create(parameters) in enthalpy.py
```

`enthalpy.parameters` supplies values for NH3. The correlation's declarations
(defaults, units, descriptions, and bounds) will be generated from Python.
**Finish the next two steps before registering this manifest.**

### 2. Write the equation

Create `build/plugin-workshop/my_ammonia/enthalpy.py` with this complete content:

```python
from dataclasses import dataclass

import numpy as np
from packed_bed.plugins import BaseCorrelation, ParameterSpec

PARAMETERS = {
    "cp_inf": ParameterSpec(70.0, "J/(mol*K)", "High-temperature Cp", minimum=1),
    "deficit": ParameterSpec(35.0, "J/(mol*K)", "Cp deficit at Tref", minimum=0),
    "t_ref": ParameterSpec(298.15, "K", "Reference temperature", minimum=1),
    "h_ref": ParameterSpec(-46000.0, "J/mol", "Enthalpy at Tref"),
}


@dataclass(frozen=True)
class InverseSquareEnthalpy(BaseCorrelation):
    cp_inf: float
    deficit: float
    t_ref: float
    h_ref: float

    def _cp(self, t):
        return self.cp_inf - self.deficit * (self.t_ref / t)**2

    def _h(self, t):
        return (
            self.h_ref
            + self.cp_inf * (t - self.t_ref)
            + self.deficit * self.t_ref**2 * (1 / t - 1 / self.t_ref)
        )

    def value(self, temperature):
        return self._h(np.asarray(temperature, dtype=float))

    def cp_value(self, temperature):
        return self._cp(np.asarray(temperature, dtype=float))

    def dae_expression(self, temperature):
        from daetools.pyDAE import Constant
        from pyUnits import J, K, mol
        return self._h(temperature / Constant(1 * K)) * Constant(1 * J / mol)

    def cp_dae_expression(self, temperature):
        from daetools.pyDAE import Constant
        from pyUnits import J, K, mol
        return self._cp(temperature / Constant(1 * K)) * Constant(1 * J / (mol * K))


def create(parameters):
    if parameters["deficit"] >= parameters["cp_inf"]:
        raise ValueError("Cp at Tref must be positive: deficit must be smaller than cp_inf.")
    return InverseSquareEnthalpy(**parameters)
```

There are three jobs here:

1. `PARAMETERS` declares the values users may edit, including their units and
   bounds.
2. The class evaluates the same equations numerically and symbolically. Numerical
   methods use NumPy and accept arrays. The solver methods normalize temperature
   to K and attach the output units.
3. `create(parameters)` checks the relationship between two parameters and
   returns the correlation object. The application supplies this dictionary.

When adapting the example, begin with `_cp`, its integral `_h`, and the parameter
declarations. Keep numerical and symbolic results consistent. More complex
expressions may need separate numerical and DAETools functions; the
[logarithmic example](enthalpy_overrides/correlation.py) shows `np.log` and
DAETools `Log`.

### 3. Generate, check, and package it

Run this from the repository root in your active Python environment:

```sh
python build/plugin-workshop/build_plugin.py build/plugin-workshop/my_ammonia dist/plugin-workshop/my_ammonia.msplugin
```

**Checkpoint:** the command prints `Checks passed.` and creates
`dist/plugin-workshop/my_ammonia.msplugin`. Open the source manifest: the
`correlations.inverse_square.parameters` declarations have been filled in.
You keep editing the equation and declarations in Python; regenerate instead of
manually synchronizing those generated entries.

Register the completed manifest or archive. Select **My ammonia model**, choose
**Enable**, allow your code, and wait for **Checks passed.** Its Ammonia species
is now available in Chemistry. Double-click the species in Plugins to see the
custom correlation's editable parameters.

### 4. Check that the equation behaves as intended

A package check verifies the software contract. Also check your equation.
Save this as `build/plugin-workshop/check_enthalpy.py`, beside `my_ammonia`:

```python
import numpy as np
from my_ammonia.enthalpy import PARAMETERS, create

parameters = {key: spec.default for key, spec in PARAMETERS.items()}
correlation = create(parameters)
temperature = np.linspace(parameters["t_ref"], 1200.0, 100)
enthalpy = correlation.value(temperature)
cp = correlation.cp_value(temperature)

assert enthalpy.shape == temperature.shape
assert cp.shape == temperature.shape
assert np.all(np.isfinite(enthalpy))
assert np.all(np.isfinite(cp)) and np.all(cp > 0)
np.testing.assert_allclose(
    correlation.value(parameters["t_ref"]), parameters["h_ref"]
)

step = 0.01
numerical_derivative = (
    correlation.value(temperature + step) - correlation.value(temperature - step)
) / (2 * step)
np.testing.assert_allclose(numerical_derivative, cp, rtol=1e-7)
print("Enthalpy reference, array output, positive Cp, and dh/dT checks passed.")
```

Run:

```sh
python build/plugin-workshop/check_enthalpy.py
```

This checks the default parameters. Also test any different species parameter
sets you supply. Before using a new correlation scientifically, compare the
symbolic implementation with the numerical one and verify it against its source
data. This small numerical check does not test the solver.

## Lesson 3: add a reaction

Keep the `my_ammonia` folder from Lesson 2. Add a simple reaction:

```text
N2 + 3 H2 → 2 NH3
r = k c_N2 c_H2
```

Concentrations are in mol/m³ **bed**, and `r` is in mol/(m³ bed·s).
The model is irreversible, first order in each reactant, and has no catalyst,
temperature dependence, or equilibrium term.

### 1. Write the rate and reaction definition

Create `build/plugin-workshop/my_ammonia/kinetics.py`:

```python
from packed_bed.plugins import ParameterSpec, ReactionDefinition, ReactionFamily

PARAMETERS = {
    "k": ParameterSpec(
        0.001, "m^3 bed/(mol*s)",
        "Rate coefficient using concentrations per bed volume", minimum=0,
    ),
}


def rate(context):
    from daetools.pyDAE import Constant, Max
    from pyUnits import m, mol, s

    model, cell = context.model, context.idx_cell
    zero = Constant(0 * mol / m**3)
    nitrogen = Max(model.c_gas(context.gas_index("N2"), cell), zero)
    hydrogen = Max(model.c_gas(context.gas_index("H2"), cell), zero)
    return Constant(context.parameters["k"] * m**3 / (mol * s)) * nitrogen * hydrogen


def create(parameters):
    reaction = ReactionDefinition(
        id="synthesis",
        name="N2 + 3 H2 → 2 NH3",
        phase="gas_gas",
        stoichiometry={"N2": -1, "H2": -3, "NH3": 2},
        required_species=("N2", "H2", "NH3"),
        source_reference="Illustrative plugin example",
        notes="Irreversible, first order in each reactant. No catalyst, equilibrium or pressure fit.",
    )
    return ReactionFamily(
        name="Ammonia synthesis",
        reactions=(reaction,),
        required_gas_species=("N2", "H2", "NH3"),
        required_solid_species=(),
        kinetics_hooks={"synthesis": rate},
    )
```

You define the chemistry in `ReactionDefinition` and the equation in `rate`.
`ReactionFamily` joins them: the `"synthesis"` key in `kinetics_hooks` must match
the reaction's `id`.

The factory must accept `parameters`; this rate reads its values later through
`context.parameters`. The index helpers resolve roles to whichever components
the case has bound to N2 and H2. `Max` protects against negative trial
concentrations during solving.

The engine multiplies your extent rate by stoichiometry: the species sources are
`−r` for N2, `−3r` for H2, and `+2r` for NH3. The coefficient `−3` does not
make the rate third order in H2; a different order must be written in the rate
equation.

### 2. Add its factory to the manifest

Append this at the **top level** of `my_ammonia/manifest.yaml`, aligned with
`species:` and `correlations:`:

```yaml
mechanisms:
  synthesis:
    name: Ammonia synthesis
    implementation: kinetics:create
```

Do not copy the reaction's stoichiometry or parameter declarations into YAML.
Run the same build command again:

```sh
python build/plugin-workshop/build_plugin.py build/plugin-workshop/my_ammonia dist/plugin-workshop/my_ammonia.msplugin
```

The generated manifest now includes the reaction, gas roles, and `k` declaration.
Your complete source folder is:

```text
build/plugin-workshop/
├── build_plugin.py
├── check_enthalpy.py
├── my_nitrogen/
│   └── manifest.yaml
└── my_ammonia/
    ├── manifest.yaml
    ├── enthalpy.py
    └── kinetics.py
```

### 3. Select the reaction in a case

Register the updated manifest or archive and confirm replacement of the project's
copy. Adding `kinetics.py` changes the code, so allow it when prompted.

In the case's **Chemistry** tab:

1. Select built-in N2 and H2, and Ammonia from **My ammonia model**.
2. Choose **+ Add reaction families…** and add **Ammonia synthesis** from your
   plugin.
3. Select its `N2 + 3 H2 → 2 NH3` reaction and inspect **Bindings…**.
4. Supply the case's gas compositions and solid-bed configuration. For example,
   use an inert solid bed and a feed of 0.25 N2 / 0.75 H2 / 0 NH3, within the
   example's 298.15–1200 K temperature interval. Complete the other reactor and
   operating inputs before running.

**Checkpoint:** the family and reaction appear in Chemistry, with NH3 bound to
your plugin species. This lesson defines the plugin; the feed suggestion is not
a complete reactor case.

### 4. Change a rate parameter without rewriting code

In **Plugins**, select your synthesis reaction and choose **Edit…**. Change
`k` from `0.001` to `0.002`, then save. At the same concentrations the rate
doubles.

To make that change in your source files instead, add `values` to the existing
`mechanisms.synthesis` entry:

```yaml
    values:
      k: 0.002
```

Keep the generated fields already in that entry. Run the build command and
re-register. `values.k` overrides the default for this plugin; the default
declaration remains `0.001`. Changing the value needs no new code approval.
Changing Python parameter names, defaults, units, bounds, or reaction declarations
requires metadata regeneration and re-registration.

To create a separate model, copy the folder and change its top-level `id` and
`name`. To add another reaction to the same family, create another
`ReactionDefinition`, add it to `reactions`, and add its rate function to
`kinetics_hooks`. Include every species the rate reads in the reaction's
`required_species` and the family's gas/solid roles.

## Use the supplied examples as templates

The repository also contains complete, ready-to-register packages:

| Folder | What it demonstrates | Files to start with |
| --- | --- | --- |
| [nitrogen_oxides](nitrogen_oxides/) | Several species using built-in equations, including an N2 variant | [manifest.yaml](nitrogen_oxides/manifest.yaml) |
| [ammonia](ammonia/) | The custom property equation and rate law built in Lessons 2–3 | [manifest.yaml](ammonia/manifest.yaml), [enthalpy.py](ammonia/enthalpy.py), [kinetics.py](ammonia/kinetics.py) |
| [enthalpy_overrides](enthalpy_overrides/) | Two species sharing a custom correlation with different parameter values | [manifest.yaml](enthalpy_overrides/manifest.yaml), [correlation.py](enthalpy_overrides/correlation.py) |

Each folder is independent. You may register its manifest directly; the Python
examples need **Enable** and local code approval. Registering them does not
automatically replace any case definitions.

For the CO2/H2O example:

1. Copy `enthalpy_overrides` into your workspace and give its manifest a new
   `id` and `name` if you want an independent plugin.
2. Read `correlation.py`. It uses
   `Cp = cp_inf − deficit t_ref/T` and
   `h = h_ref + cp_inf (T − t_ref) − deficit t_ref log(T/t_ref)`.
3. Edit the CO2 or H2O `enthalpy.parameters` in the manifest. Each species has
   its own coefficients despite using the same `logarithmic` correlation.
4. Build with the helper script, register, and enable it. Select the plugin's
   **Carbon Dioxide** or **Water**, then **Use this as replacement…** to apply it
   to existing compatible case definitions.

Those species retain the built-in molecular weights, viscosities, and reference
enthalpies; the Cp curves are illustrative replacements. Their shared code does
not require installing a separate correlation plugin.

## Share your finished example

If you edited source files, share the archive created by `build_plugin.py`.
If you edited the project copy in the application, use **Export…** from Plugins
to share those edits. These are separate copies, so choose the one you changed.

Include a README describing the equations, units, source data, intended range,
and known limitations. Recipients register the archive and approve Python code
locally. For changes to an existing plugin, keep its `id`; for a separate
alternative, give it a new one.

[Plugin guide and troubleshooting](../../docs/PLUGINS.md) ·
[Documentation index](../../README.md#documentation)
