# multisolid-CL packed bed model

`packed_bed` is a DAETools-based packed-bed reactor simulation package for
multi-species gas flow, solid inventories, heat balance, reaction source terms,
and configurable operating programs. Runs are driven by YAML files that describe
the selected gas species, solid species, reactions, inlet program, geometry,
solver tolerances, and requested outputs.

The current source tree is intended to be run from a checkout. There is no
project packaging metadata in this repository yet, so install the runtime
dependencies into your Python environment and run commands from the repository
root.

## Requirements

- Python 3.11 or Python 3.12.
- DAETools 2.6.0, including the dependencies normally required by DAETools.
- Runtime Python dependencies used directly by this package:
  - `numpy`
  - `pandas`
  - `pydantic`
  - `PyYAML`
  - `matplotlib`
  - `PyQt6`
  - `PyQt6-WebEngine`
- Optional dependencies:
  - Graphviz plus `pygraphviz`, used to render the species/reaction system graph.
  - `vtk`, for DAETools or visualization workflows that need VTK support.

Example environment setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas pydantic PyYAML matplotlib
```
Afterwards follow `daetools` installation guide.

Install optional graph rendering support only if you need `system_graph.svg`:

```powershell
python -m pip install pygraphviz
```

`pygraphviz` also needs the Graphviz system libraries and executables available
to the build and runtime environment.

## Running a case

The package entry point is:

```powershell
python -m packed_bed packed_bed\examples\medrano_case\run.yaml
```

Validate a case without integrating it:

```powershell
python -m packed_bed packed_bed\examples\medrano_case\run.yaml --validate-only
```

Open the DAETools plotter after a successful run:

```powershell
python -m packed_bed packed_bed\examples\medrano_case\run.yaml --dae-plotter
```

## Running a batch

Batch runs expand a cartesian product of named options into ordinary case
directories. Each generated case contains materialized `run.yaml`,
`chemistry.yaml`, `program.yaml`, and `solids.yaml` files, then uses the normal
single-case validation and execution path.

```powershell
python -m packed_bed batch packed_bed\examples\default_batch_case\batch.yaml
```

Validate and materialize the generated cases without integrating them:

```powershell
python -m packed_bed batch packed_bed\examples\default_batch_case\batch.yaml --validate-only
```

Kill and mark a single generated case as failed if it exceeds a per-case wall
clock timeout, while continuing with the rest of the batch:

```powershell
python -m packed_bed batch packed_bed\examples\default_batch_case\batch.yaml --case-timeout-s 600
```

The same default can be set in `batch.yaml` with `case_timeout_s: 600.0`; the
CLI option overrides the file setting.

The batch specification lives beside its own base case. Program presets can use
the normal mol/s inlet-flow form, or can set `inlet_flow.basis: ghsv_per_h`.
GHSV values are converted during batch materialization using the final case
geometry, 273.15 K, and 1 bar; the generated `program.yaml` always contains
mol/s.

The top-level `run.yaml` points to three sibling input files:

- `chemistry.yaml`: selected gas species and reaction IDs.
- `program.yaml`: inlet flow, inlet temperature, outlet pressure, and inlet
  gas composition programs.
- `solids.yaml`: selected solid species and the initial axial solid profile.

Outputs are written under the `outputs.directory` configured in `run.yaml`.
Artifacts are written under `outputs.artifacts_directory`.

Requested report data are written as pickled pandas dataframes:

- `reports.pkl` contains spatial-only reports such as temperature and pressure
  with multi-indexed columns `(feature, x_cell_m)`.
- `balances.pkl` contains requested mass and heat balance totals and errors.
- species/spatial reports are written one feature per file, for example
  `gas_mole_fraction.pkl`, with multi-indexed columns such as
  `(gas_species, x_cell_m)`.

Each report can be loaded in one line:

```python
reports = pd.read_pickle("output/reports.pkl")
```

## Input file shape

`chemistry.yaml` selects species and reactions:

```yaml
gas_species:
  - H2
  - H2O
  - N2
  - O2

reaction_ids:
  - ni_reduction_h2_medrano_an
  - ni_oxidation_o2_medrano_an
```

`program.yaml` defines scalar channels and the inlet composition channel. Every
composition mapping must contain exactly the configured gas species, and mole
fractions must sum to `1.0`.

```yaml
inlet_flow:
  initial: 1.0

inlet_temperature:
  initial: 600.0

outlet_pressure:
  initial: 5000000.0

inlet_composition:
  initial:
    H2: 0.0
    H2O: 0.0
    N2: 1.0
    O2: 0.0
  steps:
    - kind: ramp
      duration_s: 5.0
      target:
        H2: 0.0
        H2O: 0.0
        N2: 0.8
        O2: 0.2
    - kind: hold
      duration_s: 150.0
```

`solids.yaml` defines selected solid species and one or more contiguous axial
zones. The zones must start at `0.0`, must not contain gaps or overlaps, and
must end at `model.bed_length_m`.

```yaml
solid_species:
  - Ni
  - NiO

initial_profile:
  basis: bed
  zones:
    - x_start_m: 0.0
      x_end_m: 2.5
      e_b: 0.5
      e_p: 0.5
      d_p: 0.01
      values:
        Ni: 1143.0
        NiO: 0.0
```

`run.yaml` selects geometry, solver settings, numerical schemes, and reports.
Supported axial schemes are `upwind1`, `central`, `linear_upwind2`, `muscl_minmod`, `weno3`, `weno5`.

Common report IDs include `temperature`, `pressure`, `velocity`, `gas_concentration`,
`gas_mole_fraction`, `solid_concentration`, `solid_mole_fraction`, `gas_flux`,
`gas_source`, `solid_source`, `reaction_rate`, `gas_enthalpy_flux`,
`heat_balance`, and `mass_balance`.

Set `outputs.solver_incidence_matrix: true` to write solver sparsity artifacts
after DAETools initializes the model. The run writes the raw DAETools XPM plus a
labelled HTML incidence matrix and a CSV edge list under `artifacts_directory`.

## Adding new components

In this codebase, a "component" is a gas or solid species identifier such as
`H2`, `Ni`, or `Fe2O3`. Component IDs are used consistently across the property
registry, reactions, chemistry configuration, inlet composition, and solid
profiles.

To add a component:

1. Add a `SpeciesPropertyRecord` to `PROPERTY_REGISTRY` in
   `packed_bed/properties.py`.
2. Use `phase="gas"` for gas species and `phase="solid"` for solid species.
3. Provide `mw` in `kg/mol`.
4. Provide an enthalpy correlation for every gas and solid species.
5. Provide a viscosity correlation for every gas species.
6. Add the species ID to `chemistry.yaml` if it is a gas species.
7. Add the species ID to `solids.yaml` and to every solid zone's `values` map if
   it is a solid species.
8. For gas species, add that species to every `program.yaml`
   `inlet_composition.initial` and ramp `target` map.
9. If a reaction uses the component, include it in that reaction's
   `required_species`, `stoichiometry`, or `catalyst_species` as appropriate.

Gas and solid species identifiers must be disjoint. The loader rejects a run if
the same ID appears in both `gas_species` and `solid_species`.

Example gas record:

```python
"H2": SpeciesPropertyRecord(
    name="Hydrogen",
    phase="gas",
    mw=2.01588e-3,
    enthalpy=CpCubicMolar(
        h_form_ref=0.0,
        a0=2.94409905e01,
        a1=-2.38377533e-03,
        a2=6.39601662e-06,
        a3=-2.03147561e-09,
    ),
    viscosity=ViscosityQuadratic(
        a0=2.04091133e-05,
        a1=1.41343819e-08,
        a2=-2.34255119e-12,
    ),
)
```

Example solid record:

```python
"NiO": SpeciesPropertyRecord(
    name="Nickel Oxide",
    phase="solid",
    mw=74.6928e-3,
    enthalpy=CpCubicMolar(
        h_form_ref=-239701.0,
        a0=5.64774634e01,
        a1=-1.56343578e-02,
        a2=2.10045988e-05,
        a3=-4.78601077e-09,
    ),
)
```

## Adding property correlations

Property correlations are defined in `packed_bed/properties.py`.

Molar enthalpy correlations implement `MolarEnthalpyCorrelation`:

```python
class MolarEnthalpyCorrelation(ABC):
    def dae_expression(self, temperature):
        ...

    def value(self, temperature):
        ...
```

Gas viscosity correlations implement `GasViscosityCorrelation`:

```python
class GasViscosityCorrelation(ABC):
    def dae_expression(self, temperature):
        ...

    def value(self, temperature):
        ...
```

Follow the existing pattern:

- `value(...)` is the numeric NumPy implementation used for validation, reports,
  or offline calculations.
- `dae_expression(...)` returns a DAETools symbolic expression with units,
  usually by wrapping constants with `daetools.pyDAE.Constant` and `pyUnits`.
- Temperatures are in K.
- Enthalpy is in J/mol.
- Heat capacity coefficients should evaluate to J/(mol K).
- Gas viscosity is in Pa s.

Existing enthalpy/Cp bases include `CpZerothMolar`, `CpQuadraticMolar`,
`CpCubicMolar`, `CpQuarticMolar`, and `CpShomateMolar`. Existing gas viscosity
support includes `ViscosityQuadratic`.

After adding a new correlation class, use it in a `SpeciesPropertyRecord` in
`PROPERTY_REGISTRY`.

## Adding reaction definitions

Reaction metadata lives in `packed_bed/reactions.py`. Add a
`ReactionDefinition` to `REACTION_CATALOG`:

```python
"my_reaction_id": ReactionDefinition(
    id="my_reaction_id",
    name="Readable reaction name",
    phase="gas_solid",
    stoichiometry={
        "H2": -1.0,
        "NiO": -1.0,
        "Ni": 1.0,
        "H2O": 1.0,
    },
    required_species=("H2", "H2O", "Ni", "NiO"),
    source_reference="Citation or source note",
    kinetics_hook="my_kinetics_hook",
    reversible=False,
    notes="Short implementation note.",
)
```

Important rules:

- `id` must match the dictionary key.
- `phase` must be one of `gas_gas`, `gas_solid`, or `solid_solid`.
- Reactants use negative stoichiometric coefficients.
- Products use positive stoichiometric coefficients.
- Do not include zero coefficients.
- `required_species` must include every stoichiometric species and every
  catalyst species.
- Catalysts go in `catalyst_species`, not in `stoichiometry`.
- Set `kinetics_hook` to the registry key for the hook that computes this
  reaction rate.
- Add the reaction ID to `chemistry.yaml` to select it for a run.

The reaction network builder creates gas and solid source matrices from the
stoichiometry. The current solver stores reaction rates as `mol/(m^3 s)` per
total bed volume, so convert any catalyst-volume, gas-volume, or solid-volume
rate expression inside the kinetics hook before returning it.

## Adding kinetics hooks

Kinetics hooks are registered through `packed_bed/kinetics/__init__.py`.
Existing implementations are split into files such as `medrano.py`,
`medrano_an.py`, `xu_froment.py`, `numaguchi_an.py`, `coper_redox.py`, and
`fe_redox.py`.

The workflow is:

1. Add a `ReactionDefinition` in `packed_bed/reactions.py`.
2. Implement a hook function in a module under `packed_bed/kinetics`.
3. Decorate it with `@register_kinetics_hook("hook_name")`.
4. Set `ReactionDefinition.kinetics_hook="hook_name"`.
5. Ensure the kinetics module is imported from `packed_bed/kinetics/__init__.py`
   so the decorator runs at package import time.
6. Add numeric helper tests where possible. Existing kinetics modules usually
   separate numeric helper functions from DAETools expression helpers for this
   reason.

Minimal hook shape:

```python
from daetools.pyDAE import Constant, Exp
from pyUnits import K, m, mol, s

from . import KineticsContext, register_kinetics_hook


@register_kinetics_hook("my_kinetics_hook")
def my_kinetics_hook(context: KineticsContext):
    gas_idx = context.gas_index("H2")
    solid_idx = context.solid_index("NiO")

    temperature_k = context.model.T(context.idx_cell) / Constant(1.0 * K)
    h2_y = context.model.y_gas(gas_idx, context.idx_cell)
    nio_c = context.model.c_sol(solid_idx, context.idx_cell) / Constant(1.0 * mol / m**3)

    rate_expression = h2_y * nio_c * Exp(-10000.0 / temperature_k)
    return Constant(1.0 * mol / (m**3 * s)) * rate_expression
```

`KineticsContext` provides:

- `model`: the DAETools model instance.
- `idx_cell`: the distributed axial cell index for the current equation.
- `gas_index(species_id)`: index lookup for gas species variables.
- `solid_index(species_id)`: index lookup for solid species variables.
- `reaction_lookup(reaction_id)`: index lookup for selected reactions.

Frequently used model variables in kinetics hooks include:

- `model.T(idx_cell)`: bed temperature.
- `model.P(idx_cell)`: bed pressure.
- `model.y_gas(gas_idx, idx_cell)`: gas mole fraction.
- `model.c_gas(gas_idx, idx_cell)`: gas concentration.
- `model.c_sol(solid_idx, idx_cell)`: solid concentration.
- `model.R_rxn(reaction_idx, idx_cell)`: reaction rate variable.

Each selected reaction must have a resolvable kinetics hook. If a selected
reaction has `kinetics_hook=None`, or if the hook name is not registered, the
simulation assembly raises `NotImplementedError`.

## Validation behavior

`load_run_bundle(...)` validates each YAML file and then validates cross-file
rules. It checks, among other things:

- unknown species IDs,
- gas/solid phase mismatches,
- missing required property data,
- duplicate species or reaction IDs,
- inlet composition species mismatches,
- inlet composition sums,
- solid zone contiguity,
- unknown reaction IDs,
- reactions that require unselected species,
- unknown report IDs,
- missing or unknown kinetics hooks during simulation assembly.

Use `--validate-only` after editing inputs or adding new model components.
