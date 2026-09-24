# Waste iron plugin

Port of `waste_iron_kinetics` at **9660e5d497265393f49e5a4c51b6c348650415b5**.
This package contains the branch's final kinetics and two solid-property
definitions. It uses only the application and its bundled libraries.

## Use in a project

1. Choose **Plugins → Register plugin** and select this `manifest.yaml` or
   `waste_iron.msplugin`. Enable **Waste iron** and approve its Python code.
2. Add **Waste iron redox** from this plugin in Chemistry. Select oxidation,
   reduction, or both. Both reactions need built-in Fe and Fe2O3. Oxidation needs
   O2; reduction needs H2 and H2O. Set their bed loadings and feed composition.
3. Add plugin SiO2 and/or Al2O3 when required for the support. To update existing
   alumina selections, select this plugin's **Aluminium Oxide (alpha)** and choose
   **Use this as replacement…**, then choose the existing Al2O3 definition.

The reactions form a new family, **not a replacement for built-in `iron_he`**:
the waste model uses direct Fe ↔ Fe2O3 conversion, whereas `iron_he` includes
intermediate oxides. Remove other iron reactions from the case if they should
not operate alongside this family; selected reaction rates add.

For case YAML, the family selection is:

```yaml
gas_species: [N2, O2, H2, H2O]
species_definitions:
  SiO2: waste_iron:SiO2
  Al2O3: waste_iron:Al2O3
reaction_families: [fe_waste]
mechanisms:
  fe_waste:
    definition: waste_iron:fe_waste
reaction_ids: [fe_waste/fe_waste_oxidation, fe_waste/fe_waste_reduction]
```

This snippet assumes `solids.yaml` selects Fe, Fe2O3, SiO2 and Al2O3. Omit any
unused support and its definition mapping. Registering the plugin alone does
not change existing species selections or reaction families.

## Preserved kinetics

Original file: `packed_bed/kinetics/fe_waste.py`. Reaction IDs and stoichiometry
are retained:

- `fe_waste_oxidation`: **Fe + 0.75 O2 → 0.5 Fe2O3**.
- `fe_waste_reduction`: **Fe2O3 + 3 H2 → 2 Fe + 3 H2O**.

The branch attributes the kinetics to Aya's iron-waste measurements and
`Solid-State kinetic modelling (iron waste based catalyst).pdf`. This port
preserves the implementation through `9660e5d`, including its later fitted and
numerical changes; it is not a fresh transcription or validation of that PDF.

Let `F` and `H` be Fe and Fe2O3 concentrations in mol/m³ **bed**. Gas
concentrations are in mol/m³ **gas**, computed as `model.c_gas / model.gasfrac`.
Both returned reaction-extent rates are in mol/(m³ bed·s).

For oxidation, `S = H + F/2`, `X = H/S`, and
`r = 2 S k (1−X) sqrt(1−psi log(1−X+epsilon)) c_O2/(c_O2+K_O2)`.
The default `k` is 0.0073738573592314 s⁻¹, `psi` is 8, `K_O2` is 1 mol/m³ gas
and `epsilon` is 1e-10. Fe consumption is `r`, O2 consumption is `0.75 r` and
Fe2O3 production is `0.5 r`.

For reduction, `S = F + 2 H`, `X = F/S`, and
`r = S/2 · k_eff · rational(1−X) · (1−X)^4 · pade(X)`.
The normalized two-term rational approximation retains `(1.452, 11.849,
1.038, 0.170)`; the Padé numerator retains `(0.2476, 67.3172, 43.4567)` and
denominator `(1, 92.6243)`. The gas law is
`k_rxn = 0.1382 exp(−37318/(R T)) [(c_H2+1e-4)^0.7554−(1e-4)^0.7554]`,
with `k_eff = k_rxn/(1+k_rxn/0.0180)`.

Six parameters are editable in the application's family editor: reduction
prefactor, activation energy and transfer limit, plus oxidation rate constant,
O2 half-saturation and pore parameter. The fitted powers, rational/Padé
coefficients and numerical offsets stay fixed in Python. Unused constants from
the old implementation, including its unused O2 transfer limit, are omitted.

The port clamps negative trial concentrations and protects the zero-carrier
denominator and square root. A cell with no Fe or Fe2O3 therefore returns zero
rates. For physical inventories the default rates agree with the branch.

## Property definitions and scope

The two fits come from `d19a65c`, unchanged at the branch tip:

- **SiO2**: new solid, 0.060084 kg/mol, reference enthalpy −910857 J/mol.
- **Al2O3 (alpha)**: 0.101961 kg/mol, reference enthalpy −1675692 J/mol;
  replaces the built-in constant Cp with the branch's quartic fit.

Both use the existing `builtin:polynomial` correlation: Cp coefficients multiply
powers of `T−298.15 K`; enthalpy is its exact integral from the reference value.
The manifests retain the branch's coefficients. Original fit data are under
`Property_Estimation/enth_hcap_data/{sio2,al2o3-alpha}` on that branch. Their
temperature spans are recorded in the manifest; no new property fit was made.

The branch's **axial heat dispersion** option and solver changes are excluded
from this transfer. They are independent engine functionality, not a kinetics
requirement. Accordingly this plugin reproduces the rate laws and properties,
not reactor results that depended on nonzero axial heat dispersion.

## Edit and share

Edit declared values in the app, or species/parameter overrides in the manifest.
After changing Python declarations, regenerate metadata in the application’s
Python environment, then package the folder:

```python
from packed_bed.plugins.check import generate_metadata
from packed_bed.plugins.storage import pack_plugin

generate_metadata("plugins/waste_iron")  # Executes the local factory.
pack_plugin("plugins/waste_iron", "dist/plugins/waste_iron.msplugin")
```

The app's **Export** action can share an edited project copy. The archive includes
this documentation, implementation and manifest; it has no plugin dependencies.
