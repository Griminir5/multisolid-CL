# Refactor baseline

Captured on 2026-07-10 from `big-refactor-deslopping` at commit `6cfd0c2`.
The worktree contained only the untracked refactor plan before this baseline was
added. The branch is already published as `origin/big-refactor-deslopping`, so it
serves as the reviewable Phase 0 checkpoint.

## Code and import metrics

- Production package: 40 Python modules and 12,011 physical lines (`wc -l`).
- Top-level AST inventory: 277 private helpers, 166 public functions, and 76
  classes.
- `import packed_bed`: about 0.01 s and 9 MB maximum resident memory.
- `import packed_bed.config`: about 0.36 s and 96 MB maximum resident memory.
- Importing `packed_bed.config` loads 25 DAETools modules and `pyUnits`; this is
  the Phase 1 import-safety defect to remove.

The executable environment is Python 3.12.3. The repository-local `pytest`
launcher does not add the working tree to `sys.path`; invoke the suite as
`.venv/bin/python -m pytest`.

## Test status

`.venv/bin/python -m pytest -q` passes: 11 tests and 2 subtests. Two DAETools
deprecation warnings are emitted for the standard-library `cgi` and `cgitb`
modules.

## Tiny inert solver check

An in-memory, nonreactive N2/Ni case with three axial cells, first-order upwind
transport, a 0.01 s horizon, and Amesos KLU completed successfully. Two cells
are not sufficient in the current model because the DAETools cell-centre domain
requires at least two intervals.

The inert model declared 43 equations in this order:

```text
species_balance_cell_0_N2
species_balance_cell_1_N2
species_balance_cell_2_N2
solid_species_balance_cell_0_Ni
solid_species_balance_cell_1_Ni
solid_species_balance_cell_2_Ni
total_concentration_closure
solid_total_concentration_closure
molar_fraction_calc
lhs_boundary_flux_N2
face_flux_1_N2
face_flux_2_N2
rhs_boundary_flux_N2
gas_component_enthalpy_N2
energy_balance_cell_0
energy_balance_cell_1
energy_balance_cell_2
total_cell_enthalpy
solid_component_enthalpy_Ni
lhs_boundary_enthalpy_flux_N2
face_enthalpy_flux_1_N2
face_enthalpy_flux_2_N2
rhs_boundary_enthalpy_flux_N2
axial_dispersion_face
ergun_face_0
ergun_face_1
ergun_face_2
ergun_face_3
gas_equation_of_state
gas_mixture_viscosity
gas_density_closure
mass_in_total_accumulation
mass_out_total_accumulation
mass_bed_total_definition
heat_in_total_accumulation
heat_out_total_accumulation
heat_loss_total_accumulation
heat_bed_total_definition
Active_inlet_flow_smooth
Active_inlet_composition_0_smooth
Active_inlet_temperature_smooth
inlet_pressure_from_flow
Active_outlet_pressure_smooth
```

## Inert reporter schema

The same case produced two reported time samples. Domain sizes below are for
one gas, one solid, three cells, and four faces.

| Variable | Value shape | Domains | Units |
| --- | --- | --- | --- |
| `J_gas_face` | `(2, 1, 4)` | gas, face | `J/(m**2 * s)` |
| `N_gas_face` | `(2, 1, 4)` | gas, face | `mol/(m**2 * s)` |
| `c_gas` | `(2, 1, 3)` | gas, cell | `mol/m**3` |
| `c_sol` | `(2, 1, 3)` | solid, cell | `mol/m**3` |
| `pres_bed` | `(2, 3)` | cell | `Pa` |
| `temp_bed` | `(2, 3)` | cell | `K` |
| `u_s` | `(2, 4)` | face | `m/s` |
| `y_gas` | `(2, 1, 3)` | gas, cell | dimensionless |
| `heat_in_total`, `heat_out_total`, `heat_loss_total`, `heat_bed_total` | `(2,)` | none | `J` |
| `mass_in_total`, `mass_out_total`, `mass_bed_total` | `(2,)` | none | `kg` |

No files were written by either inert check.
