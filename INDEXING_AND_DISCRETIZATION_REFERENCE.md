# Packed-Bed Indexing and Discretization Reference

This note summarizes the implementation lessons from getting `CLBed_mass.py` into a DAETOOLS-safe form.

The main goal is to document:

- the cell/face indexing convention,
- which DAETOOLS expression patterns assembled cleanly,
- which patterns caused access violations during `DeclareEquations()`,
- how to extend the current first-order face flux to higher-order reconstructions such as WENO or flux-limited schemes.

This is an implementation reference, not a final physics reference.

## 1. Grid and indexing convention

The current packed-bed model uses a 1D finite-volume layout:

- cell-centered state variables:
  - `c_gas(i, x_center)`
  - `ct_gas(x_center)`
  - `y_gas(i, x_center)`
- face variables:
  - `N_gas_face(i, x_face)`
  - `Dax(x_face)`
  - `u_s(x_face)`

The axial grid is stored as:

- `x_centers`: cell-center coordinates
- `x_faces`: cell-face coordinates

The required relation is:

- `Nf = Nc + 1`

with:

- left boundary face index: `0`
- right boundary face index: `Nf - 1`
- interior face indices: `1 ... Nf - 2`
- cell indices: `0 ... Nc - 1`

For a uniform grid with `n_cells`:

- `Nc = n_cells`
- `Nf = n_cells + 1`

The current helper methods build the domains so that `NumberOfPoints` matches the actual number of points, not the number of intervals:

```python
self.x_faces.CreateStructuredGrid(face_locations.size - 1, 0, 1)
self.x_centers.CreateStructuredGrid(center_locations.size - 1, 0, 1)
```

This is consistent with DAETOOLS examples.

## 2. Current governing structure in `CLBed_mass.py`

The current model is:

- species mass balances only,
- temporary plug-flow velocity closure,
- temporary fixed `c_in` used in `u_s = F_in / (c_in A)`,
- no EOS yet,
- no momentum equation yet,
- no solids yet except `d_p` in the dispersion placeholder.

The current equations are:

1. Cell total concentration closure:

```text
ct_gas(x) = sum_i c_gas(i, x)
```

2. Cell mole fraction definition:

```text
y_gas(i, x) * ct_gas(x) = c_gas(i, x)
```

3. Left boundary face flux:

```text
N_i,0 = y_in,i * F_in / A
```

4. Placeholder axial dispersion:

```text
Dax = 0.5 * |u_s| * d_p
```

5. Interior face flux:

```text
N_i,f = u_plus * c_i,L + u_minus * c_i,R - Dax * ct_face * (y_i,R - y_i,L) / dx
```

with:

- `u_plus = max(u_s, 0)`
- `u_minus = min(u_s, 0)`
- `ct_face = 0.5 * (ct_L + ct_R)`

6. Cell species balance:

```text
dc_i/dt + (N_i,f+1/2 - N_i,f-1/2) / dx = 0
```

7. Right boundary outflow:

```text
N_i,L = u_s,L * c_i,last
```

8. Temporary plug-flow closure:

```text
u_s = F_in / (c_in * A)
```

where `A = pi * R_bed^2`.

## 3. DAETOOLS-safe patterns that worked

These patterns assembled cleanly in a minimal smoke test.

### 3.1 Use `DistributeOnDomain(...)` for equation indexing

For equations that naturally live over an entire domain, prefer one distributed equation over many Python-created scalar equations.

Safe example:

```python
eq = self.CreateEquation("total_concentration_closure")
idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, 'x')
eq.Residual = self.ct_gas(idx_cell) - Sum(self.c_gas.array('*', idx_cell))
```

Safe example:

```python
eq = self.CreateEquation("molar_fraction_calc")
idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, 'i')
idx_cell = eq.DistributeOnDomain(self.x_centers, eClosedClosed, 'x')
eq.Residual = self.y_gas(idx_gas, idx_cell) * self.ct_gas(idx_cell) - self.c_gas(idx_gas, idx_cell)
```

### 3.2 Use `Sum(var.array('*', fixed_index))` for component sums

For sums over species at a fixed cell, this worked well:

```python
Sum(self.c_gas.array('*', cell_index))
```

This was more robust than building a manual Python sum over `self.c_gas(gas_index, cell_index)`.


### 3.3 `SetValues(...)` on distributed parameters should use a NumPy array

For example:

```python
self.y_in.SetValues(np.array([0.5, 0.5]))
```

Using a plain Python list caused:

```text
'list' object has no attribute 'shape'
```

## 4. Patterns that caused DAETOOLS failures

The main failure mode was a Windows access violation during `DeclareEquations()`.

The most fragile patterns were:

- manually looping over both gas and cell indices in Python and calling `self.c_gas(i, j)` repeatedly,
- using scalar inlet quantities as DAETOOLS variables in residual expressions,
- mixing distributed indices and fixed indices inside nonlinear stencil expressions without care,
- using separately indexed algebraic helper variables in selector-weighted face reconstructions when the same quantity could be reconstructed directly from the primary variables.

In practice, the following kinds of patterns were problematic in this model:

- raw Python loops of the form:

```python
for idx_cell in range(Nc):
    for idx_gas in range(Ng):
        expr = self.c_gas(idx_gas, idx_cell)
```

## 5. Current interior-face pattern

The current interior-face equation is distributed over:

- all gas species,
- all interior faces.

Because DAETOOLS does not provide a simple native "neighbor cell" index operator, the face stencil is assembled using selector expressions.

Current pattern:

```python
eq = self.CreateEquation("face_flux")
idx_gas = eq.DistributeOnDomain(self.N_gas, eClosedClosed, 'i')
idx_face = eq.DistributeOnDomain(self.x_faces, eOpenOpen, 'x_f')

cL = Constant(0 * kmol / m**3)
cR = Constant(0 * kmol / m**3)
yL = Constant(0 * dimless)
yR = Constant(0 * dimless)
ct_face = Constant(0 * kmol / m**3)
inv_dx = Constant(0 * m**(-1))

for face_index in range(1, Nf-1):
    cell_L = face_index - 1
    cell_R = face_index
    dx = center_coords[cell_R] - center_coords[cell_L]
    selector = 1 - (idx_face() - face_index) / (idx_face() - face_index + 1E-15)
    ct_L = Sum(self.c_gas.array('*', cell_L))
    ct_R = Sum(self.c_gas.array('*', cell_R))

    cL += selector * self.c_gas(idx_gas, cell_L)
    cR += selector * self.c_gas(idx_gas, cell_R)
    yL += selector * self.y_gas(idx_gas, cell_L)
    yR += selector * self.y_gas(idx_gas, cell_R)
    ct_face += selector * 0.5 * (ct_L + ct_R)
    inv_dx += selector / dx
```

Then the flux is written once:

```python
uplus = Max(self.u_s(idx_face), 0)
uminus = Min(self.u_s(idx_face), 0)
eq.Residual = self.N_gas_face(idx_gas, idx_face) - uplus*cL - uminus*cR + self.Dax(idx_face)*ct_face*(yR-yL)*inv_dx
```

This pattern is not the most elegant one, but it assembled safely.

## 6. How to extend this to higher-order schemes

When moving to high-order reconstruction or WENO, the safest path is:

1. Keep the equation distributed over `idx_gas` and `idx_face`.
2. Build fixed-cell stencil expressions inside a Python loop over `face_index`.
3. Use selector expressions to activate the correct stencil for the current distributed face.
4. Combine those selector-weighted contributions into one final left and right reconstructed state.

That means the high-order logic should still follow this structure:

- build `c_minus(face, gas)` from left-biased stencil values,
- build `c_plus(face, gas)` from right-biased stencil values,
- compute the numerical flux using `uplus` and `uminus`,
- only then take the divergence in the cell balances.

For example, conceptually:

```text
For each interior face f:
    build c_minus(f) from cells [f-2, f-1, f, ...]
    build c_plus(f)  from cells [f+1, f, f-1, ...]
    N_f = u_plus * c_minus + u_minus * c_plus + diffusive_term
```

In DAETOOLS, the implementation should still be:

- loop over `face_index` in Python,
- construct the fixed-index stencil expressions,
- multiply by a selector for that `face_index`,
- accumulate into a symbolic face expression.

## 7. WENO and flux-limiter guidance

When you move past the current low-order scheme, these points will matter.

### 7.1 Reconstruct concentrations, not flux divergence directly

For a finite-volume scheme, reconstruct left and right face states first:

- `c_minus`
- `c_plus`

Then define the face flux.

This keeps the face equation modular and makes it easier to change:

- upwind,
- MUSCL,
- TVD limiter,
- WENO,
- hybrid convection-diffusion forms.

### 7.2 Keep boundary closures separate from interior closures

Interior WENO stencils need more cells than the current first-order scheme.

So you will need:

- interior face reconstruction,
- near-boundary reconstruction,
- boundary-condition closure.

Do not try to force one stencil to cover every face.

### 7.3 Expect ghost states or one-sided stencils

For a WENO-type method you will likely need one of:

- ghost cells,
- one-sided reconstructions near the inlet and outlet,
- explicit low-order fallback on the first few interior faces.

Given the current DAETOOLS setup, one-sided selector-weighted formulas are likely easier than trying to add a separate ghost-cell domain.

### 7.4 Reverse flow will need explicit treatment

The current upwind split allows negative `u_s`, but the boundary conditions are still essentially written for left-to-right flow.

Once you care about reverse flow, the model should switch:

- which boundary is inflow,
- which composition is imposed,
- which outlet uses simple convective outflow.

That becomes more important once a high-order convective scheme is introduced.

### 7.5 Flux limiters may require extra dimensionless guard terms

Limiter formulas often use ratios like:

```text
r = (phi_j - phi_j-1) / (phi_j+1 - phi_j + eps)
```

In DAETOOLS, keep a small `eps` in the denominator to avoid singular expressions. If a limiter formula becomes awkward for unit checking, it may be necessary to disable unit checking on that specific equation only.

## 8. Recommended implementation rules going forward

- Use `daeParameter` for true inputs.
- Use `daeVariable` for states or algebraic unknowns that are solved by the DAE.
- Prefer `eq.DistributeOnDomain(...)` over nested Python loops for equations that live over full domains.
- Prefer `Sum(var.array('*', fixed_index))` over manual component summation.
- Keep face equations on the face domain and cell balances on the cell domain.
- Build complex stencils by selector-weighted accumulation over fixed integer stencil locations.
- Keep boundary equations separate from interior equations.
- Smoke-test `DeclareEquations()` before attempting a full simulation.

## 9. Minimal smoke-test checklist

Before running a transient solve, verify:

1. `gas_species` is set correctly.
2. `self.N_gas.CreateArray(len(self.gas_species))` has happened.
3. `L_bed`, `R_bed`, `d_p`, `c_in` are set.
4. `F_in` and `y_in` are set, or an `InletProgram` is attached.
5. The grid has been created with `SetUniformAxialGrid(...)` or `SetAxialGridFromFaces(...)`.
6. `DeclareEquations()` completes without error.

Minimal assembly test:

```python
model = CLBed_mass('test', ['N2', 'CO2'])
model.pi.SetValue(3.141592653589793)
model.F_in.SetValue(1.0 * kmol / s)
model.L_bed.SetValue(1.0 * m)
model.R_bed.SetValue(0.1 * m)
model.d_p.SetValue(0.01 * m)
model.c_in.SetValue(0.025 * kmol / m**3)
model.y_in.SetValues(np.array([0.5, 0.5]))
model.SetUniformAxialGrid(3)
model.DeclareEquations()
```

If this works, the next step is testing initialization and one timestep, not immediately jumping to a long run.

## 10. Current known modeling approximations

These are intentional temporary simplifications, not indexing bugs:

- `c_in` is a temporary closure parameter until EOS and momentum equations are added.
- `Dax = 0.5 * |u_s| * d_p` is only a placeholder.
- voidage is not yet represented.
- the right boundary currently assumes simple convective outflow.
- reverse flow is not fully well-posed at the boundary level.

Those issues should be revisited later, but they are separate from the DAETOOLS indexing and assembly issues documented here.
