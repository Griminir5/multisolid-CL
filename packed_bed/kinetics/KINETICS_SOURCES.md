# Bundled kinetics families

Each module owns one swappable family: its reaction definitions, symbolic
DAETools hooks, component requirements, and local regularization constants.
Run manifests record selected definitions, their parameter values, and the
built-in definition fingerprint. They also record the source Git commit when
available; packaged installations may have no Git checkout. There is no separate
kinetics version system. See [run provenance](../../docs/CLI-GUIDE.md#outputs-and-analysis).

These references document provenance. They do not assert that a family is
appropriate for a particular material or experiment.

## Copper — San Pio et al.

Source: M. A. San Pio et al., *Chemical Engineering Science* 175 (2018),
56–71, [doi:10.1016/j.ces.2017.09.044](https://doi.org/10.1016/j.ces.2017.09.044).

- `copper_sio2_san_pio` (`copper_sio2.py`) contains the two
  pseudo-homogeneous CuO → Cu2O → Cu hydrogen-reduction steps and the
  Cu + ½O2 → CuO oxidation step. SiO2 is treated as an inert support, so the
  family requires no aluminium or spinel species.
- `copper_al2o3_san_pio` (`copper_al2o3.py`) contains those support-independent
  reductions, Cu → CuO oxidation, and the Al2O3-specific spinel reduction
  and oxidation steps.

Both families use the Cu oxidation rate law `r = k(T) c_Cu sqrt(p_O2)` with
oxygen partial pressure in bar, but the paper fits different Arrhenius
parameters: Table 5 gives `A = 0.193 s^-1 bar^-1/2` and `Ea = 180 J/mol` for
SiO2, while Table 9 gives `A = 0.854 s^-1 bar^-1/2` and `Ea = 830 J/mol` for
Al2O3. The SiO2 oxidation hook copies the Al2O3 rate expression and smooth
positive-pressure regularization, using the SiO2-specific parameter defaults.

The source states that CuO-to-Cu reduction is independent of the support and
extends that base model with CuAl2O4/CuAlO2 chemistry for the Al2O3 carrier.
The zero-order hydrogen reductions retain a local smooth gas-availability gate.

Both copper families evaluate the smooth positive pressure
`(p + sqrt(p*p + 4*epsilon*epsilon))/2`, with `epsilon = 1e-12 bar`,
using an algebraically equivalent rationalized form in `regularization.py`.
This preserves the small positive tail and its derivative for negative trial
pressures without subtracting nearly equal numbers. The rate laws and
regularization scale are unchanged.

## Nickel redox — Medrano form

`nickel_medrano` (`nickel_medrano.py`) follows the Medrano-style form documented
in Andrew Wright, *Chemical Looping Reactor Modelling – 2D*, with context from
[doi:10.1016/j.apenergy.2015.08.078](https://doi.org/10.1016/j.apenergy.2015.08.078).
The rational, solver-oriented implementation is canonical; the older duplicate
form is not retained.

The chemical rate constant includes the total-pressure correction used by
Argyris et al., *Sustainable Energy & Fuels* **6** (2022), 2755–2770,
[doi:10.1039/D2SE00351A](https://doi.org/10.1039/D2SE00351A),
Eq. (14) and Table 2:

$$
k_j(T,P)=k_{0,j}\exp\left(-\frac{E_{a,j}}{RT}\right)
\left(\frac{P}{P_{\mathrm{ref}}}\right)^{-q_j},
\qquad P_{\mathrm{ref}}=1\ \mathrm{bar}=10^5\ \mathrm{Pa}.
$$

`P` is the **local absolute total gas pressure**, recovered from `C_total R T`.
The reference pressure is fixed at 1 bar. This factor modifies the chemical
rate constant inside the shrinking-core resistance expression; the ash
diffusivity is unchanged. Gas concentration retains its existing reaction-order
dependence. At 1 bar the additional factor is exactly one. Each reaction exposes its own dimensionless, nonnegative exponent.

## Nickel-catalysed reforming

- `reforming_xu_froment` (`reforming_xu_froment.py`) implements the three
  Xu–Froment reactions from
  [doi:10.1002/aic.690350109](https://doi.org/10.1002/aic.690350109).
- `reforming_numaguchi` (`reforming_numaguchi.py`) implements the alternative
  Numaguchi–Kikuchi reforming and water-gas-shift form documented in Andrew
  Wright's technical report.

These are alternative published mechanisms, not compatibility aliases.

## Iron redox — He et al.

`iron_he` (`iron_he.py`) contains the He et al. reduction and oxidation family
from *Energy Conversion and Management* 293 (2023), 117525. Its symbolic
runtime expressions and local constants are the canonical implementation.

[Model equations](../../docs/EQUATIONS.md) · [Documentation index](../../README.md#documentation)
