# Bundled kinetics families

Each module owns one swappable family: its reaction definitions, symbolic
DAETools hooks, component requirements, and local regularization constants.
The family name and Git commit recorded in a run manifest identify the selected
implementation; there is no separate kinetics version system.

These references document provenance. They do not assert that a family is
appropriate for a particular material or experiment.

## Copper — San Pio et al.

Source: M. A. San Pio et al., *Chemical Engineering Science* 175 (2018),
56–71, [doi:10.1016/j.ces.2017.09.044](https://doi.org/10.1016/j.ces.2017.09.044).

- `copper_sio2_san_pio` (`copper_sio2.py`) contains the two
  pseudo-homogeneous CuO → Cu2O → Cu hydrogen-reduction steps. SiO2 is treated
  as an inert support, so the family requires no aluminium or spinel species.
- `copper_al2o3_san_pio` (`copper_al2o3.py`) contains those support-independent
  reductions plus the Al2O3-specific spinel reduction and oxidation steps.

The source states that CuO-to-Cu reduction is independent of the support and
extends that base model with CuAl2O4/CuAlO2 chemistry for the Al2O3 carrier.
The zero-order hydrogen reductions retain a local smooth gas-availability gate.

## Nickel redox — Medrano form

`nickel_medrano` (`nickel_medrano.py`) follows the Medrano-style form documented
in Andrew Wright, *Chemical Looping Reactor Modelling – 2D*, with context from
[doi:10.1016/j.apenergy.2015.08.078](https://doi.org/10.1016/j.apenergy.2015.08.078).
The rational, solver-oriented implementation is canonical; the older duplicate
form is not retained.

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
