# Model equations

This guide is an outline. It will describe the implemented model, with links to
the corresponding source and scientific references.

Planned coverage:

- Geometry, units, concentration bases, voidage, and modelling assumptions.
- Gas and solid species balances, stoichiometry, kinetics, and source terms.
- Internal-energy storage, enthalpy transport, heat loss, and axial heat dispersion.
- Pressure, flow, axial mass dispersion, and gas-property closures.
- Initial and boundary conditions, operating programs, and feed transitions.
- Spatial discretization, solver treatment, and conservation diagnostics.

Current implementation: [model.py](../packed_bed/model.py),
[initialization.py](../packed_bed/initialization.py), and
[axial_schemes.py](../packed_bed/axial_schemes.py). Scientific provenance is recorded
in the [kinetics sources](../packed_bed/kinetics/KINETICS_SOURCES.md).

[Documentation index](../README.md#documentation)
