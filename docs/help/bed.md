# Bed tab

Describe the reactor geometry, heat exchange, and initial solid distribution.
The preview shows the sampled bed profiles used by the model.

## Bed settings

| Field or option | What it does |
| --- | --- |
| Radius (m) | Internal bed radius; sets cross-sectional area and contributes to bed volume. |
| Length (m) | Full axial bed extent. The final zone always ends here. |
| Solid density basis — Bed volume | Interprets entered loadings per unit total bed volume. |
| Solid density basis — Solid volume | Interprets entered loadings per unit solid volume, using zone voidages to obtain inventories per bed volume. |
| Gas-occupied bed fraction — Interparticle only | Includes only the spaces between particles in gas storage. |
| Gas-occupied bed fraction — Interparticle + intraparticle | Includes spaces between particles and the pores within them in gas storage. |
| Ambient temperature (K) | Temperature outside the bed used for wall heat exchange. |
| Heat transfer (W/m²/K) | Overall wall heat-transfer coefficient. Zero disables wall exchange. |
| Heat dispersion (W/m/K) | Effective axial heat spreading. Zero disables it; dispersive heat does not cross the inlet or outlet boundaries. |
| Reversible flow | Enables signed interior transport when local flow reverses. It does not turn the outlet into a second prescribed feed. |
| Keep positions | In the length-change dialog, retains internal zone boundaries at their current positions. |
| Scale proportionally | In the length-change dialog, scales zone boundaries with the changed length. |

## Fill the material zones

| Field or option | What it does |
| --- | --- |
| Concentration units — mol/m³ / kg/m³ | Interprets loadings as molar or mass concentration on the selected density basis. Changing units or basis does not convert entered numbers. |
| Start (m) / End (m) | Zone boundaries. The first start is fixed at zero and the last end at the bed length. Zones must meet without gaps or overlaps. |
| Interparticle voidage | Fraction of the bed volume between particles. |
| Intraparticle voidage | Pore fraction within the particles. |
| Particle diameter (m) | Particle size used in transport and pressure-drop calculations. |
| Each solid-species column | Initial loading of that selected solid in this zone, using the chosen units and basis. Species come from Chemistry. |
| wt% beneath a loading | Calculated share of the zone's initial solid mass; it is not a second editable input. |

Inspect the preview after changing zones or the number of cells in General.
The selected solid species come from **Chemistry**. Bed geometry also affects
the conversion between GHSV and molar flow in **Program**.

Read more: [Bed settings](../GUI-GUIDE.md#bed-geometry-and-material-zones) ·
[Volume and concentration bases](../EQUATIONS.md#volume-fractions-and-concentration-bases) ·
[Boundary conditions](../EQUATIONS.md#boundary-conditions).
