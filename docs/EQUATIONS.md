# Model equations

This guide describes the reactor model implemented by MultiSolid-CL. The equations
below use the same concentration bases, face fluxes, boundary conditions, and
source-term conventions as [PackedBedModel](../packed_bed/model.py). Standard and
Compiled execution start from this model; their numerical treatment is described
in [Time integration](#time-integration).

The core balances apply to every selected chemistry. Individual kinetic laws and
property datasets are supplied by built-in definitions or plugins. Their validity
ranges and provenance remain part of those definitions; see
[Reaction sources](#reaction-sources).

## Contents

- [Scope and assumptions](#scope-and-assumptions)
- [Notation and grid](#notation-and-grid)
- [Volume fractions and concentration bases](#volume-fractions-and-concentration-bases)
- [Species balances and reaction sources](#species-balances-and-reaction-sources)
- [Thermodynamic and transport properties](#thermodynamic-and-transport-properties)
- [Gas transport and pressure drop](#gas-transport-and-pressure-drop)
- [Energy balance](#energy-balance)
- [Boundary conditions](#boundary-conditions)
- [Operating programs](#operating-programs)
- [Initial conditions](#initial-conditions)
- [Spatial reconstruction](#spatial-reconstruction)
- [Time integration](#time-integration)
- [Conservation diagnostics and reported quantities](#conservation-diagnostics-and-reported-quantities)
- [Configuration and implementation map](#configuration-and-implementation-map)

## Scope and assumptions

The reactor is a fixed cylindrical bed represented in one axial dimension.
Cross-sectional variations are unresolved. The implemented assumptions are:

- Gas is an ideal mixture. Each cell has one pressure, gas composition, and
  temperature. Gas and solids share that temperature: local thermal equilibrium
  is assumed, with no separate gas-to-solid heat-transfer equation.
- Solids are stationary. Their inventories change through reactions, with no
  axial solid transport. Gas moves by convection and axial dispersion.
- Bed radius, bed length, interparticle and intraparticle voidages, and particle
  diameters are prescribed and remain fixed during a run. Reaction-induced
  changes in solid inventory do not change voidage or geometry.
- The gas storage volume can include particle pores, but there is no separate
  pore-gas state or resolved intraparticle diffusion problem. Particle effects
  such as an effectiveness factor must be represented by the selected kinetics.
- The pressure gradient follows an algebraic Ergun closure. There is no momentum
  accumulation equation, gravity term, or independent prescribed interior flow.
- Species enthalpies depend on temperature and include their reference enthalpies.
  Stored energy is internal energy, with ideal-gas pressure-volume correction
  and negligible solid pressure-volume contribution. Kinetic and gravitational
  potential energy are omitted.
- Heat exchange with the environment is represented by a constant overall wall
  coefficient and ambient temperature. An optional constant axial heat-dispersion
  coefficient acts on the shared temperature. There is no separate wall energy
  state, resolved radial conduction, or radiation term.

These assumptions define the implemented model. In particular, selecting a
reaction family does not add a separate particle, wall, or gas-temperature model.

## Notation and grid

Use SI units unless a particular kinetic correlation explicitly converts them.
The gas constant in the implementation is
$R = 8.31446\ \mathrm{J\,mol^{-1}\,K^{-1}}$.

| Symbol | Meaning | Unit or basis |
| --- | --- | --- |
| $L$, $R_b$, $A=\pi R_b^2$ | Bed length, radius, cross-sectional area | m, m, m² |
| $j$, $i$, $a$, $k$ | Cell, gas species, solid species, reaction indices | — |
| $e_b$, $e_p$ | Interparticle voidage; pore fraction within particles | Dimensionless |
| $\varepsilon_g$, $\varepsilon_s$ | Modelled gas volume and solid skeleton volume per bed volume | Dimensionless |
| $c_i$, $s_a$ | Gas and solid concentrations stored by the model | mol/m³ **bed** |
| $C_i=c_i/\varepsilon_g$ | Gas-phase concentration | mol/m³ **gas** |
| $c_t=\sum_i c_i$, $C_t=c_t/\varepsilon_g$ | Total gas concentrations on the two bases | mol/m³ bed; mol/m³ gas |
| $y_i=c_i/c_t$ | Gas mole fraction | Dimensionless |
| $r_k$, $S_i^g$, $S_a^s$ | Reaction-extent rate and species source terms | mol/(m³ bed·s) |
| $u_f$ | Superficial gas velocity at a face | m/s, based on total area $A$ |
| $N_{i,f}$ | Gas species molar flux at a face | mol/(m² bed·s) |
| $h_i^g$, $h_a^s$ | Species molar enthalpies | J/mol |
| $J_{i,f}$, $q_f$ | Species enthalpy flux; axial dispersive heat flux | W/m² bed |
| $E_j$ | Cell internal energy per bed volume | J/m³ bed |
| $P$, $T$, $\rho_g$, $\mu_g$ | Gas pressure, shared temperature, gas density, mixture viscosity | Pa, K, kg/m³ gas, Pa·s |
| $D_f$, $d_{p,f}$ | Axial mass-dispersion coefficient; particle diameter | m²/s, m |
| $U$, $T_\infty$, $\lambda_{ax}$ | Wall heat-transfer coefficient, ambient temperature, axial heat dispersion | W/(m²·K), K, W/(m·K) |

Positive flux is from the inlet at $x=0$ to the outlet at $x=L$.
Mathematical cell indices here run from $1$ to $n$; Python array indices run from
$0$ to $n-1$.

The configured grid is uniform, with at least three cells:

$$
\Delta x=\frac{L}{n},\qquad
x_j=\left(j-\frac12\right)\Delta x,\qquad
x_{j-1/2}=(j-1)\Delta x.
$$

Concentrations, temperature, pressure, and gas properties live at cell centres.
Velocities, particle diameters, dispersion coefficients, and transport fluxes
live at faces. Define the conservative cell divergence of a face quantity as

$$
\mathcal D_j(F)=\frac{F_{j+1/2}-F_{j-1/2}}{\Delta x_j},
\qquad \Delta x_j=x_{j+1/2}-x_{j-1/2}.
$$

The balances retain this cell-width form, but the supplied grid builder and
higher-order reconstruction formulas use a uniform grid. The distance used in
an interior face gradient is $x_{j+1}-x_j$; the inlet and outlet pressure-drop
segments are half a cell wide.

## Volume fractions and concentration bases

The `model.gas_voidage_mode` setting defines gas storage:

$$
\varepsilon_g=
\begin{cases}
e_b+(1-e_b)e_p, & \texttt{bed\_and\_particle}\ \text{(default)},\\
e_b, & \texttt{bed\_only}.
\end{cases}
\qquad
\varepsilon_s=(1-e_b)(1-e_p).
$$

`bed_only` excludes pore gas from storage while retaining the same solid skeleton
fraction. Consequently $\varepsilon_g+\varepsilon_s$ need not equal one in that
mode: the excluded pore volume is not reassigned to solids. The chosen gas
fraction is used in initialization, the equation of state, transport
concentrations, and energy storage. Ergun resistance always uses $e_b$.

Solids are specified in contiguous zones covering the bed. Zone voidages and
initial solid values are sampled at cell centres; a narrow zone containing no
cell centre does not contribute a cell inventory, and it is the responsibility of the user to choose a suitable resolution. Particle diameters are sampled
at faces. A face exactly on a zone boundary uses the arithmetic mean of the two
adjacent zone diameters. These rules are implemented in
[solid_profiles.py](../packed_bed/solid_profiles.py).

Let $v_a$ be an entered solid value and $M_a$ its molecular weight in kg/mol.
The conversion to the model state is

$$
s_{a,0}=b\begin{cases}
v_a, & \texttt{mol\_per\_m3},\\
v_a/M_a, & \texttt{kg\_per\_m3},
\end{cases}
\qquad
b=\begin{cases}
1, & \texttt{basis: bed},\\
\varepsilon_s, & \texttt{basis: solid}.
\end{cases}
$$

Thus a value per solid volume is multiplied by the solid skeleton fraction;
mass inputs are divided by molecular weight first. Outputs and reaction source
terms continue to use mol/m³ bed regardless of the input basis. No evolving
solid-density or volume-filling closure is imposed.

## Species balances and reaction sources

### Gas and solid balances

For each gas species and cell,

$$
\frac{d c_{i,j}}{dt}+\mathcal D_j(N_i)=S^g_{i,j}.
$$

For each stationary solid species,

$$
\frac{d s_{a,j}}{dt}=S^s_{a,j}.
$$

There is no extra voidage multiplier on these accumulation or source terms:
$c_i$, $s_a$, and both sources already use total bed volume. The total
concentrations and mole fractions are algebraic closures:

$$
c_{t,j}=\sum_i c_{i,j},\qquad
s_{t,j}=\sum_a s_{a,j},\qquad
y_{i,j}c_{t,j}=c_{i,j}.
$$

Gas mole fractions sum to one when the total gas concentration is nonzero and
the closures are satisfied. Total solid concentration is not held constant.

### Stoichiometric source terms

Each selected reaction has one extent rate $r_{k,j}$. Its signed stoichiometric
coefficients $\nu^g_{ik}$ and $\nu^s_{ak}$ are negative for reactants and positive
for products:

$$
S^g_{i,j}=\sum_k\nu^g_{ik}r_{k,j},\qquad
S^s_{a,j}=\sum_k\nu^s_{ak}r_{k,j}.
$$

The reaction network builds these matrices from the selected definitions in
[reactions.py](../packed_bed/reactions.py). Sources from all selected reactions
add; selecting two mechanisms for the same conversion does not make one replace
the other. With no reactions, all source terms are zero.

For example, a reaction written as
$\mathrm{Fe_2O_3+3H_2\rightarrow2Fe+3H_2O}$ gives
$S_{H_2}^g=-3r$, $S_{H_2O}^g=3r$, $S_{Fe_2O_3}^s=-r$, and $S_{Fe}^s=2r$.
The scalar $r$ is an extent rate; the stoichiometric coefficients are applied
once by the source matrices. They do not determine kinetic reaction orders.

A kinetic hook receives the cell state and returns a symbolic rate in
mol/(m³ bed·s). A correlation defined per gas volume, catalyst mass, or another
basis must perform its own conversion to that output basis. The core model does
not automatically multiply hook results by voidage or catalyst loading. Partial
pressures follow $p_i=y_iP$; correlations using bar or other units must convert
from the model's pascals. Catalyst roles can affect a rate without having a
stoichiometric source term.

Clamping, gas-availability gates, conversion limits, and regularization constants
belong to individual kinetic implementations. There is no universal reaction-rate
clip or limiter that guarantees nonnegative concentrations. Transient solver
trial states can leave a correlation's physical domain.

### Reaction sources

| Family | Implementation |
| --- | --- |
| Copper on silica, San Pio form | [copper_sio2.py](../packed_bed/kinetics/copper_sio2.py) |
| Copper on alumina, including spinel chemistry | [copper_al2o3.py](../packed_bed/kinetics/copper_al2o3.py) |
| Nickel redox, Medrano form | [nickel_medrano.py](../packed_bed/kinetics/nickel_medrano.py) |
| Nickel-catalysed reforming, Xu–Froment form | [reforming_xu_froment.py](../packed_bed/kinetics/reforming_xu_froment.py) |
| Nickel-catalysed reforming, Numaguchi–Kikuchi form | [reforming_numaguchi.py](../packed_bed/kinetics/reforming_numaguchi.py) |
| Iron redox, He form | [iron_he.py](../packed_bed/kinetics/iron_he.py) |

The [kinetics source notes](../packed_bed/kinetics/KINETICS_SOURCES.md) identify
scientific references and implementation-specific choices. The additional
[waste iron plugin](../plugins/waste_iron/README.md) documents its own rate laws
and property fits. Plugin and built-in variants use the same core balances;
changing their rates or properties can change both composition and temperature.

## Thermodynamic and transport properties

### Equation of state and density

The ideal-gas closure is

$$
P_j\varepsilon_{g,j}=c_{t,j}RT_j,
\qquad C_{t,j}=\frac{P_j}{RT_j}.
$$

With gas molecular weights $M_i$ in kg/mol,

$$
M_{mix,j}=\sum_i y_{i,j}M_i,\qquad
\rho_{g,j}=\frac{P_jM_{mix,j}}{RT_j}
          =\sum_i C_{i,j}M_i.
$$

This is density per gas volume, as needed by Ergun. Multiplying it by
$\varepsilon_g$ gives gas mass per bed volume.

### Species enthalpy and heat capacity

Every selected gas or solid species supplies $h(T)$. Consistent heat capacity is
$C_p(T)=dh/dT$. Reference enthalpies must use a consistent chemical reference
because reaction energy is obtained from these species enthalpies.

For `PolynomialHeatCapacity`, with coefficients $a_m$ and
$\theta=T-T_{ref}$,

$$
C_p(T)=\sum_{m=0}^{p}a_m\theta^m,\qquad
h(T)=h_{ref}+\sum_{m=0}^{p}\frac{a_m}{m+1}\theta^{m+1}.
$$

The coefficients are in ascending powers of $T-T_{ref}$, not powers of absolute
$T$. The parameter `h_form_ref` supplies $h_{ref}$ in J/mol.

For `ShomateHeatCapacity`, let $T_*=1000\ \mathrm K$,
$\tau=T/T_*$, and $\tau_{ref}=T_{ref}/T_*$:

$$
C_p(T)=a_0+a_1\tau+a_2\tau^2+a_3\tau^3+a_4\tau^{-2},
$$

$$
\begin{aligned}
h(T)=h_{ref}+T_*\bigg[&a_0(\tau-\tau_{ref})
+\frac{a_1}{2}(\tau^2-\tau_{ref}^2)
+\frac{a_2}{3}(\tau^3-\tau_{ref}^3)\\
&+\frac{a_3}{4}(\tau^4-\tau_{ref}^4)
-a_4(\tau^{-1}-\tau_{ref}^{-1})\bigg].
\end{aligned}
$$

This implementation anchors the integrated heat capacity at the explicitly
supplied reference enthalpy; it does not take a separate pair of Shomate
enthalpy-offset coefficients. Custom correlations supply numerical and symbolic
enthalpy/heat-capacity functions. Scientific validity and agreement between
those functions must be established for the chosen temperature range.

### Viscosity

The built-in pure-species `QuadraticViscosity` correlation is

$$
\mu_i(T)=a_{0,i}+a_{1,i}(T-T_{ref,i})+a_{2,i}(T-T_{ref,i})^2.
$$

Mixture viscosity uses Wilke's rule:

$$
\mu_g=\sum_i\frac{y_i\mu_i}{\sum_\ell y_\ell\phi_{i\ell}},
\qquad
\phi_{i\ell}=
\frac{\left[1+\sqrt{\mu_i/\mu_\ell}(M_\ell/M_i)^{1/4}\right]^2}
{\sqrt{8(1+M_i/M_\ell)}}.
$$

The diagonal factor is exactly one. The implementation does not divide by mole
fractions, so absent species with zero mole fraction are supported when the
property values and remaining mixture are valid. Initialization and the evolving
model use the same function. See [properties.py](../packed_bed/properties.py) and
the [IDAES description of Wilke's rule](https://idaes-pse.readthedocs.io/en/stable/explanations/components/property_package/general/transport_properties/viscosity_wilke.html).

## Gas transport and pressure drop

### Interior species fluxes

At an interior face $f=j+1/2$,

$$
D_f=\frac12\lvert u_f\rvert d_{p,f},\qquad
\overline C_{t,f}=\frac12\left(
\frac{c_{t,j}}{\varepsilon_{g,j}}+
\frac{c_{t,j+1}}{\varepsilon_{g,j+1}}\right),
$$

$$
N_{i,f}=\mathcal A_f(u_f,C_i)
-D_f\overline C_{t,f}\frac{y_{i,j+1}-y_{i,j}}{x_{j+1}-x_j}.
$$

Here $\mathcal A_f$ is the scheme-dependent convective transport operator defined
in [Spatial reconstruction](#spatial-reconstruction). The dispersed gradient is
mole fraction, multiplied by the arithmetic face mean of total gas-phase
concentration. It is not the gradient of concentration per bed volume.

The mass-dispersion coefficient is common to the gas species and contains no
separate molecular-diffusion term. It vanishes at zero velocity. The dispersive
species fluxes sum to zero when both neighbouring compositions sum to one.
The boundary species fluxes are prescribed separately below; the interior
formula is not evaluated across an external ghost cell.

### Ergun closure

Define the drag function

$$
\mathcal E_f(u)=
\frac{150\mu_f(1-e_{b,f})^2}{e_{b,f}^3d_{p,f}^2}u
+\frac{1.75\rho_f(1-e_{b,f})}{e_{b,f}^3d_{p,f}}\lvert u\rvert u.
$$

For each interior face,

$$
\frac{P_j-P_{j+1}}{x_{j+1}-x_j}=\mathcal E_f(u_f).
$$

Viscosity, density, and interparticle voidage at an interior face are arithmetic
means of the adjacent cell values. Particle diameter follows the face-sampling
rule above. Boundary drag uses the adjacent cell's viscosity, density, and
interparticle voidage.

The signed $\lvert u\rvert u$ term permits either velocity sign. Velocity and
pressure are solved together with species and energy accumulation; total molar
flow need not remain equal to the inlet flow along a reacting, transient bed.
The `interior_flow_mode` setting changes transport reconstruction, not this
pressure-drop equation.

## Energy balance

### Stored internal energy

The differential energy state is

$$
E_j=\sum_i c_{i,j}h_i^g(T_j)
   +\sum_a s_{a,j}h_a^s(T_j)
   -\varepsilon_{g,j}P_j.
$$

The subtraction is the ideal-gas pressure-volume contribution per bed volume.
Using the equation of state, the same definition is

$$
E_j=\sum_i c_{i,j}\bigl[h_i^g(T_j)-RT_j\bigr]
   +\sum_a s_{a,j}h_a^s(T_j).
$$

Temperature is recovered through this algebraic relation and the species
enthalpy closures. It is shared by gas and solids. Holding the gas and solid concentrations fixed,
the derivative of this stored energy with respect to temperature is
$\sum_i c_i(C_{p,i}^g-R)+\sum_a s_aC_{p,a}^s$.

### Enthalpy transport and axial heat dispersion

At an interior face, the enthalpy flux of gas species $i$ is

$$
J_{i,f}=\mathcal A_f(N_{i,f},h_i^g),\qquad H_f=\sum_iJ_{i,f}.
$$

This uses the complete species molar flux, including its dispersive part.
The `heat_scheme` reconstructs each species enthalpy. In reversible mode, its
upwind direction follows the sign of that species flux, which need not be the
same as the sign of superficial velocity.

The additional axial heat flux is

$$
q_f=-\lambda_{ax}\frac{T_{j+1}-T_j}{x_{j+1}-x_j}
\quad\text{at interior faces},\qquad q_{1/2}=q_{n+1/2}=0.
$$

The constant `model.axial_heat_dispersion_w_per_m_k` is nonnegative and defaults
to zero. The coefficient and heat flux use the total bed cross section; no
additional gas or solid volume fraction multiplies this expression. This term
can conduct heat at zero flow and is independent of the velocity-dependent
mass-dispersion coefficient.

### Wall exchange and cell balance

The volumetric heat loss to the surroundings is

$$
Q_{wall,j}=\frac{2U}{R_b}(T_j-T_\infty).
$$

The factor $2/R_b$ is the cylindrical sidewall-area-to-bed-volume ratio. Positive
$Q_{wall}$ removes energy; it becomes a heat input when the surroundings are
hotter. The model uses one constant $U$ and one $T_\infty$ for the whole bed.

The energy equation is

$$
\frac{dE_j}{dt}+\mathcal D_j(H)+\mathcal D_j(q)+Q_{wall,j}=0.
$$

When $\lambda_{ax}=0$, the heat-dispersion divergence is omitted. Internal face
heat transfers enter neighbouring cells with opposite signs. Since dispersive
heat flux is zero at both boundaries, this term redistributes stored energy
without providing an external heat source.

### How reaction heat enters

There is no additional $-\Delta H_r r$ term in the solved energy balance.
Reactions change the species inventories, and the stored energy includes those
species' reference and sensible enthalpies. Adding a second reaction-heat source
to this formulation would count the same contribution twice.

To see the connection, define

$$
C_{p,bed,j}=\sum_i c_{i,j}C_{p,i}^g(T_j)
+\sum_a s_{a,j}C_{p,a}^s(T_j),\qquad
\Delta H_k(T)=\sum_i\nu^g_{ik}h_i^g(T)+\sum_a\nu^s_{ak}h_a^s(T).
$$

Differentiating the energy definition at fixed voidage and substituting the
species balances gives the equivalent identity

$$
\begin{aligned}
C_{p,bed,j}\frac{dT_j}{dt}-\varepsilon_{g,j}\frac{dP_j}{dt}
={}&-\mathcal D_j(H)-\mathcal D_j(q)-Q_{wall,j}\\
&+\sum_i h_i^g(T_j)\mathcal D_j(N_i)
-\sum_k\Delta H_k(T_j)r_{k,j}.
\end{aligned}
$$

This identity explains the implicit reaction heat and pressure correction; it
is not an extra equation assembled by the solver. The implementation integrates
$E_j$ directly. In particular, storing total enthalpy instead of $E_j$ would omit
the pressure contribution during transients.

## Boundary conditions

The operating program prescribes total inlet molar flow $F_{in}(t)>0$, inlet
composition $y_{i,in}(t)$, inlet temperature $T_{in}(t)$, and outlet pressure
$P_{out}(t)$. Inlet pressure is solved, not supplied independently.

### Inlet

$$
N_{i,1/2}=\frac{F_{in}}{A}y_{i,in},\qquad
J_{i,1/2}=N_{i,1/2}h_i^g(T_{in}),\qquad q_{1/2}=0.
$$

The inlet velocity and pressure obey

$$
\frac{F_{in}}{A}=u_{1/2}\frac{P_{in}}{RT_{in}},\qquad
\frac{P_{in}-P_1}{x_1-0}=\mathcal E_{1/2}(u_{1/2}).
$$

This is a prescribed total species-flux and enthalpy-flux boundary. The first
cell's composition and temperature are not fixed to the inlet values during
integration. No additional species dispersion flux is added to the prescribed
$N_{i,1/2}$.

### Outlet

$$
N_{i,n+1/2}=u_{n+1/2}\frac{c_{i,n}}{\varepsilon_{g,n}},\qquad
J_{i,n+1/2}=N_{i,n+1/2}h_i^g(T_n),\qquad q_{n+1/2}=0,
$$

$$
\frac{P_n-P_{out}}{L-x_n}=\mathcal E_{n+1/2}(u_{n+1/2}).
$$

Outlet species and enthalpy fluxes use the final cell directly, with no added
axial species-dispersion flux. There is no independently imposed outlet
composition or temperature. Solids have no boundary flux at either end.

### Flow reversal limits

`simulation.interior_flow_mode: reversible` switches interior reconstruction
according to flux direction. Both inlet and outlet equations above remain
unchanged. There is no external outlet-reservoir composition or temperature to
supply a physical backflow boundary. Reversible interior transport should
therefore not be interpreted as a model of interchangeable inlet/outlet ports.

`forward_only` always uses the forward-oriented face reconstruction. It does
not clamp negative velocity or remove the signed Ergun closure; cases with
interior reversal require the reversible reconstruction.

## Operating programs

Programs consist of an initial value and timed `hold` or `ramp` steps. A hold
keeps the preceding value; a ramp has an absolute target. Channels without steps
remain constant. Flow, temperature, and pressure inputs are positive; each
specified composition includes every selected gas species and sums to one.

### Separate channels and feed streams

In `separate_channels` mode, flow, temperature, composition, and outlet pressure
are independently programmed. In `feed_stream` mode, the inlet program instead
interpolates the molar quantities

$$
F,\qquad G_i=Fy_i,\qquad B=FT.
$$

After interpolation and smoothing,

$$
y_{i,in}(t)=\frac{G_i(t)}{F(t)},\qquad
T_{in}(t)=\frac{B(t)}{F(t)}.
$$

All inlet quantities use the same positive flow denominator. This preserves
species molar-flow interpolation when the total flow changes. The temperature
is weighted by molar flow, not by an enthalpy balance between two feed streams.
A ramp target may omit flow, temperature, or composition; omitted values carry
forward from the preceding feed. Outlet pressure remains a separate channel.

### Ramp smoothing

The default smoothing half-width is $w=1\ \mathrm s$. Define the smoothed positive
part of elapsed time $z$ by

$$
g_w(z)=\begin{cases}
0, & z\leq-w,\\
(z+w)^2/(4w), & -w<z<w,\\
z, & z\geq w.
\end{cases}
$$

A ramp from $a$ to $b$ between $t_s$ and $t_e$ contributes

$$
(b-a)\frac{g_w(t-t_s)-g_w(t-t_e)}{t_e-t_s}
$$

to the carried value. Each corner is smoothed only within $w$ of its endpoint;
outside those regions, the result agrees with the unsmoothed piecewise-linear
program. If short steps have overlapping smoothing regions, the same summed
formula applies. Values and first derivatives are continuous at the joins.

For feed streams the smoothing is applied to $F$, $G_i$, and $B$ before taking
ratios. A ramp starting at time zero can therefore alter the actual inlet state
at $t=0$ from its declared `initial` values. Initialization, previews, and both
solver backends use these smoothed conditions. The smoothing width is an engine
parameter, not a field in the current YAML schema.

With `repeat_program: true`, each channel's next cycle starts from its previous
cycle's final value; it does not reset to the initial value. The first cycle and
the carried-forward continuation are retained, and extending the simulation
horizon does not change earlier program values. Repeating channels use their
own step-list durations. For non-repeating inputs, each channel with steps must
span the configured horizon; channels without steps remain constant.

### GHSV conversion

When the flow basis is `ghsv_per_h`, the volumetric flow uses the **empty bed
volume** $V_b=AL$ and the reference conditions
$T_N=273.15\ \mathrm K$, $P_N=100000\ \mathrm{Pa}$:

$$
F_{in}=\frac{\mathrm{GHSV}}{3600}V_b\frac{P_N}{RT_N}.
$$

No voidage factor multiplies $V_b$. This conversion is shared by both program
modes. Program construction is implemented in [programs.py](../packed_bed/programs.py).

## Initial conditions

[initialization.py](../packed_bed/initialization.py) constructs the starting
state before DAETools solves for consistency:

1. Build the uniform axial grid, sample the solid zones, convert solid inputs to
   mol/m³ bed, and calculate the fixed volume fractions and face diameters.
2. Evaluate the smoothed programs at zero to obtain $F_0$, $y_{i,0}$, $T_0$, and
   $P_{out,0}$. The initializer requires positive inlet flow.
3. Evaluate the mixture properties at $T_0$ and $y_{i,0}$. Construct a positive-flow
   Ergun pressure-profile seed using the inlet molar flux $F_0/A$, face properties,
   and half-cell boundary distances. Bracket and bisect the inlet pressure until
   the constructed outlet pressure matches $P_{out,0}$.
4. Set every cell's initial temperature and gas composition to $T_0$ and $y_{i,0}$,
   while allowing pressure and gas concentration to vary axially:

$$
c_{i,j}(0)=y_{i,0}\frac{\varepsilon_{g,j}P_j(0)}{RT_0},\qquad
s_{a,j}(0)=s_{a,0,j},
$$

$$
E_j(0)=\sum_i c_{i,j}(0)h_i^g(T_0)
+\sum_a s_{a,j}(0)h_a^s(T_0)-\varepsilon_{g,j}P_j(0).
$$

Gas and solid concentrations and internal energy are supplied as differential
initial conditions. Pressure, temperature, fluxes, rates, and other algebraic
quantities receive initial guesses. DAETools' `SolveInitial()` then obtains
consistent algebraic values and differential derivatives before either backend
integrates. The pressure construction is an initial-state calculation, not a
claim that the reactive bed is initially at steady state or chemical equilibrium.
There is no independently specified axial initial gas-composition or temperature
profile in the current case format.

Cumulative inlet/outlet mass and energy transfers and cumulative wall heat loss
start at zero. Initial bed inventories are calculated from the initialized
concentrations and internal energy.

## Spatial reconstruction

`simulation.mass_scheme` and `simulation.heat_scheme` independently select how
interior face states are reconstructed. Species convection reconstructs
$C_i=c_i/\varepsilon_g$; enthalpy transport reconstructs $h_i^g(T)$.

For a generic cell quantity $v$, let $v_f^L$ and $v_f^R$ be reconstructions from
the left and right. For signed transport rate $a_f$,

$$
\mathcal A_f(a_f,v)=
\begin{cases}
a_fv_f^L, & \texttt{forward\_only},\\
a_f^+v_f^L+a_f^-v_f^R, & \texttt{reversible},
\end{cases}
\qquad a_f^\pm=\frac{a_f\pm\lvert a_f\rvert}{2}.
$$

Use $a_f=u_f$ for species convection and $a_f=N_{i,f}$ for species enthalpy
transport. Right reconstructions apply the same formulas to the reversed cell
sequence. Central reconstruction is identical from either direction.

### Upwind, central, and limited linear schemes

At the face between cells $j$ and $j+1$, the left reconstructions are:

| Scheme | $v_f^L$ |
| --- | --- |
| `upwind1` | $v_j$ |
| `central` | $(v_j+v_{j+1})/2$ |
| `linear_upwind2` | $3v_j/2-v_{j-1}/2$ |
| `muscl_minmod` | $v_j+\operatorname{minmod}(v_j-v_{j-1},v_{j+1}-v_j)/2$ |

The limiter returns the argument of smaller magnitude if both differences have
the same sign and zero otherwise. At the first interior face, schemes needing
a second upstream cell fall back to `upwind1`.

### WENO3

For $v_{j-1},v_j,v_{j+1}$, the candidate states and smoothness indicators are

$$
p_0=\frac32v_j-\frac12v_{j-1},\quad
p_1=\frac12(v_j+v_{j+1}),\quad
\beta_0=(v_j-v_{j-1})^2,\quad
\beta_1=(v_{j+1}-v_j)^2.
$$

With $d_0=1/3$, $d_1=2/3$,

$$
\alpha_m=\frac{d_m}{(\beta_m+\epsilon^2)^2},\qquad
\omega_m=\frac{\alpha_m}{\sum_\ell\alpha_\ell},\qquad
v_f^L=\sum_m\omega_mp_m.
$$

### WENO5

For $v_{j-2},v_{j-1},v_j,v_{j+1},v_{j+2}$,

$$
\begin{aligned}
p_0&=\frac13v_{j-2}-\frac76v_{j-1}+\frac{11}{6}v_j,\\
p_1&=-\frac16v_{j-1}+\frac56v_j+\frac13v_{j+1},\\
p_2&=\frac13v_j+\frac56v_{j+1}-\frac16v_{j+2},
\end{aligned}
$$

$$
\begin{aligned}
\beta_0&=\frac{13}{12}(v_{j-2}-2v_{j-1}+v_j)^2
+\frac14(v_{j-2}-4v_{j-1}+3v_j)^2,\\
\beta_1&=\frac{13}{12}(v_{j-1}-2v_j+v_{j+1})^2
+\frac14(v_{j-1}-v_{j+1})^2,\\
\beta_2&=\frac{13}{12}(v_j-2v_{j+1}+v_{j+2})^2
+\frac14(3v_j-4v_{j+1}+v_{j+2})^2.
\end{aligned}
$$

Use the same nonlinear-weight formula with $(d_0,d_1,d_2)=(0.1,0.6,0.3)$.
Where its full stencil is unavailable, `weno5` falls back to `weno3`; where the
WENO3 upstream stencil is unavailable, it falls back to `upwind1`. The same
boundary fallback rules apply to the reversed right reconstruction.

The regularization scale is $\epsilon=10^{-8}\ \mathrm{mol/m^3}$ for gas-phase
concentration and $10^{-8}\ \mathrm{J/mol}$ for enthalpy. It enters as
$\epsilon^2$ in the smoothness denominator. It is a reconstruction parameter,
not a concentration floor or a solver absolute tolerance.

These formulas, including their boundary fallbacks, are in
[axial_schemes.py](../packed_bed/axial_schemes.py). Higher-order reconstruction
and central transport do not guarantee positive concentrations. Mass and heat
dispersion continue to use centred differences regardless of the selected
convective reconstruction.

## Time integration

The spatial balances and algebraic closures form a differential-algebraic
system $\mathcal F(t,z,\dot z)=0$. Gas and solid inventories and internal energy
are differential states; optional cumulative balance reports add differential
accounting states. Fluxes, composition closures, pressure, properties, and
reaction rates are algebraic quantities in the unreduced model.

- **Standard (`solver.backend: daetools`):** DAETools assembles and integrates the
  model using IDAS and the selected linear-solver adapter.
- **Compiled (`solver.backend: compiled`):** DAETools first solves the same initial
  conditions. The backend exports the equations, substitutes explicit algebraic
  definitions, eliminates fixed states where applicable, and generates native
  residuals, analytic Jacobians, and report reconstruction. SUNDIALS IDA integrates
  the reduced system with SuperLU, KLU, or Band.

The intended physical balances and boundary conditions are shared. Algebraic
elimination changes the solver state vector and error control, so identical
settings do not imply bitwise-identical trajectories. See
[simulation.py](../packed_bed/simulation.py), the
[compiled backend](../packed_bed/compiled/__init__.py), and its
[graph reduction](../packed_bed/compiled/graph.py).

IDA uses variable-step, variable-order backward differentiation, with orders
from one to five. Its error weights have the form

$$
W_\ell=\frac{1}{\mathrm{rtol}\lvert z_\ell\rvert+\mathrm{atol}_\ell}.
$$

The resulting weighted error norm controls local integration error, not a bound
on the error in a final scientific result. The numerical method is described in
the [SUNDIALS IDA mathematical reference](https://sundials.readthedocs.io/en/latest/ida/Mathematics_link.html).

`solver.concentration_absolute_tolerance` sets the gas and solid concentration
tolerances in mol/m³ bed; the total-concentration closures use the same types.
Other variable types retain their own tolerances. Absolute tolerance is
particularly important for trace species, where the relative term is small.
`maximum_order`, nonlinear convergence controls, and `suppress_algebraic_errors`
affect solver behaviour, not the physical source terms. Suppressing algebraic
error estimates does not remove the algebraic equations. Compiled residual
scaling likewise changes numerical conditioning while retaining the residual
zeros.

`reporting_interval_s` sets saved output times, not a fixed integration step.
The reporting schedule includes zero and the final horizon, adding a shorter
last reporting interval if needed. Selecting plots does not change the balances.
Selecting mass/heat balance reports adds their accounting equations and can
therefore affect the numerical path even though the physical reactor equations
are unchanged.

## Conservation diagnostics and reported quantities

### Mass accounting

The optional `mass_balance` report stores

$$
M_{bed}=A\sum_j\Delta x_j\left(\sum_iM_i c_{i,j}+\sum_aM_a s_{a,j}\right),
$$

$$
\frac{dM_{in}}{dt}=A\sum_iM_iN_{i,1/2},\qquad
\frac{dM_{out}}{dt}=A\sum_iM_iN_{i,n+1/2}.
$$

Its signed reported error is

$$
\delta M=M_{bed}(t)-M_{bed}(0)-M_{in}(t)+M_{out}(t).
$$

Summing cell balances cancels the interior fluxes exactly at the equation
level. Zero physical mass error also requires each selected reaction to satisfy
$\sum_iM_i\nu^g_{ik}+\sum_aM_a\nu^s_{ak}=0$ with the actual selected molecular
weights. A nonzero diagnostic can therefore reflect stoichiometry/property
inconsistency as well as numerical error. The core source assembly does not
repair an unbalanced reaction.

### Energy accounting

The optional `heat_balance` report stores

$$
E_{bed}=A\sum_j\Delta x_jE_j,\qquad
\frac{dQ_{in}}{dt}=AH_{1/2},\qquad
\frac{dQ_{out}}{dt}=AH_{n+1/2},\qquad
\frac{dQ_{loss}}{dt}=A\sum_j\Delta x_jQ_{wall,j}.
$$

Its signed reported error is

$$
\delta E=E_{bed}(t)-E_{bed}(0)-Q_{in}(t)+Q_{out}(t)+Q_{loss}(t).
$$

Despite the output name `heat_bed_total`, the stored inventory is **internal
energy**. `heat_in_total` and `heat_out_total` accumulate boundary **enthalpy**
transfer. Axial dispersive heat contributes no boundary transfer, and reaction
heat is already included through the species enthalpies. The cumulative
quantities are integrated as part of the DAE, not estimated afterwards by
trapezoidal integration of saved samples. All transfers retain their signs.

These diagnostics are reported in kg and J. The summary identifies the largest
absolute error among saved reporting times, not necessarily its largest value
between those times. Good conservation alone does not establish accurate
kinetics, adequate spatial resolution, or valid property data.

### Interpreting derived outputs

[reports.py](../packed_bed/reports.py) derives the following quantities from the
retained model data:

- Outlet temperature is the last cell's temperature, $T_n$.
- Pressure drop is $P_{in}-P_{out}$, including both boundary half-cell segments.
- Outlet species flow is $A N_{i,n+1/2}$ and total outlet flow is its sum.
- Outlet composition is $N_{i,n+1/2}/\sum_\ell N_{\ell,n+1/2}$. When the magnitude
  of the total outlet flux is at most $10^{-12}\ \mathrm{mol/(m^2\,s)}$, the
  reported composition falls back to the final-cell mole fractions.
- Solid mole fraction is $s_a/\sum_b s_b$ when total solid concentration is
  positive; an empty solid inventory reports zero fractions. This is a derived
  quantity, not an additional dynamic state or a solid volume fraction.

For numerical comparisons, vary the axial grid and solver tolerances, compare
both conservation errors and quantities of interest, and retain the selected
chemistry and property definitions with the results.

## Configuration and implementation map

| Input setting | Model meaning |
| --- | --- |
| `model.bed_length_m`, `bed_radius_m`, `axial_cells` | $L$, $R_b$, and the uniform axial grid |
| `model.gas_voidage_mode` | Gas storage fraction $\varepsilon_g$ |
| `model.ambient_temperature_k` | $T_\infty$; omitted-field default 873.15 K |
| `model.heat_transfer_coefficient_w_per_m2_k` | $U$; omitted-field default 100 W/(m²·K); set zero for no wall exchange |
| `model.axial_heat_dispersion_w_per_m_k` | $\lambda_{ax}$; default zero |
| `solids.initial_profile` | Concentration unit/basis and zones supplying initial solids, $e_b$, $e_p$, and $d_p$ |
| `simulation.mass_scheme`, `heat_scheme` | Interior concentration and enthalpy reconstruction |
| `simulation.interior_flow_mode` | Forward-oriented or sign-aware interior transport |
| `simulation.program_mode`, `repeat_program` | Boundary-program construction and repetition |
| `simulation.time_horizon_s`, `reporting_interval_s` | Integration horizon and saved reporting times |
| `solver.*` | Backend, linear solver, tolerances, and numerical controls |
| `outputs.requested_reports` | Saved quantities and optional integral balance equations |

Input paths in this table refer to the corresponding `run.yaml` or `solids.yaml`
sections. Defaults are schema fallbacks; authored examples and new desktop cases
can specify different values. The current definitions are in
[config/models.py](../packed_bed/config/models.py), with cross-file validation in
[config/load.py](../packed_bed/config/load.py).

| Implementation | Responsibility |
| --- | --- |
| [model.py](../packed_bed/model.py) | Balances, source sums, state closures, fluxes, and boundary equations |
| [solid_profiles.py](../packed_bed/solid_profiles.py) | Grid, volume fractions, zone sampling, and concentration conversion |
| [properties.py](../packed_bed/properties.py) | Enthalpy, heat capacity, pure viscosity, and Wilke mixing |
| [reactions.py](../packed_bed/reactions.py) | Reaction definitions and stoichiometric source matrices |
| [kinetics/](../packed_bed/kinetics/) | Built-in symbolic reaction rates |
| [programs.py](../packed_bed/programs.py) | Smoothing, repetition, feed mixing, and GHSV conversion |
| [initialization.py](../packed_bed/initialization.py) | Initial pressure seed, inventories, and algebraic guesses |
| [axial_schemes.py](../packed_bed/axial_schemes.py) | Reconstruction stencils, nonlinear weights, and signed flux splitting |
| [simulation.py](../packed_bed/simulation.py) | DAETools initialization and execution |
| [compiled/runtime.py](../packed_bed/compiled/runtime.py) | Native IDA integration and solver controls |
| [reports.py](../packed_bed/reports.py) | Output coordinates, derived quantities, and balance errors |

[Documentation index](../README.md#documentation)
