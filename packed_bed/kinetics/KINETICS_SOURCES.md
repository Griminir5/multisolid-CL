# This is meant as a place to store references for gas-solid and gas-gas reaction kinetics, with a short description

# Gas-solid
## Copper
### San Pio CuO/SiO2 and CuO/Al2O3 (copper_redox.py)
San Pio (https://doi.org/10.1016/j.ces.2017.09.044) has very nice equations for oxidation and reduction of Cu on SiO2 (essentialy no interaction with support) and Cu on Al2O3, where spinels form. Equations are presented both for true shrinking core models as well as pseudohomogeneous reaction rates. Unfortunately only hydrogen is used as reducing gas, no CO. Additionaly, an availability term of the form $C_{H2}/(C_{H2} + eps)$ has to be included, since pseudohomogeneous reactions are order 0 wrt to H2.

## Nickel
### Medrano NiO/CaAl2O4 (medrano.py)
Source (https://doi.org/10.1016/j.apenergy.2015.08.078) for kinetics of nickel in chemical looping, with nominally no support interaction. Specifically these kinetics are regressed from data on HiFUEL® R110 from Johnson Matthey, so they may be doing some secret fuckery with it, which may render kinetics regressed in this work less valid for other Ni-based catalysts, regardless, the equation form is very nice and can be used to regress another set of parameters.

### Medrano NiO/CaAl2O4 - Andrew Wright (medrano_an.py)
Source (Technical Report: Chemical Looping Reactor Modelling – 2D) follows the idea of Medrano kinetics, but the implementation of the expression is slightly different.
Some of the terms are rearranged, rational approximations to fractional power functions are used.
Overall this set of equation is preferred vs the original Medrano implementation.

# Gas-gas
## Nickel-catalysed
### Xu-Froment Reforming (xu_froment.py)
Source (Technical Report: Chemical Looping Reactor Modelling – 2D;  https://doi.org/10.1002/aic.690350109) is an implementation of the Xu and Froment reforming reactions, which include a water gas shift reaction from carbon monoxide to carbon dioxide, reforming from methane to carbon monoxide, and an overall reaction directly from methano to carbon dioxide. All reactions are reversible and use partial pressure as the driving force.

### Xu-Froment Reforming (numaguchi_an.py)
Follows the source (Technical Report: Chemical Looping Reactor Modelling – 2D;  https://doi.org/10.1002/aic.690350109) in the implementation of two reactions, reforming and water-gas shift. In the original technical report hydrogen is used as an inhibitory term, this has been corrected to water.