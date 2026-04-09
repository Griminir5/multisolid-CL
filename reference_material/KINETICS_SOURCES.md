# This is meant as a place to store references for gas-solid and gas-gas reaction kinetics, with a short description

# Gas-solid
## Copper
### San Pio CuO/SiO2 and CuO/Al2O3
San Pio (https://doi.org/10.1016/j.ces.2017.09.044) has very nice equations for oxidation and reduction of Cu on SiO2 (essentialy no interaction with support) and Cu on Al2O3, where spinels form. Equations are presented both for true shrinking core models as well as pseudohomogeneous reaction rates. Unfortunately only hydrogen is used as reducing gas, no CO. Additionaly, an availability term of the form $C_{H2}/(C_{H2} + eps)$ has to be included, since pseudohomogeneous reactions are order 0 wrt to H2.

## Nickel
### Medrano NiO/CaAl2O4
Source (https://doi.org/10.1016/j.apenergy.2015.08.078) for kinetics of nickel in chemical looping, with nominally no support interaction. Specifically these kinetics are regressed from data on HiFUEL® R110 from Johnson Matthey, so they may be doing some secret fuckery with it, which may render kinetics regressed in this work less valid for other Ni-based catalysts, regardless, the equation form is very nice and can be used to regress another set of parameters.  