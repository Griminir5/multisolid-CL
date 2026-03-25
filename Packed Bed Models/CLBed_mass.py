__doc__ = """
This file is means as the very first step towards a model for chemical looping in packed-bed reactors.
This file implements a very simple mass balance, with pressure and temperature considered fixed.
"""

# 1. Import the modules
import sys
from time import localtime, strftime
from daetools.pyDAE import *

from pyUnits import m, s, K, kmol, Pa # this will not show up because pylance cannot get to .pyd files

#
molar_flux_type = daeVariableType(name="molar_flux_type", units=kmol/m**2/s,
                                  lowerBound=0, upperBound=100, initialGuess=0, absTolerance=1e-5)
molar_conc_type = daeVariableType(name="molar_conc_type", units=kmol/m**3,
                                  lowerBound=0, upperBound=10, initialGuess=0, absTolerance=1e-5)
molar_frac_type = daeVariableType(name="molar_frac_type", units=dimless,
                                  lowerBound=-0.1, upperBound=1.1, initialGuess=0, absTolerance=1e-5)

temp_type = daeVariableType(name="temp_type", units=K,
                            lowerBound=100, upperBound=2000, initialGuess=500, absTolerance=1e-5)

pres_type = daeVariableType(name="pres_type", units=Pa,
                            lowerBound=1e-3, upperBound=1e7, initialGuess=1e5, absTolerance=1e-5)
velocity_type = daeVariableType(name="velocity_type", units=m//s,
                                lowerBound=-100, upperBound=100, initialGuess=1, absTolerance=1e-5)

class CLBed_mass(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):