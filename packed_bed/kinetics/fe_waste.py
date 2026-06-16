from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from daetools.pyDAE import Constant, Exp, Log, Sqrt

from pyUnits import K, Pa, m, mol, s

from ..properties import PROPERTY_REGISTRY
from . import KineticsContext, register_kinetics_hook


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324

FE2O3_MW_KG_PER_MOL = PROPERTY_REGISTRY.get_record("Fe2O3").mw
FE_MW_KG_PER_MOL = PROPERTY_REGISTRY.get_record("Fe").mw