from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from daetools.pyDAE import Constant, Exp, Sqrt

from pyUnits import K, Pa, m, mol, s

from ..properties import PROPERTY_REGISTRY
from . import KineticsContext, register_kinetics_hook

