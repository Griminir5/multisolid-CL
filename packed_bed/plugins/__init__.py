"""Portable scientific plugin API. No GUI or solver imports at module import time."""
from packed_bed.parameters import ParameterSpec, parameter_group
from packed_bed.properties import BaseCorrelation, PolynomialHeatCapacity, QuadraticViscosity, ShomateHeatCapacity
from packed_bed.reactions import KineticsContext, ReactionDefinition, ReactionFamily

API_VERSION = 1
__all__ = ('API_VERSION', 'ParameterSpec', 'parameter_group', 'BaseCorrelation',
           'PolynomialHeatCapacity', 'QuadraticViscosity', 'ShomateHeatCapacity',
           'KineticsContext', 'ReactionDefinition', 'ReactionFamily')
