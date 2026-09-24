"""Illustrative Cp approaching a high-temperature limit, integrated exactly."""
from dataclasses import dataclass

import numpy as np
from packed_bed.plugins import BaseCorrelation, ParameterSpec

PARAMETERS = {
    'cp_inf': ParameterSpec(70.0, 'J/(mol*K)', 'High-temperature Cp', minimum=1),
    'deficit': ParameterSpec(35.0, 'J/(mol*K)', 'Cp deficit at Tref', minimum=0),
    't_ref': ParameterSpec(298.15, 'K', 'Reference temperature', minimum=1),
    'h_ref': ParameterSpec(-46000.0, 'J/mol', 'Enthalpy at Tref'),
}


@dataclass(frozen=True)
class InverseSquareEnthalpy(BaseCorrelation):
    cp_inf: float
    deficit: float
    t_ref: float
    h_ref: float

    def _cp(self, t):
        return self.cp_inf - self.deficit * (self.t_ref / t)**2

    def _h(self, t):
        return self.h_ref + self.cp_inf * (t - self.t_ref) + self.deficit * self.t_ref**2 * (1 / t - 1 / self.t_ref)

    def value(self, temperature):
        return self._h(np.asarray(temperature, dtype=float))

    def cp_value(self, temperature):
        return self._cp(np.asarray(temperature, dtype=float))

    def dae_expression(self, temperature):
        from daetools.pyDAE import Constant
        from pyUnits import J, K, mol
        return self._h(temperature / Constant(1 * K)) * Constant(1 * J / mol)

    def cp_dae_expression(self, temperature):
        from daetools.pyDAE import Constant
        from pyUnits import J, K, mol
        return self._cp(temperature / Constant(1 * K)) * Constant(1 * J / (mol * K))


def create(parameters):
    if parameters['deficit'] >= parameters['cp_inf']:
        raise ValueError('Cp at Tref must be positive: deficit must be smaller than cp_inf.')
    return InverseSquareEnthalpy(**parameters)
