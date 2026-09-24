"""Cp = cp_inf - deficit*Tref/T, giving a logarithmic enthalpy term."""
from dataclasses import dataclass

import numpy as np
from packed_bed.plugins import BaseCorrelation, ParameterSpec

PARAMETERS = {
    'cp_inf': ParameterSpec(50.0, 'J/(mol*K)', 'High-temperature Cp', minimum=1),
    'deficit': ParameterSpec(15.0, 'J/(mol*K)', 'Cp deficit at Tref', minimum=0),
    't_ref': ParameterSpec(298.15, 'K', 'Reference temperature', minimum=1),
    'h_ref': ParameterSpec(0.0, 'J/mol', 'Enthalpy at Tref'),
}


@dataclass(frozen=True)
class LogarithmicEnthalpy(BaseCorrelation):
    cp_inf: float
    deficit: float
    t_ref: float
    h_ref: float

    def _cp(self, t):
        return self.cp_inf - self.deficit * self.t_ref / t

    def _h(self, t, log):
        return self.h_ref + self.cp_inf * (t - self.t_ref) - self.deficit * self.t_ref * log(t / self.t_ref)

    def value(self, temperature):
        return self._h(np.asarray(temperature, dtype=float), np.log)

    def cp_value(self, temperature):
        return self._cp(np.asarray(temperature, dtype=float))

    def dae_expression(self, temperature):
        from daetools.pyDAE import Constant, Log
        from pyUnits import J, K, mol
        return self._h(temperature / Constant(1 * K), Log) * Constant(1 * J / mol)

    def cp_dae_expression(self, temperature):
        from daetools.pyDAE import Constant
        from pyUnits import J, K, mol
        return self._cp(temperature / Constant(1 * K)) * Constant(1 * J / (mol * K))


def create(parameters):
    if parameters['deficit'] >= parameters['cp_inf']:
        raise ValueError('Cp at Tref must be positive: deficit must be smaller than cp_inf.')
    return LogarithmicEnthalpy(**parameters)
