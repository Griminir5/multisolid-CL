from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping
import numpy as np

from daetools.pyDAE import Constant
from pyUnits import J, K, mol


def _as_float_array(temperature):
    return np.asarray(temperature, dtype=float)


class MolarEnthalpyCorrelation(ABC):
    """Build enthalpy as a native DAETOOLS expression."""

    @abstractmethod
    def dae_expression(self, temperature):
        raise NotImplementedError

    @abstractmethod
    def value(self, temperature):
        raise NotImplementedError

    def enthalpy(self, temperature):
        return self.value(temperature)


@dataclass(frozen=True)
class CpZerothMolar(MolarEnthalpyCorrelation):
    """Cp(T) = a0 and H(T) = h_ref + a0 * (T - T_ref)."""

    t_ref: float = 298.15
    h_form_ref: float = 0.0
    a0: float = 0.0

    def cp_dae_expression(self, temperature):
        return Constant(self.a0 * J / (mol * K))

    def cp_value(self, temperature):
        return np.zeros_like(_as_float_array(temperature), dtype=float) + self.a0

    def dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        h_form = Constant(self.h_form_ref * J / mol)
        a0 = Constant(self.a0 * J / (mol * K))
        d_t = temperature - t_ref
        return h_form + a0*d_t

    def value(self, temperature):
        d_t = _as_float_array(temperature) - self.t_ref
        return self.h_form_ref + d_t * self.a0


@dataclass(frozen=True)
class CpQuadraticMolar(MolarEnthalpyCorrelation):
    """Cp(T) = a0 + a1 * dT + a2 * dT^2 where dT = T - T_ref."""

    t_ref: float = 298.15
    h_form_ref: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0

    def cp_dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        a0 = Constant(self.a0 * J / (mol * K))
        a1 = Constant(self.a1 * J / (mol * K**2))
        a2 = Constant(self.a2 * J / (mol * K**3))
        d_t = temperature - t_ref
        return a0 + d_t * (a1 + d_t * a2)

    def cp_value(self, temperature):
        d_t = _as_float_array(temperature) - self.t_ref
        return self.a0 + d_t * (self.a1 + d_t * self.a2)

    def dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        h_form = Constant(self.h_form_ref * J / mol)
        a0 = Constant(self.a0 * J / (mol * K))
        a1_half = Constant(0.5 * self.a1 * J / (mol * K**2))
        a2_third = Constant((self.a2 / 3.0) * J / (mol * K**3))
        d_t = temperature - t_ref
        return h_form + d_t * (a0 + d_t * (a1_half + d_t * a2_third))

    def value(self, temperature):
        d_t = _as_float_array(temperature) - self.t_ref
        return self.h_form_ref + d_t * (
            self.a0 + d_t * (0.5 * self.a1 + d_t * (self.a2 / 3.0))
        )


@dataclass(frozen=True)
class CpCubicMolar(MolarEnthalpyCorrelation):
    """Cp(T) = a0 + a1 * dT + a2 * dT^2 + a3 * dT^3 where dT = T - T_ref."""

    t_ref: float = 298.15
    h_form_ref: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0

    def cp_dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        a0 = Constant(self.a0 * J / (mol * K))
        a1 = Constant(self.a1 * J / (mol * K**2))
        a2 = Constant(self.a2 * J / (mol * K**3))
        a3 = Constant(self.a3 * J / (mol * K**4))
        d_t = temperature - t_ref
        return a0 + d_t * (a1 + d_t * (a2 + d_t * a3))

    def cp_value(self, temperature):
        d_t = _as_float_array(temperature) - self.t_ref
        return self.a0 + d_t * (self.a1 + d_t * (self.a2 + d_t * self.a3))

    def dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        h_form = Constant(self.h_form_ref * J / mol)
        a0 = Constant(self.a0 * J / (mol * K))
        a1_half = Constant(0.5 * self.a1 * J / (mol * K**2))
        a2_third = Constant((self.a2 / 3.0) * J / (mol * K**3))
        a3_quarter = Constant(0.25 * self.a3 * J / (mol * K**4))
        d_t = temperature - t_ref
        return h_form + d_t * (
            a0 + d_t * (a1_half + d_t * (a2_third + d_t * a3_quarter))
        )

    def value(self, temperature):
        d_t = _as_float_array(temperature) - self.t_ref
        return self.h_form_ref + d_t * (
            self.a0
            + d_t * (0.5 * self.a1 + d_t * (self.a2 / 3.0 + d_t * (0.25 * self.a3)))
        )


@dataclass(frozen=True)
class CpQuarticMolar(MolarEnthalpyCorrelation):
    """Cp(T) = a0 + a1 * dT + a2 * dT^2 + a3 * dT^3 + a4 * dT^4."""

    t_ref: float = 298.15
    h_form_ref: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    a4: float = 0.0

    def cp_dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        a0 = Constant(self.a0 * J / (mol * K))
        a1 = Constant(self.a1 * J / (mol * K**2))
        a2 = Constant(self.a2 * J / (mol * K**3))
        a3 = Constant(self.a3 * J / (mol * K**4))
        a4 = Constant(self.a4 * J / (mol * K**5))
        d_t = temperature - t_ref
        return a0 + d_t * (a1 + d_t * (a2 + d_t * (a3 + d_t * a4)))

    def cp_value(self, temperature):
        d_t = _as_float_array(temperature) - self.t_ref
        return self.a0 + d_t * (self.a1 + d_t * (self.a2 + d_t * (self.a3 + d_t * self.a4)))

    def dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        h_form = Constant(self.h_form_ref * J / mol)
        a0 = Constant(self.a0 * J / (mol * K))
        a1_half = Constant(0.5 * self.a1 * J / (mol * K**2))
        a2_third = Constant((self.a2 / 3.0) * J / (mol * K**3))
        a3_quarter = Constant(0.25 * self.a3 * J / (mol * K**4))
        a4_fifth = Constant(0.2 * self.a4 * J / (mol * K**5))
        d_t = temperature - t_ref
        return h_form + d_t * (
            a0
            + d_t * (a1_half + d_t * (a2_third + d_t * (a3_quarter + d_t * a4_fifth)))
        )

    def value(self, temperature):
        d_t = _as_float_array(temperature) - self.t_ref
        return self.h_form_ref + d_t * (
            self.a0
            + d_t
            * (
                0.5 * self.a1
                + d_t * (self.a2 / 3.0 + d_t * (0.25 * self.a3 + d_t * (0.2 * self.a4)))
            )
        )

@dataclass(frozen=True)
class CpShomateMolar(MolarEnthalpyCorrelation):
    """Shomate Cp(T) basis with enthalpy anchored at h_form_ref when T = t_ref."""

    t_ref: float = 298.15
    h_form_ref: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    a4: float = 0.0

    def cp_dae_expression(self, temperature):
        temp_scale = Constant(1000.0 * K)
        tau = temperature / temp_scale
        a0 = Constant(self.a0 * J / (mol * K))
        a1 = Constant(self.a1 * J / (mol * K))
        a2 = Constant(self.a2 * J / (mol * K))
        a3 = Constant(self.a3 * J / (mol * K))
        a4 = Constant(self.a4 * J / (mol * K))

        return a0 + a1 * tau + a2 * tau**2 + a3 * tau**3 + a4 / tau**2

    def cp_value(self, temperature):
        tau = _as_float_array(temperature) / 1000.0
        return self.a0 + self.a1 * tau + self.a2 * tau**2 + self.a3 * tau**3 + self.a4 / tau**2

    def dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        temp_scale = Constant(1000.0 * K)
        h_form = Constant(self.h_form_ref * J / mol)
        tau = temperature / temp_scale
        tau_ref = t_ref / temp_scale
        a0 = Constant(self.a0 * J / (mol * K))
        a1 = Constant(self.a1 * J / (mol * K))
        a2 = Constant(self.a2 * J / (mol * K))
        a3 = Constant(self.a3 * J / (mol * K))
        a4 = Constant(self.a4 * J / (mol * K))

        return (
            h_form
            + a0 * (temperature - t_ref)
            + 0.5 * a1 * temp_scale * (tau**2 - tau_ref**2)
            + (a2 * temp_scale / 3.0) * (tau**3 - tau_ref**3)
            + 0.25 * a3 * temp_scale * (tau**4 - tau_ref**4)
            - a4 * temp_scale * (1.0 / tau - 1.0 / tau_ref)
        )

    def value(self, temperature):
        tau = _as_float_array(temperature) / 1000.0
        tau_ref = self.t_ref / 1000.0
        return (
            self.h_form_ref
            + self.a0 * (_as_float_array(temperature) - self.t_ref)
            + 500.0 * self.a1 * (tau**2 - tau_ref**2)
            + (1000.0 / 3.0) * self.a2 * (tau**3 - tau_ref**3)
            + 250.0 * self.a3 * (tau**4 - tau_ref**4)
            - 1000.0 * self.a4 * (1.0 / tau - 1.0 / tau_ref)
        )


@dataclass(frozen=True)
class SpeciesPropertyRecord:
    """Property data for one species, stored in canonical SI units."""

    name: str
    phase: str
    mw: float | None = None
    enthalpy: MolarEnthalpyCorrelation | None = None
    gas_viscosity: object | None = None
    gas_conductivity: object | None = None

    def __post_init__(self) -> None:
        if self.phase not in {"gas", "solid"}:
            raise ValueError(f"Unsupported phase '{self.phase}' for species '{self.name}'.")
        if self.phase != "gas":
            if self.gas_viscosity is not None:
                raise ValueError(f"Gas viscosity is only valid for gas species ('{self.name}').")
            if self.gas_conductivity is not None:
                raise ValueError(f"Gas conductivity is only valid for gas species ('{self.name}').")


@dataclass(frozen=True)
class PropertyRegistry:
    records: Mapping[str, SpeciesPropertyRecord]

    def get_record(self, species_name: str) -> SpeciesPropertyRecord:
        return self.records[species_name]

    def enthalpy_expression(self, species_name: str, temperature):
        record = self.get_record(species_name)
        if record.enthalpy is None:
            raise KeyError(f"Species '{species_name}' does not define an enthalpy correlation.")
        return record.enthalpy.dae_expression(temperature)

    def enthalpy_value(self, species_name: str, temperature):
        record = self.get_record(species_name)
        if record.enthalpy is None:
            raise KeyError(f"Species '{species_name}' does not define an enthalpy correlation.")
        return record.enthalpy.value(temperature)


DEFAULT_PROPERTY_REGISTRY = PropertyRegistry(
    records={
        "AR": SpeciesPropertyRecord(
            name="Argon",
            phase="gas",
            mw=39.948e-3,
            enthalpy=CpZerothMolar(h_form_ref=0.0, a0=20.786),
        ),
        "CH4": SpeciesPropertyRecord(
            name="Methane",
            phase="gas",
            mw=16.043e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=-74873.0,
                a0=37.6194088,
                a1=5.054146625e-2,
                a2=2.27606802e-6,
                a3=-6.82466733e-9,
            ),
        ),
        "CO": SpeciesPropertyRecord(
            name="Carbon Monoxide",
            phase="gas",
            mw=28.010e-3,
            enthalpy=CpQuarticMolar(
                h_form_ref=-110541.0,
                a0=29.1010658,
                a1=1.92542547e-3,
                a2=1.18696459e-5,
                a3=-1.12435518e-8,
                a4=2.97502474e-12,
            ),
        ),
        "CO2": SpeciesPropertyRecord(
            name="Carbon Dioxide",
            phase="gas",
            mw=44.0095e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=-393505.0,
                a0=37.0481649,
                a1=4.05995467e-2,
                a2=-2.72100644e-5,
                a3=6.84699347e-9,
            ),
        ),
        "H2": SpeciesPropertyRecord(
            name="Hydrogen",
            phase="gas",
            mw=2.01588e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=0.0,
                a0=28.6285719,
                a1=3.87550311e-3,
                a2=-6.60085498e-6,
                a3=8.02717561e-9,
            ),
        ),
        "H2O": SpeciesPropertyRecord(
            name="Water",
            phase="gas",
            mw=18.01528e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=-241826.0,
                a0=33.7806686,
                a1=6.48912262e-3,
                a2=8.12757341e-6,
                a3=-3.53127971e-9,
            ),
        ),
        "HE": SpeciesPropertyRecord(
            name="Helium",
            phase="gas",
            mw=4.002602e-3,
            enthalpy=CpZerothMolar(h_form_ref=0.0, a0=20.786),
        ),
        "N2": SpeciesPropertyRecord(
            name="Nitrogen",
            phase="gas",
            mw=28.0134e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=0.0,
                a0=29.151445,
                a1=2.69178695e-3,
                a2=4.47573614e-6,
                a3=-2.23789023e-9,
            ),
        ),
        "O2": SpeciesPropertyRecord(
            name="Oxygen",
            phase="gas",
            mw=31.9988e-3,
            enthalpy=CpQuadraticMolar(
                h_form_ref=0.0,
                a0=29.7404663,
                a1=8.82224917e-3,
                a2=-2.57179415e-6,
            ),
        ),
        "Ni": SpeciesPropertyRecord(
            name="Nickel",
            phase="solid",
            mw=58.693e-3,
            enthalpy=CpQuarticMolar(
                h_form_ref=0.0,
                a0=24.4850906,
                a1=6.33359549e-2,
                a2=-1.81736161e-4,
                a3=2.10037634e-7,
                a4=-8.01728267e-11,
            ),
        ),
        "NiO": SpeciesPropertyRecord(
            name="Nickel Oxide",
            phase="solid",
            mw=74.6928e-3,
            enthalpy=CpShomateMolar(h_form_ref=-239701, a0=179.38973769, a1=-300.19583295, a2=246.69888057, a3=-65.90651588, a4=-6.35461864),
        ),
        "CaAl2O4": SpeciesPropertyRecord(
            name="Calcium Aluminate",
            phase="solid",
            mw=158.039e-3,
            enthalpy=CpShomateMolar(h_form_ref=-2326304, a0=154.055548, a1=22.3001808, a2=-2.47833922e-4, a3=2.56391177e-4, a4=-3.54806056),
        ),
    }
)
