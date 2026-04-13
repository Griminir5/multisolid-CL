from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from daetools.pyDAE import Constant
from pyUnits import J, K, Pa, mol, s


def _as_float_array(temperature):
    return np.asarray(temperature, dtype=float)


class MolarEnthalpyCorrelation(ABC):
    @abstractmethod
    def dae_expression(self, temperature):
        raise NotImplementedError

    @abstractmethod
    def value(self, temperature):
        raise NotImplementedError


class GasViscosityCorrelation(ABC):
    @abstractmethod
    def dae_expression(self, temperature):
        raise NotImplementedError

    @abstractmethod
    def value(self, temperature):
        raise NotImplementedError


@dataclass(frozen=True)
class CpZerothMolar(MolarEnthalpyCorrelation):
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
        return h_form + a0 * (temperature - t_ref)

    def value(self, temperature):
        return self.h_form_ref + (_as_float_array(temperature) - self.t_ref) * self.a0


@dataclass(frozen=True)
class CpQuadraticMolar(MolarEnthalpyCorrelation):
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
        return h_form + d_t * (a0 + d_t * (a1_half + d_t * (a2_third + d_t * a3_quarter)))

    def value(self, temperature):
        d_t = _as_float_array(temperature) - self.t_ref
        return self.h_form_ref + d_t * (
            self.a0 + d_t * (0.5 * self.a1 + d_t * (self.a2 / 3.0 + d_t * (0.25 * self.a3)))
        )


@dataclass(frozen=True)
class CpQuarticMolar(MolarEnthalpyCorrelation):
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
            a0 + d_t * (a1_half + d_t * (a2_third + d_t * (a3_quarter + d_t * a4_fifth)))
        )

    def value(self, temperature):
        d_t = _as_float_array(temperature) - self.t_ref
        return self.h_form_ref + d_t * (
            self.a0
            + d_t * (0.5 * self.a1 + d_t * (self.a2 / 3.0 + d_t * (0.25 * self.a3 + d_t * (0.2 * self.a4))))
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
        temperature_array = _as_float_array(temperature)
        tau = temperature_array / 1000.0
        tau_ref = self.t_ref / 1000.0
        return (
            self.h_form_ref
            + self.a0 * (temperature_array - self.t_ref)
            + 500.0 * self.a1 * (tau**2 - tau_ref**2)
            + (1000.0 / 3.0) * self.a2 * (tau**3 - tau_ref**3)
            + 250.0 * self.a3 * (tau**4 - tau_ref**4)
            - 1000.0 * self.a4 * (1.0 / tau - 1.0 / tau_ref)
        )


@dataclass(frozen=True)
class ViscosityQuadratic(GasViscosityCorrelation):
    t_ref: float = 1000.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0

    def dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        a0 = Constant(self.a0 * Pa * s)
        a1 = Constant(self.a1 * (Pa * s) / K)
        a2 = Constant(self.a2 * (Pa * s) / K**2)
        d_t = temperature - t_ref
        return a0 + d_t * (a1 + a2 * d_t)

    def value(self, temperature):
        d_t = _as_float_array(temperature) - self.t_ref
        return self.a0 + d_t * (self.a1 + d_t * self.a2)


@dataclass(frozen=True)
class SpeciesPropertyRecord:
    """Property data for one species, stored in canonical SI units."""

    name: str
    phase: str
    mw: float | None = None
    enthalpy: MolarEnthalpyCorrelation | None = None
    viscosity: GasViscosityCorrelation | None = None
    gas_conductivity: object | None = None

    def __post_init__(self) -> None:
        if self.phase not in {"gas", "solid"}:
            raise ValueError(f"Unsupported phase '{self.phase}' for species '{self.name}'.")
        if self.phase != "gas":
            if self.viscosity is not None:
                raise ValueError(f"Gas viscosity is only valid for gas species ('{self.name}').")
            if self.gas_conductivity is not None:
                raise ValueError(f"Gas conductivity is only valid for gas species ('{self.name}').")


@dataclass(frozen=True)
class PropertyRegistry:
    records: Mapping[str, SpeciesPropertyRecord]

    def has_species(self, species_id: str) -> bool:
        return species_id in self.records

    def species_ids(self, phase: str | None = None) -> tuple[str, ...]:
        if phase is None:
            return tuple(self.records.keys())
        return tuple(species_id for species_id, record in self.records.items() if record.phase == phase)

    def get_record(self, species_id: str) -> SpeciesPropertyRecord:
        try:
            return self.records[species_id]
        except KeyError as exc:
            available = ", ".join(self.records.keys())
            raise KeyError(f"Unknown species '{species_id}'. Available species: {available}") from exc

    def require_species(self, species_id: str, phase: str | None = None) -> SpeciesPropertyRecord:
        record = self.get_record(species_id)
        if phase is not None and record.phase != phase:
            raise KeyError(f"Species '{species_id}' is phase '{record.phase}', expected '{phase}'.")
        return record

    def enthalpy_expression(self, species_id: str, temperature):
        record = self.get_record(species_id)
        if record.enthalpy is None:
            raise KeyError(f"Species '{species_id}' does not define an enthalpy correlation.")
        return record.enthalpy.dae_expression(temperature)

    def enthalpy_value(self, species_id: str, temperature):
        record = self.get_record(species_id)
        if record.enthalpy is None:
            raise KeyError(f"Species '{species_id}' does not define an enthalpy correlation.")
        return record.enthalpy.value(temperature)

    def viscosity_expression(self, species_id: str, temperature):
        record = self.get_record(species_id)
        if record.viscosity is None:
            raise KeyError(f"Species '{species_id}' does not define a viscosity correlation.")
        return record.viscosity.dae_expression(temperature)

    def viscosity_value(self, species_id: str, temperature):
        record = self.get_record(species_id)
        if record.viscosity is None:
            raise KeyError(f"Species '{species_id}' does not define a viscosity correlation.")
        return record.viscosity.value(temperature)


PROPERTY_REGISTRY = PropertyRegistry(
    records={
        "Ar": SpeciesPropertyRecord(
            name="Argon",
            phase="gas",
            mw=39.948e-3,
            enthalpy=CpZerothMolar(h_form_ref=0.0, a0=20.786),
            viscosity=ViscosityQuadratic(a0=5.56703098e-05, a1=3.86114742e-08, a2=-8.57834841e-12),
        ),
        "CH4": SpeciesPropertyRecord(
            name="Methane",
            phase="gas",
            mw=16.043e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=-74873.0,
                a0=3.65894663e01,
                a1=5.41242609e-02,
                a2=3.72168361e-06,
                a3=-1.07088462e-08,
            ),
            viscosity=ViscosityQuadratic(a0=2.80097001e-05, a1=2.02172611e-08, a2=-2.79160801e-12),
        ),
        "CO": SpeciesPropertyRecord(
            name="Carbon Monoxide",
            phase="gas",
            mw=28.010e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=-110541.0,
                a0=2.87504130e01,
                a1=5.05074872e-03,
                a2=3.90907850e-06,
                a3=-3.09307094e-09,
            ),
            viscosity=ViscosityQuadratic(a0=4.05943891e-05, a1=2.58571235e-08, a2=-6.43543956e-12),
        ),
        "CO2": SpeciesPropertyRecord(
            name="Carbon Dioxide",
            phase="gas",
            mw=44.0095e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=-393505.0,
                a0=3.81416029e01,
                a1=3.59619451e-02,
                a2=-2.28055090e-05,
                a3=5.90960724e-09,
            ),
            viscosity=ViscosityQuadratic(a0=3.99550354e-05, a1=2.79482484e-08, a2=-8.23314636e-12),
        ),
        "H2": SpeciesPropertyRecord(
            name="Hydrogen",
            phase="gas",
            mw=2.01588e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=0.0,
                a0=2.94409905e01,
                a1=-2.38377533e-03,
                a2=6.39601662e-06,
                a3=-2.03147561e-09,
            ),
            viscosity=ViscosityQuadratic(a0=2.04091133e-05, a1=1.41343819e-08, a2=-2.34255119e-12),
        ),
        "H2O": SpeciesPropertyRecord(
            name="Water",
            phase="gas",
            mw=18.01528e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=-241826.0,
                a0=3.34558812e01,
                a1=7.59842908e-03,
                a2=7.65622945e-06,
                a3=-3.77998371e-09,
            ),
            viscosity=ViscosityQuadratic(a0=3.77371020e-05, a1=4.16760701e-08, a2=1.73653811e-12),
        ),
        "He": SpeciesPropertyRecord(
            name="Helium",
            phase="gas",
            mw=4.002602e-3,
            enthalpy=CpZerothMolar(h_form_ref=0.0, a0=20.786),
            viscosity=ViscosityQuadratic(a0=4.62481858e-05, a1=3.30037729e-08, a2=-4.84571869e-12),
        ),
        "N2": SpeciesPropertyRecord(
            name="Nitrogen",
            phase="gas",
            mw=28.0134e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=0.0,
                a0=2.87990411e01,
                a1=3.12879912e-03,
                a2=6.03928488e-06,
                a3=-3.77238986e-09,
            ),
            viscosity=ViscosityQuadratic(a0=4.15082118e-05, a1=2.78501267e-08, a2=-6.09391431e-12),
        ),
        "O2": SpeciesPropertyRecord(
            name="Oxygen",
            phase="gas",
            mw=31.9988e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=0.0,
                a0=2.88034459e01,
                a1=1.32893606e-02,
                a2=-7.91100675e-06,
                a3=1.82341564e-09,
            ),
            viscosity=ViscosityQuadratic(a0=4.92257601e-05, a1=3.26807190e-08, a2=-8.12639449e-12),
        ),
        "Ni": SpeciesPropertyRecord(
            name="Nickel",
            phase="solid",
            mw=58.693e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=0.0,
                a0=3.22350875e01,
                a1=-1.59194574e-02,
                a2=3.55975554e-05,
                a3=-1.62743281e-08,
            ),
        ),
        "NiO": SpeciesPropertyRecord(
            name="Nickel Oxide",
            phase="solid",
            mw=74.6928e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=-239701.0,
                a0=5.64774634e01,
                a1=-1.56343578e-02,
                a2=2.10045988e-05,
                a3=-4.78601077e-09,
            ),
        ),
        "CaAl2O4": SpeciesPropertyRecord(
            name="Calcium Aluminate",
            phase="solid",
            mw=158.039e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=-2326300.0,
                a0=1.31025407e02,
                a1=1.07694151e-01,
                a2=-9.18337717e-05,
                a3=3.37682975e-08,
            ),
        ),
        "Cu": SpeciesPropertyRecord(
            name="Copper",
            phase="solid",
            mw=63.55e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=0.0,
                a0=2.43826543e+01,
                a1=1.06212521e-02,
                a2=-1.53809529e-05,
                a3=1.25658461e-08,
            ),
        ),
        "Cu2O": SpeciesPropertyRecord(
            name="Copper(I) Oxide",
            phase="solid",
            mw=143.091e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=-170707.0,
                a0=6.31729851e+01,
                a1=4.50982629e-02,
                a2=-4.40775011e-05,
                a3=2.44874168e-08,
            ),
        ),
        "CuO": SpeciesPropertyRecord(
            name="Copper(II) Oxide",
            phase="solid",
            mw=79.545e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=-156063.0,
                a0=4.28595032e+01,
                a1=3.80740189e-02,
                a2=-4.25436164e-05,
                a3=1.92484328e-08,
            ),
        ),
        "Fe": SpeciesPropertyRecord(
            name="Iron",
            phase="solid",
            mw=55.845e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=0.0,
                a0=1.52959946e+01,
                a1=1.05151203e-01,
                a2=-1.20394019e-04,
                a3=4.13092928e-08,
            ),
        ),
        "FeO": SpeciesPropertyRecord(
            name="Iron(II) Oxide",
            phase="solid",
            mw=71.844e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=-272044.0,
                a0=5.00153501e+01,
                a1=1.83032040e-02,
                a2=-8.55099620e-06,
                a3=2.13210525e-09,
            ),
        ),
        "Fe3O4": SpeciesPropertyRecord(
            name="Iron(II,III) Oxide",
            phase="solid",
            mw=231.533e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=-1118380.0,
                a0=1.38896229e+02,
                a1=4.26480881e-01,
                a2=-5.95185534e-04,
                a3=2.27414606e-07,
            ),
        ),
        "Fe2O3": SpeciesPropertyRecord(
            name="Iron(III) Oxide",
            phase="solid",
            mw=159.687e-3,
            enthalpy=CpCubicMolar(
                h_form_ref=-824248.0,
                a0=9.67074082e+01,
                a1=2.51521981e-01,
                a2=-3.28757899e-04,
                a3=1.24389558e-07,
            ),
        ),
    }
)

DEFAULT_PROPERTY_REGISTRY = PROPERTY_REGISTRY
