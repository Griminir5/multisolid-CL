from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


PolynomialCoefficients = tuple[float, ...]

def _as_float_array(temperature):
    return np.asarray(temperature, dtype=float)


def _dae_symbols():
    from daetools.pyDAE import Constant
    from pyUnits import J, K, Pa, mol, s

    return Constant, J, K, Pa, mol, s


class BaseCorrelation(ABC):
    @abstractmethod
    def dae_expression(self, temperature):
        raise NotImplementedError

    @abstractmethod
    def value(self, temperature):
        raise NotImplementedError


@dataclass(frozen=True)
class PolynomialHeatCapacity(BaseCorrelation):
    """Cp polynomial in powers of ``T - t_ref``, with SI molar coefficients."""

    coefficients: PolynomialCoefficients
    t_ref: float = 298.15
    h_form_ref: float = 0.0

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError("Polynomial heat capacity coefficients must not be empty.")
        if not all(math.isfinite(value) for value in (*self.coefficients, self.t_ref, self.h_form_ref)):
            raise ValueError("Polynomial heat capacity parameters must be finite.")

    def cp_dae_expression(self, temperature):
        Constant, J, K, _Pa, mol, _s = _dae_symbols()
        delta_temperature = temperature - Constant(self.t_ref * K)
        expression = Constant(0.0 * J / (mol * K))
        for power, coefficient in enumerate(self.coefficients):
            coefficient_units = J / (mol * K ** (power + 1))
            expression += Constant(coefficient * coefficient_units) * delta_temperature**power
        return expression

    def cp_value(self, temperature):
        delta_temperature = _as_float_array(temperature) - self.t_ref
        return np.polynomial.polynomial.polyval(delta_temperature, self.coefficients)

    def dae_expression(self, temperature):
        Constant, J, K, _Pa, mol, _s = _dae_symbols()
        delta_temperature = temperature - Constant(self.t_ref * K)
        expression = Constant(self.h_form_ref * J / mol)
        for power, coefficient in enumerate(self.coefficients, start=1):
            coefficient_units = J / (mol * K**power)
            expression += Constant((coefficient / power) * coefficient_units) * delta_temperature**power
        return expression

    def value(self, temperature):
        delta_temperature = _as_float_array(temperature) - self.t_ref
        integrated_coefficients = (self.h_form_ref,) + tuple(
            coefficient / power
            for power, coefficient in enumerate(self.coefficients, start=1)
        )
        return np.polynomial.polynomial.polyval(delta_temperature, integrated_coefficients)


@dataclass(frozen=True)
class ShomateHeatCapacity(BaseCorrelation):
    """Shomate Cp(T) basis with enthalpy anchored at h_form_ref when T = t_ref."""

    t_ref: float = 298.15
    h_form_ref: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    a4: float = 0.0

    def __post_init__(self) -> None:
        if self.t_ref <= 0.0 or not all(
            math.isfinite(value)
            for value in (self.t_ref, self.h_form_ref, self.a0, self.a1, self.a2, self.a3, self.a4)
        ):
            raise ValueError("Shomate heat capacity parameters must be finite with positive t_ref.")

    def cp_dae_expression(self, temperature):
        Constant, J, K, _Pa, mol, _s = _dae_symbols()
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
        Constant, J, K, _Pa, mol, _s = _dae_symbols()
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
class QuadraticViscosity(BaseCorrelation):
    t_ref: float = 1000.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.t_ref, self.a0, self.a1, self.a2)):
            raise ValueError("Quadratic viscosity parameters must be finite.")

    def dae_expression(self, temperature):
        Constant, _J, K, Pa, _mol, s = _dae_symbols()
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
class SpeciesProperties:
    """Property data for one species, stored in canonical SI units."""

    name: str
    phase: str
    mw: float | None = None
    enthalpy: BaseCorrelation | None = None
    viscosity: BaseCorrelation | None = None

    def __post_init__(self) -> None:
        if self.phase not in {"gas", "solid"}:
            raise ValueError(f"Unsupported phase '{self.phase}' for species '{self.name}'.")
        if self.mw is not None and (not math.isfinite(self.mw) or self.mw <= 0.0):
            raise ValueError(f"Molecular weight must be finite and positive for species '{self.name}'.")
        if self.phase != "gas":
            if self.viscosity is not None:
                raise ValueError(f"Gas viscosity is only valid for gas species ('{self.name}').")


@dataclass(frozen=True)
class PropertyRegistry:
    records: Mapping[str, SpeciesProperties]

    def __post_init__(self) -> None:
        sample_temperature = np.array([300.0, 1000.0])
        for species_id, record in self.records.items():
            if not species_id or species_id != species_id.strip():
                raise ValueError("Property registry species identifiers must not be blank or padded.")
            for property_name in ("enthalpy", "viscosity"):
                correlation = getattr(record, property_name)
                if correlation is None:
                    continue
                if not callable(getattr(correlation, "value", None)) or not callable(
                    getattr(correlation, "dae_expression", None)
                ):
                    raise ValueError(
                        f"Species '{species_id}' {property_name} correlation has an invalid interface."
                    )
                values = np.asarray(correlation.value(sample_temperature), dtype=float)
                if values.shape != sample_temperature.shape or not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Species '{species_id}' {property_name} correlation returned incompatible values."
                    )

    def has_species(self, species_id: str) -> bool:
        return species_id in self.records

    def species_ids(self, phase: str | None = None) -> tuple[str, ...]:
        if phase is None:
            return tuple(self.records.keys())
        return tuple(species_id for species_id, record in self.records.items() if record.phase == phase)

    def get_record(self, species_id: str) -> SpeciesProperties:
        try:
            return self.records[species_id]
        except KeyError as exc:
            available = ", ".join(self.records.keys())
            raise KeyError(f"Unknown species '{species_id}'. Available species: {available}") from exc

    def require_species(self, species_id: str, phase: str | None = None) -> SpeciesProperties:
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


def _polynomial_enthalpy(h_form_ref: float, *coefficients: float) -> PolynomialHeatCapacity:
    return PolynomialHeatCapacity(coefficients=coefficients, h_form_ref=h_form_ref)


def _quadratic_viscosity(a0: float, a1: float, a2: float) -> QuadraticViscosity:
    return QuadraticViscosity(a0=a0, a1=a1, a2=a2)


PROPERTY_REGISTRY = PropertyRegistry(
    records={
        "Ar": SpeciesProperties(
            "Argon", "gas", 39.948e-3,
            _polynomial_enthalpy(0.0, 20.786),
            _quadratic_viscosity(5.56703098e-05, 3.86114742e-08, -8.57834841e-12),
        ),
        "CH4": SpeciesProperties(
            "Methane", "gas", 16.043e-3,
            _polynomial_enthalpy(-74873.0, 3.65894663e01, 5.41242609e-02, 3.72168361e-06, -1.07088462e-08),
            _quadratic_viscosity(2.80097001e-05, 2.02172611e-08, -2.79160801e-12),
        ),
        "CO": SpeciesProperties(
            "Carbon Monoxide", "gas", 28.010e-3,
            _polynomial_enthalpy(-110541.0, 2.87504130e01, 5.05074872e-03, 3.90907850e-06, -3.09307094e-09),
            _quadratic_viscosity(4.05943891e-05, 2.58571235e-08, -6.43543956e-12),
        ),
        "CO2": SpeciesProperties(
            "Carbon Dioxide", "gas", 44.0095e-3,
            _polynomial_enthalpy(-393505.0, 3.81416029e01, 3.59619451e-02, -2.28055090e-05, 5.90960724e-09),
            _quadratic_viscosity(3.99550354e-05, 2.79482484e-08, -8.23314636e-12),
        ),
        "H2": SpeciesProperties(
            "Hydrogen", "gas", 2.01588e-3,
            _polynomial_enthalpy(0.0, 2.94409905e01, -2.38377533e-03, 6.39601662e-06, -2.03147561e-09),
            _quadratic_viscosity(2.04091133e-05, 1.41343819e-08, -2.34255119e-12),
        ),
        "H2O": SpeciesProperties(
            "Water", "gas", 18.01528e-3,
            _polynomial_enthalpy(-241826.0, 3.34558812e01, 7.59842908e-03, 7.65622945e-06, -3.77998371e-09),
            _quadratic_viscosity(3.77371020e-05, 4.16760701e-08, 1.73653811e-12),
        ),
        "He": SpeciesProperties(
            "Helium", "gas", 4.002602e-3,
            _polynomial_enthalpy(0.0, 20.786),
            _quadratic_viscosity(4.62481858e-05, 3.30037729e-08, -4.84571869e-12),
        ),
        "N2": SpeciesProperties(
            "Nitrogen", "gas", 28.0134e-3,
            _polynomial_enthalpy(0.0, 2.87990411e01, 3.12879912e-03, 6.03928488e-06, -3.77238986e-09),
            _quadratic_viscosity(4.15082118e-05, 2.78501267e-08, -6.09391431e-12),
        ),
        "O2": SpeciesProperties(
            "Oxygen", "gas", 31.9988e-3,
            _polynomial_enthalpy(0.0, 2.88034459e01, 1.32893606e-02, -7.91100675e-06, 1.82341564e-09),
            _quadratic_viscosity(4.92257601e-05, 3.26807190e-08, -8.12639449e-12),
        ),
        "Ni": SpeciesProperties(
            "Nickel", "solid", 58.693e-3,
            _polynomial_enthalpy(0.0, 3.22350875e01, -1.59194574e-02, 3.55975554e-05, -1.62743281e-08),
        ),
        "NiO": SpeciesProperties(
            "Nickel Oxide", "solid", 74.6928e-3,
            _polynomial_enthalpy(-239701.0, 5.64774634e01, -1.56343578e-02, 2.10045988e-05, -4.78601077e-09),
        ),
        "CaAl2O4": SpeciesProperties(
            "Calcium Aluminate", "solid", 158.039e-3,
            _polynomial_enthalpy(-2326300.0, 1.31025407e02, 1.07694151e-01, -9.18337717e-05, 3.37682975e-08),
        ),
        "Cu": SpeciesProperties(
            "Copper", "solid", 63.55e-3,
            _polynomial_enthalpy(0.0, 2.43826543e01, 1.06212521e-02, -1.53809529e-05, 1.25658461e-08),
        ),
        "Cu2O": SpeciesProperties(
            "Copper(I) Oxide", "solid", 143.091e-3,
            _polynomial_enthalpy(-170707.0, 6.31729851e01, 4.50982629e-02, -4.40775011e-05, 2.44874168e-08),
        ),
        "CuO": SpeciesProperties(
            "Copper(II) Oxide", "solid", 79.545e-3,
            _polynomial_enthalpy(-156063.0, 4.28595032e01, 3.80740189e-02, -4.25436164e-05, 1.92484328e-08),
        ),
        "Al2O3": SpeciesProperties(
            "Aluminium Oxide", "solid", 101.961e-3,
            _polynomial_enthalpy(-1675700.0, 7.9100e01, 0.0, 0.0, 0.0),
        ),
        "CuAlO2": SpeciesProperties(
            "Copper(I) Aluminate", "solid", 122.526e-3,
            _polynomial_enthalpy(-923200.0, 1.0000e02, 0.0, 0.0, 0.0),
        ),
        "CuAl2O4": SpeciesProperties(
            "Copper(II) Aluminate", "solid", 181.508e-3,
            _polynomial_enthalpy(-1831800.0, 1.4000e02, 0.0, 0.0, 0.0),
        ),
        "Fe": SpeciesProperties(
            "Iron", "solid", 55.845e-3,
            _polynomial_enthalpy(0.0, 1.52959946e01, 1.05151203e-01, -1.20394019e-04, 4.13092928e-08),
        ),
        "FeO": SpeciesProperties(
            "Iron(II) Oxide", "solid", 71.844e-3,
            _polynomial_enthalpy(-272044.0, 5.00153501e01, 1.83032040e-02, -8.55099620e-06, 2.13210525e-09),
        ),
        "Fe3O4": SpeciesProperties(
            "Iron(II,III) Oxide", "solid", 231.533e-3,
            _polynomial_enthalpy(-1118380.0, 1.38896229e02, 4.26480881e-01, -5.95185534e-04, 2.27414606e-07),
        ),
        "Fe2O3": SpeciesProperties(
            "Iron(III) Oxide", "solid", 159.687e-3,
            _polynomial_enthalpy(-824248.0, 9.67074082e01, 2.51521981e-01, -3.28757899e-04, 1.24389558e-07),
        ),
    }
)

DEFAULT_PROPERTY_REGISTRY = PROPERTY_REGISTRY
