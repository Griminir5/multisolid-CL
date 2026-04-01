from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Mapping
import numpy as np

from daetools.pyDAE import Constant
from pyUnits import J, K, mol


class MolarEnthalpyCorrelation(ABC):
    """Build enthalpy as a native DAETOOLS expression."""

    @abstractmethod
    def dae_expression(self, temperature):
        raise NotImplementedError
    
    @abstractmethod
    def value(self, temperature):
        raise NotImplementedError

@dataclass(frozen=True)
class CpZerothMolar(MolarEnthalpyCorrelation):
    """Molar heat-capacity zeroth-order polynomial and enthalpy reference data in SI units."""

    t_ref: float = 298.15
    h_form_ref: float = 0.0
    a0: float = 0.0

    def dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        h_form = Constant(self.h_form_ref * J / mol)
        a0 = Constant(self.a0 * J / (mol * K))

        return h_form # incomplete
    
    def value(self, temperature):
        return self.h_form_ref # incomplete

@dataclass(frozen=True)
class CpQuadraticMolar(MolarEnthalpyCorrelation):
    """Molar heat-capacity quadratic polynomial and enthalpy reference data in SI units."""

    t_ref: float = 298.15
    h_form_ref: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0

    def dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        h_form = Constant(self.h_form_ref * J / mol)
        a0 = Constant(self.a0 * J / (mol * K))
        a1 = Constant(self.a1 * J / (mol * K**2))
        a2 = Constant(self.a2 * J / (mol * K**3))

        return h_form # incomplete
    
    def value(self, temperature):
        return self.h_form_ref # incomplete

@dataclass(frozen=True)
class CpCubicMolar(MolarEnthalpyCorrelation):
    """Molar heat-capacity cubic polynomial and enthalpy reference data in SI units."""

    t_ref: float = 298.15
    h_form_ref: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0

    def dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        h_form = Constant(self.h_form_ref * J / mol)
        a0 = Constant(self.a0 * J / (mol * K))
        a1 = Constant(self.a1 * J / (mol * K**2))
        a2 = Constant(self.a2 * J / (mol * K**3))
        a3 = Constant(self.a3 * J / (mol * K**4))

        return h_form # incomplete
    
    def value(self, temperature):
        return self.h_form_ref # incomplete

@dataclass(frozen=True)
class CpQuarticMolar(MolarEnthalpyCorrelation):
    """Molar heat-capacity cubic polynomial and enthalpy reference data in SI units."""

    t_ref: float = 298.15
    h_form_ref: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    a4: float = 0.0

    def dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        h_form = Constant(self.h_form_ref * J / mol)
        a0 = Constant(self.a0 * J / (mol * K))
        a1 = Constant(self.a1 * J / (mol * K**2))
        a2 = Constant(self.a2 * J / (mol * K**3))
        a3 = Constant(self.a3 * J / (mol * K**4))
        a4 = Constant(self.a4 * J / (mol * K**4))

        return h_form # incomplete
    
    def value(self, temperature):
        return self.h_form_ref # incomplete

@dataclass(frozen=True)
class CpPolyMolar(MolarEnthalpyCorrelation):
    """Molar heat-capacity polynomial and enthalpy reference data in SI units."""

    t_ref: float = 298.15
    h_form_ref: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    a4: float = 0.0

    def dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        h_form = Constant(self.h_form_ref * J / mol)
        a0 = Constant(self.a0 * J / (mol * K))
        a1 = Constant(self.a1 * J / (mol * K**2))
        a2 = Constant(self.a2 * J / (mol * K**3))
        a3 = Constant(self.a3 * J / (mol * K**4))
        a4 = Constant(self.a4 * J / (mol * K**5))

        return h_form # incomplete
    
    def value(self, temperature):
        return self.h_form_ref # incomplete


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

    def enthalpy_expression(self, species_name: str, temperature):
        return self.records[species_name].enthalpy.dae_expression(temperature)
    
    def enthalpy_value(self, species_name: str, temperature):
        return self.records[species_name].enthalpy.value(temperature)


DEFAULT_PROPERTY_REGISTRY = PropertyRegistry(
    records={
        "AR": SpeciesPropertyRecord(
            name="AR",
            phase="gas",
            mw=39.948e-3,
            enthalpy=CpZerothMolar(h_form_ref=0.0, a0=20.786),
        ),
        "CH4": SpeciesPropertyRecord(
            name="CH4",
            phase="gas",
            mw=16.043e-3,
            enthalpy=CpCubicMolar(h_form_ref=-74873.0, a0=37.6194088, a1=5.054146625e-2, a2=2.27606802e-6, a3=-6.82466733e-9),
        ),
        "CO": SpeciesPropertyRecord(
            name="CO",
            phase="gas",
            mw=28.010e-3,
            enthalpy=CpQuarticMolar(h_form_ref=-110541.0, a0=29.1010658, a1=1.92542547e-3, a2=1.18696459e-5, a3=-1.12435518e-8, a4=2.97502474e-12),
        ),
        "CO2": SpeciesPropertyRecord(
            name="CO2",
            phase="gas",
            mw=44.0095e-3,
            enthalpy=CpCubicMolar(h_form_ref=-393505.0, a0=37.0481649, a1=4.05995467e-2, a2=-2.72100644e-5, a3=6.84699347e-9),
        ),
        "H2": SpeciesPropertyRecord(
            name="H2",
            phase="gas",
            mw=2.01588e-3,
            enthalpy=CpCubicMolar(h_form_ref=0.0, a0=28.6285719, a1=3.87550311e-3, a2=-6.60085498e-6, a3=8.02717561e-9),
        ),
        "H2O": SpeciesPropertyRecord(
            name="H2O",
            phase="gas",
            mw=18.01528e-3,
            enthalpy=CpCubicMolar(h_form_ref=-241826.0, a0=33.7806686, a1=6.48912262e-3, a2=8.12757341e-6, a3=-3.53127971e-9),
        ),
        "HE": SpeciesPropertyRecord(
            name="HE",
            phase="gas",
            mw=4.002602e-3,
            enthalpy=CpZerothMolar(h_form_ref=0.0, a0=20.786),
        ),
        "N2": SpeciesPropertyRecord(
            name="N2",
            phase="gas",
            mw=28.0134e-3,
            enthalpy=CpCubicMolar(h_form_ref=0.0, a0=29.151445, a1=2.69178695e-3, a2=4.47573614e-6, a3=-2.23789023e-9),
        ),
        "O2": SpeciesPropertyRecord(
            name="O2",
            phase="gas",
            mw=31.9988e-3,
            enthalpy=CpQuadraticMolar(h_form_ref=0.0, a0=29.7404663, a1=8.82224917e-3, a2=-2.57179415e-6),
        ),
        "Solid": SpeciesPropertyRecord(
            name="Solid",
            phase="solid",
            enthalpy=CpPolyMolar(h_form_ref=0.0, a0=50.0),
        ),
        "NI": SpeciesPropertyRecord(
            name="Nickel",
            phase="solid",
            enthalpy=CpPolyMolar(h_form_ref=0.0, a0=50.0),
        ),
    }
)
