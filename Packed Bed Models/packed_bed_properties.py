from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Mapping

from daetools.pyDAE import Constant
from pyUnits import J, K, mol


class MolarEnthalpyCorrelation(ABC):
    """Build enthalpy as a native DAETOOLS expression."""

    @abstractmethod
    def dae_expression(self, temperature):
        raise NotImplementedError



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

        return h_form + ()


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


DEFAULT_PROPERTY_REGISTRY = PropertyRegistry(
    records={
        "AR": SpeciesPropertyRecord(
            name="AR",
            phase="gas",
            mw=39.948e-3,
            enthalpy=CpPolyMolar(h_form_ref=0.0, a0=20.786),
        ),
        "CH4": SpeciesPropertyRecord(
            name="CH4",
            phase="gas",
            mw=16.043e-3,
            enthalpy=CpPolyMolar(h_form_ref=-74850.0, a0=35.69),
        ),
        "CO": SpeciesPropertyRecord(
            name="CO",
            phase="gas",
            mw=28.010e-3,
            enthalpy=CpPolyMolar(h_form_ref=-110530.0, a0=29.14),
        ),
        "CO2": SpeciesPropertyRecord(
            name="CO2",
            phase="gas",
            mw=44.0095e-3,
            enthalpy=CpPolyMolar(h_form_ref=-393520.0, a0=37.13),
        ),
        "H2": SpeciesPropertyRecord(
            name="H2",
            phase="gas",
            mw=2.01588e-3,
            enthalpy=CpPolyMolar(h_form_ref=0.0, a0=28.84),
        ),
        "H2O": SpeciesPropertyRecord(
            name="H2O",
            phase="gas",
            mw=18.01528e-3,
            enthalpy=CpPolyMolar(h_form_ref=-241826.0, a0=33.58),
        ),
        "HE": SpeciesPropertyRecord(
            name="HE",
            phase="gas",
            mw=4.002602e-3,
            enthalpy=CpPolyMolar(h_form_ref=0.0, a0=20.786),
        ),
        "N2": SpeciesPropertyRecord(
            name="N2",
            phase="gas",
            mw=28.0134e-3,
            enthalpy=CpPolyMolar(h_form_ref=0.0, a0=29.12),
        ),
        "O2": SpeciesPropertyRecord(
            name="O2",
            phase="gas",
            mw=31.9988e-3,
            enthalpy=CpPolyMolar(h_form_ref=0.0, a0=29.38),
        ),
        "Solid": SpeciesPropertyRecord(
            name="Solid",
            phase="solid",
            enthalpy=CpPolyMolar(h_form_ref=0.0, a0=50.0),
        ),
    }
)
