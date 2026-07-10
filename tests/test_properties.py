import numpy as np
import pytest

import packed_bed.properties as properties
from packed_bed.properties import (
    PROPERTY_REGISTRY,
    PolynomialHeatCapacity,
    QuadraticViscosity,
    ShomateHeatCapacity,
    SpeciesProperties,
)


def test_polynomial_heat_capacity_uses_delta_temperature_coefficient_order() -> None:
    correlation = PolynomialHeatCapacity(
        coefficients=(1.0, 2.0, 3.0),
        t_ref=10.0,
        h_form_ref=5.0,
    )

    assert correlation.cp_value(12.0) == pytest.approx(17.0)
    assert correlation.value(12.0) == pytest.approx(19.0)
    assert correlation.cp_value(np.array([10.0, 12.0])).shape == (2,)


def test_polynomial_heat_capacity_rejects_structurally_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        PolynomialHeatCapacity(coefficients=())
    with pytest.raises(ValueError, match="must be finite"):
        PolynomialHeatCapacity(coefficients=(float("nan"),))


def test_genuinely_different_property_concepts_remain_explicit() -> None:
    assert ShomateHeatCapacity is not PolynomialHeatCapacity
    viscosity = QuadraticViscosity(a0=1.0, a1=2.0, a2=3.0, t_ref=10.0)
    assert viscosity.value(12.0) == pytest.approx(17.0)


def test_registry_uses_one_polynomial_implementation() -> None:
    assert all(
        isinstance(record, SpeciesProperties)
        and isinstance(record.enthalpy, PolynomialHeatCapacity)
        for record in PROPERTY_REGISTRY.records.values()
    )
    assert not hasattr(properties, "CpCubicMolar")
    assert not hasattr(properties, "CpQuarticMolar")
    assert not hasattr(properties, "ViscosityQuadratic")
