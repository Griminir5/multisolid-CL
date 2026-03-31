import sys
from pathlib import Path

import numpy as np
from daetools.pyDAE import daeIDAS, daeNoOpDataReporter, daePythonStdOutLog


MODEL_DIR = Path(__file__).resolve().parents[1] / "Packed Bed Models"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from CLBed_MHMM import CLBed_mass, configure_compute_stack, simBed
from packed_bed_properties import CpPolyMolar, PropertyRegistry, SpeciesPropertyRecord


def initialize_and_solve_initial(simulation):
    configure_compute_stack()

    log = daePythonStdOutLog()
    log.PrintProgress = False
    solver = daeIDAS()
    reporter = daeNoOpDataReporter()

    simulation.ReportingInterval = 1.0
    simulation.TimeHorizon = 1.0
    simulation.Initialize(solver, reporter, log)
    simulation.SolveInitial()
    return simulation


def finalize_simulation(simulation):
    try:
        simulation.Finalize()
    except Exception:
        pass


def build_polynomial_test_registry():
    return PropertyRegistry(
        records={
            "TESTG": SpeciesPropertyRecord(
                name="TESTG",
                phase="gas",
                mw=0.028,
                enthalpy=CpPolyMolar(
                    t_ref=298.15,
                    h_form_ref=-123456.0,
                    a0=30.0,
                    a1=1.2e-2,
                    a2=-3.0e-5,
                    a3=7.0e-8,
                    a4=-2.0e-11,
                ),
            ),
            "TESTS": SpeciesPropertyRecord(
                name="TESTS",
                phase="solid",
                enthalpy=CpPolyMolar(
                    t_ref=298.15,
                    h_form_ref=-54321.0,
                    a0=60.0,
                    a1=-2.5e-2,
                    a2=4.0e-5,
                    a3=-1.5e-8,
                    a4=3.0e-12,
                ),
            ),
        }
    )


def assert_uniform(array, expected_value, atol=1e-8):
    values = np.asarray(array, dtype=float)
    if not np.allclose(values, expected_value, atol=atol, rtol=0.0):
        raise AssertionError(f"Expected uniform value {expected_value}, got {values}.")


def test_compute_stack_smoke():
    simulation = simBed()
    try:
        initialize_and_solve_initial(simulation)
        gas_enthalpy = np.asarray(simulation.model.h_gas_comp.npyValues, dtype=float)
        solid_enthalpy = np.asarray(simulation.model.h_sol_comp.npyValues, dtype=float)
        if gas_enthalpy.shape[0] != len(simulation.gas_species):
            raise AssertionError("Gas enthalpy array does not match the gas species count.")
        if solid_enthalpy.shape[0] != len(simulation.solid_species):
            raise AssertionError("Solid enthalpy array does not match the solid species count.")
        if not np.isfinite(gas_enthalpy).all():
            raise AssertionError("Gas enthalpy initialization produced non-finite values.")
        if not np.isfinite(solid_enthalpy).all():
            raise AssertionError("Solid enthalpy initialization produced non-finite values.")
    finally:
        finalize_simulation(simulation)


def test_reference_temperature_enthalpy():
    registry = build_polynomial_test_registry()
    reference_temperature = 298.15
    simulation = simBed(
        gas_species=["TESTG"],
        solid_species=["TESTS"],
        property_registry=registry,
        temperature_setpoint=reference_temperature,
    )
    try:
        initialize_and_solve_initial(simulation)
        assert_uniform(
            simulation.model.h_gas_comp.npyValues,
            registry.get_record("TESTG").enthalpy.h_form_ref,
        )
        assert_uniform(
            simulation.model.h_sol_comp.npyValues,
            registry.get_record("TESTS").enthalpy.h_form_ref,
        )
    finally:
        finalize_simulation(simulation)


def test_off_reference_polynomial_enthalpy():
    registry = build_polynomial_test_registry()
    evaluation_temperature = 650.0
    simulation = simBed(
        gas_species=["TESTG"],
        solid_species=["TESTS"],
        property_registry=registry,
        temperature_setpoint=evaluation_temperature,
    )
    try:
        initialize_and_solve_initial(simulation)
        expected_gas_enthalpy = registry.get_record("TESTG").enthalpy.enthalpy(evaluation_temperature)
        expected_solid_enthalpy = registry.get_record("TESTS").enthalpy.enthalpy(evaluation_temperature)

        assert_uniform(simulation.model.T.npyValues, evaluation_temperature)
        assert_uniform(simulation.model.h_gas_comp.npyValues, expected_gas_enthalpy)
        assert_uniform(simulation.model.h_sol_comp.npyValues, expected_solid_enthalpy)
    finally:
        finalize_simulation(simulation)


def test_bound_property_view_access():
    registry = build_polynomial_test_registry()
    gas_properties = registry.bind_species(["TESTG"], expected_phase="gas", require_enthalpy=True)
    solid_properties = registry.bind_species(["TESTS"], expected_phase="solid", require_enthalpy=True)

    expected_gas_enthalpy = registry.get_record("TESTG").enthalpy.value(650.0)
    expected_solid_enthalpy = registry.get_record("TESTS").enthalpy.value(650.0)

    if gas_properties.enthalpy_value(0, 650.0) != expected_gas_enthalpy:
        raise AssertionError("Bound gas property view returned the wrong enthalpy by index.")
    if gas_properties.enthalpy_value("TESTG", 650.0) != expected_gas_enthalpy:
        raise AssertionError("Bound gas property view returned the wrong enthalpy by name.")
    if solid_properties.enthalpy_value(0, 650.0) != expected_solid_enthalpy:
        raise AssertionError("Bound solid property view returned the wrong enthalpy by index.")


def test_missing_property_validation():
    try:
        CLBed_mass(
            "MissingProps",
            ["TESTG"],
            ["TESTS"],
            property_registry=PropertyRegistry(records={}),
        )
    except KeyError as exc:
        if "TESTG" not in str(exc):
            raise AssertionError(f"Missing-species error did not include the species name: {exc}")
    else:
        raise AssertionError("Expected missing property data to fail during model construction.")


def main():
    tests = [
        ("compute-stack smoke", test_compute_stack_smoke),
        ("reference enthalpy", test_reference_temperature_enthalpy),
        ("polynomial enthalpy", test_off_reference_polynomial_enthalpy),
        ("bound property view", test_bound_property_view_access),
        ("missing property validation", test_missing_property_validation),
    ]

    for name, test in tests:
        test()
        print(f"[ok] {name}")


if __name__ == "__main__":
    main()
