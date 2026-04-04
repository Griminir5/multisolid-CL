import argparse
import gc
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from daetools.pyDAE import *
from pyUnits import J, K, W, mol


MODEL_DIR = Path(__file__).resolve().parents[1] / "Packed Bed Models"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from packed_bed_properties import CpShomateMolar


temperature_type = daeVariableType(
    name="benchmark_temperature_type",
    units=K,
    lowerBound=200.0,
    upperBound=2000.0,
    initialGuess=700.0,
    absTolerance=1e-6,
)
molar_heat_capacity_type = daeVariableType(
    name="benchmark_molar_heat_capacity_type",
    units=J / (mol * K),
    lowerBound=0.0,
    upperBound=1e6,
    initialGuess=100.0,
    absTolerance=1e-6,
)
molar_enthalpy_type = daeVariableType(
    name="benchmark_molar_enthalpy_type",
    units=J / mol,
    lowerBound=-1e8,
    upperBound=1e8,
    initialGuess=0.0,
    absTolerance=1e-6,
)


def _as_float_array(temperature):
    return np.asarray(temperature, dtype=float)


@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    phase: str
    fraction: float
    correlation: object


@dataclass(frozen=True)
class CorrelationSet:
    gas_species: tuple[SpeciesSpec, ...]
    solid_species: tuple[SpeciesSpec, ...]


@dataclass(frozen=True)
class FitErrorSummary:
    order: int
    max_cp_abs: float
    max_cp_rel: float
    max_h_abs: float
    max_h_rel: float


@dataclass(frozen=True)
class NumPyBenchmarkResult:
    label: str
    order: int | None
    cp_seconds: float
    h_seconds: float
    combined_seconds: float


@dataclass(frozen=True)
class DAEBenchmarkResult:
    label: str
    order: int | None
    initialize_solve_initial_seconds: float
    run_seconds: float
    total_seconds: float


class CpPolynomialMolar:
    """Generic Cp(theta) polynomial with theta = (T - T_ref) / temperature_scale."""

    def __init__(self, coefficients, *, t_ref=298.15, h_form_ref=0.0, temperature_scale=1000.0):
        coeffs = tuple(float(value) for value in coefficients)
        if not coeffs:
            raise ValueError("At least one polynomial coefficient is required.")

        self.coefficients = coeffs
        self.t_ref = float(t_ref)
        self.h_form_ref = float(h_form_ref)
        self.temperature_scale = float(temperature_scale)
        self._integral_coefficients = tuple(
            coeff / (index + 1) for index, coeff in enumerate(self.coefficients)
        )

    def cp_value(self, temperature):
        theta = (_as_float_array(temperature) - self.t_ref) / self.temperature_scale
        result = np.zeros_like(theta, dtype=float) + self.coefficients[-1]
        for coefficient in reversed(self.coefficients[:-1]):
            result = coefficient + theta * result
        return result

    def value(self, temperature):
        theta = (_as_float_array(temperature) - self.t_ref) / self.temperature_scale
        result = np.zeros_like(theta, dtype=float) + self._integral_coefficients[-1]
        for coefficient in reversed(self._integral_coefficients[:-1]):
            result = coefficient + theta * result
        return self.h_form_ref + self.temperature_scale * theta * result

    def cp_dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        temp_scale = Constant(self.temperature_scale * K)
        theta = (temperature - t_ref) / temp_scale
        expression = Constant(self.coefficients[-1] * J / (mol * K))
        for power in range(len(self.coefficients) - 2, -1, -1):
            coefficient = Constant(self.coefficients[power] * J / (mol * K))
            expression = coefficient + theta * expression
        return expression

    def dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        h_form = Constant(self.h_form_ref * J / mol)
        temp_scale = Constant(self.temperature_scale * K)
        theta = (temperature - t_ref) / temp_scale
        expression = Constant(self._integral_coefficients[-1] * J / (mol * K))
        for power in range(len(self._integral_coefficients) - 2, -1, -1):
            coefficient = Constant(self._integral_coefficients[power] * J / (mol * K))
            expression = coefficient + theta * expression
        return h_form + temp_scale * theta * expression


def build_reference_shomate_set():
    return CorrelationSet(
        gas_species=(
            SpeciesSpec(
                name="GAS_A",
                phase="gas",
                fraction=0.55,
                correlation=CpShomateMolar(
                    h_form_ref=-110541.0,
                    a0=30.15,
                    a1=6.80,
                    a2=-1.15,
                    a3=0.22,
                    a4=0.08,
                ),
            ),
            SpeciesSpec(
                name="GAS_B",
                phase="gas",
                fraction=0.25,
                correlation=CpShomateMolar(
                    h_form_ref=-74873.0,
                    a0=33.10,
                    a1=7.25,
                    a2=-1.70,
                    a3=0.34,
                    a4=0.05,
                ),
            ),
            SpeciesSpec(
                name="GAS_C",
                phase="gas",
                fraction=0.20,
                correlation=CpShomateMolar(
                    h_form_ref=-241826.0,
                    a0=35.20,
                    a1=8.40,
                    a2=-2.10,
                    a3=0.31,
                    a4=0.09,
                ),
            ),
        ),
        solid_species=(
            SpeciesSpec(
                name="SOLID_A",
                phase="solid",
                fraction=0.65,
                correlation=CpShomateMolar(
                    h_form_ref=-239701.0,
                    a0=179.38973769,
                    a1=-300.19583295,
                    a2=246.69888057,
                    a3=-65.90651588,
                    a4=-6.35461864,
                ),
            ),
            SpeciesSpec(
                name="SOLID_B",
                phase="solid",
                fraction=0.35,
                correlation=CpShomateMolar(
                    h_form_ref=-2326304.0,
                    a0=154.055548,
                    a1=22.3001808,
                    a2=-2.47833922e-4,
                    a3=2.56391177e-4,
                    a4=-3.54806056,
                ),
            ),
        ),
    )


def fit_polynomial_to_enthalpy_and_cp(
    temperature_grid,
    correlation,
    order,
    *,
    cp_weight=1.0,
    h_weight=5.0,
    temperature_scale=1000.0,
):
    if order < 0:
        raise ValueError("Polynomial order must be non-negative.")

    temperatures = np.asarray(temperature_grid, dtype=float)
    cp_data = correlation.cp_value(temperatures)
    h_data = correlation.value(temperatures)
    theta = (temperatures - correlation.t_ref) / temperature_scale

    h_scale = max(np.std(h_data), 1.0)
    cp_scale = max(np.std(cp_data), 1.0)

    cp_basis = np.column_stack([theta**power for power in range(order + 1)])
    h_basis = np.column_stack(
        [
            temperature_scale * theta ** (power + 1) / (power + 1)
            for power in range(order + 1)
        ]
    )

    lhs = np.vstack(
        [
            (cp_weight / cp_scale) * cp_basis,
            (h_weight / h_scale) * h_basis,
        ]
    )
    rhs = np.concatenate(
        [
            (cp_weight / cp_scale) * cp_data,
            (h_weight / h_scale) * (h_data - correlation.h_form_ref),
        ]
    )

    coefficients, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return CpPolynomialMolar(
        coefficients=coefficients,
        t_ref=correlation.t_ref,
        h_form_ref=correlation.h_form_ref,
        temperature_scale=temperature_scale,
    )


def build_polynomial_set(reference_set, order, fit_temperatures):
    return CorrelationSet(
        gas_species=tuple(
            SpeciesSpec(
                name=species.name,
                phase=species.phase,
                fraction=species.fraction,
                correlation=fit_polynomial_to_enthalpy_and_cp(
                    fit_temperatures,
                    species.correlation,
                    order,
                ),
            )
            for species in reference_set.gas_species
        ),
        solid_species=tuple(
            SpeciesSpec(
                name=species.name,
                phase=species.phase,
                fraction=species.fraction,
                correlation=fit_polynomial_to_enthalpy_and_cp(
                    fit_temperatures,
                    species.correlation,
                    order,
                ),
            )
            for species in reference_set.solid_species
        ),
    )


def summarize_fit_errors(reference_set, candidate_set, validation_temperatures, order):
    max_cp_abs = 0.0
    max_cp_rel = 0.0
    max_h_abs = 0.0
    max_h_rel = 0.0

    for reference_species, candidate_species in zip(
        reference_set.gas_species + reference_set.solid_species,
        candidate_set.gas_species + candidate_set.solid_species,
    ):
        cp_ref = reference_species.correlation.cp_value(validation_temperatures)
        cp_fit = candidate_species.correlation.cp_value(validation_temperatures)
        h_ref = reference_species.correlation.value(validation_temperatures)
        h_fit = candidate_species.correlation.value(validation_temperatures)

        cp_abs = np.max(np.abs(cp_fit - cp_ref))
        h_abs = np.max(np.abs(h_fit - h_ref))
        cp_rel = np.max(np.abs((cp_fit - cp_ref) / np.maximum(np.abs(cp_ref), 1e-12)))
        h_rel = np.max(np.abs((h_fit - h_ref) / np.maximum(np.abs(h_ref), 1e-12)))

        max_cp_abs = max(max_cp_abs, float(cp_abs))
        max_cp_rel = max(max_cp_rel, float(cp_rel))
        max_h_abs = max(max_h_abs, float(h_abs))
        max_h_rel = max(max_h_rel, float(h_rel))

    return FitErrorSummary(
        order=order,
        max_cp_abs=max_cp_abs,
        max_cp_rel=max_cp_rel,
        max_h_abs=max_h_abs,
        max_h_rel=max_h_rel,
    )


def _evaluate_cp_workload(correlation_set, temperatures):
    total = np.zeros_like(temperatures, dtype=float)
    for species in correlation_set.gas_species:
        total += species.fraction * species.correlation.cp_value(temperatures)
    for species in correlation_set.solid_species:
        total += species.fraction * species.correlation.cp_value(temperatures)
    return total


def _evaluate_h_workload(correlation_set, temperatures):
    total = np.zeros_like(temperatures, dtype=float)
    for species in correlation_set.gas_species:
        total += species.fraction * species.correlation.value(temperatures)
    for species in correlation_set.solid_species:
        total += species.fraction * species.correlation.value(temperatures)
    return total


def _evaluate_combined_workload(correlation_set, temperatures):
    cp_total = np.zeros_like(temperatures, dtype=float)
    h_total = np.zeros_like(temperatures, dtype=float)
    for species in correlation_set.gas_species:
        cp_total += species.fraction * species.correlation.cp_value(temperatures)
        h_total += species.fraction * species.correlation.value(temperatures)
    for species in correlation_set.solid_species:
        cp_total += species.fraction * species.correlation.cp_value(temperatures)
        h_total += species.fraction * species.correlation.value(temperatures)
    return cp_total, h_total


def benchmark_callable(function, repeats):
    timings = []
    function()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = perf_counter()
            function()
            timings.append(perf_counter() - start)
    finally:
        if gc_was_enabled:
            gc.enable()
    return statistics.median(timings)


def run_numpy_benchmarks(reference_set, polynomial_sets, temperatures, repeats):
    results = []

    shomate_cp_time = benchmark_callable(
        lambda: _evaluate_cp_workload(reference_set, temperatures),
        repeats,
    )
    shomate_h_time = benchmark_callable(
        lambda: _evaluate_h_workload(reference_set, temperatures),
        repeats,
    )
    shomate_combined_time = benchmark_callable(
        lambda: _evaluate_combined_workload(reference_set, temperatures),
        repeats,
    )
    results.append(
        NumPyBenchmarkResult(
            label="Shomate",
            order=None,
            cp_seconds=shomate_cp_time,
            h_seconds=shomate_h_time,
            combined_seconds=shomate_combined_time,
        )
    )

    for order, polynomial_set in polynomial_sets.items():
        results.append(
            NumPyBenchmarkResult(
                label=f"Polynomial n={order}",
                order=order,
                cp_seconds=benchmark_callable(
                    lambda correlation_set=polynomial_set: _evaluate_cp_workload(
                        correlation_set, temperatures
                    ),
                    repeats,
                ),
                h_seconds=benchmark_callable(
                    lambda correlation_set=polynomial_set: _evaluate_h_workload(
                        correlation_set, temperatures
                    ),
                    repeats,
                ),
                combined_seconds=benchmark_callable(
                    lambda correlation_set=polynomial_set: _evaluate_combined_workload(
                        correlation_set, temperatures
                    ),
                    repeats,
                ),
            )
        )

    return results


class EnergyChainBenchmarkModel(daeModel):
    def __init__(self, name, correlation_set, n_cells, parent=None, description="Energy benchmark chain"):
        daeModel.__init__(self, name, parent, description)

        self.correlation_set = correlation_set
        self.gas_species = list(correlation_set.gas_species)
        self.solid_species = list(correlation_set.solid_species)

        self.Cells = daeDomain("Cells", self, dimless, "Axial energy cells")
        self.GasSpecies = daeDomain("GasSpecies", self, dimless, "Gas species")
        self.SolidSpecies = daeDomain("SolidSpecies", self, dimless, "Solid species")
        self.Cells.CreateArray(n_cells)
        self.GasSpecies.CreateArray(len(self.gas_species))
        self.SolidSpecies.CreateArray(len(self.solid_species))

        self.GasHoldup = daeParameter("GasHoldup", mol, self, "Gas holdup per cell")
        self.SolidHoldup = daeParameter("SolidHoldup", mol, self, "Solid holdup per cell")
        self.GasFlow = daeParameter("GasFlow", mol / s, self, "Gas molar flow")
        self.SolidFlow = daeParameter("SolidFlow", mol / s, self, "Solid molar flow")
        self.UA = daeParameter("UA", W / K, self, "Interphase heat transfer coefficient")
        self.GasInletTemperature = daeParameter("GasInletTemperature", K, self, "Gas inlet temperature")
        self.SolidInletTemperature = daeParameter("SolidInletTemperature", K, self, "Solid inlet temperature")

        self.y_gas = daeParameter(
            "y_gas",
            dimless,
            self,
            "Gas species mole fractions",
            [self.GasSpecies],
        )
        self.y_solid = daeParameter(
            "y_solid",
            dimless,
            self,
            "Solid species pseudo-mole fractions",
            [self.SolidSpecies],
        )

        self.T_gas = daeVariable(
            "T_gas",
            temperature_type,
            self,
            "Gas temperature in each cell",
            [self.Cells],
        )
        self.T_solid = daeVariable(
            "T_solid",
            temperature_type,
            self,
            "Solid temperature in each cell",
            [self.Cells],
        )

        self.cp_gas = daeVariable(
            "cp_gas",
            molar_heat_capacity_type,
            self,
            "Gas species heat capacities",
            [self.GasSpecies, self.Cells],
        )
        self.h_gas = daeVariable(
            "h_gas",
            molar_enthalpy_type,
            self,
            "Gas species enthalpies",
            [self.GasSpecies, self.Cells],
        )
        self.cp_solid = daeVariable(
            "cp_solid",
            molar_heat_capacity_type,
            self,
            "Solid species heat capacities",
            [self.SolidSpecies, self.Cells],
        )
        self.h_solid = daeVariable(
            "h_solid",
            molar_enthalpy_type,
            self,
            "Solid species enthalpies",
            [self.SolidSpecies, self.Cells],
        )

        self.cp_mix_gas = daeVariable(
            "cp_mix_gas",
            molar_heat_capacity_type,
            self,
            "Gas mixture heat capacity",
            [self.Cells],
        )
        self.h_mix_gas = daeVariable(
            "h_mix_gas",
            molar_enthalpy_type,
            self,
            "Gas mixture enthalpy",
            [self.Cells],
        )
        self.cp_mix_solid = daeVariable(
            "cp_mix_solid",
            molar_heat_capacity_type,
            self,
            "Solid mixture heat capacity",
            [self.Cells],
        )
        self.h_mix_solid = daeVariable(
            "h_mix_solid",
            molar_enthalpy_type,
            self,
            "Solid mixture enthalpy",
            [self.Cells],
        )

    def _weighted_sum(self, parameter, variable, count, cell_index, units):
        expression = Constant(0 * units)
        for species_index in range(count):
            expression += parameter(species_index) * variable(species_index, cell_index)
        return expression

    def _gas_inlet_enthalpy_expression(self):
        expression = Constant(0 * J / mol)
        for species_index, species in enumerate(self.gas_species):
            expression += self.y_gas(species_index) * species.correlation.dae_expression(
                self.GasInletTemperature()
            )
        return expression

    def _solid_inlet_enthalpy_expression(self):
        expression = Constant(0 * J / mol)
        for species_index, species in enumerate(self.solid_species):
            expression += self.y_solid(species_index) * species.correlation.dae_expression(
                self.SolidInletTemperature()
            )
        return expression

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        n_cells = self.Cells.NumberOfPoints
        n_gas = self.GasSpecies.NumberOfPoints
        n_solid = self.SolidSpecies.NumberOfPoints

        for gas_index, species in enumerate(self.gas_species):
            for cell_index in range(n_cells):
                eq = self.CreateEquation(f"gas_cp_{species.name}_{cell_index}")
                eq.Residual = self.cp_gas(gas_index, cell_index) - species.correlation.cp_dae_expression(
                    self.T_gas(cell_index)
                )

                eq = self.CreateEquation(f"gas_h_{species.name}_{cell_index}")
                eq.Residual = self.h_gas(gas_index, cell_index) - species.correlation.dae_expression(
                    self.T_gas(cell_index)
                )

        for solid_index, species in enumerate(self.solid_species):
            for cell_index in range(n_cells):
                eq = self.CreateEquation(f"solid_cp_{species.name}_{cell_index}")
                eq.Residual = self.cp_solid(
                    solid_index,
                    cell_index,
                ) - species.correlation.cp_dae_expression(self.T_solid(cell_index))

                eq = self.CreateEquation(f"solid_h_{species.name}_{cell_index}")
                eq.Residual = self.h_solid(
                    solid_index,
                    cell_index,
                ) - species.correlation.dae_expression(self.T_solid(cell_index))

        for cell_index in range(n_cells):
            eq = self.CreateEquation(f"cp_mix_gas_{cell_index}")
            eq.Residual = self.cp_mix_gas(cell_index) - self._weighted_sum(
                self.y_gas,
                self.cp_gas,
                n_gas,
                cell_index,
                J / (mol * K),
            )

            eq = self.CreateEquation(f"h_mix_gas_{cell_index}")
            eq.Residual = self.h_mix_gas(cell_index) - self._weighted_sum(
                self.y_gas,
                self.h_gas,
                n_gas,
                cell_index,
                J / mol,
            )

            eq = self.CreateEquation(f"cp_mix_solid_{cell_index}")
            eq.Residual = self.cp_mix_solid(cell_index) - self._weighted_sum(
                self.y_solid,
                self.cp_solid,
                n_solid,
                cell_index,
                J / (mol * K),
            )

            eq = self.CreateEquation(f"h_mix_solid_{cell_index}")
            eq.Residual = self.h_mix_solid(cell_index) - self._weighted_sum(
                self.y_solid,
                self.h_solid,
                n_solid,
                cell_index,
                J / mol,
            )

        gas_inlet_enthalpy = self._gas_inlet_enthalpy_expression()
        solid_inlet_enthalpy = self._solid_inlet_enthalpy_expression()

        for cell_index in range(n_cells):
            gas_upstream_enthalpy = (
                gas_inlet_enthalpy if cell_index == 0 else self.h_mix_gas(cell_index - 1)
            )
            solid_upstream_enthalpy = (
                solid_inlet_enthalpy if cell_index == 0 else self.h_mix_solid(cell_index - 1)
            )

            eq = self.CreateEquation(f"gas_energy_{cell_index}")
            eq.Residual = (
                self.GasHoldup() * self.cp_mix_gas(cell_index) * dt(self.T_gas(cell_index))
                - self.GasFlow() * (gas_upstream_enthalpy - self.h_mix_gas(cell_index))
                - self.UA() * (self.T_solid(cell_index) - self.T_gas(cell_index))
            )

            eq = self.CreateEquation(f"solid_energy_{cell_index}")
            eq.Residual = (
                self.SolidHoldup() * self.cp_mix_solid(cell_index) * dt(self.T_solid(cell_index))
                - self.SolidFlow() * (solid_upstream_enthalpy - self.h_mix_solid(cell_index))
                - self.UA() * (self.T_gas(cell_index) - self.T_solid(cell_index))
            )


class EnergyChainBenchmarkSimulation(daeSimulation):
    def __init__(self, correlation_set, n_cells):
        daeSimulation.__init__(self)
        self.correlation_set = correlation_set
        self.n_cells = n_cells
        self.model = EnergyChainBenchmarkModel("EnergyChainBenchmark", correlation_set, n_cells)

    def SetUpParametersAndDomains(self):
        self.model.GasHoldup.SetValue(6.0 * mol)
        self.model.SolidHoldup.SetValue(120.0 * mol)
        self.model.GasFlow.SetValue(1.8 * mol / s)
        self.model.SolidFlow.SetValue(0.35 * mol / s)
        self.model.UA.SetValue(45.0 * W / K)
        self.model.GasInletTemperature.SetValue(760.0 * K)
        self.model.SolidInletTemperature.SetValue(990.0 * K)
        self.model.y_gas.SetValues(
            np.asarray([species.fraction for species in self.correlation_set.gas_species], dtype=float)
        )
        self.model.y_solid.SetValues(
            np.asarray([species.fraction for species in self.correlation_set.solid_species], dtype=float)
        )

    def SetUpVariables(self):
        for cell_index in range(self.n_cells):
            gas_temperature = 930.0 - 12.0 * cell_index
            solid_temperature = 1080.0 - 10.0 * cell_index

            self.model.T_gas.SetInitialCondition(cell_index, gas_temperature * K)
            self.model.T_solid.SetInitialCondition(cell_index, solid_temperature * K)

            gas_cp_mix = 0.0
            gas_h_mix = 0.0
            for species_index, species in enumerate(self.correlation_set.gas_species):
                cp_value = float(species.correlation.cp_value(gas_temperature))
                h_value = float(species.correlation.value(gas_temperature))
                gas_cp_mix += species.fraction * cp_value
                gas_h_mix += species.fraction * h_value
                self.model.cp_gas.SetInitialGuess(
                    species_index,
                    cell_index,
                    cp_value * J / (mol * K),
                )
                self.model.h_gas.SetInitialGuess(
                    species_index,
                    cell_index,
                    h_value * J / mol,
                )

            solid_cp_mix = 0.0
            solid_h_mix = 0.0
            for species_index, species in enumerate(self.correlation_set.solid_species):
                cp_value = float(species.correlation.cp_value(solid_temperature))
                h_value = float(species.correlation.value(solid_temperature))
                solid_cp_mix += species.fraction * cp_value
                solid_h_mix += species.fraction * h_value
                self.model.cp_solid.SetInitialGuess(
                    species_index,
                    cell_index,
                    cp_value * J / (mol * K),
                )
                self.model.h_solid.SetInitialGuess(
                    species_index,
                    cell_index,
                    h_value * J / mol,
                )

            self.model.cp_mix_gas.SetInitialGuess(cell_index, gas_cp_mix * J / (mol * K))
            self.model.h_mix_gas.SetInitialGuess(cell_index, gas_h_mix * J / mol)
            self.model.cp_mix_solid.SetInitialGuess(cell_index, solid_cp_mix * J / (mol * K))
            self.model.h_mix_solid.SetInitialGuess(cell_index, solid_h_mix * J / mol)


def configure_compute_stack():
    cfg = daeGetConfig()
    cfg.SetString("daetools.core.equations.evaluationMode", "computeStack_OpenMP")


def run_dae_case(correlation_set, *, n_cells, time_horizon, reporting_interval):
    configure_compute_stack()

    simulation = EnergyChainBenchmarkSimulation(correlation_set, n_cells)
    simulation.ReportingInterval = reporting_interval
    simulation.TimeHorizon = time_horizon
    simulation.ReportTimeDerivatives = False

    solver = daeIDAS()
    reporter = daeNoOpDataReporter()
    log = daeBaseLog()

    initialize_start = perf_counter()
    simulation.Initialize(solver, reporter, log)
    simulation.SolveInitial()
    initialize_elapsed = perf_counter() - initialize_start

    run_start = perf_counter()
    simulation.Run()
    run_elapsed = perf_counter() - run_start

    try:
        simulation.Finalize()
    except Exception:
        pass

    return initialize_elapsed, run_elapsed


def benchmark_dae_case(correlation_set, *, n_cells, time_horizon, reporting_interval, repeats):
    run_dae_case(
        correlation_set,
        n_cells=n_cells,
        time_horizon=time_horizon,
        reporting_interval=reporting_interval,
    )

    initialize_times = []
    run_times = []
    for _ in range(repeats):
        initialize_elapsed, run_elapsed = run_dae_case(
            correlation_set,
            n_cells=n_cells,
            time_horizon=time_horizon,
            reporting_interval=reporting_interval,
        )
        initialize_times.append(initialize_elapsed)
        run_times.append(run_elapsed)

    initialize_median = statistics.median(initialize_times)
    run_median = statistics.median(run_times)
    return initialize_median, run_median, initialize_median + run_median


def run_dae_benchmarks(reference_set, polynomial_sets, *, n_cells, time_horizon, reporting_interval, repeats):
    results = []
    initialize_time, run_time, total_time = benchmark_dae_case(
        reference_set,
        n_cells=n_cells,
        time_horizon=time_horizon,
        reporting_interval=reporting_interval,
        repeats=repeats,
    )
    results.append(
        DAEBenchmarkResult(
            label="Shomate",
            order=None,
            initialize_solve_initial_seconds=initialize_time,
            run_seconds=run_time,
            total_seconds=total_time,
        )
    )

    for order, polynomial_set in polynomial_sets.items():
        initialize_time, run_time, total_time = benchmark_dae_case(
            polynomial_set,
            n_cells=n_cells,
            time_horizon=time_horizon,
            reporting_interval=reporting_interval,
            repeats=repeats,
        )
        results.append(
            DAEBenchmarkResult(
                label=f"Polynomial n={order}",
                order=order,
                initialize_solve_initial_seconds=initialize_time,
                run_seconds=run_time,
                total_seconds=total_time,
            )
        )

    return results


def print_fit_error_table(error_summaries):
    print("\nFit quality against the Shomate reference over 300-1400 K")
    print(
        f"{'order':>8} {'cp_abs_max':>14} {'cp_rel_max':>14} {'h_abs_max':>14} {'h_rel_max':>14}"
    )
    for summary in error_summaries:
        print(
            f"{summary.order:>8d} "
            f"{summary.max_cp_abs:>14.6g} "
            f"{summary.max_cp_rel:>14.6g} "
            f"{summary.max_h_abs:>14.6g} "
            f"{summary.max_h_rel:>14.6g}"
        )


def print_numpy_results(results):
    baseline = next(result for result in results if result.order is None)
    print("\nNumPy workload benchmark")
    print(
        f"{'backend':>18} {'cp_s':>12} {'cp_speedup':>12} "
        f"{'h_s':>12} {'h_speedup':>12} {'cp+h_s':>12} {'combo_speedup':>14}"
    )
    for result in results:
        print(
            f"{result.label:>18} "
            f"{result.cp_seconds:>12.6f} "
            f"{baseline.cp_seconds / result.cp_seconds:>12.3f} "
            f"{result.h_seconds:>12.6f} "
            f"{baseline.h_seconds / result.h_seconds:>12.3f} "
            f"{result.combined_seconds:>12.6f} "
            f"{baseline.combined_seconds / result.combined_seconds:>14.3f}"
        )


def print_dae_results(results):
    baseline = next(result for result in results if result.order is None)
    print("\nDAETools energy-model benchmark")
    print(
        f"{'backend':>18} {'init+SI_s':>12} {'init_speedup':>14} "
        f"{'run_s':>12} {'run_speedup':>12} {'total_s':>12} {'total_speedup':>14}"
    )
    for result in results:
        print(
            f"{result.label:>18} "
            f"{result.initialize_solve_initial_seconds:>12.6f} "
            f"{baseline.initialize_solve_initial_seconds / result.initialize_solve_initial_seconds:>14.3f} "
            f"{result.run_seconds:>12.6f} "
            f"{baseline.run_seconds / result.run_seconds:>12.3f} "
            f"{result.total_seconds:>12.6f} "
            f"{baseline.total_seconds / result.total_seconds:>14.3f}"
        )


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Shomate enthalpy/cp correlations against fitted dT-polynomial surrogates "
            "in raw NumPy and inside a DAETools energy model."
        )
    )
    parser.add_argument("--orders", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--numpy-points", type=int, default=250000)
    parser.add_argument("--numpy-repeats", type=int, default=25)
    parser.add_argument("--dae-cells", type=int, default=15)
    parser.add_argument("--dae-time-horizon", type=float, default=40.0)
    parser.add_argument("--dae-reporting-interval", type=float, default=5.0)
    parser.add_argument("--dae-repeats", type=int, default=3)
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    reference_set = build_reference_shomate_set()
    fit_temperatures = np.linspace(300.0, 1400.0, 600)
    validation_temperatures = np.linspace(300.0, 1400.0, 2000)
    numpy_temperatures = np.linspace(300.0, 1400.0, args.numpy_points)

    polynomial_sets = {
        order: build_polynomial_set(reference_set, order, fit_temperatures)
        for order in args.orders
    }
    fit_summaries = [
        summarize_fit_errors(
            reference_set,
            polynomial_sets[order],
            validation_temperatures,
            order,
        )
        for order in args.orders
    ]

    for species in reference_set.gas_species + reference_set.solid_species:
        cp_values = species.correlation.cp_value(validation_temperatures)
        cp_min = float(np.min(cp_values))
        if cp_min <= 0.0:
            raise RuntimeError(f"{species.name} produced non-positive cp values: min={cp_min}")

    print_fit_error_table(fit_summaries)
    numpy_results = run_numpy_benchmarks(
        reference_set,
        polynomial_sets,
        numpy_temperatures,
        args.numpy_repeats,
    )
    print_numpy_results(numpy_results)

    dae_results = run_dae_benchmarks(
        reference_set,
        polynomial_sets,
        n_cells=args.dae_cells,
        time_horizon=args.dae_time_horizon,
        reporting_interval=args.dae_reporting_interval,
        repeats=args.dae_repeats,
    )
    print_dae_results(dae_results)


if __name__ == "__main__":
    main()
