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
from pyUnits import K, Pa, m, s


PROPERTY_DIR = Path(__file__).resolve().parents[1] / "Property_Estimation"
if str(PROPERTY_DIR) not in sys.path:
    sys.path.insert(0, str(PROPERTY_DIR))

import visc_fit


temperature_type = daeVariableType(
    name="visc_benchmark_temperature_type",
    units=K,
    lowerBound=250.0,
    upperBound=2000.0,
    initialGuess=900.0,
    absTolerance=1e-6,
)
pressure_type = daeVariableType(
    name="visc_benchmark_pressure_type",
    units=Pa,
    lowerBound=1.0e4,
    upperBound=1.0e7,
    initialGuess=2.0e5,
    absTolerance=1e-4,
)
volumetric_flow_type = daeVariableType(
    name="visc_benchmark_volumetric_flow_type",
    units=m**3 / s,
    lowerBound=-10.0,
    upperBound=10.0,
    initialGuess=1.0e-4,
    absTolerance=1e-10,
)
viscosity_type = daeVariableType(
    name="visc_benchmark_viscosity_type",
    units=Pa * s,
    lowerBound=1.0e-7,
    upperBound=1.0,
    initialGuess=3.0e-5,
    absTolerance=1e-12,
)


SPECIES_FRACTIONS = {
    "h2": 0.34,
    "co": 0.18,
    "co2": 0.10,
    "h2o": 0.20,
    "n2": 0.10,
    "ch4": 0.05,
    "o2": 0.01,
    "ar": 0.01,
    "he": 0.01,
}


def _as_float_array(values):
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class SpeciesViscositySpec:
    name: str
    fraction: float
    correlation: object


@dataclass(frozen=True)
class CorrelationSet:
    species: tuple[SpeciesViscositySpec, ...]


@dataclass(frozen=True)
class SpeciesFitSummary:
    species: str
    quadratic_rmse: float
    power_rmse: float
    quadratic_max_abs: float
    power_max_abs: float
    quadratic_r2: float
    power_r2: float


@dataclass(frozen=True)
class BenchmarkResult:
    label: str
    seconds: float


class QuadraticViscosityCorrelation:
    def __init__(self, *, t_ref, a0, a1, a2):
        self.t_ref = float(t_ref)
        self.a0 = float(a0)
        self.a1 = float(a1)
        self.a2 = float(a2)

    def value(self, temperature):
        d_t = _as_float_array(temperature) - self.t_ref
        return self.a0 + d_t * (self.a1 + d_t * self.a2)

    def dae_expression(self, temperature):
        t_ref = Constant(self.t_ref * K)
        a0 = Constant(self.a0 * Pa * s)
        a1 = Constant(self.a1 * Pa * s / K)
        a2 = Constant(self.a2 * Pa * s / (K**2))
        d_t = temperature - t_ref
        return a0 + d_t * (a1 + d_t * a2)


class PowerLawViscosityCorrelation:
    def __init__(self, *, A, B):
        self.A = float(A)
        self.B = float(B)

    def value(self, temperature):
        return self.A * _as_float_array(temperature) ** self.B

    def dae_expression(self, temperature):
        tau = temperature / Constant(1.0 * K)
        a = Constant(self.A * Pa * s)
        return a * tau**self.B


def build_correlation_sets():
    quadratic_species = []
    power_species = []
    summaries = []

    for species_name, fraction in SPECIES_FRACTIONS.items():
        temperatures, viscosity_data, _ = visc_fit.load_viscosity_data(species_name)
        t_ref = float(np.mean(temperatures))

        quadratic_result = visc_fit.summarize_model_fit(
            temperatures,
            viscosity_data,
            visc_fit.make_polynomial_basis(2, t_ref=t_ref),
        )
        power_result = visc_fit.summarize_model_fit(
            temperatures,
            viscosity_data,
            visc_fit.make_power_law_basis(),
        )

        quad_theta = quadratic_result["theta"]
        power_theta = power_result["theta"]

        quadratic_species.append(
            SpeciesViscositySpec(
                name=species_name.upper(),
                fraction=fraction,
                correlation=QuadraticViscosityCorrelation(
                    t_ref=t_ref,
                    a0=quad_theta[0],
                    a1=quad_theta[1],
                    a2=quad_theta[2],
                ),
            )
        )
        power_species.append(
            SpeciesViscositySpec(
                name=species_name.upper(),
                fraction=fraction,
                correlation=PowerLawViscosityCorrelation(
                    A=power_theta[0],
                    B=power_theta[1],
                ),
            )
        )

        summaries.append(
            SpeciesFitSummary(
                species=species_name.upper(),
                quadratic_rmse=float(quadratic_result["visc_rmse"]),
                power_rmse=float(power_result["visc_rmse"]),
                quadratic_max_abs=float(quadratic_result["visc_max_abs"]),
                power_max_abs=float(power_result["visc_max_abs"]),
                quadratic_r2=float(quadratic_result["visc_r2"]),
                power_r2=float(power_result["visc_r2"]),
            )
        )

    return CorrelationSet(tuple(quadratic_species)), CorrelationSet(tuple(power_species)), summaries


def _evaluate_viscosity_workload(correlation_set, temperatures):
    species_values = []
    mix = np.zeros_like(temperatures, dtype=float)
    for species in correlation_set.species:
        mu = species.correlation.value(temperatures)
        species_values.append(mu)
        mix += species.fraction * mu
    species_matrix = np.vstack(species_values)
    return species_matrix, mix


def benchmark_callable(function, repeats):
    function()
    timings = []
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


def run_numpy_benchmarks(quadratic_set, power_set, temperatures, repeats):
    return [
        BenchmarkResult(
            label="Quadratic",
            seconds=benchmark_callable(
                lambda: _evaluate_viscosity_workload(quadratic_set, temperatures),
                repeats,
            ),
        ),
        BenchmarkResult(
            label="Power law",
            seconds=benchmark_callable(
                lambda: _evaluate_viscosity_workload(power_set, temperatures),
                repeats,
            ),
        ),
    ]


class ViscosityFlowBenchmarkModel(daeModel):
    def __init__(self, name, correlation_set, n_cells, parent=None, description="Viscosity flow benchmark"):
        daeModel.__init__(self, name, parent, description)

        self.correlation_set = correlation_set
        self.species = list(correlation_set.species)

        self.Cells = daeDomain("Cells", self, dimless, "Flow cells")
        self.Species = daeDomain("Species", self, dimless, "Gas species")
        self.Faces = daeDomain("Faces", self, dimless, "Cell faces")
        self.Cells.CreateArray(n_cells)
        self.Species.CreateArray(len(self.species))
        self.Faces.CreateArray(n_cells + 1)

        self.y = daeParameter("y", dimless, self, "Species fractions", [self.Species])
        self.T_target = daeParameter("T_target", K, self, "Cell target temperatures", [self.Cells])
        self.tau_T = daeParameter("tau_T", s, self, "Temperature time constant")
        self.pressure_compliance = daeParameter(
            "pressure_compliance",
            m**3 / Pa,
            self,
            "Gas volume compliance per cell",
        )
        self.hydraulic_conductance = daeParameter(
            "hydraulic_conductance",
            m**3,
            self,
            "Linear conductance multiplier in F = K * dP / mu",
        )
        self.P_inlet = daeParameter("P_inlet", Pa, self, "Inlet pressure")
        self.P_outlet = daeParameter("P_outlet", Pa, self, "Outlet pressure")

        self.T = daeVariable("T", temperature_type, self, "Cell temperature", [self.Cells])
        self.P = daeVariable("P", pressure_type, self, "Cell pressure", [self.Cells])
        self.mu_species = daeVariable(
            "mu_species",
            viscosity_type,
            self,
            "Species viscosities",
            [self.Species, self.Cells],
        )
        self.mu_mix = daeVariable(
            "mu_mix",
            viscosity_type,
            self,
            "Mixture viscosity",
            [self.Cells],
        )
        self.F = daeVariable(
            "F",
            volumetric_flow_type,
            self,
            "Face volumetric flow",
            [self.Faces],
        )

    def _mixture_sum(self, cell_index):
        expression = Constant(0 * Pa * s)
        for species_index in range(self.Species.NumberOfPoints):
            expression += self.y(species_index) * self.mu_species(species_index, cell_index)
        return expression

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        n_cells = self.Cells.NumberOfPoints

        for species_index, species in enumerate(self.species):
            for cell_index in range(n_cells):
                eq = self.CreateEquation(f"mu_{species.name}_{cell_index}")
                eq.Residual = self.mu_species(
                    species_index,
                    cell_index,
                ) - species.correlation.dae_expression(self.T(cell_index))

        for cell_index in range(n_cells):
            eq = self.CreateEquation(f"mu_mix_{cell_index}")
            eq.Residual = self.mu_mix(cell_index) - self._mixture_sum(cell_index)

            eq = self.CreateEquation(f"temperature_relax_{cell_index}")
            eq.Residual = self.tau_T() * dt(self.T(cell_index)) - (
                self.T_target(cell_index) - self.T(cell_index)
            )

        eq = self.CreateEquation("flow_face_0")
        eq.Residual = self.F(0) - self.hydraulic_conductance() * (
            self.P_inlet() - self.P(0)
        ) / self.mu_mix(0)

        for face_index in range(1, n_cells):
            eq = self.CreateEquation(f"flow_face_{face_index}")
            mu_face = 0.5 * (self.mu_mix(face_index - 1) + self.mu_mix(face_index))
            eq.Residual = self.F(face_index) - self.hydraulic_conductance() * (
                self.P(face_index - 1) - self.P(face_index)
            ) / mu_face

        eq = self.CreateEquation(f"flow_face_{n_cells}")
        eq.Residual = self.F(n_cells) - self.hydraulic_conductance() * (
            self.P(n_cells - 1) - self.P_outlet()
        ) / self.mu_mix(n_cells - 1)

        for cell_index in range(n_cells):
            eq = self.CreateEquation(f"pressure_balance_{cell_index}")
            eq.Residual = self.pressure_compliance() * dt(self.P(cell_index)) - self.F(
                cell_index
            ) + self.F(cell_index + 1)


class ViscosityFlowBenchmarkSimulation(daeSimulation):
    def __init__(self, correlation_set, n_cells):
        daeSimulation.__init__(self)
        self.correlation_set = correlation_set
        self.n_cells = n_cells
        self.model = ViscosityFlowBenchmarkModel("ViscosityFlowBenchmark", correlation_set, n_cells)

    def SetUpParametersAndDomains(self):
        self.model.y.SetValues(
            np.asarray([species.fraction for species in self.correlation_set.species], dtype=float)
        )
        self.model.T_target.SetValues(
            np.asarray([1080.0 - 4.0 * idx for idx in range(self.n_cells)], dtype=float)
        )
        self.model.tau_T.SetValue(35.0 * s)
        self.model.pressure_compliance.SetValue(3.0e-8 * m**3 / Pa)
        self.model.hydraulic_conductance.SetValue(3.5e-13 * m**3)
        self.model.P_inlet.SetValue(3.0e5 * Pa)
        self.model.P_outlet.SetValue(1.0e5 * Pa)

    def SetUpVariables(self):
        initial_pressures = np.linspace(2.8e5, 1.2e5, self.n_cells)
        mix_guesses = np.zeros(self.n_cells, dtype=float)

        for cell_index in range(self.n_cells):
            temperature = 720.0 + 3.0 * cell_index
            pressure = float(initial_pressures[cell_index])

            self.model.T.SetInitialCondition(cell_index, temperature * K)
            self.model.P.SetInitialCondition(cell_index, pressure * Pa)

            mix_viscosity = 0.0
            for species_index, species in enumerate(self.correlation_set.species):
                mu_value = float(species.correlation.value(temperature))
                mix_viscosity += species.fraction * mu_value
                self.model.mu_species.SetInitialGuess(
                    species_index,
                    cell_index,
                    mu_value * Pa * s,
                )

            mix_guesses[cell_index] = mix_viscosity
            self.model.mu_mix.SetInitialGuess(cell_index, mix_viscosity * Pa * s)

        conductance = self.model.hydraulic_conductance.GetValue()
        p_inlet = self.model.P_inlet.GetValue()
        p_outlet = self.model.P_outlet.GetValue()

        for face_index in range(self.n_cells + 1):
            if face_index == 0:
                left_pressure = p_inlet
                right_pressure = initial_pressures[0]
                mu_face = mix_guesses[0]
            elif face_index == self.n_cells:
                left_pressure = initial_pressures[-1]
                right_pressure = p_outlet
                mu_face = mix_guesses[self.n_cells - 1]
            else:
                left_pressure = initial_pressures[face_index - 1]
                right_pressure = initial_pressures[face_index]
                mu_face = 0.5 * (mix_guesses[face_index - 1] + mix_guesses[face_index])
            flow_guess = conductance * (left_pressure - right_pressure) / mu_face
            self.model.F.SetInitialGuess(face_index, flow_guess * m**3 / s)


def configure_compute_stack():
    daeGetConfig().SetString("daetools.core.equations.evaluationMode", "computeStack_OpenMP")


def run_dae_case(correlation_set, *, n_cells, time_horizon, reporting_interval):
    configure_compute_stack()

    simulation = ViscosityFlowBenchmarkSimulation(correlation_set, n_cells)
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

    return statistics.median(initialize_times), statistics.median(run_times)


def run_dae_benchmarks(quadratic_set, power_set, *, n_cells, time_horizon, reporting_interval, repeats):
    results = []
    for label, correlation_set in [("Quadratic", quadratic_set), ("Power law", power_set)]:
        init_time, run_time = benchmark_dae_case(
            correlation_set,
            n_cells=n_cells,
            time_horizon=time_horizon,
            reporting_interval=reporting_interval,
            repeats=repeats,
        )
        results.append(
            {
                "label": label,
                "initialize_seconds": init_time,
                "run_seconds": run_time,
                "total_seconds": init_time + run_time,
            }
        )
    return results


def print_fit_summary(summaries):
    print("\nPer-species fit quality on source viscosity data")
    print(
        f"{'species':>8} {'quad_rmse':>12} {'power_rmse':>12} "
        f"{'quad_max':>12} {'power_max':>12} {'quad_r2':>10} {'power_r2':>10}"
    )
    for summary in summaries:
        print(
            f"{summary.species:>8} "
            f"{summary.quadratic_rmse:>12.6g} "
            f"{summary.power_rmse:>12.6g} "
            f"{summary.quadratic_max_abs:>12.6g} "
            f"{summary.power_max_abs:>12.6g} "
            f"{summary.quadratic_r2:>10.6f} "
            f"{summary.power_r2:>10.6f}"
        )


def print_numpy_results(results):
    baseline = results[0]
    print("\nNumPy viscosity workload benchmark")
    print(f"{'backend':>12} {'mu_s':>12} {'speedup_vs_quad':>18}")
    for result in results:
        print(
            f"{result.label:>12} "
            f"{result.seconds:>12.6f} "
            f"{baseline.seconds / result.seconds:>18.3f}"
        )


def print_dae_results(results):
    baseline = results[0]
    print("\nDAETools viscosity-flow benchmark")
    print(
        f"{'backend':>12} {'init+SI_s':>12} {'run_s':>12} "
        f"{'total_s':>12} {'speedup_vs_quad':>18}"
    )
    for result in results:
        print(
            f"{result['label']:>12} "
            f"{result['initialize_seconds']:>12.6f} "
            f"{result['run_seconds']:>12.6f} "
            f"{result['total_seconds']:>12.6f} "
            f"{baseline['total_seconds'] / result['total_seconds']:>18.3f}"
        )


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark quadratic-polynomial and A*T^B viscosity correlations "
            "in raw NumPy and inside a DAETools flow model."
        )
    )
    parser.add_argument("--numpy-points", type=int, default=250000)
    parser.add_argument("--numpy-repeats", type=int, default=25)
    parser.add_argument("--dae-cells", type=int, default=30)
    parser.add_argument("--dae-time-horizon", type=float, default=200.0)
    parser.add_argument("--dae-reporting-interval", type=float, default=20.0)
    parser.add_argument("--dae-repeats", type=int, default=3)
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    quadratic_set, power_set, summaries = build_correlation_sets()
    numpy_temperatures = np.linspace(500.0, 1500.0, args.numpy_points)

    print_fit_summary(summaries)
    numpy_results = run_numpy_benchmarks(
        quadratic_set,
        power_set,
        numpy_temperatures,
        args.numpy_repeats,
    )
    print_numpy_results(numpy_results)

    dae_results = run_dae_benchmarks(
        quadratic_set,
        power_set,
        n_cells=args.dae_cells,
        time_horizon=args.dae_time_horizon,
        reporting_interval=args.dae_reporting_interval,
        repeats=args.dae_repeats,
    )
    print_dae_results(dae_results)


if __name__ == "__main__":
    main()
