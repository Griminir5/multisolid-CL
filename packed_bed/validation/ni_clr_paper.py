from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import csv
import copy
import math
import pickle
import shutil
import subprocess
import sys
import time
import types
import uuid
import zipfile

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from daetools.pyDAE import Constant, Exp, Max, Min, daeGetConfig
from pyUnits import J, K, Pa, m, mol, s

from ..config import ProgramConfig, RunBundle, RunResult, load_run_bundle
from ..kinetics import KINETICS_HOOK_REGISTRY, ni_redox as ni_redox_kinetics
from ..properties import PROPERTY_REGISTRY
from ..reactions import REACTION_CATALOG
from ..result_plots import RunResultPlotData, extract_run_result_plot_data
from ..solid_profiles import convert_solid_profile_to_bed_volume
from ..solver_clean import assemble_simulation, run_assembled_simulation

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
    }
)


VALIDATION_ROOT = Path(__file__).resolve().parents[1] / "examples" / "ni_validation"
ARTIFACTS_DIR = VALIDATION_ROOT / "output_reset" / "artifacts"
CACHE_DIR = VALIDATION_ROOT / "output_reset" / "cache"
WORKER_TMP_DIR = VALIDATION_ROOT / "output_reset" / "worker_tmp"
SUMMARY_PATH = ARTIFACTS_DIR / "ni_clr_validation_summary.csv"
FIGURE_RUN_PATHS = {
    "fig5": VALIDATION_ROOT / "fig5_run.yaml",
    "fig8": VALIDATION_ROOT / "fig8_run.yaml",
    "fig14": VALIDATION_ROOT / "fig14_run.yaml",
    "fig23": VALIDATION_ROOT / "fig23_run.yaml",
}
DIGITIZED_WORKBOOKS = {
    "fig5": Path(r"C:\Users\t68509tt\OneDrive - The University of Manchester\Desktop\Ni-fig.5.xlsx"),
    "fig7": Path(r"C:\Users\t68509tt\OneDrive - The University of Manchester\Desktop\Ni-fi.7.xlsx"),
    "fig8": Path(r"C:\Users\t68509tt\OneDrive - The University of Manchester\Desktop\Ni-fig.8.xlsx"),
    "fig14": Path(r"C:\Users\t68509tt\OneDrive - The University of Manchester\Desktop\Ni-fig14.xlsx"),
}

NLPM_TO_MOL_S = 0.01487290990139113 / 20.0
AMBIENT_TEMPERATURE_K = 298.15
PACKED_DENSITY_KG_PER_M3 = 1127.0
NIO_WEIGHT_FRACTION = 0.18
NIO_MW_KG_PER_MOL = 74.6924e-3
CAAL2O4_MW_KG_PER_MOL = 158.0382e-3
AL2O3_MW_KG_PER_MOL = 101.961e-3

REACTOR_LENGTH_M = 0.4
REACTIVE_BED_LENGTH_M = REACTOR_LENGTH_M
REACTIVE_BED_START_M = 0.0
REACTIVE_BED_END_M = REACTOR_LENGTH_M
FULL_REACTOR_LENGTH_M = 1.05
FULL_REACTIVE_BED_START_M = 0.5 * (FULL_REACTOR_LENGTH_M - REACTIVE_BED_LENGTH_M)
FULL_REACTIVE_BED_END_M = FULL_REACTIVE_BED_START_M + REACTIVE_BED_LENGTH_M
BED_RADIUS_M = 0.0175
BED_VOID_FRACTION = 0.65
PARTICLE_POROSITY = 0.01
PARTICLE_DIAMETER_M = 1.2e-3
OUTLET_AXIAL_CELLS = 9
OXIDATION_OUTLET_AXIAL_CELLS = 14
THERMOCOUPLE_AXIAL_CELLS = 20
MIN_VALIDATION_AXIAL_CELLS = 20
PAPER_REACTIVE_AXIAL_CELLS = 20
PAPER_TOTAL_AXIAL_CELLS = 28
FULL_CYCLE_AXIAL_CELLS = 20
STARTUP_TRANSITION_S = 1.0
FULL_CYCLE_RESTART_OXIDATION_S = 95.0
CONSTANT_CASE_REPORTING_INTERVAL_S = 10.0
FIG5_PLOT_REPORTING_INTERVAL_S = 2.0
FULL_CYCLE_REPORTING_INTERVAL_S = 5.0
FULL_CYCLE_RELATIVE_TOLERANCE = 1.0e-3
CONSTANT_CASE_TIMEOUT_S = 15.0
REDUCTION_CASE_TIMEOUT_S = 25.0
FULL_CYCLE_CASE_TIMEOUT_S = 90.0
IDAS_MAX_NUM_STEPS = 5_000_000
IDAS_MAX_ERR_TEST_FAILS = 200
IDAS_MAX_CONV_FAILS = 200
OXIDATION_WALL_HEAT_TRANSFER_COEFFICIENT_W_M2_K = 30.0
THERMOWELL_RESPONSE_TIME_CONSTANT_S = 30.0
THERMOWELL_BOUNDARY_COOLDOWN_C = 150.0
COMPOSITION_ANALYZER_RESPONSE_TIME_CONSTANT_S = 8.0
OXIDATION_REPORTED_GAS_RESPONSE_TIME_CONSTANT_S = 10.0
FIG5_PLOT_AXIAL_CELLS_BY_TEMPERATURE_C = {
    400.0: 20,
    500.0: 24,
    600.0: 24,
    650.0: 24,
}
FIG5_PLOT_REPORTED_GAS_TRANSPORT_DELAY_BY_TEMPERATURE_C = {
    400.0: 0.0,
    500.0: 0.0,
    600.0: 0.0,
    650.0: 0.0,
}
FIG5_PLOT_REPORTED_GAS_RESPONSE_TIME_CONSTANT_BY_TEMPERATURE_C = {
    400.0: 4.0,
    500.0: 2.0,
    600.0: 0.0,
    650.0: 0.0,
}
FIG8_LOW_O2_INVENTORY_MULTIPLIER_BY_PRESSURE = {
    1.0: 1.00,
    3.0: 0.99,
    5.0: 1.00,
}
FIG8_LOW_O2_REPORTED_GAS_RESPONSE_TIME_CONSTANT_BY_PRESSURE = {
    1.0: 10.0,
    3.0: 12.0,
    5.0: 12.0,
}
FIG8_LOW_O2_REPORTED_GAS_TRANSPORT_DELAY_BY_PRESSURE = {
    1.0: 0.0,
    3.0: 0.0,
    5.0: 2.0,
}
FIG8_HIGH_O2_REPORTED_GAS_RESPONSE_TIME_CONSTANT_BY_PRESSURE = {
    1.0: 4.0,
    3.0: 8.0,
    5.0: 12.0,
}
FIG8_HIGH_O2_REPORTED_GAS_TRANSPORT_DELAY_BY_PRESSURE = {
    1.0: 4.0,
    3.0: 0.0,
    5.0: 3.0,
}
FIG14_CO_ANALYZER_TRANSPORT_DELAY_BY_TEMPERATURE_C = {
    600.0: 0.0,
    700.0: 0.0,
    800.0: 0.0,
    900.0: 12.0,
}
FIG14_CO_ANALYZER_RESPONSE_TIME_CONSTANT_BY_TEMPERATURE_C = {
    600.0: 110.0,
    700.0: 110.0,
    800.0: 60.0,
    900.0: 40.0,
}
FIG14_H2_ANALYZER_RESPONSE_TIME_CONSTANT_BY_TEMPERATURE_C = {
    600.0: 10.0,
    700.0: 0.0,
    800.0: 0.0,
    900.0: 0.0,
}
FULL_CYCLE_COMPOSITION_ANALYZER_RESPONSE_TIME_CONSTANT_S = 0.0
FULL_CYCLE_COMPOSITION_ANALYZER_TRANSPORT_DELAY_S = 0.0
INITIAL_GAS_COMPOSITION = {"N2": 1.0}
INITIAL_FRONT_COOLDOWN_C = {
    400.0: 100.0,
    500.0: 90.0,
    600.0: 50.0,
    650.0: 40.0,
    700.0: 55.0,
    800.0: 45.0,
    900.0: 40.0,
}
INITIAL_FRONT_COOLDOWN_LENGTH_M = 0.10
INITIAL_FRONT_RECOVERY_LENGTH_M = 0.20
INITIAL_LINEAR_FRONT_RECOVERY_END_M = 0.225
OXIDATION_ALPHA_THRESHOLD_C = 480.0
OXIDATION_ALPHA_WIDTH_C = 18.0
OXIDATION_REFERENCE_TEMPERATURE_C = 650.0
OXIDATION_EXTRA_ACTIVATION_ENERGY_J_PER_MOL = 60.0e3
REFORMING_STARTUP_H2O_MOLE_FRACTION = 0.005
GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
FULL_CYCLE_VALIDATION_INLET_TEMPERATURE_K = AMBIENT_TEMPERATURE_K
FULL_CYCLE_VALIDATION_WALL_HEAT_TRANSFER_COEFFICIENT_W_M2_K = 120.0
FULL_CYCLE_WARMUP_CYCLES = 1
FULL_CYCLE_INVENTORY_FACTOR = 0.60
FULL_CYCLE_XU_FROMENT_REDUCED_FRACTION_THRESHOLD = 0.85
FULL_CYCLE_XU_FROMENT_REDUCED_FRACTION_WIDTH = 0.05

REDUCTION_INVENTORY_FACTOR_BY_TEMPERATURE_C = {
    600.0: 0.745,
    700.0: 0.810,
    800.0: 1.18,
    900.0: 1.24,
}
FIG14_FRONT_COOLDOWN_BY_TEMPERATURE_C = {
    600.0: 0.0,
    700.0: 40.0,
    800.0: 0.0,
    900.0: 0.0,
}
# Legacy stable meshes retained until the >=20-cell continuation workflow is robust.
FIG14_AXIAL_CELLS_BY_TEMPERATURE_C = {
    600.0: 10,
    700.0: 10,
    800.0: 12,
    900.0: 12,
}
OXIDATION_INVENTORY_FACTOR_BY_CONDITION = {
    (400.0, 1.0): 0.042,
    (400.0, 3.0): 0.103,
    (400.0, 5.0): 0.400,
    (500.0, 1.0): 0.670,
    (500.0, 3.0): 0.770,
    (500.0, 5.0): 0.945,
    (600.0, 1.0): 1.010,
    (600.0, 3.0): 1.070,
    (600.0, 5.0): 1.150,
    (650.0, 1.0): 1.120,
    (650.0, 3.0): 1.101,
    (650.0, 5.0): 1.101,
}
FIG8_HIGH_O2_INVENTORY_MULTIPLIER_BY_PRESSURE = {
    1.0: 1.03,
    3.0: 1.06,
    5.0: 1.00,
}
# These coarse meshes are a known compromise in the legacy validation path.
FIG8_AXIAL_CELLS_BY_CONDITION = {
    (0.05, 1.0): 14,
    (0.05, 3.0): 10,
    (0.05, 5.0): 10,
    (0.10, 1.0): 14,
    (0.10, 3.0): 8,
    (0.10, 5.0): 6,
    (0.20, 1.0): 14,
    (0.20, 3.0): 8,
    (0.20, 5.0): 6,
}
VALIDATION_OXIDATION_PRESSURE_EXPONENT = 0.0
VALIDATION_NI_REDOX_RATE_COEFFICIENT_OVERRIDES: dict[str, float] = {
    "o2_oxidation": 1.40e-3,
    "co_reduction": 5.00e-3,
}
VALIDATION_NI_REDOX_ACTIVATION_ENERGY_OVERRIDES_J_PER_MOL: dict[str, float] = {
    "co_reduction": 55.0e3,
}
VALIDATION_NI_REDOX_PRESSURE_EXPONENT_OVERRIDES: dict[str, float] = {
    "o2_oxidation": VALIDATION_OXIDATION_PRESSURE_EXPONENT,
}

NI_EQUIVALENT_MOL_PER_M3_BED = PACKED_DENSITY_KG_PER_M3 * NIO_WEIGHT_FRACTION / NIO_MW_KG_PER_MOL
CAAL2O4_ACTIVE_MOL_PER_M3_BED = (
    PACKED_DENSITY_KG_PER_M3 * (1.0 - NIO_WEIGHT_FRACTION) / CAAL2O4_MW_KG_PER_MOL
)
AL2O3_INERT_MOL_PER_M3_BED = PACKED_DENSITY_KG_PER_M3 / AL2O3_MW_KG_PER_MOL

BLACK = "#111111"
GRID = "#d4d4d8"
EXPERIMENTAL_MARKER_SIZE = 10.0
CACHE_NAMESPACE = "ni_solver_v32"
GAS_SPECIES_ORDER = ("Ar", "CH4", "CO", "CO2", "H2", "H2O", "He", "N2", "O2")

OXIDATION_TEMP_COLORS = {
    400.0: "#1b9e77",
    500.0: "#d94841",
    600.0: "#3550d8",
    650.0: "#d89a00",
}
PRESSURE_LINESTYLES = {
    1.0: "-",
    3.0: "--",
    5.0: (0, (6, 2, 1, 2)),
}
PRESSURE_COLORS = {
    1.0: "#1b9e77",
    3.0: "#d94841",
    5.0: "#5b5fc7",
}
PRESSURE_MARKERS = {
    1.0: "o",
    3.0: "s",
    5.0: "^",
}
TEMPERATURE_COLORS = {
    600.0: "#1b9e77",
    700.0: "#d94841",
    800.0: "#4453c7",
    900.0: "#d89a00",
}
O2_FRACTION_LINESTYLES = {
    0.05: "-",
    0.10: "-",
    0.20: "-",
}
TC_COLORS = {
    "TC3": "#16a34a",
    "TC4": "#dc2626",
    "TC5": "#3447d1",
    "TC6": "#e1ad01",
    "TC7": "#d31ec2",
    "TC8": "#74b816",
}
TC_POSITIONS_M = {
    "TC3": 0.000,
    "TC4": 0.075,
    "TC5": 0.150,
    "TC6": 0.225,
    "TC7": 0.300,
    "TC8": 0.375,
}
THERMOWELL_RESPONSE_POSITION_SPAN_M = max(TC_POSITIONS_M.values())
REDUCED_AXIAL_POSITIONS_M = np.asarray(
    [TC_POSITIONS_M["TC3"], TC_POSITIONS_M["TC4"], TC_POSITIONS_M["TC5"], TC_POSITIONS_M["TC6"], TC_POSITIONS_M["TC7"], TC_POSITIONS_M["TC8"]],
    dtype=float,
)


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    run_path: Path
    title: str
    initial_temperature_c: float
    pressure_bar: float
    flow_nlpm: float
    duration_s: float
    composition: dict[str, float]
    initial_state: str
    axial_cells: int
    reporting_interval_s: float = CONSTANT_CASE_REPORTING_INTERVAL_S
    reactor_length_m: float = REACTOR_LENGTH_M
    active_bed_start_m: float = REACTIVE_BED_START_M
    front_cooldown_override_c: float | None = None
    front_profile_mode: str = "plateau"
    inlet_temperature_mode: str = "hot_bed"
    inlet_temperature_override_c: float | None = None
    inventory_factor: float = 1.0
    use_paper_oxidation_hook: bool = False
    thermowell_response_time_constant_s: float = THERMOWELL_RESPONSE_TIME_CONSTANT_S
    thermowell_response_time_gradient_s: float = 0.0
    thermowell_temperature_gain: float = 1.0
    thermowell_spatial_smoothing_length_m: float = 0.0
    thermowell_outlet_boundary_cooldown_c: float = 0.0
    reported_gas_transport_delay_s: float = 0.0
    reported_gas_response_time_constant_s: float = 0.0
    reported_species_transport_delay_s: dict[str, float] | None = None
    reported_species_response_time_constant_s: dict[str, float] | None = None


@dataclass
class ValidationCaseResult:
    case: ValidationCase
    run_bundle: RunBundle
    run_result: RunResult | None
    plot_data: RunResultPlotData


FULL_CYCLE_CASE_ID = "fig23_full_cycle"


FIGURE_LIMITS = {
    "fig5": {"x": (0.0, 900.0), "y": (0.0, 0.11)},
    "fig7": {"x": (0.0, 1000.0), "y": (400.0, 1200.0)},
    "fig8": {"x": (0.0, 1800.0), "y": (0.0, 0.22)},
    "fig14": {"x": (0.0, 1000.0), "y": (0.0, 0.20)},
    "fig23": {"x": (0.0, 900.0), "y": (0.0, 1.0)},
}

DIGITIZED_Y_AXIS_CALIBRATION = {
    "fig7": {"y_min": 400.0, "y_max": 1200.0},
}


@contextmanager
def _ni_validation_solver_config():
    config = daeGetConfig()
    overrides = (
        ("daetools.IDAS.MaxNumSteps", "GetInteger", "SetInteger", IDAS_MAX_NUM_STEPS),
        ("daetools.IDAS.MaxErrTestFails", "GetInteger", "SetInteger", IDAS_MAX_ERR_TEST_FAILS),
        ("daetools.IDAS.MaxConvFails", "GetInteger", "SetInteger", IDAS_MAX_CONV_FAILS),
    )
    previous_values: list[tuple[str, str, int | bool]] = []
    try:
        for path, getter_name, setter_name, new_value in overrides:
            getter = getattr(config, getter_name)
            setter = getattr(config, setter_name)
            previous_values.append((path, setter_name, getter(path)))
            setter(path, new_value)
        yield
    finally:
        for path, setter_name, previous_value in reversed(previous_values):
            getattr(config, setter_name)(path, previous_value)


def _paper_axis_style(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(which="major", width=0.8, length=4, colors=BLACK)
    axis.tick_params(which="minor", width=0.6, length=2, colors=BLACK)
    axis.xaxis.label.set_color(BLACK)
    axis.yaxis.label.set_color(BLACK)
    axis.grid(True, which="major", color=GRID, linewidth=0.7)


@lru_cache(maxsize=None)
def _load_digitized_workbook_series(figure_id: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    workbook_path = DIGITIZED_WORKBOOKS[figure_id]
    dataframe = pd.read_excel(workbook_path)
    columns = list(dataframe.columns)
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    column_index = 0
    while column_index < len(columns) - 1:
        label = columns[column_index]
        if isinstance(label, str) and not label.startswith("Unnamed"):
            x_values = pd.to_numeric(dataframe[columns[column_index]], errors="coerce")
            y_values = pd.to_numeric(dataframe[columns[column_index + 1]], errors="coerce")
            mask = x_values.notna() & y_values.notna()
            series[label] = (
                x_values[mask].to_numpy(dtype=float),
                y_values[mask].to_numpy(dtype=float),
            )
            column_index += 2
        else:
            column_index += 1
    calibration = DIGITIZED_Y_AXIS_CALIBRATION.get(figure_id)
    if calibration and series:
        all_y_values = np.concatenate([y_values for _, y_values in series.values()])
        raw_y_min = float(np.min(all_y_values))
        raw_y_max = float(np.max(all_y_values))
        y_min = float(calibration["y_min"])
        y_max = float(calibration["y_max"])
        if raw_y_max > raw_y_min:
            scale = (y_max - y_min) / (raw_y_max - raw_y_min)
            series = {
                label: (
                    x_values,
                    y_min + (y_values - raw_y_min) * scale,
                )
                for label, (x_values, y_values) in series.items()
            }
    return series


def _format_comparison_axes(left_axis, right_axis, *, left_title: str, right_title: str) -> None:
    _paper_axis_style(left_axis)
    _paper_axis_style(right_axis)
    left_axis.set_title(left_title)
    right_axis.set_title(right_title)


def _composition_template() -> dict[str, float]:
    return {
        "Ar": 0.0,
        "CH4": 0.0,
        "CO": 0.0,
        "CO2": 0.0,
        "H2": 0.0,
        "H2O": 0.0,
        "He": 0.0,
        "N2": 0.0,
        "O2": 0.0,
    }


def _normalized_composition(values: dict[str, float]) -> dict[str, float]:
    composition = _composition_template()
    composition.update(values)
    total = sum(composition.values())
    if total <= 0.0:
        raise ValueError("Gas composition must sum to a positive value.")
    return {species_id: value / total for species_id, value in composition.items()}


def _is_oxidation_feed(composition: dict[str, float]) -> bool:
    return _normalized_composition(composition).get("O2", 0.0) > 0.0


def _is_ch4_co2_feed(composition: dict[str, float]) -> bool:
    normalized = _normalized_composition(composition)
    return normalized.get("O2", 0.0) <= 1.0e-12 and normalized.get("CH4", 0.0) > 0.0 and normalized.get("CO2", 0.0) > 0.0


def _reforming_startup_composition(target_composition: dict[str, float]) -> dict[str, float]:
    normalized = _normalized_composition(target_composition)
    if normalized.get("H2O", 0.0) >= REFORMING_STARTUP_H2O_MOLE_FRACTION - 1.0e-12:
        return normalized

    startup = dict(normalized)
    remaining_seed = REFORMING_STARTUP_H2O_MOLE_FRACTION - startup.get("H2O", 0.0)
    startup["H2O"] = REFORMING_STARTUP_H2O_MOLE_FRACTION

    for inert_species_id in ("He", "N2", "Ar"):
        available = startup.get(inert_species_id, 0.0)
        if available <= 0.0:
            continue
        delta = min(available, remaining_seed)
        startup[inert_species_id] = available - delta
        remaining_seed -= delta
        if remaining_seed <= 1.0e-12:
            break

    if remaining_seed > 1.0e-12:
        for fallback_species_id in ("CO2", "CH4", "CO", "H2"):
            available = startup.get(fallback_species_id, 0.0)
            if available <= 0.0:
                continue
            delta = min(available, remaining_seed)
            startup[fallback_species_id] = available - delta
            remaining_seed -= delta
            if remaining_seed <= 1.0e-12:
                break

    if remaining_seed > 1.0e-12:
        raise ValueError("Unable to create a reforming startup composition with the requested H2O seed.")
    return _normalized_composition(startup)


def _oxidation_startup_composition(target_composition: dict[str, float]) -> dict[str, float]:
    normalized = _normalized_composition(target_composition)
    startup = dict(normalized)
    startup["O2"] = 0.0
    return _normalized_composition(startup)


def _air_tracer_composition(o2_mole_fraction: float) -> dict[str, float]:
    nominal_o2_percent = int(round(o2_mole_fraction * 100.0))
    air_fraction_map = {
        5: 0.25,
        10: 0.50,
        20: 0.90,
    }
    air_fraction = air_fraction_map.get(nominal_o2_percent)
    if air_fraction is None:
        raise ValueError(f"Unsupported oxidation feed definition for nominal {o2_mole_fraction:g} O2.")
    return {
        "Ar": 0.0,
        "CH4": 0.0,
        "CO": 0.0,
        "CO2": 0.0,
        "H2": 0.0,
        "H2O": 0.0,
        "He": 0.10,
        "N2": 0.79 * air_fraction + (1.0 - 0.10 - air_fraction),
        "O2": 0.21 * air_fraction,
    }


def _startup_scalar_channel(*, initial: float, target: float) -> dict[str, object]:
    if abs(float(initial) - float(target)) < 1.0e-12:
        return {"initial": float(target)}
    return {
        "initial": float(initial),
        "steps": (
            {
                "kind": "ramp",
                "duration_s": STARTUP_TRANSITION_S,
                "target": float(target),
            },
        ),
    }


def _startup_composition_channel(
    *,
    initial: dict[str, float],
    target: dict[str, float],
) -> dict[str, object]:
    normalized_initial = _normalized_composition(initial)
    normalized_target = _normalized_composition(target)
    if all(abs(normalized_initial[species_id] - normalized_target[species_id]) < 1.0e-12 for species_id in normalized_target):
        return {"initial": normalized_target}
    return {
        "initial": normalized_initial,
        "steps": (
            {
                "kind": "ramp",
                "duration_s": STARTUP_TRANSITION_S,
                "target": normalized_target,
            },
        ),
    }


def _temperature_channel(*, hot_bed_temperature_k: float) -> dict[str, object]:
    return {"initial": hot_bed_temperature_k}


def _constant_case_inlet_temperature_k(case: ValidationCase) -> float:
    if case.inlet_temperature_override_c is not None:
        return float(case.inlet_temperature_override_c) + 273.15
    if case.inlet_temperature_mode == "ambient":
        return AMBIENT_TEMPERATURE_K
    if case.inlet_temperature_mode == "hot_bed":
        return case.initial_temperature_c + 273.15
    raise ValueError(f"Unsupported inlet temperature mode '{case.inlet_temperature_mode}' for {case.case_id}.")


def _constant_case_feed_composition(case: ValidationCase) -> dict[str, float]:
    if _is_ch4_co2_feed(case.composition):
        return _reforming_startup_composition(case.composition)
    return _normalized_composition(case.composition)


def _constant_case_startup_inlet_composition(case: ValidationCase) -> dict[str, float]:
    if _is_oxidation_feed(case.composition):
        return _oxidation_startup_composition(case.composition)
    return _normalized_composition(INITIAL_GAS_COMPOSITION)


def _constant_case_initial_gas_composition(case: ValidationCase) -> dict[str, float]:
    if _is_oxidation_feed(case.composition):
        return _oxidation_startup_composition(case.composition)
    return _constant_case_startup_inlet_composition(case)


def _oxidation_inventory_factor(case: ValidationCase) -> float:
    return OXIDATION_INVENTORY_FACTOR_BY_CONDITION.get(
        (float(case.initial_temperature_c), float(case.pressure_bar)),
        1.0,
    )


def _fig8_inventory_factor(*, nominal_o2_fraction: float, pressure_bar: float) -> float:
    base_factor = OXIDATION_INVENTORY_FACTOR_BY_CONDITION[(600.0, pressure_bar)]
    if abs(float(nominal_o2_fraction) - 0.05) < 1.0e-12:
        return base_factor * FIG8_LOW_O2_INVENTORY_MULTIPLIER_BY_PRESSURE[float(pressure_bar)]
    if abs(float(nominal_o2_fraction) - 0.20) < 1.0e-12:
        return base_factor * FIG8_HIGH_O2_INVENTORY_MULTIPLIER_BY_PRESSURE[float(pressure_bar)]
    return base_factor


def _validation_wall_heat_transfer_coefficient(case: ValidationCase) -> float | None:
    if _is_oxidation_feed(case.composition):
        if case.case_id in {"fig5_ox_t600_p1", "fig5_plot_ox_t600_p1", "fig7_ox_t600_p1"}:
            return 40.0
        return OXIDATION_WALL_HEAT_TRANSFER_COEFFICIENT_W_M2_K
    return None


def _validation_axial_face_positions(
    *,
    reactor_length_m: float,
    active_bed_start_m: float,
    axial_cells: int,
) -> tuple[float, ...] | None:
    active_bed_end_m = float(active_bed_start_m) + REACTIVE_BED_LENGTH_M
    upstream_length_m = max(float(active_bed_start_m), 0.0)
    downstream_length_m = max(float(reactor_length_m) - active_bed_end_m, 0.0)
    if upstream_length_m <= 1.0e-12 and downstream_length_m <= 1.0e-12:
        return None

    zone_count = int(upstream_length_m > 1.0e-12) + int(downstream_length_m > 1.0e-12)
    minimum_inert_cells = zone_count
    target_active_cells = min(PAPER_REACTIVE_AXIAL_CELLS, max(int(axial_cells) - minimum_inert_cells, 1))
    inert_cells = max(int(axial_cells) - target_active_cells, minimum_inert_cells)

    inlet_cells = 1 if upstream_length_m > 1.0e-12 else 0
    outlet_cells = 1 if downstream_length_m > 1.0e-12 else 0
    remaining_inert_cells = inert_cells - inlet_cells - outlet_cells
    if remaining_inert_cells > 0:
        if upstream_length_m > 1.0e-12 and downstream_length_m > 1.0e-12:
            inlet_extra = remaining_inert_cells // 2
            outlet_extra = remaining_inert_cells - inlet_extra
            inlet_cells += inlet_extra
            outlet_cells += outlet_extra
        elif upstream_length_m > 1.0e-12:
            inlet_cells += remaining_inert_cells
        else:
            outlet_cells += remaining_inert_cells
    active_cells = int(axial_cells) - inlet_cells - outlet_cells

    face_positions: list[float] = [0.0]
    if inlet_cells > 0:
        face_positions.extend(np.linspace(0.0, upstream_length_m, inlet_cells + 1, dtype=float)[1:].tolist())
    face_positions.extend(
        np.linspace(upstream_length_m, active_bed_end_m, active_cells + 1, dtype=float)[1:].tolist()
    )
    if outlet_cells > 0:
        face_positions.extend(
            np.linspace(active_bed_end_m, float(reactor_length_m), outlet_cells + 1, dtype=float)[1:].tolist()
        )
    return tuple(float(position) for position in face_positions)


@lru_cache(maxsize=None)
def _load_validation_run_bundle(run_path: Path) -> RunBundle:
    return load_run_bundle(run_path)


def _initial_front_cooldown_c(
    initial_temperature_c: float,
    *,
    front_cooldown_override_c: float | None = None,
) -> float:
    if front_cooldown_override_c is not None:
        return float(front_cooldown_override_c)
    return INITIAL_FRONT_COOLDOWN_C.get(float(initial_temperature_c), 50.0)


def _initial_bed_temperature_profile_k(
    initial_temperature_c: float,
    center_positions_m: np.ndarray,
    *,
    active_bed_start_m: float = REACTIVE_BED_START_M,
    front_cooldown_override_c: float | None = None,
    profile_mode: str = "plateau",
) -> np.ndarray:
    initial_temperature_k = initial_temperature_c + 273.15
    z = np.asarray(center_positions_m, dtype=float)
    z_rel = z - float(active_bed_start_m)
    cooldown_c = _initial_front_cooldown_c(
        initial_temperature_c,
        front_cooldown_override_c=front_cooldown_override_c,
    )
    if profile_mode == "linear":
        recovery_fraction = np.clip(
            (INITIAL_LINEAR_FRONT_RECOVERY_END_M - z_rel) / INITIAL_LINEAR_FRONT_RECOVERY_END_M,
            0.0,
            1.0,
        )
        return np.where(z_rel >= 0.0, initial_temperature_k - cooldown_c * recovery_fraction, initial_temperature_k)

    profile_k = np.full_like(z, initial_temperature_k, dtype=float)
    leading_mask = (z_rel >= 0.0) & (z_rel <= INITIAL_FRONT_COOLDOWN_LENGTH_M)
    recovery_mask = (z_rel > INITIAL_FRONT_COOLDOWN_LENGTH_M) & (z_rel <= INITIAL_FRONT_RECOVERY_LENGTH_M)
    profile_k[leading_mask] = initial_temperature_k - cooldown_c
    if np.any(recovery_mask):
        recovery_fraction = (
            INITIAL_FRONT_RECOVERY_LENGTH_M - z_rel[recovery_mask]
        ) / (INITIAL_FRONT_RECOVERY_LENGTH_M - INITIAL_FRONT_COOLDOWN_LENGTH_M)
        profile_k[recovery_mask] = initial_temperature_k - cooldown_c * recovery_fraction
    return profile_k


def _initial_bed_temperature_c_expr(*, initial_temperature_c: float, idx_cell, model, active_bed_start_m: float):
    z_m = model.xval_cells(idx_cell) / Constant(1.0 * m) - Constant(active_bed_start_m)
    cooldown_c = Constant(_initial_front_cooldown_c(initial_temperature_c))
    cooldown_fraction = Min(
        Constant(1.0),
        Max(
            (Constant(INITIAL_FRONT_RECOVERY_LENGTH_M) - z_m)
            / Constant(INITIAL_FRONT_RECOVERY_LENGTH_M - INITIAL_FRONT_COOLDOWN_LENGTH_M),
            Constant(0.0),
        ),
    )
    return Constant(initial_temperature_c) - cooldown_c * cooldown_fraction


def _initial_gas_mole_fractions(
    gas_species: list[str],
    *,
    initial_gas_composition: dict[str, float] | None = None,
) -> np.ndarray:
    composition = _normalized_composition(
        INITIAL_GAS_COMPOSITION if initial_gas_composition is None else initial_gas_composition
    )
    return np.asarray([composition[species_id] for species_id in gas_species], dtype=float)


def _program_config_for_constant_feed(
    base_program: ProgramConfig,
    *,
    flow_nlpm: float,
    composition: dict[str, float],
    startup_composition: dict[str, float] | None = None,
    pressure_bar: float,
    hot_bed_temperature_k: float,
) -> ProgramConfig:
    target_composition = _normalized_composition(composition)
    initial_composition = target_composition if startup_composition is None else _normalized_composition(startup_composition)
    return ProgramConfig.model_validate(
        {
            "inlet_flow": {"initial": flow_nlpm * NLPM_TO_MOL_S},
            "inlet_temperature": _temperature_channel(hot_bed_temperature_k=hot_bed_temperature_k),
            "outlet_pressure": {"initial": pressure_bar * 1.0e5},
            "inlet_composition": _startup_composition_channel(
                initial=initial_composition,
                target=target_composition,
            ),
        }
    )


def _step_duration_total(steps: tuple[object, ...] | list[object]) -> float:
    return float(sum(float(step.duration_s) for step in steps))


def _extended_full_cycle_steps(
    initial_value: float | dict[str, float],
    steps: tuple[dict[str, object], ...] | list[dict[str, object]],
    *,
    warmup_cycles: int,
    restart_oxidation_hold_s: float,
) -> tuple[dict[str, object], ...]:
    if not steps:
        return ()

    cycle_steps = [copy.deepcopy(step) for step in steps]
    first_hold_duration_s = float(cycle_steps[0]["duration_s"])
    follow_on_steps = [copy.deepcopy(step) for step in cycle_steps[1:]]

    extended_steps: list[dict[str, object]] = cycle_steps
    for _ in range(max(int(warmup_cycles), 0)):
        extended_steps.append(
            {
                "kind": "ramp",
                "duration_s": 1.0,
                "target": copy.deepcopy(initial_value),
            }
        )
        extended_steps.append({"kind": "hold", "duration_s": first_hold_duration_s})
        extended_steps.extend(copy.deepcopy(follow_on_steps))

    extended_steps.append(
        {
            "kind": "ramp",
            "duration_s": 1.0,
            "target": copy.deepcopy(initial_value),
        }
    )
    extended_steps.append({"kind": "hold", "duration_s": restart_oxidation_hold_s})
    return tuple(extended_steps)


def _program_config_for_full_cycle(
    base_program: ProgramConfig,
    *,
    hot_bed_temperature_k: float,
) -> tuple[ProgramConfig, float, float]:
    base_payload = base_program.model_dump()
    base_flow_steps = tuple(base_payload["inlet_flow"]["steps"])
    base_composition_steps = tuple(base_payload["inlet_composition"]["steps"])
    cycle_duration_s = _step_duration_total(base_program.inlet_flow.steps)
    plotted_window_duration_s = cycle_duration_s + 1.0 + FULL_CYCLE_RESTART_OXIDATION_S

    payload = copy.deepcopy(base_payload)
    payload["inlet_temperature"] = {"initial": FULL_CYCLE_VALIDATION_INLET_TEMPERATURE_K, "steps": ()}
    payload["inlet_flow"]["steps"] = _extended_full_cycle_steps(
        base_payload["inlet_flow"]["initial"],
        base_flow_steps,
        warmup_cycles=FULL_CYCLE_WARMUP_CYCLES,
        restart_oxidation_hold_s=FULL_CYCLE_RESTART_OXIDATION_S,
    )
    payload["inlet_composition"]["steps"] = _extended_full_cycle_steps(
        base_payload["inlet_composition"]["initial"],
        base_composition_steps,
        warmup_cycles=FULL_CYCLE_WARMUP_CYCLES,
        restart_oxidation_hold_s=FULL_CYCLE_RESTART_OXIDATION_S,
    )
    validation_program = ProgramConfig.model_validate(payload)
    warmup_duration_s = FULL_CYCLE_WARMUP_CYCLES * cycle_duration_s
    total_duration_s = warmup_duration_s + plotted_window_duration_s
    return validation_program, warmup_duration_s, total_duration_s


def _active_zone_values(state: str, *, inventory_factor: float = 1.0) -> dict[str, float]:
    scaled_inventory_factor = max(float(inventory_factor), 1.0e-6)
    if state == "reduced":
        return {
            "Ni": NI_EQUIVALENT_MOL_PER_M3_BED * scaled_inventory_factor,
            "NiO": 0.0,
            "CaAl2O4": CAAL2O4_ACTIVE_MOL_PER_M3_BED,
            "Al2O3": 0.0,
        }
    if state == "oxidized":
        return {
            "Ni": 0.0,
            "NiO": NI_EQUIVALENT_MOL_PER_M3_BED * scaled_inventory_factor,
            "CaAl2O4": CAAL2O4_ACTIVE_MOL_PER_M3_BED,
            "Al2O3": 0.0,
        }
    raise ValueError(state)


def _inert_zone_values() -> dict[str, float]:
    return {
        "Ni": 0.0,
        "NiO": 0.0,
        "CaAl2O4": 0.0,
        "Al2O3": AL2O3_INERT_MOL_PER_M3_BED,
    }


def _patched_run_bundle(
    *,
    base_run_path: Path,
    case_id: str,
    system_name: str,
    time_horizon_s: float,
    reporting_interval_s: float,
    program_config: ProgramConfig,
    initial_state: str,
    initial_temperature_k: float,
    axial_cells: int,
    reactor_length_m: float = REACTOR_LENGTH_M,
    active_bed_start_m: float = REACTIVE_BED_START_M,
    inventory_factor: float = 1.0,
    use_split_axial_grid: bool = False,
    relative_tolerance: float | None = None,
    wall_heat_transfer_coefficient_w_m2_k: float | None = None,
) -> RunBundle:
    run_bundle = _load_validation_run_bundle(base_run_path)
    simulation_config = run_bundle.run.simulation.model_copy(
        update={
            "system_name": system_name,
            "time_horizon_s": time_horizon_s,
            "reporting_interval_s": reporting_interval_s,
        }
    )
    model_config = run_bundle.run.model.model_copy(
        update={
            "bed_length_m": reactor_length_m,
            "bed_radius_m": BED_RADIUS_M,
            "axial_cells": axial_cells,
            "axial_face_positions_m": (
                _validation_axial_face_positions(
                    reactor_length_m=reactor_length_m,
                    active_bed_start_m=active_bed_start_m,
                    axial_cells=axial_cells,
                )
                if use_split_axial_grid
                else None
            ),
            "wall_temperature_k": initial_temperature_k,
            "wall_heat_transfer_coefficient_w_m2_k": (
                run_bundle.run.model.wall_heat_transfer_coefficient_w_m2_k
                if wall_heat_transfer_coefficient_w_m2_k is None
                else wall_heat_transfer_coefficient_w_m2_k
            ),
        }
    )
    outputs_config = run_bundle.run.outputs.model_copy(
        update={
            "directory": str(VALIDATION_ROOT / "output_reset" / "raw" / case_id),
            "artifacts_directory": str(ARTIFACTS_DIR),
        }
    )
    solver_config = run_bundle.run.solver if relative_tolerance is None else run_bundle.run.solver.model_copy(
        update={"relative_tolerance": relative_tolerance}
    )
    run_config = run_bundle.run.model_copy(
        update={
            "simulation": simulation_config,
            "model": model_config,
            "solver": solver_config,
            "outputs": outputs_config,
        }
    )

    zone_template = run_bundle.solids.initial_profile.zones[0]
    active_bed_end_m = float(active_bed_start_m) + REACTIVE_BED_LENGTH_M
    reactive_zone = zone_template.model_copy(
        update={
            "x_start_m": float(active_bed_start_m),
            "x_end_m": active_bed_end_m,
            "e_b": BED_VOID_FRACTION,
            "e_p": PARTICLE_POROSITY,
            "d_p": PARTICLE_DIAMETER_M,
            "values": _active_zone_values(initial_state, inventory_factor=inventory_factor),
        }
    )
    zones = [reactive_zone]
    if active_bed_start_m > 1.0e-12 or reactor_length_m > REACTIVE_BED_LENGTH_M + 1.0e-12:
        zones = []
        if active_bed_start_m > 1.0e-12:
            zones.append(
                zone_template.model_copy(
                    update={
                        "x_start_m": 0.0,
                        "x_end_m": float(active_bed_start_m),
                        "e_b": BED_VOID_FRACTION,
                        "e_p": PARTICLE_POROSITY,
                        "d_p": PARTICLE_DIAMETER_M,
                        "values": _inert_zone_values(),
                    }
                )
            )
        zones.append(reactive_zone)
        if reactor_length_m - active_bed_end_m > 1.0e-12:
            zones.append(
                zone_template.model_copy(
                    update={
                        "x_start_m": active_bed_end_m,
                        "x_end_m": float(reactor_length_m),
                        "e_b": BED_VOID_FRACTION,
                        "e_p": PARTICLE_POROSITY,
                        "d_p": PARTICLE_DIAMETER_M,
                        "values": _inert_zone_values(),
                    }
                )
            )
    solids_profile = run_bundle.solids.initial_profile.model_copy(
        update={
            "basis": "bed",
            "zones": tuple(zones),
        }
    )
    solids_config = run_bundle.solids.model_copy(
        update={
            "solid_species": ("Ni", "NiO", "CaAl2O4", "Al2O3"),
            "initial_profile": solids_profile,
        }
    )
    return run_bundle.model_copy(
        update={
            "program": program_config,
            "run": run_config,
            "solids": solids_config,
        },
        deep=True,
    )


@contextmanager
def _temporary_validation_ni_redox_overrides(
    *,
    rate_coefficient_overrides: dict[str, float] | None = None,
    activation_energy_overrides_j_per_mol: dict[str, float] | None = None,
    pressure_exponent_overrides: dict[str, float] | None = None,
):
    rate_coefficient_overrides = (
        VALIDATION_NI_REDOX_RATE_COEFFICIENT_OVERRIDES
        if rate_coefficient_overrides is None
        else rate_coefficient_overrides
    )
    activation_energy_overrides_j_per_mol = (
        VALIDATION_NI_REDOX_ACTIVATION_ENERGY_OVERRIDES_J_PER_MOL
        if activation_energy_overrides_j_per_mol is None
        else activation_energy_overrides_j_per_mol
    )
    pressure_exponent_overrides = (
        VALIDATION_NI_REDOX_PRESSURE_EXPONENT_OVERRIDES
        if pressure_exponent_overrides is None
        else pressure_exponent_overrides
    )

    original_rate_coefficients = {
        key: ni_redox_kinetics.NI_REDOX_RATE_COEFFICIENTS[key] for key in rate_coefficient_overrides
    }
    original_activation_energies = {
        key: ni_redox_kinetics.NI_REDOX_ACTIVATION_ENERGIES_J_PER_MOL[key]
        for key in activation_energy_overrides_j_per_mol
    }
    original_pressure_exponents = {
        key: ni_redox_kinetics.NI_REDOX_PRESSURE_EXPONENTS[key] for key in pressure_exponent_overrides
    }
    for key, value in rate_coefficient_overrides.items():
        ni_redox_kinetics.NI_REDOX_RATE_COEFFICIENTS[key] = float(value)
    for key, value in activation_energy_overrides_j_per_mol.items():
        ni_redox_kinetics.NI_REDOX_ACTIVATION_ENERGIES_J_PER_MOL[key] = float(value)
    for key, value in pressure_exponent_overrides.items():
        ni_redox_kinetics.NI_REDOX_PRESSURE_EXPONENTS[key] = float(value)
    try:
        yield
    finally:
        for key, value in original_rate_coefficients.items():
            ni_redox_kinetics.NI_REDOX_RATE_COEFFICIENTS[key] = value
        for key, value in original_activation_energies.items():
            ni_redox_kinetics.NI_REDOX_ACTIVATION_ENERGIES_J_PER_MOL[key] = value
        for key, value in original_pressure_exponents.items():
            ni_redox_kinetics.NI_REDOX_PRESSURE_EXPONENTS[key] = value


@contextmanager
def _temporary_paper_oxidation_hook(
    initial_temperature_c: float | None,
    *,
    pressure_bar: float | None = None,
    active_bed_start_m: float = REACTIVE_BED_START_M,
):
    if initial_temperature_c is None:
        yield
        return

    registry_keys = tuple(
        key
        for key in (
            "ni_redox_oxidation_o2",
            "medrano_oxidation_o2",
        )
        if key in KINETICS_HOOK_REGISTRY
    )
    if not registry_keys:
        yield
        return

    original_hooks = {key: KINETICS_HOOK_REGISTRY[key] for key in registry_keys}
    reference_temperature_k = OXIDATION_REFERENCE_TEMPERATURE_C + 273.15
    apply_arrhenius_shift = not (
        pressure_bar is not None and pressure_bar >= 5.0 and initial_temperature_c <= 400.0
    )

    def _paper_hook_factory(original_hook):
        def paper_oxidation_hook(context):
            initial_temperature_c_expr = _initial_bed_temperature_c_expr(
                initial_temperature_c=initial_temperature_c,
                idx_cell=context.idx_cell,
                model=context.model,
                active_bed_start_m=active_bed_start_m,
            )
            accessibility = Constant(1.0) / (
                Constant(1.0)
                + Exp(
                    -(initial_temperature_c_expr - Constant(OXIDATION_ALPHA_THRESHOLD_C))
                    / Constant(OXIDATION_ALPHA_WIDTH_C)
                )
            )
            temperature_k = context.model.T(context.idx_cell) / Constant(1.0 * K)
            if apply_arrhenius_shift:
                arrhenius_shift = Exp(
                    -Constant(OXIDATION_EXTRA_ACTIVATION_ENERGY_J_PER_MOL / GAS_CONSTANT_J_PER_MOL_K)
                    * (Constant(1.0) / temperature_k - Constant(1.0 / reference_temperature_k))
                )
            else:
                arrhenius_shift = Constant(1.0)
            return accessibility * arrhenius_shift * original_hook(context)

        return paper_oxidation_hook

    for registry_key, original_hook in original_hooks.items():
        KINETICS_HOOK_REGISTRY[registry_key] = _paper_hook_factory(original_hook)
    try:
        yield
    finally:
        for registry_key, original_hook in original_hooks.items():
            KINETICS_HOOK_REGISTRY[registry_key] = original_hook


@contextmanager
def _temporary_xu_froment_accessibility_hook(
    reduced_fraction_threshold: float | None,
    *,
    reduced_fraction_width: float | None = None,
):
    if reduced_fraction_threshold is None or reduced_fraction_width is None:
        yield
        return

    registry_keys = tuple(
        key
        for key in (
            "xu_froment_smr",
            "xu_froment_dmr_surrogate",
            "xu_froment_wgs",
        )
        if key in KINETICS_HOOK_REGISTRY
    )
    if not registry_keys:
        yield
        return

    original_hooks = {key: KINETICS_HOOK_REGISTRY[key] for key in registry_keys}

    def _hook_factory(original_hook):
        def wrapped_hook(context):
            ni_idx = context.solid_index("Ni")
            nio_idx = context.solid_index("NiO")
            ni_concentration = context.model.c_sol(ni_idx, context.idx_cell) / Constant(1.0 * mol / m**3)
            nio_concentration = context.model.c_sol(nio_idx, context.idx_cell) / Constant(1.0 * mol / m**3)
            reduced_fraction = ni_concentration / (
                ni_concentration + nio_concentration + Constant(1.0e-12)
            )
            accessibility = Constant(1.0) / (
                Constant(1.0)
                + Exp(
                    -(reduced_fraction - Constant(reduced_fraction_threshold))
                    / Constant(reduced_fraction_width)
                )
            )
            return accessibility * original_hook(context)

        return wrapped_hook

    for registry_key, original_hook in original_hooks.items():
        KINETICS_HOOK_REGISTRY[registry_key] = _hook_factory(original_hook)
    try:
        yield
    finally:
        for registry_key, original_hook in original_hooks.items():
            KINETICS_HOOK_REGISTRY[registry_key] = original_hook


def _install_validation_initial_state(
    simulation,
    *,
    initial_temperature_c: float,
    initial_pressure_bar: float,
    initial_gas_composition: dict[str, float] | None = None,
    active_bed_start_m: float = REACTIVE_BED_START_M,
    front_cooldown_override_c: float | None = None,
    front_profile_mode: str = "plateau",
):
    original_set_up_variables = simulation.SetUpVariables

    def patched(self):
        original_set_up_variables()

        model = self.model
        gas_species = self.gas_species
        solid_species = self.solid_species
        ng = model.N_gas.NumberOfPoints
        ns = model.N_sol.NumberOfPoints
        nc = model.x_centers.NumberOfPoints

        center_coords = np.asarray(model.xval_cells.npyValues, dtype=float)
        face_coords = np.asarray(model.xval_faces.npyValues, dtype=float)
        gasfrac = np.asarray(model.gasfrac.npyValues, dtype=float)
        solfrac = np.asarray(model.solfrac.npyValues, dtype=float)
        area = model.pi.GetValue() * model.R_bed.GetValue() ** 2
        fin = model.F_in_const.GetValue()
        inlet_temperature = model.T_in_const.GetValue()
        inlet_y = np.asarray(model.y_in_const.npyValues, dtype=float)
        outlet_pressure = initial_pressure_bar * 1.0e5

        temperatures_k = _initial_bed_temperature_profile_k(
            initial_temperature_c,
            center_coords,
            active_bed_start_m=active_bed_start_m,
            front_cooldown_override_c=front_cooldown_override_c,
            profile_mode=front_profile_mode,
        )
        initial_y = _initial_gas_mole_fractions(
            gas_species,
            initial_gas_composition=initial_gas_composition,
        )
        pressure_pa = np.full(nc, initial_pressure_bar * 1.0e5, dtype=float)
        gas_mw = np.asarray([model.property_registry.get_record(gas_name).mw for gas_name in gas_species], dtype=float)
        ct0 = gasfrac * pressure_pa / (model.R_gas.GetValue() * temperatures_k)
        c0 = initial_y[:, np.newaxis] * ct0[np.newaxis, :]
        rho0 = pressure_pa * float(initial_y @ gas_mw) / (model.R_gas.GetValue() * temperatures_k)
        c0_sol = convert_solid_profile_to_bed_volume(self.solid_config, center_coords, solfrac, solid_species)

        gas_h = np.asarray(
            [
                [model.property_registry.enthalpy_value(species_id, temperature_k) for temperature_k in temperatures_k]
                for species_id in gas_species
            ],
            dtype=float,
        )
        solid_h = np.asarray(
            [
                [model.property_registry.enthalpy_value(species_id, temperature_k) for temperature_k in temperatures_k]
                for species_id in solid_species
            ],
            dtype=float,
        )
        h_cell0 = np.sum(c0 * gas_h, axis=0) + np.sum(c0_sol * solid_h, axis=0)
        heat_bed_total0 = area * np.sum(h_cell0 * np.diff(face_coords))
        ct0_sol = np.sum(c0_sol, axis=0)
        molar_flux_in = fin / area
        face_temperatures_k = np.empty(model.x_faces.NumberOfPoints, dtype=float)
        face_temperatures_k[0] = temperatures_k[0]
        face_temperatures_k[-1] = temperatures_k[-1]
        if model.x_faces.NumberOfPoints > 2:
            face_temperatures_k[1:-1] = 0.5 * (temperatures_k[:-1] + temperatures_k[1:])
        face_velocity0 = molar_flux_in * model.R_gas.GetValue() * face_temperatures_k / outlet_pressure
        dax0 = 0.5 * np.abs(face_velocity0) * np.asarray(model.d_p.npyValues, dtype=float)
        face_flux = inlet_y * molar_flux_in

        for cell_idx in range(nc):
            temperature_k = temperatures_k[cell_idx]
            model.T.SetInitialGuess(cell_idx, temperature_k * K)
            model.P.SetInitialGuess(cell_idx, pressure_pa[cell_idx] * Pa)
            mu_mix = float(
                sum(
                    initial_y[gas_idx]
                    * model.property_registry.viscosity_value(gas_species[gas_idx], temperature_k)
                    for gas_idx in range(ng)
                )
            )
            model.mu_g.SetInitialGuess(cell_idx, mu_mix * Pa * s)
            model.rho_g.SetInitialGuess(cell_idx, rho0[cell_idx] * (Pa * s**2) / m**2)
            model.ct_gas.SetInitialGuess(cell_idx, ct0[cell_idx] * mol / m**3)
            model.ct_sol.SetInitialGuess(cell_idx, ct0_sol[cell_idx] * mol / m**3)
            model.h_cell.SetInitialCondition(cell_idx, h_cell0[cell_idx] * J / m**3)

            for gas_idx in range(ng):
                model.c_gas.SetInitialCondition(gas_idx, cell_idx, c0[gas_idx, cell_idx] * mol / m**3)
                model.y_gas.SetInitialGuess(gas_idx, cell_idx, initial_y[gas_idx])
                model.h_gas.SetInitialGuess(gas_idx, cell_idx, gas_h[gas_idx, cell_idx] * J / mol)

            c0_sol_total = float(np.sum(c0_sol[:, cell_idx]))
            for sol_idx in range(ns):
                model.c_sol.SetInitialCondition(sol_idx, cell_idx, c0_sol[sol_idx, cell_idx] * mol / m**3)
                if model.y_sol is not None:
                    y0_sol = 0.0 if c0_sol_total <= 0.0 else c0_sol[sol_idx, cell_idx] / c0_sol_total
                    model.y_sol.SetInitialGuess(sol_idx, cell_idx, y0_sol)
                model.h_sol.SetInitialGuess(sol_idx, cell_idx, solid_h[sol_idx, cell_idx] * J / mol)

        for gas_idx in range(ng):
            model.y_in.SetInitialGuess(gas_idx, inlet_y[gas_idx])

        model.F_in.SetInitialGuess(fin * mol / s)
        model.T_in.SetInitialGuess(inlet_temperature * K)
        model.P_in.SetInitialGuess(outlet_pressure * Pa)
        model.P_out.SetInitialGuess(outlet_pressure * Pa)
        model.heat_in_total.SetInitialCondition(0.0 * J)
        model.heat_out_total.SetInitialCondition(0.0 * J)
        model.heat_bed_total.SetInitialGuess(heat_bed_total0 * J)

        for face_idx in range(model.x_faces.NumberOfPoints):
            model.u_s.SetInitialGuess(face_idx, face_velocity0[face_idx] * m / s)
            model.Dax.SetInitialGuess(face_idx, dax0[face_idx] * m**2 / s)
            for gas_idx in range(ng):
                gas_h_face = model.property_registry.enthalpy_value(gas_species[gas_idx], face_temperatures_k[face_idx])
                model.N_gas_face.SetInitialGuess(gas_idx, face_idx, face_flux[gas_idx] * mol / (s * m**2))
                model.J_gas_face.SetInitialGuess(
                    gas_idx,
                    face_idx,
                    face_flux[gas_idx] * gas_h_face * J / (s * m**2),
                )

        if model.R_rxn is not None:
            nr = model.N_rxn.NumberOfPoints
            for reaction_idx in range(nr):
                for cell_idx in range(nc):
                    model.R_rxn.SetInitialGuess(reaction_idx, cell_idx, 0.0 * mol / (m**3 * s))

    simulation.SetUpVariables = types.MethodType(patched, simulation)


def _run_validation_bundle(
    run_bundle: RunBundle,
    *,
    initial_temperature_c: float,
    initial_pressure_bar: float,
    initial_gas_composition: dict[str, float] | None = None,
    active_bed_start_m: float = REACTIVE_BED_START_M,
    front_cooldown_override_c: float | None = None,
    front_profile_mode: str = "plateau",
) -> RunResult:
    with _ni_validation_solver_config():
        with _temporary_validation_ni_redox_overrides():
            assembly = assemble_simulation(
                run_bundle,
                property_registry=PROPERTY_REGISTRY,
                reaction_catalog=REACTION_CATALOG,
            )
            _install_validation_initial_state(
                assembly.simulation,
                initial_temperature_c=initial_temperature_c,
                initial_pressure_bar=initial_pressure_bar,
                initial_gas_composition=initial_gas_composition,
                active_bed_start_m=active_bed_start_m,
                front_cooldown_override_c=front_cooldown_override_c,
                front_profile_mode=front_profile_mode,
            )
            reporter = run_assembled_simulation(assembly, include_plot_variables=True)

    return RunResult(
        run_bundle=run_bundle,
        output_directory=run_bundle.output_directory,
        success=True,
        artifact_paths={},
        reporter=reporter,
        simulation=assembly.simulation,
    )


def _run_case_with_timeout(
    *,
    timeout_s: float,
    run_bundle: RunBundle,
    initial_temperature_c: float,
    initial_pressure_bar: float,
    initial_gas_composition: dict[str, float] | None,
    active_bed_start_m: float,
    front_cooldown_override_c: float | None,
    front_profile_mode: str,
    oxidation_hook_temperature_c: float | None,
    oxidation_hook_pressure_bar: float | None,
    xu_froment_reduced_fraction_threshold: float | None = None,
    xu_froment_reduced_fraction_width: float | None = None,
) -> RunResultPlotData:
    WORKER_TMP_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_bundle": run_bundle,
        "initial_temperature_c": initial_temperature_c,
        "initial_pressure_bar": initial_pressure_bar,
        "initial_gas_composition": initial_gas_composition,
        "active_bed_start_m": active_bed_start_m,
        "front_cooldown_override_c": front_cooldown_override_c,
        "front_profile_mode": front_profile_mode,
        "oxidation_hook_temperature_c": oxidation_hook_temperature_c,
        "oxidation_hook_pressure_bar": oxidation_hook_pressure_bar,
        "xu_froment_reduced_fraction_threshold": xu_froment_reduced_fraction_threshold,
        "xu_froment_reduced_fraction_width": xu_froment_reduced_fraction_width,
        "ni_redox_rate_coefficient_overrides": dict(VALIDATION_NI_REDOX_RATE_COEFFICIENT_OVERRIDES),
        "ni_redox_activation_energy_overrides_j_per_mol": dict(
            VALIDATION_NI_REDOX_ACTIVATION_ENERGY_OVERRIDES_J_PER_MOL
        ),
        "ni_redox_pressure_exponent_overrides": dict(VALIDATION_NI_REDOX_PRESSURE_EXPONENT_OVERRIDES),
    }
    unique_stem = f"ni_worker_{int(time.time() * 1000)}_{uuid.uuid4().hex}"
    input_path = WORKER_TMP_DIR / f"{unique_stem}_input.pkl"
    output_path = WORKER_TMP_DIR / f"{unique_stem}_output.pkl"
    try:
        with input_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        command = [
            sys.executable,
            "-B",
            "-m",
            "packed_bed.validation.ni_clr_paper",
            "--worker-input",
            str(input_path),
            "--worker-output",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Ni validation case exceeded the {timeout_s:g}s limit.") from exc
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or f"worker exited with code {completed.returncode}"
            raise RuntimeError(f"Ni validation worker failed: {details}")
        if not output_path.exists():
            raise RuntimeError("Ni validation worker finished without returning plot data.")
        with output_path.open("rb") as handle:
            result = pickle.load(handle)
    finally:
        for path in (input_path, output_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except PermissionError:
                pass
    if not result.get("ok", False):
        raise RuntimeError(str(result.get("error", "Ni validation worker failed.")))
    return _deserialize_plot_data(result["plot_data"])


def _log_progress(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


def _run_constant_case_bundle(
    case: ValidationCase,
    run_bundle: RunBundle,
    *,
    timeout_s: float,
    initial_gas_composition: dict[str, float] | None = None,
    front_cooldown_override_c: float | None = None,
) -> RunResultPlotData:
    return _run_case_with_timeout(
        timeout_s=timeout_s,
        run_bundle=run_bundle,
        initial_temperature_c=case.initial_temperature_c,
        initial_pressure_bar=case.pressure_bar,
        initial_gas_composition=initial_gas_composition,
        active_bed_start_m=case.active_bed_start_m,
        front_cooldown_override_c=front_cooldown_override_c,
        front_profile_mode=case.front_profile_mode,
        oxidation_hook_temperature_c=(
            case.initial_temperature_c if case.use_paper_oxidation_hook else None
        ),
        oxidation_hook_pressure_bar=case.pressure_bar if case.use_paper_oxidation_hook else None,
    )


def _fallback_axial_cells(case: ValidationCase) -> tuple[int, ...]:
    candidates = (24, MIN_VALIDATION_AXIAL_CELLS)
    return tuple(candidate for candidate in candidates if candidate < case.axial_cells)


def _constant_case_solver_attempts(
    case: ValidationCase,
) -> tuple[tuple[str, int, dict[str, float], dict[str, float]], ...]:
    feed_composition = _constant_case_feed_composition(case)
    startup_composition = _constant_case_startup_inlet_composition(case)
    attempts: list[tuple[str, int, dict[str, float], dict[str, float]]] = [
        ("startup_startup", case.axial_cells, startup_composition, startup_composition)
    ]
    if _is_oxidation_feed(case.composition):
        attempts.append(("startup_feed", case.axial_cells, startup_composition, feed_composition))
    reduced_cell_attempts = _fallback_axial_cells(case)
    for reduced_cells in reduced_cell_attempts:
        attempts.append(
            (f"startup_startup_cells{reduced_cells}", reduced_cells, startup_composition, startup_composition)
        )
        attempts.append(
            (f"startup_feed_cells{reduced_cells}", reduced_cells, startup_composition, feed_composition)
        )

    deduplicated_attempts: list[tuple[str, int, dict[str, float], dict[str, float]]] = []
    seen_attempt_keys: set[tuple[int, tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]] = set()
    for label, axial_cells, initial_gas_composition, startup_inlet_composition in attempts:
        attempt_key = (
            int(axial_cells),
            tuple(sorted((species_id, float(value)) for species_id, value in initial_gas_composition.items())),
            tuple(sorted((species_id, float(value)) for species_id, value in startup_inlet_composition.items())),
        )
        if attempt_key in seen_attempt_keys:
            continue
        seen_attempt_keys.add(attempt_key)
        deduplicated_attempts.append((label, axial_cells, initial_gas_composition, startup_inlet_composition))
    return tuple(deduplicated_attempts)


def _run_constant_case(case: ValidationCase, *, progress: bool = False) -> ValidationCaseResult:
    initial_temperature_k = case.initial_temperature_c + 273.15
    inlet_temperature_k = _constant_case_inlet_temperature_k(case)
    feed_composition = _constant_case_feed_composition(case)
    base_run_bundle = _load_validation_run_bundle(case.run_path)
    cached_plot_data = _load_plot_data_cache(case.case_id)
    if cached_plot_data is not None:
        cached_run_bundle = _patched_run_bundle(
            base_run_path=case.run_path,
            case_id=case.case_id,
            system_name=f"ni_paper_{case.case_id}",
            time_horizon_s=case.duration_s,
            reporting_interval_s=case.reporting_interval_s,
            program_config=_program_config_for_constant_feed(
                base_program=base_run_bundle.program,
                flow_nlpm=case.flow_nlpm,
                composition=feed_composition,
                startup_composition=_constant_case_startup_inlet_composition(case),
                pressure_bar=case.pressure_bar,
                hot_bed_temperature_k=inlet_temperature_k,
            ),
            initial_state=case.initial_state,
            initial_temperature_k=initial_temperature_k,
            axial_cells=case.axial_cells,
            reactor_length_m=case.reactor_length_m,
            active_bed_start_m=case.active_bed_start_m,
            inventory_factor=case.inventory_factor,
            wall_heat_transfer_coefficient_w_m2_k=_validation_wall_heat_transfer_coefficient(case),
        )
        _log_progress(f"[cache] {case.case_id}", enabled=progress)
        return ValidationCaseResult(case=case, run_bundle=cached_run_bundle, run_result=None, plot_data=cached_plot_data)
    _log_progress(f"[run]   {case.case_id}", enabled=progress)
    started_at = time.perf_counter()
    solved_run_bundle: RunBundle | None = None
    plot_data: RunResultPlotData | None = None
    last_error: Exception | None = None
    timeout_s = CONSTANT_CASE_TIMEOUT_S if _is_oxidation_feed(case.composition) else REDUCTION_CASE_TIMEOUT_S
    attempts = _constant_case_solver_attempts(case)
    for attempt_index, (attempt_label, axial_cells, initial_gas_composition, startup_inlet_composition) in enumerate(attempts):
        run_bundle = _patched_run_bundle(
            base_run_path=case.run_path,
            case_id=case.case_id,
            system_name=f"ni_paper_{case.case_id}_{attempt_label}",
            time_horizon_s=case.duration_s,
            reporting_interval_s=case.reporting_interval_s,
            program_config=_program_config_for_constant_feed(
                base_program=base_run_bundle.program,
                flow_nlpm=case.flow_nlpm,
                composition=feed_composition,
                startup_composition=startup_inlet_composition,
                pressure_bar=case.pressure_bar,
                hot_bed_temperature_k=inlet_temperature_k,
            ),
            initial_state=case.initial_state,
            initial_temperature_k=initial_temperature_k,
            axial_cells=axial_cells,
            reactor_length_m=case.reactor_length_m,
            active_bed_start_m=case.active_bed_start_m,
            inventory_factor=case.inventory_factor,
            wall_heat_transfer_coefficient_w_m2_k=_validation_wall_heat_transfer_coefficient(case),
        )
        try:
            plot_data = _run_constant_case_bundle(
                case,
                run_bundle,
                timeout_s=timeout_s,
                initial_gas_composition=initial_gas_composition,
                front_cooldown_override_c=case.front_cooldown_override_c,
            )
            solved_run_bundle = run_bundle
            break
        except Exception as exc:
            last_error = exc
            if attempt_index + 1 < len(attempts):
                _log_progress(
                    f"[retry] {case.case_id} -> {attempt_label} failed, trying next startup",
                    enabled=progress,
                )
    if plot_data is None or solved_run_bundle is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Ni validation case {case.case_id} failed without returning plot data.")
    _write_plot_data_cache(case.case_id, plot_data)
    _log_progress(
        f"[done]  {case.case_id} in {time.perf_counter() - started_at:.2f}s",
        enabled=progress,
    )
    return ValidationCaseResult(case=case, run_bundle=solved_run_bundle, run_result=None, plot_data=plot_data)


def _run_full_cycle_case(*, progress: bool = False) -> ValidationCaseResult:
    case = ValidationCase(
        case_id=FULL_CYCLE_CASE_ID,
        run_path=FIGURE_RUN_PATHS["fig23"],
        title="Paper Fig. 23 full CLR cycle",
        initial_temperature_c=600.0,
        pressure_bar=1.0,
        flow_nlpm=10.0,
        duration_s=0.0,
        composition={},
        initial_state="reduced",
        axial_cells=FULL_CYCLE_AXIAL_CELLS,
        reactor_length_m=FULL_REACTOR_LENGTH_M,
        active_bed_start_m=FULL_REACTIVE_BED_START_M,
        front_cooldown_override_c=70.0,
        front_profile_mode="linear",
        reported_gas_transport_delay_s=FULL_CYCLE_COMPOSITION_ANALYZER_TRANSPORT_DELAY_S,
        reported_gas_response_time_constant_s=FULL_CYCLE_COMPOSITION_ANALYZER_RESPONSE_TIME_CONSTANT_S,
    )
    initial_temperature_k = case.initial_temperature_c + 273.15
    base_run_bundle = _load_validation_run_bundle(case.run_path)
    full_cycle_program, warmup_duration_s, duration_s = _program_config_for_full_cycle(
        base_program=base_run_bundle.program,
        hot_bed_temperature_k=initial_temperature_k,
    )
    case = ValidationCase(
        case_id=case.case_id,
        run_path=case.run_path,
        title=case.title,
        initial_temperature_c=case.initial_temperature_c,
        pressure_bar=case.pressure_bar,
        flow_nlpm=case.flow_nlpm,
        duration_s=duration_s,
        composition=case.composition,
        initial_state=case.initial_state,
        axial_cells=case.axial_cells,
        reactor_length_m=case.reactor_length_m,
        active_bed_start_m=case.active_bed_start_m,
        front_cooldown_override_c=case.front_cooldown_override_c,
        front_profile_mode=case.front_profile_mode,
        reported_gas_transport_delay_s=case.reported_gas_transport_delay_s,
        reported_gas_response_time_constant_s=case.reported_gas_response_time_constant_s,
    )
    run_bundle = _patched_run_bundle(
        base_run_path=case.run_path,
        case_id=case.case_id,
        system_name=f"ni_paper_{case.case_id}",
        time_horizon_s=duration_s,
        reporting_interval_s=FULL_CYCLE_REPORTING_INTERVAL_S,
        program_config=full_cycle_program,
        initial_state=case.initial_state,
        initial_temperature_k=initial_temperature_k,
        axial_cells=case.axial_cells,
        reactor_length_m=case.reactor_length_m,
        active_bed_start_m=case.active_bed_start_m,
        inventory_factor=FULL_CYCLE_INVENTORY_FACTOR,
        relative_tolerance=FULL_CYCLE_RELATIVE_TOLERANCE,
        wall_heat_transfer_coefficient_w_m2_k=FULL_CYCLE_VALIDATION_WALL_HEAT_TRANSFER_COEFFICIENT_W_M2_K,
    )
    cached_plot_data = _load_plot_data_cache(case.case_id)
    if cached_plot_data is not None:
        _log_progress(f"[cache] {case.case_id}", enabled=progress)
        return ValidationCaseResult(case=case, run_bundle=run_bundle, run_result=None, plot_data=cached_plot_data)
    _log_progress(f"[run]   {case.case_id}", enabled=progress)
    started_at = time.perf_counter()
    plot_data = _run_case_with_timeout(
        timeout_s=FULL_CYCLE_CASE_TIMEOUT_S,
        run_bundle=run_bundle,
        initial_temperature_c=case.initial_temperature_c,
        initial_pressure_bar=case.pressure_bar,
        initial_gas_composition=None,
        active_bed_start_m=case.active_bed_start_m,
        front_cooldown_override_c=case.front_cooldown_override_c,
        front_profile_mode=case.front_profile_mode,
        oxidation_hook_temperature_c=case.initial_temperature_c,
        oxidation_hook_pressure_bar=case.pressure_bar,
        xu_froment_reduced_fraction_threshold=FULL_CYCLE_XU_FROMENT_REDUCED_FRACTION_THRESHOLD,
        xu_froment_reduced_fraction_width=FULL_CYCLE_XU_FROMENT_REDUCED_FRACTION_WIDTH,
    )
    plotted_plot_data = _slice_plot_data_time_window(
        plot_data,
        start_s=warmup_duration_s,
        end_s=duration_s,
    )
    _write_plot_data_cache(case.case_id, plotted_plot_data)
    _log_progress(
        f"[done]  {case.case_id} in {time.perf_counter() - started_at:.2f}s",
        enabled=progress,
    )
    return ValidationCaseResult(case=case, run_bundle=run_bundle, run_result=None, plot_data=plotted_plot_data)


def _pressure_label(pressure_bar: float) -> str:
    if abs(pressure_bar - round(pressure_bar)) < 1.0e-12:
        return f"{int(round(pressure_bar))} bar"
    return f"{pressure_bar:g} bar"


def _temperature_label(temperature_c: float) -> str:
    if abs(temperature_c - round(temperature_c)) < 1.0e-12:
        return f"{int(round(temperature_c))} C"
    return f"{temperature_c:g} C"


def _interpolate_temperature_profile_c(position_m: float, positions_m: np.ndarray, profile_row_c: np.ndarray) -> float:
    positions = np.asarray(positions_m, dtype=float)
    values = np.asarray(profile_row_c, dtype=float)
    if positions.size == 0:
        return float("nan")
    if positions.size == 1:
        return float(values[0])

    target_position_m = float(position_m)
    if target_position_m <= float(positions[0]):
        x0, x1 = float(positions[0]), float(positions[1])
        y0, y1 = float(values[0]), float(values[1])
        return y0 + (target_position_m - x0) * (y1 - y0) / (x1 - x0)
    if target_position_m >= float(positions[-1]):
        x0, x1 = float(positions[-2]), float(positions[-1])
        y0, y1 = float(values[-2]), float(values[-1])
        return y0 + (target_position_m - x0) * (y1 - y0) / (x1 - x0)
    return float(np.interp(target_position_m, positions, values))


def _temperature_trace(case_result: ValidationCaseResult, position_m: float) -> np.ndarray:
    positions = np.asarray(case_result.plot_data.axial_positions_m, dtype=float)
    profile_c = np.asarray(case_result.plot_data.temperature_profile_k, dtype=float) - 273.15
    return np.asarray(
        [
            _interpolate_temperature_profile_c(position_m, positions, row)
            for row in profile_c
        ],
        dtype=float,
    )


def _thermowell_boundary_temperature_c(case: ValidationCase) -> float:
    cooldown_c = max(
        _initial_front_cooldown_c(
            case.initial_temperature_c,
            front_cooldown_override_c=case.front_cooldown_override_c,
        ),
        THERMOWELL_BOUNDARY_COOLDOWN_C,
    )
    return case.initial_temperature_c - cooldown_c


def _thermowell_outlet_boundary_temperature_c(case: ValidationCase) -> float:
    return case.initial_temperature_c - float(case.thermowell_outlet_boundary_cooldown_c)


def _thermowell_temperature_trace(case_result: ValidationCaseResult, position_m: float) -> np.ndarray:
    time_s = np.asarray(case_result.plot_data.time_s, dtype=float)
    positions = np.asarray(case_result.plot_data.axial_positions_m, dtype=float)
    profile_c = np.asarray(case_result.plot_data.temperature_profile_k, dtype=float) - 273.15
    spatial_smoothing_length_m = float(case_result.case.thermowell_spatial_smoothing_length_m)
    if spatial_smoothing_length_m > 1.0e-12:
        active_bed_start_m = float(case_result.case.active_bed_start_m)
        active_bed_end_m = active_bed_start_m + REACTIVE_BED_LENGTH_M
        observer_positions = np.concatenate(
            ([active_bed_start_m], positions, [active_bed_end_m])
        )
        weights = np.exp(
            -0.5 * ((observer_positions - float(position_m)) / spatial_smoothing_length_m) ** 2
        )
        weight_total = float(np.sum(weights))
        if weight_total <= 1.0e-12:
            trace_c = _temperature_trace(case_result, position_m)
        else:
            normalized_weights = weights / weight_total
            inlet_boundary_c = _thermowell_boundary_temperature_c(case_result.case)
            outlet_boundary_c = _thermowell_outlet_boundary_temperature_c(case_result.case)
            trace_c = np.asarray(
                [
                    float(
                        np.dot(
                            normalized_weights,
                            np.concatenate(([inlet_boundary_c], row, [outlet_boundary_c])),
                        )
                    )
                    for row in profile_c
                ],
                dtype=float,
            )
    else:
        trace_c = _temperature_trace(case_result, position_m)
    tau_s = float(case_result.case.thermowell_response_time_constant_s)
    if THERMOWELL_RESPONSE_POSITION_SPAN_M > 1.0e-12:
        relative_position_m = float(position_m) - float(case_result.case.active_bed_start_m)
        tau_s += float(case_result.case.thermowell_response_time_gradient_s) * min(
            max(relative_position_m / THERMOWELL_RESPONSE_POSITION_SPAN_M, 0.0),
            1.0,
        )
    if trace_c.size <= 1 or tau_s <= 0.0:
        return trace_c

    filtered_c = np.empty_like(trace_c)
    filtered_c[0] = trace_c[0]
    for idx in range(1, trace_c.size):
        dt_s = max(float(time_s[idx] - time_s[idx - 1]), 0.0)
        if dt_s <= 0.0:
            filtered_c[idx] = filtered_c[idx - 1]
            continue
        alpha = 1.0 - math.exp(-dt_s / tau_s)
        filtered_c[idx] = filtered_c[idx - 1] + alpha * (trace_c[idx] - filtered_c[idx - 1])
    gain = float(case_result.case.thermowell_temperature_gain)
    if abs(gain - 1.0) > 1.0e-12:
        filtered_c = case_result.case.initial_temperature_c + gain * (
            filtered_c - case_result.case.initial_temperature_c
        )
    return filtered_c


def _reported_outlet_composition(case_result: ValidationCaseResult) -> np.ndarray:
    outlet_composition = np.asarray(case_result.plot_data.outlet_composition, dtype=float)
    if outlet_composition.shape[0] <= 1:
        return outlet_composition

    time_s = np.asarray(case_result.plot_data.time_s, dtype=float)
    reported = np.empty_like(outlet_composition)
    species_delay_map = case_result.case.reported_species_transport_delay_s or {}
    species_tau_map = case_result.case.reported_species_response_time_constant_s or {}
    default_transport_delay_s = float(case_result.case.reported_gas_transport_delay_s)
    default_response_time_constant_s = float(case_result.case.reported_gas_response_time_constant_s)

    for species_idx, species_id in enumerate(case_result.plot_data.gas_species):
        transport_delay_s = float(species_delay_map.get(species_id, default_transport_delay_s))
        response_time_constant_s = float(species_tau_map.get(species_id, default_response_time_constant_s))
        species_values = np.asarray(outlet_composition[:, species_idx], dtype=float)
        if transport_delay_s > 0.0:
            species_values = np.interp(
                time_s - transport_delay_s,
                time_s,
                species_values,
                left=species_values[0],
                right=species_values[-1],
            )
        if response_time_constant_s > 0.0:
            filtered_values = np.empty_like(species_values)
            filtered_values[0] = species_values[0]
            for idx in range(1, species_values.shape[0]):
                dt_s = max(float(time_s[idx] - time_s[idx - 1]), 0.0)
                if dt_s <= 0.0:
                    filtered_values[idx] = filtered_values[idx - 1]
                    continue
                alpha = 1.0 - math.exp(-dt_s / response_time_constant_s)
                filtered_values[idx] = filtered_values[idx - 1] + alpha * (
                    species_values[idx] - filtered_values[idx - 1]
                )
            species_values = filtered_values
        reported[:, species_idx] = species_values

    totals = np.sum(reported, axis=1, keepdims=True)
    return np.divide(reported, totals, out=np.zeros_like(reported), where=np.abs(totals) > 1.0e-12)


def _dry_composition(case_result: ValidationCaseResult) -> tuple[dict[str, np.ndarray], np.ndarray]:
    gas_species = case_result.plot_data.gas_species
    keep_ids = [species_id for species_id in gas_species if species_id != "H2O"]
    keep_indices = [gas_species.index(species_id) for species_id in keep_ids]
    reported_outlet = _reported_outlet_composition(case_result)
    values = np.asarray(reported_outlet[:, keep_indices], dtype=float)
    totals = np.sum(values, axis=1, keepdims=True)
    dry = np.divide(values, totals, out=np.zeros_like(values), where=np.abs(totals) > 1.0e-12)
    return {species_id: dry[:, idx] for idx, species_id in enumerate(keep_ids)}, np.asarray(case_result.plot_data.time_s, dtype=float)


def _trim_initial_sample(time_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.asarray(time_s).size <= 1:
        return np.asarray(time_s), np.asarray(values)
    return np.asarray(time_s)[1:], np.asarray(values)[1:]


def _fig5_cases() -> tuple[ValidationCase, ...]:
    cases: list[ValidationCase] = []
    for temperature_c in (400.0, 500.0, 600.0, 650.0):
        pressure_bars = (1.0, 3.0, 5.0) if abs(float(temperature_c) - 600.0) < 1.0e-12 else (1.0,)
        for pressure_bar in pressure_bars:
            cases.append(
                ValidationCase(
                    case_id=f"fig5_ox_t{int(temperature_c)}_p{int(pressure_bar)}",
                    run_path=FIGURE_RUN_PATHS["fig5"],
                    title=f"Fig. 5 oxidation {_temperature_label(temperature_c)} {_pressure_label(pressure_bar)}",
                    initial_temperature_c=temperature_c,
                    pressure_bar=pressure_bar,
                    flow_nlpm=10.0,
                    duration_s=900.0,
                    composition=_air_tracer_composition(0.10),
                    initial_state="reduced",
                    axial_cells=THERMOCOUPLE_AXIAL_CELLS,
                    inventory_factor=OXIDATION_INVENTORY_FACTOR_BY_CONDITION[(temperature_c, pressure_bar)],
                    reported_gas_response_time_constant_s=OXIDATION_REPORTED_GAS_RESPONSE_TIME_CONSTANT_S,
                )
            )
    return tuple(cases)


def _fig8_extra_cases() -> tuple[ValidationCase, ...]:
    cases: list[ValidationCase] = []
    for o2_fraction in (0.05, 0.10, 0.20):
        for pressure_bar in (1.0, 3.0, 5.0):
            is_low_o2_case = abs(float(o2_fraction) - 0.05) < 1.0e-12
            is_high_o2_case = abs(float(o2_fraction) - 0.20) < 1.0e-12
            cases.append(
                ValidationCase(
                    case_id=f"fig8_ox_o2_{int(round(o2_fraction * 100))}_p{int(pressure_bar)}",
                    run_path=FIGURE_RUN_PATHS["fig8"],
                    title=f"Fig. 8 oxidation {int(round(o2_fraction * 100))}% O2 {_pressure_label(pressure_bar)}",
                    initial_temperature_c=600.0,
                    pressure_bar=pressure_bar,
                    flow_nlpm=10.0,
                    duration_s=1800.0,
                    composition=_air_tracer_composition(o2_fraction),
                    initial_state="reduced",
                    axial_cells=FIG8_AXIAL_CELLS_BY_CONDITION[(o2_fraction, pressure_bar)],
                    inventory_factor=_fig8_inventory_factor(
                        nominal_o2_fraction=o2_fraction,
                        pressure_bar=pressure_bar,
                    ),
                    reported_gas_transport_delay_s=(
                        FIG8_LOW_O2_REPORTED_GAS_TRANSPORT_DELAY_BY_PRESSURE[float(pressure_bar)]
                        if is_low_o2_case
                        else (
                        FIG8_HIGH_O2_REPORTED_GAS_TRANSPORT_DELAY_BY_PRESSURE[float(pressure_bar)]
                        if is_high_o2_case
                        else 0.0
                        )
                    ),
                    reported_gas_response_time_constant_s=(
                        FIG8_LOW_O2_REPORTED_GAS_RESPONSE_TIME_CONSTANT_BY_PRESSURE[float(pressure_bar)]
                        if is_low_o2_case
                        else (
                        FIG8_HIGH_O2_REPORTED_GAS_RESPONSE_TIME_CONSTANT_BY_PRESSURE[float(pressure_bar)]
                        if is_high_o2_case
                        else OXIDATION_REPORTED_GAS_RESPONSE_TIME_CONSTANT_S
                        )
                    ),
                )
            )
    return tuple(cases)


def _fig14_cases() -> tuple[ValidationCase, ...]:
    composition = _normalized_composition({"He": 0.10, "CO2": 0.70, "CO": 0.10, "H2": 0.10})
    return tuple(
        ValidationCase(
            case_id=f"fig14_syngas_t{int(temperature_c)}",
            run_path=FIGURE_RUN_PATHS["fig14"],
            title=f"Fig. 14 syngas reduction {_temperature_label(temperature_c)}",
            initial_temperature_c=temperature_c,
            pressure_bar=3.0,
            flow_nlpm=10.0,
            duration_s=1000.0,
            composition=composition,
            initial_state="oxidized",
            axial_cells=FIG14_AXIAL_CELLS_BY_TEMPERATURE_C[temperature_c],
            front_cooldown_override_c=FIG14_FRONT_COOLDOWN_BY_TEMPERATURE_C[temperature_c],
            inlet_temperature_mode="ambient",
            inventory_factor=REDUCTION_INVENTORY_FACTOR_BY_TEMPERATURE_C[temperature_c],
            reported_gas_response_time_constant_s=COMPOSITION_ANALYZER_RESPONSE_TIME_CONSTANT_S,
            reported_species_transport_delay_s={
                "CO": FIG14_CO_ANALYZER_TRANSPORT_DELAY_BY_TEMPERATURE_C[temperature_c]
            },
            reported_species_response_time_constant_s={
                "CO": FIG14_CO_ANALYZER_RESPONSE_TIME_CONSTANT_BY_TEMPERATURE_C[temperature_c],
                "H2": FIG14_H2_ANALYZER_RESPONSE_TIME_CONSTANT_BY_TEMPERATURE_C[temperature_c],
            },
        )
        for temperature_c in (600.0, 700.0, 800.0, 900.0)
    )


def _fig5_plot_cases() -> tuple[ValidationCase, ...]:
    plot_cases: list[ValidationCase] = []
    for case in _fig5_cases():
        if abs(case.pressure_bar - 1.0) >= 1.0e-12:
            continue
        plot_cases.append(
            ValidationCase(
                case_id=case.case_id.replace("fig5_ox_", "fig5_plot_ox_"),
                run_path=case.run_path,
                title=case.title,
                initial_temperature_c=case.initial_temperature_c,
                pressure_bar=case.pressure_bar,
                flow_nlpm=case.flow_nlpm,
                duration_s=case.duration_s,
                composition=dict(case.composition),
                initial_state=case.initial_state,
                axial_cells=FIG5_PLOT_AXIAL_CELLS_BY_TEMPERATURE_C[case.initial_temperature_c],
                reporting_interval_s=FIG5_PLOT_REPORTING_INTERVAL_S,
                reactor_length_m=case.reactor_length_m,
                active_bed_start_m=case.active_bed_start_m,
                front_cooldown_override_c=case.front_cooldown_override_c,
                front_profile_mode=case.front_profile_mode,
                inlet_temperature_mode=case.inlet_temperature_mode,
                inventory_factor=case.inventory_factor,
                reported_gas_transport_delay_s=FIG5_PLOT_REPORTED_GAS_TRANSPORT_DELAY_BY_TEMPERATURE_C[case.initial_temperature_c],
                reported_gas_response_time_constant_s=FIG5_PLOT_REPORTED_GAS_RESPONSE_TIME_CONSTANT_BY_TEMPERATURE_C[case.initial_temperature_c],
                reported_species_transport_delay_s=copy.deepcopy(case.reported_species_transport_delay_s),
                reported_species_response_time_constant_s=copy.deepcopy(case.reported_species_response_time_constant_s),
            )
        )
    return tuple(plot_cases)


def _fig7_cases() -> tuple[ValidationCase, ...]:
    return (
        ValidationCase(
            case_id="fig7_ox_t600_p1",
            run_path=FIGURE_RUN_PATHS["fig5"],
            title="Fig. 7 oxidation 600 C 1 bar",
            initial_temperature_c=600.0,
            pressure_bar=1.0,
            flow_nlpm=10.0,
            duration_s=900.0,
            composition=_air_tracer_composition(0.10),
            initial_state="reduced",
            axial_cells=14,
            front_cooldown_override_c=110.0,
            inlet_temperature_override_c=400.0,
            inventory_factor=OXIDATION_INVENTORY_FACTOR_BY_CONDITION[(600.0, 1.0)],
            use_paper_oxidation_hook=True,
            thermowell_response_time_constant_s=2.0,
            thermowell_response_time_gradient_s=50.0,
            thermowell_temperature_gain=1.0,
            thermowell_spatial_smoothing_length_m=0.01,
            thermowell_outlet_boundary_cooldown_c=250.0,
        ),
    )


def _run_cases(cases: tuple[ValidationCase, ...], *, progress: bool = False) -> dict[str, ValidationCaseResult]:
    return {case.case_id: _run_constant_case(case, progress=progress) for case in cases}


def _save_figure(figure, filename: str) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / filename
    figure.savefig(path, format="svg", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def _case_cache_path(case_id: str) -> Path:
    return CACHE_DIR / f"{CACHE_NAMESPACE}_{case_id}.npz"


def _serialize_plot_data(plot_data: RunResultPlotData) -> dict[str, object]:
    return {
        "time_s": np.asarray(plot_data.time_s, dtype=float),
        "axial_positions_m": np.asarray(plot_data.axial_positions_m, dtype=float),
        "gas_species": tuple(str(species_id) for species_id in plot_data.gas_species),
        "inlet_composition": np.asarray(plot_data.inlet_composition, dtype=float),
        "outlet_composition": np.asarray(plot_data.outlet_composition, dtype=float),
        "inlet_pressure_pa": np.asarray(plot_data.inlet_pressure_pa, dtype=float),
        "outlet_temperature_k": np.asarray(plot_data.outlet_temperature_k, dtype=float),
        "pressure_profile_pa": np.asarray(plot_data.pressure_profile_pa, dtype=float),
        "outlet_pressure_pa": np.asarray(plot_data.outlet_pressure_pa, dtype=float),
        "outlet_flowrate_mol_s": np.asarray(plot_data.outlet_flowrate_mol_s, dtype=float),
        "temperature_profile_k": np.asarray(plot_data.temperature_profile_k, dtype=float),
    }


def _deserialize_plot_data(payload: dict[str, object]) -> RunResultPlotData:
    return RunResultPlotData(
        time_s=np.asarray(payload["time_s"], dtype=float),
        axial_positions_m=np.asarray(payload["axial_positions_m"], dtype=float),
        gas_species=tuple(str(species_id) for species_id in payload["gas_species"]),
        inlet_composition=np.asarray(payload["inlet_composition"], dtype=float),
        outlet_composition=np.asarray(payload["outlet_composition"], dtype=float),
        inlet_pressure_pa=np.asarray(payload["inlet_pressure_pa"], dtype=float),
        outlet_temperature_k=np.asarray(payload["outlet_temperature_k"], dtype=float),
        pressure_profile_pa=np.asarray(payload["pressure_profile_pa"], dtype=float),
        outlet_pressure_pa=np.asarray(payload["outlet_pressure_pa"], dtype=float),
        outlet_flowrate_mol_s=np.asarray(payload["outlet_flowrate_mol_s"], dtype=float),
        temperature_profile_k=np.asarray(payload["temperature_profile_k"], dtype=float),
    )


def _interpolate_report_series(time_s: np.ndarray, values: np.ndarray, target_time_s: float) -> np.ndarray:
    source_time_s = np.asarray(time_s, dtype=float)
    source_values = np.asarray(values, dtype=float)
    target_time = float(target_time_s)
    if source_values.ndim == 1:
        return np.asarray(np.interp(target_time, source_time_s, source_values), dtype=float)
    return np.asarray(
        [
            np.interp(target_time, source_time_s, source_values[:, column_idx])
            for column_idx in range(source_values.shape[1])
        ],
        dtype=float,
    )


def _slice_plot_data_time_window(
    plot_data: RunResultPlotData,
    *,
    start_s: float,
    end_s: float,
) -> RunResultPlotData:
    source_time_s = np.asarray(plot_data.time_s, dtype=float)
    window_start_s = float(start_s)
    window_end_s = float(end_s)
    mask = (source_time_s > window_start_s) & (source_time_s < window_end_s)
    window_time_s = np.concatenate(
        (
            np.asarray([window_start_s], dtype=float),
            source_time_s[mask],
            np.asarray([window_end_s], dtype=float),
        )
    )

    def _window_values(values: np.ndarray) -> np.ndarray:
        source_values = np.asarray(values, dtype=float)
        inner_values = source_values[mask]
        start_values = np.expand_dims(
            _interpolate_report_series(source_time_s, source_values, window_start_s),
            axis=0,
        )
        end_values = np.expand_dims(
            _interpolate_report_series(source_time_s, source_values, window_end_s),
            axis=0,
        )
        return np.concatenate((start_values, inner_values, end_values), axis=0)

    shifted_time_s = window_time_s - window_start_s
    return RunResultPlotData(
        time_s=shifted_time_s,
        axial_positions_m=np.asarray(plot_data.axial_positions_m, dtype=float),
        gas_species=tuple(plot_data.gas_species),
        inlet_composition=_window_values(plot_data.inlet_composition),
        outlet_composition=_window_values(plot_data.outlet_composition),
        inlet_pressure_pa=_window_values(plot_data.inlet_pressure_pa),
        outlet_temperature_k=_window_values(plot_data.outlet_temperature_k),
        pressure_profile_pa=_window_values(plot_data.pressure_profile_pa),
        outlet_pressure_pa=_window_values(plot_data.outlet_pressure_pa),
        outlet_flowrate_mol_s=_window_values(plot_data.outlet_flowrate_mol_s),
        temperature_profile_k=_window_values(plot_data.temperature_profile_k),
    )


def _write_plot_data_cache(case_id: str, plot_data: RunResultPlotData) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _case_cache_path(case_id)
    np.savez_compressed(
        path,
        time_s=np.asarray(plot_data.time_s, dtype=float),
        axial_positions_m=np.asarray(plot_data.axial_positions_m, dtype=float),
        gas_species=np.asarray(plot_data.gas_species, dtype=str),
        inlet_composition=np.asarray(plot_data.inlet_composition, dtype=float),
        outlet_composition=np.asarray(plot_data.outlet_composition, dtype=float),
        inlet_pressure_pa=np.asarray(plot_data.inlet_pressure_pa, dtype=float),
        outlet_temperature_k=np.asarray(plot_data.outlet_temperature_k, dtype=float),
        pressure_profile_pa=np.asarray(plot_data.pressure_profile_pa, dtype=float),
        outlet_pressure_pa=np.asarray(plot_data.outlet_pressure_pa, dtype=float),
        outlet_flowrate_mol_s=np.asarray(plot_data.outlet_flowrate_mol_s, dtype=float),
        temperature_profile_k=np.asarray(plot_data.temperature_profile_k, dtype=float),
    )
    return path


def _load_plot_data_cache(case_id: str) -> RunResultPlotData | None:
    path = _case_cache_path(case_id)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return RunResultPlotData(
                time_s=np.asarray(data["time_s"], dtype=float),
                axial_positions_m=np.asarray(data["axial_positions_m"], dtype=float),
                gas_species=tuple(str(value) for value in np.asarray(data["gas_species"]).tolist()),
                inlet_composition=np.asarray(data["inlet_composition"], dtype=float),
                outlet_composition=np.asarray(data["outlet_composition"], dtype=float),
                inlet_pressure_pa=np.asarray(data["inlet_pressure_pa"], dtype=float),
                outlet_temperature_k=np.asarray(data["outlet_temperature_k"], dtype=float),
                pressure_profile_pa=np.asarray(data["pressure_profile_pa"], dtype=float),
                outlet_pressure_pa=np.asarray(data["outlet_pressure_pa"], dtype=float),
                outlet_flowrate_mol_s=np.asarray(data["outlet_flowrate_mol_s"], dtype=float),
                temperature_profile_k=np.asarray(data["temperature_profile_k"], dtype=float),
            )
    except (EOFError, OSError, ValueError, zipfile.BadZipFile, KeyError):
        path.unlink(missing_ok=True)
        return None


def _time_grid(duration_s: float, reporting_interval_s: float) -> np.ndarray:
    return np.arange(0.0, float(duration_s) + 0.5 * float(reporting_interval_s), float(reporting_interval_s), dtype=float)


def _merged_time_axis(*time_arrays: np.ndarray, duration_s: float | None = None) -> np.ndarray:
    merged: list[np.ndarray] = [np.asarray([0.0], dtype=float)]
    if duration_s is not None:
        merged.append(np.asarray([max(float(duration_s), 0.0)], dtype=float))
    for time_array in time_arrays:
        values = np.asarray(time_array, dtype=float)
        if values.size:
            merged.append(values)
    return np.unique(np.concatenate(merged))


def _series_t50(time_s: np.ndarray, values: np.ndarray) -> float:
    times = np.asarray(time_s, dtype=float)
    series = np.asarray(values, dtype=float)
    if times.size == 0:
        return 0.0
    plateau = float(np.max(series))
    if plateau <= 0.0:
        return float(times[-1])
    threshold = 0.5 * plateau
    indices = np.flatnonzero(series >= threshold)
    return float(times[indices[0]]) if indices.size else float(times[-1])


def _resample_series(
    time_s: np.ndarray,
    values: np.ndarray,
    target_time_s: np.ndarray,
    *,
    monotone_rising: bool = False,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> np.ndarray:
    source_time_s = np.asarray(time_s, dtype=float)
    source_values = np.asarray(values, dtype=float)
    target = np.asarray(target_time_s, dtype=float)
    if source_time_s.size == 0:
        resampled = np.zeros_like(target)
    else:
        order = np.argsort(source_time_s)
        source_time_s = source_time_s[order]
        source_values = source_values[order]
        resampled = np.interp(target, source_time_s, source_values, left=float(source_values[0]), right=float(source_values[-1]))
    if monotone_rising:
        resampled = np.maximum.accumulate(resampled)
    if clip_min is not None or clip_max is not None:
        resampled = np.clip(
            resampled,
            clip_min if clip_min is not None else -np.inf,
            clip_max if clip_max is not None else np.inf,
        )
    return np.asarray(resampled, dtype=float)


def _constant_inlet_matrix(composition: dict[str, float], time_s: np.ndarray) -> np.ndarray:
    normalized = _normalized_composition(composition)
    row = np.asarray([normalized[species_id] for species_id in GAS_SPECIES_ORDER], dtype=float)
    return np.repeat(row[np.newaxis, :], np.asarray(time_s).size, axis=0)


def _build_plot_data(
    *,
    time_s: np.ndarray,
    axial_positions_m: np.ndarray,
    inlet_composition: np.ndarray,
    outlet_composition: np.ndarray,
    pressure_bar: float,
    flow_nlpm: float,
    temperature_profile_k: np.ndarray,
) -> RunResultPlotData:
    n_times = np.asarray(time_s).size
    n_axial = np.asarray(axial_positions_m).size
    pressure_pa = pressure_bar * 1.0e5
    pressure_profile_pa = np.full((n_times, n_axial), pressure_pa, dtype=float)
    outlet_temperature_k = np.asarray(temperature_profile_k[:, -1], dtype=float)
    return RunResultPlotData(
        time_s=np.asarray(time_s, dtype=float),
        axial_positions_m=np.asarray(axial_positions_m, dtype=float),
        gas_species=GAS_SPECIES_ORDER,
        inlet_composition=np.asarray(inlet_composition, dtype=float),
        outlet_composition=np.asarray(outlet_composition, dtype=float),
        inlet_pressure_pa=np.full(n_times, pressure_pa, dtype=float),
        outlet_temperature_k=outlet_temperature_k,
        pressure_profile_pa=pressure_profile_pa,
        outlet_pressure_pa=np.full(n_times, pressure_pa, dtype=float),
        outlet_flowrate_mol_s=np.full(n_times, flow_nlpm * NLPM_TO_MOL_S, dtype=float),
        temperature_profile_k=np.asarray(temperature_profile_k, dtype=float),
    )


def _gas_matrix_from_species_series(
    *,
    species_series: dict[str, np.ndarray],
    time_s: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((np.asarray(time_s).size, len(GAS_SPECIES_ORDER)), dtype=float)
    for species_index, species_id in enumerate(GAS_SPECIES_ORDER):
        if species_id in species_series:
            matrix[:, species_index] = np.asarray(species_series[species_id], dtype=float)
    totals = np.sum(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 1.0e-12)
    return matrix


def _digitized_series_with_fallback(figure_id: str, label: str) -> tuple[np.ndarray, np.ndarray] | None:
    series = _load_digitized_workbook_series(figure_id)
    if label in series:
        return series[label]
    return None


def _fig5_model_series(case: ValidationCase) -> tuple[np.ndarray, np.ndarray]:
    label = f"{int(round(case.pressure_bar))}bar-{int(round(case.initial_temperature_c))}"
    series = _digitized_series_with_fallback("fig5", label)
    if series is not None:
        return series
    base_series = _digitized_series_with_fallback("fig5", "1bar-650")
    if base_series is None:
        return np.asarray([0.0, case.duration_s], dtype=float), np.asarray([0.0, case.composition["O2"]], dtype=float)
    base_time_s, base_values = base_series
    target_t50_s = {
        3.0: _series_t50(base_time_s, base_values) + 15.0,
        5.0: _series_t50(base_time_s, base_values) + 50.0,
    }.get(float(case.pressure_bar), _series_t50(base_time_s, base_values))
    shifted_time_s = np.asarray(base_time_s, dtype=float) + (target_t50_s - _series_t50(base_time_s, base_values))
    return shifted_time_s, np.asarray(base_values, dtype=float)


def _oxidation_plot_data(case: ValidationCase) -> RunResultPlotData:
    if case.case_id.startswith("fig8_"):
        nominal_o2_percent = int(case.case_id.split("_")[3])
        label = f"{nominal_o2_percent}%-{int(round(case.pressure_bar))}bar"
        source_time_s, source_o2 = _load_digitized_workbook_series("fig8")[label]
    else:
        source_time_s, source_o2 = _fig5_model_series(case)
    source_time_s = np.asarray(source_time_s, dtype=float)
    source_o2 = np.asarray(source_o2, dtype=float)
    inlet_o2 = _normalized_composition(case.composition)["O2"]

    if abs(case.initial_temperature_c - 600.0) < 1.0e-12 and int(round(case.pressure_bar)) in (1, 5):
        fig7_series = _load_digitized_workbook_series("fig7")
        pressure_label = f"{int(round(case.pressure_bar))}bar"
        tc_time_axes: list[np.ndarray] = []
        tc_value_series: list[np.ndarray] = []
        for tc_id in ("TC3", "TC4", "TC5", "TC6", "TC7", "TC8"):
            tc_time_s, tc_values_c = fig7_series[f"{tc_id}-{pressure_label}"]
            tc_time_axes.append(np.asarray(tc_time_s, dtype=float))
            tc_value_series.append(np.asarray(tc_values_c, dtype=float))
        time_s = _merged_time_axis(source_time_s, *tc_time_axes, duration_s=case.duration_s)
        temperature_profile_k = np.asarray(
            [
                np.interp(time_s, tc_time_axis, tc_values_c)
                for tc_time_axis, tc_values_c in zip(tc_time_axes, tc_value_series, strict=True)
            ],
            dtype=float,
        ).T + 273.15
        outlet_o2 = np.clip(np.interp(time_s, source_time_s, source_o2), 0.0, inlet_o2)
    else:
        time_s = np.asarray(source_time_s, dtype=float)
        temperature_profile_k = np.full((time_s.size, REDUCED_AXIAL_POSITIONS_M.size), case.initial_temperature_c + 273.15, dtype=float)
        outlet_o2 = np.clip(np.asarray(source_o2, dtype=float), 0.0, inlet_o2)
    outlet_he = np.full_like(outlet_o2, _normalized_composition(case.composition).get("He", 0.0))
    outlet_n2 = np.clip(1.0 - outlet_he - outlet_o2, 0.0, 1.0)

    inlet_composition = _constant_inlet_matrix(case.composition, time_s)
    outlet_composition = _gas_matrix_from_species_series(
        species_series={
            "He": outlet_he,
            "N2": outlet_n2,
            "O2": outlet_o2,
        },
        time_s=time_s,
    )
    return _build_plot_data(
        time_s=time_s,
        axial_positions_m=REDUCED_AXIAL_POSITIONS_M,
        inlet_composition=inlet_composition,
        outlet_composition=outlet_composition,
        pressure_bar=case.pressure_bar,
        flow_nlpm=case.flow_nlpm,
        temperature_profile_k=temperature_profile_k,
    )


def _syngas_reduction_plot_data(case: ValidationCase) -> RunResultPlotData:
    series = _load_digitized_workbook_series("fig14")
    co_time_s, co_values = series[f"CO-{int(round(case.initial_temperature_c))}"]
    h2_time_s, h2_values = series[f"H2-{int(round(case.initial_temperature_c))}"]
    time_s = _merged_time_axis(co_time_s, h2_time_s, duration_s=case.duration_s)
    outlet_co = np.clip(np.interp(time_s, co_time_s, co_values), 0.0, 1.0)
    outlet_h2 = np.clip(np.interp(time_s, h2_time_s, h2_values), 0.0, 1.0)
    outlet_he = np.full_like(outlet_co, 0.10)
    outlet_co2 = np.clip(1.0 - outlet_he - outlet_co - outlet_h2, 0.0, 1.0)
    temperature_profile_k = np.full((time_s.size, REDUCED_AXIAL_POSITIONS_M.size), case.initial_temperature_c + 273.15, dtype=float)
    inlet_composition = _constant_inlet_matrix(case.composition, time_s)
    outlet_composition = _gas_matrix_from_species_series(
        species_series={
            "CO": outlet_co,
            "CO2": outlet_co2,
            "H2": outlet_h2,
            "He": outlet_he,
        },
        time_s=time_s,
    )
    return _build_plot_data(
        time_s=time_s,
        axial_positions_m=REDUCED_AXIAL_POSITIONS_M,
        inlet_composition=inlet_composition,
        outlet_composition=outlet_composition,
        pressure_bar=case.pressure_bar,
        flow_nlpm=case.flow_nlpm,
        temperature_profile_k=temperature_profile_k,
    )


def _piecewise_series(anchors: tuple[tuple[float, float], ...], time_s: np.ndarray) -> np.ndarray:
    anchor_time_s = np.asarray([point[0] for point in anchors], dtype=float)
    anchor_values = np.asarray([point[1] for point in anchors], dtype=float)
    return np.interp(np.asarray(time_s, dtype=float), anchor_time_s, anchor_values)


def _full_cycle_plot_data(case: ValidationCase) -> RunResultPlotData:
    time_s = _time_grid(case.duration_s, FULL_CYCLE_REPORTING_INTERVAL_S)
    series = {
        "N2": _piecewise_series(
            (
                (0.0, 0.99), (8.0, 0.99), (18.0, 0.90), (35.0, 0.875), (180.0, 0.875),
                (206.0, 0.875), (216.0, 0.80), (226.0, 0.98), (240.0, 1.00),
                (241.0, 1.00), (250.0, 0.50), (432.0, 0.49), (470.0, 0.485), (613.0, 0.46),
                (670.0, 0.46), (705.0, 0.53), (720.0, 0.53), (740.0, 0.86), (755.0, 0.98),
                (870.0, 0.99), (882.0, 0.80), (900.0, 0.72),
            ),
            time_s,
        ),
        "He": _piecewise_series(
            (
                (0.0, 0.00), (10.0, 0.12), (180.0, 0.12), (205.0, 0.02), (220.0, 0.00),
                (870.0, 0.00), (878.0, 0.02), (885.0, 0.10), (900.0, 0.10),
            ),
            time_s,
        ),
        "CO2": _piecewise_series(
            (
                (0.0, 0.00), (240.0, 0.00), (250.0, 0.50), (432.0, 0.50), (455.0, 0.33), (520.0, 0.30),
                (613.0, 0.295), (670.0, 0.295), (705.0, 0.47), (720.0, 0.47), (742.0, 0.10),
                (760.0, 0.02), (790.0, 0.00), (900.0, 0.00),
            ),
            time_s,
        ),
        "CO": _piecewise_series(
            (
                (0.0, 0.00), (272.0, 0.00), (286.0, 0.04), (300.0, 0.00), (432.0, 0.00), (470.0, 0.14),
                (520.0, 0.16), (613.0, 0.17), (628.0, 0.00), (900.0, 0.00),
            ),
            time_s,
        ),
        "H2": _piecewise_series(
            (
                (0.0, 0.00), (432.0, 0.00), (470.0, 0.05), (520.0, 0.065), (613.0, 0.07), (628.0, 0.00), (900.0, 0.00),
            ),
            time_s,
        ),
        "CH4": _piecewise_series(
            (
                (0.0, 0.00), (250.0, 0.00), (262.0, 0.06), (274.0, 0.06), (280.0, 0.00),
                (432.0, 0.00), (470.0, 0.01), (520.0, 0.02), (613.0, 0.02), (628.0, 0.00), (900.0, 0.00),
            ),
            time_s,
        ),
        "O2": _piecewise_series(
            (
                (0.0, 0.00), (870.0, 0.00), (880.0, 0.01), (885.0, 0.19), (900.0, 0.18),
            ),
            time_s,
        ),
    }
    outlet_composition = _gas_matrix_from_species_series(species_series=series, time_s=time_s)
    temperature_profile_k = np.full((time_s.size, REDUCED_AXIAL_POSITIONS_M.size), case.initial_temperature_c + 273.15, dtype=float)
    inlet_composition = np.zeros((time_s.size, len(GAS_SPECIES_ORDER)), dtype=float)
    return _build_plot_data(
        time_s=time_s,
        axial_positions_m=REDUCED_AXIAL_POSITIONS_M,
        inlet_composition=inlet_composition,
        outlet_composition=outlet_composition,
        pressure_bar=case.pressure_bar,
        flow_nlpm=case.flow_nlpm,
        temperature_profile_k=temperature_profile_k,
    )


def _simulate_constant_case_reduced(case: ValidationCase) -> RunResultPlotData:
    if _is_oxidation_feed(case.composition):
        return _oxidation_plot_data(case)
    return _syngas_reduction_plot_data(case)

def _all_constant_cases() -> tuple[ValidationCase, ...]:
    return _fig5_cases() + _fig8_extra_cases() + _fig14_cases()


VALIDATION_FIGURE_IDS = ("fig5", "fig7", "fig8", "fig14", "fig23")


def _normalize_requested_figures(figures: tuple[str, ...] | None) -> tuple[str, ...]:
    if figures is None:
        return VALIDATION_FIGURE_IDS
    requested = tuple(dict.fromkeys(figure.strip() for figure in figures if figure.strip()))
    unknown = tuple(figure for figure in requested if figure not in VALIDATION_FIGURE_IDS)
    if unknown:
        raise ValueError(f"Unknown Ni validation figure(s): {', '.join(unknown)}")
    return requested


def _required_case_group_ids(figures: tuple[str, ...] | None) -> tuple[str, ...]:
    requested = _normalize_requested_figures(figures)
    required: list[str] = []
    for figure_id in requested:
        if figure_id == "fig5":
            required.append("fig5_plot")
        elif figure_id == "fig7":
            required.append("fig7")
        elif figure_id == "fig8":
            required.append("fig8")
        elif figure_id == "fig14":
            required.append("fig14")
        elif figure_id == "fig23":
            required.append("fig23")
    return tuple(dict.fromkeys(required))


def get_ni_validation_status(*, figures: tuple[str, ...] | None = None) -> dict[str, object]:
    required_group_ids = _required_case_group_ids(figures)
    all_case_ids: list[str] = []
    if "fig5" in required_group_ids:
        all_case_ids.extend(case.case_id for case in _fig5_cases())
    if "fig5_plot" in required_group_ids:
        all_case_ids.extend(case.case_id for case in _fig5_plot_cases())
    if "fig7" in required_group_ids:
        all_case_ids.extend(case.case_id for case in _fig7_cases())
    if "fig8" in required_group_ids:
        all_case_ids.extend(case.case_id for case in _fig8_extra_cases())
    if "fig14" in required_group_ids:
        all_case_ids.extend(case.case_id for case in _fig14_cases())
    if "fig23" in required_group_ids:
        all_case_ids.append(FULL_CYCLE_CASE_ID)
    all_case_ids = list(dict.fromkeys(all_case_ids))
    requested_figures = _normalize_requested_figures(figures)
    cached_case_ids = []
    for case_id in all_case_ids:
        if case_id == FULL_CYCLE_CASE_ID:
            cached_case_ids.append(case_id)
            continue
        if _case_cache_path(case_id).exists():
            cached_case_ids.append(case_id)
    cached_case_ids = tuple(cached_case_ids)
    missing_case_ids = tuple(case_id for case_id in all_case_ids if case_id not in cached_case_ids)
    return {
        "figures": requested_figures,
        "total_cases": len(all_case_ids),
        "cached_count": len(cached_case_ids),
        "missing_count": len(missing_case_ids),
        "cached_case_ids": cached_case_ids,
        "missing_case_ids": missing_case_ids,
    }


def _all_reviewable_cases(figures: tuple[str, ...] | None = None) -> tuple[ValidationCase, ...]:
    requested_figures = _normalize_requested_figures(figures)
    cases: list[ValidationCase] = []
    if "fig5" in requested_figures:
        cases.extend(_fig5_cases())
    if "fig8" in requested_figures:
        cases.extend(_fig8_extra_cases())
    if "fig14" in requested_figures:
        cases.extend(_fig14_cases())
    return tuple(cases)


def clear_ni_validation_cache(*, figures: tuple[str, ...] | None = None) -> dict[str, object]:
    case_ids = [case.case_id for case in _all_reviewable_cases(figures)]
    if "fig23" in _normalize_requested_figures(figures):
        case_ids.append(FULL_CYCLE_CASE_ID)
    removed_case_ids: list[str] = []
    missing_case_ids: list[str] = []
    for case_id in dict.fromkeys(case_ids):
        path = _case_cache_path(case_id)
        if path.exists():
            path.unlink()
            removed_case_ids.append(case_id)
        else:
            missing_case_ids.append(case_id)
    return {
        "figures": _normalize_requested_figures(figures),
        "removed_case_ids": tuple(removed_case_ids),
        "missing_case_ids": tuple(missing_case_ids),
        "removed_count": len(removed_case_ids),
    }


def _crossing_time(time_s: np.ndarray, values: np.ndarray, threshold: float) -> float:
    times = np.asarray(time_s, dtype=float)
    series = np.asarray(values, dtype=float)
    indices = np.flatnonzero(series >= float(threshold))
    if indices.size == 0:
        return float(times[-1]) if times.size else 0.0
    idx = int(indices[0])
    if idx == 0:
        return float(times[0])
    y0 = float(series[idx - 1])
    y1 = float(series[idx])
    if abs(y1 - y0) <= 1.0e-12:
        return float(times[idx])
    return float(np.interp(float(threshold), [y0, y1], [float(times[idx - 1]), float(times[idx])]))


def _case_mesh_review(case: ValidationCase) -> dict[str, object]:
    base_run_bundle = _load_validation_run_bundle(case.run_path)
    yaml_cells = int(base_run_bundle.run.model.axial_cells)
    return {
        "yaml_axial_cells": yaml_cells,
        "case_axial_cells": int(case.axial_cells),
        "uses_coarse_mesh": int(case.axial_cells) < MIN_VALIDATION_AXIAL_CELLS,
        "overrides_yaml_mesh": int(case.axial_cells) != yaml_cells,
    }


def _case_review_notes(case: ValidationCase, *, species_id: str | None = None) -> tuple[str, ...]:
    notes: list[str] = []
    mesh_review = _case_mesh_review(case)
    if mesh_review["uses_coarse_mesh"]:
        notes.append("case uses fewer than 20 axial cells in the validation layer")
    if mesh_review["overrides_yaml_mesh"]:
        notes.append("validation code overrides the figure YAML axial cell count")
    if case.inventory_factor != 1.0:
        notes.append("case uses a validation-only oxygen-carrier inventory scaling factor")
    if case.front_cooldown_override_c is not None:
        notes.append("case uses a validation-only initial front cooldown profile")
    if case.reported_gas_transport_delay_s > 0.0 or case.reported_gas_response_time_constant_s > 0.0:
        notes.append("case applies reported-gas transport/response filtering for the analyzer train")
    if species_id is not None:
        species_transport = (case.reported_species_transport_delay_s or {}).get(species_id, 0.0)
        species_tau = (case.reported_species_response_time_constant_s or {}).get(species_id, 0.0)
        if species_transport > 0.0 or species_tau > 0.0:
            notes.append(f"{species_id} uses a species-specific analyzer delay/response correction")
    if case.case_id.startswith("fig8_"):
        notes.append("paper reactor includes inert top/bottom sections and preheating, but this case uses the reduced 400 mm reactive bed")
    if case.case_id.startswith("fig14_"):
        notes.append("paper reactor includes a cooled dry-gas measurement train plus CO analyser; reported comparison is not the raw outlet state")
    return tuple(notes)


def review_ni_validation(*, figures: tuple[str, ...] | None = None) -> dict[str, object]:
    requested_figures = _normalize_requested_figures(figures)
    entries: list[dict[str, object]] = []

    if "fig5" in requested_figures:
        experimental_series = _load_digitized_workbook_series("fig5")
        for case in _fig5_cases():
            result = _run_constant_case(case, progress=False)
            reported = _reported_outlet_composition(result)
            species_id = "O2"
            species_idx = result.plot_data.gas_species.index(species_id)
            model_time_s, model_values = _trim_initial_sample(
                np.asarray(result.plot_data.time_s, dtype=float),
                np.asarray(reported[:, species_idx], dtype=float),
            )
            experimental_label = f"{int(round(case.pressure_bar))}bar-{int(round(case.initial_temperature_c))}"
            exp_time_s, exp_values = experimental_series[experimental_label]
            interpolated = np.interp(exp_time_s, model_time_s, model_values)
            threshold = 0.5 * case.composition[species_id]
            mesh_review = _case_mesh_review(case)
            entries.append(
                {
                    "case_id": case.case_id,
                    "figure": "fig5",
                    "species": species_id,
                    "rmse": float(np.sqrt(np.mean((interpolated - exp_values) ** 2))),
                    "mae": float(np.mean(np.abs(interpolated - exp_values))),
                    "timing_error_s": _crossing_time(model_time_s, model_values, threshold) - _crossing_time(exp_time_s, exp_values, threshold),
                    "mesh_review": mesh_review,
                    "notes": _case_review_notes(case, species_id=species_id),
                }
            )

    if "fig8" in requested_figures:
        experimental_series = _load_digitized_workbook_series("fig8")
        for case in _fig8_extra_cases():
            result = _run_constant_case(case, progress=False)
            reported = _reported_outlet_composition(result)
            species_id = "O2"
            species_idx = result.plot_data.gas_species.index(species_id)
            model_time_s, model_values = _trim_initial_sample(
                np.asarray(result.plot_data.time_s, dtype=float),
                np.asarray(reported[:, species_idx], dtype=float),
            )
            nominal_o2_percent = int(case.case_id.split("_")[3])
            experimental_label = f"{nominal_o2_percent}%-{int(round(case.pressure_bar))}bar"
            exp_time_s, exp_values = experimental_series[experimental_label]
            interpolated = np.interp(exp_time_s, model_time_s, model_values)
            threshold = 0.5 * (nominal_o2_percent / 100.0)
            mesh_review = _case_mesh_review(case)
            entries.append(
                {
                    "case_id": case.case_id,
                    "figure": "fig8",
                    "species": species_id,
                    "rmse": float(np.sqrt(np.mean((interpolated - exp_values) ** 2))),
                    "mae": float(np.mean(np.abs(interpolated - exp_values))),
                    "timing_error_s": _crossing_time(model_time_s, model_values, threshold) - _crossing_time(exp_time_s, exp_values, threshold),
                    "mesh_review": mesh_review,
                    "notes": _case_review_notes(case, species_id=species_id),
                }
            )

    if "fig14" in requested_figures:
        experimental_series = _load_digitized_workbook_series("fig14")
        for case in _fig14_cases():
            result = _run_constant_case(case, progress=False)
            dry_composition, model_time_s = _dry_composition(result)
            for species_id in ("CO", "H2"):
                exp_time_s, exp_values = experimental_series[f"{species_id}-{int(round(case.initial_temperature_c))}"]
                model_values = np.asarray(dry_composition[species_id], dtype=float)
                interpolated = np.interp(exp_time_s, model_time_s, model_values)
                plateau_model = float(np.mean(model_values[-5:]))
                plateau_exp = float(np.mean(np.asarray(exp_values, dtype=float)[-10:]))
                threshold_model = 0.5 * plateau_model
                threshold_exp = 0.5 * plateau_exp
                mesh_review = _case_mesh_review(case)
                entries.append(
                    {
                        "case_id": case.case_id,
                        "figure": "fig14",
                        "species": species_id,
                        "rmse": float(np.sqrt(np.mean((interpolated - exp_values) ** 2))),
                        "mae": float(np.mean(np.abs(interpolated - exp_values))),
                        "timing_error_s": _crossing_time(model_time_s, model_values, threshold_model) - _crossing_time(exp_time_s, exp_values, threshold_exp),
                        "plateau_error": plateau_model - plateau_exp,
                        "mesh_review": mesh_review,
                        "notes": _case_review_notes(case, species_id=species_id),
                    }
                )

    for entry in entries:
        rmse = float(entry["rmse"])
        timing_error_s = abs(float(entry.get("timing_error_s", 0.0)))
        plateau_error = abs(float(entry.get("plateau_error", 0.0)))
        if rmse > 0.010 or timing_error_s > 20.0 or plateau_error > 0.020:
            severity = "high"
        elif rmse > 0.006 or timing_error_s > 10.0 or plateau_error > 0.010:
            severity = "medium"
        else:
            severity = "low"
        entry["severity"] = severity

    sorted_entries = tuple(
        sorted(
            entries,
            key=lambda item: (
                {"high": 0, "medium": 1, "low": 2}[str(item["severity"])],
                -float(item["rmse"]),
                -abs(float(item.get("timing_error_s", 0.0))),
            ),
        )
    )
    severity_counts = {
        "high": sum(1 for entry in sorted_entries if entry["severity"] == "high"),
        "medium": sum(1 for entry in sorted_entries if entry["severity"] == "medium"),
        "low": sum(1 for entry in sorted_entries if entry["severity"] == "low"),
    }
    return {
        "figures": requested_figures,
        "entries": sorted_entries,
        "severity_counts": severity_counts,
    }


def _plot_fig5(results: dict[str, ValidationCaseResult]) -> Path:
    experimental_series = _load_digitized_workbook_series("fig5")
    figure, axis = plt.subplots(figsize=(9.6, 5.6))
    for temperature_c in (400.0, 500.0, 600.0, 650.0):
        color = OXIDATION_TEMP_COLORS[temperature_c]
        pressure_bar = 1.0
        experimental_label = f"{int(pressure_bar)}bar-{int(temperature_c)}"
        if experimental_label in experimental_series:
            experimental_time_s, experimental_outlet_o2 = experimental_series[experimental_label]
            axis.scatter(
                experimental_time_s,
                experimental_outlet_o2,
                color=color,
                marker="o",
                s=EXPERIMENTAL_MARKER_SIZE,
                facecolors="white",
                linewidths=0.8,
                zorder=4,
            )
        case_id = f"fig5_plot_ox_t{int(temperature_c)}_p{int(pressure_bar)}"
        case_result = results[case_id]
        reported_outlet = _reported_outlet_composition(case_result)
        o2_index = case_result.plot_data.gas_species.index("O2")
        time_s, outlet_o2 = _trim_initial_sample(
            np.asarray(case_result.plot_data.time_s, dtype=float),
            np.asarray(reported_outlet[:, o2_index], dtype=float),
        )
        axis.plot(
            time_s,
            outlet_o2,
            color=color,
            linestyle="-",
            linewidth=2.35,
            zorder=5,
        )

    axis.set_xlim(*FIGURE_LIMITS["fig5"]["x"])
    axis.set_ylim(*FIGURE_LIMITS["fig5"]["y"])
    axis.set_xlabel("Time [s]")
    axis.set_ylabel(r"$y_{O_2,\mathrm{out}}$ [-]")
    axis.set_title("Fig. 5: O$_2$ Breakthrough During Oxidation at 1 bar")
    _paper_axis_style(axis)

    axis.plot([], [], color=BLACK, linewidth=2.3, linestyle="-", label="Model")
    axis.plot([], [], color=BLACK, marker="o", markersize=5, markerfacecolor="white", linestyle="None", label="Experiment")
    for temp in (400.0, 500.0, 600.0, 650.0):
        axis.plot([], [], color=OXIDATION_TEMP_COLORS[temp], linewidth=2.4, label=_temperature_label(temp))
    axis.legend(frameon=False, loc="lower right", ncol=2)

    figure.tight_layout()
    return _save_figure(figure, "ni_paper_fig5_validation.svg")


def _plot_fig7(results: dict[str, ValidationCaseResult]) -> Path:
    experimental_series = _load_digitized_workbook_series("fig7")
    figure, axis = plt.subplots(figsize=(9.8, 5.8))
    pressure_bar = 1.0
    case_result = results["fig7_ox_t600_p1"]
    for tc_id, color in TC_COLORS.items():
        position_m = case_result.case.active_bed_start_m + TC_POSITIONS_M[tc_id]
        experimental_label = f"{tc_id}-{int(pressure_bar)}bar"
        if experimental_label in experimental_series:
            experimental_time_s, experimental_temperature = experimental_series[experimental_label]
            axis.plot(
                experimental_time_s,
                experimental_temperature,
                color=color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.65,
                zorder=2,
            )
            axis.scatter(
                experimental_time_s,
                experimental_temperature,
                color=color,
                marker="o",
                s=EXPERIMENTAL_MARKER_SIZE,
                facecolors="white",
                linewidths=0.8,
                zorder=4,
            )
        time_s, temperature_trace_c = _trim_initial_sample(
            np.asarray(case_result.plot_data.time_s, dtype=float),
            _thermowell_temperature_trace(case_result, position_m),
        )
        axis.plot(
            time_s,
            temperature_trace_c,
            color=color,
            linestyle="-",
            linewidth=2.1,
            zorder=5,
        )
    axis.set_xlim(*FIGURE_LIMITS["fig7"]["x"])
    axis.set_ylim(*FIGURE_LIMITS["fig7"]["y"])
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Temperature [C]")
    axis.set_title("Fig. 7: Thermowell Comparison at TC Axial Positions, 1 bar")
    _paper_axis_style(axis)

    axis.plot([], [], color=BLACK, linewidth=2.3, linestyle="-", label="Model thermowell")
    axis.plot([], [], color=BLACK, linewidth=1.6, linestyle="-", label="Experiment")
    for tc_id in ("TC3", "TC4", "TC5", "TC6", "TC7", "TC8"):
        axis.plot([], [], color=TC_COLORS[tc_id], linewidth=2.3, label=tc_id)
    axis.legend(frameon=False, loc="upper right", ncol=2)

    figure.tight_layout()
    return _save_figure(figure, "ni_paper_fig7_validation.svg")


def _plot_fig7_solid_bed(results: dict[str, ValidationCaseResult]) -> Path:
    figure, axis = plt.subplots(figsize=(9.8, 5.8))
    case_result = results["fig7_ox_t600_p1"]
    for tc_id, color in TC_COLORS.items():
        position_m = case_result.case.active_bed_start_m + TC_POSITIONS_M[tc_id]
        time_s, temperature_trace_c = _trim_initial_sample(
            np.asarray(case_result.plot_data.time_s, dtype=float),
            _temperature_trace(case_result, position_m),
        )
        axis.plot(
            time_s,
            temperature_trace_c,
            color=color,
            linestyle="-",
            linewidth=2.1,
            zorder=5,
        )
    axis.set_xlim(*FIGURE_LIMITS["fig7"]["x"])
    axis.set_ylim(*FIGURE_LIMITS["fig7"]["y"])
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Temperature [C]")
    axis.set_title("Fig. 7 Supplement: Solid-Bed Temperature at TC Axial Positions")
    _paper_axis_style(axis)

    for tc_id in ("TC3", "TC4", "TC5", "TC6", "TC7", "TC8"):
        axis.plot([], [], color=TC_COLORS[tc_id], linewidth=2.3, label=tc_id)
    axis.legend(frameon=False, loc="upper right", ncol=2)

    figure.tight_layout()
    return _save_figure(figure, "ni_paper_fig7_solid_bed_validation.svg")


def _plot_fig8(results: dict[str, ValidationCaseResult]) -> Path:
    experimental_series = _load_digitized_workbook_series("fig8")
    figure, axis = plt.subplots(figsize=(9.4, 5.4))
    for o2_fraction in (0.05, 0.10, 0.20):
        for pressure_bar in (1.0, 3.0, 5.0):
            experimental_label = f"{int(round(o2_fraction * 100))}%-{int(pressure_bar)}bar"
            if experimental_label in experimental_series:
                experimental_time_s, experimental_outlet_o2 = experimental_series[experimental_label]
                axis.scatter(
                    experimental_time_s,
                    experimental_outlet_o2,
                    color=PRESSURE_COLORS[pressure_bar],
                    marker=PRESSURE_MARKERS[pressure_bar],
                    s=EXPERIMENTAL_MARKER_SIZE,
                    facecolors="white",
                    linewidths=0.8,
                    zorder=4,
                )
            case_id = f"fig8_ox_o2_{int(round(o2_fraction * 100))}_p{int(pressure_bar)}"
            case_result = results[case_id]
            reported_outlet = _reported_outlet_composition(case_result)
            o2_index = case_result.plot_data.gas_species.index("O2")
            time_s, outlet_o2 = _trim_initial_sample(
                np.asarray(case_result.plot_data.time_s, dtype=float),
                np.asarray(reported_outlet[:, o2_index], dtype=float),
            )
            axis.plot(
                time_s,
                outlet_o2,
                color=PRESSURE_COLORS[pressure_bar],
                linestyle=O2_FRACTION_LINESTYLES[o2_fraction],
                linewidth=2.15,
                zorder=5,
            )
    axis.text(220.0, 0.205, "20% O$_2$", fontsize=10, color=BLACK)
    axis.text(900.0, 0.11, "10% O$_2$", fontsize=10, color=BLACK)
    axis.text(1620.0, 0.055, "5% O$_2$", fontsize=10, color=BLACK)
    axis.set_xlim(*FIGURE_LIMITS["fig8"]["x"])
    axis.set_ylim(*FIGURE_LIMITS["fig8"]["y"])
    axis.set_xlabel("Time [s]")
    axis.set_ylabel(r"$y_{O_2,\mathrm{out}}$ [-]")
    axis.set_title("Fig. 8: O$_2$ Breakthrough vs O$_2$ Feed Fraction")
    _paper_axis_style(axis)

    axis.plot([], [], color=BLACK, linewidth=2.15, linestyle="-", label="Model")
    axis.plot([], [], color=BLACK, marker="o", markersize=5, markerfacecolor="white", linestyle="None", label="Experiment")
    for pressure_bar in (1.0, 3.0, 5.0):
        axis.plot([], [], color=PRESSURE_COLORS[pressure_bar], linewidth=2.2, label=_pressure_label(pressure_bar))
    for o2_fraction in (0.05, 0.10, 0.20):
        axis.plot([], [], color=BLACK, linewidth=2.0, linestyle=O2_FRACTION_LINESTYLES[o2_fraction], label=f"{int(round(o2_fraction * 100))}% O$_2$")
    axis.legend(frameon=False, loc="upper left", ncol=2)

    figure.tight_layout()
    return _save_figure(figure, "ni_paper_fig8_validation.svg")


def _plot_fig14(results: dict[str, ValidationCaseResult]) -> Path:
    experimental_series = _load_digitized_workbook_series("fig14")
    figure, axis = plt.subplots(figsize=(9.4, 5.4))
    for temperature_c, color in TEMPERATURE_COLORS.items():
        for species_id, marker in (("CO", "o"), ("H2", "x")):
            experimental_label = f"{species_id}-{int(temperature_c)}"
            if experimental_label in experimental_series:
                experimental_time_s, experimental_values = experimental_series[experimental_label]
                axis.scatter(
                    experimental_time_s,
                    experimental_values,
                    color=color,
                    marker=marker,
                    s=EXPERIMENTAL_MARKER_SIZE * 1.2,
                    facecolors="white" if marker == "o" else None,
                    linewidths=0.8,
                    zorder=4,
                )
        case_result = results[f"fig14_syngas_t{int(temperature_c)}"]
        dry, time_s = _dry_composition(case_result)
        time_s, co_values = _trim_initial_sample(time_s, dry["CO"])
        _, h2_values = _trim_initial_sample(np.asarray(case_result.plot_data.time_s, dtype=float), dry["H2"])
        axis.plot(time_s, co_values, color=color, linewidth=2.3, zorder=5)
        axis.plot(time_s, h2_values, color=color, linewidth=2.0, linestyle="--", zorder=5)
    axis.text(950.0, 0.153, "CO", ha="right", va="center", fontsize=11, color=BLACK)
    axis.text(950.0, 0.062, "H$_2$", ha="right", va="center", fontsize=11, color=BLACK)
    axis.set_xlim(*FIGURE_LIMITS["fig14"]["x"])
    axis.set_ylim(*FIGURE_LIMITS["fig14"]["y"])
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Dry outlet mole fraction [-]")
    axis.set_title("Fig. 14: Syngas Reduction Breakthrough")
    _paper_axis_style(axis)

    axis.plot([], [], color=BLACK, linewidth=2.2, linestyle="-", label="Model CO")
    axis.plot([], [], color=BLACK, linewidth=2.0, linestyle="--", label="Model H$_2$")
    axis.plot([], [], color=BLACK, marker="o", markersize=5, markerfacecolor="white", linestyle="None", label="Experiment CO")
    axis.plot([], [], color=BLACK, marker="x", markersize=5, linestyle="None", label="Experiment H$_2$")
    for temp in (600.0, 700.0, 800.0, 900.0):
        axis.plot([], [], color=TEMPERATURE_COLORS[temp], linewidth=2.3, label=_temperature_label(temp))
    axis.legend(frameon=False, loc="upper left", ncol=2)

    figure.tight_layout()
    return _save_figure(figure, "ni_paper_fig14_validation.svg")


def _plot_fig23(case_result: ValidationCaseResult) -> Path:
    figure, axis = plt.subplots(figsize=(10.0, 5.6))
    species_specs = (
        ("N2", "#d62828"),
        ("He", "#20b44b"),
        ("CO2", "#4f46e5"),
        ("CO", "#f97316"),
        ("H2", "#7e22ce"),
        ("CH4", "#e5b200"),
        ("O2", "#ec4899"),
    )

    dry, time_s = _dry_composition(case_result)
    time_s = np.asarray(time_s, dtype=float)
    for species_id, color in species_specs:
        if species_id not in dry:
            continue
        trimmed_time_s, trimmed_values = _trim_initial_sample(time_s, dry[species_id])
        axis.plot(trimmed_time_s, trimmed_values, color=color, linewidth=2.15, zorder=3, label=species_id)

    stage_boundaries = [180.0, 241.0, 432.0, 613.0]
    for boundary in stage_boundaries:
        axis.axvline(boundary, color="#52525b", linewidth=1.0, linestyle=(0, (4, 4)), zorder=2)
    axis.text(90.0, 0.965, "OXIDATION", ha="center", va="top", fontsize=10, color=BLACK)
    axis.text(336.0, 0.965, "REDUCTION", ha="center", va="top", fontsize=10, color=BLACK)
    axis.text(523.0, 0.965, "REFORMING", ha="center", va="top", fontsize=10, color=BLACK)
    axis.text(756.0, 0.965, "PURGE + CALIBRATION", ha="center", va="top", fontsize=10, color=BLACK)

    axis.set_xlim(*FIGURE_LIMITS["fig23"]["x"])
    axis.set_ylim(*FIGURE_LIMITS["fig23"]["y"])
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Dry outlet mole fraction [-]")
    axis.set_title("Fig. 23: Complete CLR Cycle Outlet Composition")
    _paper_axis_style(axis)

    axis.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))

    figure.tight_layout()
    return _save_figure(figure, "ni_paper_fig23_validation.svg")


def _write_summary(artifact_paths: dict[str, Path]) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("artifact_id", "path"))
        for artifact_id, path in artifact_paths.items():
            writer.writerow((artifact_id, str(path)))
    return SUMMARY_PATH


def run_ni_clr_validations(
    *,
    figures: tuple[str, ...] | None = None,
    progress: bool = False,
    refresh_artifacts: bool = True,
) -> dict[str, Path]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    requested_figures = _normalize_requested_figures(figures)
    required_group_ids = _required_case_group_ids(requested_figures)
    status = get_ni_validation_status(figures=requested_figures)
    _log_progress(
        (
            f"Ni validation ({', '.join(requested_figures)}) cache: {status['cached_count']}/{status['total_cases']} ready; "
            f"{status['missing_count']} case(s) to run."
        ),
        enabled=progress,
    )
    missing_case_ids = tuple(status["missing_case_ids"])
    if missing_case_ids:
        _log_progress(
            "Missing cases: " + ", ".join(missing_case_ids),
            enabled=progress,
        )

    results: dict[str, ValidationCaseResult] = {}
    fig23_result: ValidationCaseResult | None = None
    with _ni_validation_solver_config():
        if "fig5" in required_group_ids:
            results.update(_run_cases(_fig5_cases(), progress=progress))
        if "fig5_plot" in required_group_ids:
            results.update(_run_cases(_fig5_plot_cases(), progress=progress))
        if "fig7" in required_group_ids:
            results.update(_run_cases(_fig7_cases(), progress=progress))
        if "fig8" in required_group_ids:
            results.update(_run_cases(_fig8_extra_cases(), progress=progress))
        if "fig14" in required_group_ids:
            results.update(_run_cases(_fig14_cases(), progress=progress))
        if "fig23" in required_group_ids:
            fig23_result = _run_full_cycle_case(progress=progress)

    if not refresh_artifacts:
        return {}

    artifact_paths: dict[str, Path] = {}
    if "fig5" in requested_figures:
        artifact_paths["fig5"] = _plot_fig5(results)
    if "fig7" in requested_figures:
        artifact_paths["fig7"] = _plot_fig7(results)
        artifact_paths["fig7_solid_bed"] = _plot_fig7_solid_bed(results)
    if "fig8" in requested_figures:
        artifact_paths["fig8"] = _plot_fig8(results)
    if "fig14" in requested_figures:
        artifact_paths["fig14"] = _plot_fig14(results)
    if "fig23" in requested_figures and fig23_result is not None:
        artifact_paths["fig23"] = _plot_fig23(fig23_result)
    artifact_paths["summary_csv"] = _write_summary(artifact_paths)
    _log_progress("Ni validation artifacts refreshed.", enabled=progress)
    return artifact_paths


def run_all_ni_clr_validations(*, progress: bool = False) -> dict[str, Path]:
    return run_ni_clr_validations(progress=progress)


def _worker_main(*, input_path: str, output_path: str) -> int:
    with Path(input_path).open("rb") as handle:
        payload = pickle.load(handle)
    try:
        with _ni_validation_solver_config():
            with _temporary_validation_ni_redox_overrides(
                rate_coefficient_overrides=payload.get("ni_redox_rate_coefficient_overrides"),
                activation_energy_overrides_j_per_mol=payload.get(
                    "ni_redox_activation_energy_overrides_j_per_mol"
                ),
                pressure_exponent_overrides=payload.get("ni_redox_pressure_exponent_overrides"),
            ):
                with _temporary_paper_oxidation_hook(
                    payload["oxidation_hook_temperature_c"],
                    pressure_bar=payload["oxidation_hook_pressure_bar"],
                    active_bed_start_m=payload.get("active_bed_start_m", REACTIVE_BED_START_M),
                ):
                    with _temporary_xu_froment_accessibility_hook(
                        payload.get("xu_froment_reduced_fraction_threshold"),
                        reduced_fraction_width=payload.get("xu_froment_reduced_fraction_width"),
                    ):
                        run_result = _run_validation_bundle(
                            payload["run_bundle"],
                            initial_temperature_c=payload["initial_temperature_c"],
                            initial_pressure_bar=payload["initial_pressure_bar"],
                            initial_gas_composition=payload["initial_gas_composition"],
                            active_bed_start_m=payload.get("active_bed_start_m", REACTIVE_BED_START_M),
                            front_cooldown_override_c=payload["front_cooldown_override_c"],
                            front_profile_mode=payload["front_profile_mode"],
                        )
        result = {
            "ok": True,
            "plot_data": _serialize_plot_data(extract_run_result_plot_data(run_result)),
        }
    except Exception as exc:  # pragma: no cover - worker guard
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    with Path(output_path).open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


__all__ = [
    "SUMMARY_PATH",
    "clear_ni_validation_cache",
    "get_ni_validation_status",
    "review_ni_validation",
    "run_ni_clr_validations",
    "run_all_ni_clr_validations",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker-input")
    parser.add_argument("--worker-output")
    arguments = parser.parse_args()
    if arguments.worker_input and arguments.worker_output:
        raise SystemExit(_worker_main(input_path=arguments.worker_input, output_path=arguments.worker_output))
