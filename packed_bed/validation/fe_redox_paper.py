from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import matplotlib
import yaml

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..kinetics.fe_redox import (
    FE2O3_MW_KG_PER_MOL,
    FE3O4_MW_KG_PER_MOL,
    FEO_MW_KG_PER_MOL,
    FE_MW_KG_PER_MOL,
    FULLY_OXIDIZED_FE2O3_CONCENTRATION_MOL_PER_M3,
    co_reaction_constant_value,
    equilibrium_constant_co_fe3o4_to_feo_value,
    fe2o3_ch4_reduction_rate_value,
    fe2o3_co_reduction_rate_value,
    fe2o3_h2_reduction_rate_value,
    fe3o4_co_reduction_rate_value,
    fe3o4_h2_reduction_rate_value,
    feo_co_reduction_rate_value,
    feo_h2_reduction_rate_value,
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
    }
)


VALIDATION_ROOT = Path(__file__).resolve().parents[1] / "examples" / "fe_redox_validation"
ARTIFACTS_DIR = VALIDATION_ROOT / "output" / "artifacts"
SUMMARY_PATH = ARTIFACTS_DIR / "fe_redox_validation_summary.csv"

RUN_FILE_NAMES = (
    "figs2_run.yaml",
)

MAX_INTERNAL_REDUCTION_STEP_S = 0.01
SUPPLEMENTARY_SEQUENCE_MAX_INTERNAL_STEP_S = {
    "H2": 1.0e-4,
    "CO": 1.0e-2,
}
SUPPLEMENTARY_STAGE_MAX_INTERNAL_STEP_S = {
    "fe2o3_ch4": 1.0e-3,
}

BLACK = "#111111"
REDUCTION_CURVE_COLORS = {
    "x_fe2o3": "#0071bc",
    "x_fe3o4": "#d95218",
    "x_feo": "#ecb01f",
}
REDUCTION_CURVE_LABELS = {
    "x_fe2o3": r"Fe$_2$O$_3$-H$_2$+CO+CH$_4$",
    "x_fe3o4": r"Fe$_3$O$_4$-H$_2$+CO",
    "x_feo": r"FeO-H$_2$+CO",
}
REDUCTION_CURVE_KEYS = ("x_fe2o3", "x_fe3o4", "x_feo")

SUPPLEMENTARY_REDUCTION_CURVE_SPECS = {
    "fe2o3_h2": {
        "label": r"Fe$_2$O$_3$-H$_2$",
        "color": "#0071bc",
        "phase": "Fe2O3",
        "gas_species_id": "H2",
        "conversion_key": "x_fe2o3",
    },
    "fe2o3_co": {
        "label": r"Fe$_2$O$_3$-CO",
        "color": "#d95319",
        "phase": "Fe2O3",
        "gas_species_id": "CO",
        "conversion_key": "x_fe2o3",
    },
    "fe2o3_ch4": {
        "label": r"Fe$_2$O$_3$-CH$_4$",
        "color": "#edb120",
        "phase": "Fe2O3",
        "gas_species_id": "CH4",
        "conversion_key": "x_fe2o3",
    },
    "fe3o4_h2": {
        "label": r"Fe$_3$O$_4$-H$_2$",
        "color": "#7e2f8e",
        "phase": "Fe3O4",
        "gas_species_id": "H2",
        "conversion_key": "x_fe3o4",
    },
    "fe3o4_co": {
        "label": r"Fe$_3$O$_4$-CO",
        "color": "#77ac30",
        "phase": "Fe3O4",
        "gas_species_id": "CO",
        "conversion_key": "x_fe3o4",
    },
    "feo_h2": {
        "label": r"FeO-H$_2$",
        "color": "#4dbeee",
        "phase": "FeO",
        "gas_species_id": "H2",
        "conversion_key": "x_feo",
    },
    "feo_co": {
        "label": r"FeO-CO",
        "color": "#a2142f",
        "phase": "FeO",
        "gas_species_id": "CO",
        "conversion_key": "x_feo",
    },
}
SUPPLEMENTARY_REDUCTION_CURVE_IDS = tuple(SUPPLEMENTARY_REDUCTION_CURVE_SPECS)


@dataclass(frozen=True)
class ReductionPanelConfig:
    mode: str
    panel_id: str
    title: str
    temperature_k: float
    initial_equivalent_fe2o3_mol_per_kg: float
    oc_density_kg_per_m3: float
    time_horizon_s: float
    time_step_s: float
    mean_gas_concentrations_mol_per_m3: dict[str, float]
    paper_curve_csv_path: Path
    artifact_stem: str
    x_limits_s: tuple[float, float]
    y_limits: tuple[float, float]
    x_major_step_s: float
    x_minor_step_s: float
    y_major_step: float
    y_minor_step: float


@dataclass(frozen=True)
class SupplementaryReductionConfig:
    mode: str
    panel_id: str
    title: str
    temperature_k: float
    total_pressure_pa: float
    oc_density_kg_per_m3: float
    time_horizon_s: float
    time_step_s: float
    gas_mole_fractions: dict[str, float]
    paper_curve_csv_path: Path
    artifact_stem: str
    x_limits_s: tuple[float, float]
    y_limits: tuple[float, float]
    x_major_step_s: float
    x_minor_step_s: float
    y_major_step: float
    y_minor_step: float
    inset_x_limits_s: tuple[float, float]
    inset_y_limits: tuple[float, float]


PanelConfig = ReductionPanelConfig | SupplementaryReductionConfig


@dataclass(frozen=True)
class PanelSummary:
    panel_id: str
    mode: str
    artifact_path: Path
    rmse_mean: float
    initial_equivalent_fe2o3_mol_per_kg: float | None = None
    rmse_fe2o3: float | None = None
    rmse_fe3o4: float | None = None
    rmse_feo: float | None = None
    curve_rmses: dict[str, float] | None = None


def _paper_axis_style(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.tick_params(which="major", width=0.8, length=4, colors=BLACK)
    axis.tick_params(which="minor", width=0.6, length=2, colors=BLACK)
    axis.xaxis.label.set_color(BLACK)
    axis.yaxis.label.set_color(BLACK)


def _secondary_axis_style(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.tick_params(which="major", width=0.8, length=4, colors=BLACK)
    axis.tick_params(which="minor", width=0.6, length=2, colors=BLACK)
    axis.yaxis.label.set_color(BLACK)


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a top-level mapping in {path}")
    return data


def _as_float(mapping: dict, key: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _load_reduction_paper_curves(csv_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 0], {
        "x_fe2o3": data[:, 1],
        "x_fe3o4": data[:, 2],
        "x_feo": data[:, 3],
    }


def _load_supplementary_reduction_paper_curves(csv_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 0], {
        curve_id: data[:, idx + 1]
        for idx, curve_id in enumerate(SUPPLEMENTARY_REDUCTION_CURVE_IDS)
    }


def _load_panel_config(run_path: Path) -> PanelConfig:
    run_data = _read_yaml(run_path)
    references = run_data["references"]
    validation = run_data["validation"]
    base_dir = run_path.parent

    chemistry = _read_yaml((base_dir / references["chemistry_file"]).resolve())
    program = _read_yaml((base_dir / references["program_file"]).resolve())
    solids = _read_yaml((base_dir / references["solids_file"]).resolve())
    mode = str(chemistry.get("validation_mode"))

    if mode == "reduction":
        return ReductionPanelConfig(
            mode=mode,
            panel_id=str(validation["panel_id"]),
            title=str(validation["title"]),
            temperature_k=_as_float(program, "temperature_k"),
            initial_equivalent_fe2o3_mol_per_kg=_as_float(solids, "initial_equivalent_fe2o3_mol_per_kg"),
            oc_density_kg_per_m3=_as_float(solids, "oc_density_kg_per_m3"),
            time_horizon_s=_as_float(program, "time_horizon_s"),
            time_step_s=_as_float(program, "time_step_s"),
            mean_gas_concentrations_mol_per_m3={
                species_id: float(value)
                for species_id, value in program["mean_gas_concentrations_mol_per_m3"].items()
            },
            paper_curve_csv_path=(base_dir / validation["paper_curve_csv"]).resolve(),
            artifact_stem=str(validation["artifact_stem"]),
            x_limits_s=(float(validation["x_limits_s"][0]), float(validation["x_limits_s"][1])),
            y_limits=(float(validation["y_limits"][0]), float(validation["y_limits"][1])),
            x_major_step_s=_as_float(validation, "x_major_step_s"),
            x_minor_step_s=_as_float(validation, "x_minor_step_s"),
            y_major_step=_as_float(validation, "y_major_step"),
            y_minor_step=_as_float(validation, "y_minor_step"),
        )

    if mode == "supplementary_reduction":
        return SupplementaryReductionConfig(
            mode=mode,
            panel_id=str(validation["panel_id"]),
            title=str(validation["title"]),
            temperature_k=_as_float(program, "temperature_k"),
            total_pressure_pa=_as_float(program, "total_pressure_pa"),
            oc_density_kg_per_m3=_as_float(solids, "oc_density_kg_per_m3"),
            time_horizon_s=_as_float(program, "time_horizon_s"),
            time_step_s=_as_float(program, "time_step_s"),
            gas_mole_fractions={
                species_id: float(value)
                for species_id, value in program["gas_mole_fractions"].items()
            },
            paper_curve_csv_path=(base_dir / validation["paper_curve_csv"]).resolve(),
            artifact_stem=str(validation["artifact_stem"]),
            x_limits_s=(float(validation["x_limits_s"][0]), float(validation["x_limits_s"][1])),
            y_limits=(float(validation["y_limits"][0]), float(validation["y_limits"][1])),
            x_major_step_s=_as_float(validation, "x_major_step_s"),
            x_minor_step_s=_as_float(validation, "x_minor_step_s"),
            y_major_step=_as_float(validation, "y_major_step"),
            y_minor_step=_as_float(validation, "y_minor_step"),
            inset_x_limits_s=(float(validation["inset_x_limits_s"][0]), float(validation["inset_x_limits_s"][1])),
            inset_y_limits=(float(validation["inset_y_limits"][0]), float(validation["inset_y_limits"][1])),
        )

    raise ValueError(f"Unsupported Fe validation mode {mode!r} in {run_path}")


def _load_panel_configs() -> tuple[PanelConfig, ...]:
    return tuple(_load_panel_config(VALIDATION_ROOT / file_name) for file_name in RUN_FILE_NAMES)


def _initial_state_from_equivalent_fe2o3(
    *,
    equivalent_fe2o3_mol_per_kg: float,
    oc_density_kg_per_m3: float,
) -> np.ndarray:
    c_full_mol_per_kg = FULLY_OXIDIZED_FE2O3_CONCENTRATION_MOL_PER_M3 / oc_density_kg_per_m3

    if equivalent_fe2o3_mol_per_kg >= (8.0 / 9.0) * c_full_mol_per_kg:
        c_fe2o3_mol_per_kg = 9.0 * equivalent_fe2o3_mol_per_kg - 8.0 * c_full_mol_per_kg
        c_fe3o4_mol_per_kg = 6.0 * (c_full_mol_per_kg - equivalent_fe2o3_mol_per_kg)
        c_feo_mol_per_kg = 0.0
        c_fe_mol_per_kg = 0.0
    elif equivalent_fe2o3_mol_per_kg >= (2.0 / 3.0) * c_full_mol_per_kg:
        c_fe2o3_mol_per_kg = 0.0
        c_fe3o4_mol_per_kg = 3.0 * equivalent_fe2o3_mol_per_kg - 2.0 * c_full_mol_per_kg
        c_feo_mol_per_kg = 8.0 * c_full_mol_per_kg - 9.0 * equivalent_fe2o3_mol_per_kg
        c_fe_mol_per_kg = 0.0
    else:
        c_fe2o3_mol_per_kg = 0.0
        c_fe3o4_mol_per_kg = 0.0
        c_feo_mol_per_kg = 3.0 * equivalent_fe2o3_mol_per_kg
        c_fe_mol_per_kg = 2.0 * c_full_mol_per_kg - 3.0 * equivalent_fe2o3_mol_per_kg

    return oc_density_kg_per_m3 * np.array(
        [c_fe2o3_mol_per_kg, c_fe3o4_mol_per_kg, c_feo_mol_per_kg, c_fe_mol_per_kg],
        dtype=float,
    )


def _state_conversions(state_vector: np.ndarray) -> np.ndarray:
    c_fe2o3, c_fe3o4, c_feo, _c_fe = state_vector
    x_fe2o3 = 1.0 - c_fe2o3 / FULLY_OXIDIZED_FE2O3_CONCENTRATION_MOL_PER_M3
    x_fe3o4 = x_fe2o3 - 1.5 * c_fe3o4 / FULLY_OXIDIZED_FE2O3_CONCENTRATION_MOL_PER_M3
    x_feo = x_fe3o4 - 0.5 * c_feo / FULLY_OXIDIZED_FE2O3_CONCENTRATION_MOL_PER_M3
    return np.array([x_fe2o3, x_fe3o4, x_feo], dtype=float)


def _oc_mass_density(state_vector: np.ndarray) -> float:
    return float(
        state_vector[0] * FE2O3_MW_KG_PER_MOL
        + state_vector[1] * FE3O4_MW_KG_PER_MOL
        + state_vector[2] * FEO_MW_KG_PER_MOL
        + state_vector[3] * FE_MW_KG_PER_MOL
    )


def _reduction_rhs(state_vector: np.ndarray, config: ReductionPanelConfig) -> np.ndarray:
    c_fe2o3, c_fe3o4, c_feo, _c_fe = state_vector
    x_fe2o3, x_fe3o4, x_feo = _state_conversions(state_vector)
    gas = config.mean_gas_concentrations_mol_per_m3
    # The CO and CH4 expressions in the paper are written per kg_OC, so the
    # conversion to volumetric source terms must use the full oxygen-carrier
    # density, not just the reactive Fe-phase mass remaining in the particle.
    oc_density = config.oc_density_kg_per_m3

    r1 = 0.0
    if c_fe2o3 > 0.0:
        r1 += fe2o3_h2_reduction_rate_value(
            temperature_k=config.temperature_k,
            c_fe2o3_mol_per_m3=c_fe2o3,
            c_h2_mol_per_m3=gas.get("H2", 0.0),
            c_h2o_mol_per_m3=gas.get("H2O", 0.0),
            x_fe2o3=x_fe2o3,
        )
        r1 += fe2o3_co_reduction_rate_value(
            temperature_k=config.temperature_k,
            oc_mass_density_kg_per_m3=oc_density,
            c_co_mol_per_m3=gas.get("CO", 0.0),
            c_co2_mol_per_m3=gas.get("CO2", 0.0),
            x_fe2o3=x_fe2o3,
        )
        r1 += fe2o3_ch4_reduction_rate_value(
            temperature_k=config.temperature_k,
            oc_mass_density_kg_per_m3=oc_density,
            c_ch4_mol_per_m3=gas.get("CH4", 0.0),
            x_fe2o3=x_fe2o3,
        )

    r2 = fe3o4_h2_reduction_rate_value(
        temperature_k=config.temperature_k,
        c_fe3o4_mol_per_m3=c_fe3o4,
        c_h2_mol_per_m3=gas.get("H2", 0.0),
        c_h2o_mol_per_m3=gas.get("H2O", 0.0),
        x_fe3o4=x_fe3o4,
    )
    r2 += fe3o4_co_reduction_rate_value(
        temperature_k=config.temperature_k,
        oc_mass_density_kg_per_m3=oc_density,
        c_co_mol_per_m3=gas.get("CO", 0.0),
        c_co2_mol_per_m3=gas.get("CO2", 0.0),
        x_fe3o4=x_fe3o4,
    )

    r3 = feo_h2_reduction_rate_value(
        temperature_k=config.temperature_k,
        c_feo_mol_per_m3=c_feo,
        c_h2_mol_per_m3=gas.get("H2", 0.0),
        c_h2o_mol_per_m3=gas.get("H2O", 0.0),
        x_feo=x_feo,
    )
    r3 += feo_co_reduction_rate_value(
        temperature_k=config.temperature_k,
        c_feo_mol_per_m3=c_feo,
        c_co_mol_per_m3=gas.get("CO", 0.0),
        c_co2_mol_per_m3=gas.get("CO2", 0.0),
        x_feo=x_feo,
    )

    return np.array(
        [
            -r1,
            (2.0 / 3.0) * r1 - r2,
            3.0 * r2 - r3,
            r3,
        ],
        dtype=float,
    )


def _simulate_reduction_case(
    *,
    temperature_k: float,
    initial_equivalent_fe2o3_mol_per_kg: float,
    oc_density_kg_per_m3: float,
    time_horizon_s: float,
    time_step_s: float,
    gas_concentrations_mol_per_m3: dict[str, float],
    max_internal_step_s: float = MAX_INTERNAL_REDUCTION_STEP_S,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    config = ReductionPanelConfig(
        mode="reduction",
        panel_id="internal",
        title="internal",
        temperature_k=temperature_k,
        initial_equivalent_fe2o3_mol_per_kg=initial_equivalent_fe2o3_mol_per_kg,
        oc_density_kg_per_m3=oc_density_kg_per_m3,
        time_horizon_s=time_horizon_s,
        time_step_s=time_step_s,
        mean_gas_concentrations_mol_per_m3=gas_concentrations_mol_per_m3,
        paper_curve_csv_path=VALIDATION_ROOT / "digitized" / "figs2_paper_curves.csv",
        artifact_stem="internal",
        x_limits_s=(0.0, time_horizon_s),
        y_limits=(0.0, 1.0),
        x_major_step_s=100.0,
        x_minor_step_s=20.0,
        y_major_step=0.2,
        y_minor_step=0.1,
    )
    state = _initial_state_from_equivalent_fe2o3(
        equivalent_fe2o3_mol_per_kg=initial_equivalent_fe2o3_mol_per_kg,
        oc_density_kg_per_m3=oc_density_kg_per_m3,
    )
    time_s = np.arange(0.0, time_horizon_s + time_step_s, time_step_s)
    curves = {curve_key: np.zeros_like(time_s) for curve_key in REDUCTION_CURVE_KEYS}
    internal_step_count = max(1, int(np.ceil(time_step_s / max_internal_step_s)))
    internal_step_s = time_step_s / internal_step_count

    for idx, _time in enumerate(time_s):
        conversions = _state_conversions(state)
        curves["x_fe2o3"][idx] = conversions[0]
        curves["x_fe3o4"][idx] = conversions[1]
        curves["x_feo"][idx] = conversions[2]

        for _ in range(internal_step_count):
            k1 = _reduction_rhs(state, config)
            k2 = _reduction_rhs(state + 0.5 * internal_step_s * k1, config)
            k3 = _reduction_rhs(state + 0.5 * internal_step_s * k2, config)
            k4 = _reduction_rhs(state + internal_step_s * k3, config)
            state = np.maximum(
                state + (internal_step_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
                0.0,
            )

    return time_s, curves


def _simulate_reduction_panel(config: ReductionPanelConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return _simulate_reduction_case(
        temperature_k=config.temperature_k,
        initial_equivalent_fe2o3_mol_per_kg=config.initial_equivalent_fe2o3_mol_per_kg,
        oc_density_kg_per_m3=config.oc_density_kg_per_m3,
        time_horizon_s=config.time_horizon_s,
        time_step_s=config.time_step_s,
        gas_concentrations_mol_per_m3=config.mean_gas_concentrations_mol_per_m3,
    )


def _simulate_supplementary_bohn_co_curve(
    *,
    phase: str,
    temperature_k: float,
    total_pressure_pa: float,
    time_horizon_s: float,
    time_step_s: float,
    gas_mole_fractions: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    # Figure S2 adopts the first two CO stages from Bohn et al. In that source,
    # the standalone curves are expressed in terms of stage conversion and inlet
    # gas vol%; reproducing the supplementary panel is therefore more faithful if
    # we integrate the stage-conversion ODE directly on a partial-pressure basis.
    if phase == "Fe2O3":
        phase_mw_kg_per_mol = FE2O3_MW_KG_PER_MOL
        exponent = 0.4
    elif phase == "Fe3O4":
        phase_mw_kg_per_mol = FE3O4_MW_KG_PER_MOL
        exponent = 1.2
    else:  # pragma: no cover - only called for the Bohn-derived CO stages
        raise ValueError(f"Unsupported Bohn CO phase {phase!r}")

    p_co_bar = max(float(gas_mole_fractions.get("CO", 0.0)), 0.0) * total_pressure_pa / 1.0e5
    p_co2_bar = max(float(gas_mole_fractions.get("CO2", 0.0)), 0.0) * total_pressure_pa / 1.0e5
    keq = float(equilibrium_constant_co_fe3o4_to_feo_value(temperature_k))
    driving_force_bar = max(p_co_bar - p_co2_bar / keq, 0.0)
    rate_constant = float(co_reaction_constant_value(phase, temperature_k=temperature_k))

    def rhs(x_stage: float) -> float:
        x_safe = min(max(x_stage, 0.0), 1.0)
        return phase_mw_kg_per_mol * rate_constant * driving_force_bar * (1.0 - x_safe) ** exponent

    time_s = np.arange(0.0, time_horizon_s + time_step_s, time_step_s)
    conversion = np.zeros_like(time_s)
    x_stage = 0.0
    internal_step_count = max(1, int(np.ceil(time_step_s / MAX_INTERNAL_REDUCTION_STEP_S)))
    internal_step_s = time_step_s / internal_step_count

    for idx, _time in enumerate(time_s):
        conversion[idx] = x_stage
        for _ in range(internal_step_count):
            k1 = rhs(x_stage)
            k2 = rhs(x_stage + 0.5 * internal_step_s * k1)
            k3 = rhs(x_stage + 0.5 * internal_step_s * k2)
            k4 = rhs(x_stage + internal_step_s * k3)
            x_stage = min(max(x_stage + (internal_step_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0), 1.0)

    return time_s, conversion


def _supplementary_stage_initial_concentration_mol_per_m3(*, phase: str) -> float:
    if phase == "Fe2O3":
        return FULLY_OXIDIZED_FE2O3_CONCENTRATION_MOL_PER_M3
    if phase == "Fe3O4":
        return (2.0 / 3.0) * FULLY_OXIDIZED_FE2O3_CONCENTRATION_MOL_PER_M3
    if phase == "FeO":
        return 2.0 * FULLY_OXIDIZED_FE2O3_CONCENTRATION_MOL_PER_M3
    raise ValueError(f"Unsupported supplementary Fe phase {phase!r}")


def _integrate_supplementary_stage_conversion(
    *,
    time_horizon_s: float,
    time_step_s: float,
    max_internal_step_s: float,
    rhs,
) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.arange(0.0, time_horizon_s + time_step_s, time_step_s)
    conversion = np.zeros_like(time_s)
    x_stage = 0.0
    internal_step_count = max(1, int(np.ceil(time_step_s / max_internal_step_s)))
    internal_step_s = time_step_s / internal_step_count

    for idx, _time in enumerate(time_s):
        conversion[idx] = x_stage
        for _ in range(internal_step_count):
            k1 = rhs(x_stage)
            k2 = rhs(x_stage + 0.5 * internal_step_s * k1)
            k3 = rhs(x_stage + 0.5 * internal_step_s * k2)
            k4 = rhs(x_stage + internal_step_s * k3)
            x_stage = min(
                max(x_stage + (internal_step_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0),
                1.0,
            )

    return time_s, conversion


def _simulate_supplementary_stage_curve(
    *,
    curve_id: str,
    phase: str,
    temperature_k: float,
    total_pressure_pa: float,
    time_horizon_s: float,
    time_step_s: float,
    oc_density_kg_per_m3: float,
    gas_mole_fractions: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    gas_concentrations = _gas_concentrations_from_mole_fractions(
        total_pressure_pa=total_pressure_pa,
        temperature_k=temperature_k,
        gas_mole_fractions=gas_mole_fractions,
    )
    initial_phase_concentration = _supplementary_stage_initial_concentration_mol_per_m3(phase=phase)

    def rhs(x_stage: float) -> float:
        x_safe = min(max(x_stage, 0.0), 1.0 - 1.0e-12)
        phase_concentration = initial_phase_concentration * (1.0 - x_safe)

        if curve_id == "fe2o3_h2":
            rate = fe2o3_h2_reduction_rate_value(
                temperature_k=temperature_k,
                c_fe2o3_mol_per_m3=phase_concentration,
                c_h2_mol_per_m3=gas_concentrations.get("H2", 0.0),
                c_h2o_mol_per_m3=gas_concentrations.get("H2O", 0.0),
                x_fe2o3=x_safe,
            )
        elif curve_id == "fe2o3_ch4":
            rate = fe2o3_ch4_reduction_rate_value(
                temperature_k=temperature_k,
                oc_mass_density_kg_per_m3=oc_density_kg_per_m3,
                c_ch4_mol_per_m3=gas_concentrations.get("CH4", 0.0),
                x_fe2o3=x_safe,
            )
        elif curve_id == "fe3o4_h2":
            rate = fe3o4_h2_reduction_rate_value(
                temperature_k=temperature_k,
                c_fe3o4_mol_per_m3=phase_concentration,
                c_h2_mol_per_m3=gas_concentrations.get("H2", 0.0),
                c_h2o_mol_per_m3=gas_concentrations.get("H2O", 0.0),
                x_fe3o4=x_safe,
            )
        elif curve_id == "feo_h2":
            rate = feo_h2_reduction_rate_value(
                temperature_k=temperature_k,
                c_feo_mol_per_m3=phase_concentration,
                c_h2_mol_per_m3=gas_concentrations.get("H2", 0.0),
                c_h2o_mol_per_m3=gas_concentrations.get("H2O", 0.0),
                x_feo=x_safe,
            )
        elif curve_id == "feo_co":
            rate = feo_co_reduction_rate_value(
                temperature_k=temperature_k,
                c_feo_mol_per_m3=phase_concentration,
                c_co_mol_per_m3=gas_concentrations.get("CO", 0.0),
                c_co2_mol_per_m3=gas_concentrations.get("CO2", 0.0),
                x_feo=x_safe,
            )
        else:  # pragma: no cover - caller filters supported curve ids
            raise ValueError(f"Unsupported supplementary stage curve {curve_id!r}")

        return rate / initial_phase_concentration

    max_internal_step_s = SUPPLEMENTARY_STAGE_MAX_INTERNAL_STEP_S.get(
        curve_id,
        MAX_INTERNAL_REDUCTION_STEP_S,
    )
    return _integrate_supplementary_stage_conversion(
        time_horizon_s=time_horizon_s,
        time_step_s=time_step_s,
        max_internal_step_s=max_internal_step_s,
        rhs=rhs,
    )


def _supplementary_full_oxidized_equivalent_fe2o3_mol_per_kg(*, oc_density_kg_per_m3: float) -> float:
    return FULLY_OXIDIZED_FE2O3_CONCENTRATION_MOL_PER_M3 / oc_density_kg_per_m3


def _simulate_supplementary_h2_sequence(
    config: SupplementaryReductionConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    gas_mole_fractions = {species_id: 0.0 for species_id in ("H2", "H2O", "CO", "CO2", "CH4")}
    gas_mole_fractions["H2"] = config.gas_mole_fractions.get("H2", 0.0)
    gas_concentrations = _gas_concentrations_from_mole_fractions(
        total_pressure_pa=config.total_pressure_pa,
        temperature_k=config.temperature_k,
        gas_mole_fractions=gas_mole_fractions,
    )
    time_s, curves = _simulate_reduction_case(
        temperature_k=config.temperature_k,
        initial_equivalent_fe2o3_mol_per_kg=_supplementary_full_oxidized_equivalent_fe2o3_mol_per_kg(
            oc_density_kg_per_m3=config.oc_density_kg_per_m3,
        ),
        oc_density_kg_per_m3=config.oc_density_kg_per_m3,
        time_horizon_s=config.time_horizon_s,
        time_step_s=config.time_step_s,
        gas_concentrations_mol_per_m3=gas_concentrations,
        max_internal_step_s=SUPPLEMENTARY_SEQUENCE_MAX_INTERNAL_STEP_S["H2"],
    )
    return time_s, {
        "fe2o3_h2": curves["x_fe2o3"],
        "fe3o4_h2": curves["x_fe3o4"],
        "feo_h2": curves["x_feo"],
    }


def _simulate_supplementary_co_sequence(
    config: SupplementaryReductionConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    c_full = FULLY_OXIDIZED_FE2O3_CONCENTRATION_MOL_PER_M3
    c_fe3o4_initial = (2.0 / 3.0) * c_full
    gas_mole_fractions = {species_id: 0.0 for species_id in ("H2", "H2O", "CO", "CO2", "CH4")}
    gas_mole_fractions["CO"] = config.gas_mole_fractions.get("CO", 0.0)
    gas_mole_fractions["CO2"] = config.gas_mole_fractions.get("CO2", 0.0)
    gas_concentrations = _gas_concentrations_from_mole_fractions(
        total_pressure_pa=config.total_pressure_pa,
        temperature_k=config.temperature_k,
        gas_mole_fractions=gas_mole_fractions,
    )
    p_co_bar = max(float(gas_mole_fractions.get("CO", 0.0)), 0.0) * config.total_pressure_pa / 1.0e5
    p_co2_bar = max(float(gas_mole_fractions.get("CO2", 0.0)), 0.0) * config.total_pressure_pa / 1.0e5
    p_co_driving_fe2o3_fe3o4_bar = max(
        p_co_bar - p_co2_bar / float(equilibrium_constant_co_fe3o4_to_feo_value(config.temperature_k)),
        0.0,
    )
    p_co_driving_fe3o4_feo_bar = max(
        p_co_bar - p_co2_bar / float(equilibrium_constant_co_fe3o4_to_feo_value(config.temperature_k)),
        0.0,
    )
    k_fe2o3_co = float(co_reaction_constant_value("Fe2O3", temperature_k=config.temperature_k))
    k_fe3o4_co = float(co_reaction_constant_value("Fe3O4", temperature_k=config.temperature_k))
    state = _initial_state_from_equivalent_fe2o3(
        equivalent_fe2o3_mol_per_kg=_supplementary_full_oxidized_equivalent_fe2o3_mol_per_kg(
            oc_density_kg_per_m3=config.oc_density_kg_per_m3,
        ),
        oc_density_kg_per_m3=config.oc_density_kg_per_m3,
    )
    time_s = np.arange(0.0, config.time_horizon_s + config.time_step_s, config.time_step_s)
    model_curves = {
        "fe2o3_co": np.zeros_like(time_s),
        "fe3o4_co": np.zeros_like(time_s),
        "feo_co": np.zeros_like(time_s),
    }
    internal_step_count = max(
        1,
        int(np.ceil(config.time_step_s / SUPPLEMENTARY_SEQUENCE_MAX_INTERNAL_STEP_S["CO"])),
    )
    internal_step_s = config.time_step_s / internal_step_count

    for idx, _time in enumerate(time_s):
        x_fe2o3, x_fe3o4, x_feo = _state_conversions(state)
        model_curves["fe2o3_co"][idx] = x_fe2o3
        model_curves["fe3o4_co"][idx] = x_fe3o4
        model_curves["feo_co"][idx] = x_feo

        for _ in range(internal_step_count):
            x_fe2o3, x_fe3o4, x_feo = _state_conversions(state)
            x_fe2o3 = min(max(float(x_fe2o3), 0.0), 1.0 - 1.0e-12)
            x_fe3o4 = min(max(float(x_fe3o4), 0.0), 1.0 - 1.0e-12)
            x_feo = min(max(float(x_feo), 0.0), 1.0 - 1.0e-12)

            r_fe2o3_co = (
                c_full
                * FE2O3_MW_KG_PER_MOL
                * k_fe2o3_co
                * p_co_driving_fe2o3_fe3o4_bar
                * (1.0 - x_fe2o3) ** 0.4
            )
            r_fe3o4_co = (
                c_fe3o4_initial
                * FE3O4_MW_KG_PER_MOL
                * k_fe3o4_co
                * p_co_driving_fe3o4_feo_bar
                * (1.0 - x_fe3o4) ** 1.2
            )
            r_feo_co = feo_co_reduction_rate_value(
                temperature_k=config.temperature_k,
                c_feo_mol_per_m3=max(float(state[2]), 0.0),
                c_co_mol_per_m3=gas_concentrations.get("CO", 0.0),
                c_co2_mol_per_m3=gas_concentrations.get("CO2", 0.0),
                x_feo=x_feo,
            )
            state += internal_step_s * np.array(
                [
                    -r_fe2o3_co,
                    (2.0 / 3.0) * r_fe2o3_co - r_fe3o4_co,
                    3.0 * r_fe3o4_co - r_feo_co,
                    r_feo_co,
                ],
                dtype=float,
            )
            state = np.maximum(state, 0.0)

    return time_s, model_curves


def _gas_concentrations_from_mole_fractions(
    *,
    total_pressure_pa: float,
    temperature_k: float,
    gas_mole_fractions: dict[str, float],
) -> dict[str, float]:
    gas_constant_j_per_mol_k = 8.31446261815324
    total_concentration_mol_per_m3 = total_pressure_pa / (gas_constant_j_per_mol_k * temperature_k)
    return {
        species_id: total_concentration_mol_per_m3 * max(float(mole_fraction), 0.0)
        for species_id, mole_fraction in gas_mole_fractions.items()
    }


def _simulate_supplementary_reduction_panel(
    config: SupplementaryReductionConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    paper_time_s, paper_curves = _load_supplementary_reduction_paper_curves(config.paper_curve_csv_path)
    model_time_s, model_curves = _simulate_supplementary_h2_sequence(config)
    co_time_s, co_curves = _simulate_supplementary_co_sequence(config)
    if co_time_s.shape != model_time_s.shape or not np.allclose(co_time_s, model_time_s):
        raise ValueError("Supplementary CO-sequence time grid does not match the H2-sequence time grid.")
    model_curves.update(co_curves)
    ch4_gas = {species_id: 0.0 for species_id in ("H2", "H2O", "CO", "CO2", "CH4")}
    ch4_gas["CH4"] = config.gas_mole_fractions.get("CH4", 0.0)
    _ch4_time_s, ch4_curve = _simulate_supplementary_stage_curve(
        curve_id="fe2o3_ch4",
        phase="Fe2O3",
        temperature_k=config.temperature_k,
        total_pressure_pa=config.total_pressure_pa,
        time_horizon_s=config.time_horizon_s,
        time_step_s=config.time_step_s,
        oc_density_kg_per_m3=config.oc_density_kg_per_m3,
        gas_mole_fractions=ch4_gas,
    )
    model_curves["fe2o3_ch4"] = ch4_curve
    return model_time_s, model_curves, paper_time_s, paper_curves


def _curve_rmse(paper_x: np.ndarray, paper_y: np.ndarray, model_x: np.ndarray, model_y: np.ndarray) -> float:
    model_interp = np.interp(paper_x, model_x, model_y)
    return float(np.sqrt(np.mean((model_interp - paper_y) ** 2)))


def _save_figure(figure, artifact_path: Path) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(artifact_path, format="svg", dpi=300, bbox_inches="tight")


def _plot_reduction_panel(
    *,
    config: ReductionPanelConfig,
    model_time_s: np.ndarray,
    model_curves: dict[str, np.ndarray],
    paper_time_s: np.ndarray,
    paper_curves: dict[str, np.ndarray],
) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))

    for curve_key in REDUCTION_CURVE_KEYS:
        color = REDUCTION_CURVE_COLORS[curve_key]
        axis.plot(
            model_time_s,
            model_curves[curve_key],
            color=color,
            linewidth=2.0,
            label=f"{REDUCTION_CURVE_LABELS[curve_key]} model",
        )
        axis.plot(
            paper_time_s,
            paper_curves[curve_key],
            color=color,
            linewidth=1.6,
            linestyle=(0, (5, 3)),
            alpha=0.9,
            label=f"{REDUCTION_CURVE_LABELS[curve_key]} paper",
        )

    axis.set_xlim(*config.x_limits_s)
    axis.set_ylim(*config.y_limits)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel(r"$X_i$ [-]")
    axis.set_xticks(np.arange(config.x_limits_s[0], config.x_limits_s[1] + 1.0e-9, config.x_major_step_s))
    axis.set_xticks(
        np.arange(config.x_limits_s[0], config.x_limits_s[1] + 1.0e-9, config.x_minor_step_s),
        minor=True,
    )
    axis.set_yticks(np.arange(config.y_limits[0], config.y_limits[1] + 1.0e-9, config.y_major_step))
    axis.set_yticks(np.arange(config.y_limits[0], config.y_limits[1] + 1.0e-9, config.y_minor_step), minor=True)

    _paper_axis_style(axis)
    axis.legend(
        loc="lower right",
        frameon=False,
        fontsize=9,
        ncol=2,
    )

    artifact_path = ARTIFACTS_DIR / f"{config.artifact_stem}.svg"
    _save_figure(figure, artifact_path)
    plt.close(figure)
    return artifact_path


def _plot_supplementary_reduction_panel(
    *,
    config: SupplementaryReductionConfig,
    model_time_s: np.ndarray,
    model_curves: dict[str, np.ndarray],
    paper_time_s: np.ndarray,
    paper_curves: dict[str, np.ndarray],
) -> Path:
    figure, axis = plt.subplots(figsize=(7.8, 5.4))

    for curve_id, curve_spec in SUPPLEMENTARY_REDUCTION_CURVE_SPECS.items():
        color = str(curve_spec["color"])
        label = str(curve_spec["label"])
        axis.plot(model_time_s, model_curves[curve_id], color=color, linewidth=2.2, label=f"{label} model")
        axis.plot(
            paper_time_s,
            paper_curves[curve_id],
            color=color,
            linewidth=1.7,
            linestyle=(0, (5, 3)),
            alpha=0.95,
            label=f"{label} paper",
        )

    axis.set_xlim(*config.x_limits_s)
    axis.set_ylim(*config.y_limits)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel(r"$X_i$ [-]")
    axis.set_xticks(np.arange(config.x_limits_s[0], config.x_limits_s[1] + 1.0e-9, config.x_major_step_s))
    axis.set_xticks(
        np.arange(config.x_limits_s[0], config.x_limits_s[1] + 1.0e-9, config.x_minor_step_s),
        minor=True,
    )
    axis.set_yticks(np.arange(config.y_limits[0], config.y_limits[1] + 1.0e-9, config.y_major_step))
    axis.set_yticks(np.arange(config.y_limits[0], config.y_limits[1] + 1.0e-9, config.y_minor_step), minor=True)

    _paper_axis_style(axis)
    axis.legend(loc="upper right", frameon=False, fontsize=10, ncol=1, handlelength=2.8)

    artifact_path = ARTIFACTS_DIR / f"{config.artifact_stem}.svg"
    _save_figure(figure, artifact_path)
    plt.close(figure)
    return artifact_path


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _format_curve_rmses(curve_rmses: dict[str, float] | None) -> str:
    if not curve_rmses:
        return ""
    return "; ".join(f"{curve_id}={value:.6f}" for curve_id, value in curve_rmses.items())


def _write_summary_csv(summaries: list[PanelSummary]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "panel_id",
                "mode",
                "rmse_mean",
                "rmse_fe2o3",
                "rmse_fe3o4",
                "rmse_feo",
                "initial_equivalent_fe2o3_mol_per_kg",
                "curve_rmses",
                "artifact_path",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.panel_id,
                    summary.mode,
                    f"{summary.rmse_mean:.6f}",
                    _format_optional_float(summary.rmse_fe2o3),
                    _format_optional_float(summary.rmse_fe3o4),
                    _format_optional_float(summary.rmse_feo),
                    _format_optional_float(summary.initial_equivalent_fe2o3_mol_per_kg),
                    _format_curve_rmses(summary.curve_rmses),
                    summary.artifact_path.as_posix(),
                ]
            )


def run_all_fe_redox_validations() -> tuple[PanelSummary, ...]:
    summaries: list[PanelSummary] = []

    for config in _load_panel_configs():
        if isinstance(config, ReductionPanelConfig):
            paper_time_s, paper_curves = _load_reduction_paper_curves(config.paper_curve_csv_path)
            model_time_s, model_curves = _simulate_reduction_panel(config)
            artifact_path = _plot_reduction_panel(
                config=config,
                model_time_s=model_time_s,
                model_curves=model_curves,
                paper_time_s=paper_time_s,
                paper_curves=paper_curves,
            )

            rmse_fe2o3 = _curve_rmse(
                paper_time_s,
                paper_curves["x_fe2o3"],
                model_time_s,
                model_curves["x_fe2o3"],
            )
            rmse_fe3o4 = _curve_rmse(
                paper_time_s,
                paper_curves["x_fe3o4"],
                model_time_s,
                model_curves["x_fe3o4"],
            )
            rmse_feo = _curve_rmse(
                paper_time_s,
                paper_curves["x_feo"],
                model_time_s,
                model_curves["x_feo"],
            )
            rmse_mean = float(np.mean([rmse_fe2o3, rmse_fe3o4, rmse_feo]))

            summaries.append(
                PanelSummary(
                    panel_id=config.panel_id,
                    mode=config.mode,
                    artifact_path=artifact_path,
                    rmse_mean=rmse_mean,
                    rmse_fe2o3=rmse_fe2o3,
                    rmse_fe3o4=rmse_fe3o4,
                    rmse_feo=rmse_feo,
                    initial_equivalent_fe2o3_mol_per_kg=config.initial_equivalent_fe2o3_mol_per_kg,
                )
            )
            continue

        if isinstance(config, SupplementaryReductionConfig):
            model_time_s, model_curves, paper_time_s, paper_curves = _simulate_supplementary_reduction_panel(config)
            artifact_path = _plot_supplementary_reduction_panel(
                config=config,
                model_time_s=model_time_s,
                model_curves=model_curves,
                paper_time_s=paper_time_s,
                paper_curves=paper_curves,
            )
            curve_rmses = {
                curve_id: _curve_rmse(paper_time_s, paper_curves[curve_id], model_time_s, model_curves[curve_id])
                for curve_id in SUPPLEMENTARY_REDUCTION_CURVE_IDS
            }
            rmse_fe2o3 = float(np.mean([curve_rmses["fe2o3_h2"], curve_rmses["fe2o3_co"], curve_rmses["fe2o3_ch4"]]))
            rmse_fe3o4 = float(np.mean([curve_rmses["fe3o4_h2"], curve_rmses["fe3o4_co"]]))
            rmse_feo = float(np.mean([curve_rmses["feo_h2"], curve_rmses["feo_co"]]))
            rmse_mean = float(np.mean(list(curve_rmses.values())))

            summaries.append(
                PanelSummary(
                    panel_id=config.panel_id,
                    mode=config.mode,
                    artifact_path=artifact_path,
                    rmse_mean=rmse_mean,
                    rmse_fe2o3=rmse_fe2o3,
                    rmse_fe3o4=rmse_fe3o4,
                    rmse_feo=rmse_feo,
                    curve_rmses=curve_rmses,
                )
            )
            continue

    _write_summary_csv(summaries)
    return tuple(summaries)


if __name__ == "__main__":
    for summary in run_all_fe_redox_validations():
        if summary.mode == "reduction":
            print(
                f"{summary.panel_id}: mean RMSE={summary.rmse_mean:.6f}, "
                f"Fe2O3={summary.rmse_fe2o3:.6f}, "
                f"Fe3O4={summary.rmse_fe3o4:.6f}, "
                f"FeO={summary.rmse_feo:.6f}"
            )
        elif summary.mode == "supplementary_reduction":
            print(
                f"{summary.panel_id}: mean RMSE={summary.rmse_mean:.6f}, "
                f"Fe2O3={summary.rmse_fe2o3:.6f}, "
                f"Fe3O4={summary.rmse_fe3o4:.6f}, "
                f"FeO={summary.rmse_feo:.6f}, "
                f"{_format_curve_rmses(summary.curve_rmses)}"
            )
