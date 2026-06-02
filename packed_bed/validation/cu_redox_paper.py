from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import matplotlib
import yaml

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..kinetics.cu_redox import (
    cu2o_h2_reduction_rate_value,
    cu_al2o3_ox1_rate_value,
    cu_al2o3_ox2_rate_value,
    cu_al2o3_ox3_rate_value,
    cu_al2o3_sp1_rate_value,
    cu_al2o3_sp2_rate_value,
    cu_al2o3_sp3_rate_value,
    cuo_h2_reduction_rate_value,
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


VALIDATION_ROOT = Path(__file__).resolve().parents[1] / "examples" / "cu_redox_validation"
ARTIFACTS_DIR = VALIDATION_ROOT / "output" / "artifacts"
SUMMARY_PATH = ARTIFACTS_DIR / "cu_redox_validation_summary.csv"

RUN_FILE_NAMES = (
    "fig19a_run.yaml",
    "fig19b_run.yaml",
    "fig20a_run.yaml",
    "fig20b_run.yaml",
)

MW_KG_PER_MOL = {
    "Cu": 63.55e-3,
    "Cu2O": 143.091e-3,
    "CuO": 79.545e-3,
    "CuAl2O4": 181.508e-3,
    "CuAlO2": 122.526e-3,
    "Al2O3": 101.961e-3,
}

# Table 3 densities from the paper.
PHASE_DENSITY_KG_PER_M3 = {
    "Cu": 8.96e3,
    "Cu2O": 6.0e3,
    "CuO": 6.315e3,
    "Al2O3": 3.98e3,
}

# The paper does not tabulate spinel densities. These effective values follow
# the additive-volume interpretation of Eq. (27) using the Table 3 oxide data.
PHASE_DENSITY_KG_PER_M3["CuAl2O4"] = MW_KG_PER_MOL["CuAl2O4"] / (
    MW_KG_PER_MOL["CuO"] / PHASE_DENSITY_KG_PER_M3["CuO"]
    + MW_KG_PER_MOL["Al2O3"] / PHASE_DENSITY_KG_PER_M3["Al2O3"]
)
PHASE_DENSITY_KG_PER_M3["CuAlO2"] = MW_KG_PER_MOL["CuAlO2"] / (
    0.5 * MW_KG_PER_MOL["Cu2O"] / PHASE_DENSITY_KG_PER_M3["Cu2O"]
    + 0.5 * MW_KG_PER_MOL["Al2O3"] / PHASE_DENSITY_KG_PER_M3["Al2O3"]
)

BLACK = "#111111"
RED = "#d62828"


@dataclass(frozen=True)
class PanelConfig:
    panel_id: str
    title: str
    mode: str
    temperature_k: float
    pressure_pa: float
    p_h2_pa: float
    p_o2_pa: float
    sample_mass_mg: float
    time_offset_s: float
    t_end_s: float
    dt_s: float
    y_limits_mg: tuple[float, float]
    y_major_step_mg: float
    y_minor_step_mg: float
    x_major_step_s: float
    x_minor_step_s: float
    experimental_csv_path: Path
    artifact_stem: str
    initial_basis_masses_g: dict[str, float]


@dataclass(frozen=True)
class PanelSummary:
    panel_id: str
    artifact_path: Path
    sample_mass_mg: float
    model_delta_mass_mg: float
    experimental_delta_mass_mg: float
    rmse_mg: float


def _paper_axis_style(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.tick_params(which="major", width=0.8, length=4, colors=BLACK)
    axis.tick_params(which="minor", width=0.6, length=2, colors=BLACK)
    axis.xaxis.label.set_color(BLACK)
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


def _load_experimental_trace(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def _cu_equivalent_to_phase_masses_g(cu_equivalent_wt: dict[str, float]) -> dict[str, float]:
    masses_g: dict[str, float] = {}
    for species_id, cu_equivalent_mass_g in cu_equivalent_wt.items():
        if species_id == "Cu":
            masses_g["Cu"] = cu_equivalent_mass_g
            continue
        if species_id == "Cu2O":
            masses_g["Cu2O"] = cu_equivalent_mass_g * MW_KG_PER_MOL["Cu2O"] / (2.0 * MW_KG_PER_MOL["Cu"])
            continue
        if species_id == "CuO":
            masses_g["CuO"] = cu_equivalent_mass_g * MW_KG_PER_MOL["CuO"] / MW_KG_PER_MOL["Cu"]
            continue
        if species_id == "CuAl2O4":
            masses_g["CuAl2O4"] = cu_equivalent_mass_g * MW_KG_PER_MOL["CuAl2O4"] / MW_KG_PER_MOL["Cu"]
            continue
        if species_id == "CuAlO2":
            masses_g["CuAlO2"] = cu_equivalent_mass_g * MW_KG_PER_MOL["CuAlO2"] / MW_KG_PER_MOL["Cu"]
            continue
        raise ValueError(f"Unsupported Cu phase '{species_id}' in validation solids.")
    return masses_g


def _total_al_moles_from_basis_masses(masses_g: dict[str, float]) -> float:
    return (
        2.0 * masses_g.get("Al2O3", 0.0) / (MW_KG_PER_MOL["Al2O3"] * 1.0e3)
        + 2.0 * masses_g.get("CuAl2O4", 0.0) / (MW_KG_PER_MOL["CuAl2O4"] * 1.0e3)
        + masses_g.get("CuAlO2", 0.0) / (MW_KG_PER_MOL["CuAlO2"] * 1.0e3)
    )


def _complete_basis_masses_g(
    *,
    reference_oxidized_cu_equivalent_wt: dict[str, float],
    initial_cu_equivalent_wt: dict[str, float],
) -> dict[str, float]:
    reference_masses_g = _cu_equivalent_to_phase_masses_g(reference_oxidized_cu_equivalent_wt)
    reference_masses_g["Al2O3"] = 100.0 - sum(reference_masses_g.values())
    total_al_moles = _total_al_moles_from_basis_masses(reference_masses_g)

    initial_masses_g = _cu_equivalent_to_phase_masses_g(initial_cu_equivalent_wt)
    al_moles_locked = (
        2.0 * initial_masses_g.get("CuAl2O4", 0.0) / (MW_KG_PER_MOL["CuAl2O4"] * 1.0e3)
        + initial_masses_g.get("CuAlO2", 0.0) / (MW_KG_PER_MOL["CuAlO2"] * 1.0e3)
    )
    initial_masses_g["Al2O3"] = (total_al_moles - al_moles_locked) * 0.5 * MW_KG_PER_MOL["Al2O3"] * 1.0e3
    return initial_masses_g


def _load_panel_config(run_path: Path) -> PanelConfig:
    run_data = _read_yaml(run_path)
    references = run_data["references"]
    validation = run_data["validation"]
    base_dir = run_path.parent

    chemistry_path = (base_dir / references["chemistry_file"]).resolve()
    program_path = (base_dir / references["program_file"]).resolve()
    solids_path = (base_dir / references["solids_file"]).resolve()

    chemistry = _read_yaml(chemistry_path)
    program = _read_yaml(program_path)
    solids = _read_yaml(solids_path)

    gas_composition = {species_id: float(value) for species_id, value in program["gas_composition"].items()}
    pressure_pa = _as_float(program, "pressure_pa")
    initial_basis_masses_g = _complete_basis_masses_g(
        reference_oxidized_cu_equivalent_wt=solids["reference_oxidized_cu_equivalent_wt"],
        initial_cu_equivalent_wt=solids["initial_cu_equivalent_wt"],
    )

    return PanelConfig(
        panel_id=str(validation["panel_id"]),
        title=str(validation["title"]),
        mode=str(chemistry["validation_mode"]),
        temperature_k=_as_float(program, "temperature_k"),
        pressure_pa=pressure_pa,
        p_h2_pa=pressure_pa * float(gas_composition.get("H2", 0.0)),
        p_o2_pa=pressure_pa * float(gas_composition.get("O2", 0.0)),
        sample_mass_mg=_as_float(program, "sample_mass_mg"),
        time_offset_s=float(program.get("time_offset_s", 0.0)),
        t_end_s=_as_float(program, "time_horizon_s"),
        dt_s=_as_float(program, "time_step_s"),
        y_limits_mg=(
            float(validation["y_limits_mg"][0]),
            float(validation["y_limits_mg"][1]),
        ),
        y_major_step_mg=_as_float(validation, "y_major_step_mg"),
        y_minor_step_mg=_as_float(validation, "y_minor_step_mg"),
        x_major_step_s=_as_float(validation, "x_major_step_s"),
        x_minor_step_s=_as_float(validation, "x_minor_step_s"),
        experimental_csv_path=(base_dir / validation["experimental_csv"]).resolve(),
        artifact_stem=str(validation["artifact_stem"]),
        initial_basis_masses_g=initial_basis_masses_g,
    )


def _load_panel_configs() -> tuple[PanelConfig, ...]:
    return tuple(_load_panel_config(VALIDATION_ROOT / file_name) for file_name in RUN_FILE_NAMES)


def _moles_from_basis_masses(masses_g: dict[str, float]) -> dict[str, float]:
    return {
        species_id: masses_g[species_id] / 1.0e3 / MW_KG_PER_MOL[species_id]
        for species_id in masses_g
    }


def _mass_mg_from_state_vector(state_vector: np.ndarray, species_order: tuple[str, ...]) -> float:
    return 1.0e6 * sum(
        amount_mol * MW_KG_PER_MOL[species_id]
        for amount_mol, species_id in zip(state_vector, species_order, strict=True)
    )


def _particle_volume_m3(state: dict[str, float]) -> float:
    return sum(
        state[species_id] * MW_KG_PER_MOL[species_id] / PHASE_DENSITY_KG_PER_M3[species_id]
        for species_id in state
    )


def _rk4_integrate(
    rhs,
    *,
    initial_state: np.ndarray,
    t_end_s: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    times_s = np.arange(0.0, t_end_s + 0.5 * dt_s, dt_s)
    states = np.empty((times_s.size, initial_state.size), dtype=float)
    states[0] = initial_state

    for idx in range(times_s.size - 1):
        state = states[idx]
        k1 = rhs(state)
        k2 = rhs(np.maximum(state + 0.5 * dt_s * k1, 0.0))
        k3 = rhs(np.maximum(state + 0.5 * dt_s * k2, 0.0))
        k4 = rhs(np.maximum(state + dt_s * k3, 0.0))
        states[idx + 1] = np.maximum(state + dt_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, 0.0)

    return times_s, states


def _simulate_reduction_panel(config: PanelConfig) -> tuple[np.ndarray, np.ndarray]:
    species_order = ("CuO", "Cu2O", "Cu", "CuAl2O4", "CuAlO2", "Al2O3")
    initial_moles = _moles_from_basis_masses(config.initial_basis_masses_g)
    initial_state = np.array([initial_moles.get(species_id, 0.0) for species_id in species_order], dtype=float)

    def rhs(state_vector: np.ndarray) -> np.ndarray:
        state = {species_id: max(state_vector[idx], 0.0) for idx, species_id in enumerate(species_order)}
        particle_volume_m3 = _particle_volume_m3(state)
        concentrations = {
            species_id: amount_mol / particle_volume_m3
            for species_id, amount_mol in state.items()
        }
        red1 = cuo_h2_reduction_rate_value(
            temperature_k=config.temperature_k,
            c_cuo_mol_per_m3=concentrations["CuO"],
            p_h2_pa=config.p_h2_pa,
        )
        red2 = cu2o_h2_reduction_rate_value(
            temperature_k=config.temperature_k,
            c_cu2o_mol_per_m3=concentrations["Cu2O"],
            p_h2_pa=config.p_h2_pa,
        )
        sp1 = cu_al2o3_sp1_rate_value(
            temperature_k=config.temperature_k,
            c_cual2o4_mol_per_m3=concentrations["CuAl2O4"],
            p_h2_pa=config.p_h2_pa,
        )
        sp2 = cu_al2o3_sp2_rate_value(
            temperature_k=config.temperature_k,
            c_cual2o4_mol_per_m3=concentrations["CuAl2O4"],
            p_h2_pa=config.p_h2_pa,
        )
        sp3 = cu_al2o3_sp3_rate_value(
            temperature_k=config.temperature_k,
            c_cualo2_mol_per_m3=concentrations["CuAlO2"],
            p_h2_pa=config.p_h2_pa,
        )
        return particle_volume_m3 * np.array(
            (
                -2.0 * red1,
                red1 - red2,
                2.0 * red2 + sp1 + sp3,
                -sp1 - sp2,
                sp2 - sp3,
                sp1 + 0.5 * sp2 + 0.5 * sp3,
            )
        )

    times_s, states = _rk4_integrate(rhs, initial_state=initial_state, t_end_s=config.t_end_s, dt_s=config.dt_s)
    masses_mg = np.array(
        [_mass_mg_from_state_vector(state_vector, species_order) for state_vector in states],
        dtype=float,
    )
    return times_s, masses_mg


def _simulate_oxidation_panel(config: PanelConfig) -> tuple[np.ndarray, np.ndarray]:
    species_order = ("CuO", "Cu", "CuAl2O4", "CuAlO2", "Al2O3")
    initial_moles = _moles_from_basis_masses(config.initial_basis_masses_g)
    initial_state = np.array([initial_moles.get(species_id, 0.0) for species_id in species_order], dtype=float)

    def rhs(state_vector: np.ndarray) -> np.ndarray:
        state = {species_id: max(state_vector[idx], 0.0) for idx, species_id in enumerate(species_order)}
        particle_volume_m3 = _particle_volume_m3(state)
        concentrations = {
            species_id: amount_mol / particle_volume_m3
            for species_id, amount_mol in state.items()
        }
        ox1 = cu_al2o3_ox1_rate_value(
            temperature_k=config.temperature_k,
            c_cu_mol_per_m3=concentrations["Cu"],
            p_o2_pa=config.p_o2_pa,
        )
        ox2 = cu_al2o3_ox2_rate_value(
            temperature_k=config.temperature_k,
            c_cuo_mol_per_m3=concentrations["CuO"],
            c_al2o3_mol_per_m3=concentrations["Al2O3"],
        )
        ox3 = cu_al2o3_ox3_rate_value(
            temperature_k=config.temperature_k,
            c_cualo2_mol_per_m3=concentrations["CuAlO2"],
            c_al2o3_mol_per_m3=concentrations["Al2O3"],
            p_o2_pa=config.p_o2_pa,
        )
        return particle_volume_m3 * np.array(
            (
                ox1 - ox2,
                -ox1,
                ox2 + ox3,
                -ox3,
                -ox2 - 0.5 * ox3,
            )
        )

    times_s, states = _rk4_integrate(rhs, initial_state=initial_state, t_end_s=config.t_end_s, dt_s=config.dt_s)
    masses_mg = np.array(
        [_mass_mg_from_state_vector(state_vector, species_order) for state_vector in states],
        dtype=float,
    )
    return times_s, masses_mg


def _simulate_panel(config: PanelConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.mode == "reduction":
        return _simulate_reduction_panel(config)
    if config.mode == "oxidation":
        return _simulate_oxidation_panel(config)
    raise ValueError(f"Unsupported panel mode '{config.mode}' for {config.panel_id}")


def _scale_model_to_sample_mass(config: PanelConfig, basis_mass_mg: np.ndarray) -> np.ndarray:
    return basis_mass_mg * (config.sample_mass_mg / basis_mass_mg[0])


def _align_model_to_paper_axis(experimental_mass_mg: np.ndarray, model_absolute_mass_mg: np.ndarray) -> np.ndarray:
    return experimental_mass_mg[0] + (model_absolute_mass_mg - model_absolute_mass_mg[0])


def _rmse_delta_mg(experimental_mass_mg: np.ndarray, model_display_mg: np.ndarray) -> float:
    experimental_delta_mg = experimental_mass_mg - experimental_mass_mg[0]
    model_delta_mg = model_display_mg - model_display_mg[0]
    return float(np.sqrt(np.mean((model_delta_mg - experimental_delta_mg) ** 2)))


def _smooth_display_curve(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x_values.size < 4:
        return x_values, y_values

    dense_x = np.linspace(float(x_values[0]), float(x_values[-1]), max(400, x_values.size * 8))
    dense_y = np.interp(dense_x, x_values, y_values)

    # Keep smoothing display-only and gentle so the plotted digitized trace
    # looks less jagged without changing the underlying validation data.
    kernel = np.array((1.0, 2.0, 3.0, 2.0, 1.0), dtype=float)
    kernel /= kernel.sum()
    padded_y = np.pad(dense_y, (2, 2), mode="edge")
    smooth_y = np.convolve(padded_y, kernel, mode="valid")
    return dense_x, smooth_y


def _save_figure(figure, stem: str) -> Path:
    svg_path = ARTIFACTS_DIR / f"{stem}.svg"
    figure.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight")
    return svg_path


def _plot_panel(config: PanelConfig) -> PanelSummary:
    experimental_time_s, experimental_mass_mg = _load_experimental_trace(config.experimental_csv_path)
    model_time_s, model_basis_mass_mg = _simulate_panel(config)
    aligned_model_time_s = np.clip(experimental_time_s - config.time_offset_s, 0.0, model_time_s[-1])
    model_basis_mass_mg = np.interp(aligned_model_time_s, model_time_s, model_basis_mass_mg)
    model_absolute_mass_mg = _scale_model_to_sample_mass(config, model_basis_mass_mg)
    model_display_mass_mg = _align_model_to_paper_axis(experimental_mass_mg, model_absolute_mass_mg)
    rmse_mg = _rmse_delta_mg(experimental_mass_mg, model_display_mass_mg)
    display_experimental_time_s, display_experimental_mass_mg = _smooth_display_curve(
        experimental_time_s,
        experimental_mass_mg,
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(
        experimental_time_s,
        model_display_mass_mg,
        color=BLACK,
        linewidth=2.0,
        label="Model",
    )
    axis.plot(
        display_experimental_time_s,
        display_experimental_mass_mg,
        color=RED,
        linewidth=1.6,
        linestyle=(0, (5, 3)),
        alpha=0.9,
        label="Paper",
    )

    axis.set_xlim(0.0, config.t_end_s)
    axis.set_ylim(*config.y_limits_mg)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Mass change [mg]")
    axis.set_xticks(np.arange(0.0, config.t_end_s + 1.0, config.x_major_step_s))
    axis.set_xticks(np.arange(0.0, config.t_end_s + 1.0, config.x_minor_step_s), minor=True)
    major_y_start = np.ceil(config.y_limits_mg[0])
    axis.set_yticks(np.arange(major_y_start, config.y_limits_mg[1] + 1.0e-9, config.y_major_step_mg))
    axis.set_yticks(np.arange(config.y_limits_mg[0], config.y_limits_mg[1] + 1.0e-9, config.y_minor_step_mg), minor=True)
    axis.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
    axis.text(
        0.03,
        0.97,
        f"RMSE ($\\Delta m$): {rmse_mg:.3f} mg",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=BLACK,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )
    _paper_axis_style(axis)

    artifact_path = _save_figure(figure, config.artifact_stem)
    plt.close(figure)
    return PanelSummary(
        panel_id=config.panel_id,
        artifact_path=artifact_path,
        sample_mass_mg=config.sample_mass_mg,
        model_delta_mass_mg=float(model_display_mass_mg[-1] - model_display_mass_mg[0]),
        experimental_delta_mass_mg=float(experimental_mass_mg[-1] - experimental_mass_mg[0]),
        rmse_mg=rmse_mg,
    )


def _write_summary_file(panel_summaries: list[PanelSummary]) -> None:
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "panel_id",
                "sample_mass_mg",
                "model_delta_mass_mg",
                "experimental_delta_mass_mg",
                "rmse_delta_mass_mg",
                "artifact_svg",
            )
        )
        for summary in panel_summaries:
            writer.writerow(
                (
                    summary.panel_id,
                    f"{summary.sample_mass_mg:.6f}",
                    f"{summary.model_delta_mass_mg:.6f}",
                    f"{summary.experimental_delta_mass_mg:.6f}",
                    f"{summary.rmse_mg:.6f}",
                    str(summary.artifact_path),
                )
            )


def run_all_cu_redox_validations() -> dict[str, Path]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [_plot_panel(config) for config in _load_panel_configs()]
    _write_summary_file(summaries)
    return {summary.panel_id: summary.artifact_path for summary in summaries}


if __name__ == "__main__":
    run_all_cu_redox_validations()
