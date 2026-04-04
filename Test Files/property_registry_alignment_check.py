from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROPERTY_MODULE_PATH = REPO_ROOT / "Packed Bed Models" / "packed_bed_properties.py"
ENTHALPY_DATA_ROOT = REPO_ROOT / "Property_Estimation" / "enth_hcap_data"
VISCOSITY_DATA_ROOT = REPO_ROOT / "Property_Estimation" / "visc_data"

ENTHALPY_REFIT_ATOL = 1e-6
VISCOSITY_REFIT_ATOL = 1e-12
ENTHALPY_WEIGHT = 9.0
HEAT_CAPACITY_WEIGHT = 1.0

DATA_TO_REGISTRY_KEY = {
    "ar": "AR",
    "caal2o4": "CaAl2O4",
    "ch4": "CH4",
    "co": "CO",
    "co2": "CO2",
    "h2": "H2",
    "h2o": "H2O",
    "he": "HE",
    "n2": "N2",
    "ni": "Ni",
    "nio": "NiO",
    "o2": "O2",
}


def load_property_module():
    spec = importlib.util.spec_from_file_location("packed_bed_properties", PROPERTY_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_enthalpy_series(path: Path) -> np.ndarray:
    return np.asarray(np.genfromtxt(path, dtype=float, skip_header=1), dtype=float)


def load_viscosity_series(path: Path) -> np.ndarray:
    return np.asarray(np.genfromtxt(path, dtype=float), dtype=float)


def safe_r_squared(y_true, y_pred) -> float:
    total_sum_squares = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total_sum_squares <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / total_sum_squares


def current_enthalpy_order(module, correlation) -> int:
    if isinstance(correlation, module.CpZerothMolar):
        return 0
    if isinstance(correlation, module.CpQuadraticMolar):
        return 2
    if isinstance(correlation, module.CpCubicMolar):
        return 3
    if isinstance(correlation, module.CpQuarticMolar):
        return 4
    raise TypeError(f"Unsupported enthalpy correlation type: {type(correlation).__name__}")


def build_enthalpy_matrices(module, correlation, temperature):
    delta_temperature = np.asarray(temperature, dtype=float) - correlation.t_ref

    if isinstance(correlation, module.CpZerothMolar):
        cp_matrix = np.column_stack([np.ones_like(delta_temperature)])
        h_matrix = np.column_stack([delta_temperature])
    elif isinstance(correlation, module.CpQuadraticMolar):
        cp_matrix = np.column_stack(
            [np.ones_like(delta_temperature), delta_temperature, delta_temperature**2]
        )
        h_matrix = np.column_stack(
            [
                delta_temperature,
                0.5 * delta_temperature**2,
                (delta_temperature**3) / 3.0,
            ]
        )
    elif isinstance(correlation, module.CpCubicMolar):
        cp_matrix = np.column_stack(
            [
                np.ones_like(delta_temperature),
                delta_temperature,
                delta_temperature**2,
                delta_temperature**3,
            ]
        )
        h_matrix = np.column_stack(
            [
                delta_temperature,
                0.5 * delta_temperature**2,
                (delta_temperature**3) / 3.0,
                0.25 * delta_temperature**4,
            ]
        )
    elif isinstance(correlation, module.CpQuarticMolar):
        cp_matrix = np.column_stack(
            [
                np.ones_like(delta_temperature),
                delta_temperature,
                delta_temperature**2,
                delta_temperature**3,
                delta_temperature**4,
            ]
        )
        h_matrix = np.column_stack(
            [
                delta_temperature,
                0.5 * delta_temperature**2,
                (delta_temperature**3) / 3.0,
                0.25 * delta_temperature**4,
                0.2 * delta_temperature**5,
            ]
        )
    else:
        raise TypeError(f"Unsupported enthalpy correlation type: {type(correlation).__name__}")

    return cp_matrix, h_matrix


def refit_enthalpy_coefficients(module, correlation, temperature, enthalpy, heat_capacity):
    cp_matrix, h_matrix = build_enthalpy_matrices(module, correlation, temperature)
    enthalpy_scale = max(float(np.std(enthalpy)), 1.0)
    heat_capacity_scale = max(float(np.std(heat_capacity)), 1.0)

    design_matrix = np.vstack(
        [
            (HEAT_CAPACITY_WEIGHT / heat_capacity_scale) * cp_matrix,
            (ENTHALPY_WEIGHT / enthalpy_scale) * h_matrix,
        ]
    )
    rhs = np.concatenate(
        [
            (HEAT_CAPACITY_WEIGHT / heat_capacity_scale) * heat_capacity,
            (ENTHALPY_WEIGHT / enthalpy_scale) * (enthalpy - correlation.h_form_ref),
        ]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, rhs, rcond=None)
    return coefficients


def get_enthalpy_coefficients(correlation):
    coefficients = []
    for attr_name in ("a0", "a1", "a2", "a3", "a4"):
        if hasattr(correlation, attr_name):
            coefficients.append(getattr(correlation, attr_name))
    return np.asarray(coefficients, dtype=float)


def summarize_enthalpy_alignment(module, registry, registry_key):
    correlation = registry.get_record(registry_key).enthalpy
    species_dir = ENTHALPY_DATA_ROOT / registry_key.lower()
    if not species_dir.is_dir():
        species_dir = ENTHALPY_DATA_ROOT / next(
            key for key, value in DATA_TO_REGISTRY_KEY.items() if value == registry_key
        )

    temperature = load_enthalpy_series(species_dir / "temp.csv")
    enthalpy = load_enthalpy_series(species_dir / "enth.csv")
    heat_capacity = load_enthalpy_series(species_dir / "hcap.csv")

    enthalpy_fit = correlation.value(temperature)
    heat_capacity_fit = correlation.cp_value(temperature)
    refit_coefficients = refit_enthalpy_coefficients(
        module, correlation, temperature, enthalpy, heat_capacity
    )
    stored_coefficients = get_enthalpy_coefficients(correlation)

    return {
        "order": current_enthalpy_order(module, correlation),
        "cp_rmse": float(np.sqrt(np.mean((heat_capacity_fit - heat_capacity) ** 2))),
        "cp_r2": safe_r_squared(heat_capacity, heat_capacity_fit),
        "h_rmse": float(np.sqrt(np.mean((enthalpy_fit - enthalpy) ** 2))),
        "h_r2": safe_r_squared(enthalpy, enthalpy_fit),
        "max_coeff_delta": float(np.max(np.abs(stored_coefficients - refit_coefficients))),
        "refit_matches": bool(
            np.allclose(stored_coefficients, refit_coefficients, atol=ENTHALPY_REFIT_ATOL, rtol=0.0)
        ),
        "t_min": float(np.min(temperature)),
        "t_max": float(np.max(temperature)),
    }


def refit_viscosity_coefficients(correlation, temperature, viscosity):
    delta_temperature = np.asarray(temperature, dtype=float) - correlation.t_ref
    design_matrix = np.column_stack(
        [np.ones_like(delta_temperature), delta_temperature, delta_temperature**2]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, viscosity, rcond=None)
    return coefficients


def summarize_viscosity_alignment(registry, registry_key):
    correlation = registry.get_record(registry_key).viscosity
    species_dir = VISCOSITY_DATA_ROOT / registry_key.lower()
    if not species_dir.is_dir():
        species_dir = VISCOSITY_DATA_ROOT / next(
            key for key, value in DATA_TO_REGISTRY_KEY.items() if value == registry_key
        )

    temperature = load_viscosity_series(species_dir / "temp.csv")
    viscosity = load_viscosity_series(species_dir / "visc.csv")

    viscosity_fit = correlation.value(temperature)
    refit_coefficients = refit_viscosity_coefficients(correlation, temperature, viscosity)
    stored_coefficients = np.asarray([correlation.a0, correlation.a1, correlation.a2], dtype=float)

    return {
        "order": 2,
        "rmse": float(np.sqrt(np.mean((viscosity_fit - viscosity) ** 2))),
        "r2": safe_r_squared(viscosity, viscosity_fit),
        "max_coeff_delta": float(np.max(np.abs(stored_coefficients - refit_coefficients))),
        "refit_matches": bool(
            np.allclose(stored_coefficients, refit_coefficients, atol=VISCOSITY_REFIT_ATOL, rtol=0.0)
        ),
        "t_min": float(np.min(temperature)),
        "t_max": float(np.max(temperature)),
    }


def build_enthalpy_order_matrix(temperature, order, t_ref):
    delta_temperature = np.asarray(temperature, dtype=float) - t_ref
    cp_matrix = np.column_stack([delta_temperature**power for power in range(order + 1)])
    h_matrix = np.column_stack(
        [delta_temperature ** (power + 1) / (power + 1) for power in range(order + 1)]
    )
    return cp_matrix, h_matrix


def best_enthalpy_bic_order(temperature, enthalpy, heat_capacity, h_form_ref, t_ref):
    enthalpy_scale = max(float(np.std(enthalpy)), 1.0)
    heat_capacity_scale = max(float(np.std(heat_capacity)), 1.0)
    best_order = None
    best_bic = None

    for order in range(6):
        cp_matrix, h_matrix = build_enthalpy_order_matrix(temperature, order, t_ref)
        design_matrix = np.vstack(
            [
                (HEAT_CAPACITY_WEIGHT / heat_capacity_scale) * cp_matrix,
                (ENTHALPY_WEIGHT / enthalpy_scale) * h_matrix,
            ]
        )
        rhs = np.concatenate(
            [
                (HEAT_CAPACITY_WEIGHT / heat_capacity_scale) * heat_capacity,
                (ENTHALPY_WEIGHT / enthalpy_scale) * (enthalpy - h_form_ref),
            ]
        )
        coefficients, _, _, _ = np.linalg.lstsq(design_matrix, rhs, rcond=None)
        cp_error = cp_matrix @ coefficients - heat_capacity
        h_error = h_form_ref + h_matrix @ coefficients - enthalpy
        weighted_rss = float(
            np.sum(((HEAT_CAPACITY_WEIGHT / heat_capacity_scale) * cp_error) ** 2)
            + np.sum(((ENTHALPY_WEIGHT / enthalpy_scale) * h_error) ** 2)
        )
        n_points = 2 * temperature.size
        n_params = order + 1
        bic = n_points * np.log(weighted_rss / n_points) + n_params * np.log(n_points)
        if best_bic is None or bic < best_bic:
            best_order = order
            best_bic = bic

    return best_order


def best_viscosity_bic_order(temperature, viscosity, t_ref):
    best_order = None
    best_bic = None
    scale = max(float(np.std(viscosity)), 1.0)

    for order in range(4):
        delta_temperature = np.asarray(temperature, dtype=float) - t_ref
        design_matrix = np.column_stack(
            [delta_temperature**power for power in range(order + 1)]
        )
        coefficients, _, _, _ = np.linalg.lstsq(design_matrix, viscosity, rcond=None)
        residual = design_matrix @ coefficients - viscosity
        weighted_rss = float(np.sum((residual / scale) ** 2))
        n_points = temperature.size
        n_params = order + 1
        bic = n_points * np.log(weighted_rss / n_points) + n_params * np.log(n_points)
        if best_bic is None or bic < best_bic:
            best_order = order
            best_bic = bic

    return best_order


def main():
    module = load_property_module()
    registry = module.DEFAULT_PROPERTY_REGISTRY

    failures = []
    enthalpy_lines = []
    viscosity_lines = []
    advisory_lines = []

    enthalpy_data_keys = {
        DATA_TO_REGISTRY_KEY[path.name] for path in ENTHALPY_DATA_ROOT.iterdir() if path.is_dir()
    }
    viscosity_data_keys = {
        DATA_TO_REGISTRY_KEY[path.name] for path in VISCOSITY_DATA_ROOT.iterdir() if path.is_dir()
    }

    for registry_key in sorted(enthalpy_data_keys):
        summary = summarize_enthalpy_alignment(module, registry, registry_key)
        enthalpy_lines.append(
            f"{registry_key:8s} order={summary['order']} "
            f"T=[{summary['t_min']:.1f}, {summary['t_max']:.1f}] "
            f"cp_rmse={summary['cp_rmse']:.6g} h_rmse={summary['h_rmse']:.6g} "
            f"coeff_delta={summary['max_coeff_delta']:.3e}"
        )
        if not summary["refit_matches"]:
            failures.append(
                f"Enthalpy coefficients for {registry_key} differ from the same-basis refit "
                f"by {summary['max_coeff_delta']:.3e}."
            )

        species_dir = ENTHALPY_DATA_ROOT / next(
            key for key, value in DATA_TO_REGISTRY_KEY.items() if value == registry_key
        )
        temperature = load_enthalpy_series(species_dir / "temp.csv")
        enthalpy = load_enthalpy_series(species_dir / "enth.csv")
        heat_capacity = load_enthalpy_series(species_dir / "hcap.csv")
        best_order = best_enthalpy_bic_order(
            temperature,
            enthalpy,
            heat_capacity,
            registry.get_record(registry_key).enthalpy.h_form_ref,
            registry.get_record(registry_key).enthalpy.t_ref,
        )
        if best_order != summary["order"]:
            advisory_lines.append(
                f"Enthalpy: {registry_key} uses order {summary['order']}, but order {best_order} "
                "is the best BIC fit on the current CSV data."
            )

    for registry_key in sorted(viscosity_data_keys):
        summary = summarize_viscosity_alignment(registry, registry_key)
        viscosity_lines.append(
            f"{registry_key:8s} order={summary['order']} "
            f"T=[{summary['t_min']:.1f}, {summary['t_max']:.1f}] "
            f"rmse={summary['rmse']:.6g} coeff_delta={summary['max_coeff_delta']:.3e}"
        )
        if not summary["refit_matches"]:
            failures.append(
                f"Viscosity coefficients for {registry_key} differ from the same-basis refit "
                f"by {summary['max_coeff_delta']:.3e}."
            )

        species_dir = VISCOSITY_DATA_ROOT / next(
            key for key, value in DATA_TO_REGISTRY_KEY.items() if value == registry_key
        )
        temperature = load_viscosity_series(species_dir / "temp.csv")
        viscosity = load_viscosity_series(species_dir / "visc.csv")
        best_order = best_viscosity_bic_order(
            temperature,
            viscosity,
            registry.get_record(registry_key).viscosity.t_ref,
        )
        if best_order != summary["order"]:
            advisory_lines.append(
                f"Viscosity: {registry_key} uses order {summary['order']}, but order {best_order} "
                "is the best BIC fit on the current CSV data."
            )

    unverified_enthalpy_keys = sorted(
        key
        for key, record in registry.records.items()
        if record.enthalpy is not None and key not in enthalpy_data_keys
    )

    print("Enthalpy registry/data alignment")
    for line in enthalpy_lines:
        print(f"  {line}")
    if unverified_enthalpy_keys:
        print(
            "  Unverified against local enthalpy CSVs: "
            + ", ".join(unverified_enthalpy_keys)
        )

    print()
    print("Viscosity registry/data alignment")
    for line in viscosity_lines:
        print(f"  {line}")

    if advisory_lines:
        print()
        print("Model-order advisories")
        for line in advisory_lines:
            print(f"  {line}")

    if failures:
        print()
        print("Failures")
        for line in failures:
            print(f"  {line}")
        raise SystemExit(1)

    print()
    print("Alignment check passed.")


if __name__ == "__main__":
    main()
