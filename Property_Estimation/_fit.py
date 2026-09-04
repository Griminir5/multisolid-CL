"""Shared array checks and linear-basis evaluation for the fitting tools."""

import numpy as np


def _as_1d_float_array(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _evaluate_basis_matrix(T_data, basis_funcs, name):
    if not basis_funcs:
        raise ValueError(f"{name} cannot be empty")

    columns = []
    for idx, basis_func in enumerate(basis_funcs):
        values = np.asarray(basis_func(T_data), dtype=float)
        if values.ndim == 0:
            values = np.full_like(T_data, float(values), dtype=float)
        if values.shape != T_data.shape:
            raise ValueError(
                f"{name}[{idx}] returned shape {values.shape}, expected {T_data.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name}[{idx}] returned non-finite values")
        columns.append(values)

    return np.column_stack(columns)


def evaluate_linear_basis_model(T_data, basis_funcs, params, offset=0.0):
    T_data = _as_1d_float_array(T_data, "T_data")
    params = _as_1d_float_array(params, "params")

    if len(basis_funcs) != len(params):
        raise ValueError("basis_funcs and params must have the same length")

    basis_matrix = _evaluate_basis_matrix(T_data, basis_funcs, "basis_funcs")
    return offset + basis_matrix @ params


def _safe_r_squared(y_true, y_pred):
    total_sum_squares = np.sum((y_true - np.mean(y_true)) ** 2)
    if total_sum_squares <= 0.0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / total_sum_squares
