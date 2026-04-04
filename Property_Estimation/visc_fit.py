from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

species = "o2"
max_poly_order = 2
plot_all_models = True


def load_viscosity_data(species):
    data_root = Path(__file__).resolve().parent / "visc_data"
    data_dir = data_root / species

    if not data_dir.is_dir():
        available_species = ", ".join(sorted(path.name for path in data_root.iterdir() if path.is_dir()))
        raise FileNotFoundError(
            f"No viscosity data found for '{species}'. Available species: {available_species}"
        )

    temp_path = data_dir / "temp.csv"
    visc_path = data_dir / "visc.csv"

    if not temp_path.is_file():
        raise FileNotFoundError(f"Temperature data file is missing: {temp_path}")
    if not visc_path.is_file():
        raise FileNotFoundError(f"Viscosity data file is missing: {visc_path}")

    T_data = np.genfromtxt(temp_path, dtype=float)
    visc_data = np.genfromtxt(visc_path, dtype=float)
    return T_data, visc_data, data_dir


def _as_1d_float_array(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _safe_scale(values):
    values = _as_1d_float_array(values, "values")
    scale = float(np.std(values))
    if np.isfinite(scale) and scale > 0.0:
        return scale

    fallback = float(np.max(np.abs(values)))
    if np.isfinite(fallback) and fallback > 0.0:
        return fallback
    return 1.0


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


def fit_linear_basis_viscosity(T_data, visc_data, basis_funcs):
    T_data = _as_1d_float_array(T_data, "T_data")
    visc_data = _as_1d_float_array(visc_data, "visc_data")

    if T_data.shape != visc_data.shape:
        raise ValueError("T_data and visc_data must have the same shape")

    A = _evaluate_basis_matrix(T_data, basis_funcs, "basis_funcs")
    theta, residuals, rank, singular_values = np.linalg.lstsq(A, visc_data, rcond=None)
    return theta, residuals, rank, singular_values


def evaluate_linear_basis_model(T_data, basis_funcs, params):
    T_data = _as_1d_float_array(T_data, "T_data")
    params = _as_1d_float_array(params, "params")

    if len(basis_funcs) != len(params):
        raise ValueError("basis_funcs and params must have the same length")

    basis_matrix = _evaluate_basis_matrix(T_data, basis_funcs, "basis_funcs")
    return basis_matrix @ params


def _poly_basis(power, t_ref):
    def basis_func(T_data):
        delta_temperature = np.asarray(T_data, dtype=float) - t_ref
        return delta_temperature**power

    return basis_func


def make_polynomial_basis(order, t_ref):
    if order < 0:
        raise ValueError("Polynomial order must be non-negative")

    return {
        "name": f"poly_order_{order}",
        "kind": "linear_basis",
        "basis_funcs": [_poly_basis(power, t_ref) for power in range(order + 1)],
        "reference_temperature": t_ref,
    }


def power_law_model(T_data, A, B):
    T_data = np.asarray(T_data, dtype=float)
    return A * T_data**B


def _power_law_jacobian(T_data, A, B):
    T_data = _as_1d_float_array(T_data, "T_data")
    power_term = T_data**B
    return np.column_stack([power_term, A * power_term * np.log(T_data)])


def fit_power_law_model(T_data, visc_data):
    T_data = _as_1d_float_array(T_data, "T_data")
    visc_data = _as_1d_float_array(visc_data, "visc_data")

    if T_data.shape != visc_data.shape:
        raise ValueError("T_data and visc_data must have the same shape")
    if np.any(T_data <= 0.0):
        raise ValueError("Power-law fitting requires strictly positive temperatures")
    if np.any(visc_data <= 0.0):
        raise ValueError("Power-law fitting requires strictly positive viscosities")

    log_design = np.column_stack([np.ones_like(T_data), np.log(T_data)])
    log_theta, _, _, _ = np.linalg.lstsq(log_design, np.log(visc_data), rcond=None)
    initial_guess = np.array([np.exp(log_theta[0]), log_theta[1]], dtype=float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            theta, covariance = curve_fit(
                power_law_model,
                T_data,
                visc_data,
                p0=initial_guess,
                bounds=([0.0, -np.inf], [np.inf, np.inf]),
                maxfev=100000,
            )
        fit_method = "nonlinear_least_squares"
    except RuntimeError:
        theta = initial_guess
        covariance = np.full((2, 2), np.nan)
        fit_method = "log_linear_fallback"

    jacobian = _power_law_jacobian(T_data, theta[0], theta[1])
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    rank = np.linalg.matrix_rank(jacobian)

    return theta, covariance, rank, singular_values, fit_method


def make_power_law_basis():
    return {
        "name": "power_law_A*T^B",
        "kind": "power_law",
        "parameter_labels": ("A", "B"),
    }


def build_default_model_sweep(t_ref, max_poly_order=5):
    models = [
        make_polynomial_basis(order, t_ref=t_ref)
        for order in range(max_poly_order + 1)
    ]
    models.append(make_power_law_basis())
    return models


def _safe_r_squared(y_true, y_pred):
    total_sum_squares = np.sum((y_true - np.mean(y_true)) ** 2)
    if total_sum_squares <= 0.0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / total_sum_squares


def evaluate_model(T_data, model, params):
    if model["kind"] == "linear_basis":
        return evaluate_linear_basis_model(T_data, model["basis_funcs"], params)
    if model["kind"] == "power_law":
        return power_law_model(T_data, *params)
    raise ValueError(f"Unsupported model kind: {model['kind']}")


def summarize_model_fit(T_data, visc_data, model):
    T_data = _as_1d_float_array(T_data, "T_data")
    visc_data = _as_1d_float_array(visc_data, "visc_data")

    if T_data.shape != visc_data.shape:
        raise ValueError("T_data and visc_data must have the same shape")

    if model["kind"] == "linear_basis":
        theta, residuals, rank, singular_values = fit_linear_basis_viscosity(
            T_data=T_data,
            visc_data=visc_data,
            basis_funcs=model["basis_funcs"],
        )
        fit_note = "linear_least_squares"
    elif model["kind"] == "power_law":
        theta, covariance, rank, singular_values, fit_note = fit_power_law_model(
            T_data=T_data,
            visc_data=visc_data,
        )
        residuals = np.asarray([], dtype=float)
    else:
        raise ValueError(f"Unsupported model kind: {model['kind']}")

    visc_fit = evaluate_model(T_data, model, theta)
    visc_error = visc_fit - visc_data
    visc_scale = _safe_scale(visc_data)

    weighted_rss = np.sum((visc_error / visc_scale) ** 2)
    residual_rss = np.sum(visc_error**2)
    if residuals.size == 0:
        residuals = np.array([residual_rss], dtype=float)

    n_points = T_data.size
    n_params = theta.size
    dof = max(n_points - n_params, 0)
    rss_per_dof = weighted_rss / dof if dof > 0 else np.nan

    if weighted_rss > 0.0:
        aic = n_points * np.log(weighted_rss / n_points) + 2 * n_params
        bic = n_points * np.log(weighted_rss / n_points) + n_params * np.log(n_points)
    else:
        aic = -np.inf
        bic = -np.inf

    condition_number = np.inf
    if singular_values.size > 0 and singular_values[-1] > 0.0:
        condition_number = singular_values[0] / singular_values[-1]

    return {
        "name": model["name"],
        "kind": model["kind"],
        "model": model,
        "theta": theta,
        "residuals": residuals,
        "rank": rank,
        "singular_values": singular_values,
        "n_params": n_params,
        "weighted_rss": weighted_rss,
        "rss_per_dof": rss_per_dof,
        "aic": aic,
        "bic": bic,
        "condition_number": condition_number,
        "visc_rmse": np.sqrt(np.mean(visc_error**2)),
        "visc_mae": np.mean(np.abs(visc_error)),
        "visc_max_abs": np.max(np.abs(visc_error)),
        "visc_r2": _safe_r_squared(visc_data, visc_fit),
        "visc_fit": visc_fit,
        "fit_note": fit_note,
    }


def sweep_model_fits(T_data, visc_data, models):
    results = [summarize_model_fit(T_data=T_data, visc_data=visc_data, model=model) for model in models]
    return sorted(results, key=lambda result: (result["weighted_rss"], result["n_params"]))


def format_sweep_table(results):
    header = (
        f"{'model':<18} {'kind':<12} {'n_par':>5} {'rank':>4} {'rss_scaled':>12} "
        f"{'rmse':>10} {'max_abs':>10} {'r2':>10} {'cond':>12}"
    )
    divider = "-" * len(header)

    lines = [header, divider]
    for result in results:
        lines.append(
            f"{result['name']:<18} "
            f"{result['kind']:<12} "
            f"{result['n_params']:>5d} "
            f"{result['rank']:>4d} "
            f"{result['weighted_rss']:>12.5g} "
            f"{result['visc_rmse']:>10.5g} "
            f"{result['visc_max_abs']:>10.5g} "
            f"{result['visc_r2']:>10.5g} "
            f"{result['condition_number']:>12.5g}"
        )
    return "\n".join(lines)


def print_fit_details(result, title):
    print(title)
    print(f"  model           = {result['name']}")
    print(f"  kind            = {result['kind']}")
    print(f"  theta           = {result['theta']}")
    if result["kind"] == "power_law":
        print(f"  A               = {result['theta'][0]:.12g}")
        print(f"  B               = {result['theta'][1]:.12g}")
    print(f"  fit_method      = {result['fit_note']}")
    print(f"  weighted_rss    = {result['weighted_rss']:.6g}")
    print(f"  rss_per_dof     = {result['rss_per_dof']:.6g}")
    print(f"  visc_rmse       = {result['visc_rmse']:.6g}")
    print(f"  visc_max_abs    = {result['visc_max_abs']:.6g}")
    print(f"  visc_r2         = {result['visc_r2']:.6g}")
    print(f"  rank            = {result['rank']}")
    print(f"  condition_num   = {result['condition_number']:.6g}")
    print(f"  AIC             = {result['aic']:.6g}")
    print(f"  BIC             = {result['bic']:.6g}")


def plot_fit_comparison(
    T_data,
    visc_data,
    primary_result,
    all_results=None,
    plot_all_models=True,
    secondary_result=None,
):
    T_plot = np.linspace(np.min(T_data), np.max(T_data), num=2000)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    ax1.scatter(T_data, visc_data, label="Original", zorder=3)
    ax1.set_ylabel("Viscosity [Pa*s]")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    if plot_all_models:
        if all_results is None:
            raise ValueError("all_results is required when plot_all_models=True")

        for result in all_results:
            curve = evaluate_model(T_plot, result["model"], result["theta"])
            line_width = 2.5 if result["name"] == primary_result["name"] else 1.1
            alpha = 0.95 if result["name"] == primary_result["name"] else 0.55
            label = (
                f"Best RSS: {result['name']}"
                if result["name"] == primary_result["name"]
                else result["name"]
            )
            ax1.plot(T_plot, curve, linewidth=line_width, alpha=alpha, label=label)
    else:
        primary_curve = evaluate_model(T_plot, primary_result["model"], primary_result["theta"])
        ax1.plot(T_plot, primary_curve, label=f"Best RSS: {primary_result['name']}")

    if secondary_result is not None and secondary_result["name"] != primary_result["name"]:
        secondary_curve = evaluate_model(
            T_plot,
            secondary_result["model"],
            secondary_result["theta"],
        )
        ax1.plot(
            T_plot,
            secondary_curve,
            "--",
            linewidth=2.0,
            label=f"Best BIC: {secondary_result['name']}",
        )

    ax2.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax2.scatter(
        T_data,
        primary_result["visc_fit"] - visc_data,
        label=f"Residuals: {primary_result['name']}",
        s=22,
    )
    if secondary_result is not None and secondary_result["name"] != primary_result["name"]:
        ax2.scatter(
            T_data,
            secondary_result["visc_fit"] - visc_data,
            label=f"Residuals: {secondary_result['name']}",
            s=22,
            marker="x",
        )

    ax2.set_xlabel("Temperature [K]")
    ax2.set_ylabel("Residual [Pa*s]")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax1.legend(loc="best", fontsize="small")
    ax2.legend(loc="best", fontsize="small")
    plt.tight_layout()

    # if "agg" in plt.get_backend().lower():
    #     plt.close(fig)
    #     return

    plt.show()


def main():
    T_data, visc_data, data_dir = load_viscosity_data(species)
    T_data = _as_1d_float_array(T_data, "T_data")
    visc_data = _as_1d_float_array(visc_data, "visc_data")

    if T_data.shape != visc_data.shape:
        raise ValueError("T_data and visc_data must have the same shape")

    poly_reference_temperature = float(np.mean(T_data))
    models = build_default_model_sweep(
        t_ref=poly_reference_temperature,
        max_poly_order=max_poly_order,
    )
    results = sweep_model_fits(
        T_data=T_data,
        visc_data=visc_data,
        models=models,
    )

    best_rss = results[0]
    best2nd_rss = results[1] if len(results) > 1 else results[0]
    best_bic = min(results, key=lambda result: result["bic"])

    print(f"Species: {species}")
    print(f"Data directory: {data_dir}")
    print(f"Polynomial reference temperature = {poly_reference_temperature:.6g} K")
    print()
    print("Model sweep summary")
    print(format_sweep_table(results))
    print()
    print_fit_details(best_rss, "Best weighted residual fit")
    print()
    print_fit_details(best2nd_rss, "Second best weighted residual fit")
    print()
    print_fit_details(best_bic, "Best BIC fit")

    plot_fit_comparison(
        T_data,
        visc_data,
        best_rss,
        all_results=results,
        plot_all_models=plot_all_models,
        secondary_result=best_bic,
    )


if __name__ == "__main__":
    main()
