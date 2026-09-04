import argparse
from pathlib import Path

if __package__:
    from ._fit import _as_1d_float_array, _evaluate_basis_matrix, _safe_r_squared, evaluate_linear_basis_model
else:
    from _fit import _as_1d_float_array, _evaluate_basis_matrix, _safe_r_squared, evaluate_linear_basis_model


import numpy as np
import matplotlib.pyplot as plt

def fit_linear_basis_enthcp(
    T_data,
    h_data,
    cp_data,
    cp_basis_funcs,
    h_basis_funcs,
    h_ref,
    h_weight=1.0,
    cp_weight=1.0,
):
    T_data = _as_1d_float_array(T_data, "T_data")
    h_data = _as_1d_float_array(h_data, "h_data")
    cp_data = _as_1d_float_array(cp_data, "cp_data")

    if not (T_data.shape == h_data.shape == cp_data.shape):
        raise ValueError("T_data, h_data, and cp_data must have the same shape")
    if len(cp_basis_funcs) != len(h_basis_funcs):
        raise ValueError("cp_basis_funcs and h_basis_funcs must have the same length")

    h_scale = max(np.std(h_data), 1.0)
    cp_scale = max(np.std(cp_data), 1.0)

    A_cp = _evaluate_basis_matrix(T_data, cp_basis_funcs, "cp_basis_funcs")
    A_h = _evaluate_basis_matrix(T_data, h_basis_funcs, "h_basis_funcs")

    b_cp = cp_data
    b_h = h_data - h_ref

    A_cp_scaled = (cp_weight / cp_scale) * A_cp
    b_cp_scaled = (cp_weight / cp_scale) * b_cp
    A_h_scaled = (h_weight / h_scale) * A_h
    b_h_scaled = (h_weight / h_scale) * b_h

    A = np.vstack([A_cp_scaled, A_h_scaled])
    b = np.concatenate([b_cp_scaled, b_h_scaled])

    theta, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    return theta, residuals, rank, singular_values


def _poly_cp_basis(power, t_ref):
    def basis_func(T_data):
        dT = np.asarray(T_data, dtype=float) - t_ref
        return dT ** power

    return basis_func


def _poly_h_basis(power, t_ref):
    def basis_func(T_data):
        dT = np.asarray(T_data, dtype=float) - t_ref
        return dT ** (power + 1) / (power + 1)

    return basis_func


def make_polynomial_basis(order, t_ref):
    if order < 0:
        raise ValueError("Polynomial order must be non-negative")

    return {
        "name": f"poly_dT_order_{order}",
        "cp_basis_funcs": [_poly_cp_basis(power, t_ref) for power in range(order + 1)],
        "h_basis_funcs": [_poly_h_basis(power, t_ref) for power in range(order + 1)],
    }


def make_shomate_basis(t_ref):
    tau_ref = t_ref / 1000.0

    return {
        "name": "shomate",
        "cp_basis_funcs": [
            lambda T_data: np.ones_like(np.asarray(T_data, dtype=float)),
            lambda T_data: np.asarray(T_data, dtype=float) / 1000.0,
            lambda T_data: (np.asarray(T_data, dtype=float) / 1000.0) ** 2,
            lambda T_data: (np.asarray(T_data, dtype=float) / 1000.0) ** 3,
            lambda T_data: (1000.0 / np.asarray(T_data, dtype=float)) ** 2,
        ],
        "h_basis_funcs": [
            lambda T_data: np.asarray(T_data, dtype=float) - t_ref,
            lambda T_data: 500.0
            * ((np.asarray(T_data, dtype=float) / 1000.0) ** 2 - tau_ref**2),
            lambda T_data: (1000.0 / 3.0)
            * ((np.asarray(T_data, dtype=float) / 1000.0) ** 3 - tau_ref**3),
            lambda T_data: 250.0
            * ((np.asarray(T_data, dtype=float) / 1000.0) ** 4 - tau_ref**4),
            lambda T_data: -1000.0
            * (
                1.0 / (np.asarray(T_data, dtype=float) / 1000.0)
                - 1.0 / tau_ref
            ),
        ],
    }


def make_log_reciprocal_basis(t_ref):
    return {
        "name": "log_reciprocal",
        "cp_basis_funcs": [
            lambda T_data: np.ones_like(np.asarray(T_data, dtype=float)),
            lambda T_data: np.asarray(T_data, dtype=float) - t_ref,
            lambda T_data: (np.asarray(T_data, dtype=float) - t_ref) ** 2,
            lambda T_data: 1.0 / np.asarray(T_data, dtype=float),
            lambda T_data: 1.0 / np.asarray(T_data, dtype=float) ** 2,
        ],
        "h_basis_funcs": [
            lambda T_data: np.asarray(T_data, dtype=float) - t_ref,
            lambda T_data: 0.5 * (np.asarray(T_data, dtype=float) - t_ref) ** 2,
            lambda T_data: ((np.asarray(T_data, dtype=float) - t_ref) ** 3) / 3.0,
            lambda T_data: np.log(np.asarray(T_data, dtype=float) / t_ref),
            lambda T_data: -(1.0 / np.asarray(T_data, dtype=float) - 1.0 / t_ref),
        ],
    }


def build_default_basis_sweep(t_ref, max_poly_order=5):
    bases = [make_polynomial_basis(order, t_ref) for order in range(max_poly_order + 1)]
    #bases.append(make_shomate_basis(t_ref))
    #bases.append(make_log_reciprocal_basis(t_ref))
    return bases


def summarize_basis_fit(
    T_data,
    h_data,
    cp_data,
    basis,
    h_ref,
    h_weight=1.0,
    cp_weight=1.0,
):
    theta, residuals, rank, singular_values = fit_linear_basis_enthcp(
        T_data=T_data,
        h_data=h_data,
        cp_data=cp_data,
        cp_basis_funcs=basis["cp_basis_funcs"],
        h_basis_funcs=basis["h_basis_funcs"],
        h_ref=h_ref,
        h_weight=h_weight,
        cp_weight=cp_weight,
    )

    T_data = _as_1d_float_array(T_data, "T_data")
    h_data = _as_1d_float_array(h_data, "h_data")
    cp_data = _as_1d_float_array(cp_data, "cp_data")

    cp_fit = evaluate_linear_basis_model(T_data, basis["cp_basis_funcs"], theta)
    h_fit = evaluate_linear_basis_model(T_data, basis["h_basis_funcs"], theta, offset=h_ref)

    cp_error = cp_fit - cp_data
    h_error = h_fit - h_data

    h_scale = max(np.std(h_data), 1.0)
    cp_scale = max(np.std(cp_data), 1.0)
    weighted_rss = np.sum(((cp_weight / cp_scale) * cp_error) ** 2) + np.sum(
        ((h_weight / h_scale) * h_error) ** 2
    )

    n_points = 2 * T_data.size
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
        "name": basis["name"],
        "basis": basis,
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
        "cp_rmse": np.sqrt(np.mean(cp_error**2)),
        "cp_mae": np.mean(np.abs(cp_error)),
        "cp_max_abs": np.max(np.abs(cp_error)),
        "cp_r2": _safe_r_squared(cp_data, cp_fit),
        "h_rmse": np.sqrt(np.mean(h_error**2)),
        "h_mae": np.mean(np.abs(h_error)),
        "h_max_abs": np.max(np.abs(h_error)),
        "h_r2": _safe_r_squared(h_data, h_fit),
        "cp_fit": cp_fit,
        "h_fit": h_fit,
    }


def sweep_basis_fits(
    T_data,
    h_data,
    cp_data,
    bases,
    h_ref,
    h_weight=1.0,
    cp_weight=1.0,
):
    results = [
        summarize_basis_fit(
            T_data=T_data,
            h_data=h_data,
            cp_data=cp_data,
            basis=basis,
            h_ref=h_ref,
            h_weight=h_weight,
            cp_weight=cp_weight,
        )
        for basis in bases
    ]
    return sorted(results, key=lambda result: (result["weighted_rss"], result["n_params"]))


def format_sweep_table(results):
    header = (
        f"{'basis':<18} {'n_par':>5} {'rank':>4} {'rss_scaled':>12} "
        f"{'cp_rmse':>10} {'h_rmse':>10} {'cp_max':>10} {'h_max':>10} {'cond':>12}"
    )
    divider = "-" * len(header)

    lines = [header, divider]
    for result in results:
        lines.append(
            f"{result['name']:<18} "
            f"{result['n_params']:>5d} "
            f"{result['rank']:>4d} "
            f"{result['weighted_rss']:>12.5g} "
            f"{result['cp_rmse']:>10.5g} "
            f"{result['h_rmse']:>10.5g} "
            f"{result['cp_max_abs']:>10.5g} "
            f"{result['h_max_abs']:>10.5g} "
            f"{result['condition_number']:>12.5g}"
        )
    return "\n".join(lines)


def print_fit_details(result, title):
    print(title)
    print(f"  basis           = {result['name']}")
    print(f"  theta           = {result['theta']}")
    print(f"  weighted_rss    = {result['weighted_rss']:.6g}")
    print(f"  rss_per_dof     = {result['rss_per_dof']:.6g}")
    print(f"  cp_rmse         = {result['cp_rmse']:.6g}")
    print(f"  cp_max_abs      = {result['cp_max_abs']:.6g}")
    print(f"  cp_r2           = {result['cp_r2']:.6g}")
    print(f"  h_rmse          = {result['h_rmse']:.6g}")
    print(f"  h_max_abs       = {result['h_max_abs']:.6g}")
    print(f"  h_r2            = {result['h_r2']:.6g}")
    print(f"  rank            = {result['rank']}")
    print(f"  condition_num   = {result['condition_number']:.6g}")
    print(f"  AIC             = {result['aic']:.6g}")
    print(f"  BIC             = {result['bic']:.6g}")


def plot_fit_comparison(T_data, h_data, cp_data, results, h_ref):
    temperatures = np.linspace(np.min(T_data), np.max(T_data), num=2000)
    figure, axes = plt.subplots(2, 1, sharex=True)
    best_rss, best_bic = results[0], min(results, key=lambda result: result["bic"])
    for axis, data, basis_key, offset, label in (
        (axes[0], cp_data, "cp_basis_funcs", 0.0, "Cp [J/(mol*K)]"),
        (axes[1], h_data, "h_basis_funcs", h_ref, "H [J/mol]"),
    ):
        axis.scatter(T_data, data, label="Original")
        for result in results:
            curve = evaluate_linear_basis_model(
                temperatures, result["basis"][basis_key], result["theta"], offset=offset,
            )
            axis.plot(temperatures, curve, label=result["name"],
                      linewidth=2.5 if result is best_rss else 1.1,
                      linestyle="--" if result is best_bic and result is not best_rss else "-")
        axis.set_ylabel(label)
        axis.legend(loc="best", fontsize="small")
    axes[1].set_xlabel("Temperature [K]")
    figure.tight_layout()
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit heat capacity and enthalpy from species CSV data.")
    parser.add_argument("species", help="Species directory under the data root.")
    parser.add_argument("--h-ref", type=float, required=True, help="Reference enthalpy in J/mol.")
    parser.add_argument("--t-ref", type=float, default=298.15)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "enth_hcap_data")
    parser.add_argument("--output", type=Path, help="Save a figure instead of opening a window.")
    args = parser.parse_args(argv)
    if args.order < 0:
        parser.error("--order must be nonnegative")
    directory = args.data_root / args.species
    temperature, enthalpy, capacity = (
        np.genfromtxt(directory / f"{name}.csv", dtype=float)[1:] for name in ("temp", "enth", "hcap")
    )
    results = sweep_basis_fits(
        temperature, enthalpy, capacity,
        build_default_basis_sweep(t_ref=args.t_ref, max_poly_order=args.order),
        args.h_ref, h_weight=9.0, cp_weight=1.0,
    )
    print(format_sweep_table(results))
    for result, label in zip(results[:2], ("Best residual fit", "Second residual fit")):
        print_fit_details(result, label)
    print_fit_details(min(results, key=lambda item: item["bic"]), "Best BIC fit")
    figure = plot_fit_comparison(temperature, enthalpy, capacity, results, args.h_ref)
    if args.output:
        figure.savefig(args.output, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
