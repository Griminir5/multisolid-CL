from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import RunBundle
from .reporting import REPORT_VARIABLE_REGISTRY


def _get_reported_variable(reporter, variable_path):
    try:
        return reporter.Process.dictVariables[variable_path]
    except KeyError as exc:
        available = sorted(reporter.Process.dictVariables.keys())
        raise KeyError(
            f"Reported variable '{variable_path}' was not found. Available variables: {available}"
        ) from exc


def _flatten_variable(variable, run_bundle: RunBundle):
    values = np.asarray(variable.Values, dtype=float)
    time_values = np.asarray(variable.TimeValues, dtype=float)
    domain_names = [domain.Name for domain in variable.Domains]
    domain_points = [np.asarray(domain.Points, dtype=float) for domain in variable.Domains]

    rows = []
    if values.ndim == 1:
        for time_index, time_value in enumerate(time_values):
            rows.append({"time_s": float(time_value), "value": float(values[time_index])})
        return rows

    for time_index, time_value in enumerate(time_values):
        for point_index in np.ndindex(values.shape[1:]):
            row = {"time_s": float(time_value)}
            for axis_index, domain_name in enumerate(domain_names):
                item_index = point_index[axis_index]
                if domain_name == "Gas_comps":
                    row["species"] = run_bundle.chemistry.gas_species[item_index]
                elif domain_name == "Solid_comps":
                    row["species"] = run_bundle.chemistry.solid_species[item_index]
                elif domain_name == "Cell_centers":
                    row["x_cell_m"] = float(domain_points[axis_index][item_index])
                elif domain_name == "Cell_faces":
                    row["x_face_m"] = float(domain_points[axis_index][item_index])
                else:
                    row[domain_name] = float(domain_points[axis_index][item_index])
            row["value"] = float(values[(time_index, *point_index)])
            rows.append(row)
    return rows


def _write_rows(path, rows):
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _extract_balance_data(reporter, variable_prefix):
    material_in_var = _get_reported_variable(reporter, f"{variable_prefix}.material_in_total")
    material_out_var = _get_reported_variable(reporter, f"{variable_prefix}.material_out_total")
    material_bed_var = _get_reported_variable(reporter, f"{variable_prefix}.material_bed_total")
    heat_in_var = _get_reported_variable(reporter, f"{variable_prefix}.heat_in_total")
    heat_out_var = _get_reported_variable(reporter, f"{variable_prefix}.heat_out_total")
    heat_bed_var = _get_reported_variable(reporter, f"{variable_prefix}.heat_bed_total")

    time_values = np.asarray(material_in_var.TimeValues, dtype=float)
    material_in_total = np.asarray(material_in_var.Values, dtype=float).reshape(-1)
    material_out_total = np.asarray(material_out_var.Values, dtype=float).reshape(-1)
    material_bed_total = np.asarray(material_bed_var.Values, dtype=float).reshape(-1)
    heat_in_total = np.asarray(heat_in_var.Values, dtype=float).reshape(-1)
    heat_out_total = np.asarray(heat_out_var.Values, dtype=float).reshape(-1)
    heat_bed_total = np.asarray(heat_bed_var.Values, dtype=float).reshape(-1)

    cumulative_mass_boundary = material_out_total - material_in_total
    cumulative_heat_boundary = heat_out_total - heat_in_total
    material_balance_error = (material_bed_total - material_bed_total[0]) + cumulative_mass_boundary
    heat_balance_error = (heat_bed_total - heat_bed_total[0]) + cumulative_heat_boundary

    return {
        "time_s": time_values,
        "material_in_total_mol": material_in_total,
        "material_out_total_mol": material_out_total,
        "material_bed_total_mol": material_bed_total,
        "material_balance_error_mol": material_balance_error,
        "heat_in_total_J": heat_in_total,
        "heat_out_total_J": heat_out_total,
        "heat_bed_total_J": heat_bed_total,
        "heat_balance_error_J": heat_balance_error,
    }


def export_run_outputs(reporter, run_bundle: RunBundle, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variable_prefix = run_bundle.run.system_name

    balance_data = _extract_balance_data(reporter, variable_prefix)
    balances_path = output_dir / "balances.csv"
    balance_rows = []
    for index, time_value in enumerate(balance_data["time_s"]):
        balance_rows.append(
            {
                "time_s": float(time_value),
                "material_in_total_mol": float(balance_data["material_in_total_mol"][index]),
                "material_out_total_mol": float(balance_data["material_out_total_mol"][index]),
                "material_bed_total_mol": float(balance_data["material_bed_total_mol"][index]),
                "material_balance_error_mol": float(balance_data["material_balance_error_mol"][index]),
                "heat_in_total_J": float(balance_data["heat_in_total_J"][index]),
                "heat_out_total_J": float(balance_data["heat_out_total_J"][index]),
                "heat_bed_total_J": float(balance_data["heat_bed_total_J"][index]),
                "heat_balance_error_J": float(balance_data["heat_balance_error_J"][index]),
            }
        )
    _write_rows(balances_path, balance_rows)

    report_paths = {}
    for report_id in run_bundle.run.outputs.requested_reports:
        report_definition = REPORT_VARIABLE_REGISTRY[report_id]
        output_path = output_dir / f"{report_id}.csv"

        if report_id == "material_balance":
            rows = []
            for index, time_value in enumerate(balance_data["time_s"]):
                for metric_name in (
                    "material_in_total_mol",
                    "material_out_total_mol",
                    "material_bed_total_mol",
                    "material_balance_error_mol",
                ):
                    rows.append(
                        {
                            "time_s": float(time_value),
                            "metric": metric_name,
                            "value": float(balance_data[metric_name][index]),
                        }
                    )
            _write_rows(output_path, rows)
        elif report_id == "heat_balance":
            rows = []
            for index, time_value in enumerate(balance_data["time_s"]):
                for metric_name in (
                    "heat_in_total_J",
                    "heat_out_total_J",
                    "heat_bed_total_J",
                    "heat_balance_error_J",
                ):
                    rows.append(
                        {
                            "time_s": float(time_value),
                            "metric": metric_name,
                            "value": float(balance_data[metric_name][index]),
                        }
                    )
            _write_rows(output_path, rows)
        else:
            variable = _get_reported_variable(reporter, f"{variable_prefix}.{report_definition.variable_name}")
            _write_rows(output_path, _flatten_variable(variable, run_bundle))

        report_paths[report_id] = output_path

    summary_path = output_dir / "run_summary.csv"
    _write_rows(
        summary_path,
        [
            {"metric": "success", "value": "True"},
            {"metric": "gas_species", "value": ";".join(run_bundle.chemistry.gas_species)},
            {"metric": "solid_species", "value": ";".join(run_bundle.chemistry.solid_species)},
            {"metric": "reaction_ids", "value": ";".join(run_bundle.chemistry.reaction_ids)},
            {"metric": "time_horizon_s", "value": run_bundle.run.time_horizon_s},
            {"metric": "reporting_interval_s", "value": run_bundle.run.reporting_interval_s},
            {"metric": "requested_reports", "value": ";".join(run_bundle.run.outputs.requested_reports)},
            {
                "metric": "max_abs_material_balance_error_mol",
                "value": float(np.max(np.abs(balance_data["material_balance_error_mol"]))),
            },
            {
                "metric": "max_abs_heat_balance_error_J",
                "value": float(np.max(np.abs(balance_data["heat_balance_error_J"]))),
            },
        ],
    )

    return {
        "summary_path": summary_path,
        "balances_path": balances_path,
        "report_paths": report_paths,
    }
