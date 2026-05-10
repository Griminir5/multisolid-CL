from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReportDefinition:
    id: str
    description: str
    variable_name: str | None = None  # Actual name inside the DAETools model definition.


REPORT_VARIABLE_REGISTRY = {
    "temperature": ReportDefinition(
        id="temperature",
        description="Bed temperature by cell center.",
        variable_name="temp_bed",
    ),
    "pressure": ReportDefinition(
        id="pressure",
        description="Bed pressure by cell center.",
        variable_name="pres_bed",
    ),
    "velocity": ReportDefinition(
        id="velocity",
        description="Face superficial velocity by cell face.",
        variable_name="u_s",
    ),
    "gas_concentration": ReportDefinition(
        id="gas_concentration",
        description="Gas concentration by species and cell center.",
        variable_name="c_gas",
    ),
    "gas_mole_fraction": ReportDefinition(
        id="gas_mole_fraction",
        description="Gas mole fraction by species and cell center.",
        variable_name="y_gas",
    ),
    "solid_concentration": ReportDefinition(
        id="solid_concentration",
        description="Solid concentration by species and cell center.",
        variable_name="c_sol",
    ),
    "solid_mole_fraction": ReportDefinition(
        id="solid_mole_fraction",
        description="Solid mole fraction by species and cell center.",
        variable_name="y_sol",
    ),
    "gas_flux": ReportDefinition(
        id="gas_flux",
        description="Gas molar flux by species and face.",
        variable_name="N_gas_face",
    ),
    "gas_source": ReportDefinition(
        id="gas_source",
        description="Net gas-phase source term by species and cell center.",
        variable_name="S_gas",
    ),
    "solid_source": ReportDefinition(
        id="solid_source",
        description="Net solid-phase source term by species and cell center.",
        variable_name="S_sol",
    ),
    "reaction_rate": ReportDefinition(
        id="reaction_rate",
        description="Reaction rate by reaction and cell center.",
        variable_name="R_rxn",
    ),
    "gas_enthalpy_flux": ReportDefinition(
        id="gas_enthalpy_flux",
        description="Gas enthalpy flux by species and face.",
        variable_name="J_gas_face",
    ),
    "heat_balance": ReportDefinition(
        id="heat_balance",
        description="Heat balance totals and error over time.",
    ),
    "mass_balance": ReportDefinition(
        id="mass_balance",
        description="Mass balance totals and error over time.",
    ),
}


DERIVED_REPORT_VARIABLE_NAMES = {
    "heat_balance": (
        "heat_in_total",
        "heat_out_total",
        "heat_loss_total",
        "heat_bed_total",
    ),
    "mass_balance": (
        "mass_in_total",
        "mass_out_total",
        "mass_bed_total",
    ),
}

PLOT_REPORT_IDS = (
    "temperature",
    "pressure",
    "gas_mole_fraction",
    "gas_flux",
)

PLOT_EXTRA_VARIABLE_NAMES = (
    "P_in",
    "P_out",
)

PLOT_EXTRA_PARAMETER_NAMES = (
    "xval_cells",
)

BENCHMARK_SNAPSHOT_VARIABLE_NAMES = (
    "temp_bed",
    "pres_bed",
    "y_gas",
    "c_sol",
    "heat_in_total",
    "heat_out_total",
    "heat_loss_total",
    "heat_bed_total",
    "mass_in_total",
    "mass_out_total",
    "mass_bed_total",
    "material_in_total",
    "material_out_total",
    "material_bed_total",
)


def report_variable_names(report_ids) -> tuple[str, ...]:
    variable_names: list[str] = []
    for report_id in report_ids:
        definition = REPORT_VARIABLE_REGISTRY[report_id]
        if definition.variable_name is not None:
            variable_names.append(definition.variable_name)
        variable_names.extend(DERIVED_REPORT_VARIABLE_NAMES.get(report_id, ()))
    return tuple(dict.fromkeys(variable_names))


def reporting_targets(
    report_ids,
    *,
    include_plot_variables: bool = False,
    include_benchmark_snapshot: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    expanded_report_ids = list(report_ids)
    variable_names: list[str] = []
    parameter_names: list[str] = []

    if include_plot_variables:
        expanded_report_ids.extend(PLOT_REPORT_IDS)
        variable_names.extend(PLOT_EXTRA_VARIABLE_NAMES)
        parameter_names.extend(PLOT_EXTRA_PARAMETER_NAMES)

    variable_names.extend(report_variable_names(expanded_report_ids))

    if include_benchmark_snapshot:
        variable_names.extend(BENCHMARK_SNAPSHOT_VARIABLE_NAMES)

    return (
        tuple(dict.fromkeys(variable_names)),
        tuple(dict.fromkeys(parameter_names)),
    )


__all__ = [
    "BENCHMARK_SNAPSHOT_VARIABLE_NAMES",
    "DERIVED_REPORT_VARIABLE_NAMES",
    "PLOT_EXTRA_PARAMETER_NAMES",
    "PLOT_EXTRA_VARIABLE_NAMES",
    "PLOT_REPORT_IDS",
    "REPORT_VARIABLE_REGISTRY",
    "ReportDefinition",
    "report_variable_names",
    "reporting_targets",
]
