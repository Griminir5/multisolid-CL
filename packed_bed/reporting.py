from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReportDefinition:
    id: str
    description: str
    variable_name: str | None = None


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
    "gas_enthalpy_flux": ReportDefinition(
        id="gas_enthalpy_flux",
        description="Gas enthalpy flux by species and face.",
        variable_name="J_gas_face",
    ),
    "material_balance": ReportDefinition(
        id="material_balance",
        description="Material balance totals and error over time.",
    ),
    "heat_balance": ReportDefinition(
        id="heat_balance",
        description="Heat balance totals and error over time.",
    ),
}

