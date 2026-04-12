from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReportDefinition:
    id: str 
    description: str
    variable_name: str | None = None # actual name inside the DAETools model definition


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
}
