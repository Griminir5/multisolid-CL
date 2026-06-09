from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from packed_bed.axial_schemes import SUPPORTED_SCHEMES

from .validators import (
    ConfigString,
    FrozenConfigModel,
    NonNegativeFloat,
    PositiveFloat,
    UniqueStringTuple,
)


class ReferencesConfig(FrozenConfigModel):
    chemistry_file: ConfigString
    program_file: ConfigString
    solids_file: ConfigString


class SimulationConfig(FrozenConfigModel):
    system_name: ConfigString
    time_horizon_s: PositiveFloat
    reporting_interval_s: PositiveFloat
    repeat_program: bool = False
    mass_scheme: ConfigString
    heat_scheme: ConfigString
    report_time_derivatives: bool

    @field_validator("mass_scheme", "heat_scheme")
    @classmethod
    def validate_scheme(cls, value: str) -> str:
        if value not in SUPPORTED_SCHEMES:
            raise ValueError(f"must be one of: {', '.join(SUPPORTED_SCHEMES)}.")
        return value


class ModelConfig(FrozenConfigModel):
    bed_length_m: PositiveFloat
    bed_radius_m: PositiveFloat
    axial_cells: int = Field(ge=1)
    ambient_temperature_k: PositiveFloat = 873.15
    heat_transfer_coefficient_w_per_m2_k: NonNegativeFloat = 100.0


class SolverConfig(FrozenConfigModel):
    name: Literal[
        "trilinos_klu",
        "trilinos_umfpack",
        "trilinos_lapack",
        "trilinos_aztecoo",
        "trilinos_aztecoo_ifpack",
        "trilinos_aztecoo_ml",
        "superlu",
        "superlu_mt",
        "intel_pardiso",
    ] = "trilinos_klu"
    threads: int = Field(default=0, ge=0)
    relative_tolerance: PositiveFloat


class OutputConfig(FrozenConfigModel):
    directory: ConfigString
    artifacts_directory: ConfigString
    requested_reports: UniqueStringTuple
    solver_incidence_matrix: bool = False


class RunConfig(FrozenConfigModel):
    references: ReferencesConfig
    simulation: SimulationConfig
    model: ModelConfig
    solver: SolverConfig
    outputs: OutputConfig

    @model_validator(mode="after")
    def validate_reporting_window(self) -> "RunConfig":
        if self.simulation.reporting_interval_s > self.simulation.time_horizon_s:
            raise ValueError("reporting_interval_s must not exceed time_horizon_s.")
        return self
