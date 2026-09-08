from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from packed_bed.axial_schemes import SUPPORTED_SCHEMES


def _require_string(value: str) -> str:
    if value == "" or value != value.strip():
        raise ValueError("must not be blank or padded with whitespace.")
    return value


def _require_unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"contains duplicates: {', '.join(duplicates)}.")
    return values


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("must be provided as a YAML sequence.")
    return tuple(value)


def _require_fraction_mapping(mapping: dict[str, float]) -> dict[str, float]:
    total = sum(mapping.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"must sum to 1.0 exactly, got {total:.16g}.")
    return mapping


class ConfigModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


ConfigString = Annotated[str, AfterValidator(_require_string)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
UnitFraction = Annotated[float, Field(gt=0.0, lt=1.0, allow_inf_nan=False)]
UniqueStringTuple = Annotated[
    tuple[ConfigString, ...],
    BeforeValidator(_as_tuple),
    AfterValidator(_require_unique_strings),
]
NonEmptyUniqueStringTuple = Annotated[
    tuple[ConfigString, ...],
    BeforeValidator(_as_tuple),
    Field(min_length=1),
    AfterValidator(_require_unique_strings),
]
FractionMapping = Annotated[
    dict[ConfigString, NonNegativeFloat],
    Field(min_length=1),
    AfterValidator(_require_fraction_mapping),
]


class ChemistryConfig(ConfigModel):
    gas_species: NonEmptyUniqueStringTuple
    reaction_families: UniqueStringTuple
    reaction_ids: UniqueStringTuple


class HoldStep(ConfigModel):
    kind: Literal["hold"]
    duration_s: PositiveFloat


class ScalarRampStep(ConfigModel):
    kind: Literal["ramp"]
    duration_s: PositiveFloat
    target: PositiveFloat


class CompositionRampStep(ConfigModel):
    kind: Literal["ramp"]
    duration_s: PositiveFloat
    target: FractionMapping


ScalarStep = Annotated[HoldStep | ScalarRampStep, Field(discriminator="kind")]
CompositionStep = Annotated[HoldStep | CompositionRampStep, Field(discriminator="kind")]
ScalarSteps = Annotated[tuple[ScalarStep, ...], BeforeValidator(_as_tuple)]
CompositionSteps = Annotated[tuple[CompositionStep, ...], BeforeValidator(_as_tuple)]


class ScalarChannelConfig(ConfigModel):
    initial: PositiveFloat
    steps: ScalarSteps = Field(default_factory=tuple)


class InletFlowConfig(ScalarChannelConfig):
    basis: Literal["mol_per_s", "ghsv_per_h"] = "mol_per_s"


class CompositionChannelConfig(ConfigModel):
    initial: FractionMapping
    steps: CompositionSteps = Field(default_factory=tuple)


class ProgramConfig(ConfigModel):
    inlet_flow: InletFlowConfig
    inlet_temperature: ScalarChannelConfig
    outlet_pressure: ScalarChannelConfig
    inlet_composition: CompositionChannelConfig


class FeedState(ConfigModel):
    flow: PositiveFloat
    temperature: PositiveFloat
    composition: FractionMapping


class FeedRampTarget(ConfigModel):
    flow: PositiveFloat | None = None
    temperature: PositiveFloat | None = None
    composition: FractionMapping | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "FeedRampTarget":
        if self.flow is None and self.temperature is None and self.composition is None:
            raise ValueError("must specify at least one of flow, temperature, or composition; use hold otherwise.")
        return self


class FeedRampStep(ConfigModel):
    kind: Literal["ramp"]
    duration_s: PositiveFloat
    target: FeedRampTarget


FeedStep = Annotated[HoldStep | FeedRampStep, Field(discriminator="kind")]


class FeedStreamConfig(ConfigModel):
    basis: Literal["mol_per_s", "ghsv_per_h"] = "mol_per_s"
    initial: FeedState
    steps: Annotated[tuple[FeedStep, ...], BeforeValidator(_as_tuple)] = Field(default_factory=tuple)


class FeedProgramConfig(ConfigModel):
    feed_stream: FeedStreamConfig
    outlet_pressure: ScalarChannelConfig


class ReferencesConfig(ConfigModel):
    chemistry_file: ConfigString
    program_file: ConfigString
    solids_file: ConfigString


class SimulationConfig(ConfigModel):
    system_name: ConfigString
    time_horizon_s: PositiveFloat
    reporting_interval_s: PositiveFloat
    repeat_program: bool = False
    program_mode: Literal["separate_channels", "feed_stream"] = "separate_channels"
    interior_flow_mode: Literal["forward_only", "reversible"] = "forward_only"
    mass_scheme: ConfigString
    heat_scheme: ConfigString
    report_time_derivatives: bool

    @field_validator("mass_scheme", "heat_scheme")
    @classmethod
    def validate_scheme(cls, value: str) -> str:
        if value not in SUPPORTED_SCHEMES:
            raise ValueError(f"must be one of: {', '.join(SUPPORTED_SCHEMES)}.")
        return value


class ModelConfig(ConfigModel):
    bed_length_m: PositiveFloat
    bed_radius_m: PositiveFloat
    axial_cells: int = Field(ge=3)
    gas_voidage_mode: Literal["bed_only", "bed_and_particle"] = "bed_and_particle"
    ambient_temperature_k: PositiveFloat = 873.15
    heat_transfer_coefficient_w_per_m2_k: NonNegativeFloat = 100.0


class SolverConfig(ConfigModel):
    backend: Literal["daetools", "compiled"] = "daetools"
    name: Literal[
        "trilinos_klu",
        "trilinos_umfpack",
        "trilinos_lapack",
        "trilinos_aztecoo",
        "trilinos_aztecoo_ifpack",
        "trilinos_aztecoo_ml",
        "sundials_gmres_ifpack",
        "superlu",
        "superlu_mt",
        "band",
        "intel_pardiso",
    ] = "trilinos_klu"
    threads: int = Field(default=0, ge=0)
    relative_tolerance: PositiveFloat
    concentration_absolute_tolerance: PositiveFloat = 1.0e-5
    suppress_algebraic_errors: bool = False
    max_nonlinear_iterations: int = Field(default=4, ge=1)
    nonlinear_convergence_coefficient: PositiveFloat = 0.33
    maximum_order: int = Field(default=5, ge=1, le=5)
    scale_residuals: bool = False
    step_growth_threshold: float = Field(default=2.0, ge=1.0, allow_inf_nan=False)
    nonlinear_refresh_interval: int = Field(default=0, ge=0, strict=True)
    vector_exponentials: bool = Field(default=False, strict=True)
    band_reciprocals: bool = Field(default=False, strict=True)


class OutputConfig(ConfigModel):
    directory: ConfigString
    artifacts_directory: ConfigString
    requested_reports: UniqueStringTuple
    requested_plots: UniqueStringTuple
    solver_incidence_matrix: bool = False


class RunConfig(ConfigModel):
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

    @model_validator(mode="after")
    def validate_compiled_backend(self) -> "RunConfig":
        if self.solver.band_reciprocals and (
            self.solver.backend != "compiled" or self.solver.name != "band"
        ):
            raise ValueError(
                "band_reciprocals requires solver.backend: compiled and solver.name: band."
            )
        if self.solver.vector_exponentials and self.solver.backend != "compiled":
            raise ValueError(
                "vector_exponentials currently requires solver.backend: compiled."
            )
        if self.solver.nonlinear_refresh_interval and self.solver.backend != "compiled":
            raise ValueError(
                "nonlinear_refresh_interval currently requires solver.backend: compiled."
            )
        if self.solver.name == "band" and self.solver.backend != "compiled":
            raise ValueError("solver.name: band requires solver.backend: compiled.")
        if self.solver.scale_residuals and self.solver.backend != "compiled":
            raise ValueError(
                "scale_residuals currently requires solver.backend: compiled."
            )
        if (
            self.solver.step_growth_threshold != 2.0
            and self.solver.backend != "compiled"
        ):
            raise ValueError(
                "step_growth_threshold currently requires solver.backend: compiled."
            )
        if self.solver.backend == "compiled":
            if self.solver.name not in {"superlu", "superlu_mt", "band"}:
                raise ValueError(
                    "The compiled backend requires solver.name: superlu, superlu_mt or band."
                )
            if self.simulation.report_time_derivatives:
                raise ValueError(
                    "The compiled backend does not support report_time_derivatives: true."
                )
            if self.outputs.solver_incidence_matrix:
                raise ValueError(
                    "The compiled backend does not support solver_incidence_matrix: true."
                )
        return self


class SolidZoneConfig(ConfigModel):
    x_start_m: NonNegativeFloat
    x_end_m: PositiveFloat
    e_b: UnitFraction
    e_p: UnitFraction
    d_p: PositiveFloat
    values: dict[ConfigString, NonNegativeFloat]

    @model_validator(mode="after")
    def validate_bounds(self) -> "SolidZoneConfig":
        if self.x_end_m <= self.x_start_m:
            raise ValueError("x_end_m must be greater than x_start_m.")
        if not self.values:
            raise ValueError("values must not be empty.")
        return self


SolidZones = Annotated[
    tuple[SolidZoneConfig, ...],
    BeforeValidator(_as_tuple),
    Field(min_length=1),
]


class SolidProfileConfig(ConfigModel):
    basis: Literal["solid", "bed"]
    zones: SolidZones


class SolidConfig(ConfigModel):
    solid_species: NonEmptyUniqueStringTuple
    initial_profile: SolidProfileConfig

    @model_validator(mode="after")
    def validate_zone_species(self) -> "SolidConfig":
        expected = set(self.solid_species)
        for zone_index, zone in enumerate(self.initial_profile.zones):
            actual = set(zone.values)
            if actual == expected:
                continue
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            differences = []
            if missing:
                differences.append(f"missing {', '.join(missing)}")
            if extra:
                differences.append(f"unexpected {', '.join(extra)}")
            raise ValueError(
                f"solids.initial_profile.zones[{zone_index}].values species mismatch: "
                f"{'; '.join(differences)}."
            )
        return self


__all__ = (
    "ChemistryConfig",
    "CompositionChannelConfig",
    "CompositionRampStep",
    "FeedProgramConfig",
    "FeedRampStep",
    "FeedRampTarget",
    "FeedState",
    "FeedStreamConfig",
    "HoldStep",
    "InletFlowConfig",
    "ModelConfig",
    "OutputConfig",
    "ProgramConfig",
    "ReferencesConfig",
    "RunConfig",
    "ScalarChannelConfig",
    "ScalarRampStep",
    "SimulationConfig",
    "SolidConfig",
    "SolidProfileConfig",
    "SolidZoneConfig",
    "SolverConfig",
)
