from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Annotated, Literal

from pydantic import Field, field_validator, model_validator

from packed_bed.axial_schemes import SUPPORTED_SCHEMES
from .io import resolve_path
from .validation import PROGRAM_DURATION_SUM_ABS_TOLERANCE_S
from packed_bed.programs import (
    ProgramSegment,
    ScalarProgram,
    VectorProgram,
    compile_program_segments,
    sum_step_durations,
)
from .validators import (
    ConfigString,
    FrozenConfigModel,
    NonNegativeFloat,
    PositiveFloat,
    UnitFraction,
    _as_tuple,
    _require_exact_keys,
    _require_fraction_mapping,
    _require_nonempty_unique_strings,
    _require_unique_strings,
)


class HoldStep(FrozenConfigModel):
    kind: Literal["hold"]
    duration_s: PositiveFloat


class ScalarRampStep(FrozenConfigModel):
    kind: Literal["ramp"]
    duration_s: PositiveFloat
    target: PositiveFloat


class CompositionRampStep(FrozenConfigModel):
    kind: Literal["ramp"]
    duration_s: PositiveFloat
    target: dict[ConfigString, NonNegativeFloat]

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: dict[str, float]) -> dict[str, float]:
        return _require_fraction_mapping(value)


ScalarStep = Annotated[HoldStep | ScalarRampStep, Field(discriminator="kind")]
CompositionStep = Annotated[HoldStep | CompositionRampStep, Field(discriminator="kind")]


class ChemistryConfig(FrozenConfigModel):
    gas_species: tuple[ConfigString, ...]
    reaction_ids: tuple[ConfigString, ...]

    @field_validator("gas_species", "reaction_ids", mode="before")
    @classmethod
    def coerce_sequences(cls, value: Any) -> tuple[Any, ...]:
        return _as_tuple(value)

    @field_validator("gas_species")
    @classmethod
    def validate_gas_species(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _require_nonempty_unique_strings(value)

    @field_validator("reaction_ids")
    @classmethod
    def validate_reaction_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _require_unique_strings(value)


class SolidZoneConfig(FrozenConfigModel):
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

    @property
    def values_mol_per_m3(self) -> dict[str, float]:
        return self.values


class SolidProfileConfig(FrozenConfigModel):
    basis: Literal["solid", "bed"]
    zones: tuple[SolidZoneConfig, ...]

    @field_validator("zones", mode="before")
    @classmethod
    def coerce_zones(cls, value: Any) -> tuple[Any, ...]:
        return _as_tuple(value)

    @field_validator("zones")
    @classmethod
    def validate_zones(cls, value: tuple[SolidZoneConfig, ...]) -> tuple[SolidZoneConfig, ...]:
        if not value:
            raise ValueError("must not be empty.")
        return value


class SolidConfig(FrozenConfigModel):
    solid_species: tuple[ConfigString, ...]
    initial_profile: SolidProfileConfig

    @field_validator("solid_species", mode="before")
    @classmethod
    def coerce_solid_species(cls, value: Any) -> tuple[Any, ...]:
        return _as_tuple(value)

    @field_validator("solid_species")
    @classmethod
    def validate_solid_species(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _require_nonempty_unique_strings(value)

    @model_validator(mode="after")
    def validate_zone_species(self) -> "SolidConfig":
        for zone_index, zone in enumerate(self.initial_profile.zones):
            _require_exact_keys(
                set(zone.values),
                self.solid_species,
                f"solids.initial_profile.zones[{zone_index}].values",
            )
        return self

    @property
    def concentration_unit(self) -> str:
        return "mol_per_m3_solid" if self.initial_profile.basis == "solid" else "mol_per_m3_bed"

    @property
    def initial_profile_zones(self) -> tuple[SolidZoneConfig, ...]:
        return self.initial_profile.zones


class ScalarChannelConfig(FrozenConfigModel):
    initial: PositiveFloat
    steps: tuple[ScalarStep, ...] = Field(default_factory=tuple)

    @field_validator("steps", mode="before")
    @classmethod
    def coerce_steps(cls, value: Any) -> tuple[Any, ...]:
        return _as_tuple(value)

    def compile_program(self, *, repeat: bool = False, time_horizon: float | None = None) -> ScalarProgram:
        segments = compile_program_segments(
            self.initial,
            self.steps,
            repeat=repeat,
            time_horizon=time_horizon,
            resolve_next_value=lambda _step_index, step, current_value: (
                current_value if isinstance(step, HoldStep) else step.target
            ),
        )
        return ScalarProgram(initial_value=self.initial, segments=segments)


class CompositionChannelConfig(FrozenConfigModel):
    initial: dict[ConfigString, NonNegativeFloat]
    steps: tuple[CompositionStep, ...] = Field(default_factory=tuple)

    @field_validator("steps", mode="before")
    @classmethod
    def coerce_steps(cls, value: Any) -> tuple[Any, ...]:
        return _as_tuple(value)

    @field_validator("initial")
    @classmethod
    def validate_initial(cls, value: dict[str, float]) -> dict[str, float]:
        return _require_fraction_mapping(value)

    def compile_program(
        self,
        species_order: tuple[str, ...],
        *,
        repeat: bool = False,
        time_horizon: float | None = None,
    ) -> VectorProgram:
        _require_exact_keys(set(self.initial), species_order, "program.inlet_composition.initial")
        initial_value = tuple(self.initial[species_id] for species_id in species_order)

        def resolve_next_value(step_index: int, step: HoldStep | CompositionRampStep, current_value: tuple[float, ...]):
            if isinstance(step, HoldStep):
                return current_value

            _require_exact_keys(
                set(step.target),
                species_order,
                f"program.inlet_composition.steps[{step_index}].target",
            )
            return tuple(step.target[species_id] for species_id in species_order)

        segments = compile_program_segments(
            initial_value,
            self.steps,
            repeat=repeat,
            time_horizon=time_horizon,
            resolve_next_value=resolve_next_value,
        )

        return VectorProgram(
            initial_value=initial_value,
            segments=segments,
        )


class ProgramConfig(FrozenConfigModel):
    inlet_flow: ScalarChannelConfig
    inlet_temperature: ScalarChannelConfig
    outlet_pressure: ScalarChannelConfig
    inlet_composition: CompositionChannelConfig


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
    axial_cells: int
    ambient_temperature_k: PositiveFloat = 873.15
    heat_transfer_coefficient_w_per_m2_k: NonNegativeFloat = 100.0

    @field_validator("axial_cells")
    @classmethod
    def validate_axial_cells(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be at least 1.")
        return value


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
    requested_reports: tuple[ConfigString, ...]
    solver_incidence_matrix: bool = False

    @field_validator("requested_reports", mode="before")
    @classmethod
    def coerce_requested_reports(cls, value: Any) -> tuple[Any, ...]:
        return _as_tuple(value)

    @field_validator("requested_reports")
    @classmethod
    def validate_requested_reports(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _require_unique_strings(value)


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

    @property
    def system_name(self) -> str:
        return self.simulation.system_name

    @property
    def time_horizon_s(self) -> float:
        return self.simulation.time_horizon_s

    @property
    def reporting_interval_s(self) -> float:
        return self.simulation.reporting_interval_s

    @property
    def repeat_program(self) -> bool:
        return self.simulation.repeat_program

    @property
    def mass_scheme(self) -> str:
        return self.simulation.mass_scheme

    @property
    def heat_scheme(self) -> str:
        return self.simulation.heat_scheme

    @property
    def report_time_derivatives(self) -> bool:
        return self.simulation.report_time_derivatives


class RunBundle(FrozenConfigModel):
    run_path: Path
    chemistry_path: Path
    solids_path: Path
    program_path: Path
    chemistry: ChemistryConfig
    solids: SolidConfig
    program: ProgramConfig
    run: RunConfig

    @property
    def output_directory(self) -> Path:
        return resolve_path(self.run_path.parent, self.run.outputs.directory)

    @property
    def artifacts_directory(self) -> Path:
        return resolve_path(self.run_path.parent, self.run.outputs.artifacts_directory)

    @model_validator(mode="after")
    def validate_cross_file_rules(self) -> "RunBundle":
        overlap = sorted(set(self.chemistry.gas_species) & set(self.solids.solid_species))
        if overlap:
            raise ValueError(
                "Gas and solid species identifiers must be disjoint. "
                f"Found duplicates: {', '.join(overlap)}."
            )

        zones = self.solids.initial_profile_zones
        previous_end: float | None = None
        for zone_index, zone in enumerate(zones):
            if zone_index == 0 and not math.isclose(zone.x_start_m, 0.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("solids.initial_profile.zones must start at x = 0.")
            if previous_end is not None and not math.isclose(zone.x_start_m, previous_end, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("solids.initial_profile.zones must be contiguous without gaps or overlaps.")
            previous_end = zone.x_end_m

        if previous_end is None:
            raise ValueError("solids.initial_profile.zones must not be empty.")
        if not math.isclose(previous_end, self.run.model.bed_length_m, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("solids.initial_profile.zones must end at model.bed_length_m.")

        _require_exact_keys(
            set(self.program.inlet_composition.initial),
            self.chemistry.gas_species,
            "program.inlet_composition.initial",
        )
        for step_index, step in enumerate(self.program.inlet_composition.steps):
            if isinstance(step, CompositionRampStep):
                _require_exact_keys(
                    set(step.target),
                    self.chemistry.gas_species,
                    f"program.inlet_composition.steps[{step_index}].target",
                )

        horizon = self.run.time_horizon_s
        cycle_durations = (
            ("program.inlet_flow.steps", sum_step_durations(self.program.inlet_flow.steps)),
            ("program.inlet_temperature.steps", sum_step_durations(self.program.inlet_temperature.steps)),
            ("program.outlet_pressure.steps", sum_step_durations(self.program.outlet_pressure.steps)),
            ("program.inlet_composition.steps", sum_step_durations(self.program.inlet_composition.steps)),
        )
        if not self.run.repeat_program:
            for label, duration in cycle_durations:
                if duration == 0.0:
                    continue
                if not math.isclose(
                    duration,
                    horizon,
                    rel_tol=0.0,
                    abs_tol=PROGRAM_DURATION_SUM_ABS_TOLERANCE_S,
                ):
                    difference = duration - horizon
                    raise ValueError(
                        f"{label} must sum to time_horizon_s ({horizon:.16g}) within "
                        f"{PROGRAM_DURATION_SUM_ABS_TOLERANCE_S:.1e} s, got {duration:.17g} "
                        f"(difference {difference:+.3e} s)."
                    )

        return self
