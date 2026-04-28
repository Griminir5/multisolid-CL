from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Annotated, Literal

import yaml
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, field_validator, model_validator
from yaml.resolver import BaseResolver

from .axial_schemes import SUPPORTED_SCHEMES
from .properties import PROPERTY_REGISTRY
from .reactions import REACTION_CATALOG, build_reaction_network
from .reporting import REPORT_VARIABLE_REGISTRY


class PackedBedValidationError(ValueError):
    pass


class FrozenConfigModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )


def _require_string(value: str) -> str:
    if value == "" or value != value.strip():
        raise ValueError("must not be blank or padded with whitespace.")
    return value


def _require_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise ValueError("must be written as a float value.")
    if not math.isfinite(value):
        raise ValueError("must be finite.")
    return value


def _require_positive(value: float) -> float:
    if value <= 0.0:
        raise ValueError("must be strictly positive.")
    return value


def _require_nonnegative(value: float) -> float:
    if value < 0.0:
        raise ValueError("must be non-negative.")
    return value


def _require_unit_fraction(value: float) -> float:
    if not (0.0 < value < 1.0):
        raise ValueError("must lie strictly between 0 and 1.")
    return value


ConfigString = Annotated[str, AfterValidator(_require_string)]
PositiveFloat = Annotated[float, BeforeValidator(_require_float), AfterValidator(_require_positive)]
NonNegativeFloat = Annotated[float, BeforeValidator(_require_float), AfterValidator(_require_nonnegative)]
UnitFraction = Annotated[float, BeforeValidator(_require_float), AfterValidator(_require_unit_fraction)]


def _require_nonempty_unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        raise ValueError("must not be empty.")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"contains duplicates: {', '.join(duplicates)}.")
    return values


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
    if not mapping:
        raise ValueError("must not be empty.")
    total = sum(mapping.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"must sum to 1.0 exactly, got {total:.16g}.")
    return mapping


def _require_exact_keys(actual: set[str], expected: tuple[str, ...], label: str) -> None:
    expected_keys = set(expected)
    if actual == expected_keys:
        return
    missing = sorted(expected_keys - actual)
    extra = sorted(actual - expected_keys)
    parts: list[str] = []
    if missing:
        parts.append(f"missing {', '.join(missing)}")
    if extra:
        parts.append(f"unexpected {', '.join(extra)}")
    raise ValueError(f"{label} species mismatch: {'; '.join(parts)}.")


def _sum_step_durations(steps: tuple["HoldStep | ScalarRampStep | CompositionRampStep", ...]) -> float:
    return sum(step.duration_s for step in steps)


def _interpolate_program_value(
    start_value: float | tuple[float, ...],
    end_value: float | tuple[float, ...],
    fraction: float,
) -> float | tuple[float, ...]:
    if isinstance(start_value, tuple):
        if not isinstance(end_value, tuple):
            raise TypeError("Expected tuple-valued program endpoints.")
        return tuple(
            start_component + (end_component - start_component) * fraction
            for start_component, end_component in zip(start_value, end_value)
        )

    if isinstance(end_value, tuple):
        raise TypeError("Expected scalar-valued program endpoints.")
    return start_value + (end_value - start_value) * fraction


def _compile_program_segments(
    initial_value: float | tuple[float, ...],
    steps: tuple["HoldStep | ScalarRampStep | CompositionRampStep", ...],
    *,
    repeat: bool,
    time_horizon: float | None,
    resolve_next_value,
) -> tuple["ProgramSegment", ...]:
    if repeat and time_horizon is None:
        raise ValueError("time_horizon must be provided when repeat=True.")

    current_time = 0.0
    current_value = initial_value
    segments: list[ProgramSegment] = []

    while True:
        for step_index, step in enumerate(steps):
            next_time = current_time + step.duration_s
            next_value = resolve_next_value(step_index, step, current_value)

            if time_horizon is not None and next_time > time_horizon:
                if current_time >= time_horizon:
                    return tuple(segments)

                fraction = (time_horizon - current_time) / step.duration_s
                segments.append(
                    ProgramSegment(
                        start_time=current_time,
                        end_time=time_horizon,
                        start_value=current_value,
                        end_value=_interpolate_program_value(current_value, next_value, fraction),
                    )
                )
                return tuple(segments)

            segments.append(
                ProgramSegment(
                    start_time=current_time,
                    end_time=next_time,
                    start_value=current_value,
                    end_value=next_value,
                )
            )
            current_time = next_time
            current_value = next_value

        if not repeat or not steps or (time_horizon is not None and current_time >= time_horizon):
            return tuple(segments)


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _format_validation_error(label: str, path: Path, exc: ValidationError) -> PackedBedValidationError:
    lines = [f"{label} is invalid: {path}"]
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"]) or "<root>"
        lines.append(f"- {location}: {error['msg']}")
    return PackedBedValidationError("\n".join(lines))


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line = key_node.start_mark.line + 1
            raise PackedBedValidationError(f"Duplicate key {key!r} at line {line}.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.load(handle, Loader=_UniqueKeyLoader)
    except PackedBedValidationError:
        raise
    except FileNotFoundError as exc:
        raise PackedBedValidationError(f"{label} was not found: {path}") from exc
    except OSError as exc:
        raise PackedBedValidationError(f"Could not read {label}: {path}") from exc
    except yaml.YAMLError as exc:
        raise PackedBedValidationError(f"{label} contains invalid YAML: {path}") from exc

    if not isinstance(data, dict):
        raise PackedBedValidationError(f"{label} must contain a top-level mapping: {path}")
    return data


def _validate_model(model_type, data: dict[str, Any], label: str, path: Path):
    try:
        return model_type.model_validate(data)
    except ValidationError as exc:
        raise _format_validation_error(label, path, exc) from exc


@dataclass(frozen=True)
class ProgramSegment:
    start_time: float
    end_time: float
    start_value: float | tuple[float, ...]
    end_value: float | tuple[float, ...]


@dataclass(frozen=True)
class ScalarProgram:
    initial_value: float
    segments: tuple[ProgramSegment, ...]

    def build_segments(self) -> tuple[ProgramSegment, ...]:
        return self.segments


@dataclass(frozen=True)
class VectorProgram:
    initial_value: tuple[float, ...]
    segments: tuple[ProgramSegment, ...]

    def build_segments(self) -> tuple[ProgramSegment, ...]:
        return self.segments


@dataclass(frozen=True)
class RunResult:
    run_bundle: RunBundle
    output_directory: Path
    success: bool
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    report_paths: dict[str, Path] = field(default_factory=dict)
    balance_errors: dict[str, Any] = field(default_factory=dict)
    summary_path: Path | None = None
    balances_path: Path | None = None
    reporter: Any | None = None
    simulation: Any | None = None


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
        segments = _compile_program_segments(
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

        segments = _compile_program_segments(
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
        return _resolve_path(self.run_path.parent, self.run.outputs.directory)

    @property
    def artifacts_directory(self) -> Path:
        return _resolve_path(self.run_path.parent, self.run.outputs.artifacts_directory)

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
            ("program.inlet_flow.steps", _sum_step_durations(self.program.inlet_flow.steps)),
            ("program.inlet_temperature.steps", _sum_step_durations(self.program.inlet_temperature.steps)),
            ("program.outlet_pressure.steps", _sum_step_durations(self.program.outlet_pressure.steps)),
            ("program.inlet_composition.steps", _sum_step_durations(self.program.inlet_composition.steps)),
        )
        if not self.run.repeat_program:
            for label, duration in cycle_durations:
                if math.isclose(duration, 0.0, rel_tol=0.0, abs_tol=1e-12):
                    continue
                if not math.isclose(duration, horizon, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"{label} must sum exactly to time_horizon_s ({horizon:.16g}), got {duration:.16g}.")

        return self


def validate_run_bundle(
    run_bundle: RunBundle,
    property_registry=PROPERTY_REGISTRY,
    reaction_catalog=REACTION_CATALOG,
    report_variable_registry=REPORT_VARIABLE_REGISTRY,
) -> RunBundle:
    errors: list[str] = []
    has_unknown_reaction_ids = False
    gas_species = set(run_bundle.chemistry.gas_species)
    solid_species = set(run_bundle.solids.solid_species)
    selected_species = gas_species | solid_species

    for species_id in run_bundle.chemistry.gas_species:
        if not property_registry.has_species(species_id):
            errors.append(f"Unknown gas species '{species_id}'.")
            continue
        record = property_registry.get_record(species_id)
        if record.phase != "gas":
            errors.append(f"Species '{species_id}' is phase '{record.phase}', not gas.")
        if record.mw is None:
            errors.append(f"Gas species '{species_id}' must define molecular weight.")
        if record.enthalpy is None:
            errors.append(f"Gas species '{species_id}' must define an enthalpy correlation.")
        if record.viscosity is None:
            errors.append(f"Gas species '{species_id}' must define a viscosity correlation.")

    for species_id in run_bundle.solids.solid_species:
        if not property_registry.has_species(species_id):
            errors.append(f"Unknown solid species '{species_id}'.")
            continue
        record = property_registry.get_record(species_id)
        if record.phase != "solid":
            errors.append(f"Species '{species_id}' is phase '{record.phase}', not solid.")
        if record.mw is None:
            errors.append(f"Solid species '{species_id}' must define molecular weight.")
        if record.enthalpy is None:
            errors.append(f"Solid species '{species_id}' must define an enthalpy correlation.")

    for reaction_id in run_bundle.chemistry.reaction_ids:
        reaction = reaction_catalog.get(reaction_id)
        if reaction is None:
            errors.append(f"Unknown reaction id '{reaction_id}'.")
            has_unknown_reaction_ids = True
            continue

        missing = sorted(
            species_id
            for species_id in reaction.all_species
            if species_id not in selected_species
        )
        if missing:
            errors.append(f"Reaction '{reaction_id}' requires unselected species: {', '.join(missing)}.")

        invalid_refs = sorted(
            species_id
            for species_id in reaction.stoichiometry
            if species_id not in selected_species
        )
        for species_id in invalid_refs:
            errors.append(
                f"Reaction '{reaction_id}' references species '{species_id}' that is not selected in the current gas/solid configuration."
            )

    if not has_unknown_reaction_ids:
        try:
            build_reaction_network(
                run_bundle.chemistry.reaction_ids,
                run_bundle.chemistry.gas_species,
                run_bundle.solids.solid_species,
                reaction_catalog=reaction_catalog,
            )
        except ValueError as exc:
            error_message = str(exc)
            if error_message not in errors:
                errors.append(error_message)

    unknown_reports = sorted(
        report_id
        for report_id in run_bundle.run.outputs.requested_reports
        if report_id not in report_variable_registry
    )
    if unknown_reports:
        errors.append(f"outputs.requested_reports contains unknown ids: {', '.join(unknown_reports)}.")

    if errors:
        raise PackedBedValidationError("\n".join(errors))
    return run_bundle


def load_run_bundle(run_yaml_path: str | Path) -> RunBundle:
    run_path = Path(run_yaml_path).resolve()
    run = _validate_model(RunConfig, _read_yaml_mapping(run_path, "run.yaml"), "run.yaml", run_path)

    base_dir = run_path.parent
    chemistry_path = _resolve_path(base_dir, run.references.chemistry_file)
    program_path = _resolve_path(base_dir, run.references.program_file)
    solids_path = _resolve_path(base_dir, run.references.solids_file)

    for label, path in (
        ("run.references.chemistry_file", chemistry_path),
        ("run.references.program_file", program_path),
        ("run.references.solids_file", solids_path),
    ):
        if not path.exists():
            raise PackedBedValidationError(f"{label} does not exist: {path}")
        if not path.is_file():
            raise PackedBedValidationError(f"{label} must point to a file: {path}")

    chemistry = _validate_model(
        ChemistryConfig,
        _read_yaml_mapping(chemistry_path, "chemistry.yaml"),
        "chemistry.yaml",
        chemistry_path,
    )
    program = _validate_model(
        ProgramConfig,
        _read_yaml_mapping(program_path, "program.yaml"),
        "program.yaml",
        program_path,
    )
    solids = _validate_model(
        SolidConfig,
        _read_yaml_mapping(solids_path, "solids.yaml"),
        "solids.yaml",
        solids_path,
    )

    try:
        run_bundle = RunBundle(
            run_path=run_path,
            chemistry_path=chemistry_path,
            solids_path=solids_path,
            program_path=program_path,
            chemistry=chemistry,
            solids=solids,
            program=program,
            run=run,
        )
    except ValidationError as exc:
        raise _format_validation_error("run bundle", run_path, exc) from exc

    return validate_run_bundle(run_bundle)
