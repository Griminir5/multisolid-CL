from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import ValidationError
import yaml
from yaml.resolver import BaseResolver

from packed_bed.programs import CompiledProgram, compile_program_channels, sum_step_durations

from .models import (
    ChemistryConfig,
    CompositionRampStep,
    ProgramConfig,
    RunConfig,
    SolidConfig,
)

if TYPE_CHECKING:
    from packed_bed.reactions import ReactionFamily


_PROGRAM_DURATION_SUM_ABS_TOLERANCE_S = 1.0e-9
_REQUIRED_PROPERTIES_BY_PHASE = {
    "gas": (
        ("mw", "molecular weight"),
        ("enthalpy", "an enthalpy correlation"),
        ("viscosity", "a viscosity correlation"),
    ),
    "solid": (
        ("mw", "molecular weight"),
        ("enthalpy", "an enthalpy correlation"),
    ),
}


class PackedBedValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Case:
    """One resolved and structurally validated runtime handoff."""

    run_path: Path
    chemistry_path: Path
    solids_path: Path
    program_path: Path
    chemistry: ChemistryConfig
    solids: SolidConfig
    run: RunConfig
    reaction_families: tuple[ReactionFamily, ...]
    inlet_flow_program: CompiledProgram
    inlet_composition_program: CompiledProgram
    inlet_temperature_program: CompiledProgram
    outlet_pressure_program: CompiledProgram

    @property
    def output_directory(self) -> Path:
        return resolve_path(self.run_path.parent, self.run.outputs.directory)

    @property
    def artifacts_directory(self) -> Path:
        return resolve_path(self.run_path.parent, self.run.outputs.artifacts_directory)


def load_case(run_yaml_path: str | Path) -> Case:
    """Load, resolve, compile, and structurally validate one case."""

    run_path = Path(run_yaml_path).resolve()
    run_data = read_yaml_mapping(run_path, "run")
    run = _parse_config_model(RunConfig, run_data, "run", run_path)
    base_dir = run_path.parent
    input_paths = {
        "chemistry": resolve_path(base_dir, run.references.chemistry_file),
        "program": resolve_path(base_dir, run.references.program_file),
        "solids": resolve_path(base_dir, run.references.solids_file),
    }

    path_errors = []
    for name, path in input_paths.items():
        reference_path = f"run.references.{name}_file"
        if not path.exists():
            path_errors.append(f"{reference_path} does not exist: {path}")
        elif not path.is_file():
            path_errors.append(f"{reference_path} must point to a file: {path}")
    if path_errors:
        raise PackedBedValidationError("\n".join(path_errors))

    return _build_case(
        run_path=run_path,
        chemistry_path=input_paths["chemistry"],
        program_path=input_paths["program"],
        solids_path=input_paths["solids"],
        run=run,
        chemistry=_parse_config_model(
            ChemistryConfig,
            read_yaml_mapping(input_paths["chemistry"], "chemistry"),
            "chemistry",
            input_paths["chemistry"],
        ),
        program=_parse_config_model(
            ProgramConfig,
            read_yaml_mapping(input_paths["program"], "program"),
            "program",
            input_paths["program"],
        ),
        solids=_parse_config_model(
            SolidConfig,
            read_yaml_mapping(input_paths["solids"], "solids"),
            "solids",
            input_paths["solids"],
        ),
    )


def resolve_case(
    *,
    run_path: str | Path,
    chemistry_path: str | Path,
    program_path: str | Path,
    solids_path: str | Path,
    run_data: dict[str, Any],
    chemistry_data: dict[str, Any],
    program_data: dict[str, Any],
    solids_data: dict[str, Any],
) -> Case:
    """Resolve an in-memory case without reading or writing files."""

    paths = {
        "run": Path(run_path).resolve(),
        "chemistry": Path(chemistry_path).resolve(),
        "program": Path(program_path).resolve(),
        "solids": Path(solids_path).resolve(),
    }
    return _build_case(
        run_path=paths["run"],
        chemistry_path=paths["chemistry"],
        program_path=paths["program"],
        solids_path=paths["solids"],
        run=_parse_config_model(RunConfig, run_data, "run", paths["run"]),
        chemistry=_parse_config_model(
            ChemistryConfig, chemistry_data, "chemistry", paths["chemistry"]
        ),
        program=_parse_config_model(ProgramConfig, program_data, "program", paths["program"]),
        solids=_parse_config_model(SolidConfig, solids_data, "solids", paths["solids"]),
    )


def _build_case(
    *,
    run_path: Path,
    chemistry_path: Path,
    program_path: Path,
    solids_path: Path,
    run: RunConfig,
    chemistry: ChemistryConfig,
    program: ProgramConfig,
    solids: SolidConfig,
) -> Case:

    shape_errors = _validate_input_shapes(chemistry, solids, program, run)
    if shape_errors:
        raise PackedBedValidationError("\n".join(shape_errors))

    from packed_bed.kinetics import load_reaction_families

    try:
        reaction_families = load_reaction_families(chemistry.reaction_families)
    except ValueError as exc:
        raise PackedBedValidationError(f"chemistry.reaction_families: {exc}") from exc

    programs = compile_program_channels(
        program,
        chemistry.gas_species,
        run.model,
        repeat=run.simulation.repeat_program,
        time_horizon=run.simulation.time_horizon_s,
    )
    case = Case(
        run_path=run_path,
        chemistry_path=chemistry_path,
        solids_path=solids_path,
        program_path=program_path,
        chemistry=chemistry,
        solids=solids,
        run=run,
        reaction_families=reaction_families,
        inlet_flow_program=programs[0],
        inlet_composition_program=programs[1],
        inlet_temperature_program=programs[2],
        outlet_pressure_program=programs[3],
    )
    return validate_case(case)


def validate_case(
    case: Case,
    *,
    property_registry=None,
    report_variable_registry=None,
) -> Case:
    """Validate resolved component, reaction, property, and report references."""

    if property_registry is None:
        from packed_bed.properties import PROPERTY_REGISTRY

        property_registry = PROPERTY_REGISTRY
    if report_variable_registry is None:
        from packed_bed.reports import REPORT_SPECS

        report_variable_registry = REPORT_SPECS

    from packed_bed.reactions import build_reaction_network, reaction_catalog

    errors: list[str] = []
    gas_species = set(case.chemistry.gas_species)
    solid_species = set(case.solids.solid_species)
    selected_species = gas_species | solid_species
    catalog = reaction_catalog(case.reaction_families)
    unknown_reactions = False

    _validate_species_group(
        errors,
        species_ids=case.chemistry.gas_species,
        expected_phase="gas",
        property_registry=property_registry,
    )
    _validate_species_group(
        errors,
        species_ids=case.solids.solid_species,
        expected_phase="solid",
        property_registry=property_registry,
    )

    for reaction_id in case.chemistry.reaction_ids:
        reaction = catalog.get(reaction_id)
        if reaction is None:
            errors.append(f"chemistry.reaction_ids contains unknown id '{reaction_id}'.")
            unknown_reactions = True
            continue
        missing = sorted(species_id for species_id in reaction.all_species if species_id not in selected_species)
        if missing:
            errors.append(
                f"Reaction '{reaction_id}' requires unselected species: {', '.join(missing)}."
            )

    if not unknown_reactions:
        try:
            build_reaction_network(
                case.chemistry.reaction_ids,
                case.chemistry.gas_species,
                case.solids.solid_species,
                families=case.reaction_families,
            )
        except (KeyError, ValueError) as exc:
            message = str(exc).strip("'")
            if message not in errors:
                errors.append(message)

    unknown_reports = sorted(
        report_id
        for report_id in case.run.outputs.requested_reports
        if report_id not in report_variable_registry
    )
    if unknown_reports:
        errors.append(
            "run.outputs.requested_reports contains unknown ids: "
            f"{', '.join(unknown_reports)}."
        )
    unavailable_reports = sorted(
        report_id
        for report_id in case.run.outputs.requested_reports
        if report_id in report_variable_registry
        and getattr(report_variable_registry[report_id], "requires_reactions", False)
        and not case.chemistry.reaction_ids
    )
    if unavailable_reports:
        errors.append(
            "run.outputs.requested_reports requires at least one selected reaction for: "
            f"{', '.join(unavailable_reports)}."
        )

    if errors:
        raise PackedBedValidationError("\n".join(errors))
    return case


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _parse_config_model(model_type, data: dict[str, Any], label: str, path: Path):
    try:
        return model_type.model_validate(data)
    except ValidationError as exc:
        lines = [f"{label} is invalid: {path}"]
        for error in exc.errors():
            location = ".".join(str(item) for item in error["loc"]) or "<root>"
            lines.append(f"- {label}.{location}: {error['msg']}")
        raise PackedBedValidationError("\n".join(lines)) from exc


def read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.load(handle, Loader=_UniqueKeyLoader)
    except PackedBedValidationError as exc:
        raise PackedBedValidationError(f"{label} is invalid: {path}\n- {exc}") from exc
    except FileNotFoundError as exc:
        raise PackedBedValidationError(f"{label} was not found: {path}") from exc
    except OSError as exc:
        raise PackedBedValidationError(f"Could not read {label}: {path}") from exc
    except yaml.YAMLError as exc:
        raise PackedBedValidationError(f"{label} contains invalid YAML: {path}") from exc

    if not isinstance(data, dict):
        raise PackedBedValidationError(f"{label} must contain a top-level mapping: {path}")
    return data


def _validate_input_shapes(
    chemistry: ChemistryConfig,
    solids: SolidConfig,
    program: ProgramConfig,
    run: RunConfig,
) -> list[str]:
    errors: list[str] = []
    overlap = sorted(set(chemistry.gas_species) & set(solids.solid_species))
    if overlap:
        errors.append(
            "chemistry.gas_species and solids.solid_species must be disjoint; "
            f"found: {', '.join(overlap)}."
        )

    previous_end: float | None = None
    for zone_index, zone in enumerate(solids.initial_profile.zones):
        if zone_index == 0 and not math.isclose(zone.x_start_m, 0.0, rel_tol=0.0, abs_tol=1e-12):
            errors.append("solids.initial_profile.zones must start at x = 0.")
        if previous_end is not None and not math.isclose(
            zone.x_start_m, previous_end, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append("solids.initial_profile.zones must be contiguous without gaps or overlaps.")
        previous_end = zone.x_end_m
    if previous_end is None:
        errors.append("solids.initial_profile.zones must not be empty.")
    elif not math.isclose(previous_end, run.model.bed_length_m, rel_tol=0.0, abs_tol=1e-12):
        errors.append("solids.initial_profile.zones must end at run.model.bed_length_m.")

    expected_gases = set(chemistry.gas_species)
    _append_key_mismatch(
        errors,
        set(program.inlet_composition.initial),
        expected_gases,
        "program.inlet_composition.initial",
    )
    for step_index, step in enumerate(program.inlet_composition.steps):
        if isinstance(step, CompositionRampStep):
            _append_key_mismatch(
                errors,
                set(step.target),
                expected_gases,
                f"program.inlet_composition.steps.{step_index}.target",
            )

    if not run.simulation.repeat_program:
        durations = (
            ("program.inlet_flow.steps", sum_step_durations(program.inlet_flow.steps)),
            (
                "program.inlet_temperature.steps",
                sum_step_durations(program.inlet_temperature.steps),
            ),
            ("program.outlet_pressure.steps", sum_step_durations(program.outlet_pressure.steps)),
            (
                "program.inlet_composition.steps",
                sum_step_durations(program.inlet_composition.steps),
            ),
        )
        for path, duration in durations:
            if duration == 0.0 or math.isclose(
                duration,
                run.simulation.time_horizon_s,
                rel_tol=0.0,
                abs_tol=_PROGRAM_DURATION_SUM_ABS_TOLERANCE_S,
            ):
                continue
            difference = duration - run.simulation.time_horizon_s
            errors.append(
                f"{path} must sum to run.simulation.time_horizon_s "
                f"({run.simulation.time_horizon_s:.16g}) within "
                f"{_PROGRAM_DURATION_SUM_ABS_TOLERANCE_S:.1e} s, got {duration:.17g} "
                f"(difference {difference:+.3e} s)."
            )
    return errors


def _append_key_mismatch(
    errors: list[str],
    actual: set[str],
    expected: set[str],
    path: str,
) -> None:
    if actual == expected:
        return
    differences = []
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        differences.append(f"missing {', '.join(missing)}")
    if extra:
        differences.append(f"unexpected {', '.join(extra)}")
    errors.append(f"{path} species mismatch: {'; '.join(differences)}.")


def _validate_species_group(
    errors: list[str],
    *,
    species_ids: tuple[str, ...],
    expected_phase: str,
    property_registry,
) -> None:
    for species_id in species_ids:
        if not property_registry.has_species(species_id):
            errors.append(f"Unknown {expected_phase} species '{species_id}'.")
            continue
        record = property_registry.get_record(species_id)
        if record.phase != expected_phase:
            errors.append(
                f"Species '{species_id}' is phase '{record.phase}', not {expected_phase}."
            )
        for property_name, description in _REQUIRED_PROPERTIES_BY_PHASE[expected_phase]:
            if getattr(record, property_name) is None:
                errors.append(
                    f"{expected_phase.capitalize()} species '{species_id}' must define {description}."
                )


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line = key_node.start_mark.line + 1
            raise PackedBedValidationError(f"duplicate key {key!r} at line {line}.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


__all__ = (
    "Case",
    "PackedBedValidationError",
    "load_case",
    "read_yaml_mapping",
    "resolve_case",
    "resolve_path",
    "validate_case",
)
