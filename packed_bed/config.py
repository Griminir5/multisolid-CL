from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml

from .programs import ProgramStep, ScalarProgram, VectorProgram, coerce_composition_mapping


@dataclass(frozen=True)
class ChemistryConfig:
    gas_species: tuple[str, ...]
    solid_species: tuple[str, ...]
    reaction_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScalarChannelConfig:
    initial: float
    steps: tuple[ProgramStep, ...] = ()

    def compile_program(self) -> ScalarProgram:
        program = ScalarProgram(self.initial)
        for step in self.steps:
            if step.kind == "hold":
                program.hold(step.duration)
            else:
                program.ramp(step.duration, step.target)
        return program


@dataclass(frozen=True)
class CompositionChannelConfig:
    initial: Mapping[str, float]
    steps: tuple[ProgramStep, ...] = ()

    def compile_program(self, species_order) -> VectorProgram:
        initial_vector = coerce_composition_mapping(
            self.initial,
            species_order,
            label="Inlet composition program initial value",
        )
        program = VectorProgram(initial_vector)
        for step in self.steps:
            if step.kind == "hold":
                program.hold(step.duration)
            else:
                target_vector = coerce_composition_mapping(
                    step.target,
                    species_order,
                    label="Inlet composition program target",
                )
                program.ramp(step.duration, target_vector)
        return program


@dataclass(frozen=True)
class ProgramConfig:
    inlet_flow: ScalarChannelConfig | None = None
    inlet_temperature: ScalarChannelConfig | None = None
    outlet_pressure: ScalarChannelConfig | None = None
    inlet_composition: CompositionChannelConfig | None = None


@dataclass(frozen=True)
class ModelConfig:
    bed_length_m: float = 2.5
    bed_radius_m: float = 0.1
    particle_diameter_m: float = 0.01
    axial_cells: int = 20
    interparticle_voidage: float = 0.5
    intraparticle_voidage: float = 0.5
    initial_solid_concentration_mol_per_m3: Mapping[str, float] = field(default_factory=dict)
    gas_constant: float = 8.314462
    pi_value: float = 3.14


@dataclass(frozen=True)
class SolverConfig:
    relative_tolerance: float = 1e-6


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    artifacts_directory: Path
    requested_reports: tuple[str, ...]


@dataclass(frozen=True)
class RunConfig:
    chemistry_file: Path
    program_file: Path
    time_horizon_s: float
    reporting_interval_s: float
    mass_scheme: str
    heat_scheme: str
    report_time_derivatives: bool
    model: ModelConfig
    solver: SolverConfig
    outputs: OutputConfig
    system_name: str = "MassTrsf"


@dataclass(frozen=True)
class RunBundle:
    run_path: Path
    chemistry_path: Path
    program_path: Path
    chemistry: ChemistryConfig
    program: ProgramConfig
    run: RunConfig


def _read_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file '{path}' must contain a top-level mapping.")
    return data


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _coerce_string_tuple(value, label):
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{label} must be provided as a list.")
    return tuple(str(item) for item in value)


def _coerce_float_mapping(value, label):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be provided as a mapping.")
    return {str(key): float(raw_value) for key, raw_value in value.items()}


def _parse_steps(raw_steps, label, composition=False):
    if raw_steps is None:
        return ()
    if not isinstance(raw_steps, list):
        raise ValueError(f"{label} steps must be provided as a list.")

    steps = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise ValueError(f"{label} step {index} must be a mapping.")
        kind = str(raw_step.get("kind", ""))
        duration = float(raw_step.get("duration_s"))
        if kind not in {"hold", "ramp"}:
            raise ValueError(f"{label} step {index} kind must be 'hold' or 'ramp'.")
        target = raw_step.get("target")
        if kind == "ramp" and target is None:
            raise ValueError(f"{label} step {index} must define a target for a ramp.")
        if composition and target is not None and not isinstance(target, dict):
            raise ValueError(f"{label} step {index} target must be a species-keyed mapping.")
        if not composition and target is not None:
            target = float(target)
        steps.append(ProgramStep(duration=duration, kind=kind, target=target))
    return tuple(steps)


def _parse_scalar_channel(raw_channel, label):
    if raw_channel is None:
        return None
    if not isinstance(raw_channel, dict):
        raise ValueError(f"{label} must be a mapping.")
    return ScalarChannelConfig(
        initial=float(raw_channel.get("initial")),
        steps=_parse_steps(raw_channel.get("steps"), label),
    )


def _parse_composition_channel(raw_channel, label):
    if raw_channel is None:
        return None
    if not isinstance(raw_channel, dict):
        raise ValueError(f"{label} must be a mapping.")
    initial = raw_channel.get("initial")
    if not isinstance(initial, dict):
        raise ValueError(f"{label} initial value must be a species-keyed mapping.")
    return CompositionChannelConfig(
        initial={str(key): float(value) for key, value in initial.items()},
        steps=_parse_steps(raw_channel.get("steps"), label, composition=True),
    )


def load_run_bundle(run_yaml_path) -> RunBundle:
    run_path = Path(run_yaml_path).resolve()
    run_data = _read_yaml(run_path)
    base_dir = run_path.parent

    references = run_data.get("references") or {}
    if not isinstance(references, dict):
        raise ValueError("run.yaml 'references' section must be a mapping.")

    chemistry_path = _resolve_path(base_dir, references.get("chemistry_file", "chemistry.yaml"))
    program_path = _resolve_path(base_dir, references.get("program_file", "program.yaml"))

    chemistry_data = _read_yaml(chemistry_path)
    program_data = _read_yaml(program_path)

    chemistry = ChemistryConfig(
        gas_species=_coerce_string_tuple(chemistry_data.get("gas_species"), "chemistry.gas_species"),
        solid_species=_coerce_string_tuple(chemistry_data.get("solid_species"), "chemistry.solid_species"),
        reaction_ids=_coerce_string_tuple(chemistry_data.get("reaction_ids"), "chemistry.reaction_ids"),
    )

    program = ProgramConfig(
        inlet_flow=_parse_scalar_channel(program_data.get("inlet_flow"), "program.inlet_flow"),
        inlet_temperature=_parse_scalar_channel(program_data.get("inlet_temperature"), "program.inlet_temperature"),
        outlet_pressure=_parse_scalar_channel(program_data.get("outlet_pressure"), "program.outlet_pressure"),
        inlet_composition=_parse_composition_channel(program_data.get("inlet_composition"), "program.inlet_composition"),
    )

    simulation_data = run_data.get("simulation") or {}
    model_data = run_data.get("model") or {}
    outputs_data = run_data.get("outputs") or {}
    solver_data = run_data.get("solver") or {}

    if not isinstance(simulation_data, dict):
        raise ValueError("run.yaml 'simulation' section must be a mapping.")
    if not isinstance(model_data, dict):
        raise ValueError("run.yaml 'model' section must be a mapping.")
    if not isinstance(outputs_data, dict):
        raise ValueError("run.yaml 'outputs' section must be a mapping.")
    if not isinstance(solver_data, dict):
        raise ValueError("run.yaml 'solver' section must be a mapping.")

    outputs_directory = _resolve_path(base_dir, outputs_data.get("directory", "output"))
    artifacts_directory = _resolve_path(
        base_dir,
        outputs_data.get("artifacts_directory", outputs_directory / "artifacts"),
    )

    run = RunConfig(
        chemistry_file=chemistry_path,
        program_file=program_path,
        time_horizon_s=float(simulation_data.get("time_horizon_s", 30000.0)),
        reporting_interval_s=float(simulation_data.get("reporting_interval_s", 10.0)),
        mass_scheme=str(simulation_data.get("mass_scheme", "weno3")),
        heat_scheme=str(simulation_data.get("heat_scheme", "central")),
        report_time_derivatives=bool(simulation_data.get("report_time_derivatives", False)),
        system_name=str(simulation_data.get("system_name", "MassTrsf")),
        model=ModelConfig(
            bed_length_m=float(model_data.get("bed_length_m", 2.5)),
            bed_radius_m=float(model_data.get("bed_radius_m", 0.1)),
            particle_diameter_m=float(model_data.get("particle_diameter_m", 0.01)),
            axial_cells=int(model_data.get("axial_cells", 20)),
            interparticle_voidage=float(model_data.get("interparticle_voidage", 0.5)),
            intraparticle_voidage=float(model_data.get("intraparticle_voidage", 0.5)),
            initial_solid_concentration_mol_per_m3=_coerce_float_mapping(
                model_data.get("initial_solid_concentration_mol_per_m3"),
                "model.initial_solid_concentration_mol_per_m3",
            ),
            gas_constant=float(model_data.get("gas_constant", 8.314462)),
            pi_value=float(model_data.get("pi_value", 3.14)),
        ),
        solver=SolverConfig(
            relative_tolerance=float(solver_data.get("relative_tolerance", 1e-6)),
        ),
        outputs=OutputConfig(
            directory=outputs_directory,
            artifacts_directory=artifacts_directory,
            requested_reports=_coerce_string_tuple(outputs_data.get("requested_reports"), "outputs.requested_reports"),
        ),
    )

    return RunBundle(
        run_path=run_path,
        chemistry_path=chemistry_path,
        program_path=program_path,
        chemistry=chemistry,
        program=program,
        run=run,
    )
