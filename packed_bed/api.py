from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import RunBundle, load_run_bundle
from .export import export_run_outputs
from .properties import DEFAULT_PROPERTY_REGISTRY
from .reactions import DEFAULT_REACTION_CATALOG
from .solver import assemble_simulation, run_assembled_simulation
from .validation import validate_run_bundle
from .visualization import build_system_graph, render_operating_program, render_system_graph


@dataclass(frozen=True)
class RunResult:
    run_bundle: RunBundle
    output_directory: Path
    artifact_paths: dict[str, Path]
    report_paths: dict[str, Path]
    summary_path: Path
    balances_path: Path
    success: bool = True


def _coerce_run_bundle(run_yaml_path_or_bundle) -> RunBundle:
    if isinstance(run_yaml_path_or_bundle, RunBundle):
        return run_yaml_path_or_bundle
    return load_run_bundle(run_yaml_path_or_bundle)


def run_simulation(
    run_yaml_path_or_bundle,
    *,
    property_registry=DEFAULT_PROPERTY_REGISTRY,
    reaction_catalog=DEFAULT_REACTION_CATALOG,
) -> RunResult:
    run_bundle = _coerce_run_bundle(run_yaml_path_or_bundle)
    validate_run_bundle(
        run_bundle,
        property_registry=property_registry,
        reaction_catalog=reaction_catalog,
    )

    output_directory = Path(run_bundle.run.outputs.directory)
    artifacts_directory = Path(run_bundle.run.outputs.artifacts_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    artifacts_directory.mkdir(parents=True, exist_ok=True)

    system_graph = build_system_graph(
        run_bundle,
        property_registry=property_registry,
        reaction_catalog=reaction_catalog,
    )
    artifact_paths = {}
    artifact_paths.update(render_system_graph(system_graph, artifacts_directory))
    artifact_paths.update(render_operating_program(run_bundle, artifacts_directory))

    assembly = assemble_simulation(
        run_bundle,
        property_registry=property_registry,
        reaction_catalog=reaction_catalog,
    )

    try:
        reporter = run_assembled_simulation(assembly)
        export_data = export_run_outputs(reporter, run_bundle, output_directory)
        return RunResult(
            run_bundle=run_bundle,
            output_directory=output_directory,
            artifact_paths=artifact_paths,
            report_paths=export_data["report_paths"],
            summary_path=export_data["summary_path"],
            balances_path=export_data["balances_path"],
            success=True,
        )
    finally:
        try:
            assembly.simulation.Finalize()
        except Exception:
            pass
