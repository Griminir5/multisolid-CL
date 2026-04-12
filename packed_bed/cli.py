from __future__ import annotations

import argparse
from pathlib import Path

from .config import RunBundle, RunResult, load_run_bundle
from .properties import PROPERTY_REGISTRY
from .reactions import REACTION_CATALOG
from .solver import assemble_simulation, run_assembled_simulation
from .visualization import build_system_graph, render_initial_solid_profile, render_operating_program, render_system_graph

def generate_artifacts(run_bundle: RunBundle) -> dict[str, Path]:
    output_directory = run_bundle.output_directory
    artifacts_directory = run_bundle.artifacts_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    artifacts_directory.mkdir(parents=True, exist_ok=True)



    system_graph = build_system_graph(
        run_bundle,
        property_registry=PROPERTY_REGISTRY,
        reaction_catalog=REACTION_CATALOG,
    )
    artifact_paths: dict[str, Path] = {}
    artifact_paths.update(render_system_graph(system_graph, artifacts_directory))
    artifact_paths.update(render_operating_program(run_bundle, artifacts_directory))
    artifact_paths.update(render_initial_solid_profile(run_bundle, artifacts_directory))
    return artifact_paths


def run_simulation(
    run_bundle: RunBundle,
    property_registry=PROPERTY_REGISTRY,
    reaction_catalog=REACTION_CATALOG,
    artifact_paths: dict[str, Path] | None = None,
) -> RunResult:
    output_directory = run_bundle.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    assembly = assemble_simulation(
        run_bundle,
        property_registry=property_registry,
        reaction_catalog=reaction_catalog,
    )
    reporter = run_assembled_simulation(assembly)

    return RunResult(
        run_bundle=run_bundle,
        output_directory=output_directory,
        success=True,
        artifact_paths=dict(artifact_paths or {}),
        reporter=reporter,
        simulation=assembly.simulation,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a packed-bed simulation from YAML input files."
    )
    parser.add_argument(
        "run_yaml",
        help="Path to the top-level run.yaml file.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the YAML bundle and exit without running the simulation.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    run_bundle = load_run_bundle(args.run_yaml)
    artifact_paths = generate_artifacts(run_bundle)

    if args.validate_only:
        print(f"Validation passed: {run_bundle.run_path}")
        return 0

    run_simulation(run_bundle, artifact_paths=artifact_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
