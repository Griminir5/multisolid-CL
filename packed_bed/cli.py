from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .api import run_simulation
from .config import OutputConfig, load_run_bundle
from .validation import validate_run_bundle


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a packed-bed simulation from YAML input files."
    )
    parser.add_argument(
        "run_yaml",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "examples" / "default_case" / "run.yaml"),
        help="Path to the top-level run.yaml file.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override the directory where CSV outputs are written.",
    )
    parser.add_argument(
        "--artifacts-dir",
        help="Override the directory where graph and plot artifacts are written.",
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
    if args.output_dir or args.artifacts_dir:
        outputs = OutputConfig(
            directory=Path(args.output_dir).resolve() if args.output_dir else run_bundle.run.outputs.directory,
            artifacts_directory=Path(args.artifacts_dir).resolve() if args.artifacts_dir else run_bundle.run.outputs.artifacts_directory,
            requested_reports=run_bundle.run.outputs.requested_reports,
        )
        run_bundle = replace(run_bundle, run=replace(run_bundle.run, outputs=outputs))

    validate_run_bundle(run_bundle)
    if args.validate_only:
        print(f"Validation passed: {run_bundle.run_path}")
        return 0

    result = run_simulation(run_bundle)
    print(f"Simulation finished successfully: {run_bundle.run_path}")
    print(f"Summary CSV: {result.summary_path}")
    print(f"Balances CSV: {result.balances_path}")
    print(f"Artifacts directory: {run_bundle.run.outputs.artifacts_directory}")
    print(f"Output directory: {result.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
