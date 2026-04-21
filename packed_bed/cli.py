from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import datetime

from .config import RunBundle, RunResult, load_run_bundle
from .properties import PROPERTY_REGISTRY
from .result_reports import compute_balance_errors, export_requested_report_csvs, format_balance_error_lines
from .result_plots import render_run_result_plots
from .reactions import REACTION_CATALOG
from .solver_clean import assemble_simulation, run_assembled_simulation
from .visualization import (
    build_system_graph,
    is_pygraphviz_available,
    render_initial_solid_profile,
    render_operating_program,
    render_system_graph,
)

def generate_artifacts(run_bundle: RunBundle) -> dict[str, Path]:
    output_directory = run_bundle.output_directory
    artifacts_directory = run_bundle.artifacts_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    artifacts_directory.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, Path] = {}
    if is_pygraphviz_available():
        system_graph = build_system_graph(
            run_bundle,
            property_registry=PROPERTY_REGISTRY,
            reaction_catalog=REACTION_CATALOG,
        )
        artifact_paths.update(render_system_graph(system_graph, artifacts_directory))
    artifact_paths.update(render_operating_program(run_bundle, artifacts_directory))
    artifact_paths.update(render_initial_solid_profile(run_bundle, artifacts_directory))
    return artifact_paths


def launch_daetools_plotter(run_result: RunResult) -> int:
    reporter = run_result.reporter
    if reporter is None or not hasattr(reporter, "Process"):
        raise ValueError("RunResult does not contain a DAETools reporter with a Process payload.")

    process = reporter.Process
    if process is None or not hasattr(process, "dictVariables"):
        raise ValueError("RunResult reporter does not expose Process.dictVariables.")

    try:
        from daetools.dae_plotter.data_receiver_io import dataReceiverProcess
        from daetools.dae_plotter.plotter import QtWidgets, daeMainWindow
    except Exception as exc:
        raise RuntimeError(
            "Cannot launch the DAETools plotter GUI. Ensure DAETools and PyQt6 are installed."
        ) from exc

    class OfflinePlotterServer:
        DataReceivers = ()

        def Stop(self):
            return None

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(["dae_plotter"])

    main_window = daeMainWindow(OfflinePlotterServer())
    main_window.loadedProcesses.append(dataReceiverProcess(process))
    main_window.show()
    return app.exec()


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
    runtime_report_ids = tuple(
        dict.fromkeys(
            (
                *run_bundle.run.outputs.requested_reports,
                "heat_balance",
                "mass_balance",
            )
        )
    )
    reporter = run_assembled_simulation(
        assembly,
        report_ids=runtime_report_ids,
        include_plot_variables=True,
    )

    run_result = RunResult(
        run_bundle=run_bundle,
        output_directory=output_directory,
        success=True,
        artifact_paths=dict(artifact_paths or {}),
        reporter=reporter,
        simulation=assembly.simulation,
    )
    report_paths = export_requested_report_csvs(run_result)
    balance_errors = compute_balance_errors(run_result)
    plot_paths = render_run_result_plots(run_result)
    return replace(
        run_result,
        artifact_paths={
            **run_result.artifact_paths,
            **plot_paths,
        },
        report_paths=report_paths,
        balance_errors=balance_errors,
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
    parser.add_argument(
        "--dae-plotter",
        "--plotter",
        dest="launch_dae_plotter",
        action="store_true",
        help="Open the DAETools plotter GUI after the run using the captured simulation results.",
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
    start_time = datetime.datetime.now()
    run_result = run_simulation(run_bundle, artifact_paths=artifact_paths)
    end_time = datetime.datetime.now()
    print(f"simulation took {end_time-start_time} seconds")
    for line in format_balance_error_lines(run_result.balance_errors):
        print(line)
    if args.launch_dae_plotter:
        print("Opening DAETools plotter. Close the plotter window to exit.")
        return launch_daetools_plotter(run_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
