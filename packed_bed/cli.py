from __future__ import annotations

import argparse
import math
import sys
from typing import TYPE_CHECKING

from .batch import BatchResult, run_batch_file
from .config import PackedBedValidationError, load_case

if TYPE_CHECKING:
    from .reports import RunResult


def _positive_float(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero.")
    return value


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a packed-bed simulation from YAML input files."
    )
    parser.add_argument("run_yaml", help="Path to the top-level run.yaml file.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the YAML bundle and exit without creating files or running the simulation.",
    )
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help="Generate the system, program, and initial-profile artifacts before the run.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Capture plot variables and render the standard result plots after the run.",
    )
    parser.add_argument(
        "--dae-plotter",
        dest="launch_dae_plotter",
        action="store_true",
        help="Open the DAETools plotter GUI after the run using the captured simulation results.",
    )
    return parser


def build_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m packed_bed batch",
        description="Run a batch of packed-bed simulations from a batch YAML file.",
    )
    parser.add_argument("batch_yaml", help="Path to the batch.yaml file.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Expand and validate every batch case without creating files.",
    )
    parser.add_argument(
        "--case-timeout-s",
        type=_positive_float,
        default=None,
        help="Kill an individual batch case if it runs longer than this many seconds.",
    )
    return parser


def _print_failed_batch_records(batch_result: BatchResult) -> None:
    for record in batch_result.records:
        if record.status.endswith("_failed"):
            print(f"{record.case_id}: {record.error}", file=sys.stderr)


def _run_cli(argv: list[str]) -> int:
    if argv and argv[0] == "batch":
        args = build_batch_parser().parse_args(argv[1:])
        batch_result = run_batch_file(
            args.batch_yaml,
            validate_only=args.validate_only,
            case_timeout_s=args.case_timeout_s,
        )
        passed = batch_result.total_count - batch_result.failed_count
        if args.validate_only:
            print(f"Batch validation complete: {passed}/{batch_result.total_count} cases passed.")
            _print_failed_batch_records(batch_result)
            return 2 if batch_result.failed_count else 0

        if batch_result.summary_path is None:
            print(
                f"Batch aborted: {batch_result.failed_count}/{batch_result.total_count} cases "
                "failed validation; no simulations were started.",
                file=sys.stderr,
            )
            _print_failed_batch_records(batch_result)
            return 2

        succeeded = sum(1 for record in batch_result.records if record.status == "success")
        print(
            f"Batch complete: {succeeded}/{batch_result.total_count} cases succeeded. "
            f"Summary: {batch_result.summary_path}"
        )
        _print_failed_batch_records(batch_result)
        return 1 if batch_result.failed_count else 0

    args = build_parser().parse_args(argv)
    case = load_case(args.run_yaml)
    if args.validate_only:
        print(f"Validation passed: {case.run_path}")
        return 0

    artifact_paths = {}
    if args.artifacts:
        from .plots import generate_artifacts

        artifact_paths = generate_artifacts(case)
    from .simulation import run_case

    run_result = run_case(
        case,
        artifact_paths=artifact_paths,
        render_plots=args.plots or args.launch_dae_plotter,
    )
    print(f"Simulation took {run_result.runtime_s:.3f} seconds.")

    from .reports import format_balance_error_lines

    for line in format_balance_error_lines(run_result.balance_errors):
        print(line)
    if args.launch_dae_plotter:
        print("Opening DAETools plotter. Close the plotter window to exit.")
        return launch_daetools_plotter(run_result)
    return 0


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        return _run_cli(arguments)
    except PackedBedValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
