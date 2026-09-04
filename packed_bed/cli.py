from __future__ import annotations

import argparse
import math
import sys
from typing import TYPE_CHECKING

from .batch import run_batch_file
from .config import PackedBedValidationError, load_case

if TYPE_CHECKING:
    from .reports import RunResult


def _positive_float(raw_value: str) -> float:
    value = float(raw_value)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero.")
    return value


def _positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be greater than zero.")
    return value


def launch_daetools_plotter(run_result: RunResult) -> int:
    process = getattr(run_result.reporter, "Process", None)
    if not hasattr(process, "dictVariables"):
        raise ValueError("The run has no retained DAETools process for the plotter.")

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


def build_parser(*, batch: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m packed_bed" + (" batch" if batch else ""),
        description="Run packed-bed simulations from YAML input files.",
        epilog=None if batch else "For batches: python -m packed_bed batch --help",
    )
    parser.add_argument("input_yaml", help="Path to batch.yaml." if batch else "Path to run.yaml.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without creating files.")
    parser.add_argument("--debug", action="store_true", help="Show a traceback for unexpected errors.")
    if batch:
        parser.add_argument("--case-timeout-s", type=_positive_float, help="Stop a case after this many seconds.")
        parser.add_argument("--workers", type=_positive_int, help="Number of single-threaded worker processes.")
    else:
        parser.add_argument("--artifacts", action="store_true", help="Create input diagrams before the run.")
        parser.add_argument("--dae-plotter", action="store_true", help="Open the DAETools plotter after the run.")
    return parser


def _run_cli(argv: list[str]) -> int:
    batch = bool(argv and argv[0] == "batch")
    args = build_parser(batch=batch).parse_args(argv[1:] if batch else argv)
    if batch:
        batch_result = run_batch_file(
            args.input_yaml,
            validate_only=args.validate_only,
            case_timeout_s=args.case_timeout_s,
            workers=args.workers,
        )
        for record in batch_result.records:
            if record.error:
                print(f"{record.case_id}: {record.error}", file=sys.stderr)
            for plot_id, error in record.plot_errors.items():
                print(f"{record.case_id} plot '{plot_id}': {error}", file=sys.stderr)
        passed = batch_result.total_count - batch_result.failed_count
        if args.validate_only:
            print(f"Batch validation complete: {passed}/{batch_result.total_count} cases passed.")
            return 2 if batch_result.failed_count else 0

        if batch_result.summary_path is None:
            print(
                f"Batch aborted: {batch_result.failed_count}/{batch_result.total_count} cases "
                "failed validation; no simulations were started.",
                file=sys.stderr,
            )
            return 2

        succeeded = sum(1 for record in batch_result.records if record.status == "success")
        print(
            f"Batch complete: {succeeded}/{batch_result.total_count} cases succeeded. "
            f"Workers: {batch_result.workers}. Summary: {batch_result.summary_path}"
        )
        return 1 if batch_result.failed_count or batch_result.plot_failed_count else 0

    case = load_case(args.input_yaml)
    if args.validate_only:
        print(f"Validation passed: {case.run_path}")
        return 0

    artifact_paths = {}
    if args.artifacts:
        from .artifacts import generate_artifacts

        artifact_paths = generate_artifacts(case)
    from .simulation import run_case

    run_result = run_case(
        case,
        artifact_paths=artifact_paths,
        retain_reporter=args.dae_plotter,
    )
    print(f"Simulation took {run_result.runtime_s:.3f} seconds.")

    from .reports import format_balance_error_lines

    for line in format_balance_error_lines(run_result.balance_errors):
        print(line)
    for plot_id, error in run_result.plot_errors.items():
        print(f"plot '{plot_id}' failed: {error}", file=sys.stderr)
    exit_code = 1 if run_result.plot_errors else 0
    if args.dae_plotter:
        print("Opening DAETools plotter. Close the plotter window to exit.")
        return launch_daetools_plotter(run_result) or exit_code
    return exit_code


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        return _run_cli(arguments)
    except PackedBedValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        if "--debug" in arguments:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
