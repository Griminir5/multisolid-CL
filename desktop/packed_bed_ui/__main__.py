"""One entry point for the desktop window and its separate solver worker."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    from multiprocessing import freeze_support

    freeze_support()
    parser = argparse.ArgumentParser(description="MultiSolid desktop")
    parser.add_argument("project", nargs="?", type=Path, help="Project folder or project.json")
    parser.add_argument("--worker", type=Path, metavar="RUN_FOLDER", help=argparse.SUPPRESS)
    parser.add_argument("--project-worker", type=Path, metavar="EXECUTION_FILE", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.project_worker is not None:
        from .worker import run_project_job

        return run_project_job(args.project_worker)
    if args.worker is not None:
        from .worker import run_snapshot

        return run_snapshot(args.worker)

    from PyQt6.QtWidgets import QApplication
    from .window import MainWindow

    app = QApplication(["MultiSolid"])
    app.setApplicationName("MultiSolid")
    window = MainWindow()
    if args.project is not None:
        window.open_project(args.project)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
