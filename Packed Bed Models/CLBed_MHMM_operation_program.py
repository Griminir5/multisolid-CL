__doc__ = """
Compatibility shim for the packed-bed operation-program model.

The implementation now lives in the repo-root `packed_bed` package.
"""

import sys

from daetools.pyDAE import daeCreateQtApplication

from packed_bed.programs import ProgramSegment, ProgramStep, ScalarProgram, VectorProgram
from packed_bed.solver import CLBed_mass, build_idas_solver, configure_evaluation_mode, guiRun, simBed


__all__ = [
    "CLBed_mass",
    "ProgramSegment",
    "ProgramStep",
    "ScalarProgram",
    "VectorProgram",
    "build_idas_solver",
    "configure_evaluation_mode",
    "guiRun",
    "simBed",
]


if __name__ == "__main__":
    qtApp = daeCreateQtApplication(sys.argv)
    guiRun(qtApp)
