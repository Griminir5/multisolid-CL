from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chemistry import ChemistryConfig
from .io import resolve_path
from .program import ProgramConfig
from .run import RunConfig
from .solids import SolidConfig


@dataclass(frozen=True)
class RunBundle:
    run_path: Path
    chemistry_path: Path
    solids_path: Path
    program_path: Path
    chemistry: ChemistryConfig
    solids: SolidConfig
    program: ProgramConfig
    run: RunConfig

    @property
    def output_directory(self) -> Path:
        return resolve_path(self.run_path.parent, self.run.outputs.directory)

    @property
    def artifacts_directory(self) -> Path:
        return resolve_path(self.run_path.parent, self.run.outputs.artifacts_directory)
