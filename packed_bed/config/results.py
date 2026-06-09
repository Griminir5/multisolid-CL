from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import RunBundle


@dataclass(frozen=True)
class RunResult:
    run_bundle: RunBundle
    output_directory: Path
    success: bool
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    report_paths: dict[str, Path] = field(default_factory=dict)
    balance_errors: dict[str, Any] = field(default_factory=dict)
    summary_path: Path | None = None
    balances_path: Path | None = None
    reporter: Any | None = None
    simulation: Any | None = None
