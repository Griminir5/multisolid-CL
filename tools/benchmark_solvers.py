"""Compatibility helpers for the documented solver benchmark cases."""

from pathlib import Path
import yaml
from .benchmark_compiled import prepare_case as _prepare


def prepare_case(documents, folder, profile):
    return _prepare(documents, folder, profile)[0]


def scenarios(scope):
    from packed_bed.config import load_case
    root = Path(__file__).resolve().parents[1] / "packed_bed/examples"
    if scope in {"default", "all"}:
        case = load_case(root / "default_case/run.yaml")
        documents = {name: yaml.safe_load(getattr(case, name + "_path").read_text()) for name in ("run", "program", "solids", "chemistry")}
        yield "default", "default", documents, {}


if __name__ == "__main__":
    from .benchmark_compiled import main
    main()
