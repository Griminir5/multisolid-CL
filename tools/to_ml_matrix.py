"""Convert explicitly selected labelled result variables into a time-by-feature CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stack selected results.nc variables into an ML-oriented CSV matrix."
    )
    parser.add_argument("results_nc", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--variables",
        nargs="+",
        required=True,
        help="Dataset variable names to include; no permanent feature list is assumed.",
    )
    return parser


def to_ml_matrix(dataset: xr.Dataset, variable_names: list[str]):
    unknown = sorted(set(variable_names) - set(dataset.data_vars))
    if unknown:
        raise ValueError(f"Unknown result variables: {', '.join(unknown)}.")
    without_time = [name for name in variable_names if "time" not in dataset[name].dims]
    if without_time:
        raise ValueError(f"Variables do not depend on time: {', '.join(without_time)}.")
    return dataset[variable_names].to_stacked_array("feature", sample_dims=("time",))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    dataset = xr.load_dataset(args.results_nc, engine="scipy")
    matrix = to_ml_matrix(dataset, args.variables).to_pandas()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
