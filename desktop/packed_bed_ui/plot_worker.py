"""Read or render a retained plot outside the desktop process."""

import hashlib
import json
from pathlib import Path


def _digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def render_plot(results_path, plot_id, destination):
    from packed_bed.plotting import PLOT_REGISTRY

    results_path, destination = Path(results_path), Path(destination)
    spec = PLOT_REGISTRY[plot_id]
    # Desktop snapshots use this fixed location. Manifest paths may still refer
    # to the original project after a move, so use hashes and local paths.
    saved = results_path.parent / "artifacts" / spec.filename
    try:
        manifest = json.loads((results_path.parent / "manifest.json").read_text(encoding="utf-8"))
        outputs = manifest["outputs"]
        if (outputs[plot_id]["status"] == "success"
                and outputs["results"]["sha256"] == _digest(results_path)):
            svg = saved.read_bytes()
            if outputs[plot_id]["sha256"] == hashlib.sha256(svg).hexdigest():
                destination.write_bytes(svg)
                return
    except (OSError, ValueError, KeyError, TypeError):
        # Missing/stale artifacts are normal for unrequested plots or old runs.
        pass

    import xarray as xr

    with xr.open_dataset(results_path, engine="scipy") as source:
        dataset = source.load()
    spec.render(dataset, destination)


def main(results_path, plot_id, destination):
    import sys

    try:
        render_plot(results_path, plot_id, destination)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0
