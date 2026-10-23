"""A standardized PCA projection of every recorded time-dependent report."""

from __future__ import annotations

from pathlib import Path

from .definitions import PlotSpec, pyplot, save_figure


def pca_scores(dataset):
    """Return two scores per saved time and their explained-variance fractions.

    Each species/location in each numeric report output is a separate feature.
    Coordinates (including time) and variables without time are not features.
    Fit one PCA basis across the entire run, using population-standardized
    features. Missing values are rejected instead of dropping timesteps.
    """
    import numpy as np

    if "time" not in dataset.coords or dataset.time.dims != ("time",):
        raise ValueError("PCA trajectory requires a one-dimensional time coordinate.")
    time = np.asarray(dataset.time.values, dtype=float)
    if not time.size or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("PCA trajectory requires finite, increasing reporting times.")

    features = []
    for name in sorted(dataset.data_vars):
        variable = dataset[name]
        if "time" not in variable.dims or variable.dtype.kind not in "biuf":
            continue
        values = np.asarray(variable.transpose("time", ...).values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"PCA trajectory requires finite values in report variable '{name}'.")
        features.append(values.reshape(time.size, -1))
    if not features or not any(values.shape[1] for values in features):
        raise ValueError("PCA trajectory requires at least one recorded numeric report with time samples.")

    matrix = np.concatenate(features, axis=1)
    # Test constants before centering: e.g. repeated 0.1 can acquire a tiny
    # nonzero variance from rounding its mean. Constant columns carry no signal.
    matrix = matrix[:, np.any(matrix != matrix[0], axis=0)]
    scores = np.zeros((time.size, 2))
    explained = np.zeros(2)
    if not matrix.shape[1]:
        return scores, explained

    # Rescale first to avoid overflow/underflow for reports with extreme units.
    # This cancels out when each column is standardized to unit variance.
    matrix /= np.max(np.abs(matrix), axis=0)
    matrix -= matrix.mean(axis=0)
    matrix /= matrix.std(axis=0)
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    tolerance = np.finfo(float).eps * max(matrix.shape) * singular_values[0]
    components = min(2, int(np.count_nonzero(singular_values > tolerance)))
    for index in range(components):
        # Fix SVD's arbitrary sign using the largest loading for stable plots.
        pivot = np.argmax(np.abs(right[index]))
        sign = 1.0 if right[index, pivot] >= 0 else -1.0
        scores[:, index] = sign * left[:, index] * singular_values[index]
    explained[:components] = singular_values[:components] ** 2 / np.sum(singular_values ** 2)
    return scores, explained


def render(dataset, path: Path) -> None:
    scores, explained = pca_scores(dataset)
    plt = pyplot()
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.plot(scores[:, 0], scores[:, 1], color="0.55", linewidth=1, alpha=0.6, zorder=1)
    points = axis.scatter(
        scores[:, 0], scores[:, 1], c=dataset.time.values, cmap="viridis", s=18, zorder=2,
        rasterized=True,
    )
    figure.colorbar(points, ax=axis, label="Time [s]")
    for index, marker, label in ((0, "o", "Start"), (-1, "X", "End")):
        axis.scatter(
            scores[index, 0], scores[index, 1], marker=marker, s=110,
            facecolors="none", edgecolors="black", linewidths=1.5, label=label, zorder=3,
        )
    axis.set(
        title="PCA Trajectory of Selected Reports",
        xlabel=f"PC1 ({explained[0]:.1%} explained variance)",
        ylabel=f"PC2 ({explained[1]:.1%} explained variance)",
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    if not explained.any():
        axis.text(0.5, 0.95, "No variation across reporting times", ha="center", transform=axis.transAxes)
    figure.tight_layout()
    save_figure(figure, path)


PLOT = PlotSpec(
    id="pca_trajectory",
    description=(
        "Two-dimensional PCA trajectory of all selected reports. Each species/location "
        "is standardized over time; points are connected in time order and colored by time."
    ),
    required_reports=(),
    filename="pca_trajectory.svg",
    render=render,
    title="PCA trajectory",
)
