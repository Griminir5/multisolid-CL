"""Numerically stable regularizations for symbolic rate expressions."""

from .runtime import Abs, Constant, Sqrt


def smooth_positive(value, epsilon: float):
    """Evaluate (x + sqrt(x*x + 4*epsilon*epsilon))/2 without cancellation.

    Negative trial concentrations are possible during integration. Rationalize
    the small positive tail so its value and derivative remain accurate even
    when abs(x) is much larger than epsilon.
    """
    magnitude = Abs(value)
    radius = Sqrt(value**2 + Constant((2.0 * epsilon) ** 2))
    return (
        Constant(0.5) * (value + magnitude)
        + Constant(2.0 * epsilon**2) / (radius + magnitude)
    )
