"""Axial reconstruction and conservative face-flux utilities."""


SUPPORTED_SCHEMES = (
    "upwind1",
    "central",
    "linear_upwind2",
    "muscl_minmod",
    "weno3",
    "weno5",
)


def validate_scheme_name(scheme_name):
    if scheme_name not in SUPPORTED_SCHEMES:
        raise ValueError(
            f"Unsupported axial scheme '{scheme_name}'. "
            f"Choose from {', '.join(SUPPORTED_SCHEMES)}."
        )
    return scheme_name


def split_face_flux(transport_rate, left_state, right_state, *, absolute=abs):
    """Select a left/right state from the sign of a numeric or symbolic rate."""

    magnitude = absolute(transport_rate)
    positive = 0.5 * (transport_rate + magnitude)
    negative = 0.5 * (transport_rate - magnitude)
    return positive * left_state + negative * right_state


def _minmod(delta_up, delta_down, zero_like, minimum, maximum):
    return maximum(zero_like, minimum(delta_up, delta_down)) + minimum(
        zero_like,
        maximum(delta_up, delta_down),
    )


def _weno3(upstream, center, downstream, epsilon):
    q0 = 1.5 * center - 0.5 * upstream
    q1 = 0.5 * (center + downstream)
    beta0 = (center - upstream) ** 2
    beta1 = (downstream - center) ** 2
    alpha0 = (1.0 / 3.0) / (beta0 + epsilon**2) ** 2
    alpha1 = (2.0 / 3.0) / (beta1 + epsilon**2) ** 2
    return (alpha0 * q0 + alpha1 * q1) / (alpha0 + alpha1)


def _weno5(v0, v1, v2, v3, v4, epsilon):
    q0 = (1.0 / 3.0) * v0 - (7.0 / 6.0) * v1 + (11.0 / 6.0) * v2
    q1 = -(1.0 / 6.0) * v1 + (5.0 / 6.0) * v2 + (1.0 / 3.0) * v3
    q2 = (1.0 / 3.0) * v2 + (5.0 / 6.0) * v3 - (1.0 / 6.0) * v4

    beta0 = (13.0 / 12.0) * (v0 - 2.0 * v1 + v2) ** 2 + 0.25 * (
        v0 - 4.0 * v1 + 3.0 * v2
    ) ** 2
    beta1 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2 + 0.25 * (v1 - v3) ** 2
    beta2 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2 + 0.25 * (
        3.0 * v2 - 4.0 * v3 + v4
    ) ** 2

    alpha0 = 0.1 / (beta0 + epsilon**2) ** 2
    alpha1 = 0.6 / (beta1 + epsilon**2) ** 2
    alpha2 = 0.3 / (beta2 + epsilon**2) ** 2
    return (alpha0 * q0 + alpha1 * q1 + alpha2 * q2) / (alpha0 + alpha1 + alpha2)


def reconstruct_face_states(
    cell_value,
    face_index,
    cell_count,
    scheme_name,
    epsilon,
    *,
    minimum=min,
    maximum=max,
):
    """Reconstruct the left and right states at one interior face."""

    validate_scheme_name(scheme_name)
    left_index = face_index - 1
    right_index = face_index
    left_cell = cell_value(left_index)
    right_cell = cell_value(right_index)

    if scheme_name == "upwind1":
        return left_cell, right_cell
    if scheme_name == "central":
        centered = 0.5 * (left_cell + right_cell)
        return centered, centered

    if face_index < 2:
        left_state = left_cell
    else:
        left_left = cell_value(left_index - 1)
        if scheme_name == "linear_upwind2":
            left_state = 1.5 * left_cell - 0.5 * left_left
        elif scheme_name == "muscl_minmod":
            slope = _minmod(
                left_cell - left_left,
                right_cell - left_cell,
                0.0 * epsilon,
                minimum,
                maximum,
            )
            left_state = left_cell + 0.5 * slope
        elif scheme_name == "weno3" or face_index < 3 or face_index > cell_count - 2:
            left_state = _weno3(left_left, left_cell, right_cell, epsilon)
        else:
            left_state = _weno5(
                cell_value(left_index - 2),
                left_left,
                left_cell,
                right_cell,
                cell_value(right_index + 1),
                epsilon,
            )

    if face_index > cell_count - 2:
        right_state = right_cell
    else:
        right_right = cell_value(right_index + 1)
        if scheme_name == "linear_upwind2":
            right_state = 1.5 * right_cell - 0.5 * right_right
        elif scheme_name == "muscl_minmod":
            slope = _minmod(
                right_right - right_cell,
                right_cell - left_cell,
                0.0 * epsilon,
                minimum,
                maximum,
            )
            right_state = right_cell - 0.5 * slope
        elif scheme_name == "weno3" or face_index < 2 or face_index > cell_count - 3:
            right_state = _weno3(right_right, right_cell, left_cell, epsilon)
        else:
            right_state = _weno5(
                cell_value(right_index + 2),
                right_right,
                right_cell,
                left_cell,
                cell_value(left_index - 1),
                epsilon,
            )

    return left_state, right_state


__all__ = (
    "SUPPORTED_SCHEMES",
    "reconstruct_face_states",
    "split_face_flux",
    "validate_scheme_name",
)
