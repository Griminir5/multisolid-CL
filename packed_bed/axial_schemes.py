"""
Reusable axial face-reconstruction schemes for DAETOOLS finite-volume models.
"""

from daetools.pyDAE import Max, Min


__all__ = (
    "SUPPORTED_SCHEMES",
    "validate_scheme_name",
    "reconstruct_face_left_value",
    "reconstruct_face_right_value",
    "reconstruct_face_states",
)


SUPPORTED_SCHEMES = (
    "upwind1",
    "central",
    "linear_upwind2",
    "linear_upwind5",
    "muscl_minmod",
    "weno3",
    "weno5",
    "weno7",
)


def validate_scheme_name(scheme_name):
    if scheme_name not in SUPPORTED_SCHEMES:
        raise ValueError(
            f"Unsupported axial scheme '{scheme_name}'. "
            f"Choose from {', '.join(SUPPORTED_SCHEMES)}."
        )
    return scheme_name


def _minmod(delta_up, delta_dn, zero_like):
    return Max(zero_like, Min(delta_up, delta_dn)) + Min(zero_like, Max(delta_up, delta_dn))


def _weno3_left_value(phi_LL, phi_L, phi_R, small_eps):
    q0 = 1.5 * phi_L - 0.5 * phi_LL
    q1 = 0.5 * (phi_L + phi_R)
    beta0 = (phi_L - phi_LL) ** 2
    beta1 = (phi_R - phi_L) ** 2
    alpha0 = (1.0 / 3.0) / (beta0 + small_eps**2) ** 2
    alpha1 = (2.0 / 3.0) / (beta1 + small_eps**2) ** 2
    return (alpha0 * q0 + alpha1 * q1) / (alpha0 + alpha1)


def _weno3_right_value(phi_L, phi_R, phi_RR, small_eps):
    q0 = 1.5 * phi_R - 0.5 * phi_RR
    q1 = 0.5 * (phi_L + phi_R)
    beta0 = (phi_RR - phi_R) ** 2
    beta1 = (phi_R - phi_L) ** 2
    alpha0 = (1.0 / 3.0) / (beta0 + small_eps**2) ** 2
    alpha1 = (2.0 / 3.0) / (beta1 + small_eps**2) ** 2
    return (alpha0 * q0 + alpha1 * q1) / (alpha0 + alpha1)


def _weno5_value(v0, v1, v2, v3, v4, small_eps):
    q0 = (1.0 / 3.0) * v0 - (7.0 / 6.0) * v1 + (11.0 / 6.0) * v2
    q1 = -(1.0 / 6.0) * v1 + (5.0 / 6.0) * v2 + (1.0 / 3.0) * v3
    q2 = (1.0 / 3.0) * v2 + (5.0 / 6.0) * v3 - (1.0 / 6.0) * v4

    beta0 = (13.0 / 12.0) * (v0 - 2.0 * v1 + v2) ** 2 + 0.25 * (v0 - 4.0 * v1 + 3.0 * v2) ** 2
    beta1 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2 + 0.25 * (v1 - v3) ** 2
    beta2 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2 + 0.25 * (3.0 * v2 - 4.0 * v3 + v4) ** 2

    alpha0 = 0.1 / (beta0 + small_eps**2) ** 2
    alpha1 = 0.6 / (beta1 + small_eps**2) ** 2
    alpha2 = 0.3 / (beta2 + small_eps**2) ** 2
    alpha_sum = alpha0 + alpha1 + alpha2
    return (alpha0 * q0 + alpha1 * q1 + alpha2 * q2) / alpha_sum


def _weno7_value(v0, v1, v2, v3, v4, v5, v6, small_eps):
    q0 = -0.25 * v0 + (13.0 / 12.0) * v1 - (23.0 / 12.0) * v2 + (25.0 / 12.0) * v3
    q1 = (1.0 / 12.0) * v1 - (5.0 / 12.0) * v2 + (13.0 / 12.0) * v3 + 0.25 * v4
    q2 = -(1.0 / 12.0) * v2 + (7.0 / 12.0) * v3 + (7.0 / 12.0) * v4 - (1.0 / 12.0) * v5
    q3 = 0.25 * v3 + (13.0 / 12.0) * v4 - (5.0 / 12.0) * v5 + (1.0 / 12.0) * v6

    beta0 = (
        547.0 * v0**2
        - 3882.0 * v0 * v1
        + 4642.0 * v0 * v2
        - 1854.0 * v0 * v3
        + 7043.0 * v1**2
        - 17246.0 * v1 * v2
        + 7042.0 * v1 * v3
        + 11003.0 * v2**2
        - 9402.0 * v2 * v3
        + 2107.0 * v3**2
    ) / 240.0
    beta1 = (
        267.0 * v1**2
        - 1642.0 * v1 * v2
        + 1602.0 * v1 * v3
        - 494.0 * v1 * v4
        + 2843.0 * v2**2
        - 5966.0 * v2 * v3
        + 1922.0 * v2 * v4
        + 3443.0 * v3**2
        - 2522.0 * v3 * v4
        + 547.0 * v4**2
    ) / 240.0
    beta2 = (
        547.0 * v2**2
        - 2522.0 * v2 * v3
        + 1922.0 * v2 * v4
        - 494.0 * v2 * v5
        + 3443.0 * v3**2
        - 5966.0 * v3 * v4
        + 1602.0 * v3 * v5
        + 2843.0 * v4**2
        - 1642.0 * v4 * v5
        + 267.0 * v5**2
    ) / 240.0
    beta3 = (
        2107.0 * v3**2
        - 9402.0 * v3 * v4
        + 7042.0 * v3 * v5
        - 1854.0 * v3 * v6
        + 11003.0 * v4**2
        - 17246.0 * v4 * v5
        + 4642.0 * v4 * v6
        + 7043.0 * v5**2
        - 3882.0 * v5 * v6
        + 547.0 * v6**2
    ) / 240.0

    alpha0 = (1.0 / 35.0) / (beta0 + small_eps**2) ** 2
    alpha1 = (12.0 / 35.0) / (beta1 + small_eps**2) ** 2
    alpha2 = (18.0 / 35.0) / (beta2 + small_eps**2) ** 2
    alpha3 = (4.0 / 35.0) / (beta3 + small_eps**2) ** 2
    alpha_sum = alpha0 + alpha1 + alpha2 + alpha3
    return (alpha0 * q0 + alpha1 * q1 + alpha2 * q2 + alpha3 * q3) / alpha_sum


def _linear_upwind5_value(v0, v1, v2, v3, v4):
    return (2.0 * v0 - 13.0 * v1 + 47.0 * v2 + 27.0 * v3 - 3.0 * v4) / 60.0


def reconstruct_face_left_value(cell_accessor, face_index, n_cells, scheme_name, small_eps):
    scheme_name = validate_scheme_name(scheme_name)

    idx_cell_L = face_index - 1
    idx_cell_R = face_index

    phi_L = cell_accessor(idx_cell_L)
    phi_R = cell_accessor(idx_cell_R)

    if scheme_name == "upwind1":
        return phi_L

    if scheme_name == "central":
        return 0.5 * (phi_L + phi_R)

    if face_index < 2:
        return phi_L

    phi_LL = cell_accessor(idx_cell_L - 1)

    if scheme_name == "linear_upwind2":
        return 1.5 * phi_L - 0.5 * phi_LL

    if scheme_name == "linear_upwind5":
        if face_index < 3 or (n_cells is not None and face_index > n_cells - 2):
            return 1.5 * phi_L - 0.5 * phi_LL
        phi_LLL = cell_accessor(idx_cell_L - 2)
        phi_RR = cell_accessor(idx_cell_R + 1)
        return _linear_upwind5_value(phi_LLL, phi_LL, phi_L, phi_R, phi_RR)

    if scheme_name == "muscl_minmod":
        zero_like = 0.0 * small_eps
        delta_up = phi_L - phi_LL
        delta_dn = phi_R - phi_L
        slope = _minmod(delta_up, delta_dn, zero_like)
        return phi_L + 0.5 * slope

    if scheme_name == "weno3":
        return _weno3_left_value(phi_LL, phi_L, phi_R, small_eps)

    if scheme_name == "weno5":
        if face_index < 3 or (n_cells is not None and face_index > n_cells - 2):
            return _weno3_left_value(phi_LL, phi_L, phi_R, small_eps)
        phi_LLL = cell_accessor(idx_cell_L - 2)
        phi_RR = cell_accessor(idx_cell_R + 1)
        return _weno5_value(phi_LLL, phi_LL, phi_L, phi_R, phi_RR, small_eps)

    if scheme_name == "weno7":
        if face_index < 4 or (n_cells is not None and face_index > n_cells - 3):
            return reconstruct_face_left_value(cell_accessor, face_index, "weno5", small_eps, n_cells=n_cells)
        phi_LLLL = cell_accessor(idx_cell_L - 3)
        phi_LLL = cell_accessor(idx_cell_L - 2)
        phi_RR = cell_accessor(idx_cell_R + 1)
        phi_RRR = cell_accessor(idx_cell_R + 2)
        return _weno7_value(phi_LLLL, phi_LLL, phi_LL, phi_L, phi_R, phi_RR, phi_RRR, small_eps)

    raise RuntimeError(f"Unsupported scheme '{scheme_name}' passed reconstruction.")


def reconstruct_face_right_value(cell_accessor, face_index, n_cells, scheme_name, small_eps):
    scheme_name = validate_scheme_name(scheme_name)

    idx_cell_L = face_index - 1
    idx_cell_R = face_index

    phi_L = cell_accessor(idx_cell_L)
    phi_R = cell_accessor(idx_cell_R)

    if scheme_name == "upwind1":
        return phi_R

    if scheme_name == "central":
        return 0.5 * (phi_L + phi_R)

    if face_index > n_cells - 2:
        return phi_R

    phi_RR = cell_accessor(idx_cell_R + 1)

    if scheme_name == "linear_upwind2":
        return 1.5 * phi_R - 0.5 * phi_RR

    if scheme_name == "linear_upwind5":
        if face_index < 2 or face_index > n_cells - 3:
            return 1.5 * phi_R - 0.5 * phi_RR
        phi_RRR = cell_accessor(idx_cell_R + 2)
        phi_LL = cell_accessor(idx_cell_L - 1)
        return _linear_upwind5_value(phi_RRR, phi_RR, phi_R, phi_L, phi_LL)

    if scheme_name == "muscl_minmod":
        zero_like = 0.0 * small_eps
        delta_up = phi_RR - phi_R
        delta_dn = phi_R - phi_L
        slope = _minmod(delta_up, delta_dn, zero_like)
        return phi_R - 0.5 * slope

    if scheme_name == "weno3":
        return _weno3_right_value(phi_L, phi_R, phi_RR, small_eps)

    if scheme_name == "weno5":
        if face_index < 2 or face_index > n_cells - 3:
            return _weno3_right_value(phi_L, phi_R, phi_RR, small_eps)
        phi_RRR = cell_accessor(idx_cell_R + 2)
        phi_LL = cell_accessor(idx_cell_L - 1)
        return _weno5_value(phi_RRR, phi_RR, phi_R, phi_L, phi_LL, small_eps)

    if scheme_name == "weno7":
        if face_index < 3 or face_index > n_cells - 4:
            return reconstruct_face_right_value(cell_accessor, face_index, n_cells, "weno5", small_eps)
        phi_RRR = cell_accessor(idx_cell_R + 2)
        phi_RRRR = cell_accessor(idx_cell_R + 3)
        phi_LL = cell_accessor(idx_cell_L - 1)
        phi_LLL = cell_accessor(idx_cell_L - 2)
        return _weno7_value(phi_RRRR, phi_RRR, phi_RR, phi_R, phi_L, phi_LL, phi_LLL, small_eps)

    raise RuntimeError(f"Unsupported scheme '{scheme_name}' passed reconstruction.")


def reconstruct_face_states(cell_accessor, face_index, n_cells, scheme_name, small_eps):
    return (
        reconstruct_face_left_value(cell_accessor, face_index, n_cells, scheme_name, small_eps),
        reconstruct_face_right_value(cell_accessor, face_index, n_cells, scheme_name, small_eps),
    )
