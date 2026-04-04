"""
Reusable axial face-reconstruction schemes for DAETOOLS finite-volume models.

The functions in this module return symbolic DAETOOLS expressions, so they can
be used directly inside equation residuals. The intended pattern for general
convective fluxes with possible flow reversal is:

    phi_L, phi_R = reconstruct_face_states(field, face_index, n_cells, scheme, eps)
    flux = uplus * phi_L + uminus * phi_R

For strictly positive flow, reconstruct_face_left_value(...) is sufficient.
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
    "muscl_minmod",
    "weno3",
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


def reconstruct_face_left_value(cell_accessor, face_index, scheme_name, small_eps):
    """
    Left-biased reconstruction at an interior face.

    This is the state to multiply by the positive-flow contribution `uplus`.
    `cell_accessor(i)` must return the cell-centered symbolic variable or
    expression at index `i`.
    """

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

    if scheme_name == "muscl_minmod":
        zero_like = 0.0 * small_eps
        delta_up = phi_L - phi_LL
        delta_dn = phi_R - phi_L
        slope = _minmod(delta_up, delta_dn, zero_like)
        return phi_L + 0.5 * slope

    if scheme_name == "weno3":
        q0 = 1.5 * phi_L - 0.5 * phi_LL
        q1 = 0.5 * (phi_L + phi_R)
        beta0 = (phi_L - phi_LL) ** 2
        beta1 = (phi_R - phi_L) ** 2
        alpha0 = (1.0 / 3.0) / (beta0 + small_eps ** 2) ** 2
        alpha1 = (2.0 / 3.0) / (beta1 + small_eps ** 2) ** 2
        return (alpha0 * q0 + alpha1 * q1) / (alpha0 + alpha1)

    raise RuntimeError(f"Unsupported scheme '{scheme_name}' passed reconstruction.")


def reconstruct_face_right_value(cell_accessor, face_index, n_cells, scheme_name, small_eps):
    """
    Right-biased reconstruction at an interior face.

    This is the state to multiply by the negative-flow contribution `uminus`.
    """

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

    if scheme_name == "muscl_minmod":
        zero_like = 0.0 * small_eps
        delta_up = phi_RR - phi_R
        delta_dn = phi_R - phi_L
        slope = _minmod(delta_up, delta_dn, zero_like)
        return phi_R - 0.5 * slope

    if scheme_name == "weno3":
        q0 = 1.5 * phi_R - 0.5 * phi_RR
        q1 = 0.5 * (phi_L + phi_R)
        beta0 = (phi_RR - phi_R) ** 2
        beta1 = (phi_R - phi_L) ** 2
        alpha0 = (1.0 / 3.0) / (beta0 + small_eps ** 2) ** 2
        alpha1 = (2.0 / 3.0) / (beta1 + small_eps ** 2) ** 2
        return (alpha0 * q0 + alpha1 * q1) / (alpha0 + alpha1)

    raise RuntimeError(f"Unsupported scheme '{scheme_name}' passed reconstruction.")


def reconstruct_face_states(cell_accessor, face_index, n_cells, scheme_name, small_eps):
    """
    Return `(phi_L, phi_R)` reconstructed at an interior face.

    `phi_L` is the left-biased state for positive velocity and `phi_R` is the
    right-biased state for negative velocity.
    """

    return (
        reconstruct_face_left_value(cell_accessor, face_index, scheme_name, small_eps),
        reconstruct_face_right_value(cell_accessor, face_index, n_cells, scheme_name, small_eps),
    )
