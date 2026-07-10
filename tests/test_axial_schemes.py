from __future__ import annotations

import pytest

from packed_bed.axial_schemes import (
    SUPPORTED_SCHEMES,
    reconstruct_face_states,
    split_face_flux,
)


def test_only_the_retained_schemes_are_available() -> None:
    assert SUPPORTED_SCHEMES == (
        "upwind1",
        "central",
        "linear_upwind2",
        "muscl_minmod",
        "weno3",
        "weno5",
    )


@pytest.mark.parametrize("scheme", SUPPORTED_SCHEMES)
def test_all_schemes_preserve_constant_states(scheme: str) -> None:
    for face_index in range(1, 7):
        assert reconstruct_face_states(
            lambda _index: 3.25,
            face_index,
            7,
            scheme,
            1.0e-8,
        ) == pytest.approx((3.25, 3.25))


@pytest.mark.parametrize(
    "scheme",
    ("central", "linear_upwind2", "muscl_minmod", "weno3", "weno5"),
)
def test_higher_order_schemes_reconstruct_a_linear_interior_state(scheme: str) -> None:
    left, right = reconstruct_face_states(
        lambda cell_index: 2.0 * (cell_index + 0.5) - 1.0,
        3,
        7,
        scheme,
        1.0e-8,
    )

    assert left == pytest.approx(5.0)
    assert right == pytest.approx(5.0)


@pytest.mark.parametrize("scheme", SUPPORTED_SCHEMES)
def test_boundary_stencils_never_read_outside_the_cells(scheme: str) -> None:
    values = (2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0)

    def bounded_value(index: int) -> float:
        if index < 0 or index >= len(values):
            raise AssertionError(f"out-of-range stencil index: {index}")
        return values[index]

    reconstruct_face_states(bounded_value, 1, len(values), scheme, 1.0e-8)
    reconstruct_face_states(bounded_value, len(values) - 1, len(values), scheme, 1.0e-8)


@pytest.mark.parametrize("scheme", SUPPORTED_SCHEMES)
def test_left_and_right_reconstruction_are_mirrored(scheme: str) -> None:
    values = (1.0, 1.5, 4.0, 3.0, 8.0, 6.0, 9.0)
    face_index = 3
    left, right = reconstruct_face_states(
        values.__getitem__,
        face_index,
        len(values),
        scheme,
        1.0e-8,
    )
    mirrored = tuple(reversed(values))
    mirrored_left, mirrored_right = reconstruct_face_states(
        mirrored.__getitem__,
        len(values) - face_index,
        len(values),
        scheme,
        1.0e-8,
    )

    assert left == pytest.approx(mirrored_right)
    assert right == pytest.approx(mirrored_left)


@pytest.mark.parametrize(
    ("velocity", "expected"),
    ((2.0, 20.0), (-2.0, -40.0), (0.0, 0.0)),
)
def test_convective_flux_selects_the_upstream_state(velocity: float, expected: float) -> None:
    assert split_face_flux(velocity, 10.0, 20.0) == expected
