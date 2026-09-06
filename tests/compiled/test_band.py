from __future__ import annotations
import numpy as np
import pytest


@pytest.mark.parametrize("vectorize", (False, True))
@pytest.mark.parametrize(
    "n,upper,lower", ((1, 0, 0), (11, 4, 0), (11, 0, 4), (17, 3, 5))
)
@pytest.mark.parametrize("reuse_diagonal", (False, True))
def test_band_pivoting_workspace_isolation_and_singular_matrix(
    native_tools,
    tmp_path,
    vectorize,
    n,
    upper,
    lower,
    reuse_diagonal,
):
    import ctypes as C
    from contextlib import ExitStack

    from scipy.sparse import csc_matrix

    from packed_bed.compiled import _load_kernel
    from packed_bed.compiled.band import prepare_band_library, supports_avx2
    from packed_bed.compiled.compiler import compile_kernel
    from packed_bed.compiled.graph import Graph
    from packed_bed.compiled.runtime import NativeIDA, callback_source
    from packed_bed.compiled.structure import band_layout

    if vectorize and not supports_avx2():
        pytest.skip("AVX2 is unavailable")
    optimized, metadata = prepare_band_library(
        tmp_path, vectorize=vectorize, reuse_diagonal=reuse_diagonal
    )
    assert metadata["linear_diagonal_reciprocals"] == reuse_diagonal
    rr, cc = np.indices((n, n))
    pattern = (cc - rr <= upper) & (rr - cc <= lower)
    sparse = csc_matrix(pattern)
    graph = Graph()
    roots = [graph.make("dot", i) for i in range(n)]
    mapping = dict(enumerate(range(n)))
    jacobian = [
        graph.make("cj") if row == col else graph.zero
        for col in range(n)
        for row in sparse.indices[sparse.indptr[col] : sparse.indptr[col + 1]]
    ]
    source = "#include <cmath>\n#include <cstring>\n"
    source += graph.emit("evaluate", roots, mapping)
    source += graph.emit("jacobian", jacobian, mapping)
    source += graph.emit(
        "reconstruct", [graph.make("var", i) for i in range(n)], mapping
    )
    source += callback_source(sparse, linear_solver="band")
    path, _ = compile_kernel(source, tmp_path)
    kernel = _load_kernel(path)
    rng = np.random.default_rng(65427)
    a = rng.normal(size=(n, n)) * pattern + np.eye(n) * 4.0
    if lower:
        # Bring the furthest permitted row up to the diagonal, extending U.
        # A lower triangular matrix still needs a nonzero first diagonal.
        a[0, 0] = 0.1 if upper == 0 else 0.0
        a[lower, 0] = 16.0
    b = (
        np.eye(n) * 2.0
    )  # Different, empty margins expose accidentally shared workspaces.
    _, _, stored, leading = band_layout(sparse)
    with ExitStack() as stack:
        solvers = [
            stack.enter_context(
                NativeIDA(
                    kernel,
                    np.zeros(n),
                    np.zeros(n),
                    np.full(n, 1e-8),
                    np.ones(n, dtype=bool),
                    sparse,
                    linear_solver="band",
                    band_library=library,
                )
            )
            for library in (optimized, optimized, None)
        ]
        core = solvers[0].libs["core"]
        setup = core.SUNLinSolSetup
        setup.argtypes = [C.c_void_p, C.c_void_p]
        setup.restype = C.c_int
        solve = core.SUNLinSolSolve
        solve.argtypes = [C.c_void_p] * 4 + [C.c_double]
        solve.restype = C.c_int
        last_flag = core.SUNLinSolLastFlag
        last_flag.argtypes = [C.c_void_p]
        last_flag.restype = C.c_int
        get_id = core.SUNLinSolGetID
        get_id.argtypes = [C.c_void_p]
        get_id.restype = C.c_int
        assert get_id(solvers[0].linear) == get_id(solvers[2].linear)

        def storage(solver):
            return np.ctypeslib.as_array(
                solver.SUNBandMatrix_Data(solver.matrix), shape=(n * leading,)
            ).reshape(n, leading)

        def fill(solver, dense):
            data = storage(solver)
            data[:] = 0.0
            for col in range(n):
                rows = np.arange(max(0, col - upper), min(n, col + lower + 1))
                data[col, stored + rows - col] = dense[rows, col]

        for dense_a in (a, a + np.eye(n) * 0.1):
            for solver, dense in zip(solvers, (dense_a, b, dense_a)):
                fill(solver, dense)
                assert setup(solver.linear, solver.matrix) == 0
            np.testing.assert_array_equal(storage(solvers[0]), storage(solvers[2]))
            rhs = rng.normal(size=n)
            for solver, dense in zip(solvers, (dense_a, b, dense_a)):
                solver.derivatives[:] = rhs
                assert (
                    solve(solver.linear, solver.matrix, solver.y, solver.yp, 0.0) == 0
                )
                np.testing.assert_allclose(
                    dense @ solver.values, rhs, rtol=0, atol=1e-12
                )
            if reuse_diagonal:
                np.testing.assert_allclose(
                    solvers[0].values, solvers[2].values, rtol=2e-13, atol=2e-15
                )
            else:
                np.testing.assert_array_equal(solvers[0].values, solvers[2].values)
        if n == 1:
            # A tiny divisor has an overflowing reciprocal; a huge one has a
            # subnormal reciprocal. Division still gives an accurate finite root.
            for diagonal in (1e-310, 1e308, -1e-310, -1e308, 2.0):
                for solver in (solvers[0], solvers[2]):
                    fill(solver, np.array([[diagonal]]))
                    assert setup(solver.linear, solver.matrix) == 0
                    solver.derivatives[:] = diagonal
                    assert (
                        solve(solver.linear, solver.matrix, solver.y, solver.yp, 0.0)
                        == 0
                    )
                    np.testing.assert_array_equal(solver.values, [1.0])
        for solver in solvers:
            fill(solver, np.zeros((n, n)))
            assert setup(solver.linear, solver.matrix) > 0
            assert last_flag(solver.linear) > 0
