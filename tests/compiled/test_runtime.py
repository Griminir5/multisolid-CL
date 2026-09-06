from __future__ import annotations
import numpy as np
import pytest


@pytest.mark.parametrize(
    "linear_solver", ("superlu", "band", "band_scalar", "band_avx2", "band_reciprocal")
)
@pytest.mark.parametrize("step_growth_threshold", (2.0, 1.25))
@pytest.mark.parametrize(
    "nonlinear_refresh_interval", (0, 2),
)
def test_scaled_native_callbacks_preserve_known_dae_solution(
    native_tools,
    tmp_path,
    linear_solver,
    step_growth_threshold,
    nonlinear_refresh_interval,
):
    from scipy.sparse import csc_matrix

    from packed_bed.compiled import _load_kernel
    from packed_bed.compiled.compiler import compile_kernel
    from packed_bed.compiled.graph import Graph
    from packed_bed.compiled.runtime import NativeIDA, callback_source

    nonlinear_library = None
    if nonlinear_refresh_interval:
        from packed_bed.compiled.nonlinear import prepare_nonlinear_library

        nonlinear_library, _ = prepare_nonlinear_library(tmp_path)

    band_library = None
    if linear_solver in {"band_scalar", "band_avx2", "band_reciprocal"}:
        from packed_bed.compiled.band import prepare_band_library, supports_avx2

        vectorize = linear_solver != "band_scalar"
        if vectorize and not supports_avx2():
            pytest.skip("AVX2 is unavailable")
        band_library, _ = prepare_band_library(
            tmp_path,
            vectorize=vectorize,
            reuse_diagonal=linear_solver == "band_reciprocal",
        )
        linear_solver = "band"

    graph = Graph()
    x = graph.make("var", 0)
    z = graph.make("var", 1)
    roots = [
        graph.make("add", graph.make("dot", 0), x),
        graph.make("sub", z, graph.make("mul", x, x)),
    ]
    sparsity = csc_matrix(np.array([[1.0, 0.0], [1.0, 1.0]]))
    jac = [
        graph.gradient(roots[i])[j]
        for j in range(2)
        for i in sparsity.indices[sparsity.indptr[j] : sparsity.indptr[j + 1]]
    ]
    source = "#include <cmath>\n#include <cstring>\n"
    source += graph.emit("evaluate", roots, {0: 0, 1: 1})
    source += graph.emit("jacobian", jac, {0: 0, 1: 1})
    source += graph.emit("reconstruct", [x, z], {0: 0, 1: 1})
    source += callback_source(sparsity, linear_solver=linear_solver)
    library, _ = compile_kernel(source, tmp_path)
    times = np.linspace(0.0, 2.0, 21)
    with NativeIDA(
        _load_kernel(library),
        [1, 1],
        [-1, -2],
        [1e-10, 1e-10],
        [True, False],
        sparsity,
        rtol=1e-8,
        row_scale=[1e-7, 1e8],
        linear_solver=linear_solver,
        band_library=band_library,
        step_growth_threshold=step_growth_threshold,
        nonlinear_refresh_interval=nonlinear_refresh_interval,
        nonlinear_library=nonlinear_library,
    ) as solver:
        initial, initial_derivatives = solver.solve([0.0])
        np.testing.assert_array_equal(initial, [[1.0, 1.0]])
        np.testing.assert_array_equal(initial_derivatives, [[-1.0, -2.0]])
        for invalid in ([], [0.0, 0.0], [1.0, 2.0], [0.0, np.nan], [[0.0, 1.0]]):
            with pytest.raises(ValueError, match="Reporting times"):
                solver.solve(invalid)
        values, _ = solver.solve(times)
    np.testing.assert_allclose(values[:, 0], np.exp(-times), rtol=0, atol=2e-6)
    np.testing.assert_allclose(values[:, 1], np.exp(-2 * times), rtol=0, atol=2e-6)


@pytest.mark.parametrize("threshold", (0.0, 0.99, float("inf"), float("nan")))
def test_native_step_growth_threshold_validation(threshold):
    from packed_bed.compiled.runtime import NativeIDA

    with pytest.raises(ValueError, match="Step growth threshold"):
        NativeIDA(None, None, None, None, None, None, step_growth_threshold=threshold)


@pytest.mark.parametrize("interval", (-1, 1.5, True, float("inf")))
def test_native_nonlinear_refresh_validation(interval):
    from packed_bed.compiled.runtime import NativeIDA

    with pytest.raises(ValueError, match="Nonlinear refresh interval"):
        NativeIDA(
            None, None, None, None, None, None, nonlinear_refresh_interval=interval
        )


def test_native_nonlinear_refresh_requires_matching_helper():
    from packed_bed.compiled.runtime import NativeIDA

    for interval, helper in ((4, None), (0, object())):
        with pytest.raises(ValueError, match="Custom Newton requires"):
            NativeIDA(
                None,
                None,
                None,
                None,
                None,
                None,
                nonlinear_refresh_interval=interval,
                nonlinear_library=helper,
            )


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("failure", ("early_stop", "nonfinite_derivative"))
def test_reporting_rejects_incomplete_or_nonfinite_states(native_tools, tmp_path, native, failure):
    import ctypes as C
    from types import SimpleNamespace
    from scipy.sparse import csc_matrix
    from packed_bed.compiled.compiler import compile_kernel
    from packed_bed.compiled.runtime import NativeIDA, callback_source

    # Inject an IDASolve result through both reporting paths. A positive return
    # code alone is insufficient evidence that tout was reached.
    solver = NativeIDA.__new__(NativeIDA)
    solver.n = 1
    solver.values = np.array([1.])
    solver.derivatives = np.array([-1.])
    solver.mem = solver.y = solver.yp = None
    solver.IDASetStopTime = lambda *args: 0
    solver.kernel = SimpleNamespace()
    signature = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_double, C.POINTER(C.c_double),
                           C.c_void_p, C.c_void_p, C.c_int)

    @signature
    def fake_solve(mem, requested, reached, y, yp, mode):
        reached[0] = requested / 2 if failure == "early_stop" else requested
        if failure == "nonfinite_derivative":
            solver.derivatives[0] = np.nan
        return 1

    solver.IDASolve = fake_solve
    if native:
        source = """
#include <cmath>
#include <cstring>
void evaluate(double,const double*,const double*,double,double*) {}
void jacobian(double,const double*,const double*,double,double*) {}
""" + callback_source(csc_matrix([[1.]]))
        path, _ = compile_kernel(source, tmp_path)
        library = C.CDLL(str(path))
        library.report_loop.argtypes = ([C.c_void_p] * 6 + [C.c_int] * 2
                                       + [C.c_void_p] * 5)
        library.report_loop.restype = C.c_int
        solver.kernel = library
    with pytest.raises(RuntimeError, match="Compiled IDA"):
        solver.solve([0., 1.])


