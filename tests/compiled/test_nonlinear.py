from __future__ import annotations
import numpy as np
import pytest


@pytest.mark.parametrize("scenario", ("refresh", "residual", "linear", "fatal"))
def test_nonlinear_refresh_retains_valid_iterate_and_propagates_errors(
    native_tools, tmp_path, scenario
):
    import ctypes as C
    from types import SimpleNamespace

    from packed_bed.compiled.nonlinear import (
        make_nonlinear_solver,
        nonlinear_statistics,
        prepare_nonlinear_library,
    )
    from packed_bed.compiled.runtime import D, I, L, NativeIDA, P, check_runtime

    folder = check_runtime()
    libs = {
        name: C.CDLL(str(next(folder.glob(f"sundials_{name}-*.dll"))))
        for name in ("core", "ida", "nvecserial")
    }

    def bind(owner, name, result, *arguments):
        function = getattr(libs[owner], name)
        function.argtypes, function.restype = arguments, result
        return function

    new_context = bind("core", "SUNContext_Create", I, I, C.POINTER(P))
    free_context = bind("core", "SUNContext_Free", I, C.POINTER(P))
    new_vector = bind("nvecserial", "N_VNew_Serial", P, I, P)
    destroy = bind("core", "N_VDestroy", None, P)
    data = bind("core", "N_VGetArrayPointer", C.POINTER(D), P)
    solve = bind("core", "SUNNonlinSolSolve", I, P, P, P, P, D, I, P)
    get_failures = bind("core", "SUNNonlinSolGetNumConvFails", I, P, C.POINTER(L))
    helper, _ = prepare_nonlinear_library(tmp_path)
    context = P()
    assert new_context(0, C.byref(context)) == 0
    y, weights = new_vector(1, context), new_vector(1, context)
    nls = None
    try:
        data(y)[0], data(weights)[0] = 0.0, 1.0
        runtime = SimpleNamespace(
            ctx=context,
            y=y,
            libs=libs,
            check=NativeIDA.check,
            n=1,
            N_VGetArrayPointer=data,
        )
        nls = make_nonlinear_solver(runtime, helper, 2)
        current, inverse, linear_calls, setup_points = [1.0], [0.05], [0], []
        if scenario == "residual":
            inverse[0] = 1.0

        @C.CFUNCTYPE(I, P, P, P)
        def system(correction, residual, memory):
            current[0] = 1.0 + data(correction)[0]
            data(residual)[0] = 2.0 * current[0] - 4.0
            if scenario == "fatal":
                return -1
            return int(scenario == "residual" and current[0] > 2.5)

        @C.CFUNCTYPE(I, I, C.POINTER(I), P)
        def setup(bad, fresh, memory):
            setup_points.append(current[0])
            inverse[0], fresh[0] = 0.5, 1
            return 0

        @C.CFUNCTYPE(I, P, P)
        def linear(delta, memory):
            linear_calls[0] += 1
            if scenario == "linear" and linear_calls[0] == 1:
                return 1
            data(delta)[0] *= inverse[0]
            return 0

        @C.CFUNCTYPE(I, P, P, P, D, P, P)
        def convergence(solver, correction, delta, tolerance, weights, memory):
            return 0 if abs(data(delta)[0]) <= tolerance else 901

        for name, callback in (
            ("SetSysFn", system),
            ("SetLSetupFn", setup),
            ("SetLSolveFn", linear),
        ):
            assert bind("core", "SUNNonlinSol" + name, I, P, P)(nls, callback) == 0
        assert (
            bind("core", "SUNNonlinSolSetConvTestFn", I, P, P, P)(
                nls, convergence, None
            )
            == 0
        )
        assert bind("core", "SUNNonlinSolSetMaxIters", I, P, I)(nls, 10) == 0
        assert bind("core", "SUNNonlinSolInitialize", I, P)(nls) == 0
        flag = solve(nls, y, y, weights, 1e-10, 0, None)
        counters = nonlinear_statistics(helper, nls)
        failures = L()
        assert get_failures(nls, C.byref(failures)) == 0
        if scenario == "fatal":
            assert flag == -1 and setup_points == []
            assert counters["NumNonlinearRestarts"] == 0
        else:
            assert flag == 0
            assert 1.0 + data(y)[0] == pytest.approx(2.0, abs=1e-12)
            assert counters["NumNonlinearRestarts"] == 1
            if scenario == "refresh":
                assert setup_points == pytest.approx([1.1])
                assert counters["NumJacobianRefreshes"] == 1
                assert failures.value == 0
            else:
                assert setup_points == [1.0]
                assert counters["NumJacobianRefreshes"] == 0
                assert failures.value == 1
    finally:
        if nls is not None:
            assert helper.free_solver(nls) == 0
        destroy(y)
        destroy(weights)
        assert free_context(C.byref(context)) == 0


