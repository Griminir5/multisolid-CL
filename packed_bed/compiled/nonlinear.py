"""Owned Newton solver with early Jacobian refresh."""

import ctypes as C
from pathlib import Path
from time import perf_counter

from .compiler import compile_kernel


def prepare_nonlinear_library(cache_directory):
    started = perf_counter()
    source = Path(__file__).with_suffix(".cpp").read_text(encoding="utf-8")
    path, compilation = compile_kernel(source, cache_directory)
    library = C.CDLL(str(path))
    library.make_solver.argtypes = [C.c_void_p, C.c_void_p, C.c_int] + [C.c_void_p] * 7
    library.make_solver.restype = C.c_void_p
    library.free_solver.argtypes = [C.c_void_p]
    library.free_solver.restype = C.c_int
    library.get_statistics.argtypes = [C.c_void_p, C.POINTER(C.c_longlong)]
    library.get_statistics.restype = None
    metadata = {
        "nonlinear_solver_implementation": "Newton with early Jacobian refresh",
        "nonlinear_kernel_sha256": compilation["kernel_sha256"],
        "nonlinear_compile_s": compilation["compile_s"],
        "nonlinear_cache_hit": compilation["cache_hit"],
        "nonlinear_generation_s": perf_counter() - started - compilation["compile_s"],
    }
    return library, metadata


def make_nonlinear_solver(runtime, library, refresh_interval):
    """Verify public operation slots against the pinned runtime's Newton solver."""
    pointer = C.c_void_p
    native = runtime.libs["ida"]
    constructor = native.SUNNonlinSol_Newton
    constructor.argtypes = [pointer, pointer]
    constructor.restype = pointer
    release = runtime.libs["core"].SUNNonlinSolFree
    release.argtypes = [pointer]
    release.restype = C.c_int
    prototype = constructor(runtime.y, runtime.ctx)
    if not prototype:
        raise MemoryError("Newton interface verification allocation")
    try:
        ops_address = C.cast(prototype, C.POINTER(pointer))[1]
        operations = C.cast(ops_address, C.POINTER(pointer))
        slots = {
            0: "GetType",
            1: "Initialize",
            3: "Solve",
            4: "Free",
            5: "SetSysFn",
            6: "SetLSetupFn",
            7: "SetLSolveFn",
            8: "SetConvTestFn",
            10: "SetMaxIters",
            11: "GetNumIters",
            12: "GetCurIter",
            13: "GetNumConvFails",
        }
        for slot, name in slots.items():
            function = getattr(native, "SUNNonlinSol" + name + "_Newton")
            if operations[slot] != C.cast(function, pointer).value:
                raise RuntimeError("Unsupported public SUNNonlinearSolver layout.")
    finally:
        runtime.check(release(prototype))
    core = runtime.libs["core"]
    functions = (
        "SUNNonlinSolNewEmpty",
        "SUNNonlinSolFreeEmpty",
        "N_VClone",
        "N_VDestroy",
        "N_VScale",
        "N_VLinearSum",
        "N_VConst",
    )
    result = library.make_solver(
        runtime.ctx,
        runtime.y,
        refresh_interval,
        *(C.cast(getattr(core, name), pointer) for name in functions),
    )
    if not result:
        raise MemoryError("Newton solver allocation")
    return result


def nonlinear_statistics(library, solver):
    values = (C.c_longlong * 3)()
    library.get_statistics(solver, values)
    result = dict(
        zip(
            ("NumNonlinearSolves", "NumNonlinearRestarts", "NumJacobianRefreshes"),
            values,
        )
    )
    return result
