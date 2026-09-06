"""Partial-pivoted band LU with bounded updates and vectorized substitution."""

import ctypes as C
from pathlib import Path
from time import perf_counter

import numpy as np

from .compiler import compile_kernel
from .structure import band_layout


def supports_avx2():
    query = C.WinDLL("kernel32").IsProcessorFeaturePresent
    query.argtypes = [C.c_uint32]
    query.restype = C.c_int
    return bool(query(40))  # PF_AVX2_INSTRUCTIONS_AVAILABLE, including OS support.


def prepare_band_library(cache_directory, *, vectorize=None, reuse_diagonal=False):
    started = perf_counter()
    available = supports_avx2()
    if vectorize is None:
        vectorize = available
    if vectorize and not available:
        raise ValueError("AVX2 is not available on this computer.")
    source = f"#define USE_AVX2 {int(vectorize)}\n"
    source += f"#define REUSE_DIAGONAL {int(reuse_diagonal)}\n"
    source += Path(__file__).with_suffix(".cpp").read_text(encoding="utf-8")
    path, compilation = compile_kernel(
        source,
        cache_directory,
        extra_flags=("/arch:AVX2",) if vectorize else (),
    )
    library = C.CDLL(str(path))
    library.make_solver.argtypes = [C.c_void_p] * 5 + [C.c_int] * 4 + [C.c_void_p] * 3
    library.make_solver.restype = C.c_void_p
    metadata = {
        "linear_solver_implementation": "active-range band / "
        + ("AVX2" if vectorize else "SSE2")
        + (" / reciprocal diagonal" if reuse_diagonal else ""),
        "linear_diagonal_reciprocals": reuse_diagonal,
        "linear_kernel_sha256": compilation["kernel_sha256"],
        "linear_compile_s": compilation["compile_s"],
        "linear_cache_hit": compilation["cache_hit"],
        "linear_generation_s": perf_counter() - started - compilation["compile_s"],
    }
    return library, metadata


def make_band_solver(runtime, library, prototype, sparsity):
    """Construct an owned SUNLinearSolver through its public operation interface.

    The pinned runtime's exported band routines identify and verify operation
    slots. Its opaque implementation-specific content is never inspected.
    """
    pointer = C.c_void_p
    address = lambda function: C.cast(function, pointer).value
    ops_address = C.cast(prototype, C.POINTER(pointer))[1]
    operations = C.cast(ops_address, C.POINTER(pointer))
    native = runtime.libs["sunlinsolband"]
    names = (
        "GetType",
        "GetID",
        "Initialize",
        "Setup",
        "Solve",
        "LastFlag",
        "Space",
        "Free",
    )
    functions = [getattr(native, "SUNLinSol" + name + "_Band") for name in names]
    # GetType and GetID can share an address after identical-code folding.
    # Their distinct public slots are checked explicitly.
    if operations[0] != address(functions[0]) or operations[1] != address(functions[1]):
        raise RuntimeError("Unsupported public SUNLinearSolver operation layout.")
    existing = list(operations[:16])
    slots = [0, 1]
    for function in functions[2:]:
        if existing.count(address(function)) != 1:
            raise RuntimeError("Cannot identify a required band solver operation.")
        slots.append(existing.index(address(function)))
    slots = np.asarray(slots, dtype=np.int32)
    core = runtime.libs["core"]
    upper, lower, stored, _ = band_layout(sparsity)
    result = library.make_solver(
        runtime.ctx,
        core.SUNLinSolNewEmpty,
        core.SUNLinSolFreeEmpty,
        runtime.libs["sunmatrixband"].SUNBandMatrix_Cols,
        runtime.N_VGetArrayPointer,
        sparsity.shape[0],
        upper,
        lower,
        stored,
        slots.ctypes.data,
        functions[0],
        functions[1],
    )
    if not result:
        raise MemoryError("Optimized band solver allocation")
    return result
