"""Select the optional AVX2/FMA3 exponential implementation before kernel generation."""

import ctypes as C
from time import perf_counter

from .compiler import compile_kernel

CPU_SOURCE = """
#include <intrin.h>
extern "C" __declspec(dllexport) int has_fma3() {
    int registers[4]; __cpuid(registers, 1);
    return (registers[2] & (1 << 12)) != 0;
}
"""


def select_vector_exponentials(cache_directory, requested, avx2_available):
    """OS AVX2 support is checked separately; this generic probe uses no AVX/FMA."""
    if not requested or not avx2_available:
        return False, {}
    started = perf_counter()
    path, compilation = compile_kernel(CPU_SOURCE, cache_directory)
    library = C.CDLL(str(path))
    query = library.has_fma3
    query.argtypes = []
    query.restype = C.c_int
    enabled = bool(query())
    metadata = {
        "vector_math_cpu_compile_s": compilation["compile_s"],
        "vector_math_cpu_cache_hit": compilation["cache_hit"],
        "vector_math_cpu_generation_s": perf_counter()
        - started
        - compilation["compile_s"],
        "vector_math_cpu_kernel_sha256": compilation["kernel_sha256"],
        "vector_math_fma3_available": enabled,
    }
    return enabled, metadata
