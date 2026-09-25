"""Select the optional AVX2/FMA3 exponential implementation before kernel generation."""

import ctypes as C
from time import perf_counter

from .compiler import compile_kernel

CPU_SOURCE = """
#if defined(_MSC_VER)
#include <intrin.h>
static void cpuid(int leaf, int sub, unsigned* r) {
    __cpuidex(reinterpret_cast<int*>(r), leaf, sub);
}
static unsigned long long xcr0() { return _xgetbv(0); }
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
static void cpuid(int leaf, int sub, unsigned* r) {
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
}
static unsigned long long xcr0() {
    unsigned lo, hi;
    __asm__ volatile ("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<unsigned long long>(hi) << 32) | lo;
}
#endif
PB_EXPORT int has_avx2() {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
    unsigned r[4]; cpuid(0,0,r); if(r[0]<7)return 0;
    cpuid(1,0,r);
    if((r[2] & ((1u<<27)|(1u<<28))) != ((1u<<27)|(1u<<28)))return 0;
    if((xcr0() & 6) != 6)return 0;
    cpuid(7,0,r); return (r[1] & (1u<<5)) != 0;
#else
    return 0;
#endif
}
PB_EXPORT int has_fma3() {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
    if(!has_avx2())return 0;
    unsigned r[4]; cpuid(1,0,r); return (r[2] & (1u<<12)) != 0;
#else
    return 0;
#endif
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
