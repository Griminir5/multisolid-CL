// Compiler-specific annotations used by generated kernels and native helpers.
#ifndef PACKED_BED_PORTABLE_HPP
#define PACKED_BED_PORTABLE_HPP
#if defined(_MSC_VER)
#define PB_EXPORT extern "C" __declspec(dllexport)
#define PB_INLINE __forceinline
#define PB_NOINLINE __declspec(noinline)
#define PB_RESTRICT __restrict
#else
#define PB_EXPORT extern "C" __attribute__((visibility("default")))
#define PB_INLINE inline __attribute__((always_inline))
#define PB_NOINLINE __attribute__((noinline))
#define PB_RESTRICT __restrict__
#endif
#endif
