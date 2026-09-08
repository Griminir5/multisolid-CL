// Four independent cells, with precise scalar transcendental functions per lane.
#include <cmath>
#include <immintrin.h>

struct CellPacket {
    __m256d value;
    CellPacket() = default;
    PB_INLINE CellPacket(double x) : value(_mm256_set1_pd(x)) {}
    PB_INLINE CellPacket(__m256d x) : value(x) {}
};
static PB_INLINE CellPacket operator+(CellPacket a, CellPacket b) {
    return _mm256_add_pd(a.value, b.value);
}
static PB_INLINE CellPacket operator-(CellPacket a, CellPacket b) {
    return _mm256_sub_pd(a.value, b.value);
}
static PB_INLINE CellPacket operator*(CellPacket a, CellPacket b) {
    return _mm256_mul_pd(a.value, b.value);
}
static PB_INLINE CellPacket operator/(CellPacket a, CellPacket b) {
    return _mm256_div_pd(a.value, b.value);
}
static PB_INLINE CellPacket operator-(CellPacket a) {
    return _mm256_xor_pd(a.value, _mm256_set1_pd(-0.));
}
static PB_INLINE void store_cells(double *output, CellPacket a) {
    _mm256_storeu_pd(output, a.value);
}
static PB_INLINE void scatter_cells(double *output, const int *offsets, int stride,
                                        CellPacket packet) {
    __m128d lo = _mm256_castpd256_pd128(packet.value);
    __m128d hi = _mm256_extractf128_pd(packet.value, 1);
    _mm_store_sd(output + offsets[0], lo);
    _mm_storeh_pd(output + offsets[stride], lo);
    _mm_store_sd(output + offsets[2 * stride], hi);
    _mm_storeh_pd(output + offsets[3 * stride], hi);
}
static PB_INLINE CellPacket gather_cells(const double *input, const int *indices, int stride) {
    return _mm256_set_pd(input[indices[3 * stride]], input[indices[2 * stride]],
                         input[indices[stride]], input[indices[0]]);
}
static PB_INLINE CellPacket cell_parameter(const double *input, int offset, int stride) {
    return _mm256_set_pd(input[offset + 3 * stride], input[offset + 2 * stride],
                         input[offset + stride], input[offset]);
}
static PB_INLINE CellPacket fabs(CellPacket a) {
    return _mm256_andnot_pd(_mm256_set1_pd(-0.), a.value);
}
static PB_INLINE CellPacket sqrt(CellPacket a) { return _mm256_sqrt_pd(a.value); }
#define CELL_UNARY(name)                                                                           \
    static PB_INLINE CellPacket name(CellPacket a) {                                           \
        double x[4];                                                                               \
        store_cells(x, a);                                                                         \
        for (int i = 0; i < 4; ++i)                                                                \
            x[i] = std::name(x[i]);                                                                \
        return _mm256_loadu_pd(x);                                                                 \
    }
#define CELL_BINARY(name)                                                                          \
    static PB_INLINE CellPacket name(CellPacket a, CellPacket b) {                             \
        double x[4], y[4];                                                                         \
        store_cells(x, a);                                                                         \
        store_cells(y, b);                                                                         \
        for (int i = 0; i < 4; ++i)                                                                \
            x[i] = std::name(x[i], y[i]);                                                          \
        return _mm256_loadu_pd(x);                                                                 \
    }
CELL_UNARY(exp)
CELL_UNARY(log)
CELL_UNARY(log10)
CELL_BINARY(pow)
CELL_BINARY(fmin)
CELL_BINARY(fmax)
#undef CELL_UNARY
#undef CELL_BINARY
