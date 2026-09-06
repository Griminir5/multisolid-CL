// Public SUNDIALS 7.5 operation interface; all solver workspaces are per instance.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#ifndef REUSE_DIAGONAL
#define REUSE_DIAGONAL 0
#endif
#if USE_AVX2
#include <immintrin.h>
#endif

using Ptr = void *;
struct Linear {
    Ptr content;
    Ptr *ops;
    Ptr context;
};
using NewEmpty = Linear *(*)(Ptr);
using FreeEmpty = void (*)(Linear *);
using MatrixCols = double **(*)(Ptr);
using VectorData = double *(*)(Ptr);

struct Content {
    int n, upper, lower, stored_upper, flag = 0;
    MatrixCols columns;
    VectorData vector;
    FreeEmpty release;
    std::vector<int> pivots, ranges;
#if REUSE_DIAGONAL
    std::vector<double> inverse;
    bool safe_inverse = true;
#endif
    Content(int n, int u, int l, int s, MatrixCols c, VectorData v, FreeEmpty r)
        : n(n), upper(u), lower(l), stored_upper(s), columns(c), vector(v), release(r), pivots(n),
          ranges(4 * n)
#if REUSE_DIAGONAL
          ,
          inverse(n)
#endif
    {
    }
};

static inline void axpy(int n, double alpha, const double *__restrict x, double *__restrict y) {
    int i = 0;
#if USE_AVX2
    __m256d a = _mm256_set1_pd(alpha);
    for (; i + 4 <= n; i += 4) {
        _mm256_storeu_pd(
            y + i, _mm256_add_pd(_mm256_loadu_pd(y + i), _mm256_mul_pd(a, _mm256_loadu_pd(x + i))));
    }
    if (i + 2 <= n) {
        _mm_storeu_pd(y + i, _mm_add_pd(_mm_loadu_pd(y + i),
                                        _mm_mul_pd(_mm_set1_pd(alpha), _mm_loadu_pd(x + i))));
        i += 2;
    }
#endif
    for (; i < n; i++)
        y[i] += alpha * x[i];
}

static int initialize(Linear *solver) {
    static_cast<Content *>(solver->content)->flag = 0;
    return 0;
}

static int active_factor(double **a, int n, int upper, int lower, int stored, int *pivots) {
    // Clear the extra upper diagonals reserved for fill from row pivoting.
    for (int j = 0; j < n; j++)
        std::fill(a[j], a[j] + stored - upper, 0.);
    int last_column = 0;
    for (int k = 0; k < n - 1; k++) {
        double *col = a[k];
        int end = std::min(lower, n - 1 - k), pivot = 0;
        double largest = std::abs(col[stored]);
        for (int i = 1; i <= end; i++) {
            double value = std::abs(col[stored + i]);
            if (value > largest) {
                largest = value;
                pivot = i;
            }
        }
        pivots[k] = k + pivot;
        if (col[stored + pivot] == 0.)
            return k + 1;
        if (pivot)
            std::swap(col[stored + pivot], col[stored]);
        double inverse = -1. / col[stored];
        for (int i = 1; i <= end; i++)
            col[stored + i] *= inverse;
        int first = 1, last = end;
        while (first <= last && col[stored + first] == 0.)
            first++;
        while (last >= first && col[stored + last] == 0.)
            last--;
        // A pivot can bring entries through pivot_row + upper into this row.
        // Retain earlier fill bounds as elimination advances down the matrix.
        last_column = std::max(last_column, std::min(n - 1, k + pivot + upper));
        for (int j = k + 1; j <= last_column; j++) {
            double *other = a[j];
            double value = other[stored + k + pivot - j];
            if (pivot)
                std::swap(other[stored + k + pivot - j], other[stored + k - j]);
            if (value != 0.)
                axpy(last - first + 1, value, col + stored + first, other + stored + k + first - j);
        }
    }
    pivots[n - 1] = n - 1;
    return a[n - 1][stored] == 0. ? n : 0;
}

static int setup(Linear *solver, Ptr matrix) {
    auto &c = *static_cast<Content *>(solver->content);
    double **a = c.columns(matrix);
    c.flag = active_factor(a, c.n, c.upper, c.lower, c.stored_upper, c.pivots.data());
    if (c.flag)
        return c.flag > 0 ? 808 : c.flag; // SUNLS_LUFACT_FAIL
#if REUSE_DIAGONAL
    c.safe_inverse = true;
#endif
    for (int k = 0; k < c.n; k++) {
#if REUSE_DIAGONAL
        c.inverse[k] = 1. / a[k][c.stored_upper];
        // Subnormal or overflowing reciprocals can lose essential information.
        // Keep ordinary division for the whole solve in that case.
        c.safe_inverse = c.safe_inverse && std::isnormal(c.inverse[k]);
#endif
        int first_lower = 1, last_lower = std::min(c.lower, c.n - 1 - k);
        while (first_lower <= last_lower && a[k][c.stored_upper + first_lower] == 0.)
            first_lower++;
        while (last_lower >= first_lower && a[k][c.stored_upper + last_lower] == 0.)
            last_lower--;
        int first_upper = std::max(0, k - c.stored_upper), last_upper = k - 1;
        while (first_upper <= last_upper && a[k][c.stored_upper + first_upper - k] == 0.)
            first_upper++;
        while (last_upper >= first_upper && a[k][c.stored_upper + last_upper - k] == 0.)
            last_upper--;
        c.ranges[4 * k] = first_lower;
        c.ranges[4 * k + 1] = last_lower;
        c.ranges[4 * k + 2] = first_upper;
        c.ranges[4 * k + 3] = last_upper;
    }
    return 0;
}

static int solve(Linear *solver, Ptr matrix, Ptr x, Ptr rhs, double tolerance) {
    auto &c = *static_cast<Content *>(solver->content);
    double **a = c.columns(matrix);
    double *b = c.vector(x);
    const double *original = c.vector(rhs);
    if (b != original)
        memcpy(b, original, c.n * sizeof(double));
    // Preserve the native pivot sequence and update order.
    for (int k = 0; k < c.n - 1; k++) {
        int pivot = c.pivots[k];
        double value = b[pivot];
        if (pivot != k)
            std::swap(b[pivot], b[k]);
        int first = c.ranges[4 * k], last = c.ranges[4 * k + 1];
        axpy(last - first + 1, value, a[k] + c.stored_upper + first, b + k + first);
    }
#if REUSE_DIAGONAL
    if (c.safe_inverse) {
        for (int k = c.n - 1; k >= 0; k--) {
            b[k] *= c.inverse[k];
            int first = c.ranges[4 * k + 2], last = c.ranges[4 * k + 3];
            axpy(last - first + 1, -b[k], a[k] + c.stored_upper + first - k, b + first);
        }
    } else
#endif
        for (int k = c.n - 1; k >= 0; k--) {
            b[k] /= a[k][c.stored_upper];
            int first = c.ranges[4 * k + 2], last = c.ranges[4 * k + 3];
            axpy(last - first + 1, -b[k], a[k] + c.stored_upper + first - k, b + first);
        }
    c.flag = 0;
    return 0;
}

static int last_flag(Linear *solver) { return static_cast<Content *>(solver->content)->flag; }
static int space(Linear *solver, long *real_size, long *integer_size) {
    *real_size = REUSE_DIAGONAL ? static_cast<Content *>(solver->content)->n : 0;
    *integer_size = static_cast<Content *>(solver->content)->n * 5;
    return 0;
}
static int destroy(Linear *solver) {
    auto *content = static_cast<Content *>(solver->content);
    auto release = content->release;
    delete content;
    solver->content = nullptr;
    release(solver);
    return 0;
}

extern "C" __declspec(dllexport) Linear *make_solver(Ptr context, NewEmpty create,
                                                     FreeEmpty release, MatrixCols columns,
                                                     VectorData vector, int n, int upper, int lower,
                                                     int stored_upper, const int *slots,
                                                     Ptr get_type, Ptr get_id) {
    Linear *solver = create(context);
    if (!solver)
        return nullptr;
    try {
        solver->content = new Content(n, upper, lower, stored_upper, columns, vector, release);
    } catch (...) {
        release(solver);
        return nullptr;
    }
    Ptr operations[] = {get_type,
                        get_id,
                        reinterpret_cast<Ptr>(initialize),
                        reinterpret_cast<Ptr>(setup),
                        reinterpret_cast<Ptr>(solve),
                        reinterpret_cast<Ptr>(last_flag),
                        reinterpret_cast<Ptr>(space),
                        reinterpret_cast<Ptr>(destroy)};
    for (int k = 0; k < 8; k++)
        solver->ops[slots[k]] = operations[k];
    return solver;
}
