from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.parametrize("local_time_coefficient", (False, True))
@pytest.mark.parametrize("vectorize", (False, True))
@pytest.mark.parametrize("batch_uniform", (False, True))
def test_shared_cell_kernels_preserve_distinct_constants_and_time_functions(
    native_tools,
    tmp_path,
    local_time_coefficient,
    vectorize,
    batch_uniform,
):
    from scipy.sparse import csc_matrix

    from packed_bed.compiled import _load_kernel
    from packed_bed.compiled.band import supports_avx2
    from packed_bed.compiled.codegen import emit_model
    from packed_bed.compiled.compiler import compile_kernel, simd_flags
    from packed_bed.compiled.graph import Graph

    if vectorize and not supports_avx2():
        pytest.skip("This computer cannot execute the AVX2 cell kernel")

    graph = Graph()
    count = 10
    keep = list(range(count))
    locations = {i: ("concentration", 0, i) for i in keep}
    clock = graph.make("time")
    roots = []
    for i in keep:
        coefficient = graph.constant(3.25 + (i // 4 if batch_uniform else i) / 10)
        decay = graph.make(
            "exp",
            graph.make(
                "mul",
                clock,
                coefficient if local_time_coefficient else graph.constant(-0.2),
            ),
        )
        loss = graph.make(
            "mul", graph.make("mul", graph.make("var", i), coefficient), decay
        )
        roots.append(graph.make("add", graph.make("dot", i), loss))
    sparsity = csc_matrix(np.eye(count))
    jac = [graph.gradient(root)[i] for i, root in enumerate(roots)]
    reports = [graph.make("var", i) for i in keep] + [clock]
    source, metadata = emit_model(
        graph,
        keep,
        roots,
        jac,
        reports,
        sparsity,
        locations,
        np.arange(count),
        count,
        vectorize=vectorize,
    )
    mapping = dict(enumerate(keep))
    for name, selected in (
        ("evaluate", roots),
        ("jacobian", jac),
        ("reconstruct", reports),
    ):
        source += graph.emit("reference_" + name, selected, mapping)
    library, _ = compile_kernel(
        source, tmp_path, extra_flags=simd_flags(avx2=vectorize)
    )
    kernel = _load_kernel(library)
    assert metadata["evaluate_templates"] == (
        (3 if batch_uniform else count) if local_time_coefficient else 1
    )
    assert metadata["vectorized_residual_cells"] == (
        8 if vectorize and (not local_time_coefficient or batch_uniform) else 0
    )
    if vectorize and batch_uniform:
        assert metadata["broadcast_cell_parameters"] > 0
    for name, width in (
        ("evaluate", count),
        ("jacobian", count),
        ("reconstruct", count + 1),
    ):
        reference = getattr(kernel, "reference_" + name)
        reference.argtypes = getattr(kernel, name).argtypes
        reference.restype = None
        actual, expected = np.empty(width), np.empty(width)
        y = np.linspace(0.2, 1.1, count)
        yp = np.linspace(-0.5, 2.0, count)
        for time in (0.0, 0.5, 0.5, 0.0, 1.7):  # Reuse time, and restart a run at zero.
            args = (time, y.ctypes.data, yp.ctypes.data, 12.3)
            getattr(kernel, name)(*args, actual.ctypes.data)
            reference(*args, expected.ctypes.data)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("vectorize", (False, True))
def test_shared_neighbor_expressions_preserve_aliases_permutations_and_fresh_inputs(
    native_tools, tmp_path, vectorize
):
    from concurrent.futures import ThreadPoolExecutor

    from scipy.sparse import csc_matrix

    from packed_bed.compiled import _load_kernel
    from packed_bed.compiled.band import supports_avx2
    from packed_bed.compiled.codegen import emit_model
    from packed_bed.compiled.compiler import compile_kernel, simd_flags
    from packed_bed.compiled.graph import Graph

    if vectorize and not supports_avx2():
        pytest.skip("This computer cannot execute the AVX2 cell kernel")
    g = Graph()
    count = 9
    x = [g.make("var", i) for i in range(count)]
    clock = g.make("time")
    decay = g.make("exp", g.make("mul", g.constant(-0.2), clock))
    faces = []
    for i in range(count - 1):
        base = g.make(
            "add",
            g.make(
                "div",
                g.make("exp", x[i]),
                g.make("add", g.one, g.make("pow", x[i + 1], g.constant(2))),
            ),
            g.make("abs", g.make("sub", x[i], x[i + 1])),
        )
        # Identical expression shapes at one face, distinct coefficients.
        faces.append(
            tuple(
                g.make("mul", g.make("mul", base, g.constant(c)), decay)
                for c in (3.25 + 0.1 * i, 5.75 + 0.2 * i)
            )
        )
    roots = []
    for i in range(count):
        root = g.make("dot", i)
        if i:
            root = g.make(
                "add",
                root,
                g.make(
                    "sub",
                    faces[i - 1][0],
                    g.make("mul", g.constant(2), faces[i - 1][1]),
                ),
            )
        if i < count - 1:
            root = g.make(
                "sub",
                root,
                g.make("sub", g.make("mul", g.constant(2), faces[i][0]), faces[i][1]),
            )
        roots.append(root)
    keep = [4, 1, 7, 0, 6, 8, 2, 5, 3]
    row_order = [6, 3, 1, 8, 5, 0, 7, 2, 4]
    roots = [roots[i] for i in row_order]
    mapping = {old: new for new, old in enumerate(keep)}
    matrix_nodes = {
        (row, mapping[col]): node
        for row, root in enumerate(roots)
        for col, node in g.gradient(root).items()
    }
    rows, columns = zip(*matrix_nodes, strict=True)
    sparsity = csc_matrix((np.ones(len(rows)), (rows, columns)), shape=(count, count))
    jacobian = [
        matrix_nodes[row, col]
        for col in range(count)
        for row in sparsity.indices[sparsity.indptr[col] : sparsity.indptr[col + 1]]
    ]
    reports = [*x, clock]
    source, metadata = emit_model(
        g,
        keep,
        roots,
        jacobian,
        reports,
        sparsity,
        {i: ("concentration", 0, i) for i in range(count)},
        np.array([mapping[i] for i in row_order]),
        count,
        vectorize=vectorize,
    )
    assert metadata["shared_expressions"]["evaluate"] > 0
    assert metadata["shared_expressions"]["jacobian"] > 0
    assert metadata["shared_kernel_metadata"]["evaluate"]["residual_lanes"] == (
        4 if vectorize else 1
    )
    for name, selected in (
        ("evaluate", roots),
        ("jacobian", jacobian),
        ("reconstruct", reports),
    ):
        source += g.emit("reference_" + name, selected, mapping)
    library, _ = compile_kernel(
        source, tmp_path, extra_flags=simd_flags(avx2=vectorize)
    )
    kernel = _load_kernel(library)
    for name in ("evaluate", "jacobian", "reconstruct"):
        reference = getattr(kernel, "reference_" + name)
        reference.argtypes = getattr(kernel, name).argtypes
        reference.restype = None

    def check(seed):
        rng = np.random.default_rng(seed)
        for iteration in range(12):
            y = rng.uniform(0.2, 1.2, count)
            if iteration % 3 == 0:
                y.fill(0.5)  # Exercise derivative ties in shared expressions.
            yp = rng.uniform(-2, 2, count)
            # Equal times with different states, derivatives and cj must refresh
            # shared values. Alternate zero with another time to test restarts.
            args = (
                0.5 if iteration % 3 else 0.0,
                y.ctypes.data,
                yp.ctypes.data,
                iteration + 1.0,
            )
            for name, width in (
                ("evaluate", count),
                ("jacobian", len(jacobian)),
                ("reconstruct", count + 1),
            ):
                guarded = np.full(width + 8, 1765.25)
                actual, expected = guarded[4:-4], np.empty(width)
                getattr(kernel, name)(*args, actual.ctypes.data)
                getattr(kernel, "reference_" + name)(*args, expected.ctypes.data)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(guarded[:4], 1765.25)
                np.testing.assert_array_equal(guarded[-4:], 1765.25)

    # One DLL can serve independent simulations and calling threads.
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(check, (21, 36)))


def test_cell_packet_matches_scalar_arithmetic_and_special_values(
    native_tools, tmp_path
):
    import ctypes as C

    from packed_bed.compiled import codegen
    from packed_bed.compiled.band import supports_avx2
    from packed_bed.compiled.compiler import compile_kernel, simd_flags

    if not supports_avx2():
        pytest.skip("This computer cannot execute the AVX2 cell kernel")
    expressions = [
        "a+b",
        "a-b",
        "a*b",
        "a/b",
        "-a",
        "fabs(a)",
        "sqrt(a)",
        "exp(a)",
        "log(a)",
        "log10(a)",
        "pow(a,b)",
        "fmin(a,b)",
        "fmax(a,b)",
    ]
    source = Path(codegen.__file__).with_name("cell.hpp").read_text()
    source += (
        'PB_EXPORT void packet(const double* x, const double* y, double* out) {\n'
        "CellPacket a(_mm256_loadu_pd(x)),b(_mm256_loadu_pd(y));\n"
        + "\n".join(
            f"store_cells(out+{4 * i},{expression});"
            for i, expression in enumerate(expressions)
        )
        + "\n}\n"
    )
    source += (
        'PB_EXPORT void scalar(const double* x, const double* y, double* out) {\n'
        "for(int j=0;j<4;++j) {double a=x[j],b=y[j];\n"
        + "\n".join(
            f"out[{4 * i}+j]={expression};" for i, expression in enumerate(expressions)
        )
        + "\n}}\n"
    )
    source += (
        'PB_EXPORT void scatter(const double* x,const int* offsets,int stride,double* out) {\n'
        "scatter_cells(out,offsets,stride,CellPacket(_mm256_loadu_pd(x)));\n}\n"
    )
    library, _ = compile_kernel(source, tmp_path, extra_flags=simd_flags(avx2=True))
    kernel = C.CDLL(str(library))
    for name in ("packet", "scalar"):
        getattr(kernel, name).argtypes = [C.c_void_p] * 3
        getattr(kernel, name).restype = None
    kernel.scatter.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p]
    kernel.scatter.restype = None
    offsets = np.array([[12, 2, 3], [1, 2, 3], [9, 2, 3], [4, 2, 3]], dtype=np.int32)
    lanes = np.array([-0.0, np.inf, np.nan, -3.5])
    actual = np.full(17, 1765.25)
    expected = actual.copy()
    expected[offsets[:, 0]] = lanes
    kernel.scatter(lanes.ctypes.data, offsets.ctypes.data, 3, actual.ctypes.data)
    # Stores must preserve lane bits and leave all other output slots alone.
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    batches = [
        ([0.2, 1.3, -2.0, 3.0], [2.5, -0.5, 0.0, 4.0]),
        ([0.0, -0.0, np.inf, -np.inf], [-0.0, 0.0, np.nan, 1.0]),
        ([np.nan, 1e-200, 1e200, -1e200], [2.0, 0.5, 1e200, 0.5]),
    ]
    for x, y in batches:
        x, y = np.array(x), np.array(y)
        expected, actual = (
            np.empty(4 * len(expressions)),
            np.empty(4 * len(expressions)),
        )
        kernel.scalar(x.ctypes.data, y.ctypes.data, expected.ctypes.data)
        kernel.packet(x.ctypes.data, y.ctypes.data, actual.ctypes.data)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        valid = ~np.isnan(expected)
        # Include zero sign and infinity sign; NaN payloads are not specified.
        np.testing.assert_array_equal(
            actual[valid].view(np.uint64), expected[valid].view(np.uint64)
        )


@pytest.mark.parametrize("requested,avx2", ((False, True), (True, False)))
def test_vector_exp_selection_skips_unsupported_or_disabled_probe(
    monkeypatch, tmp_path, requested, avx2
):
    from packed_bed.compiled import vector_math

    def unexpected(*args, **kwargs):
        pytest.fail("Disabled or non-AVX2 paths must not compile a CPU probe")

    monkeypatch.setattr(vector_math, "compile_kernel", unexpected)
    assert vector_math.select_vector_exponentials(tmp_path, requested, avx2) == (
        False,
        {},
    )


@pytest.mark.parametrize("has_fma", (False, True))
def test_vector_exp_selection_requires_fma3(monkeypatch, tmp_path, has_fma):
    from types import SimpleNamespace

    from packed_bed.compiled import vector_math

    def query():
        return int(has_fma)

    monkeypatch.setattr(
        vector_math.C, "CDLL", lambda path: SimpleNamespace(has_fma3=query)
    )
    monkeypatch.setattr(
        vector_math,
        "compile_kernel",
        lambda *args: (
            tmp_path / "probe.dll",
            {"compile_s": 0.0, "cache_hit": True, "kernel_sha256": "probe"},
        ),
    )
    enabled, metadata = vector_math.select_vector_exponentials(tmp_path, True, True)
    assert enabled is has_fma
    assert metadata["vector_math_fma3_available"] is has_fma


def test_vector_exponential_cell_kernel_accuracy_and_scalar_fallback(
    native_tools, tmp_path
):
    from decimal import Decimal, localcontext

    from scipy.sparse import csc_matrix

    from packed_bed.compiled import _load_kernel
    from packed_bed.compiled.band import supports_avx2
    from packed_bed.compiled.codegen import emit_model
    from packed_bed.compiled.compiler import compile_kernel, simd_flags
    from packed_bed.compiled.graph import Graph
    from packed_bed.compiled.vector_math import select_vector_exponentials

    enabled, _ = select_vector_exponentials(tmp_path, True, supports_avx2())
    if not enabled:
        pytest.skip("The vector exponential requires AVX2 and FMA3")
    graph = Graph()
    roots = [graph.make("exp", graph.make("var", i)) for i in range(6)]
    sparse = csc_matrix(np.eye(6))
    arguments = (
        graph,
        list(range(6)),
        roots,
        roots,
        roots,
        sparse,
        {i: ("temperature", 0, i) for i in range(6)},
        np.arange(6),
        6,
    )
    fallback, fallback_metadata = emit_model(*arguments, vector_exponentials=True)
    assert "Sleef_expd4" not in fallback
    assert fallback_metadata["vector_exponentials"] == "scalar libm"
    source, metadata = emit_model(*arguments, vectorize=True, vector_exponentials=True)
    assert metadata["vectorized_residual_cells"] == 4
    assert metadata["vector_exponentials"] == "SLEEF 3.9.0 AVX2/FMA3 exp_u10"
    library, _ = compile_kernel(source, tmp_path, extra_flags=simd_flags(avx2=True, fma=True))
    kernel = _load_kernel(library)
    yp, actual, reference = np.zeros(6), np.empty(6), np.empty(6)
    rng = np.random.default_rng(1006)
    batches = list(rng.uniform(-745, 709, (40, 6)))
    batches += [
        np.array([-np.inf, np.inf, np.nan, -0.0, 0.0, 1e-300]),
        np.array(
            [-1000.0, 1000.0, -745.1332191019411, 709.782712893384, 1e-16, -1e-16]
        ),
    ]
    with localcontext() as context:
        context.prec = 100
        for y in batches:
            kernel.evaluate(0, y.ctypes.data, yp.ctypes.data, 0, actual.ctypes.data)
            kernel.jacobian(0, y.ctypes.data, yp.ctypes.data, 0, reference.ctypes.data)
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(reference))
            np.testing.assert_array_equal(np.isinf(actual), np.isinf(reference))
            assert not np.signbit(actual[np.isfinite(actual)]).any()
            # The incomplete two-cell batch retains the scalar implementation.
            np.testing.assert_array_equal(
                actual[4:].view(np.uint64), reference[4:].view(np.uint64)
            )
            for x, result in zip(y[:4], actual[:4], strict=True):
                if not np.isfinite(x) or not np.isfinite(result):
                    continue
                exact = Decimal.from_float(float(x)).exp()
                rounded = float(exact)
                spacing = Decimal.from_float(float(np.spacing(rounded)))
                error = abs(Decimal.from_float(float(result)) - exact) / spacing
                assert error <= 1, (x, result, error)
