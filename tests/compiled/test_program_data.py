"""Runtime program semantics, parameter algebra, and shared-library isolation."""

import ctypes as C
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import csc_matrix

from packed_bed.compiled.program_data import RuntimePrograms, program_source, program_wrappers
from packed_bed.programs import CompiledProgram, ProgramSegment, RatioProgram


def long_program(initial, phase=0):
    segments = []
    value = initial
    for i in range(211):
        end = initial + (i % 5 + phase) * .03
        segments.append(ProgramSegment(i * 2., (i + 1) * 2., value, end))
        value = end
    return CompiledProgram(initial, tuple(segments))


def test_native_programs_match_reference_and_isolate_runs(native_tools, tmp_path):
    from packed_bed.compiled.compiler import compile_kernel

    source = program_source() + """
PB_EXPORT void probe(double t,ProgramData* p,double* out) {
    const double* values=program_values(t,p);
    for(int i=0;i<p->outputs;i++)out[i]=values[i];
}
"""
    path, _ = compile_kernel(source, tmp_path)
    library = C.CDLL(str(path))
    library.probe.argtypes = [C.c_double, C.c_void_p, C.c_void_p]
    library.probe.restype = None
    flow = long_program(2.)
    temperature_flow = long_program(600., 2)
    species_flow = CompiledProgram((.2, 1.8), tuple(
        ProgramSegment(s.start_time, s.end_time, (.1*s.start_value, .9*s.start_value),
                       (.1*s.end_value, .9*s.end_value)) for s in flow.segments))
    independent = [(long_program(4.), None), (long_program(350.), None),
                   (CompiledProgram(1., ()), None), (CompiledProgram(0., ()), None)]
    feed = [(flow, None), (RatioProgram(temperature_flow, flow), None),
            (RatioProgram(species_flow, flow), 0), (RatioProgram(species_flow, flow), 1)]
    contexts = [RuntimePrograms(independent, .4), RuntimePrograms(feed, 1.3)]
    assert contexts[0].fingerprint != contexts[1].fingerprint

    def check(index):
        context = contexts[index]
        programs = (independent, feed)[index]
        for time in (0., 2., 2., 1.9, 100., 422., 1000., -10., 0.):
            actual = np.empty(4)
            library.probe(time, context.pointer, actual.ctypes.data)
            expected = []
            for program, component in programs:
                value = program.value_at(time, smooth_ramp_width_s=context.data.width)
                expected.append(value if component is None else value[component])
            np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-12)

    # Alternate different runs at equal times on the SAME thread first.
    for _ in range(2):
        check(0)
        check(1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(check, (0, 1)))


def test_parameters_survive_folding_and_fixed_state_proof():
    from packed_bed.compiled.graph import Graph
    from packed_bed.compiled.structure import eliminate_fixed_states

    g = Graph()
    feed = g.parameter("feed", 0)
    loss = g.parameter("loss", 1)
    y = g.make("var", 0)
    root = g.make("sub", g.make("add", g.make("dot", 0), g.make("mul", loss, y)), feed)
    keep, roots, _, fixed = eliminate_fixed_states(g, [0], [root], [y], np.array([0.]), [[0]])
    assert keep == [0] and not fixed
    assert g.gradient(roots[0])[0] == g.make("add", g.make("cj"), loss)
    before = g.fingerprint(roots)
    g.constant(123.)  # Unreachable instructions do not alter kernel identity.
    assert g.fingerprint(roots) == before


def test_parameterized_cell_kernels_share_library_without_stale_values(native_tools, tmp_path):
    from packed_bed.compiled import _load_kernel
    from packed_bed.compiled.band import supports_avx2
    from packed_bed.compiled.codegen import emit_model
    from packed_bed.compiled.compiler import compile_kernel, simd_flags
    from packed_bed.compiled.graph import Graph

    g = Graph()
    count = 8
    feed, loss = g.parameter("feed", 0), g.parameter("loss", 1)
    y = [g.make("var", i) for i in range(count)]
    # Repeated cross-cell expressions also exercise the helper parameter ABI.
    common = g.make("mul", loss, g.make("add", y[0], y[1]))
    roots = [g.make("sub", g.make("add", g.make("dot", i),
             g.make("add", g.make("mul", loss, value), common)), feed) for i, value in enumerate(y)]
    jacobian = {(r, c): v for r, root in enumerate(roots) for c, v in g.gradient(root).items()}
    rows, cols = zip(*jacobian)
    sparsity = csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(count, count))
    jac = [jacobian[r, c] for c in range(count) for r in sparsity.indices[sparsity.indptr[c]:sparsity.indptr[c+1]]]
    vectorize = supports_avx2()
    source, _ = emit_model(g, list(range(count)), roots, jac, y + [feed, loss], sparsity,
                           {i: ("concentration", 0, i) for i in range(count)}, np.arange(count), count,
                           vectorize=vectorize)
    for name in ("evaluate", "jacobian", "reconstruct"):
        source = source.replace(f"PB_EXPORT void {name}(", f"static void {name}_values(")
    path, _ = compile_kernel(program_source() + source + program_wrappers(), tmp_path,
                             extra_flags=simd_flags(avx2=vectorize))
    library = _load_kernel(path, shared_programs=True)
    contexts = [RuntimePrograms([(CompiledProgram(f, ()), None), (CompiledProgram(k, ()), None)], 1.)
                for f, k in ((0., 0.), (3., .4))]
    values, yp = np.linspace(.1, .8, count), np.ones(count)
    for index in (0, 1, 0, 1):
        context = contexts[index]
        f, k = ((0., 0.), (3., .4))[index]
        actual, j, reported = np.empty(count), np.empty(sparsity.nnz), np.empty(count+2)
        args = (0., values.ctypes.data, yp.ctypes.data, 2.)
        library.evaluate(*args, actual.ctypes.data, context.pointer)
        np.testing.assert_allclose(actual, yp+k*values+k*(values[0]+values[1])-f)
        library.jacobian(*args, j.ctypes.data, context.pointer)
        matrix = csc_matrix((j, sparsity.indices, sparsity.indptr), shape=(count, count)).toarray()
        expected = np.eye(count)*(2.+k)
        expected[:, :2] += k
        np.testing.assert_allclose(matrix, expected)
        library.reconstruct(*args, reported.ctypes.data, context.pointer)
        np.testing.assert_array_equal(reported, np.r_[values, f, k])
