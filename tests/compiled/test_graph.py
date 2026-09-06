from __future__ import annotations
import numpy as np
import pytest


def test_generated_residual_and_jacobian_against_independent_formula(
    native_tools, tmp_path
):
    from packed_bed.compiled import _load_kernel
    from packed_bed.compiled.compiler import compile_kernel
    from packed_bed.compiled.graph import Graph

    graph = Graph()
    x = graph.make("var", 0)
    z = graph.make("var", 1)
    clock = graph.make("time")
    nonlinear = graph.make(
        "div",
        graph.make("exp", x),
        graph.make("add", graph.one, graph.make("pow", z, graph.constant(3))),
    )
    residual = graph.make(
        "add", graph.make("dot", 0), graph.make("mul", clock, nonlinear)
    )
    limited = graph.make("sub", graph.make("max", x, graph.zero), graph.make("abs", z))
    roots = [residual, limited]
    jac = [graph.gradient(root).get(j, graph.zero) for root in roots for j in range(2)]
    source = "#include <cmath>\n" + graph.emit("evaluate", roots, {0: 0, 1: 1})
    source += graph.emit("jacobian", jac, {0: 0, 1: 1})
    source += graph.emit("reconstruct", [x, z], {0: 0, 1: 1})
    library, _ = compile_kernel(source, tmp_path)
    kernel = _load_kernel(library)
    y = np.array([0.4, 1.2])
    yp = np.array([2.0, 0.0])
    t = 2.3
    cj = 7.0
    output = np.empty(2)
    actual_jac = np.empty(4)
    kernel.evaluate(t, y.ctypes.data, yp.ctypes.data, cj, output.ctypes.data)
    expected = np.array(
        [yp[0] + t * np.exp(y[0]) / (1 + y[1] ** 3), max(y[0], 0) - abs(y[1])]
    )
    np.testing.assert_allclose(output, expected, rtol=1e-14)
    kernel.jacobian(t, y.ctypes.data, yp.ctypes.data, cj, actual_jac.ctypes.data)
    finite_difference = np.empty((2, 2))
    for j in range(2):
        step = 1e-6
        plus = y.copy()
        minus = y.copy()
        dplus = yp.copy()
        dminus = yp.copy()
        plus[j] += step
        minus[j] -= step
        dplus[j] += cj * step
        dminus[j] -= cj * step

        def formula(v, dv):
            return np.array(
                [dv[0] + t * np.exp(v[0]) / (1 + v[1] ** 3), max(v[0], 0) - abs(v[1])]
            )

        finite_difference[:, j] = (formula(plus, dplus) - formula(minus, dminus)) / (
            2 * step
        )
    np.testing.assert_allclose(
        actual_jac.reshape(2, 2), finite_difference, rtol=1e-8, atol=1e-9
    )


def test_reduction_rejects_nonlinear_and_cyclic_definitions():
    from packed_bed.compiled.graph import Graph

    graph = Graph()
    x = graph.make("var", 0)
    y = graph.make("var", 1)
    with pytest.raises(ValueError, match="Nonlinear"):
        graph.isolate(graph.make("mul", x, x), 0)
    with pytest.raises(ValueError, match="Cyclic"):
        graph.substitute([x], {0: y, 1: x})


def test_fixed_state_proof_keeps_future_feeds_and_reaction_products():
    from packed_bed.compiled.graph import Graph
    from packed_bed.compiled.structure import eliminate_fixed_states

    graph = Graph()
    values = [graph.make("var", i) for i in range(6)]
    dots = [graph.make("dot", i) for i in range(6)]
    later_feed = graph.make(
        "max", graph.zero, graph.make("sub", graph.make("time"), graph.constant(10))
    )
    roots = [
        dots[0],  # A nonzero inert constant.
        graph.make("add", dots[1], graph.make("mul", values[0], values[1])),
        graph.make("add", dots[2], graph.make("sub", values[2], values[1])),
        graph.make("sub", dots[3], later_feed),  # Zero now, supplied later.
        graph.make("sub", dots[4], values[0]),  # Zero now, reaction product.
        graph.make("add", dots[5], values[5]),  # Nonzero decaying species.
    ]
    keep, remaining, reports, fixed = eliminate_fixed_states(
        graph,
        list(range(6)),
        roots,
        values + dots,
        np.array([3.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        [[1, 2], [3], [4], [5]],
    )
    assert fixed == {0: 3.0, 1: 0.0, 2: 0.0}
    assert keep == [3, 4, 5]
    assert len(remaining) == 3
    assert graph.nodes[reports[0]] == ("const", 3.0)
    assert reports[1:3] == [graph.zero, graph.zero]
    assert reports[6:9] == [graph.zero] * 3
    assert not set(fixed).intersection(
        set().union(*(graph.dependencies(r) for r in remaining))
    )
