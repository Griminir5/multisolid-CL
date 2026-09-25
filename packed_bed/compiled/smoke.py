"""Mandatory installed-runtime check: compile and solve a DAE with every solver."""

import tempfile
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix

from . import _load_kernel
from .band import prepare_band_library
from .compiler import compile_kernel
from .graph import Graph
from .nonlinear import prepare_nonlinear_library
from .runtime import NativeIDA, callback_source
from .vector_math import select_vector_exponentials


def check():
    with tempfile.TemporaryDirectory(prefix="multisolid compiled check ") as temporary:
        cache = Path(temporary)
        graph = Graph()
        x, z = graph.make("var", 0), graph.make("var", 1)
        roots = [graph.make("add", graph.make("dot", 0), x), graph.make("sub", z, graph.make("mul", x, x))]
        sparsity = csc_matrix([[1., 0.], [1., 1.]])
        jacobian = [graph.gradient(roots[row])[col] for col in range(2)
                    for row in sparsity.indices[sparsity.indptr[col]:sparsity.indptr[col + 1]]]
        source = "#include <cmath>\n#include <cstring>\n"
        for name, values in (("evaluate", roots), ("jacobian", jacobian), ("reconstruct", [x, z])):
            source += graph.emit(name, values, {0: 0, 1: 1})
        nonlinear, _ = prepare_nonlinear_library(cache)
        band, _ = prepare_band_library(cache)
        select_vector_exponentials(cache, True, True)
        for label, solver, threads in (("superlu", "superlu", 1), ("superlu_mt", "superlu", 2),
                                       ("klu", "klu", 1), ("band", "band", 1)):
            code = source + callback_source(sparsity, linear_solver=solver)
            path, _ = compile_kernel(code, cache)
            _, cached = compile_kernel(code, cache)
            # KLU and SuperLU intentionally share identical CSC callbacks.
            assert cached["cache_hit"]
            with NativeIDA(_load_kernel(path), [1., 1.], [-1., -2.], [1e-10, 1e-10], [True, False],
                           sparsity, rtol=1e-8, linear_solver=solver, threads=threads,
                           band_library=band if solver == "band" else None,
                           nonlinear_library=nonlinear, nonlinear_refresh_interval=2) as native:
                times = np.linspace(0, 2, 21)
                values, _ = native.solve(times)
                np.testing.assert_allclose(values[:, 0], np.exp(-times), rtol=0, atol=2e-6)
                np.testing.assert_allclose(values[:, 1], np.exp(-2 * times), rtol=0, atol=2e-6)
            print(f"Compiled {label}: passed", flush=True)


if __name__ == "__main__":
    check()
