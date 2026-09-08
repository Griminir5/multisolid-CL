"""Optional compiled CPU backend; DAETools remains the model and IC provider."""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csc_matrix

from .band import supports_avx2
from .codegen import emit_model
from .compiler import compile_kernel, library_suffix, platform_identity, simd_flags
from .graph import export_model
from .runtime import NativeIDA, callback_source, check_runtime
from .structure import (
    band_layout,
    eliminate_fixed_states,
    match_rows,
    reorder_for_band,
    state_locations,
)
from .vector_math import select_vector_exponentials


@dataclass
class CompiledModel:
    library: object
    keep: np.ndarray
    sparsity: csc_matrix
    report_variables: list
    report_width: int
    metadata: dict


def prepare_model(simulation, cache_directory: Path) -> CompiledModel:
    """Generate an exact sparse Jacobian and kernels for the initialized model."""
    from .compiler import compiler_identity

    check_runtime()
    started = perf_counter()
    cache_directory = cache_directory.resolve()
    cache_directory.mkdir(parents=True, exist_ok=True)
    # Hash DAETools' actual instructions, including numeric constants. This also
    # invalidates the cache for custom kinetics/property hooks and changed inputs.
    digest = hashlib.sha256()
    # A model-cache hit skips compile_kernel, so it must check the installed
    # compiler too. Otherwise a compiler update could keep loading an old binary.
    toolchain, compiler_version = compiler_identity()
    digest.update(platform_identity().encode())
    digest.update(str(toolchain).encode())
    digest.update(compiler_version.encode())
    sources = list(Path(__file__).parent.glob("*.py")) + list(
        Path(__file__).parent.glob("*.hpp")
    )
    for path in sorted(sources):
        digest.update(path.read_bytes())
    vectorize = supports_avx2()
    digest.update(b"AVX2 cells" if vectorize else b"scalar cells")
    requested = simulation.case.run.solver.vector_exponentials
    vector_exponentials, cpu_metadata = select_vector_exponentials(
        cache_directory, requested, vectorize
    )
    cpu_compile_s = cpu_metadata.get("vector_math_cpu_compile_s", 0.0)
    digest.update(b"SLEEF exp" if vector_exponentials else b"scalar exp")
    linear_solver = "band" if simulation.case.run.solver.name == "band" else "superlu"
    digest.update(linear_solver.encode())
    # Fixed-state elimination embeds constants from differential initial data.
    # Algebraic IC roundoff and integration tolerances do not define the kernel.
    from daetools.pyDAE import cnDifferential

    differential = np.asarray(simulation.VariableTypes) == cnDifferential
    digest.update(np.asarray(simulation.Values, dtype=float)[differential].tobytes())
    with tempfile.TemporaryDirectory(
        prefix="fingerprint-", dir=cache_directory
    ) as temporary:
        stacks = Path(temporary) / "equations.bin"
        indexes = Path(temporary) / "jacobian.bin"
        simulation.ExportComputeStackStructs(str(stacks), str(indexes))
        digest.update(stacks.read_bytes())
        digest.update(indexes.read_bytes())
    descriptions = [
        (v.Name, v.NumberOfPoints, v.ReportingOn, v.OverallIndex)
        for v in simulation.model.Variables
    ]
    digest.update(
        json.dumps(
            [
                descriptions,
                simulation.IndexMappings,
                [eq.Name for eq in simulation.model.Equations],
            ],
            sort_keys=True,
        ).encode()
    )
    cache_index = cache_directory / f"model-{digest.hexdigest()}.json"
    report_variables = []
    report_indices = []
    for variable in simulation.model.Variables:
        if variable.ReportingOn:
            begin = len(report_indices)
            report_indices.extend(
                simulation.IndexMappings[variable.OverallIndex + i]
                for i in range(variable.NumberOfPoints)
            )
            report_variables.append((variable, slice(begin, len(report_indices))))
    if cache_index.is_file():
        cached = json.loads(cache_index.read_text())
        filename = cached["library"]
        suffix = library_suffix()
        # The JSON contains data and a local content-addressed filename only.
        if (
            not isinstance(filename, str)
            or len(filename) != 64 + len(suffix)
            or not filename.endswith(suffix)
            or any(c not in "0123456789abcdef" for c in filename[:64])
        ):
            raise ValueError("Invalid compiled kernel cache filename.")
        library_path = cache_directory / filename
        if library_path.is_file():
            keep = np.asarray(cached["keep"], dtype=int)
            sparsity = csc_matrix(
                (np.ones(len(cached["indices"])), cached["indices"], cached["indptr"]),
                shape=(len(keep), len(keep)),
            )
            metadata = {
                **cached["metadata"],
                **cpu_metadata,
                "cache_hit": cpu_metadata.get("vector_math_cpu_cache_hit", True),
                "compile_s": cpu_compile_s,
                "generation_s": perf_counter() - started - cpu_compile_s,
                "vector_exponentials_requested": requested,
            }
            return CompiledModel(
                _load_kernel(library_path),
                keep,
                sparsity,
                report_variables,
                len(report_indices),
                metadata,
            )
    graph, keep, residuals, reconstruction, _ = export_model(simulation)
    locations, groups = state_locations(simulation)
    keep, residuals, reconstruction, fixed = eliminate_fixed_states(
        graph,
        keep,
        residuals,
        reconstruction,
        np.asarray(simulation.Values),
        groups,
    )
    if len(residuals) != len(keep):
        raise RuntimeError("Algebraic reduction did not produce a square system.")
    owners = match_rows(graph, residuals, keep)
    if linear_solver == "band":
        residuals, keep = reorder_for_band(graph, residuals, keep, owners)
        owners = np.arange(len(keep))
    variable_map = {old: new for new, old in enumerate(keep)}
    jacobian = {
        (row, variable_map[column]): value
        for row, residual in enumerate(residuals)
        for column, value in graph.gradient(residual).items()
    }
    rows, columns = zip(*jacobian)
    sparsity = csc_matrix(
        (np.ones(len(jacobian)), (rows, columns)),
        shape=(len(keep), len(keep)),
    )
    jacobian_values = [
        jacobian[row, column]
        for column in range(len(keep))
        for row in sparsity.indices[
            sparsity.indptr[column] : sparsity.indptr[column + 1]
        ]
    ]
    report_roots = [reconstruction[i] for i in report_indices]
    source, generation_metadata = emit_model(
        graph,
        keep,
        residuals,
        jacobian_values,
        report_roots,
        sparsity,
        locations,
        owners,
        simulation.case.run.model.axial_cells,
        vectorize=vectorize,
        vector_exponentials=vector_exponentials,
    )
    source += callback_source(sparsity, linear_solver=linear_solver)
    generation_s = perf_counter() - started - cpu_compile_s
    library_path, compilation = compile_kernel(
        source,
        cache_directory,
        extra_flags=simd_flags(
            avx2=generation_metadata["residual_lanes"] == 4,
            fma=vector_exponentials,
        ),
    )
    metadata = {
        **compilation,
        **generation_metadata,
        **cpu_metadata,
        "cache_hit": compilation["cache_hit"]
        and cpu_metadata.get("vector_math_cpu_cache_hit", True),
        "compile_s": compilation["compile_s"] + cpu_compile_s,
        "vector_exponentials_requested": requested,
        "generation_s": generation_s,
        "original_unknowns": simulation.NumberOfEquations,
        "compiler_version": compiler_version,
        "reduced_unknowns": len(keep),
        "jacobian_nonzeros": sparsity.nnz,
        "fixed_states_removed": len(fixed),
        "linear_solver": linear_solver,
        "runtime": "SUNDIALS 7.5.0 IDA / "
        + ("SUNLinSol_Band" if linear_solver == "band" else "SuperLU_MT"),
    }
    if linear_solver == "band":
        upper, lower, _, _ = band_layout(sparsity)
        metadata.update(upper_bandwidth=upper, lower_bandwidth=lower)
    record = {
        "library": library_path.name,
        "keep": keep,
        # CPU probe timings and capabilities belong to this process, not the
        # reusable model. Recompute them even when loading the same library.
        "metadata": {
            key: value
            for key, value in metadata.items()
            if not key.startswith("vector_math_")
        },
        "indices": sparsity.indices.tolist(),
        "indptr": sparsity.indptr.tolist(),
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=cache_directory, encoding="utf-8", delete=False
    ) as temporary:
        json.dump(record, temporary)
    Path(temporary.name).replace(cache_index)
    return CompiledModel(
        _load_kernel(library_path),
        np.asarray(keep),
        sparsity,
        report_variables,
        len(report_roots),
        metadata,
    )


def _load_kernel(library_path):
    library = C.CDLL(str(library_path))
    for name in ("evaluate", "jacobian", "reconstruct"):
        function = getattr(library, name)
        function.argtypes = [C.c_double, C.c_void_p, C.c_void_p, C.c_double, C.c_void_p]
        function.restype = None
    return library


def _process(simulation, compiled, times, values):
    """Copy metadata to Python so the retained results outlive the DAE model."""
    domains = {}
    variables = []
    for variable, selection in compiled.report_variables:
        variable_domains = []
        for domain in variable.Domains:
            if domain.CanonicalName not in domains:
                domains[domain.CanonicalName] = SimpleNamespace(
                    Name=domain.CanonicalName,
                    Type=domain.Type,
                    Units=str(domain.Units),
                    NumberOfPoints=domain.NumberOfPoints,
                    Points=np.asarray(domain.Points).copy(),
                    Coordinates=[],
                )
            variable_domains.append(domains[domain.CanonicalName])
        shape = tuple(domain.NumberOfPoints for domain in variable.Domains)
        variables.append(
            SimpleNamespace(
                Name=variable.CanonicalName,
                Units=str(variable.VariableType.Units),
                NumberOfPoints=variable.NumberOfPoints,
                Domains=variable_domains,
                TimeValues=times,
                Values=values[:, selection].reshape((len(times), *shape)),
            )
        )
    return SimpleNamespace(
        Name=simulation.case.run.simulation.system_name,
        Domains=list(domains.values()),
        Variables=variables,
        dictDomains=domains,
        dictVariables={v.Name: v for v in variables},
    )


def integrate(simulation):
    """Integrate an initialized model and return a reporter-compatible process."""
    from daetools.pyDAE import cnDifferential

    from ..reports import _scheduled_time

    case = simulation.case
    settings = case.run.solver
    cache = Path(
        os.environ.get(
            "PACKED_BED_COMPILED_CACHE", case.run_path.parent / ".packed_bed_cache"
        )
    )
    compiled = prepare_model(simulation, cache)
    band_library = None
    if settings.name == "band":
        from .band import prepare_band_library

        band_library, linear_metadata = prepare_band_library(
            cache, reuse_diagonal=case.run.solver.band_reciprocals
        )
        compiled.metadata.update(linear_metadata)
        compiled.metadata["compile_s"] += linear_metadata["linear_compile_s"]
        compiled.metadata["generation_s"] += linear_metadata["linear_generation_s"]
        compiled.metadata["cache_hit"] &= linear_metadata["linear_cache_hit"]
    keep = compiled.keep
    nonlinear_library = None
    if settings.nonlinear_refresh_interval:
        from .nonlinear import prepare_nonlinear_library

        nonlinear_library, nonlinear_metadata = prepare_nonlinear_library(cache)
        compiled.metadata.update(nonlinear_metadata)
        compiled.metadata["compile_s"] += nonlinear_metadata["nonlinear_compile_s"]
        compiled.metadata["generation_s"] += nonlinear_metadata[
            "nonlinear_generation_s"
        ]
        compiled.metadata["cache_hit"] &= nonlinear_metadata["nonlinear_cache_hit"]
    y0 = np.asarray(simulation.Values)[keep].copy()
    yp0 = np.asarray(simulation.TimeDerivatives)[keep].copy()
    absolute_tolerances = np.asarray(simulation.AbsoluteTolerances)[keep].copy()
    differential = np.asarray(simulation.VariableTypes)[keep] == cnDifferential
    times = _scheduled_time(case)
    row_scale = np.ones(len(keep))
    compiled.metadata["row_scaling"] = "none"
    compiled.metadata["step_growth_threshold"] = settings.step_growth_threshold
    compiled.metadata["nonlinear_refresh_interval"] = (
        settings.nonlinear_refresh_interval
    )
    if settings.scale_residuals:
        # Apply the same fixed positive factors to F and its Jacobian. This
        # normalizes initial matrix rows without changing the state error weights.
        initial_jacobian = np.empty(compiled.sparsity.nnz)
        compiled.library.jacobian(
            0, y0.ctypes.data, yp0.ctypes.data, 1, initial_jacobian.ctypes.data
        )
        if not np.isfinite(initial_jacobian).all():
            raise RuntimeError(
                "The initial compiled Jacobian contains nonfinite values."
            )
        row_norm = np.zeros(len(keep))
        np.maximum.at(row_norm, compiled.sparsity.indices, abs(initial_jacobian))
        row_scale = 1.0 / np.maximum(row_norm, 1e-300)
        compiled.metadata["row_scaling"] = "inverse initial Jacobian row maximum (cj=1)"
    with NativeIDA(
        compiled.library,
        y0,
        yp0,
        absolute_tolerances,
        differential,
        compiled.sparsity,
        rtol=settings.relative_tolerance,
        max_order=settings.maximum_order,
        nonlinear_coef=settings.nonlinear_convergence_coefficient,
        max_nonlin_iters=settings.max_nonlinear_iterations,
        suppress_algebraic_errors=settings.suppress_algebraic_errors,
        threads=settings.threads,
        row_scale=row_scale,
        linear_solver=compiled.metadata["linear_solver"],
        band_library=band_library,
        step_growth_threshold=settings.step_growth_threshold,
        nonlinear_refresh_interval=settings.nonlinear_refresh_interval,
        nonlinear_library=nonlinear_library,
    ) as solver:
        started = perf_counter()
        values, derivatives = solver.solve(times)
        compiled.metadata.update(
            integration_s=perf_counter() - started, integrator=solver.stats()
        )
    started = perf_counter()
    reported = np.empty((len(times), compiled.report_width))
    for i, time in enumerate(times):
        compiled.library.reconstruct(
            time,
            values[i].ctypes.data,
            derivatives[i].ctypes.data,
            0,
            reported[i].ctypes.data,
        )
    if not np.isfinite(reported).all():
        raise RuntimeError("Compiled result reconstruction produced nonfinite values.")
    process = _process(simulation, compiled, times, reported)
    compiled.metadata["reconstruction_s"] = perf_counter() - started
    return process, compiled.metadata
