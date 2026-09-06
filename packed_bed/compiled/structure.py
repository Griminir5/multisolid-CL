"""Exact fixed-state elimination and equation/variable matching."""

from functools import lru_cache

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching, reverse_cuthill_mckee


def derivative_variables(graph, roots):
    @lru_cache(None)
    def visit(node):
        op, *args = graph.nodes[node]
        if op == "dot":
            return frozenset(args)
        if op in ("const", "var", "time", "cj"):
            return frozenset()
        return frozenset().union(*(visit(arg) for arg in args))

    return [visit(root) for root in roots]


def substitute_constants(graph, roots, replacements):
    """Substitute constant states and their identically zero derivatives."""

    @lru_cache(None)
    def visit(node):
        op, *args = graph.nodes[node]
        if op == "var" and args[0] in replacements:
            return graph.constant(replacements[args[0]])
        if op == "dot" and args[0] in replacements:
            return graph.zero
        if op in ("const", "var", "dot", "time", "cj"):
            return node
        return graph.make(op, *(visit(arg) for arg in args))

    return [visit(root) for root in roots]


def eliminate_fixed_states(graph, keep, residuals, reconstruction, initial, groups):
    """Remove only states whose constant trajectory is proved by substitution.

    First find explicit y'=0 equations. Then check each all-zero species group
    against the entire symbolic forcing, including every future inlet segment.
    A zero initial value or a zero initial reaction rate alone is insufficient.
    """
    dots = derivative_variables(graph, residuals)
    rows = {next(iter(d)): row for row, d in enumerate(dots) if len(d) == 1}
    replacements = {
        old: float(initial[old])
        for old, row in rows.items()
        if graph.nodes[residuals[row]] == ("dot", old)
    }
    for group in groups:
        candidates = [old for old in group if old in rows and old not in replacements]
        if not candidates or any(initial[old] != 0 for old in candidates):
            continue
        proposed = {**replacements, **dict.fromkeys(candidates, 0.0)}
        expressions = substitute_constants(
            graph, [residuals[rows[old]] for old in candidates], proposed
        )
        if all(root == graph.zero for root in expressions):
            replacements = proposed
    removed_rows = {rows[old] for old in replacements}
    remaining = [root for row, root in enumerate(residuals) if row not in removed_rows]
    return (
        [old for old in keep if old not in replacements],
        substitute_constants(graph, remaining, replacements),
        substitute_constants(graph, reconstruction, replacements),
        replacements,
    )


def state_locations(simulation):
    """Identify variable family, component and spatial index from DAE domains."""
    spatial = {
        simulation.model.x_centers.CanonicalName,
        simulation.model.x_faces.CanonicalName,
    }
    locations = {}
    groups = []
    for variable in simulation.model.Variables:
        domains = variable.Domains
        distributed = bool(domains and domains[-1].CanonicalName in spatial)
        count = domains[-1].NumberOfPoints if distributed else 1
        components = {}
        for point in range(variable.NumberOfPoints):
            old = simulation.IndexMappings[variable.OverallIndex + point]
            component, position = divmod(point, count)
            locations[old] = (
                variable.Name,
                component,
                position if distributed else None,
            )
            components.setdefault(component, []).append(old)
        if variable.Name in ("c_gas", "c_sol"):
            groups.extend(components.values())
    return locations, groups


def match_rows(graph, residuals, keep):
    """Return the state column associated with each equation row."""
    columns = {old: new for new, old in enumerate(keep)}
    dots = derivative_variables(graph, residuals)
    if any(len(d) > 1 for d in dots):
        raise ValueError("Cell kernels require at most one derivative per equation.")
    assigned = {row: columns[next(iter(d))] for row, d in enumerate(dots) if d}
    if len(set(assigned.values())) != len(assigned):
        raise ValueError("Differential equations do not have a unique state matching.")
    algebraic_rows = [row for row in range(len(residuals)) if row not in assigned]
    algebraic_cols = [col for col in range(len(keep)) if col not in assigned.values()]
    if algebraic_rows:
        local_cols = {keep[col]: j for j, col in enumerate(algebraic_cols)}
        entries = [
            (i, local_cols[old])
            for i, row in enumerate(algebraic_rows)
            for old in graph.dependencies(residuals[row])
            if old in local_cols
        ]
        row, col = zip(*entries) if entries else ([], [])
        pattern = csc_matrix(
            (np.ones(len(entries)), (row, col)),
            shape=(len(algebraic_rows), len(algebraic_cols)),
        )
        matches = maximum_bipartite_matching(pattern, perm_type="column")
        if np.any(matches < 0):
            raise ValueError("Algebraic equations do not have a complete matching.")
        assigned.update(
            (row, algebraic_cols[col]) for row, col in zip(algebraic_rows, matches)
        )
    return np.array([assigned[row] for row in range(len(residuals))], dtype=int)


def reorder_for_band(graph, residuals, keep, owners):
    rows = np.argsort(owners)
    columns = {old: col for col, old in enumerate(keep)}
    entries = [
        (row, columns[old])
        for row, previous in enumerate(rows)
        for old in graph.dependencies(residuals[previous])
    ]
    rr, cc = zip(*entries)
    pattern = csc_matrix(
        (np.ones(len(entries)), (rr, cc)), shape=(len(keep), len(keep))
    )
    permutation = reverse_cuthill_mckee(pattern + pattern.T, symmetric_mode=True)
    return (
        [residuals[rows[i]] for i in permutation],
        [keep[i] for i in permutation],
    )


def band_layout(sparsity):
    rows, cols = sparsity.nonzero()
    upper = max(0, int(np.max(cols - rows, initial=0)))
    lower = max(0, int(np.max(rows - cols, initial=0)))
    stored_upper = min(sparsity.shape[0] - 1, upper + lower)
    return upper, lower, stored_upper, stored_upper + lower + 1
