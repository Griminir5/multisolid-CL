"""Materialize exact expression values used by more than one cell function."""

from collections import Counter, defaultdict
from dataclasses import dataclass

from .graph import Graph, LEAVES


@dataclass
class SharedInputs:
    nodes: list
    graph: Graph
    mapping: dict
    locations: dict
    extended_keep: list


def shared_inputs(graph, roots, root_cells, keep, locations):
    """Cut the frontier of repeated subgraphs; retain each original expression.

    A repeated expression is computed once per callback, not reused between
    solver calls. Its synthetic input occupies a unique slot, including when
    two expressions have the same shape but different coefficients.
    """
    groups = defaultdict(list)
    for cell, root in zip(root_cells, roots, strict=True):
        groups[cell].append(root)
    users = defaultdict(int)

    for group, selected in enumerate(groups.values()):
        bit = 1 << group
        for node in graph.postorder(selected):
            users[node] |= bit

    time_only = {}
    for node in graph.postorder(roots):
        op, *args = graph.nodes[node]
        time_only[node] = (op in ("const", "time") if op in LEAVES else
                           all(time_only[arg] for arg in args))

    repeated = {
        node
        for node, bits in users.items()
        if bits & (bits - 1)
        and graph.nodes[node][0] not in LEAVES
        and not time_only[node]
    }
    frontier = set(roots) & repeated
    for node in users:
        op, *args = graph.nodes[node]
        if node not in repeated and op not in LEAVES:
            frontier.update(arg for arg in args if arg in repeated)
    nodes = sorted(frontier)
    if not nodes:
        return None

    canonical = {}

    shapes = defaultdict(dict)

    def shape(node, anchor):
        known = shapes[anchor]
        for current in graph.postorder([node], known):
            op, *args = graph.nodes[current]
            if op in ("var", "dot"):
                name, component, position = locations[args[0]]
                offset = (
                    position - anchor
                    if position is not None and anchor is not None
                    else position
                )
                key = (op, name, component, offset)
            elif op in LEAVES:
                key = (op,)
            else:
                key = (op, *(known[arg] for arg in args))
            if key not in canonical:
                canonical[key] = len(canonical)
            known[current] = canonical[key]
        return known[node]

    extended = Graph()
    extended.nodes = graph.nodes.copy()
    mapping = {old: new for new, old in enumerate(keep)}
    extended_locations = locations.copy()
    aliases = Counter()
    first = max(locations) + 1
    new_ids = list(range(first, first + len(nodes)))
    for i, (node, new_id) in enumerate(zip(nodes, new_ids, strict=True)):
        positions = [
            locations[old][2]
            for old in graph.dependencies(node)
            if locations[old][2] is not None
        ]
        anchor = min(positions) if positions else None
        kind = shape(node, anchor)
        alias = aliases[kind, anchor]
        aliases[kind, anchor] += 1
        extended.nodes[node] = ("var", new_id)
        extended_locations[new_id] = (f"shared_{kind}", alias, anchor)
        mapping[new_id] = len(keep) + i
    extended.intern = {expression: i for i, expression in enumerate(extended.nodes)}
    return SharedInputs(
        nodes, extended, mapping, extended_locations, list(keep) + new_ids
    )
