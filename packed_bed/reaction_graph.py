"""Shared reaction-network description and the separate Graphviz executable.

No solver, Qt, or Python Graphviz bindings are needed here. Desktop previews may
describe incomplete selections; CLI exports use exactly the same DOT and SVG.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap


RENDER_TIMEOUT_S = 20


class GraphvizError(RuntimeError):
    """Graphviz is unavailable or could not render a network."""


class GraphvizNotFound(GraphvizError):
    pass


@dataclass(frozen=True)
class GraphNode:
    id: str
    key: tuple[str, str]
    label: str
    tooltip: str


@dataclass(frozen=True)
class GraphEdge:
    id: str
    source: str
    target: str
    tooltip: str


@dataclass(frozen=True)
class ReactionGraph:
    dot: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def linked(self, node_id: str) -> set[str]:
        """The selected node, its direct neighbours and their connecting edges."""
        linked = {node_id}
        for edge in self.edges:
            if node_id in (edge.source, edge.target):
                linked.update((edge.id, edge.source, edge.target))
        return linked


def _quote(value: str) -> str:
    # DOT quoted strings, not HTML labels; escape backslashes before newlines.
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('\r', '').replace('\n', '\\n') + '"'


def build_reaction_graph(gases, solids, reactions, property_registry=None) -> ReactionGraph:
    if property_registry is None:
        from .properties import PROPERTY_REGISTRY
        property_registry = PROPERTY_REGISTRY

    gases, solids = list(dict.fromkeys(gases)), list(dict.fromkeys(solids))
    reactions = list({reaction.id: reaction for reaction in reactions}.values())
    selected = set(gases) | set(solids)
    missing = {species for reaction in reactions for species in reaction.all_species} - selected
    for species in sorted(missing):
        record = property_registry.records.get(species)
        (gases if record and record.phase == "gas" else solids).append(species)

    lines = [
        'digraph system_graph {',
        '  graph [bgcolor="transparent", outputorder="edgesfirst", overlap="false"];',
        '  node [fontname="DejaVu Sans", fontsize=12, style="filled", '
        'color="#a4b0bc", fontcolor="#253748"];',
        '  edge [color="#648494", penwidth=1.5, arrowsize=0.7];',
    ]
    nodes, edges, ids = [], [], {}

    def node(key, label, tooltip, shape, color):
        node_id = f"node{len(nodes)}"
        ids[key] = node_id
        nodes.append(GraphNode(node_id, key, label, tooltip))
        wrapped = '\n'.join(textwrap.wrap(label, 22, break_long_words=False, break_on_hyphens=False))
        lines.append(f'  {node_id} [id="{node_id}", label={_quote(wrapped)}, '
                     f'shape="{shape}", fillcolor="{color}"];')

    for species_ids, color in ((gases, "#d9eee8"), (solids, "#f5e6ca")):
        for species in species_ids:
            if ("species", species) in ids:
                continue
            node(("species", species), species,
                 f"{species}\nMissing required species" if species in missing else species,
                 "ellipse", "#ffe0dd" if species in missing else color)
    for reaction in reactions:
        # Coefficients remain in the scientific editor's equations, not the graph.
        reactants = " + ".join(s for s, c in reaction.stoichiometry.items() if c < 0)
        products = " + ".join(s for s, c in reaction.stoichiometry.items() if c > 0)
        tooltip = (reaction.name + '\n' + reactants + (' ⇌ ' if reaction.reversible else ' → ')
                   + products + '\n' + reaction.source_reference)
        node(("reaction", reaction.id), reaction.name, tooltip, "box", "#e3e5f5")
    for reaction in reactions:
        reaction_id = ids[("reaction", reaction.id)]
        for species in reaction.all_species:
            coefficient = reaction.stoichiometry.get(species, 0)
            species_id = ids[("species", species)]
            source, target = ((reaction_id, species_id) if coefficient > 0 else (species_id, reaction_id))
            dependency = coefficient == 0
            edge_id = f"edge{len(edges)}"
            tooltip = f"{species}: catalyst / rate dependency" if dependency else f"{species} — {reaction.name}"
            edges.append(GraphEdge(edge_id, source, target, tooltip))
            lines.append(f'  {source} -> {target} [id="{edge_id}", '
                         f'dir="{"both" if reaction.reversible and not dependency else "forward"}", '
                         f'style="{"dashed" if dependency else "solid"}", '
                         f'color="{"#9272a1" if dependency else "#648494"}"];')
    lines.append("}")
    return ReactionGraph('\n'.join(lines), tuple(nodes), tuple(edges))


@dataclass(frozen=True)
class GraphvizCommand:
    executable: Path
    bundle: Path | None = None

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.bundle is not None:
            # Only the child receives the native-library/plugin paths.
            for name, paths in (
                ("PATH", [self.bundle / "bin"]),
                ("LD_LIBRARY_PATH", [self.bundle / "lib", self.bundle / "lib" / "graphviz"]),
            ):
                env[name] = os.pathsep.join([*(str(p) for p in paths), env.get(name, "")])
            plugins = self.bundle / "lib" / "graphviz"
            # Windows distributions keep plugins and config beside the executables.
            env["GVBINDIR"] = str(plugins if plugins.is_dir() else self.bundle / "bin")
            fonts = self.bundle / "etc" / "fonts" / "fonts.conf"
            if fonts.is_file():
                env["FONTCONFIG_FILE"] = str(fonts)
        elif getattr(sys, "frozen", False):
            # PyInstaller's libraries must not leak into an explicitly chosen
            # external Graphviz installation.
            env.pop("LD_LIBRARY_PATH", None)
            if "LD_LIBRARY_PATH_ORIG" in env:
                env["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH_ORIG"]
        return env


def _bundle_roots() -> list[Path]:
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).parent / "graphviz")
        if hasattr(sys, "_MEIPASS"):
            roots.append(Path(sys._MEIPASS) / "graphviz")
    roots.append(Path(__file__).resolve().parents[1] / "desktop" / "vendor" / "graphviz")
    return roots


def find_graphviz() -> GraphvizCommand:
    """Prefer the adjacent runtime; PATH is a source-development fallback only."""
    name = "neato.exe" if sys.platform == "win32" else "neato"
    for root in _bundle_roots():
        executable = root / "bin" / name
        if executable.is_file():
            return GraphvizCommand(executable, root)
    override = os.environ.get("MULTISOLID_GRAPHVIZ")
    if override:
        executable = Path(override).expanduser().resolve()
        if not executable.is_file():
            raise GraphvizError(f"Graphviz executable does not exist: {executable}")
        return GraphvizCommand(executable)
    if not getattr(sys, "frozen", False):
        executable = shutil.which(name)
        if executable:
            return GraphvizCommand(Path(executable))
    raise GraphvizNotFound("Graphviz neato was not found. Restore the bundled graphviz runtime "
                           "or, for source development, install Graphviz or set MULTISOLID_GRAPHVIZ.")


def render_svg(graph: ReactionGraph, command: GraphvizCommand | None = None) -> bytes:
    command = command or find_graphviz()
    try:
        result = subprocess.run(
            [str(command.executable), "-Tsvg"], input=graph.dot.encode("utf-8"),
            capture_output=True, timeout=RENDER_TIMEOUT_S, env=command.environment(),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except subprocess.TimeoutExpired as exc:
        raise GraphvizError(f"Graphviz timed out after {RENDER_TIMEOUT_S} seconds.") from exc
    except OSError as exc:
        raise GraphvizError(f"Could not start Graphviz: {exc}") from exc
    if result.returncode or b"<svg" not in result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[:2000]
        raise GraphvizError(f"Graphviz could not render the graph: {detail or 'no SVG output'}")
    return result.stdout
