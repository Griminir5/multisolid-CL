"""Shared desktop/CLI graph semantics and separate-runtime integration."""

from dataclasses import replace
from pathlib import Path
import subprocess
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest

from packed_bed import reaction_graph as rg
from packed_bed.kinetics import FAMILY_REGISTRY
from packed_bed.reactions import ReactionDefinition


@pytest.fixture
def reaction():
    return ReactionDefinition(
        id="example", name="Example reaction", phase="gas_solid",
        stoichiometry={"H2": -2, "H2O": 2},
        required_species=("H2", "H2O", "Ni", "N2"),
        catalyst_species=("Ni",), source_reference="Test reference", reversible=True,
    )


def neato_or_skip():
    try:
        return rg.find_graphviz()
    except rg.GraphvizNotFound:
        pytest.skip("Graphviz is not installed in this test environment")


def test_graph_keeps_semantics_without_coefficients(reaction):
    graph = rg.build_reaction_graph(["H2", "N2", "CO"], ["Ni"], [reaction])
    nodes = {node.key: node.id for node in graph.nodes}
    reaction_id = nodes[("reaction", "example")]
    hydrogen = nodes[("species", "H2")]
    water = nodes[("species", "H2O")]
    assert "overlap=\"false\"" in graph.dot
    assert 'fillcolor="#ffe0dd"' in graph.dot
    assert "Missing required species" in next(n.tooltip for n in graph.nodes if n.id == water)
    assert f'{hydrogen} -> {reaction_id}' in graph.dot
    assert f'{reaction_id} -> {water}' in graph.dot
    assert graph.dot.count('dir="both"') == 2
    assert graph.dot.count('style="dashed"') == 2  # catalyst AND rate-only dependency
    assert all("label=" not in line for line in graph.dot.splitlines() if " -> " in line)
    assert '2 H2' not in next(n.tooltip for n in graph.nodes if n.id == reaction_id)
    linked = graph.linked(hydrogen)
    assert reaction_id in linked
    assert water not in linked  # neighbours, not the entire connected component
    assert len(linked) == 3
    assert nodes[("species", "CO")] not in graph.linked(reaction_id)


def test_arbitrary_names_are_quoted_without_identity_collisions(reaction):
    odd = 'reaction:example "quoted" \\name\n<node>'
    reaction = replace(reaction, name='Quoted "reaction" \\N',
                       stoichiometry={odd: -1, "H2": 1}, required_species=(odd, "H2"), catalyst_species=())
    graph = rg.build_reaction_graph([odd, "H2", odd], [], [reaction, reaction])
    assert len(graph.nodes) == 3
    assert len({node.id for node in graph.nodes}) == 3
    svg = rg.render_svg(graph, neato_or_skip())
    root = ET.fromstring(svg)
    assert len(root.findall('.//{*}g[@class="node"]')) == 3


@pytest.mark.parametrize("families", [("nickel_medrano",), tuple(FAMILY_REGISTRY)])
def test_example_and_large_network_have_svg_elements_for_all_nodes_and_edges(families):
    selected = [FAMILY_REGISTRY[key] for key in families]
    graph = rg.build_reaction_graph(
        [s for family in selected for s in family.required_gas_species],
        [s for family in selected for s in family.required_solid_species],
        [r for family in selected for r in family.reactions],
    )
    svg = rg.render_svg(graph, neato_or_skip())
    root = ET.fromstring(svg)
    groups = {group.attrib["id"]: group for group in root.findall('.//{*}g')}
    assert all(node.id in groups for node in graph.nodes)
    for edge in graph.edges:
        assert edge.id in groups
        assert not groups[edge.id].findall('.//{*}text')


def test_bundle_discovery_and_child_environment(monkeypatch, tmp_path):
    root = tmp_path / "runtime with spaces"
    exe = root / "bin" / ("neato.exe" if rg.sys.platform == "win32" else "neato")
    exe.parent.mkdir(parents=True)
    exe.touch()
    (root / "lib" / "graphviz").mkdir(parents=True)
    monkeypatch.setattr(rg, "_bundle_roots", lambda: [root])
    monkeypatch.setenv("MULTISOLID_GRAPHVIZ", "/does/not/exist")
    monkeypatch.setenv("GVBINDIR", "host-plugin-path")
    command = rg.find_graphviz()
    assert command.executable == exe
    assert command.environment()["GVBINDIR"] == str(root / "lib" / "graphviz")
    assert rg.os.environ["GVBINDIR"] == "host-plugin-path"


def test_frozen_discovery_does_not_fall_back_to_host(monkeypatch, tmp_path):
    monkeypatch.setattr(rg.sys, "frozen", True, raising=False)
    monkeypatch.setattr(rg.sys, "executable", str(tmp_path / "app"))
    monkeypatch.delenv("MULTISOLID_GRAPHVIZ", raising=False)
    monkeypatch.setattr(rg, "_bundle_roots", lambda: [])
    monkeypatch.setattr(rg.shutil, "which", lambda _: pytest.fail("Frozen app must use its bundle"))
    with pytest.raises(rg.GraphvizNotFound):
        rg.find_graphviz()


def test_frozen_roots_include_executable_and_internal_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(rg.sys, "frozen", True, raising=False)
    monkeypatch.setattr(rg.sys, "executable", str(tmp_path / "app"))
    monkeypatch.setattr(rg.sys, "_MEIPASS", str(tmp_path / "_internal"), raising=False)
    assert rg._bundle_roots()[:2] == [tmp_path / "graphviz", tmp_path / "_internal" / "graphviz"]


@pytest.mark.parametrize("failure", ["timeout", "failed", "empty", "missing"])
def test_render_failures_are_actionable(monkeypatch, failure):
    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 20)
        if failure == "missing":
            raise FileNotFoundError("neato")
        return SimpleNamespace(returncode=1 if failure == "failed" else 0, stdout=b"", stderr=b"missing layout plugin")
    monkeypatch.setattr(rg.subprocess, "run", run)
    with pytest.raises(rg.GraphvizError, match="Graphviz"):
        rg.render_svg(rg.build_reaction_graph(["N2"], [], []), rg.GraphvizCommand(Path("neato")))


def test_cli_export_uses_shared_svg(tmp_path):
    from packed_bed.artifacts import _render_system_graph
    from packed_bed.config import load_case
    from packed_bed.properties import PROPERTY_REGISTRY
    from packed_bed.reactions import reaction_catalog

    neato_or_skip()
    case = load_case("packed_bed/examples/default_case/run.yaml")
    catalog = reaction_catalog(case.reaction_families)
    graph = rg.build_reaction_graph(case.chemistry.gas_species, case.solids.solid_species,
                                    [catalog[key] for key in case.chemistry.reaction_ids])
    paths = _render_system_graph(case, tmp_path, PROPERTY_REGISTRY)
    assert paths["system_graph_svg"].read_bytes() == rg.render_svg(graph)


def test_artifact_failure_warns_and_removes_stale_graph(monkeypatch, tmp_path):
    from packed_bed import artifacts

    case = SimpleNamespace(output_directory=tmp_path, artifacts_directory=tmp_path)
    (tmp_path / "system_graph.svg").write_text("stale")
    def fail(*_):
        raise rg.GraphvizNotFound("neato not found")
    monkeypatch.setattr(artifacts, "_render_system_graph", fail)
    monkeypatch.setattr(artifacts, "render_operating_program", lambda *_: {"program": tmp_path})
    monkeypatch.setattr(artifacts, "render_initial_solid_profile", lambda *_: {"solids": tmp_path})
    with pytest.warns(RuntimeWarning, match="neato not found"):
        assert artifacts.generate_artifacts(case) == {"program": tmp_path, "solids": tmp_path}
    assert not (tmp_path / "system_graph.svg").exists()
