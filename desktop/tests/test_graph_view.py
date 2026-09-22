"""Graph navigation and asynchronous render lifecycle, without a solver."""

import sys
import time

import pytest

from packed_bed.kinetics import FAMILY_REGISTRY
from packed_bed.reaction_graph import GraphvizCommand, GraphvizError, find_graphviz, render_svg


def wait_until(condition, timeout=5):
    from PyQt6.QtTest import QTest
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        QTest.qWait(10)
    assert condition(), "Graph condition did not complete"


@pytest.fixture
def view(qt_app):
    from PyQt6.QtCore import QCoreApplication, QEvent
    from packed_bed_ui.reaction_graph import NetworkView
    view = NetworkView()
    view.resize(650, 500)
    view.show()
    yield view
    view.close()
    view.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt_app.processEvents()


@pytest.fixture
def rendered(view):
    try:
        find_graphviz()
    except GraphvizError:
        pytest.skip("Graphviz is not installed")
    family = FAMILY_REGISTRY["nickel_medrano"]
    # Include a disconnected node and deliberately leave NiO missing.
    view.draw([*family.required_gas_species, "N2"], ["Ni"], family.reactions)
    wait_until(lambda: view.graph is not None)
    assert view.status == ""
    return view


def test_click_highlights_neighbours_and_background_clears(rendered):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    view = rendered
    node = next(n.id for n in view.graph.nodes if n.key == ("species", "NiO"))
    position = view.mapFromScene(view.node_items[node].sceneBoundingRect().center())
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert view.selected_node == node
    linked = view.graph.linked(node)
    for key, item in (*view.node_items.items(), *view.edge_items.items()):
        assert item.opacity() == (1 if key in linked else .18)
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert view.selected_node is None
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(2, 2))
    assert view.selected_node is None
    assert all(i.opacity() == 1 for i in view.node_items.values())


def test_zoom_pan_fit_and_resize(rendered, qt_app):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtTest import QTest
    view = rendered
    bounds = view.scene().itemsBoundingRect()
    def is_fitted():
        visible = view.mapToScene(view.viewport().rect()).boundingRect()
        assert visible.contains(bounds)
    is_fitted()
    original_scale = view.transform().m11()
    event = QWheelEvent(QPointF(200, 200), QPointF(view.mapToGlobal(QPoint(200, 200))),
                        QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    qt_app.sendEvent(view.viewport(), event)
    assert view.transform().m11() > original_scale
    center = view.mapToScene(view.viewport().rect().center())
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
    QTest.mouseMove(view.viewport(), QPoint(180, 150))
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(180, 150))
    assert view.mapToScene(view.viewport().rect().center()) != center
    assert view.selected_node is None
    view.fit()
    is_fitted()
    view.zoom(2)
    view.resize(430, 750)
    qt_app.processEvents()
    is_fitted()
    assert view._zoom == 1


def test_empty_selection_clears_old_graph_immediately(rendered):
    rendered.draw([], [], [])
    assert rendered.graph is None
    assert not rendered.node_items
    assert rendered.status == ""
    assert "Add species" in rendered.scene().items()[0].toPlainText()


def test_failed_render_retains_and_labels_previous_graph_then_retries(rendered, monkeypatch):
    from packed_bed_ui import reaction_graph as ui
    original = rendered.graph
    real_find = ui.find_graphviz
    def unavailable():
        raise GraphvizError("Missing layout plugin")
    monkeypatch.setattr(ui, "find_graphviz", unavailable)
    rendered.draw(["N2"], [], [])
    assert "Updating" in rendered.status and "Previous graph shown" in rendered.status
    wait_until(lambda: "Missing layout plugin" in rendered.status)
    assert rendered.graph is original
    assert "Previous graph shown" in rendered.status
    monkeypatch.setattr(ui, "find_graphviz", real_find)
    rendered.retry()
    wait_until(lambda: rendered.status == "")
    assert [n.key for n in rendered.graph.nodes] == [("species", "N2")]


def test_invalid_svg_preserves_previous_graph(rendered):
    graph = rendered.graph
    rendered._display(graph, b"<svg xmlns='http://www.w3.org/2000/svg'/>")
    assert rendered.graph is graph
    assert "invalid SVG" in rendered.status
    assert "Previous graph shown" in rendered.status


def test_obsolete_success_and_failure_cannot_replace_newer_selection(rendered):
    old_graph, old_revision = rendered.graph, rendered._revision
    old_svg = render_svg(old_graph)
    rendered.draw(["N2"], [], [])
    wait_until(lambda: rendered.status == "")
    current = rendered.graph
    rendered._rendered(old_revision, old_graph, old_svg, "")
    rendered._rendered(old_revision, old_graph, b"", "Obsolete failure")
    assert rendered.graph is current
    assert rendered.status == ""


def test_rapid_edits_only_start_latest_layout(view, monkeypatch):
    from packed_bed_ui import reaction_graph as ui
    calls = []
    def unavailable():
        calls.append(view._pending)
        raise GraphvizError("Expected test error")
    monkeypatch.setattr(ui, "find_graphviz", unavailable)
    for species in ["H2", "O2", "N2"]:
        view.draw([species], [], [])
    wait_until(lambda: "Expected test error" in view.status)
    assert len(calls) == 1
    assert calls[0].nodes[0].key == ("species", "N2")


def test_failed_to_start_is_nonblocking_and_visible(view, monkeypatch, tmp_path):
    from packed_bed_ui import reaction_graph as ui
    monkeypatch.setattr(ui, "find_graphviz", lambda: GraphvizCommand(tmp_path / "nonexistent"))
    view.draw(["N2"], [], [])
    wait_until(lambda: "Graphviz failed" in view.status)
    assert view.graph is None
    assert view._process is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX helper executable")
def test_timeout_kills_process_and_keeps_ui_responsive(view, monkeypatch, tmp_path):
    from PyQt6.QtCore import QTimer
    from packed_bed_ui import reaction_graph as ui
    executable = tmp_path / "slow neato"
    executable.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(20)\n")
    executable.chmod(0o755)
    monkeypatch.setattr(ui, "find_graphviz", lambda: GraphvizCommand(executable))
    monkeypatch.setattr(ui, "RENDER_TIMEOUT_S", .1)
    ticks = []
    timer = QTimer(view, interval=10)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    view.draw(["N2"], [], [])
    wait_until(lambda: view._process is not None)
    process = view._process
    wait_until(lambda: "timed out" in view.status)
    assert len(ticks) > 5
    assert process.done
    assert view._process is None


def test_destroy_view_cancels_active_process(qt_app):
    from PyQt6.QtCore import QCoreApplication, QEvent
    from packed_bed_ui.reaction_graph import NetworkView
    try:
        find_graphviz()
    except GraphvizError:
        pytest.skip("Graphviz is not installed")
    view = NetworkView()
    view.draw(["N2"], [], [])
    view._debounce.stop()
    view._start_render()
    process = view._process
    view.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert process.cancelled
    qt_app.processEvents()
