"""Asynchronous neato SVG preview with local node highlighting and navigation."""

from PyQt6.QtCore import QByteArray, QProcess, QProcessEnvironment, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFontDatabase, QPainter, QPen
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtSvgWidgets import QGraphicsSvgItem
from PyQt6.QtWidgets import QApplication, QGraphicsItem, QGraphicsScene, QGraphicsView

from packed_bed.reaction_graph import (
    GraphvizError, RENDER_TIMEOUT_S, build_reaction_graph, find_graphviz,
)


class _RenderProcess(QProcess):
    completed = pyqtSignal(int, object, bytes, str)

    def __init__(self, command, revision, graph):
        # Outlive a closing view long enough to reap the killed subprocess.
        # Bound QObject slots automatically disconnect during object destruction.
        super().__init__(QApplication.instance())
        self.revision, self.graph = revision, graph
        self.cancelled = self.done = False
        environment = QProcessEnvironment()
        for key, value in command.environment().items():
            environment.insert(key, value)
        self.setProcessEnvironment(environment)
        self.setProgram(str(command.executable))
        self.setArguments(["-Tsvg"])
        self.timer = QTimer(self, interval=int(RENDER_TIMEOUT_S * 1000), singleShot=True)
        self.timer.timeout.connect(self._timeout)
        self.started.connect(self._send_dot)
        self.finished.connect(self._finished)
        self.errorOccurred.connect(self._error)
        QApplication.instance().aboutToQuit.connect(self._shutdown)
        QApplication.instance().destroyed.connect(self._shutdown)

    def launch(self):
        self.timer.start()
        self.start()

    def _send_dot(self):
        self.write(self.graph.dot.encode("utf-8"))
        self.closeWriteChannel()

    def cancel(self):
        self.cancelled = True
        self.timer.stop()
        if self.state() != QProcess.ProcessState.NotRunning:
            self.kill()
        else:
            self.deleteLater()

    def _shutdown(self):
        self.cancel()
        if self.state() != QProcess.ProcessState.NotRunning:
            self.waitForFinished(1000)

    def _timeout(self):
        self._complete(f"Graphviz timed out after {RENDER_TIMEOUT_S} seconds.")

    def _error(self, _error):
        self._complete(f"Graphviz failed: {self.errorString()}")

    def _finished(self, code, status):
        detail = bytes(self.readAllStandardError()).decode("utf-8", errors="replace").strip()[:2000]
        error = f"Graphviz could not render the graph: {detail or 'process failed'}" if code or status != QProcess.ExitStatus.NormalExit else ""
        self._complete(error)
        self.deleteLater()

    def _complete(self, error):
        if self.done:
            return
        self.done = True
        self.timer.stop()
        if not self.cancelled:
            self.completed.emit(self.revision, self.graph, bytes(self.readAllStandardOutput()), error)
        if self.state() != QProcess.ProcessState.NotRunning:
            self.kill()
        else:
            self.deleteLater()


class NetworkView(QGraphicsView):
    statusChanged = pyqtSignal(str)
    _loaded_fonts = set()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMinimumSize(0, 0)
        self.setBackgroundBrush(QColor("#f8fafc"))
        self.setAccessibleName("Species and reactions graph")
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # Avoid scrollbar/fit resize recursion. Dragging still moves the view.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._renderer = None
        self._process = None
        self._revision = 0
        self._pending = None
        self.graph = None
        self.node_items = {}
        self.edge_items = {}
        self.selected_node = None
        self._outline = None
        self._press = None
        self._dragged = False
        self._zoom = 1.0
        self.status = ""
        self._debounce = QTimer(self, interval=150, singleShot=True)
        self._debounce.timeout.connect(self._start_render)
        self.scene().addText("Add species and reaction families to build the system graph.")

    def _set_status(self, message):
        self.status = message
        self.statusChanged.emit(message)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def showEvent(self, event):
        super().showEvent(event)
        self.fit()

    def fit(self):
        bounds = self.scene().itemsBoundingRect().adjusted(-20, -20, 20, 20)
        # Extra scene space allows panning even along the shorter fitted axis.
        self.setSceneRect(bounds.adjusted(-bounds.width(), -bounds.height(), bounds.width(), bounds.height()))
        self.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 1.0

    def zoom(self, factor):
        new_zoom = min(12.0, max(0.2, self._zoom * factor))
        self.scale(new_zoom / self._zoom, new_zoom / self._zoom)
        self._zoom = new_zoom

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.pixelDelta().y()
        self.zoom(1.2 ** (max(-600, min(600, delta)) / 120))
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom(1.2)
        elif event.key() == Qt.Key.Key_Minus:
            self.zoom(1 / 1.2)
        elif event.key() == Qt.Key.Key_0:
            self.fit()
        elif event.key() == Qt.Key.Key_Escape:
            self.select_node(None)
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = event.position().toPoint()
            self._dragged = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press is not None:
            self._dragged |= ((event.position().toPoint() - self._press).manhattanLength()
                              >= QApplication.startDragDistance())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and self._press is not None:
            if not self._dragged:
                node = next((item.data(0) for item in self.items(event.position().toPoint())
                             if item.data(0) in self.node_items), None)
                self.select_node(None if node == self.selected_node else node)
            self._press = None

    def select_node(self, node_id):
        self.selected_node = node_id if node_id in self.node_items else None
        linked = self.graph.linked(self.selected_node) if self.selected_node else None
        for key, item in (*self.node_items.items(), *self.edge_items.items()):
            item.setOpacity(1.0 if linked is None or key in linked else 0.18)
        if self._outline is not None:
            self.scene().removeItem(self._outline)
            self._outline = None
        if self.selected_node:
            bounds = self.node_items[self.selected_node].sceneBoundingRect().adjusted(-3, -3, 3, 3)
            pen = QPen(QColor("#2563eb"), 2)
            pen.setCosmetic(True)
            self._outline = self.scene().addRect(bounds, pen)
            self._outline.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._outline.setZValue(2)

    def draw(self, gases, solids, reactions):
        self._revision += 1
        self._debounce.stop()
        if self._process is not None:
            # The revision changes before killing: its queued signals are stale.
            self._process.cancel()
            self._process = None
        self._pending = build_reaction_graph(gases, solids, reactions)
        if not self._pending.nodes:
            self._clear()
            self.scene().addText("Add species and reaction families to build the system graph.")
            self._set_status("")
            self.fit()
            return
        self._set_status("Updating graph… Previous graph shown." if self.graph else "Updating graph…")
        self._debounce.start()

    def _clear(self):
        self.scene().clear()
        self.node_items.clear()
        self.edge_items.clear()
        self._outline = None
        self.selected_node = None
        self.graph = None
        if self._renderer is not None:
            self._renderer.deleteLater()
            self._renderer = None

    def _failed(self, detail):
        suffix = " Previous graph shown." if self.graph else ""
        self._set_status(f"{detail}{suffix}")
        if self.graph is None:
            self.scene().clear()
            self.scene().addText("Graph unavailable. See the message below; edit the selection or click Retry.")
            self.fit()

    def retry(self):
        if self._pending and self._pending.nodes:
            self._revision += 1
            if self._process is not None:
                self._process.cancel()
                self._process = None
            self._set_status("Updating graph… Previous graph shown." if self.graph else "Updating graph…")
            self._debounce.start()

    def _start_render(self):
        revision, graph = self._revision, self._pending
        try:
            command = find_graphviz()
        except GraphvizError as exc:
            self._failed(str(exc))
            return
        if command.bundle:
            # Match the font used for Graphviz's label metrics on machines that
            # do not have the bundled face installed in the OS font database.
            for font in (command.bundle / "share" / "fonts").glob("*.ttf"):
                if font not in self._loaded_fonts:
                    if QFontDatabase.addApplicationFont(str(font)) >= 0:
                        self._loaded_fonts.add(font)
        process = _RenderProcess(command, revision, graph)
        self._process = process
        self.destroyed.connect(process.cancel)
        process.completed.connect(self._rendered)
        process.launch()

    def _rendered(self, revision, graph, svg, error):
        if revision != self._revision:
            return
        self._process = None
        if error:
            self._failed(error)
        else:
            self._display(graph, svg)

    def _display(self, graph, svg):
        renderer = QSvgRenderer(QByteArray(svg), self)
        if not renderer.isValid() or any(not renderer.elementExists(item.id) for item in (*graph.nodes, *graph.edges)):
            renderer.deleteLater()
            self._failed("Graphviz returned an invalid SVG graph.")
            return
        self._clear()
        self._renderer = renderer
        self.graph = graph
        for records, items, z in ((graph.edges, self.edge_items, 0), (graph.nodes, self.node_items, 1)):
            for record in records:
                item = QGraphicsSvgItem()
                item.setSharedRenderer(renderer)
                item.setElementId(record.id)
                item.setCacheMode(QGraphicsItem.CacheMode.NoCache)
                bounds = renderer.boundsOnElement(record.id)
                transform = renderer.transformForElement(record.id)
                # Each SVG item has a local origin; restore Graphviz's coordinates
                # and the parent graph transform without recomputing any routes.
                item.setTransform(transform.translate(bounds.x(), bounds.y()))
                item.setData(0, record.id)
                item.setToolTip(record.tooltip)
                item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                item.setZValue(z)
                self.scene().addItem(item)
                items[record.id] = item
        self._set_status("")
        self.fit()
