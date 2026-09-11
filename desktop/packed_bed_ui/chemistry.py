"""Species lists, reaction families and a fitted, native Qt reaction graph."""

from math import atan2, cos, sin

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from packed_bed.kinetics import FAMILY_REGISTRY
from packed_bed.properties import PROPERTY_REGISTRY

from .editor_widgets import SelectionList, action_button, choose_items


def equation(reaction):
    def side(sign):
        return " + ".join((f"{abs(value):g} " if abs(value) != 1 else "") + species
                          for species, value in reaction.stoichiometry.items() if value * sign > 0)
    return side(-1) + (" ⇌ " if reaction.reversible else " → ") + side(1)


class NetworkView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMinimumSize(0, 0)
        self.setBackgroundBrush(QColor("#f8fafc"))
        self.setAccessibleName("Species and reactions graph")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def showEvent(self, event):
        super().showEvent(event)
        self.fit()

    def fit(self):
        bounds = self.scene().itemsBoundingRect().adjusted(-25, -25, 25, 25)
        self.setSceneRect(bounds)
        self.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)

    def draw(self, gases, solids, reactions):
        scene = self.scene()
        scene.clear()
        selected = set(gases) | set(solids)
        required = {species for reaction in reactions for species in reaction.all_species}
        missing = required - selected
        gases = list(gases) + sorted(species for species in missing
                                    if PROPERTY_REGISTRY.records.get(species)
                                    and PROPERTY_REGISTRY.records[species].phase == "gas")
        solids = list(solids) + sorted(missing - set(gases))
        positions = {}
        height = max(len(gases), len(solids), len(reactions), 1) * 85
        nodes = []
        for ids, x, color in ((gases, 0, "#d9eee8"), (solids, 540, "#f5e6ca")):
            for index, species in enumerate(ids):
                y = (index + 0.5) * height / max(len(ids), 1)
                positions[species] = QPointF(x, y)
                nodes.append((species, species, x, y, "#ffe0dd" if species in missing else color, False))
        for index, reaction in enumerate(reactions):
            key = "reaction:" + reaction.id
            y = (index + 0.5) * height / len(reactions)
            positions[key] = QPointF(270, y)
            nodes.append((key, reaction.name, 270, y, "#e3e5f5", True))
        for reaction in reactions:
            target = positions["reaction:" + reaction.id]
            for species in reaction.all_species:
                source = positions[species]
                coefficient = reaction.stoichiometry.get(species, 0)
                start, end = (source, target) if coefficient <= 0 else (target, source)
                direction = 1 if end.x() > start.x() else -1
                start = start + QPointF(direction * (76 if start == target else 52), 0)
                end = end - QPointF(direction * (76 if end == target else 52), 0)
                path = QPainterPath(start)
                path.cubicTo(start + QPointF(direction * 70, 0), end - QPointF(direction * 70, 0), end)
                color = QColor("#9272a1" if coefficient == 0 else "#648494")
                pen = QPen(color, 1.5)
                if coefficient == 0:
                    pen.setStyle(Qt.PenStyle.DashLine)
                edge = scene.addPath(path, pen)
                edge.setToolTip(f"{species}: {'catalyst / rate dependency' if coefficient == 0 else abs(coefficient)}")
                arrows = [(end, path.pointAtPercent(0.97))]
                if reaction.reversible and coefficient:
                    arrows.append((start, path.pointAtPercent(0.03)))
                for tip, tail in arrows:
                    angle = atan2(tip.y() - tail.y(), tip.x() - tail.x())
                    arrow = QPolygonF([tip, tip - QPointF(9 * cos(angle - .4), 9 * sin(angle - .4)),
                                       tip - QPointF(9 * cos(angle + .4), 9 * sin(angle + .4))])
                    scene.addPolygon(arrow, QPen(color), QBrush(color))
        for key, label, x, y, color, is_reaction in nodes:
            width, node_height = (152, 64) if is_reaction else (104, 42)
            pen, brush = QPen(QColor("#a4b0bc")), QBrush(QColor(color))
            shape = (scene.addRect(x - width / 2, y - node_height / 2, width, node_height, pen, brush)
                     if is_reaction else scene.addEllipse(x - width / 2, y - node_height / 2, width, node_height, pen, brush))
            text = scene.addText(label)
            text.setDefaultTextColor(QColor("#253748"))
            text.setTextWidth(width - 8)
            option = text.document().defaultTextOption()
            option.setAlignment(Qt.AlignmentFlag.AlignCenter)
            text.document().setDefaultTextOption(option)
            text.setPos(x - (width - 8) / 2, y - text.boundingRect().height() / 2)
            tooltip = "Missing required species" if key in missing else label
            if is_reaction:
                reaction = next(item for item in reactions if "reaction:" + item.id == key)
                tooltip = equation(reaction) + "\n" + reaction.source_reference
            shape.setToolTip(tooltip)
            text.setToolTip(tooltip)
        if not nodes:
            scene.addText("Add species and reaction families to build the system graph.")
        self.fit()


class ChemistryPage(QWidget):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.loading = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        left = QWidget()
        lists = QVBoxLayout(left)
        lists.setContentsMargins(0, 0, 0, 0)
        for phase, title in (("gas", "Gas species"), ("solid", "Solid species")):
            group = QGroupBox(title)
            group_layout = QVBoxLayout(group)
            selection = SelectionList({key: (f"{key} · {record.name}", record.name)
                                       for key, record in PROPERTY_REGISTRY.records.items() if record.phase == phase},
                                      f"{phase} species")
            setattr(self, phase + "_list", selection)
            group_layout.addWidget(selection)
            lists.addWidget(group, 1)
            selection.changed.connect(lambda values, phase=phase: editor.set_species(phase, values))
        group = QGroupBox("Reaction families")
        group_layout = QVBoxLayout(group)
        self.families = QTreeWidget()
        self.families.setColumnCount(3)
        self.families.setHeaderHidden(True)
        self.families.setMinimumHeight(100)
        self.families.setAlternatingRowColors(True)
        self.families.header().setStretchLastSection(False)
        self.families.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.families.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.families.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.families.setColumnWidth(1, 88)
        self.families.setColumnWidth(2, 36)
        self.families.itemChanged.connect(self.toggle_reaction)
        group_layout.addWidget(self.families)
        group_layout.addWidget(action_button("+ Add reaction families…", self.add_families))
        lists.addWidget(group, 2)
        layout.addWidget(left, 1)
        graph = QGroupBox("Species and reactions")
        graph_layout = QVBoxLayout(graph)
        self.graph = NetworkView()
        graph_layout.addWidget(self.graph, 1)
        key = QLabel("Gas · green     Solid · sand     Reaction · violet\nDashed links: catalysts / rate dependencies. Red nodes: missing species.")
        key.setWordWrap(True)
        graph_layout.addWidget(key)
        layout.addWidget(graph, 1)

    def load(self):
        self.loading = True
        self.gas_list.set_values(self.editor.get(("chemistry", "gas_species"), []))
        self.solid_list.set_values(self.editor.get(("solids", "solid_species"), []))
        self.families.clear()
        selected = self.editor.get(("chemistry", "reaction_ids"), [])
        for name in self.editor.get(("chemistry", "reaction_families"), []):
            family = FAMILY_REGISTRY.get(name)
            root = QTreeWidgetItem(self.families, [name.replace("_", " ").title()])
            root.setData(0, Qt.ItemDataRole.UserRole, name)
            self.families.setItemWidget(root, 2, action_button("×", lambda _, name=name: self.remove_family(name),
                                                             tooltip=f"Remove {name}"))
            if family:
                self.families.setItemWidget(root, 1, action_button("+ Species", lambda _, family=family: self.add_requirements(family),
                                                                  tooltip="Add required species; compositions and loadings remain unfinished"))
                for reaction in family.reactions:
                    child = QTreeWidgetItem(root, [reaction.name])
                    child.setData(0, Qt.ItemDataRole.UserRole, reaction.id)
                    child.setCheckState(0, Qt.CheckState.Checked if reaction.id in selected else Qt.CheckState.Unchecked)
                    child.setToolTip(0, equation(reaction) + "\nRequires: " + ", ".join(reaction.all_species)
                                    + "\n" + reaction.source_reference + "\n" + reaction.notes)
            root.setExpanded(True)
        self.loading = False
        self.draw_graph()

    def add_families(self):
        existing = self.editor.get(("chemistry", "reaction_families"), [])
        added = choose_items(self, "Add reaction families", {
            key: (key.replace("_", " ").title(), "\n".join(equation(r) for r in family.reactions))
            for key, family in FAMILY_REGISTRY.items()
        }, existing)
        if added:
            self.editor.put(("chemistry", "reaction_families"), existing + added)
            reactions = self.editor.get(("chemistry", "reaction_ids"), [])
            self.editor.put(("chemistry", "reaction_ids"), list(dict.fromkeys(
                reactions + [reaction.id for key in added for reaction in FAMILY_REGISTRY[key].reactions])))
            self.load()

    def remove_family(self, name):
        self.editor.put(("chemistry", "reaction_families"), [key for key in self.editor.get(("chemistry", "reaction_families"), []) if key != name])
        removed = set(FAMILY_REGISTRY[name].reaction_ids) if name in FAMILY_REGISTRY else set()
        self.editor.put(("chemistry", "reaction_ids"), [key for key in self.editor.get(("chemistry", "reaction_ids"), []) if key not in removed])
        self.load()

    def add_requirements(self, family):
        for phase, required, path in (("gas", family.required_gas_species, ("chemistry", "gas_species")),
                                       ("solid", family.required_solid_species, ("solids", "solid_species"))):
            self.editor.set_species(phase, list(dict.fromkeys(self.editor.get(path, []) + list(required))))

    def toggle_reaction(self, item, column):
        if self.loading or item.parent() is None:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        selected = list(self.editor.get(("chemistry", "reaction_ids"), []))
        if item.checkState(0) == Qt.CheckState.Checked and key not in selected:
            selected.append(key)
        elif item.checkState(0) != Qt.CheckState.Checked:
            selected = [value for value in selected if value != key]
        self.editor.put(("chemistry", "reaction_ids"), selected)
        self.draw_graph()

    def draw_graph(self):
        selected = self.editor.get(("chemistry", "reaction_ids"), [])
        reactions = [reaction for key in self.editor.get(("chemistry", "reaction_families"), [])
                     if key in FAMILY_REGISTRY for reaction in FAMILY_REGISTRY[key].reactions if reaction.id in selected]
        self.graph.draw(self.editor.get(("chemistry", "gas_species"), []),
                        self.editor.get(("solids", "solid_species"), []), reactions)
