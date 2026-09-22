"""Species lists, reaction families and a fitted Graphviz reaction graph."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from packed_bed.kinetics import FAMILY_REGISTRY
from packed_bed.properties import PROPERTY_REGISTRY

from .editor_widgets import SelectionList, action_button, choose_items
from .reaction_graph import NetworkView


def equation(reaction):
    def side(sign):
        return " + ".join((f"{abs(value):g} " if abs(value) != 1 else "") + species
                          for species, value in reaction.stoichiometry.items() if value * sign > 0)
    return side(-1) + (" ⇌ " if reaction.reversible else " → ") + side(1)


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
        controls = QHBoxLayout()
        controls.addWidget(action_button("−", lambda: self.graph.zoom(1 / 1.2), tooltip="Zoom out"))
        controls.addWidget(action_button("+", lambda: self.graph.zoom(1.2), tooltip="Zoom in"))
        controls.addWidget(action_button("Fit", self.graph.fit, tooltip="Fit graph to window (0)"))
        controls.addStretch()
        self.graph_retry = action_button("Retry", self.graph.retry)
        controls.addWidget(self.graph_retry)
        graph_layout.addLayout(controls)
        self.graph_status = QLabel()
        self.graph_status.setWordWrap(True)
        self.graph_status.setTextFormat(Qt.TextFormat.PlainText)
        graph_layout.addWidget(self.graph_status)
        self.graph.statusChanged.connect(self.graph_status.setText)
        self.graph.statusChanged.connect(lambda text: self.graph_status.setVisible(bool(text)))
        self.graph.statusChanged.connect(lambda text: self.graph_retry.setVisible(bool(text) and not text.startswith("Updating")))
        self.graph_status.hide()
        self.graph_retry.hide()
        key = QLabel("Gas · green     Solid · sand     Reaction · violet\nDashed links: catalysts / rate dependencies. Red nodes: missing species.\nClick a node to highlight its links; click again or empty space to clear. Scroll to zoom; drag to pan.")
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
