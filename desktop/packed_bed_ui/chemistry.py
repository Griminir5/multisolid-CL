"""Species lists, reaction families and a fitted Graphviz reaction graph."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget, QDialog, QFormLayout, QComboBox,
)

from types import SimpleNamespace
from packed_bed.definitions import bound_reactions, resolve_bindings, binding_candidates
from packed_bed.plugins.catalogue import builtin_manifest, split_ref

from .editor_widgets import SelectionList, action_button, choose_items, dialog_buttons
from .catalogue_widgets import species_choices, component_key, definition_details
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
            selection = SelectionList({}, f"{phase} species")
            setattr(self, phase + "_list", selection)
            group_layout.addWidget(selection)
            lists.addWidget(group, 1)
            selection.changed.connect(lambda values, phase=phase: editor.set_species(phase, values))
        group = QGroupBox("Reaction families")
        group_layout = QVBoxLayout(group)
        self.families = QTreeWidget()
        self.families.setColumnCount(4)
        self.families.setHeaderHidden(True)
        self.families.setMinimumHeight(100)
        self.families.setAlternatingRowColors(True)
        self.families.header().setStretchLastSection(False)
        self.families.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 4):
            self.families.header().setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
        self.families.setColumnWidth(1, 88)
        self.families.setColumnWidth(2, 85)
        self.families.setColumnWidth(3, 36)
        self.families.itemChanged.connect(self.toggle_reaction)
        group_layout.addWidget(self.families)
        group_layout.addWidget(action_button("+ Add reaction families…", self.add_families))
        lists.addWidget(group, 2)
        self.cautions = QLabel()
        self.cautions.setWordWrap(True)
        lists.addWidget(self.cautions)
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
        try:
            catalogue = self.editor.catalogue()
            for phase in ('gas', 'solid'):
                getattr(self, phase + '_list').catalog, _ = species_choices(
                    catalogue, phase, self.editor.get(('chemistry', 'species_definitions'), {}))
        except (ValueError, OSError) as exc:
            self.graph_status.setText(str(exc))
            self.graph_status.show()
            self.loading = False
            return
        self.gas_list.set_values(self.editor.get(("chemistry", "gas_species"), []))
        self.solid_list.set_values(self.editor.get(("solids", "solid_species"), []))
        self.families.clear()
        selected = self.editor.get(("chemistry", "reaction_ids"), [])
        for name in self.editor.get(("chemistry", "reaction_families"), []):
            family = self.family(name)
            root = QTreeWidgetItem(self.families, [family.label if family else name + ' — Missing definition'])
            root.setData(0, Qt.ItemDataRole.UserRole, name)
            self.families.setItemWidget(root, 3, action_button("×", lambda _, name=name: self.remove_family(name),
                                                             tooltip=f"Remove {name}"))
            if family:
                root.setToolTip(0, definition_details(catalogue, 'mechanisms', family.reference))
                self.families.setItemWidget(root, 2, action_button('Bindings…', lambda _, name=name: self.bind_roles(name)))
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
        catalogue = self.editor.catalogue()
        definitions = {component_key(ref): ref for ref in catalogue.entries('mechanisms')}
        added = choose_items(self, "Add reaction families", {
            key: (catalogue.label('mechanisms', ref), definition_details(catalogue, 'mechanisms', ref))
            for key, ref in definitions.items()
        }, existing)
        if added:
            mappings = self.editor.get(('chemistry', 'mechanisms'), {}).copy()
            for key in added:
                if definitions[key] != 'builtin:' + key:
                    mappings[key] = {'definition': definitions[key], 'bindings': {}}
            if mappings:
                self.editor.put(('chemistry', 'mechanisms'), mappings)
            self.editor.put(("chemistry", "reaction_families"), existing + added)
            reactions = self.editor.get(("chemistry", "reaction_ids"), [])
            self.editor.put(("chemistry", "reaction_ids"), list(dict.fromkeys(
                reactions + [reaction.id for key in added for reaction in self.family(key).reactions])))
            self.update_bindings()
            self.load()

    def remove_family(self, name):
        family = self.family(name)
        remaining = [key for key in self.editor.get(("chemistry", "reaction_families"), []) if key != name]
        self.editor.put(("chemistry", "reaction_families"), remaining)
        selected = self.editor.get(("chemistry", "reaction_ids"), [])
        removed = {r.id for r in family.reactions} if family else set()
        # Ownership must remain recoverable when a plugin is missing or changed.
        legacy = builtin_manifest().mechanisms.get(name)
        if legacy:
            removed.update(r.id for r in legacy.reactions)
        else:
            other_prefixes = tuple(key + '/' for key in remaining if key.startswith(name + '/'))
            removed.update(key for key in selected if key.startswith(name + '/') and not key.startswith(other_prefixes))
        mappings = self.editor.get(('chemistry', 'mechanisms'), {}).copy()
        mappings.pop(name, None)
        if self.editor.get(('chemistry', 'mechanisms')) is not None:
            self.editor.put(('chemistry', 'mechanisms'), mappings)
        self.editor.put(("chemistry", "reaction_ids"), [key for key in selected if key not in removed])
        self.load()

    def add_requirements(self, family):
        for phase, required, path in (("gas", family.required_gas_species, ("chemistry", "gas_species")),
                                       ("solid", family.required_solid_species, ("solids", "solid_species"))):
            catalogue = self.editor.catalogue()
            provider, _ = split_ref(family.reference)
            existing = self.editor.get(path, [])
            selected_refs = self.editor.get(('chemistry', 'species_definitions'), {})
            _, available = species_choices(catalogue, phase, selected_refs)
            additions = []
            for role in required:
                if role in family.bindings:
                    continue
                if any(catalogue.get('species', selected_refs.get(key, 'builtin:' + key)).chemical_key == role for key in existing):
                    continue
                own = [ref for ref, spec in catalogue.entries('species').items()
                       if spec.chemical_key == role and spec.phase == phase and split_ref(ref)[0] == provider]
                reference = own[0] if len(own) == 1 else 'builtin:' + role
                try:
                    if catalogue.get('species', reference).phase == phase:
                        additions.append(next(key for key, ref in available.items() if ref == reference))
                except ValueError:
                    continue
            self.editor.set_species(phase, list(dict.fromkeys(existing + additions)))

    def species(self):
        catalogue, selected = self.editor.catalogue(), {}
        mappings = self.editor.get(('chemistry', 'species_definitions'), {})
        for key in self.editor.get(('chemistry', 'gas_species'), []) + self.editor.get(('solids', 'solid_species'), []):
            try:
                selected[key] = catalogue.get('species', mappings.get(key, 'builtin:' + key)).model_dump()
            except ValueError:
                pass
        return selected

    def family(self, instance):
        catalogue = self.editor.catalogue()
        declared = self.editor.get(('chemistry', 'mechanisms'), {}).get(instance, {})
        reference = declared.get('definition', 'builtin:' + instance)
        try:
            definition = catalogue.get('mechanisms', reference)
        except ValueError:
            return None
        species = {key: {'definition': spec} for key, spec in self.species().items()}
        bindings, _ = resolve_bindings(instance, definition, declared.get('bindings', {}), species,
                                      self.editor.get(('chemistry', 'reaction_ids'), []), reference, repair=True)
        reactions = bound_reactions(instance, definition, bindings, reference)
        return SimpleNamespace(reference=reference, definition=definition, bindings=bindings, reactions=reactions,
                               label=instance + ' · ' + catalogue.label('mechanisms', reference),
                               required_gas_species=definition.gases, required_solid_species=definition.solids)

    def update_bindings(self):
        """Save UI defaults when the user changes the available species or families."""
        mappings = self.editor.get(('chemistry', 'mechanisms'), {}).copy()
        for name in self.editor.get(('chemistry', 'reaction_families'), []):
            family = self.family(name)
            if family:
                mappings[name] = {'definition': family.reference, 'bindings': dict(family.bindings)}
        if mappings != self.editor.get(('chemistry', 'mechanisms'), {}):
            self.editor.put(('chemistry', 'mechanisms'), mappings)

    def bind_roles(self, name):
        family = self.family(name)
        dialog = QDialog(self)
        dialog.setWindowTitle('Species bindings — ' + family.label)
        layout, form = QVBoxLayout(dialog), QFormLayout()
        fields = {}
        for role, candidates in binding_candidates(family.definition, self.species()).items():
            field = QComboBox()
            field.setObjectName(role)
            for component in candidates:
                field.addItem(self.editor.species_label(component), component)
            field.setCurrentIndex(max(0, field.findData(family.bindings.get(role))))
            field.setPlaceholderText('No matching species selected')
            field.setEnabled(bool(candidates))
            fields[role] = field
            form.addRow(role, field)
        layout.addLayout(form)
        if any(not field.count() for field in fields.values()):
            layout.addWidget(QLabel('Use + Species to add any missing components, then choose their bindings here.'))
        layout.addWidget(dialog_buttons(dialog))
        if dialog.exec() == QDialog.DialogCode.Accepted:
            mappings = self.editor.get(('chemistry', 'mechanisms'), {}).copy()
            mappings[name] = {'definition': family.reference,
                              'bindings': {role: field.currentData() for role, field in fields.items() if field.currentData()}}
            self.editor.put(('chemistry', 'mechanisms'), mappings)
            self.load()

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
        families = [self.family(key) for key in self.editor.get(('chemistry', 'reaction_families'), [])]
        reactions = [r for family in families if family for r in family.reactions if r.id in selected]
        gases = self.editor.get(('chemistry', 'gas_species'), [])
        solids = self.editor.get(('solids', 'solid_species'), [])
        labels = {key: self.editor.species_label(key) for key in gases + solids}
        for family in families:
            if family:
                labels.update({r.id: r.name + ' — ' + self.editor.catalogue().manifest(split_ref(family.reference)[0]).name for r in family.reactions})
        equations = [tuple(sorted(r.stoichiometry.items())) for r in reactions]
        self.cautions.setText('Overlapping selected reactions contribute additive rates. Review the mechanism selections and bindings.'
                              if len(equations) != len(set(equations)) else '')
        records = {key: SimpleNamespace(phase=phase) for phase, keys in (('gas', gases), ('solid', solids)) for key in keys}
        self.graph.draw(self.editor.get(("chemistry", "gas_species"), []),
                        self.editor.get(("solids", "solid_species"), []), reactions, SimpleNamespace(records=records), labels=labels)
