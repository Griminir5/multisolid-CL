"""One project catalogue tree, with scoped plugin and definition actions."""
from uuid import uuid4

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QInputDialog, QLineEdit, QMessageBox, QSplitter, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget)

from packed_bed.plugins.catalogue import split_ref
from packed_bed.plugins.storage import inspect_package, has_code, code_approved
from .catalogue_widgets import DefinitionDetails
from .editor_widgets import action_button, choose_items, dialog_buttons
from .plugin_forms import PluginEditor, PluginCheck, available_key


class PluginsDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project, self.actions = project, {}
        self.setWindowTitle('Project plugins')
        self.resize(1100, 720)
        layout = QVBoxLayout(self)
        self.browser = QWidget()
        body = QVBoxLayout(self.browser)
        self.add_actions(body, 'Plugins', (('Register plugin…', self.add), ('Create plugin…', self.new),
            ('Enable / disable', self.toggle), ('Remove', self.remove), ('Check', self.check), ('Export…', self.export)))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText('Filter plugins and definitions…')
        body.addWidget(self.filter)
        splitter = QSplitter()
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Plugin / definition', 'Kind'])
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        splitter.addWidget(self.tree)
        right = QWidget()
        details = QVBoxLayout(right)
        self.add_actions(details, 'Definitions', (('Add species…', self.add_species), ('Create variant…', self.variant),
            ('Edit…', self.edit), ('Replace uses in project…', self.replace)))
        self.details = DefinitionDetails()
        details.addWidget(self.details, 1)
        self.uses = QLabel()
        self.uses.setWordWrap(True)
        details.addWidget(self.uses)
        splitter.addWidget(right)
        splitter.setSizes([440, 660])
        body.addWidget(splitter, 1)
        layout.addWidget(self.browser, 1)
        self.checker = PluginCheck(self)
        self.checker.busyChanged.connect(lambda busy: self.browser.setEnabled(not busy))
        layout.addWidget(self.checker)
        layout.addWidget(dialog_buttons(self, QDialogButtonBox.StandardButton.Close))
        self.tree.currentItemChanged.connect(self.load_definition)
        self.tree.itemDoubleClicked.connect(lambda *_: self.actions['Create variant…' if self.provider() == 'builtin' else 'Edit…'].click())
        self.filter.textChanged.connect(self.filter_definitions)
        self.reload()

    def add_actions(self, layout, title, entries):
        group = QGroupBox(title)
        buttons = QHBoxLayout(group)
        for label, callback in entries:
            self.actions[label] = action_button(label, lambda _, fn=callback: self.perform(fn))
            buttons.addWidget(self.actions[label])
        layout.addWidget(group)

    def perform(self, function):
        try:
            self.checker.message('')
            function()
            if not self.checker.busy:
                self.reload()
        except (ValueError, OSError, KeyError) as exc:
            self.checker.message(str(exc))

    def provider(self):
        item = self.tree.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole)[0] if item else 'builtin'

    def selection(self):
        item = self.tree.currentItem()
        return (item.data(0, Qt.ItemDataRole.UserRole)[1:] or None) if item else None

    def reload(self, selected=None, definition=None):
        definition = definition or (self.selection() if selected in (None, self.provider()) else None)
        selected = (selected or self.provider(), *(definition or ()))
        self.catalogue, self.catalogue_errors = self.project.plugins.browse_catalogue(include_disabled=True)
        self.tree.blockSignals(True)
        self.tree.clear()
        items = {}
        def add(parent, text, kind, key):
            item = QTreeWidgetItem(parent, [text, kind])
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            items[key] = item
            return item
        for entry in [{'id': 'builtin', 'enabled': True}, *self.project.plugins.entries]:
            provider = entry['id']
            error = self.catalogue_errors.get(provider)
            manifest = None if error else self.catalogue.manifest(provider)
            status = 'Unavailable' if error else 'Always enabled' if provider == 'builtin' else 'Enabled' if entry['enabled'] else 'Disabled'
            if manifest and provider != 'builtin' and entry['enabled'] and has_code(manifest) and not code_approved(self.catalogue.paths[provider]):
                status += ' · Needs local approval'
            root = add(self.tree, f'{manifest.name if manifest else provider} — {status}', 'Plugin', (provider,))
            root.setExpanded(provider == selected[0] and len(selected) > 1)
            if error:
                continue
            for kind, label in (('species', 'Species'), ('mechanisms', 'Reaction family'), ('correlations', 'Correlation')):
                for key, definition in getattr(manifest, kind).items():
                    identity = (provider, kind, provider + ':' + key)
                    item = add(root, definition.name, label, identity)
                    if kind == 'mechanisms':
                        for reaction in definition.reactions:
                            add(item, reaction.name, 'Reaction', (*identity, reaction.id))
        item = items.get(selected, items.get(selected[:1], items[('builtin',)]))
        self.tree.setCurrentItem(item)
        if item.parent():
            item.parent().setExpanded(True)
        self.tree.blockSignals(False)
        self.load_definition()
        self.filter_definitions()

    def filter_definitions(self):
        query = self.filter.text().casefold()
        matches = []
        def visible(item, matched=False):
            if query in item.text(0).casefold():
                matches.append(item)
                matched = True
            children = [visible(item.child(i), matched) for i in range(item.childCount())]
            item.setHidden(not (matched or any(children)))
            if query:
                item.setExpanded(any(children))
            return not item.isHidden()
        for i in range(self.tree.topLevelItemCount()):
            visible(self.tree.topLevelItem(i))
        selected = self.tree.currentItem()
        if selected is None or selected.isHidden():
            self.tree.setCurrentItem(matches[0] if matches else None)

    def load_definition(self, *_):
        provider, selection = self.provider(), self.selection()
        editable = bool(selection and selection[0] in ('species', 'mechanisms'))
        local = provider != 'builtin'
        available = provider not in self.catalogue_errors
        enabled = provider == 'builtin' or next(entry['enabled'] for entry in self.project.plugins.entries if entry['id'] == provider)
        for label in ('Enable / disable', 'Check', 'Export…', 'Add species…'):
            self.actions[label].setEnabled(local and available)
        self.actions['Remove'].setEnabled(local)
        self.actions['Edit…'].setEnabled(local and editable)
        self.actions['Create variant…'].setEnabled(editable)
        self.actions['Replace uses in project…'].setEnabled(editable and len(selection) == 2 and enabled)
        self.uses.setText('Used by: ' + ('; '.join(self.project.plugins.users(provider)) or 'No current cases or studies'))
        if not available:
            self.details.setPlainText(self.catalogue_errors[provider])
            return
        manifest = self.catalogue.manifest(provider)
        needs_approval = local and has_code(manifest) and not code_approved(self.catalogue.paths[provider])
        self.actions['Enable / disable'].setText('Allow code…' if enabled and needs_approval else 'Disable' if enabled else 'Enable')
        if selection:
            self.details.show_definition(self.catalogue, *selection)
        else:
            self.details.setPlainText(f'{manifest.name}\n{manifest.description}\nSource: {manifest.source or "Not specified"}\n\n'
                                      f'{len(manifest.species)} species, {len(manifest.mechanisms)} reaction families')

    def add(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Register plugin archive', '', 'MultiSolid plugins (*.msplugin *.zip)')
        if not path:
            return
        with inspect_package(path) as package:
            previous = next((entry for entry in self.project.plugins.entries if entry['id'] == package.manifest.id), None)
            replacing = bool(previous and self.catalogue.hashes.get(previous['id']) != package.digest)
            if replacing:
                answer = QMessageBox.question(self, 'Replace plugin',
                    'Replace the project’s copy of this plugin? Cases and studies will use the new contents. Saved runs keep their inputs.',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                if answer != QMessageBox.StandardButton.Yes:
                    return
            if not has_code(package.manifest) or (previous and previous['enabled']):
                self.start_check(path, lambda checked: self.project.plugins.add(checked.folder))
                return
        entry = self.project.plugins.add(path)
        self.reload(entry['id'])

    def toggle(self):
        ident = self.provider()
        entry = next(item for item in self.project.plugins.entries if item['id'] == ident)
        needs_approval = has_code(self.catalogue.manifest(ident)) and not code_approved(self.catalogue.paths[ident])
        enabling = not entry['enabled'] or needs_approval
        if enabling:
            source = self.catalogue.paths[ident]
            self.start_check(source, lambda _: self.project.plugins.set_enabled(ident, True))
        else:
            self.project.plugins.set_enabled(ident, False)

    def remove(self):
        self.project.plugins.remove(self.provider())

    def new(self):
        name, accepted = QInputDialog.getText(self, 'Create plugin', 'Plugin name')
        if accepted and name.strip():
            entry = self.project.plugins.create(name.strip())
            self.reload(entry['id'])

    def open_editor(self, editor):
        if editor.exec() == QDialog.DialogCode.Accepted:
            self.reload(editor.data['id'], editor.saved_selection)
        editor.deleteLater()

    def add_species(self):
        provider = self.provider()
        data = self.catalogue.manifest(provider).model_dump(mode='json')
        key = available_key('new_species', data['species'])
        data['species'][key] = {}
        self.open_editor(PluginEditor(self.project, data, 'species', key, source=self.catalogue.paths[provider],
                                      editing=True, adding=True, parent=self))

    def variant(self):
        kind, ref, *reaction = self.selection()
        provider, key = split_ref(ref)
        source = None
        if provider == 'builtin':
            data = {kind: {key: self.catalogue.get(kind, ref).model_dump(mode='json')}}
        else:
            data = self.catalogue.manifest(provider).model_dump(mode='json')
            source = self.catalogue.paths[provider]
        data.update(id='plugin_' + uuid4().hex[:8], name=self.catalogue.get(kind, ref).name + ' variant',
                    derived_from=ref)
        self.open_editor(PluginEditor(self.project, data, kind, key, source=source, reaction_id=reaction[0] if reaction else None, parent=self))

    def edit(self):
        kind, ref, *reaction = self.selection()
        provider, key = split_ref(ref)
        self.open_editor(PluginEditor(self.project, self.catalogue.manifest(provider).model_dump(mode='json'), kind, key,
                         source=self.catalogue.paths[provider], editing=True, reaction_id=reaction[0] if reaction else None, parent=self))

    def check(self):
        source = self.catalogue.paths[self.provider()]
        self.start_check(source)

    def start_check(self, source, on_success=None):
        def done(package):
            if on_success:
                on_success(package)
            self.reload(package.manifest.id)
        self.checker.run(source, done)

    def reject(self):
        self.checker.cancel()
        super().reject()

    def export(self):
        ident = self.provider()
        path, _ = QFileDialog.getSaveFileName(self, 'Export plugin', ident + '.msplugin', 'MultiSolid plugins (*.msplugin)')
        if path:
            self.project.plugins.export(ident, path)

    def replace(self):
        kind, source = self.selection()
        choices = {}
        catalogue, _ = self.project.plugins.browse_catalogue()
        for ref in catalogue.entries(kind):
            if ref == source:
                continue
            try:
                self.project.plugins.replacement_uses(kind, source, ref)
                choices[ref] = (catalogue.label(kind, ref), ref)
            except ValueError:
                continue
        added = choose_items(self, 'Choose one compatible replacement', choices, [])
        if not added:
            return
        if len(added) != 1:
            raise ValueError('Choose exactly one replacement.')
        target = added[0]
        uses = self.project.plugins.replacement_uses(kind, source, target)
        if not uses:
            raise ValueError('No ordinary cases or reusable definitions use this definition.')
        answer = QMessageBox.question(self, 'Replace uses in project',
            f'{catalogue.label(kind, source)}\n→ {catalogue.label(kind, target)}\n\n' + '\n'.join(uses)
            + '\n\nComponent IDs, compositions and bindings stay the same. Study selections and saved runs are unchanged. Apply these edits?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.project.plugins.replace_uses(kind, source, target)
