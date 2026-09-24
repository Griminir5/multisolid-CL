from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic

import pytest
import yaml

from packed_bed.config import load_case
from packed_bed.plugins.catalogue import builtin_manifest
from packed_bed.plugins.storage import approve_hash, package_hash, inspect_package, pack_plugin, code_hash
from packed_bed_ui.project import Project, read_json, write_json
from packed_bed_ui.worker import activate_snapshot, verify_snapshot
from packed_bed_ui.studies import Factor, ReusableDefinition
from packed_bed_ui.inputs import definition_payload

EXAMPLES = Path(__file__).parents[2] / 'packed_bed/examples/plugins'


def wait_for_check(dialog):
    from PyQt6.QtTest import QTest
    deadline = monotonic() + 15
    while dialog.checker.busy:
        assert monotonic() < deadline, 'Plugin check did not finish.'
        QTest.qWait(10)


@pytest.fixture(autouse=True)
def isolated_approval(monkeypatch, tmp_path):
    monkeypatch.setenv('MULTISOLID_PLUGIN_APPROVALS', str(tmp_path / 'approvals.json'))


def variant(tmp_path, species='N2', ident='lab', factor=1.1):
    folder = tmp_path / ident
    folder.mkdir(exist_ok=True)
    definition = builtin_manifest().species[species].model_dump(mode='json')
    definition['mw'] *= factor
    data = {'id': ident, 'name': 'Lab properties', 'species': {species: definition}}
    (folder / 'manifest.yaml').write_text(yaml.safe_dump(data))
    return folder


def success(project, case):
    job = read_json(project.prepare_execution([case]))
    folder = activate_snapshot(case.root / f'.pending-{job["attempt_id"]}')
    write_json(folder / 'status.json', {'state': 'completed'})
    return folder


def test_incomplete_snapshot_lock_cannot_use_active_definitions(tmp_path, source_case):
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files(source_case)
    folder = success(project, case)
    write_json(folder / 'inputs/definitions.json', {'root': '..', 'lock': {}})
    with pytest.raises(ValueError, match='exact definition lock'):
        load_case(folder / 'inputs/run.yaml')


def test_add_replace_update_studies_and_remove_preserve_scientific_identity(tmp_path, source_case):
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files(source_case)
    original = deepcopy(case.documents)
    before = case.fingerprint()
    entry = project.plugins.add(variant(tmp_path))
    assert case.documents == original and case.fingerprint() == before
    project.plugins.replace_uses('species', 'builtin:N2', 'lab:N2')
    assert case.documents['chemistry']['gas_species'] == ['N2']
    assert case.documents['program'] == original['program']
    first = success(project, case)
    recorded = (first / 'snapshot.json').read_bytes()
    study = project.study_store.create('Frozen', case)
    study.factors = [Factor('cells', 'axial_cells', [3, 4])]
    project.study_store.save_study(study)
    generated = project.study_store.apply_preview(project.study_store.preview(study))
    frozen_fingerprints = [item.fingerprint() for item in generated]
    source = variant(tmp_path, factor=1.2)
    updated = project.plugins.add(source)
    assert updated['hash'] != entry['hash'] and case.state()['stale']
    assert not project.study_store.needs_update(study.id)
    assert [item.fingerprint() for item in generated] != frozen_fingerprints
    for item in generated:
        item.validate_for_run()
        runtime = load_case(item.root / 'inputs/run.yaml')
        assert runtime.definitions.selection.lock['plugins']['lab'] == updated['hash']
    assert (first / 'snapshot.json').read_bytes() == recorded
    assert verify_snapshot(first).definitions.selection.lock['plugins']['lab'] == entry['hash']
    with pytest.raises(ValueError, match='Study baseline'):
        project.plugins.remove('lab')
    project.plugins.replace_uses('species', 'lab:N2', 'builtin:N2')
    assert all(item.documents['chemistry']['species_definitions']['N2'] == 'lab:N2' for item in generated)
    project.study_store.delete_study(study.id)
    project.plugins.remove('lab')
    # Result-only references retain their immutable files after removal.
    assert verify_snapshot(first).definitions.selection.lock['plugins']['lab'] == entry['hash']
    reopened = Project.open(project.root)
    assert not reopened.plugins.entries


def test_replacement_updates_reusable_definitions_atomically_and_rejects_other_chemistry(tmp_path, source_case, monkeypatch):
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files(source_case)
    project.plugins.add(variant(tmp_path))
    definition = ReusableDefinition('program', 'Feed', 'program', definition_payload('program', case.documents))
    project.study_store.save_definition(definition)
    original = deepcopy(case.documents)
    original_save = __import__('packed_bed_ui.study_store', fromlist=['write_json']).write_json
    def fail(path, value):
        if path == project.root / 'project.json':
            raise OSError('Interrupted commit')
        original_save(path, value)
    with monkeypatch.context() as scoped:
        scoped.setattr('packed_bed_ui.study_store.write_json', fail)
        with pytest.raises(OSError, match='Interrupted'):
            project.plugins.replace_uses('species', 'builtin:N2', 'lab:N2')
    assert case.documents == original
    assert Project.open(project.root).cases[0].documents == original
    project.plugins.replace_uses('species', 'builtin:N2', 'lab:N2')
    assert project.study_store.definitions['program'].payload['species_definitions']['N2'] == 'lab:N2'
    with pytest.raises(ValueError, match='chemical key'):
        project.plugins.replace_uses('species', 'builtin:Ni', 'lab:N2')


@pytest.mark.parametrize('occupied_alias', [False, True])
def test_original_species_can_be_added_after_project_replacement(tmp_path, source_case, qt_app, occupied_alias):
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtWidgets import QListWidget
    from packed_bed_ui.catalogue_widgets import species_choices
    from packed_bed_ui.editor import CaseEditor

    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files(source_case)
    gases = ['H2O', 'H2O_2'] if occupied_alias else ['H2O']
    case.documents['chemistry']['gas_species'] = gases
    case.documents['chemistry']['species_definitions'] = {'H2O_2': 'builtin:N2'} if occupied_alias else {}
    case.documents['program']['inlet_composition']['initial'] = {key: 1 / len(gases) for key in gases}
    case.save()
    project.plugins.add(variant(tmp_path, species='H2O'))
    project.plugins.replace_uses('species', 'builtin:H2O', 'lab:H2O')
    editor = CaseEditor()
    assert editor.set_case(case)
    mappings = deepcopy(case.documents['chemistry']['species_definitions'])
    choices, references = species_choices(editor.catalogue(), 'gas', mappings)
    original = next(key for key, ref in references.items() if ref == 'builtin:H2O')
    assert original not in gases
    assert 'Built-in' in choices[original][0]
    assert 'Lab properties' in choices['H2O'][0]
    assert list(references.values()).count('lab:H2O') == 1
    assert mappings == case.documents['chemistry']['species_definitions']

    def choose_original():
        dialog = qt_app.activeModalWidget()
        items = dialog.findChild(QListWidget)
        item = next(items.item(i) for i in range(items.count())
                    if items.item(i).data(Qt.ItemDataRole.UserRole) == original)
        item.setCheckState(Qt.CheckState.Checked)
        dialog.accept()
    QTimer.singleShot(0, choose_original)
    editor.chemistry.gas_list.add_button.click()
    assert case.documents['chemistry']['gas_species'] == gases + [original]
    composition = {key: 0.0 for key in gases}
    composition.update({'H2O': .5, original: .5})
    editor.put(('program', 'inlet_composition', 'initial'), composition)
    assert editor.save()
    editor.close()
    reopened = Project.open(project.root).cases[0]
    reopened.validate_for_run()
    assert reopened.documents['chemistry']['species_definitions'] == {**mappings, original: 'builtin:H2O'}
    runtime = load_case(reopened.root / 'inputs/run.yaml')
    records = runtime.definitions.properties.records
    water_mw = builtin_manifest().species['H2O'].mw
    assert records[original].mw == water_mw
    assert records['H2O'].mw == pytest.approx(water_mw * 1.1)


@pytest.mark.parametrize('instance', ['lab.nickel', 'nickel_medrano'])
def test_removing_unreadable_family_clears_its_reactions(tmp_path, qt_app, instance):
    from packed_bed.definitions import reaction_instance_id
    from packed_bed_ui.editor import CaseEditor

    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files('packed_bed/examples/default_case/run.yaml')
    definition = builtin_manifest().mechanisms['nickel_medrano']
    plugin = tmp_path / 'lab'
    plugin.mkdir()
    (plugin / 'manifest.yaml').write_text(yaml.safe_dump({
        'id': 'lab', 'name': 'Lab kinetics', 'mechanisms': {'nickel': definition.model_dump(mode='json')}}))
    project.plugins.add(plugin)
    retained = [r.id for r in builtin_manifest().mechanisms['reforming_xu_froment'].reactions]
    case.documents['chemistry'].update(
        reaction_families=['reforming_xu_froment', instance],
        reaction_ids=retained + [reaction_instance_id(instance, r.id, 'lab:nickel') for r in definition.reactions],
        mechanisms={instance: {'definition': 'lab:nickel', 'bindings': {}}})
    case.save()
    case.validate_for_run()
    installed = project.plugins.catalogue().paths['lab']
    (installed / 'manifest.yaml').write_text('invalid: [')
    editor = CaseEditor()
    assert editor.set_case(case)
    assert editor.chemistry.family(instance) is None

    row = editor.chemistry.families.topLevelItem(1)
    editor.chemistry.families.itemWidget(row, 3).click()
    assert editor.save()
    editor.close()

    reopened = Project.open(project.root).cases[0]
    assert reopened.documents['chemistry']['reaction_families'] == ['reforming_xu_froment']
    assert reopened.documents['chemistry']['reaction_ids'] == retained
    assert not reopened.documents['chemistry']['mechanisms']
    reopened.validate_for_run()


def test_browsing_code_metadata_never_imports_it_and_copy_never_grants_approval(tmp_path, source_case, qt_app):
    from packed_bed_ui.plugins_dialog import PluginsDialog
    from packed_bed_ui.editor import CaseEditor
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files(source_case)
    source = tmp_path / 'code'
    shutil.copytree(EXAMPLES / 'correlation', source)
    module = source / 'correlation.py'
    module.write_text('raise RuntimeError("GUI imported executable code")\n' + module.read_text())
    entry = project.plugins.add(source)
    assert not entry['enabled']
    with pytest.raises(ValueError, match='[Aa]llow|approval'):
        project.plugins.set_enabled(entry['id'], True)
    # A copied project's enabled flag is metadata, not permission to import.
    project.metadata['plugins'][0]['enabled'] = True
    case.documents['chemistry']['gas_species'] = ['ArDemo']
    case.documents['chemistry']['species_definitions'] = {'ArDemo': 'example_correlation:ArDemo'}
    case.documents['program']['inlet_composition']['initial'] = {'ArDemo': 1.0}
    case.save()
    assert case.resolve().selection.species['ArDemo']['definition']['mw'] == .039948
    with pytest.raises(ValueError, match='[Aa]llow|approval'):
        case.validate_for_run()
    dialog = PluginsDialog(project)
    assert 'Always enabled' in dialog.tree.topLevelItem(0).text(0)
    dialog.tree.setCurrentItem(dialog.tree.topLevelItem(1))
    assert 'Example property correlation' in dialog.details.toPlainText()
    editor = CaseEditor()
    editor.set_case(case)
    assert 'Example property correlation' in editor.chemistry.gas_list.table.item(0, 0).text()
    assert not any(name.startswith('_multisolid_plugin_' + entry['hash']) for name in sys.modules)
    editor.chemistry.graph._debounce.stop()
    editor.close()
    dialog.close()


def test_browser_tree_distinguishes_plugins_families_and_reactions(tmp_path, qt_app):
    from PyQt6.QtCore import Qt
    from packed_bed_ui.plugins_dialog import PluginsDialog

    project = Project.create(tmp_path / 'project')
    project.plugins.add(variant(tmp_path))
    dialog = PluginsDialog(project)
    builtin = dialog.tree.topLevelItem(0)
    assert not builtin.isExpanded()  # All project plugins are visible initially.
    family = next(builtin.child(i) for i in range(builtin.childCount())
                  if builtin.child(i).data(0, Qt.ItemDataRole.UserRole)[-1] == 'builtin:nickel_medrano')
    dialog.tree.setCurrentItem(family)
    assert all(family.child(i).text(0) in dialog.details.toPlainText() for i in range(family.childCount()))
    reaction = family.child(0)
    dialog.tree.setCurrentItem(reaction)
    assert reaction.text(0) in dialog.details.toPlainText()
    assert family.child(1).text(0) not in dialog.details.toPlainText()
    assert 'R0_M.H2' in dialog.details.toPlainText()
    assert not dialog.actions['Replace uses in project…'].isEnabled()
    dialog.tree.setCurrentItem(dialog.tree.topLevelItem(1))
    assert dialog.actions['Add species…'].isEnabled()
    assert not dialog.actions['Edit…'].isEnabled()
    dialog.filter.setText(reaction.text(0))
    assert builtin.isExpanded() and family.isExpanded() and not reaction.isHidden()
    assert dialog.tree.currentItem() is reaction
    dialog.close()


def test_invalid_forms_do_not_save_and_preserve_coefficient_basis(tmp_path, qt_app):
    from packed_bed_ui.plugin_forms import PluginEditor, SpeciesForm
    project = Project.create(tmp_path / 'project')
    data = {'id': 'mine', 'name': 'My plugin', 'species': {'water': builtin_manifest().species['H2O'].model_dump(mode='json')}}
    editor = PluginEditor(project, data, 'species', 'water')
    editor.form.fields['mw'].setText('unfinished')
    editor.save_plugin()
    assert not editor.checker.busy
    assert 'mw' in editor.checker.details.toPlainText()
    assert editor.form.fields['mw'].text() == 'unfinished'
    editor.reject()
    assert not (project.root / '.plugin-drafts').exists()
    assert project.plugins.entries == []
    form = SpeciesForm(data['species']['water'])
    form.correlation_fields['coefficients'].setText('20, 0.1, 0.0001')
    assert form.value()['enthalpy']['parameters']['coefficients'] == [20, .1, .0001]
    form.model.setCurrentIndex(1)
    assert form.value()['enthalpy']['model'] == 'builtin:shomate'
    form.model.setCurrentIndex(0)
    assert form.value()['enthalpy']['parameters']['coefficients'] == [20, .1, .0001]
    form.close()


def test_broken_unrelated_provider_stays_visible_and_can_be_removed(tmp_path, source_case, qt_app):
    from packed_bed_ui.plugins_dialog import PluginsDialog
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files(source_case)
    previous = case.fingerprint()
    entry = project.plugins.add(variant(tmp_path))
    installed = project.root / 'plugins' / entry['id'] / 'current'
    (installed / 'manifest.yaml').write_text('invalid: [')
    assert case.fingerprint() == previous
    case.validate_for_run()
    dialog = PluginsDialog(project)
    dialog.tree.setCurrentItem(dialog.tree.topLevelItem(1))
    assert 'Unavailable' in dialog.tree.currentItem().text(0)
    assert 'invalid YAML' in dialog.details.toPlainText()
    assert dialog.actions['Remove'].isEnabled()
    dialog.remove()
    assert not project.plugins.entries
    dialog.close()


def test_copied_code_plugin_can_be_approved_without_disabling_and_failed_check_grants_nothing(tmp_path, source_case, qt_app, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    import packed_bed_ui.plugin_forms as forms
    from packed_bed_ui.plugins_dialog import PluginsDialog
    from packed_bed.plugins.storage import approved_hashes, code_approved
    monkeypatch.setattr(QMessageBox, 'question', lambda *_: QMessageBox.StandardButton.Yes)
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files(source_case)
    entry = project.plugins.add(EXAMPLES / 'correlation')
    project.metadata['plugins'][0]['enabled'] = True
    case.documents['chemistry']['gas_species'] = ['ArDemo']
    case.documents['chemistry']['species_definitions'] = {'ArDemo': 'example_correlation:ArDemo'}
    case.documents['program']['inlet_composition']['initial'] = {'ArDemo': 1.0}
    case.save()
    dialog = PluginsDialog(project)
    dialog.tree.setCurrentItem(dialog.tree.topLevelItem(1))
    assert 'Needs local approval' in dialog.tree.currentItem().text(0)
    dialog.toggle()
    assert dialog.checker.busy
    wait_for_check(dialog)
    assert project.plugins.entries[0]['enabled']
    assert code_approved(dialog.catalogue.paths[entry['id']])
    case.validate_for_run()
    dialog.close()
    broken = tmp_path / 'broken'
    shutil.copytree(EXAMPLES / 'correlation', broken)
    module = broken / 'correlation.py'
    module.write_text('raise RuntimeError("Check deliberately failed")\n' + module.read_text())
    with inspect_package(broken) as package:
        digest = package.digest
    check = forms.PluginCheck()
    check.run(broken)
    deadline = monotonic() + 15
    from PyQt6.QtTest import QTest
    while check.busy:
        assert monotonic() < deadline
        QTest.qWait(10)
    assert 'failed' in check.status.text()
    assert not code_approved(broken)
    assert digest not in approved_hashes()


@pytest.mark.parametrize('cancel', [False, True])
def test_inline_check_failure_or_cancellation_never_enables_or_approves(tmp_path, qt_app, monkeypatch, cancel):
    from PyQt6.QtWidgets import QMessageBox
    from packed_bed_ui.plugins_dialog import PluginsDialog
    from packed_bed.plugins.storage import approved_hashes, code_approved
    monkeypatch.setattr(QMessageBox, 'question', lambda *_: QMessageBox.StandardButton.Yes)
    project = Project.create(tmp_path / 'project')
    source = tmp_path / 'code'
    shutil.copytree(EXAMPLES / 'correlation', source)
    module = source / 'correlation.py'
    prefix = 'import time\ntime.sleep(30)\n' if cancel else 'raise RuntimeError("Deliberate check failure")\n'
    module.write_text(prefix + module.read_text())
    entry = project.plugins.add(source)
    dialog = PluginsDialog(project)
    dialog.tree.setCurrentItem(dialog.tree.topLevelItem(1))
    dialog.toggle()
    worker = dialog.checker
    if cancel:
        dialog.reject()
    wait_for_check(dialog)
    assert worker.cancelled is cancel
    assert not project.plugins.entries[0]['enabled']
    assert entry['hash'] not in approved_hashes()
    if not cancel:
        assert 'checks failed' in dialog.checker.status.text()
        assert 'Deliberate check failure' in dialog.checker.details.toPlainText()
    dialog.close()


def test_add_species_updates_one_plugin_and_preserves_existing_definitions(tmp_path, qt_app, monkeypatch):
    import packed_bed_ui.plugin_forms as forms
    from packed_bed_ui.plugins_dialog import PluginsDialog
    project = Project.create(tmp_path / 'project')
    from PyQt6.QtWidgets import QInputDialog
    monkeypatch.setattr(QInputDialog, 'getText', lambda *_: ('Lab properties', True))
    definitions = iter(('N2', 'H2O'))
    def finish_editor(editor):
        assert editor.editing and editor.adding
        definition = builtin_manifest().species[next(definitions)].model_dump(mode='json')
        definition['mw'] *= 1.1
        editor.form = forms.SpeciesForm(definition, editor)
        editor.save_plugin()
        wait_for_check(editor)
        assert editor.result() == forms.QDialog.DialogCode.Accepted, editor.checker.status.text()
        return editor.result()
    monkeypatch.setattr(forms.PluginEditor, 'exec', finish_editor)
    dialog = PluginsDialog(project)
    dialog.new()
    ident = dialog.provider()
    assert dialog.actions['Add species…'].isEnabled()
    dialog.add_species()
    original = project.plugins.entries[0]
    dialog.add_species()
    assert len(project.plugins.entries) == 1
    assert project.plugins.entries[0]['hash'] != original['hash']
    manifest = Project.open(project.root).plugins.catalogue().manifest(ident)
    assert set(manifest.species) == {'N2', 'H2O'}
    assert manifest.species['N2'].mw == pytest.approx(builtin_manifest().species['N2'].mw * 1.1)
    assert manifest.species['H2O'].chemical_key == 'H2O'
    assert dialog.selection() == ('species', ident + ':H2O')
    assert not (project.root / '.plugin-drafts').exists()
    assert [p.name for p in (project.root / 'plugins' / ident).iterdir()] == ['current']
    dialog.close()


def project_files(project):
    return {str(path.relative_to(project.root)): path.read_bytes() for path in project.root.rglob('*') if path.is_file()}


def test_real_browser_create_add_cancel_edit_and_noop_save(tmp_path, qt_app):
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QDialogButtonBox
    from packed_bed_ui.plugins_dialog import PluginsDialog
    project = Project.create(tmp_path / 'project')
    dialog = PluginsDialog(project)
    dialog.show()
    def name_plugin():
        prompt = qt_app.activeModalWidget()
        prompt.setTextValue('My properties')
        prompt.accept()
    QTimer.singleShot(0, name_plugin)
    dialog.actions['Create plugin…'].click()
    ident = dialog.provider()
    assert project.plugins.catalogue().manifest(ident).species == {}
    before = project_files(project)
    QTimer.singleShot(0, lambda: qt_app.activeModalWidget().close())
    dialog.actions['Add species…'].click()
    assert project_files(project) == before
    assert dialog.provider() == ident

    def fill_species():
        editor = qt_app.activeModalWidget()
        for key, value in {'name': 'Nitrogen', 'chemical_key': 'N2', 'mw': '.0280134'}.items():
            editor.form.fields[key].setText(value)
        editor.form.correlation_fields['h_form_ref'].setText('0')
        editor.form.correlation_fields['coefficients'].setText('29')
        editor.form.viscosity_fields['a0'].setText('0.00003')
        editor.buttons.button(QDialogButtonBox.StandardButton.Save).click()
    QTimer.singleShot(0, fill_species)
    dialog.actions['Add species…'].click()
    assert dialog.selection() == ('species', ident + ':N2')
    before = project_files(project)
    def cancel_changes():
        editor = qt_app.activeModalWidget()
        editor.form.fields['name'].setText('Discard me')
        QTest.keyClick(editor, Qt.Key.Key_Escape)
    for action in ('Add species…', 'Edit…', 'Create variant…'):
        QTimer.singleShot(0, cancel_changes)
        dialog.actions[action].click()
        assert project_files(project) == before
        assert dialog.selection() == ('species', ident + ':N2')
    QTimer.singleShot(0, lambda: qt_app.activeModalWidget().buttons.button(QDialogButtonBox.StandardButton.Save).click())
    dialog.actions['Edit…'].click()
    assert project_files(project) == before
    dialog.close()


def test_reaction_edit_preserves_other_parameters_and_disabled_state(tmp_path, qt_app):
    from packed_bed_ui.plugin_forms import PluginEditor
    from packed_bed.plugins.parameter_details import reaction_parameter_groups
    project = Project.create(tmp_path / 'project')
    family = builtin_manifest().mechanisms['reforming_xu_froment'].model_dump(mode='json')
    folder = tmp_path / 'rates'
    folder.mkdir()
    (folder / 'manifest.yaml').write_text(yaml.safe_dump({'id': 'rates', 'name': 'My rates', 'mechanisms': {'xf': family}}))
    project.plugins.add(folder)
    project.plugins.set_enabled('rates', False)
    cat = project.plugins.catalogue(include_disabled=True)
    groups = reaction_parameter_groups(cat.manifest('rates').mechanisms['xf'])
    reaction, keys = next(iter(groups.items()))
    other = next(key for key in family['parameters'] if key not in keys)
    data = cat.manifest('rates').model_dump(mode='json')
    data['mechanisms']['xf']['values'][other] = family['parameters'][other]['default'] * 1.2
    editor = PluginEditor(project, data, 'mechanisms', 'xf', source=cat.paths['rates'], editing=True, reaction_id=reaction)
    key = editor.form.keys[0]
    editor.form.parameters.item(0, 1).setText(str(family['parameters'][key]['default'] * 1.1))
    assert other not in editor.form.keys
    editor.save_plugin()
    wait_for_check(editor)
    assert editor.result() == editor.DialogCode.Accepted, editor.checker.status.text()
    assert not project.plugins.entries[0]['enabled']
    saved = project.plugins.catalogue(include_disabled=True).manifest('rates').mechanisms['xf'].values
    assert saved[key] == pytest.approx(family['parameters'][key]['default'] * 1.1)
    assert saved[other] == pytest.approx(family['parameters'][other]['default'] * 1.2)


def test_plugin_save_rolls_back_files_and_metadata_together(tmp_path, monkeypatch):
    project = Project.create(tmp_path / 'project')
    project.plugins.add(variant(tmp_path))
    before = project_files(project)
    original = __import__('packed_bed_ui.study_store', fromlist=['write_json']).write_json
    def fail(path, value):
        if path == project.root / 'project.json':
            raise OSError('Interrupted save')
        original(path, value)
    monkeypatch.setattr('packed_bed_ui.study_store.write_json', fail)
    with pytest.raises(OSError, match='Interrupted save'):
        project.plugins.add(variant(tmp_path, factor=1.2))
    assert project_files(project) == before
    assert Project.open(project.root).plugins.entries == project.plugins.entries


def test_importing_case_cannot_silently_replace_a_project_plugin(tmp_path, source_case):
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files(source_case)
    project.plugins.add(variant(tmp_path))
    project.plugins.replace_uses('species', 'builtin:N2', 'lab:N2')
    saved = success(project, case)
    project.plugins.add(variant(tmp_path, factor=1.2))
    before = project_files(project)
    with pytest.raises(ValueError, match='Register plugin'):
        project.add_case_from_files(saved / 'inputs/run.yaml')
    assert project_files(project) == before


@pytest.mark.parametrize('workers', [1, 2])
def test_real_spawned_runs_use_pinned_data_and_code_definitions(tmp_path, source_case, workers):
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files(source_case)
    data_entry = project.plugins.add(variant(tmp_path, species='H2O', factor=1.01))
    code_entry = project.plugins.add(EXAMPLES / 'kinetics')
    approve_hash(code_hash(project.plugins.catalogue(include_disabled=True).paths[code_entry['id']]))
    project.plugins.set_enabled(code_entry['id'], True)
    gases = ['CO', 'CO2', 'H2', 'H2O', 'other_water']
    case.documents['chemistry'] = {
        'gas_species': gases, 'species_definitions': {'other_water': 'lab:H2O'},
        'reaction_families': ['shift'], 'reaction_ids': ['shift/shift'],
        'mechanisms': {'shift': {'definition': 'example_kinetics:shift', 'bindings': {'H2O': 'other_water'}}}}
    case.documents['program']['inlet_composition']['initial'] = {key: .2 for key in gases}
    case.documents['program']['inlet_temperature']['initial'] = 900.0
    project.duplicate_case(case, 'Second')
    job = project.prepare_execution(project.cases, max_workers=workers)
    # Updating active data after preparation must not affect the scheduled execution.
    project.plugins.add(variant(tmp_path, species='H2O', factor=1.02))
    result = subprocess.run([sys.executable, '-m', 'packed_bed_ui', '--project-worker', str(job)],
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).parents[1])}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr + job.read_text()
    for item in project.cases:
        manifest = read_json(item.run_folder / 'output/manifest.json')
        definitions = manifest['configuration']['definitions']
        assert definitions['lock']['plugins']['lab'] == data_entry['hash']
        assert definitions['mechanisms']['shift']['bindings']['H2O'] == 'other_water'
        assert 'Lab properties' in definitions['species']['other_water']['label']
        assert item.state()['stale']


def reacting_project(tmp_path, source_case):
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files(source_case)
    project.plugins.add(variant(tmp_path, species='H2O', factor=1.0))
    entry = project.plugins.add(EXAMPLES / 'kinetics')
    approve_hash(code_hash(project.plugins.catalogue(include_disabled=True).paths[entry['id']]))
    project.plugins.set_enabled(entry['id'], True)
    gases = ['CO', 'CO2', 'H2', 'lab.water']
    case.documents['chemistry'] = {
        'gas_species': gases, 'species_definitions': {'lab.water': 'lab:H2O'},
        'reaction_families': ['shift'], 'reaction_ids': ['shift/shift'],
        'mechanisms': {'shift': {'definition': 'example_kinetics:shift', 'bindings': {'H2O': 'lab.water'}}}}
    case.documents['program']['inlet_composition']['initial'] = {key: .25 for key in gases}
    case.documents['program']['inlet_temperature']['initial'] = 900.0
    case.save()
    case.documents['run']['outputs']['requested_reports'].append('reaction_rate')
    case.save()
    return project, case


def test_plugins_match_across_daetools_and_compiled_backends(tmp_path, source_case):
    pytest.importorskip('sksundae')
    if shutil.which('g++') is None:
        pytest.skip('Compiled solver requires a C++ compiler.')
    import numpy as np
    from packed_bed.simulation import run_case
    from packed_bed.reports import load_dataset
    project, case = reacting_project(tmp_path, source_case)
    runtime = load_case(case.root / 'inputs/run.yaml')
    results = []
    for backend in ('daetools', 'compiled'):
        outputs = runtime.run.outputs.model_copy(update={'directory': str(tmp_path / backend)})
        solver = runtime.run.solver.model_copy(update={'backend': backend, 'relative_tolerance': 1e-7})
        resolved = replace(runtime, run=runtime.run.model_copy(update={'outputs': outputs, 'solver': solver}))
        result = run_case(resolved)
        results.append(load_dataset(result.results_path))
    try:
        for key in results[0].data_vars:
            # Pressure drop subtracts two ~100 kPa values. Compare it in Pa,
            # rather than using relative error against the tiny difference.
            absolute = 1e-4 if results[0][key].attrs.get('units') == 'Pa' else 1e-9
            np.testing.assert_allclose(results[0][key], results[1][key], rtol=1e-7, atol=absolute)
        assert 'lab.water' in results[1].gas_species.values
        assert json.loads(results[1].attrs['definition_selection'])['lock']['plugins']['lab']
    finally:
        for dataset in results:
            dataset.close()


def test_cli_batch_workers_reconstruct_portable_pinned_plugins(tmp_path, source_case):
    project, case = reacting_project(tmp_path, source_case)
    path = project.root / 'batch.yaml'
    path.write_text(yaml.safe_dump({'base_case': f'cases/{case.id}/inputs/run.yaml',
        'output_directory': 'batch-output', 'workers': 2,
        'axes': [{'id': 'grid', 'values': [{'id': 'three', 'patch': {'run': {'model': {'axial_cells': 3}}}},
                                          {'id': 'four', 'patch': {'run': {'model': {'axial_cells': 4}}}}]}]}))
    completed = subprocess.run([sys.executable, '-m', 'packed_bed', 'batch', str(path)],
                               cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for generated in (project.root / 'batch-output/cases').iterdir():
        descriptor = read_json(generated / 'definitions.json')
        assert not Path(descriptor['root']).is_absolute()
        assert load_case(generated / 'run.yaml').definitions.selection.to_dict()['lock'] == descriptor['lock']
