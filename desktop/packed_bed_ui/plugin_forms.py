"""Small data forms; Python implementations are authored outside the application."""
from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import yaml
from pydantic import ValidationError
from PyQt6.QtCore import QProcess, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QScrollArea, QHeaderView,
    QVBoxLayout, QWidget, QStackedWidget)

from packed_bed.plugins.schema import Manifest
from packed_bed.plugins.storage import has_code, inspect_package, approve_hash, code_hash, code_approved, package_files, package_hash
from .editor_widgets import action_button, number, display, table, cell, choices, dialog_buttons
from .project import write_text


class PluginCheck(QWidget):
    """One inline, cancellable check and approval flow for every plugin action."""
    busyChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.busy = False
        self.package = None
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.finished.connect(self.finish)
        self.process.errorOccurred.connect(self.process_error)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.status = QLabel()
        self.status.setWordWrap(True)
        row.addWidget(self.status, 1)
        self.stop = action_button('Cancel check', self.cancel)
        self.stop.hide()
        row.addWidget(self.stop)
        layout.addLayout(row)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(130)
        self.details.hide()
        layout.addWidget(self.details)

    def message(self, text, details=''):
        self.status.setText(text)
        self.details.setPlainText(details)
        self.details.setVisible(bool(details))

    def run(self, source, on_success=None):
        if self.busy:
            return
        self.inspection = ExitStack()
        package = self.inspection.enter_context(inspect_package(source))
        try:
            if has_code(package.manifest) and not code_approved(package.folder):
                answer = QMessageBox.question(self, 'Allow plugin code',
                    f'{package.manifest.name} contains Python code that will run with your permissions.\n'
                    f'Source: {package.manifest.source or "Not specified"}\n\nAllow this code?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                if answer != QMessageBox.StandardButton.Yes:
                    self.inspection.close()
                    return
        except Exception:
            self.inspection.close()
            raise
        self.package, self.on_success = package, on_success
        self.cancelled, self.busy = False, True
        self.message(f'Checking {package.manifest.name}…')
        self.stop.show()
        self.busyChanged.emit(True)
        args = ['--check-plugin', str(package.folder)]
        self.process.start(sys.executable, args if getattr(sys, 'frozen', False) else ['-m', 'packed_bed_ui', *args])

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.finish(-1, QProcess.ExitStatus.CrashExit)

    def finish(self, code, status):
        if not self.busy:
            return
        output = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace').strip()
        package = self.package
        try:
            if self.cancelled:
                self.message('Check cancelled.')
            elif code or status != QProcess.ExitStatus.NormalExit:
                self.message('Plugin checks failed.', output or self.process.errorString())
            else:
                if package_hash(package.folder) != package.digest:
                    raise ValueError('Plugin changed during checking. Check the new contents again.')
                if has_code(package.manifest):
                    approve_hash(code_hash(package.folder))
                if self.on_success:
                    self.on_success(package)
                self.message('Checks passed.')
        except (ValueError, OSError, KeyError) as exc:
            self.message(str(exc))
        finally:
            self.inspection.close()
            self.package, self.busy = None, False
            self.stop.hide()
            self.busyChanged.emit(False)

    def cancel(self):
        if self.busy:
            self.cancelled = True
            self.process.kill()
            self.process.waitForFinished(1000)


class ParameterForm(QWidget):
    def __init__(self, definition, parent=None, *, reaction_id=None):
        super().__init__(parent)
        self.definition = deepcopy(definition)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name, self.source = QLineEdit(definition.get('name', '')), QLineEdit(definition.get('source', ''))
        form.addRow('Family name', self.name)
        form.addRow('Scientific source', self.source)
        layout.addLayout(form)
        if reaction_id:
            selected = next(r for r in definition['reactions'] if r['id'] == reaction_id)
            layout.addWidget(QLabel(selected['name']))
        self.parameters = table(['Parameter / group', 'Value', 'Unit / basis'])
        header = self.parameters.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column, width in ((1, 160), (2, 180)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            header.resizeSection(column, width)
        from packed_bed.plugins.schema import Mechanism
        from packed_bed.plugins.parameter_details import reaction_parameter_groups
        groups = reaction_parameter_groups(Mechanism.model_validate(definition))
        self.keys = tuple(groups[reaction_id]) if reaction_id and groups is not None else tuple(definition['parameters'])
        self.parameters.setRowCount(len(self.keys))
        for row, key in enumerate(self.keys):
            spec = definition['parameters'][key]
            cell(self.parameters, row, 0, key, editable=False, tooltip=spec['description'])
            cell(self.parameters, row, 1, definition.get('values', {}).get(key, spec['default']))
            cell(self.parameters, row, 2, spec['unit'], editable=False)
        self.parameters.resizeRowsToContents()
        layout.addWidget(self.parameters)
        layout.addWidget(action_button('Reset shown values', self.reset))
        layout.addWidget(QLabel('Shared parameters also affect the other reactions that use them.'))

    def reset(self):
        for row, key in enumerate(self.keys):
            spec = self.definition['parameters'][key]
            self.parameters.item(row, 1).setText(display(spec['default']))

    def value(self):
        return {**self.definition, 'name': self.name.text(), 'source': self.source.text(),
                'values': {**self.definition.get('values', {}), **{key: number(self.parameters.item(row, 1).text())
                           for row, key in enumerate(self.keys)}}}


def text_fields(form, rows, values):
    fields = {}
    for key, label, default in rows:
        value = values.get(key, default)
        field = QLineEdit(', '.join(map(display, value)) if isinstance(value, (tuple, list)) else display(value))
        field.setObjectName(key)
        form.addRow(label, field)
        fields[key] = field
    return fields


def numeric_values(fields, *, polynomial=False):
    return {key: [number(v.strip()) for v in field.text().split(',')] if polynomial and key == 'coefficients' else number(field.text())
            for key, field in fields.items()}


class SpeciesForm(QWidget):
    def __init__(self, definition, parent=None, *, correlations=None):
        super().__init__(parent)
        self.definition = deepcopy(definition)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.fields = text_fields(form, (
            ('name', 'Name', ''), ('chemical_key', 'Chemical key', ''), ('mw', 'Molecular weight (kg/mol)', ''),
            ('source', 'Scientific source', ''), ('notes', 'Notes', '')), definition)
        self.phase = choices(['gas', 'solid'])
        self.phase.setCurrentText(definition.get('phase', 'gas'))
        form.addRow('Phase', self.phase)
        limits = dict(zip(('low', 'high'), definition.get('temperature_range', [298.15, 1200])))
        self.limits = text_fields(form, (('low', 'Minimum T (K)', 298.15), ('high', 'Maximum T (K)', 1200)), limits)
        self.model = choices([('Polynomial Cp', 'builtin:polynomial'), ('Shomate Cp', 'builtin:shomate')])
        correlations = correlations or {}
        for key, spec in correlations.items():
            if spec['kind'] == 'enthalpy':
                self.model.addItem(spec.get('name') or key, key)
        enthalpy = definition.get('enthalpy', {'model': 'builtin:polynomial', 'parameters': {}})
        if self.model.findData(enthalpy['model']) < 0:
            self.model.addItem('External: ' + enthalpy['model'], enthalpy['model'])
        form.addRow('Heat capacity / enthalpy', self.model)
        layout.addLayout(form)
        self.correlations = QStackedWidget()
        self.correlation_forms = {}
        for i in range(self.model.count()):
            model = self.model.itemData(i)
            page = QWidget()
            fields = QFormLayout(page)
            rows = [('t_ref', 'Tref (K)', 298.15), ('h_form_ref', 'H at Tref (J/mol)', '')]
            if model == 'builtin:polynomial':
                fields.addRow(QLabel('Cp = Σ ai·(T − Tref)^i; ai in J/(mol·K^(i+1)).\nH integrates Cp from Tref.'))
                rows.append(('coefficients', 'Cp coefficients a0, a1, …', ''))
            elif model == 'builtin:shomate':
                fields.addRow(QLabel('τ = T/1000 K; Cp = a0 + a1τ + a2τ² + a3τ³ + a4/τ².\nAll ai in J/(mol·K); H integrates Cp from Tref.'))
                rows.extend((f'a{i}', f'a{i} (J/(mol·K))', '') for i in range(5))
            else:
                specs = correlations.get(model, {}).get('parameters', {})
                rows = [(key, f"{key} ({spec['unit']})", spec['default']) for key, spec in specs.items()]
                if model not in correlations:
                    fields.addRow(QLabel('Edit this correlation in its Python package.'))
            self.correlation_forms[model] = text_fields(fields, rows, enthalpy['parameters'] if model == enthalpy['model'] else {})
            self.correlations.addWidget(page)
        self.model.currentIndexChanged.connect(self.correlations.setCurrentIndex)
        self.model.setCurrentIndex(self.model.findData(enthalpy['model']))
        layout.addWidget(self.correlations)
        self.viscosity = QWidget()
        viscosity_form = QFormLayout(self.viscosity)
        viscosity = definition.get('viscosity') or {'model': 'builtin:quadratic_viscosity', 'parameters': {}}
        self.custom_viscosity = viscosity['model'] != 'builtin:quadratic_viscosity'
        rows = [] if self.custom_viscosity else [('t_ref', 'Tref (K)', 1000), ('a0', 'a0 (Pa·s)', ''),
                                               ('a1', 'a1 (Pa·s/K)', 0), ('a2', 'a2 (Pa·s/K²)', 0)]
        viscosity_form.addRow(QLabel('Edit viscosity in its Python package.' if self.custom_viscosity else
                                    'Viscosity = a0 + a1·(T − Tref) + a2·(T − Tref)²'))
        self.viscosity_fields = text_fields(viscosity_form, rows, viscosity['parameters'])
        layout.addWidget(self.viscosity)
        self.phase.currentTextChanged.connect(lambda value: self.viscosity.setVisible(value == 'gas'))
        self.viscosity.setVisible(self.phase.currentText() == 'gas')

    @property
    def correlation_fields(self):
        return self.correlation_forms[self.model.currentData()]

    def value(self):
        fields = {key: field.text() for key, field in self.fields.items()}
        fields['mw'] = number(fields['mw'])
        model = self.model.currentData()
        parameters = dict(self.definition['enthalpy']['parameters']) if model == self.definition.get('enthalpy', {}).get('model') else {}
        parameters.update(numeric_values(self.correlation_fields, polynomial=model == 'builtin:polynomial'))
        enthalpy = {'model': model, 'parameters': parameters}
        viscosity = None
        if self.phase.currentText() == 'gas':
            viscosity = self.definition['viscosity'] if self.custom_viscosity else {
                'model': 'builtin:quadratic_viscosity', 'parameters': numeric_values(self.viscosity_fields)}
        return {**self.definition, **fields, 'phase': self.phase.currentText(),
                'temperature_range': list(numeric_values(self.limits).values()), 'enthalpy': enthalpy, 'viscosity': viscosity}


def available_key(name, existing):
    """Generate a stable local ID without making users fill in another name."""
    import re
    base = re.sub(r'[^A-Za-z0-9_.-]', '_', name.strip()) or 'species'
    if not base[0].isalpha():
        base = 'species_' + base
    key, suffix = base, 2
    while key in existing:
        key, suffix = f'{base}_{suffix}', suffix + 1
    return key


class PluginEditor(QDialog):
    """Edit in memory; only a successful Save writes to the project."""
    def __init__(self, project, manifest, kind, key, *, source=None, editing=False, adding=False,
                 reaction_id=None, parent=None):
        super().__init__(parent)
        self.project, self.data, self.kind, self.key = project, deepcopy(manifest), kind, key
        self.editing, self.adding, self.reaction_id = editing, adding, reaction_id
        self.source = Path(source) if source else None
        self.temporary = None
        self.setWindowTitle(f'Add species — {manifest["name"]}' if adding else 'Edit definition' if editing else 'Create variant')
        self.resize(1000 if kind == 'mechanisms' else 800, 720)
        layout = QVBoxLayout(self)
        self.fields = QWidget()
        fields = QVBoxLayout(self.fields)
        fields.setContentsMargins(0, 0, 0, 0)
        self.name = QLineEdit(self.data['name'])
        if not editing:
            identity = QFormLayout()
            identity.addRow('New plugin name', self.name)
            fields.addLayout(identity)
        else:
            fields.addWidget(QLabel('Plugin: ' + self.data['name']))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.form = (SpeciesForm(self.data[kind][key], correlations=self.data.get('correlations')) if kind == 'species'
                     else ParameterForm(self.data[kind][key], reaction_id=reaction_id))
        scroll.setWidget(self.form)
        fields.addWidget(scroll)
        layout.addWidget(self.fields, 1)
        self.checker = PluginCheck(self)
        layout.addWidget(self.checker)
        self.buttons = dialog_buttons(self, accept=self.save_plugin)
        layout.addWidget(self.buttons)
        self.checker.busyChanged.connect(self.checking)
        self.initial = self.value()

    def value(self):
        data = deepcopy(self.data)
        data['name'] = self.name.text().strip()
        definition = self.form.value()
        data[self.kind].pop(self.key, None)
        key = available_key(definition.get('chemical_key', ''), data[self.kind]) if self.adding else self.key
        data[self.kind][key] = definition
        self.saved_selection = (self.kind, data['id'] + ':' + key, *((self.reaction_id,) if self.reaction_id else ()))
        return data

    def write_package(self, folder, data):
        folder.mkdir(parents=True, exist_ok=True)
        if self.source and self.source != folder:
            for name, path in package_files(self.source):
                target = folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        write_text(folder / 'manifest.yaml', yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    def save_plugin(self):
        try:
            data = self.value()
            if self.editing and not self.adding and data == self.initial:
                self.accept()
                return
            Manifest.model_validate(data)
            self.temporary = TemporaryDirectory(prefix='multisolid-author-')
            folder = Path(self.temporary.name)
            self.write_package(folder, data)
            def save(package):
                entry = self.project.plugins.add(package.folder)
                if not self.editing and not entry['enabled']:
                    self.project.plugins.set_enabled(entry['id'], True)
                self.accept()
            self.checker.run(folder, save)
        except ValidationError as exc:
            self.checker.message('Please correct these fields before saving.',
                                 '\n'.join(f'{" → ".join(map(str, error["loc"]))}: {error["msg"]}' for error in exc.errors()))
        except (ValueError, OSError, TypeError) as exc:
            self.checker.message(str(exc))
        if not self.checker.busy:
            self.checking(False)

    def checking(self, busy):
        self.fields.setEnabled(not busy)
        for button in self.buttons.buttons():
            if self.buttons.buttonRole(button) != QDialogButtonBox.ButtonRole.RejectRole:
                button.setEnabled(not busy)
        if not busy and self.temporary:
            self.temporary.cleanup()
            self.temporary = None

    def reject(self):
        self.checker.cancel()
        super().reject()
