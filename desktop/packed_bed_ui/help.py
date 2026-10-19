"""Offline documentation entry points and application information."""

from importlib import metadata
from pathlib import Path
import platform
import sys

from PyQt6.QtCore import PYQT_VERSION_STR, QUrl, qVersion
from PyQt6.QtGui import QDesktopServices, QKeySequence
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLabel, QMessageBox, QPlainTextEdit, QVBoxLayout

from packed_bed.release import application_identity


HELP_ROOT = Path(__file__).resolve().parent / 'assets/help'
PROJECT_URL = 'https://github.com/Griminir5/multisolid-CL'


def help_url(topic='index', anchor=''):
    path = HELP_ROOT / f'{topic}.html'
    if path.parent != HELP_ROOT or not path.is_file():
        raise FileNotFoundError(f'The bundled help page is missing: {path}')
    url = QUrl.fromLocalFile(str(path))
    if anchor:
        url.setFragment(anchor)
    return url


def open_help(parent, topic='index', anchor=''):
    try:
        url = help_url(topic, anchor)
    except FileNotFoundError as exc:
        QMessageBox.warning(parent, 'Help unavailable', str(exc))
        return
    if not QDesktopServices.openUrl(url):
        QMessageBox.information(parent, 'Open help', 'Open this help page in your browser:\n' + url.toLocalFile())


def screen_topic(window):
    page = window.pages.currentWidget()
    if page is window.case_page:
        current = window.editor.tabs.currentWidget()
        for name in ('general', 'chemistry', 'bed', 'program', 'report'):
            if current is getattr(window.editor, name).tab_container:
                return name
    return {window.welcome: 'landing', window.home: 'project', window.results: 'results',
            window.study_editor: 'studies'}.get(page, 'index')


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return 'Version unavailable'


def application_version():
    return application_identity()['ui_version'] or 'Unknown'


def diagnostic_information():
    """An inspectable version summary, without project data or machine paths."""
    identity = application_identity()
    dirty = {True: 'yes', False: 'no', None: 'unknown'}[identity['source_dirty']]
    return '\n'.join((
        f'MultiSolid: {identity["ui_version"] or "Unknown"}',
        f'Engine: {identity["engine_version"] or "Unknown"}',
        f'Source commit: {identity["source_commit"] or "Unknown"}',
        f'Source modified: {dirty}',
        f'Build identifier: {identity["build_id"] or "Unavailable"}',
        f'Operating system: {platform.system()} {platform.release()} ({platform.machine()})',
        f'Python: {platform.python_version()}',
        f'Qt: {qVersion()}',
        f'PyQt: {PYQT_VERSION_STR}',
        f'DAETools package: {package_version("daetools")}',
        f'Installation: {"Packaged desktop" if getattr(sys, "frozen", False) else "Python environment"}',
    ))


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('About MultiSolid')
        self.resize(610, 390)
        layout = QVBoxLayout(self)
        title = QLabel(f'MultiSolid {application_version()}')
        title.setProperty('role', 'section')
        layout.addWidget(title)
        description = QLabel('Packed-bed reactor simulations, parameter studies, and result analysis.')
        description.setWordWrap(True)
        layout.addWidget(description)
        link = QLabel(f'<a href="{PROJECT_URL}">MultiSolid project on GitHub</a>')
        link.setOpenExternalLinks(True)
        layout.addWidget(link)
        layout.addWidget(QLabel('Diagnostic information'))
        self.details = QPlainTextEdit(diagnostic_information())
        self.details.setReadOnly(True)
        self.details.setAccessibleName('Diagnostic information to copy')
        layout.addWidget(self.details)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy = buttons.addButton('Copy diagnostic information', QDialogButtonBox.ButtonRole.ActionRole)
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.details.toPlainText()))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def show_about(parent):
    dialog = AboutDialog(parent)
    dialog.exec()
    dialog.deleteLater()


def install_help_menu(window):
    menu = window.masthead.menus.addMenu('Help')

    def entry(label, topic, anchor=''):
        return menu.addAction(label, lambda: open_help(window, topic, anchor))

    entry('Getting started', 'getting-started')
    entry('User guide', 'gui-guide')
    window.screen_help_action = menu.addAction('Help for this screen', lambda: open_help(window, screen_topic(window)))
    window.screen_help_action.setShortcut(QKeySequence('F1'))
    window.addAction(window.screen_help_action)
    entry('Search help…', 'index', 'search')
    reference = menu.addMenu('Reference guides')
    for label, topic in (('Model equations', 'equations'), ('Plugin guide', 'plugins'),
                         ('Plugin authoring examples', 'plugin-examples'), ('CLI guide', 'cli-guide'),
                         ('Kinetics sources', 'kinetics')):
        reference.addAction(label, lambda checked=False, topic=topic: open_help(window, topic))
    entry('Troubleshooting', 'gui-guide', 'troubleshooting')
    menu.addSeparator()
    menu.addAction('About MultiSolid', lambda: show_about(window))
    menu.addAction('Licences and notices', window._show_licenses)
    return menu
