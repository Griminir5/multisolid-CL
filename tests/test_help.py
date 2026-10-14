"""Help must work from local files and route F1 without touching project inputs."""

import json
import os
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/multisolid-help-test-mpl')

import pytest
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from build_help import DESTINATION, PageParser, build_pages
from packed_bed_ui import help as help_ui


def test_bundled_help_is_current_with_valid_local_links():
    output = build_pages()
    for name, content in output.items():
        assert (DESTINATION / name).read_bytes() == content.encode('utf-8'), name
    pages = {name: PageParser(content) for name, content in output.items() if name.endswith('.html')}
    for name, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = url.path or name
            assert target in pages, (name, link)
            if url.fragment:
                assert unquote(url.fragment) in pages[target].ids, (name, link)
    assert '<math ' in output['equations.html']
    assert '$$' not in output['equations.html']
    assert r'bed\_only' not in output['equations.html']
    search = json.loads(output['search-data.js'].removeprefix('window.helpSearch = ').removesuffix(';\n'))
    for topic in ('landing', 'project', 'general', 'chemistry', 'bed', 'program', 'report', 'results', 'studies'):
        assert any(entry['url'].startswith(topic + '.html#') for entry in search)


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_f1_routes_every_screen_and_preserves_inputs(app, tmp_path, monkeypatch):
    from packed_bed_ui.project import Project
    from packed_bed_ui.window import MainWindow

    opened = []
    monkeypatch.setattr(help_ui.QDesktopServices, 'openUrl', lambda url: opened.append(url) or True)
    project = Project.create(tmp_path / 'Project with spaces α', 'Help check')
    case = project.add_case('Draft case')
    window = MainWindow(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))
    window.show()
    window.activateWindow()
    QTest.qWait(40)

    def press_f1(topic):
        before = len(opened)
        QTest.keyClick(window, Qt.Key.Key_F1)
        app.processEvents()
        assert len(opened) == before + 1
        assert opened[-1] == help_ui.help_url(topic)

    try:
        press_f1('landing')
        window.open_project(project.root)
        press_f1('project')
        window._show_case(window.project.cases[0])
        for topic in ('general', 'chemistry', 'bed', 'program', 'report'):
            window.editor.tabs.setCurrentWidget(getattr(window.editor, topic).tab_container)
            app.processEvents()
            press_f1(topic)
        before = {p: p.read_bytes() for p in project.root.rglob('*') if p.is_file() and '.draft' not in str(p)}
        press_f1('report')
        assert {p: p.read_bytes() for p in before} == before
        window.results.set_project(window.project)
        window.pages.setCurrentWidget(window.results)
        press_f1('results')
        window.pages.setCurrentWidget(window.study_editor)
        press_f1('studies')
    finally:
        window.close()


def test_help_handles_missing_files_and_about_copy(app, tmp_path, monkeypatch):
    errors = []
    monkeypatch.setattr(help_ui, 'HELP_ROOT', tmp_path)
    monkeypatch.setattr(help_ui.QMessageBox, 'warning', lambda *args: errors.append(args[2]))
    help_ui.open_help(None, 'missing')
    assert len(errors) == 1 and 'missing' in errors[0]
    (tmp_path / 'general.html').write_text('<h1>General</h1>')
    failures = []
    monkeypatch.setattr(help_ui.QDesktopServices, 'openUrl', lambda url: False)
    monkeypatch.setattr(help_ui.QMessageBox, 'information', lambda *args: failures.append(args[2]))
    help_ui.open_help(None, 'general')
    assert str(tmp_path / 'general.html') in failures[0]
    dialog = help_ui.AboutDialog()
    from PyQt6.QtWidgets import QPushButton
    button = next(b for b in dialog.findChildren(QPushButton) if b.text() == 'Copy diagnostic information')
    button.click()
    assert app.clipboard().text() == dialog.details.toPlainText()
    assert 'Python:' in app.clipboard().text()
    dialog.close()
