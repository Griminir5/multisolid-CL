"""Render native desktop review captures without running a simulation.

Run with QT_QPA_PLATFORM=offscreen; use QT_SCALE_FACTOR=2 for the scaling pass.
Outputs and temporary projects are kept outside the source tree by default.
"""
from argparse import ArgumentParser
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from PyQt6.QtCore import QSettings
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from packed_bed_ui.project import Project
from packed_bed_ui.theme import apply_theme
from packed_bed_ui.window import MainWindow


def main():
    parser = ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('/tmp/multisolid-design-review'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    theme = apply_theme(app)
    root = Path(__file__).resolve().parents[2]
    audit = []
    with TemporaryDirectory(prefix='multisolid-design-') as temp:
        project = Project.create(Path(temp) / 'project', 'Packed-bed study')
        case = project.add_case_from_files(root / 'packed_bed/examples/default_case/run_feed_stream.yaml', 'Cyclic packed bed')
        settings = QSettings(str(Path(temp) / 'settings.ini'), QSettings.Format.IniFormat)
        for dark in (False, True):
            theme.apply(dark)
            window = MainWindow(settings=settings)
            window.hero.animation.set_paused(True)
            for width, height in ((1280, 800), (1024, 768), (1920, 1080)):
                window.resize(width, height)
                window.show()
                window.pages.setCurrentWidget(window.welcome)
                for name, tab in [('welcome', None), ('general', 0), ('chemistry', 1), ('bed', 2), ('program', 3), ('report', 4)]:
                    if tab is not None:
                        if window.project is None:
                            window._set_project(project)
                        window._show_case(case)
                        window.editor.tabs.setCurrentIndex(tab)
                    QTest.qWait(300)
                    actual = [window.width(), window.height()]
                    assert actual == [width, height], (name, actual, [width, height])
                    path = args.output / f'{"dark" if dark else "light"}-{width}x{height}-{name}.png'
                    assert window.grab().save(str(path))
                    row = dict(dark=dark, requested=[width, height], actual=actual, page=name, image=path.name)
                    if name == 'bed':
                        row['bed_split'] = window.editor.bed.settings_split.sizes()
                    if name == 'welcome':
                        buttons = [*window.hero.actions, window.hero.pause_button]
                        row['action_baselines'] = [b.mapTo(window, b.rect().bottomLeft()).y() for b in buttons]
                        assert len(set(row['action_baselines'])) == 1
                    audit.append(row)
                before = deepcopy(case.documents)
                theme.apply(not dark)
                theme.apply(dark)
                assert case.documents == before
                window.pages.setCurrentWidget(window.welcome)
            window.close()
            window.deleteLater()
            app.processEvents()
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(f'{len(audit)} captures: {args.output}')


if __name__ == '__main__':
    main()
