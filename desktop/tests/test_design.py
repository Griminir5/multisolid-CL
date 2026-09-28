"""Presentation behavior that can affect authoring, navigation or data integrity."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QCoreApplication, QEvent, QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QLineEdit, QStyleOptionViewItem

from packed_bed_ui.choice import BinaryChoice
from packed_bed_ui.editor import CaseEditor, InputEditor
from packed_bed_ui.editor_widgets import select_value
from packed_bed_ui.project import Project
from packed_bed_ui.theme import apply_theme, NUMERIC_FAMILY, LIGHT, DARK
from packed_bed_ui.welcome_animation import CycleEmblem


@pytest.fixture
def design(qt_app):
    theme = apply_theme(qt_app)
    theme.apply(False)
    yield theme
    for widget in qt_app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    theme.follow_system()


def test_binary_choice_keyboard_unknown_and_signal_blocking(qt_app, design):
    choice = BinaryChoice([('First', 'a'), ('Second', 'b')])
    changes = []
    choice.currentIndexChanged.connect(changes.append)
    choice.show()
    choice.buttons[0].setFocus()
    QTest.keyClick(choice.buttons[0], Qt.Key.Key_Right)
    assert choice.currentData() == 'b'
    assert changes == [1]
    choice.blockSignals(True)
    select_value(choice, 'imported-unknown')
    choice.blockSignals(False)
    assert changes == [1]
    assert choice.currentData() == 'imported-unknown'
    assert choice.unknown.isVisible()
    assert not any(button.isChecked() for button in choice.buttons)
    QTest.mouseClick(choice.buttons[0], Qt.MouseButton.LeftButton)
    assert choice.currentData() == 'a'
    assert not choice.unknown.isVisible()


def test_numeric_drafts_and_read_only_selectors_round_trip(qt_app, design, tmp_path, source_case):
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files(source_case)
    case.documents['solids']['initial_profile']['zones'][0]['e_b'] = '1e-'
    case.documents['solids']['initial_profile']['basis'] = 'unknown'
    case.save()
    editor = InputEditor()
    editor.resize(1280, 700)
    editor.set_case(case, read_only=True)
    before = deepcopy(case.documents)
    for choice in editor.findChildren(BinaryChoice):
        assert not choice.isEnabled()
    editor.set_case(case)
    assert editor.bed.basis.currentData() == 'unknown'
    assert all(button.isEnabled() for button in editor.general.backend.buttons)
    assert case.documents == before
    editor.tabs.setCurrentWidget(editor.bed)
    editor.show()
    qt_app.processEvents()
    item = editor.bed.zones.item(0, 2)
    assert item.text() == '1e-'
    assert item.font().family() == NUMERIC_FAMILY
    index = editor.bed.zones.indexFromItem(item)
    delegate = editor.bed.zones.itemDelegate()
    control = delegate.createEditor(editor.bed.zones, QStyleOptionViewItem(), index)
    assert control.font().family() == NUMERIC_FAMILY
    control.deleteLater()
    assert editor.fields['model', 'bed_radius_m'].font().family() == NUMERIC_FAMILY


def test_animation_variation_continuity_and_lifecycle(qt_app, design, tmp_path):
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    art = CycleEmblem(settings=settings)
    for bed in range(3):
        for seconds in (0, 1, 12, 23.99, 24, 112, 719.9):
            assert art.particle_states(bed, seconds) == art.particle_states(bed, seconds + 720)
        assert art.particle_states(bed, 12) != art.particle_states(bed, 60)
        assert art.particle_states(bed, 720 - 1e-7) == art.particle_states(bed, 720)
    art.resize(600, 380)
    art.show()
    QTest.qWait(80)
    assert art.seconds > .04 and art.timer.isActive()
    art.set_paused(True)
    seconds, frame = art.seconds, art.grab().toImage()
    QTest.qWait(70)
    assert art.seconds == seconds and art.grab().toImage() == frame
    second = CycleEmblem(settings=settings)
    assert second.paused
    art.set_paused(False)
    art.hide()
    stopped = art.seconds
    QTest.qWait(70)
    assert art.seconds == stopped and not art.timer.isActive()
    art.show()
    QTest.qWait(70)
    assert stopped < art.seconds < stopped + .2
    art.close()
    second.close()


def test_welcome_corners_archive_action_and_navigation(qt_app, design, tmp_path, monkeypatch):
    from packed_bed_ui.window import MainWindow
    called = []
    monkeypatch.setattr(MainWindow, '_import_archive', lambda self: called.append(True))
    window = MainWindow()
    window.show()
    qt_app.processEvents()
    buttons = [*window.hero.actions, window.hero.pause_button]
    assert len({b.mapTo(window, b.rect().bottomLeft()).y() for b in buttons}) == 1
    assert window.hero.art_pane.isAncestorOf(window.hero.pause_button)
    assert window.hero.pause_button.palette().button().color().name() == '#242720'
    assert window.masthead.isVisible()
    QTest.mouseClick(window.hero.actions[2], Qt.MouseButton.LeftButton)
    assert called == [True]
    project = Project.create(tmp_path / 'project')
    window._set_project(project)
    assert not window.masthead.isVisible()
    assert not window.hero.animation.timer.isActive()
    window._close_project()
    assert window.masthead.isVisible()
    window.close()


@pytest.mark.parametrize('size', [(1280, 800), (1024, 768), (1920, 1080)])
def test_layouts_fit_and_theme_keeps_documents_and_preview_arrays(qt_app, design, tmp_path, size):
    from packed_bed_ui.window import MainWindow
    window = MainWindow()
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files('packed_bed/examples/default_case/run_feed_stream.yaml')
    window._set_project(project)
    window._show_case(case)
    window.resize(*size)
    window.show()
    before = deepcopy(case.documents)
    original = [line.get_ydata().copy() for axis in window.editor.figures[0].axes for line in axis.lines]
    for dark in (False, True):
        design.apply(dark)
        for tab in (0, 1, 2, 3, 4):
            window.editor.tabs.setCurrentIndex(tab)
            qt_app.processEvents()
            assert (window.width(), window.height()) == size
        for preview in (window.editor.bed.preview, window.editor.program.preview):
            for axis in preview.figure.axes:
                assert all(t.get_fontfamily() == [NUMERIC_FAMILY] for t in axis.get_xticklabels())
        assert case.documents == before
        for expected, actual in zip(original, [line.get_ydata() for axis in window.editor.figures[0].axes for line in axis.lines]):
            np.testing.assert_array_equal(expected, actual)
    window.editor.tabs.setCurrentWidget(window.editor.bed)
    qt_app.processEvents()
    options = window.editor.bed.settings_split.widget(0)
    fields = [options.layout().itemAt(i).widget() for i in range(options.layout().count())]
    fields = [field for field in fields if field and field.isVisible()]
    for i, field in enumerate(fields):
        for other in fields[i+1:]:
            assert not field.geometry().intersects(other.geometry()), (field.objectName(), other.objectName())
    if size == (1280, 800):
        left, right = window.editor.bed.settings_split.sizes()
        assert .20 <= left / (left + right) <= .30
    window.close()


def test_graph_style_is_optional_and_component_ids_stay_distinct(qt_app, design, tmp_path, source_case):
    from packed_bed.reaction_graph import build_reaction_graph, GraphStyle
    default = build_reaction_graph(['N2'], ['Ni'], [])
    assert 'fillcolor="#d9eee8"' in default.dot
    styled = build_reaction_graph(['N2', 'N2_secondary'], [], [],
                                 tooltips={'N2_secondary': 'N2_secondary · Nitrogen · Custom source'},
                                 style=GraphStyle(gas=DARK['gas']))
    assert [n.label for n in styled.nodes] == ['N2', 'N2_secondary']
    assert 'Custom source' in styled.nodes[1].tooltip
    assert DARK['gas'] in styled.dot
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files(source_case)
    editor = CaseEditor()
    editor.set_case(case)
    assert [n.label for n in editor.chemistry.graph._pending.nodes] == ['N2', 'Ni']
    assert any('Nitrogen' in n.tooltip for n in editor.chemistry.graph._pending.nodes)
    revision = editor.chemistry.graph._revision
    design.apply(True)
    assert editor.chemistry.graph._revision > revision
    assert DARK['gas'] in editor.chemistry.graph._pending.dot


def test_theme_contrast_and_font_resources(qt_app, design):
    def luminance(value):
        rgb = [int(value[i:i+2], 16) / 255 for i in (1, 3, 5)]
        return sum((x / 12.92 if x <= .04045 else ((x + .055) / 1.055) ** 2.4) * w
                   for x, w in zip(rgb, (.2126, .7152, .0722)))
    def contrast(a, b):
        light, dark = sorted((luminance(a), luminance(b)), reverse=True)
        return (light + .05) / (dark + .05)
    for palette in (LIGHT, DARK):
        for text in ('ink', 'muted', 'error', 'warning', 'success', 'active'):
            assert contrast(palette[text], palette['surface']) >= 4.5
        assert contrast(palette['contrastText'], palette['contrastSurface']) >= 4.5
        for text, background in [('warning', 'warningFill'), ('active', 'activeFill'), ('success', 'gas'), ('error', 'missing')]:
            assert contrast(palette[text], palette[background]) >= 4.5
        assert contrast(palette['boundary'], palette['surface']) >= 3
        assert contrast(palette['focus'], palette['paper']) >= 3
    from matplotlib import get_data_path
    from PyQt6.QtGui import QFontDatabase
    assert (Path(get_data_path()) / 'fonts/ttf/DejaVuSansMono.ttf').is_file()
    assert NUMERIC_FAMILY in QFontDatabase.families()


def test_empty_project_authoring_with_two_zones_and_program_steps(qt_app, design, tmp_path):
    """Author a reacting draft through widgets, then reopen and check exact inputs."""
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QDialogButtonBox, QListWidget, QPushButton
    from packed_bed.kinetics import FAMILY_REGISTRY
    from packed_bed.preview import preview_case
    from packed_bed_ui.window import MainWindow

    window = MainWindow()
    project = Project.create(tmp_path / 'authored')
    case = project.add_case('Authored case')
    window._set_project(project)
    window._show_case(case)
    window.show()
    editor = window.editor

    def type_text(field, text):
        field.setFocus()
        QTest.keyClick(field, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClicks(field, text)
        QTest.keyClick(field, Qt.Key.Key_Tab)

    def edit_cell(table, row, col, text):
        table.setCurrentCell(row, col)
        table.editItem(table.item(row, col))
        qt_app.processEvents()
        field = next(field for field in table.findChildren(QLineEdit) if field.isVisible())
        QTest.keyClick(field, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClicks(field, text)
        QTest.keyClick(field, Qt.Key.Key_Return)

    def in_dialog(callback, launch):
        errors = []
        def fill():
            dialog = qt_app.activeModalWidget()
            try:
                callback(dialog)
                buttons = dialog.findChild(QDialogButtonBox)
                button = buttons.button(QDialogButtonBox.StandardButton.Ok) or buttons.button(QDialogButtonBox.StandardButton.Save)
                QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            except Exception as exc:
                errors.append(exc)
                dialog.reject()
        QTimer.singleShot(0, fill)
        launch()
        assert not errors, errors

    def choose(keys, launch):
        def fill(dialog):
            items = dialog.findChild(QListWidget)
            for key in keys:
                item = next(items.item(i) for i in range(items.count()) if items.item(i).data(Qt.ItemDataRole.UserRole) == key)
                items.setCurrentItem(item)
                items.scrollToItem(item)
                QTest.keyClick(items, Qt.Key.Key_Space)
                assert item.checkState() == Qt.CheckState.Checked
        in_dialog(fill, launch)

    editor.tabs.setCurrentWidget(editor.chemistry)
    family = FAMILY_REGISTRY['nickel_medrano']
    choose([*family.required_gas_species, 'N2'], lambda: editor.chemistry.gas_list.add_button.click())
    choose(family.required_solid_species, lambda: editor.chemistry.solid_list.add_button.click())
    add_reactions = next(b for b in editor.chemistry.findChildren(QPushButton) if b.text() == '+ Add reaction families…')
    choose(['nickel_medrano'], add_reactions.click)
    editor.tabs.setCurrentWidget(editor.general)
    type_text(editor.fields['simulation', 'time_horizon_s'], '30')
    type_text(editor.fields['simulation', 'reporting_interval_s'], '1')
    editor.tabs.setCurrentWidget(editor.bed)
    type_text(editor.bed.length, '1')
    type_text(editor.fields['model', 'bed_radius_m'], '.01')
    QTest.mouseClick(editor.bed.add_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(editor.bed.add_button, Qt.MouseButton.LeftButton)
    for row in range(2):
        for col, value in [(2, '.4'), (3, '.5'), (4, '.001'), *[(i + 5, '1') for i in range(len(family.required_solid_species))]]:
            edit_cell(editor.bed.zones, row, col, value)
    editor.tabs.setCurrentWidget(editor.program)
    feed = editor.program.channels['feed_stream']
    def fill_feed(dialog):
        for key, value in [('flow', '1e-8'), ('temperature', '300'),
                           *[(key, '1' if key == 'N2' else '0') for key in case.documents['chemistry']['gas_species']]]:
            type_text(dialog.findChild(QLineEdit, key), value)
    in_dialog(fill_feed, lambda: feed.table.cellWidget(0, 2).click())
    edit_cell(editor.program.channels['outlet_pressure'].table, 0, 2, '100000')
    for row in range(1, 4):
        QTest.mouseClick(feed.add_button, Qt.MouseButton.LeftButton)
        edit_cell(feed.table, row, 1, '10')
    before = deepcopy(case.documents)
    QTest.mouseClick(editor.program.mode.buttons[0], Qt.MouseButton.LeftButton)
    QTest.mouseClick(editor.program.mode.buttons[1], Qt.MouseButton.LeftButton)
    assert case.documents == before
    assert editor.save()
    restored = Project.open(project.root).cases[0]
    assert restored.documents == case.documents
    assert restored.state()['inputs'] == 'Ready'
    zones = restored.documents['solids']['initial_profile']['zones']
    assert [(z['x_start_m'], z['x_end_m']) for z in zones] == [(0, .5), (.5, 1)]
    assert all((z['e_b'], z['e_p'], z['d_p']) == (.4, .5, .001) for z in zones)
    assert [step['duration_s'] for step in restored.documents['program']['feed_stream']['steps']] == [10, 10, 10]
    expected = preview_case(restored.resolve())
    np.testing.assert_array_equal(editor.program.preview.figure.axes[0].lines[0].get_ydata(), expected.flow_mol_s)
    window.close()


def test_theme_graph_render_preserves_selection_and_discards_old_revision(qt_app, design):
    from packed_bed.reaction_graph import GraphvizError, find_graphviz
    from packed_bed_ui.reaction_graph import NetworkView
    try:
        find_graphviz()
    except GraphvizError:
        pytest.skip('Graphviz is unavailable')
    view = NetworkView()
    view.resize(600, 400)
    view.show()
    view.draw(['N2'], ['Ni'], [])
    for _ in range(300):
        if view.graph is not None:
            break
        QTest.qWait(10)
    assert view.graph is not None
    node = view.graph.nodes[0].id
    view.select_node(node)
    view.zoom(1.5)
    revision = view._revision
    design.apply(True)
    view._rendered(revision, view.graph, b'', 'Obsolete failed render')
    assert 'Obsolete' not in view.status
    for _ in range(300):
        if not view.status:
            break
        QTest.qWait(10)
    assert not view.status
    assert DARK['gas'] in view.graph.dot
    assert view.selected_node == node
    assert view._zoom == 1.5
    view.close()


def test_trailing_actions_and_zero_based_step_indices(qt_app, design, tmp_path, source_case):
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files(source_case)
    editor = InputEditor()
    editor.set_case(case)
    editor.resize(1280, 800)
    editor.show()
    editor.tabs.setCurrentWidget(editor.program)
    channel = editor.program.channels['inlet_flow']
    original = deepcopy(channel.channel())
    count = len(original.get('steps', []))
    assert channel.table.cellWidget(count + 1, 0) is channel.add_button
    assert channel.table.columnSpan(count + 1, 0) == channel.table.columnCount()
    channel.add_button.setFocus()
    QTest.keyClick(channel.add_button, Qt.Key.Key_Space)
    channel.table.item(count + 1, 1).setText('2.75')
    assert channel.channel()['steps'][-1]['duration_s'] == 2.75
    assert channel.table.verticalHeaderItem(count + 1).text() == str(count)
    assert channel.table.verticalHeader().font().family() == NUMERIC_FAMILY
    assert f'Remove step {count} ' in channel.table.cellWidget(count + 1, 3).toolTip()
    channel.table.cellWidget(count + 1, 3).click()
    assert channel.channel() == original
    if count:
        channel.remove_step(0)
    for index in range(len(channel.channel().get('steps', []))):
        assert channel.table.verticalHeaderItem(index + 1).text() == str(index)
    editor.tabs.setCurrentWidget(editor.bed)
    old_zones = deepcopy(editor.get(('solids', 'initial_profile', 'zones')))
    assert editor.bed.zones.cellWidget(len(old_zones), 0) is editor.bed.add_button
    editor.bed.add_button.click()
    zones = editor.get(('solids', 'initial_profile', 'zones'))
    assert len(zones) == len(old_zones) + 1
    assert editor.bed.zones.cellWidget(len(zones), 0) is editor.bed.add_button
    assert editor.bed.zones.columnSpan(len(zones), 0) == editor.bed.zones.columnCount()
    assert editor.save()
    assert Project.open(project.root).cases[0].documents == case.documents
    editor.set_case(case, read_only=True)
    assert not editor.bed.add_button.isEnabled()
    assert all(not ch.add_button.isEnabled() for ch in editor.program.channels.values())


def test_chemistry_collapse_reclaims_space_with_keyboard_and_read_only(qt_app, design, tmp_path):
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files('packed_bed/examples/default_case/run_feed_stream.yaml')
    editor = InputEditor()
    editor.resize(1280, 800)
    editor.set_case(case)
    editor.tabs.setCurrentWidget(editor.chemistry)
    editor.show()
    qt_app.processEvents()
    before = deepcopy(case.documents)
    sections = editor.chemistry.sections
    old_height = sections['reactions'].height()
    for key in ('gas', 'solid'):
        sections[key].toggle.setFocus()
        QTest.keyClick(sections[key].toggle, Qt.Key.Key_Space)
    qt_app.processEvents()
    assert sections['reactions'].height() > old_height
    assert all(sections[key].content.isHidden() for key in ('gas', 'solid'))
    editor.chemistry.load()
    assert sections['gas'].content.isHidden()
    sections['reactions'].toggle.click()
    qt_app.processEvents()
    assert all(section.height() < 80 for section in sections.values())
    sections['gas'].toggle.click()
    qt_app.processEvents()
    assert sections['gas'].height() > old_height
    assert case.documents == before and not editor.dirty
    editor.set_case(case, read_only=True)
    for section in sections.values():
        assert section.toggle.isEnabled()
        section.toggle.click()
    assert not editor.chemistry.gas_list.add_button.isEnabled()
    assert case.documents == before and not editor.dirty


def test_validation_details_keyboard_save_failure_and_recovery(qt_app, design, tmp_path, monkeypatch):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QDialog, QPlainTextEdit
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files('packed_bed/examples/default_case/run_feed_stream.yaml')
    editor = InputEditor()
    editor.resize(1280, 800)
    editor.set_case(case)
    editor.show()
    qt_app.processEvents()
    assert editor.validation_details.isHidden()
    radius = editor.fields['model', 'bed_radius_m']
    old_value = radius.text()
    radius.setText('unfinished')
    assert editor.save()
    assert editor.validation_details.isVisible()
    details = editor.validation.toolTip()
    assert details and 'bed_radius_m' in details
    errors, seen = [], []

    def inspect():
        dialog = qt_app.activeModalWidget()
        try:
            assert isinstance(dialog, QDialog)
            text = dialog.findChild(QPlainTextEdit)
            assert text.isReadOnly()
            assert text.toPlainText() == details
            text.selectAll()
            assert text.textCursor().selectedText()
            seen.append(True)
            QTest.keyClick(dialog, Qt.Key.Key_Escape)
        except Exception as exc:
            errors.append(exc)
            dialog.reject()

    editor.validation_details.setFocus()
    QTimer.singleShot(0, inspect)
    QTest.keyClick(editor.validation_details, Qt.Key.Key_Space)
    assert seen and not errors, errors
    radius.setText(old_value)
    assert editor.validation_details.isHidden()  # Do not show stale errors during autosave.
    with monkeypatch.context() as patch:
        def fail():
            raise OSError('Storage unavailable\nThe draft is still open.')
        patch.setattr(case, 'save', fail)
        assert not editor.save()
        assert editor.dirty
        assert editor.validation_details.isVisible()
        assert editor.validation.property('state') == 'error'
        assert 'Storage unavailable' in editor.validation.toolTip()
    assert editor.save()
    assert not editor.dirty
    assert editor.validation_details.isHidden()
    assert editor.validation.property('state') == 'success'


@pytest.mark.parametrize('dark', [False, True])
def test_binary_selection_has_no_radio_indicator_and_clear_fill(qt_app, design, dark):
    from PyQt6.QtWidgets import QPushButton, QRadioButton
    from packed_bed_ui.theme import ACID
    design.apply(dark)
    choice = BinaryChoice([('Standard', 'standard'), ('Compiled', 'compiled')])
    choice.resize(360, 40)
    choice.show()
    qt_app.processEvents()
    assert not choice.findChildren(QRadioButton)
    for index in (0, 1):
        choice.setCurrentIndex(index)
        qt_app.processEvents()
        selected = choice.buttons[index]
        other = choice.buttons[1-index]
        assert isinstance(selected, QPushButton) and selected.isCheckable()
        assert selected.isChecked() and not other.isChecked()
        # Sample inside the border, away from the centered text and keyboard focus.
        assert selected.grab().toImage().pixelColor(5, 5).name() == ACID
        assert other.grab().toImage().pixelColor(5, 5).name() != ACID


def test_chemistry_family_actions_fit_their_rows(qt_app, design, tmp_path):
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files('packed_bed/examples/default_case/run_feed_stream.yaml')
    editor = InputEditor()
    editor.resize(1280, 800)
    editor.set_case(case)
    editor.tabs.setCurrentWidget(editor.chemistry)
    editor.show()
    qt_app.processEvents()
    tree = editor.chemistry.families
    for index in range(tree.topLevelItemCount()):
        root = tree.topLevelItem(index)
        for column in (1, 2):
            button = tree.itemWidget(root, column)
            assert button.height() >= button.sizeHint().height()
            assert button.width() >= button.sizeHint().width()


@pytest.mark.parametrize('kind', ['bed', 'program'])
def test_definition_dialog_fits_screen_with_accessible_preview_and_actions(qt_app, design, tmp_path, kind):
    from PyQt6.QtWidgets import QDialogButtonBox
    from packed_bed_ui.definition_editor import DefinitionDialog
    project = Project.create(tmp_path / 'project')
    case = project.add_case_from_files('packed_bed/examples/default_case/run_feed_stream.yaml')
    original = deepcopy(case.documents)
    dialog = DefinitionDialog(project.study_store, kind, case.documents)
    dialog.show()
    qt_app.processEvents()
    assert dialog.frameGeometry().height() <= dialog.screen().availableGeometry().height()
    assert dialog.height() <= 740
    preview = getattr(dialog.editor, kind).preview
    assert preview.canvas.height() >= (300 if kind == 'program' else 180)
    save = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Save)
    assert dialog.rect().contains(save.mapTo(dialog, save.rect().bottomRight()))
    assert not dialog.editor.tabs.tabBar().isVisible()
    if kind == 'bed':
        control = dialog.editor.fields['simulation', 'interior_flow_mode']
        dialog.controls_scroll.ensureWidgetVisible(control)
        qt_app.processEvents()
        center = control.mapTo(dialog.controls_scroll.viewport(), control.rect().center())
        assert dialog.controls_scroll.viewport().rect().contains(center)
    dialog.reject()
    assert case.documents == original
    assert not project.study_store.definitions


def test_plot_icons_follow_live_theme_without_resetting_navigation(qt_app, design):
    from packed_bed_ui.editor_widgets import Preview
    preview = Preview()
    preview.resize(600, 400)
    axis = preview.figure.subplots()
    axis.plot([0, 1, 2], [0, 2, 1])
    preview.show()
    preview.draw()
    axis.set_xlim(.25, .75)
    preview.toolbar.push_current()
    preview.toolbar.pan()
    action = preview.toolbar._actions['home']
    light = action.icon().pixmap(24, 24).toImage()
    design.apply(True)
    qt_app.processEvents()
    dark = action.icon().pixmap(24, 24).toImage()
    assert dark != light
    assert tuple(axis.get_xlim()) == (.25, .75)
    assert preview.toolbar._actions['pan'].isChecked()
    design.apply(False)
    qt_app.processEvents()
    assert action.icon().pixmap(24, 24).toImage() == light
    assert tuple(axis.get_xlim()) == (.25, .75)


def test_report_headers_use_available_space_and_preserve_full_labels(qt_app, design):
    from packed_bed_ui.report import ReportPage
    page = ReportPage()
    page.resize(1280, 800)
    page.show()
    heading = 'Temperature (K) | Cell position=0.1666666666666667 m'
    page.show_table(['Time (s)', heading], [[0, 300], [1, 303]])
    qt_app.processEvents()
    assert page.table.horizontalHeaderItem(1).text() == heading
    assert page.table.horizontalHeaderItem(1).toolTip() == heading
    assert page.table.columnWidth(1) > 170
    assert page.table.item(1, 1).text() == '303'
    assert page.table.horizontalHeader().textElideMode() == Qt.TextElideMode.ElideRight


def test_species_prose_uses_text_font_and_numeric_drafts_keep_monospace(qt_app, design):
    from packed_bed.plugins.catalogue import builtin_manifest
    from packed_bed_ui.plugin_forms import SpeciesForm
    definition = builtin_manifest().model_dump(mode='json')['species']['Ni']
    form = SpeciesForm(definition)
    form.show()
    qt_app.processEvents()
    for key in ('name', 'chemical_key', 'source', 'notes'):
        assert form.fields[key].font().family() == design.body
        assert not form.fields[key].property('numeric')
    form.fields['mw'].setText('1e-')
    assert form.fields['mw'].font().family() == NUMERIC_FAMILY
    assert form.value()['mw'] == '1e-'
