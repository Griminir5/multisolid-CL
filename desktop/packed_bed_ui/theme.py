"""Desktop-only typography, palette and figure styling; follow the system scheme."""
from functools import lru_cache
from html import escape
import re
from pathlib import Path

from matplotlib import get_data_path
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt6.QtWidgets import QApplication, QLabel

INK, ACID = '#242720', '#d9f36d'
NUMERIC_FAMILY = 'DejaVu Sans Mono'
NUMERIC_ROLE = Qt.ItemDataRole.UserRole + 71
LOCKED_ROLE = Qt.ItemDataRole.UserRole + 72
LIGHT = dict(paper='#f1f0e8', surface='#faf9f3', ink=INK, muted='#65685c', line='#d0d1c4',
             boundary='#636957', contrastSurface=INK, contrastText='#f1f0e8', contrastBorder=INK, focus='#245b83', selected='#e3edbc', alternate='#edeee5', disabled='#858978',
             error='#9b3d2a', warning='#715926', success='#3d5028', active='#245b83',
             warningFill='#ede3c9', activeFill='#ddeaf3', gas='#e0ecc0', solid='#f0dfcd', reaction='#e7e9df', missing='#f2ddd4')
DARK = dict(paper='#1e221c', surface='#292e25', ink='#f1f0e8', muted='#b7bda9', line='#454d3e',
            boundary='#7f8975', contrastSurface=ACID, contrastText=INK, contrastBorder=ACID, focus='#8dcaf2', selected='#414d2b', alternate='#30362b', disabled='#89917f',
            error='#ffb6a2', warning='#eac779', success='#c5e393', active='#a0d2f2',
            warningFill='#3e3622', activeFill='#263947', gas='#3e5030', solid='#5a4231', reaction='#394033', missing='#4a2c25')

@lru_cache(maxsize=16)
def font_family(*candidates):
    available = set(QFontDatabase.families())
    return next((name for name in candidates if name in available), QApplication.font().family())

def numeric_font(size=14):
    font = QFont(NUMERIC_FAMILY)
    font.setPixelSize(size)
    return font

def numeric(widget, size=14):
    widget.setProperty('numeric', True)
    widget.setFont(numeric_font(size))
    return widget

class NumberLabel(QLabel):
    """Body text with numeric spans; public text remains plain for status consumers."""
    def __init__(self, text='', **kwargs):
        super().__init__(**kwargs)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setText(text)

    def setText(self, text):
        self._plain = str(text)
        parts, offset = [], 0
        for match in re.finditer(r"(?<![\w])[-+]?(?:\d+(?:[.,]\d+)*|\.\d+)(?:[eE][-+]?\d+)?(?![\w])", self._plain):
            parts.append(escape(self._plain[offset:match.start()]))
            parts.append(f'<span style="font-family: {NUMERIC_FAMILY}">{escape(match.group())}</span>')
            offset = match.end()
        parts.append(escape(self._plain[offset:]))
        super().setText(''.join(parts).replace('\n', '<br>'))

    def text(self):
        return self._plain


def set_state(widget, state):
    if widget.property('state') != state:
        widget.setProperty('state', state)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


def label(text, role='', *, wrap=False):
    widget = QLabel(text)
    widget.setProperty('role', role)
    widget.setWordWrap(wrap)
    return widget

def manager():
    return getattr(QApplication.instance(), '_multisolid_theme', None)

def colors():
    return manager().colors if manager() else LIGHT

class Theme(QObject):
    changed = pyqtSignal()

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self._native_dark = app.palette().window().color().lightness() < 128
        self.colors = LIGHT
        # The same bundled font files serve both Qt and Matplotlib, offline.
        fonts = Path(get_data_path()) / 'fonts' / 'ttf'
        for name in ('DejaVuSansMono.ttf', 'DejaVuSansMono-Bold.ttf', 'DejaVuSansMono-Oblique.ttf'):
            QFontDatabase.addApplicationFont(str(fonts / name))
        app.setStyle('Fusion')
        self.body = font_family('Segoe UI', 'Helvetica Neue', 'DejaVu Sans')
        self.display = font_family('Impact', 'Arial Narrow', 'DejaVu Sans')
        from .branding import brand_family
        self.display = brand_family(self.display)
        self.follow_system()
        app.styleHints().colorSchemeChanged.connect(self.follow_system)
        app.paletteChanged.connect(self.native_palette_changed)

    def native_palette_changed(self, palette):
        if not getattr(self, '_applying', False) and self.app.styleHints().colorScheme() == Qt.ColorScheme.Unknown:
            self._native_dark = palette.window().color().lightness() < 128
            self.follow_system()

    def follow_system(self, *_):
        scheme = self.app.styleHints().colorScheme()
        self.apply(scheme == Qt.ColorScheme.Dark if scheme != Qt.ColorScheme.Unknown else self._native_dark)

    def apply(self, dark):
        """Apply a resolved scheme; explicit selection is only for visual tests."""
        self._applying = True
        self.colors = c = DARK if dark else LIGHT
        font = QFont(self.body)
        font.setPixelSize(14)
        self.app.setFont(font)
        palette = QPalette()
        for role, token in dict(Window='paper', WindowText='ink', Base='surface', AlternateBase='alternate',
                                Text='ink', Button='surface', ButtonText='ink', Highlight='selected',
                                HighlightedText='ink', ToolTipBase='surface', ToolTipText='ink',
                                PlaceholderText='muted', Mid='boundary', Light='surface', Dark='muted',
                                Link='focus', BrightText='ink').items():
            palette.setColor(getattr(QPalette.ColorRole, role), QColor(c[token]))
        for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
            palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(c['disabled']))
        self.app.setPalette(palette)
        style = STYLES
        tokens = {**c, 'body': self.body, 'display': self.display, 'mono': NUMERIC_FAMILY,
                  'assets': Path(__file__).with_name('assets').as_posix(),
                  'mode': 'dark' if dark else 'light', 'checkink': c['paper']}
        for key in sorted(tokens, key=len, reverse=True):
            value = tokens[key]
            style = style.replace('@' + key, value)
        self.app.setStyleSheet(style)
        self._applying = False
        self.changed.emit()

def apply_theme(app):
    if not getattr(app, '_multisolid_theme', None):
        app._multisolid_theme = Theme(app)
    return app._multisolid_theme

def style_figure(figure):
    c = colors()
    figure.set_facecolor(c['paper'])
    body = manager().body if manager() else font_family('DejaVu Sans')
    palette = (('#86c8ee', '#e7ab80', '#8fd6b0', '#d6a3d0', '#e1cf85', '#c1c9d2') if c is DARK else
               ('#0072b2', '#be492b', '#007a5e', '#924c8e', '#875800', '#56616d'))
    strokes = ('-', '--', '-.', ':')
    legends = []
    for axis in figure.axes:
        axis.set_facecolor(c['surface'])
        axis.tick_params(colors=c['muted'], labelsize=9)
        for text in (*axis.get_xticklabels(), *axis.get_yticklabels(), axis.xaxis.offsetText, axis.yaxis.offsetText):
            text.set_fontfamily(NUMERIC_FAMILY)
        for text in (axis.xaxis.label, axis.yaxis.label, axis.title):
            text.set_color(c['ink'])
            text.set_fontfamily(body)
            text.set_fontsize(10)
        for position, spine in axis.spines.items():
            spine.set_color(c['boundary'])
            spine.set_visible(position in ('left', 'bottom'))
        axis.grid(color=c['line'], alpha=.7, linewidth=.6)
        series = 0
        for artist in (*axis.lines, *axis.patches):
            if getattr(artist, '_multisolid_guide', False):
                artist.set_color(c['muted'])
                continue
            artist.set_color(palette[series % len(palette)])
            artist.set_linestyle(strokes[series % len(strokes)])
            series += 1
        legend = axis.get_legend()
        if legend:
            legends.append((legend, axis.get_legend_handles_labels()[0]))
    for legend in figure.legends:
        legends.append((legend, [handle for axis in figure.axes for handle in axis.get_legend_handles_labels()[0]]))
    for legend, handles in legends:
        if legend:
            for handle, source in zip(legend.legend_handles, handles):
                if hasattr(source, 'get_color') and hasattr(handle, 'set_color'):
                    handle.set_color(source.get_color())
                    handle.set_linestyle(source.get_linestyle())
                elif hasattr(source, 'get_edgecolor'):
                    handle.set_color(source.get_edgecolor())
            legend.get_frame().set_facecolor(c['surface'])
            legend.get_frame().set_edgecolor(c['line'])
            for text in legend.get_texts():
                text.set_color(c['ink'])
                text.set_fontfamily(body)

STYLES = '''
QWidget { color: @ink; font-family: "@body"; font-size: 14px; }
QMainWindow, QDialog { background: @paper; }
QLabel { background: transparent; }
QLabel[role="muted"], QLabel[role="note"] { color: @muted; font-size: 13px; }
QLabel[role="section"] { font-weight: 600; }
QLabel[role="pageTitle"] { font-size: 26px; font-weight: 700; }
QLabel[role="projectTitle"] { font-family: "@display"; font-size: 32px; }
QLabel[role="display"] { font-family: "@display"; font-size: 64px; font-weight: 900; }
QLabel[role="emptyTitle"] { font-size: 28px; font-weight: 700; }
*[numeric="true"], QSpinBox, QDoubleSpinBox { font-family: "@mono"; }
QLabel[role="validation"] { color: @muted; background: @alternate; border-left: 4px solid @muted; padding: 6px 10px; }
QLabel[role="validation"][state="error"] { background: @missing; border-color: @error; }
QLabel[role="validation"][state="warning"] { background: @warningFill; border-color: @warning; }
QLabel[role="validation"][state="success"] { background: @gas; border-color: @success; }
QLabel[role="validation"][state="active"] { background: @activeFill; border-color: @active; }
QLabel[state="error"] { color: @error; }
QLabel[state="warning"] { color: @warning; }
QLabel[state="success"] { color: @success; }
QLabel[state="active"] { color: @active; }
QWidget#masthead { background: #242720; }
QLabel#brandIndex { background: #d9f36d; color: #242720; padding: 5px 10px; font-family: "@display"; font-size: 24px; }
QMenuBar#ribbonMenus { background: #242720; color: #f1f0e8; }
QMenuBar#ribbonMenus::item { background: transparent; padding: 9px 14px; font-weight: 600; }
QMenuBar#ribbonMenus::item:selected { background: #d9f36d; color: #242720; }
QMenuBar#ribbonMenus::item:disabled { color: #858978; }
QLabel#wordmark { font-family: "@display"; color: #f1f0e8; font-size: 24px; font-weight: 900; }
QLabel#brandDescriptor, QLabel#brandEdition { color: #b8bbad; font-size: 11px; }
QFrame#welcomeHero, QFrame#recentEmpty { background: @surface; border: 2px solid @boundary; }
QPushButton, QToolButton {
    background: @surface; border: 2px solid @boundary; border-radius: 0; padding: 7px 12px; font-weight: 600;
}
QPushButton:hover, QToolButton:hover { background: @selected; }
QPushButton:pressed { background: @contrastSurface; color: @contrastText; border-color: @contrastBorder; }
QToolButton:checked { background: @selected; }
QPushButton[segment="true"] { spacing: 0; padding: 5px 8px; font-weight: 600; }
QPushButton[segment="true"]:checked { background: #d9f36d; color: #242720; border-color: @contrastBorder; }
QPushButton[role="welcomeAction"] { font-size: 13px; padding: 6px 7px; }
QWidget#animationPane { background: #242720; }
QPushButton[primary="true"], QPushButton[role="primary"] { background: #d9f36d; color: #242720; border-color: #242720; font-weight: 700; }
QPushButton[primary="true"]:hover, QPushButton[role="primary"]:hover { background: #e4fc91; }
QPushButton:disabled, QToolButton:disabled { color: @disabled; border-color: @line; background: @alternate; }
QPushButton[segment="true"]:checked:disabled { background: @selected; color: @disabled; border-color: @line; }
QToolButton { padding: 4px; }
QToolButton[role="channelTitle"] { border: none; text-align: left; font-weight: 700; background: transparent; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {
    background: @surface; border: 1px solid @boundary; border-radius: 0; padding: 5px 6px;
    selection-background-color: @selected; selection-color: @ink;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled { color: @disabled; background: @alternate; }
QComboBox { padding-right: 22px; combobox-popup: 0; }
QComboBox::drop-down { width: 20px; border: none; }
QComboBox::down-arrow { image: url("@assets/chevron-down-@mode.svg"); width: 12px; height: 12px; }
QSpinBox, QDoubleSpinBox { padding-right: 22px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 20px; border-left: 1px solid @boundary; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 20px; border-left: 1px solid @boundary; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url("@assets/chevron-up-@mode.svg"); width: 10px; height: 10px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url("@assets/chevron-down-@mode.svg"); width: 10px; height: 10px; }
QCheckBox::indicator, QTreeView::indicator, QTableView::indicator { width: 14px; height: 14px; border: 1px solid @boundary; background: @surface; }
QCheckBox::indicator:checked, QTreeView::indicator:checked, QTableView::indicator:checked { background: @focus; image: url("@assets/check-@mode.svg"); }
QCheckBox::indicator:indeterminate, QTreeView::indicator:indeterminate { background: @focus; image: url("@assets/mixed-@mode.svg"); }
QComboBox QAbstractItemView { padding: 0; border: 1px solid @boundary; }
QComboBox QAbstractItemView::item { min-height: 28px; padding: 0 6px; }
QCheckBox { spacing: 7px; }
QGroupBox { border: 2px solid @line; border-top-color: @boundary; margin-top: 22px; padding: 9px 5px 5px; font-weight: 700; }
QGroupBox::title { subcontrol-origin: margin; left: 0; top: 0; padding-right: 6px; }
QGroupBox[role="channel"] { margin-top: 0; padding: 0; }
QTabWidget::pane { border: 0; border-top: 1px solid @boundary; padding-top: 6px; }
QTabBar::tab { padding: 10px 20px; background: @alternate; border-bottom: 4px solid transparent; font-weight: 600; }
QTabBar::tab:selected { background: @contrastSurface; color: @contrastText; border-bottom-color: #d9f36d; }
QTabBar::tab:!selected:hover { background: @selected; }
QTabBar::tab:selected:focus { border-bottom-color: @focus; }
QTableView, QTreeView, QListView { background: @surface; alternate-background-color: @alternate;
    border: 1px solid @line; gridline-color: @line; selection-background-color: @selected; selection-color: @ink; }
QTreeView::item, QListView::item { padding: 4px; }
QHeaderView::section { background: @alternate; color: @ink; border: 0; border-bottom: 2px solid @boundary; padding: 5px; font-size: 12px; font-weight: 600; }
QHeaderView::section:vertical { border-bottom: 1px solid @line; border-right: 1px solid @line; padding: 0 5px; }
QTreeView QPushButton, QTableView QPushButton, QTableView QPushButton[segment="true"], QTableView QLineEdit, QTableView QComboBox { border-width: 1px; padding: 2px 5px; font-size: 13px; }
QTableView QPushButton[role="tableAction"] { text-align: left; padding: 6px 10px; border: 0; background: @alternate; font-weight: 600; }
QTableView QPushButton[role="tableAction"]:hover { background: @selected; }
QTableView QPushButton[role="tableAction"]:disabled { color: @disabled; }
QMenuBar, QStatusBar, QToolBar { background: @paper; }
QMenu { background: @surface; border: 1px solid @boundary; padding: 4px; }
QMenu::item { padding: 6px 22px; }
QMenu::item:selected, QMenuBar::item:selected { background: @selected; }
QMenu::item:disabled { color: @disabled; }
QStatusBar { border-top: 1px solid @line; font-size: 12px; }
QToolTip { background: @surface; color: @ink; border: 1px solid @boundary; padding: 5px; }
QScrollArea, QScrollArea > QWidget > QWidget { border: none; background: @paper; }
QSplitter::handle { background: @line; }
QSplitter::handle:hover { background: @boundary; }
QToolBar { border: 0; spacing: 2px; }
QToolBar QToolButton { border: none; background: transparent; }
QToolBar QToolButton:hover { background: @selected; }
QPushButton:focus, QPushButton[role]:focus, QToolButton:focus, QPushButton[segment="true"]:focus { border: 2px solid @focus; }
QToolButton[role="channelTitle"]:focus { padding: 2px; }
QTableView QPushButton[role="tableAction"]:focus { padding: 4px 8px; }
QTreeView QPushButton:focus, QTableView QPushButton[segment="true"]:focus, QTableView QPushButton:focus { padding: 1px 4px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus { border: 2px solid @focus; padding: 4px 5px; }
QSpinBox:focus, QDoubleSpinBox:focus { padding-right: 21px; }
QComboBox:focus { padding-right: 21px; }
QCheckBox:focus { outline: 2px solid @focus; }
QLineEdit[invalidInput="true"], QComboBox[invalidInput="true"], QSpinBox[invalidInput="true"],
QTableView[invalidInput="true"], QTreeView[invalidInput="true"],
QPushButton[invalidInput="true"], QCheckBox[invalidInput="true"] {
    background: @missing; border: 1px solid @error;
}
QTableView QPushButton[role="tableAction"][invalidInput="true"] {
    background: @missing; border: 1px solid @error; color: @error;
}
QWidget[invalidInput="true"] QPushButton[segment="true"] {
    background: @missing; border-color: @error; color: @ink;
}
QToolButton[role="channelTitle"][invalidInput="true"] { color: @error; background: @missing; }
QLineEdit[invalidInput="true"]:focus, QComboBox[invalidInput="true"]:focus,
QSpinBox[invalidInput="true"]:focus, QPushButton[invalidInput="true"]:focus { border: 2px solid @focus; }
'''
