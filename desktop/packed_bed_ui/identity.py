"""Shared application identity and welcome actions."""
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter
from .theme import manager, colors
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QMenuBar, QPushButton, QVBoxLayout, QWidget
from .theme import label, numeric
from .welcome_animation import CycleEmblem


class Wordmark(QWidget):
    """Maratype identity, with a condensed fallback if the asset is unavailable."""
    def __init__(self, size=64, *, ribbon=False):
        super().__init__()
        self.pixel_size, self.ribbon = size, ribbon
        self.setAccessibleName("MULTISOLID")
        self.setFixedHeight(round(size * 1.25))

    def wordmark_font(self):
        font = QFont(manager().display)
        font.setPixelSize(self.pixel_size)
        font.setWeight(QFont.Weight.Normal if font.family() == 'Maratype' else QFont.Weight.Black)
        if font.family() == "DejaVu Sans":
            font.setStretch(80)
        return font

    def sizeHint(self):
        return QSize(QFontMetrics(self.wordmark_font()).horizontalAdvance("MULTISOLID"), self.height())

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setFont(self.wordmark_font())
        painter.setPen(Qt.GlobalColor.white if self.ribbon else QColor(colors()['ink']))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "MULTISOLID")


class Masthead(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName('masthead')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 4, 24, 4)
        layout.setSpacing(16)
        for text, name in (('M/S', 'brandIndex'),
                           ('REACTIVE PACKED BEDS\nTRANSIENT PROCESS SIMULATION', 'brandDescriptor')):
            widget = label(text)
            widget.setObjectName(name)
            layout.addWidget(widget)
            if name == 'brandDescriptor':
                self.descriptor = widget
        self.menus = QMenuBar()
        self.menus.setNativeMenuBar(False)
        self.menus.setObjectName('ribbonMenus')
        layout.addWidget(self.menus)
        layout.addStretch()
        edition = label('HEAT & MASS TRANSFER / HETEROGENEOUS REACTIONS')
        edition.setObjectName('brandEdition')
        layout.addWidget(edition)
        self.edition = edition

    def set_workspace(self, active):
        self.descriptor.setVisible(not active)
        self.edition.setVisible(not active)


class WelcomeHero(QFrame):
    def __init__(self, create, open_project, import_archive, settings):
        super().__init__()
        self.setObjectName('welcomeHero')
        self.setMinimumHeight(330)
        self.setMaximumHeight(470)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        introduction = QWidget()
        left = QVBoxLayout(introduction)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)
        content = QVBoxLayout()
        content.setContentsMargins(24, 24, 24, 16)
        content.setSpacing(12)
        content.addWidget(Wordmark())
        content.addWidget(label('Resolve axial profiles of temperature, pressure, velocity, and concentration '
                                'in packed beds with changing gas feeds.', wrap=True))
        content.addWidget(label('Combustion. Reforming. Energy storage.', 'muted', wrap=True))
        content.addWidget(label('Nickel, iron, copper, and custom materials.', 'muted', wrap=True))
        content.addStretch()
        left.addLayout(content, 1)
        footer = QHBoxLayout()
        footer.setContentsMargins(20, 12, 16, 12)
        footer.setSpacing(6)
        self.actions = []
        for text, action in (('Create new project…', create), ('Open existing project…', open_project),
                             ('Import archived project…', import_archive)):
            button = QPushButton(text)
            button.setProperty('role', 'welcomeAction')
            button.setFixedHeight(40)
            button.setAccessibleName(text)
            button.clicked.connect(action)
            footer.addWidget(button)
            self.actions.append(button)
        self.actions[0].setProperty('primary', True)
        footer.addStretch()
        left.addLayout(footer)
        layout.addWidget(introduction, 1)
        self.art_pane = QWidget()
        self.art_pane.setObjectName('animationPane')
        self.art_pane.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        art = QVBoxLayout(self.art_pane)
        art.setContentsMargins(0, 0, 0, 0)
        art.setSpacing(0)
        self.animation = CycleEmblem(settings=settings)
        art.addWidget(self.animation, 1)
        art_footer = QHBoxLayout()
        art_footer.setContentsMargins(20, 12, 16, 12)
        caption = numeric(label('CHEMICAL LOOPING'), 11)
        caption.setObjectName('artCaption')
        art_footer.addWidget(caption)
        art_footer.addStretch()
        self.pause_button = QPushButton()
        self.pause_button.setProperty('role', 'artControl')
        self.pause_button.setFixedHeight(40)
        self.pause_button.clicked.connect(lambda: self.animation.set_paused(not self.animation.paused))
        self.animation.pausedChanged.connect(self.update_pause)
        self.update_pause(self.animation.paused)
        art_footer.addWidget(self.pause_button)
        art.addLayout(art_footer)
        layout.addWidget(self.art_pane, 1)

    def update_pause(self, paused):
        self.pause_button.setText('Play animation' if paused else 'Pause animation')
        self.pause_button.setAccessibleName(self.pause_button.text())


def recent_empty():
    frame = QFrame()
    frame.setObjectName('recentEmpty')
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(24, 20, 24, 20)
    layout.addWidget(label('NO RECENT PROJECTS', 'muted'))
    layout.addWidget(label('Define a simulation project', 'emptyTitle'))
    layout.addWidget(label('A project groups individual simulation cases, parameter studies, bed configurations, '
                           'reaction mechanisms, and operating programs.', 'muted', wrap=True))
    return frame
