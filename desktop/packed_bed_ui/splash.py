"""Paint a launch indicator before importing the scientific workspace."""
from pathlib import Path

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QSplashScreen

from .branding import brand_family


def show_splash(app):
    body_font = QFont(app.font())
    ratio = app.primaryScreen().devicePixelRatio()
    canvas = QPixmap(round(520 * ratio), round(240 * ratio))
    canvas.setDevicePixelRatio(ratio)
    canvas.fill(QColor('#242720'))
    painter = QPainter(canvas)
    icon = QIcon(str(Path(__file__).parent / 'assets/multisolid.svg'))
    icon.paint(painter, QRect(28, 48, 124, 124))
    font = QFont(brand_family(app.font().family()))
    font.setPixelSize(40)
    painter.setFont(font)
    painter.setPen(QColor('#f1f0e8'))
    painter.drawText(QRect(172, 60, 328, 64), Qt.AlignmentFlag.AlignVCenter, 'MULTISOLID')
    font = body_font
    font.setPixelSize(15)
    painter.setFont(font)
    painter.setPen(QColor('#d9f36d'))
    painter.drawText(QRect(174, 130, 310, 32), Qt.AlignmentFlag.AlignVCenter, 'Launching your workspace…')
    painter.end()
    splash = QSplashScreen(canvas)
    splash.setWindowTitle('Launching MultiSolid')
    splash.show()
    app.processEvents()
    return splash
