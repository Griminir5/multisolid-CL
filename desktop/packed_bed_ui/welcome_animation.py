"""A slow identity animation; never a calculated process preview."""

from functools import lru_cache
from math import cos, floor, pi, sin
from random import Random

from PyQt6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from .theme import ACID, INK, font_family


PARTICLE_COLUMNS = 11


def _packed_circles():
    random = Random(731)
    circles = []
    for row in range(26):
        for column in range(PARTICLE_COLUMNS):
            theta = -pi / 2 + (column + .25 * (row % 2)) * pi / (PARTICLE_COLUMNS - 1)
            # Project a cylindrical surface: compressed edges and gently curved rows.
            circles.append((33 * sin(theta) + random.uniform(-.12, .12),
                            (row - 12.5) * 7.2 + 4 * cos(theta) + random.uniform(-.2, .2)))
    return tuple(circles)


PARTICLES = _packed_circles()


@lru_cache(maxsize=90)
def _switch_points(bed, sweep):
    """One fixed, random switching time per grain for this upward sweep.

    Spatially bounded jitter scatters the transition around a moving front.
    Sampling at paint time would cause flicker; this schedule is independent of
    frame rate, and changes only when the next front enters the bed.
    """
    random = Random(1709 + bed * 101 + sweep * 7919)
    phase = random.uniform(-pi, pi)
    curve = random.uniform(.012, .024)
    speed = random.uniform(-.05, .05)
    positions = [(96 - y) / 192 + curve * sin(x / 12 + phase)
                 + speed * sin(pi * (96 - y) / 192) + random.uniform(-.065, .065)
                 for x, y in PARTICLES]
    low, high = min(positions), max(positions)
    return tuple(.025 + .95 * (position - low) / (high - low) for position in positions)


class CycleEmblem(QWidget):
    """Packed circles switch state behind slowly rotating, colour-shifting helices.

    Three phase offsets evoke cyclic operation without prescribing reactor
    connections, products or calculated conversion fronts. Time advances only
    while visible.
    """

    SUPER_CYCLE_SECONDS = 720
    SWEEPS = 30
    PERIOD_SECONDS = 48
    HELIX_ROTATION_SECONDS = 36
    HELIX_TURNS = 1 / 6
    COPPER = "#d79a72"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._seconds = 0.0
        self._clock = QElapsedTimer()
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.update)
        self.setMinimumWidth(350)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("Chemical looping animation")
        self.setAccessibleDescription(
            "Three beds of stationary circles switch between lime and copper, grain by grain, "
            "around upward-moving fronts. Outer helices rotate and gradually change colour. "
            "An abstract motif of material states and retained heat."
        )

    @property
    def seconds(self):
        return self._seconds + (self._clock.elapsed() / 1000 if self._clock.isValid() else 0)

    def _start(self):
        if self.isVisible() and not self.timer.isActive():
            self._clock.start()
            self.timer.start()

    def _stop(self):
        self._seconds = self.seconds
        self._clock.invalidate()
        self.timer.stop()

    def showEvent(self, event):
        super().showEvent(event)
        self._start()

    def hideEvent(self, event):
        self._stop()
        super().hideEvent(event)

    @classmethod
    def particle_states(cls, bed, seconds):
        sweep_time = (seconds + bed * cls.PERIOD_SECONDS / 3) / (cls.PERIOD_SECONDS / 2)
        sweep = floor(sweep_time)
        progress = sweep_time - sweep
        return tuple((progress >= point) != bool(sweep % 2) for point in _switch_points(bed, sweep % cls.SWEEPS))

    @staticmethod
    def _blend(first, second, amount):
        return QColor.fromRgbF(
            first.redF() * (1 - amount) + second.redF() * amount,
            first.greenF() * (1 - amount) + second.greenF() * amount,
            first.blueF() * (1 - amount) + second.blueF() * amount,
        )

    def _helices(self, bed, seconds):
        rotation = seconds * 2 * pi / self.HELIX_ROTATION_SECONDS + bed * .7
        colour_phase = seconds * 2 * pi / 60 + bed * 2 * pi / 3
        lime, copper = QColor(ACID), QColor(self.COPPER)
        helices = []
        for strand in range(8):
            back, front = QPainterPath(), QPainterPath()
            previous, previous_front = None, None
            for step in range(97):
                t = step / 96
                angle = strand * 2 * pi / 8 + rotation + t * 2 * pi * self.HELIX_TURNS
                depth = sin(angle)
                radius = 44 + 4 * sin(pi * t)
                point = QPointF(radius * cos(angle), -108 + 216 * t + 7 * depth)
                in_front = depth >= 0
                path = front if in_front else back
                if previous is None:
                    path.moveTo(point)
                else:
                    if in_front != previous_front:
                        path.moveTo(previous)
                    path.lineTo(point)
                previous, previous_front = point, in_front
            colour = self._blend(copper, lime, (1 + sin(colour_phase + strand * .45)) / 2)
            helices.append((back, front, colour))
        return helices

    def _bed(self, painter, bed, seconds):
        painter.save()
        painter.translate((bed - 1) * 155, 0)
        helices = self._helices(bed, seconds)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for back, _, colour in helices:
            colour.setAlpha(75)
            painter.setPen(QPen(colour, .8))
            painter.drawPath(back)

        # The grains stay fixed. Only their binary state changes; no crossfade.
        shell = QPainterPath()
        shell.moveTo(-35, -102)
        shell.cubicTo(-35, -111, 35, -111, 35, -102)
        shell.lineTo(35, 102)
        shell.cubicTo(35, 111, -35, 111, -35, 102)
        shell.closeSubpath()
        shade = QLinearGradient(-35, 0, 35, 0)
        shade.setColorAt(0, QColor("#191e17"))
        shade.setColorAt(.38, QColor("#343a2c"))
        shade.setColorAt(.7, QColor("#292f23"))
        shade.setColorAt(1, QColor("#191e17"))
        painter.fillPath(shell, shade)
        painter.setBrush(shade)
        painter.setPen(QPen(QColor("#626c50"), .6))
        painter.drawEllipse(QRectF(-35, -109, 70, 14))
        painter.setPen(Qt.PenStyle.NoPen)
        for (x, y), state in zip(PARTICLES, self.particle_states(bed, seconds)):
            depth = max(0, 1 - (x / 33) ** 2) ** .5
            colour = QColor(ACID if state else self.COPPER)
            colour.setAlpha(round(100 + 155 * depth))
            painter.setBrush(colour)
            radius = 1.45 + 1.4 * depth
            painter.drawEllipse(QPointF(x, y), radius, radius)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        for _, front, colour in helices:
            colour.setAlpha(175)
            painter.setPen(QPen(colour, .85))
            painter.drawPath(front)
        painter.setPen(QPen(QColor("#8c9677"), .65))
        for end in (-108, 108):
            painter.drawEllipse(QRectF(-44, end - 7, 88, 14))
        painter.restore()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(INK))
        width, height = self.width(), self.height()
        painter.save()
        scale = min((width - 44) / 520, (height - 75) / 270)
        painter.translate(width / 2, height / 2 - 4)
        painter.scale(scale, scale)
        painter.rotate(-12)

        # A quiet recurring contour ties the triptych together without arrows.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#61694c"), 1))
        painter.drawEllipse(QRectF(-252, -72, 504, 144))
        seconds = self.seconds % self.SUPER_CYCLE_SECONDS
        for bed in range(3):
            self._bed(painter, bed, seconds)
        painter.restore()

        font = QFont(font_family("Cascadia Mono", "Consolas", "DejaVu Sans Mono"))
        font.setPixelSize(11)
        painter.setFont(font)
        painter.setPen(QColor("#b7c0a2"))
        # Labels frame the main diagonal; plus marks occupy the other corners.
        labels = QRectF(22, 18, width - 44, height - 36)
        painter.drawText(labels, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                         "OXIDATION / REDUCTION")
        painter.drawText(labels, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                         "CHEMICAL LOOPING")
        painter.setPen(QPen(QColor(ACID), 1))
        for x, y in ((width - 24, 23), (24, height - 23)):
            painter.drawLine(x - 5, y, x + 5, y)
            painter.drawLine(x, y - 5, x, y + 5)
