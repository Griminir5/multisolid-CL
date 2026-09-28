"""Visible, exclusive choices with the value interface used by document bindings."""
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class BinaryChoice(QWidget):
    currentIndexChanged = pyqtSignal(int)
    currentTextChanged = pyqtSignal(str)

    def __init__(self, items=(), parent=None, *, vertical=False):
        super().__init__(parent)
        self._index = -1
        self._model = QStandardItemModel(self)
        self.buttons = []
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(3)
        self.options = QVBoxLayout() if vertical else QHBoxLayout()
        self.options.setSpacing(0)
        outer.addLayout(self.options)
        self.unknown = QLabel()
        self.unknown.setProperty('state', 'error')
        self.unknown.setWordWrap(True)
        self.unknown.hide()
        outer.addWidget(self.unknown)
        for item in items:
            title, value = item if isinstance(item, tuple) else (item, item)
            self.addItem(title, value)
        self.group.idClicked.connect(self.setCurrentIndex)
        self._model.dataChanged.connect(self._sync)
        self.setMinimumHeight(56 if vertical else 28)

    def addItem(self, title, value=None):
        item = QStandardItem(str(title))
        item.setData(value, Qt.ItemDataRole.UserRole)
        index = self.count()
        self._model.appendRow(item)
        # Extra imported values remain visible, but aren't offered as valid choices.
        if index < 2:
            button = QPushButton(str(title))
            button.setCheckable(True)
            button.setAutoDefault(False)
            button.setProperty('segment', True)
            button.setMinimumHeight(28)
            button.installEventFilter(self)
            button.setAccessibleName(str(title))
            self.group.addButton(button, index)
            self.options.addWidget(button)
            self.buttons.append(button)
        if self._index < 0:
            self.setCurrentIndex(0)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyPress and event.key() in (
                Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            direction = -1 if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Up) else 1
            index = self.buttons.index(watched)
            for offset in range(1, len(self.buttons) + 1):
                target = (index + offset * direction) % len(self.buttons)
                if self.buttons[target].isEnabled():
                    self.buttons[target].setFocus()
                    self.setCurrentIndex(target)
                    break
            return True
        return super().eventFilter(watched, event)

    def count(self):
        return self._model.rowCount()

    def model(self):
        return self._model

    def itemData(self, index):
        item = self._model.item(index)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def itemText(self, index):
        item = self._model.item(index)
        return item.text() if item else ''

    def findData(self, value):
        return next((i for i in range(self.count()) if self.itemData(i) == value), -1)

    def findText(self, text):
        return next((i for i in range(self.count()) if self.itemText(i) == text), -1)

    def currentIndex(self):
        return self._index

    def currentData(self):
        return self.itemData(self._index)

    def currentText(self):
        return self.itemText(self._index)

    def setCurrentText(self, text):
        if self.findText(text) < 0:
            self.addItem(text, text)
        self.setCurrentIndex(self.findText(text))

    def setCurrentIndex(self, index):
        if index < -1 or index >= self.count():
            return
        changed = index != self._index
        self._index = index
        self._sync()
        if changed:
            self.currentIndexChanged.emit(index)
            self.currentTextChanged.emit(self.currentText())

    def _sync(self, *_):
        self.group.setExclusive(False)
        for i, button in enumerate(self.buttons):
            button.setChecked(i == self._index)
            button.setEnabled(self._model.item(i).isEnabled())
            button.setToolTip(self._model.item(i).toolTip())
        self.group.setExclusive(True)
        self.unknown.setText('Unsupported value: ' + (self.currentText() or 'Select a value'))
        self.unknown.setVisible(self._index not in (0, 1))
