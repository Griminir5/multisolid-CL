"""Study validation and staging workers; Qt widgets stay on the GUI thread."""

from time import perf_counter

from PyQt6.QtCore import QThread, pyqtSignal

from .project import Project


class PreviewTask(QThread):
    batch = pyqtSignal(object, object)
    completed = pyqtSignal(object, str)

    def __init__(self, preview, parent=None):
        super().__init__(parent)
        self.preview = preview

    def run(self):
        batch, started = [], perf_counter()
        try:
            for candidate in self.preview.remaining:
                if self.isInterruptionRequested():
                    return
                batch.append(candidate)
                if len(batch) >= 20 or perf_counter() - started >= .05:
                    self.batch.emit(self.preview, batch)
                    batch, started = [], perf_counter()
            if batch and not self.isInterruptionRequested():
                self.batch.emit(self.preview, batch)
            if not self.isInterruptionRequested():
                self.completed.emit(self.preview, "")
        except Exception as exc:
            self.completed.emit(self.preview, str(exc))


class GenerationTask(QThread):
    progress = pyqtSignal(str)
    completed = pyqtSignal(object, str)

    def __init__(self, root, preview, parent=None):
        super().__init__(parent)
        self.root, self.preview = root, preview

    def run(self):
        try:
            # This project has no GUI callbacks or shared mutable case objects.
            project = Project.open(self.root)
            project.study_store.apply_preview(self.preview, cancelled=self.isInterruptionRequested,
                                               progress=self.progress.emit)
            self.completed.emit(project, "")
        except Exception as exc:
            self.completed.emit(None, str(exc))
