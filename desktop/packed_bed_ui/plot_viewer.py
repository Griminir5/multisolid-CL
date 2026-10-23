"""Cancellable plot windows; data loading and Matplotlib run in a subprocess."""

from pathlib import Path
import sys

from PyQt6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt, pyqtSignal
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLabel, QProgressBar, QVBoxLayout

from packed_bed.file_io import TemporaryDirectory
from packed_bed.plotting import PLOT_REGISTRY


def result_revision(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


class PlotProcess(QProcess):
    completed = pyqtSignal(bytes, str)

    def __init__(self, path, plot_id):
        # Keep the process alive until it has exited, even if its view closes.
        super().__init__(QApplication.instance())
        self.path = Path(path)
        self.revision = result_revision(self.path)
        self.cancelled = self.done = False
        self.temporary = TemporaryDirectory(prefix="multisolid-plot-")
        self.destination = Path(self.temporary.name) / PLOT_REGISTRY[plot_id].filename
        environment = QProcessEnvironment.systemEnvironment()
        # Large BLAS thread pools slow these decompositions and compete with
        # the GUI/simulation. This changes only the isolated plotting process.
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            environment.insert(name, "1")
        self.setProcessEnvironment(environment)
        self.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        args = ["--render-plot", str(self.path), plot_id, str(self.destination)]
        if not getattr(sys, "frozen", False):
            args = ["-m", "packed_bed_ui", *args]
        self.setProgram(sys.executable)
        self.setArguments(args)
        self.finished.connect(self._finished)
        self.errorOccurred.connect(self._error)
        QApplication.instance().aboutToQuit.connect(self._shutdown)
        QApplication.instance().destroyed.connect(self._shutdown)

    def launch(self):
        if not self.cancelled and not self.done:
            self.start()

    def cancel(self):
        self.cancelled = True
        if self.state() != QProcess.ProcessState.NotRunning:
            self.kill()
        else:
            self._complete()

    def _shutdown(self):
        self.cancel()
        if self.state() != QProcess.ProcessState.NotRunning:
            self.waitForFinished(1000)

    def _error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._complete(error=f"Could not start plot preparation: {self.errorString()}")

    def _finished(self, code, status):
        svg, error = b"", ""
        if not self.cancelled:
            if code or status != QProcess.ExitStatus.NormalExit:
                detail = bytes(self.readAllStandardOutput()).decode("utf-8", errors="replace").strip()[-4000:]
                error = detail or "Plot preparation failed."
            else:
                try:
                    if result_revision(self.path) != self.revision:
                        raise ValueError("Results changed while preparing this plot. Close it and click Show again.")
                    svg = self.destination.read_bytes()
                except (OSError, ValueError) as exc:
                    error = str(exc)
        self._complete(svg, error)

    def _complete(self, svg=b"", error=""):
        if self.done:
            return
        self.done = True
        try:
            self.temporary.cleanup()
        except OSError:
            pass  # A temporary-file cleanup failure must not hide the plot.
        if not self.cancelled:
            self.completed.emit(svg, error)
        self.deleteLater()


class PlotDialog(QDialog):
    def __init__(self, path, plot_id, case_name, status, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"{case_name} · {PLOT_REGISTRY[plot_id].label}")
        self.resize(950, 720)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Retained run · " + status))
        self.message = QLabel("Preparing plot… You can keep working or close this window to cancel.")
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        layout.addWidget(self.progress)
        self.plot = QSvgWidget()
        self.plot.hide()
        layout.addWidget(self.plot, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.process = PlotProcess(path, plot_id)
        self.request = (Path(path), self.process.revision, plot_id)
        self.process.completed.connect(self._rendered)
        self.finished.connect(self.process.cancel)
        self.destroyed.connect(self.process.cancel)
        QTimer.singleShot(0, self.process.launch)

    def _rendered(self, svg, error):
        self.progress.hide()
        if not error:
            self.plot.load(svg)
            if not self.plot.renderer().isValid():
                error = "Plot preparation returned an invalid SVG."
        if error:
            self.message.setText("This plot could not be drawn from the retained results.\n" + error)
        else:
            self.message.hide()
            self.plot.show()
