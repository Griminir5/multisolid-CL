"""The selected case's small form and previews, separate from project navigation."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QFormLayout, QLabel, QLineEdit, QTabWidget, QVBoxLayout, QWidget

from packed_bed.preview import preview_case


class CaseEditor(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.case = None
        self.dirty = False
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(350)
        self.debounce.timeout.connect(self.save)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        form = QFormLayout()
        self.fields = {}
        for section, key, label in (
            ("simulation", "time_horizon_s", "Run duration (s)"),
            ("simulation", "reporting_interval_s", "Recording interval (s)"),
            ("model", "axial_cells", "Number of axial cells"),
            ("solver", "threads", "Numerical threads (0 = environment)"),
        ):
            field = QLineEdit()
            field.textEdited.connect(self.queue_edit)
            self.fields[(section, key)] = field
            form.addRow(label, field)
        layout.addLayout(form)
        self.validation = QLabel()
        self.validation.setWordWrap(True)
        layout.addWidget(self.validation)
        self.tabs = QTabWidget()
        self.figures, self.canvases = [], []
        for title in ("Operating program", "Bed"):
            page = QWidget()
            page_layout = QVBoxLayout(page)
            figure = Figure(layout="constrained")
            canvas = FigureCanvasQTAgg(figure)
            page_layout.addWidget(NavigationToolbar2QT(canvas, page))
            page_layout.addWidget(canvas)
            self.tabs.addTab(page, title)
            self.figures.append(figure)
            self.canvases.append(canvas)
        layout.addWidget(self.tabs)

    def set_case(self, case):
        if not self.save():
            return False
        self.case = case
        for (section, key), field in self.fields.items():
            values = case.documents["run"].get(section, {})
            field.setText(str(values.get(key, 0 if key == "threads" else "")) if isinstance(values, dict) else "")
        self.refresh()
        return True

    def queue_edit(self):
        self.dirty = True
        self.debounce.start()
        self.changed.emit()

    def save(self):
        self.debounce.stop()
        if self.case is None or not self.dirty:
            return True
        for (section, key), field in self.fields.items():
            text = field.text().strip()
            try:
                value = int(text) if key in ("axial_cells", "threads") else float(text)
            except ValueError:
                value = text
            values = self.case.documents["run"].setdefault(section, {})
            if not isinstance(values, dict):
                self.validation.setText(f"Cannot edit: run.{section} must be a mapping.")
                return False
            values[key] = value
        try:
            self.case.save()
        except OSError as exc:
            self.validation.setText(f"Draft could not be saved: {exc}")
            return False
        self.dirty = False
        self.refresh()
        self.changed.emit()
        return True

    def refresh(self):
        for figure in self.figures:
            figure.clear()
        for (section, _), field in self.fields.items():
            field.setEnabled(isinstance(self.case.documents["run"].get(section, {}), dict))
        self.summary.setText(self.case.name)
        try:
            case = self.case.resolve()
            self._draw_preview(case, preview_case(case))
            self.summary.setText(
                f"{self.case.name} · {case.run.simulation.program_mode.replace('_', ' ')}\n"
                f"Bed length {case.run.model.bed_length_m:g} m · Radius {case.run.model.bed_radius_m:g} m\n"
                f"Gases: {', '.join(case.chemistry.gas_species)} · Solids: {', '.join(case.solids.solid_species)}"
            )
            self.case.validate_for_run()
        except (ValueError, OSError) as exc:
            self.validation.setText(f"Draft — cannot run: {exc}")
        else:
            self.validation.setText(
                "Ready: structural checks passed; scientific suitability still needs assessment.\n"
                "Previews include smoothing and cycle carry-over. GHSV uses 273.15 K and 100,000 Pa."
            )
        for canvas in self.canvases:
            canvas.draw_idle()
    def _draw_preview(self, case, preview):
        axes = self.figures[0].subplots(4, 1, sharex=True)
        for axis, values, label in zip(axes, (
            preview.flow_mol_s, preview.temperature_k, preview.pressure_pa,
        ), ("Inlet flow\n(mol/s)", "Inlet\ntemperature\n(K)", "Outlet\npressure\n(Pa)")):
            axis.plot(preview.time_s, values)
            axis.set_ylabel(label)
        axes[3].plot(preview.time_s, preview.mole_fractions, label=case.chemistry.gas_species)
        axes[3].legend(loc="upper right", fontsize="small", ncols=3)
        axes[3].set(ylabel="Mole fraction", xlabel="Time (s)", ylim=(-0.02, 1.02))
        for axis in axes:
            axis.grid(alpha=0.25)
        axes = self.figures[1].subplots(3, 1, sharex=True)
        for name, values in zip(case.solids.solid_species, preview.solid_concentrations_mol_m3_bed):
            axes[0].stairs(values, preview.face_positions_m, label=name)
        axes[0].set_ylabel("Concentration\n(mol/m³ bed)")
        if case.solids.solid_species:
            axes[0].legend(fontsize="small", ncols=3)
        axes[1].stairs(preview.interparticle_voidage, preview.face_positions_m, label="Interparticle")
        axes[1].stairs(preview.particle_voidage, preview.face_positions_m, label="Particle")
        axes[1].set(ylabel="Voidage (fraction)", ylim=(0, 1))
        axes[1].legend(fontsize="small")
        axes[2].plot(preview.face_positions_m, preview.particle_diameter_m, marker=".")
        axes[2].set(ylabel="Particle diameter (m)", xlabel="Axial position (m)")
        axes[0].set_title("Grid-sampled profiles; dashed lines show authored zone boundaries")
        for axis in axes:
            for boundary in preview.zone_edges_m:
                axis.axvline(boundary, color="grey", linestyle="--", alpha=0.4)
            axis.grid(alpha=0.25)
