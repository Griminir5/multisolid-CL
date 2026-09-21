"""Five-tab case authoring, autosave and engine-backed previews."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QAbstractButton, QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QLabel, QLineEdit,
    QMessageBox, QSizePolicy, QSpinBox, QTabWidget, QTreeWidget, QVBoxLayout, QWidget,
)

from packed_bed.plotting import PLOT_REGISTRY
from packed_bed.preview import preview_case
from packed_bed.reports import RESULTS_FILENAME

from .bed import BedPage
from .chemistry import ChemistryPage
from .editor_widgets import choices, display, number, select_value
from .general import GeneralPage
from .program_editor import ProgramPage
from .inputs import get_value, input_readiness, resolve_documents, set_value


@dataclass
class InputSession:
    documents: dict
    metadata: dict
    on_change: object = None

    def save(self):
        if self.on_change:
            self.on_change(self.documents, self.metadata)

    def resolve(self):
        return resolve_documents(self.documents)

    def validate_for_run(self):
        readiness, message = input_readiness(self.documents)
        if readiness != "Ready":
            raise ValueError(message)


class InputEditor(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.case = None
        self.read_only = False
        self.dirty = False
        self.loading = False
        self.bindings = []
        self.fields = {}
        self.plot_windows = []
        self.frozen_controls = []
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(350)
        self.debounce.timeout.connect(self.save)
        self.setObjectName("caseEditor")
        self.setStyleSheet("""
            QGroupBox { font-weight: 600; border: 1px solid palette(mid);
                        border-radius: 5px; margin-top: 12px; padding-top: 9px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLineEdit, QComboBox, QSpinBox { min-height: 25px; }
            QLineEdit:disabled { background: palette(alternate-base); color: palette(mid); }
            QTabBar::tab { min-width: 85px; padding: 9px 15px; }
            QTableWidget, QTreeWidget { border: 0; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.general = GeneralPage(self)
        self.chemistry = ChemistryPage(self)
        self.bed = BedPage(self)
        self.program = ProgramPage(self)
        for title, page in (("General", self.general), ("Chemistry", self.chemistry),
                            ("Bed", self.bed), ("Program", self.program)):
            self.tabs.addTab(page, title)
        layout.addWidget(self.tabs, 1)
        self.validation = QLabel()
        self.validation.setWordWrap(True)
        self.validation.setMaximumHeight(52)
        self.validation.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.validation)
        # These remain useful to callers inspecting the numerical previews.
        self.figures = [self.program.preview.figure, self.bed.preview.figure]
        self.canvases = [self.program.preview.canvas, self.bed.preview.canvas]

    def get(self, path, default=None):
        return get_value(self.case.documents, path, default) if self.case else default

    def put(self, path, value):
        if self.loading or self.read_only or self.case is None:
            return
        try:
            set_value(self.case.documents, path, value)
        except ValueError as exc:
            self.validation.setText(f"Cannot edit {'.'.join(path)}: {exc}")
            return
        self.queue_edit()

    def field(self, form, path, label, *, kind="text", options=None, default=None, bounds=(0, 1000000), checked_values=None):
        if options is not None:
            widget = choices(options)
            signal = widget.currentIndexChanged
            read = widget.currentData
        elif kind == "spin":
            widget = QSpinBox()
            widget.setRange(*bounds)
            signal = widget.valueChanged
            read = widget.value
        elif kind == "check":
            widget = QCheckBox(label)
            signal = widget.toggled
            read = (lambda: checked_values[int(widget.isChecked())]) if checked_values else widget.isChecked
        else:
            widget = QLineEdit()
            signal = widget.textChanged
            read = lambda: number(widget.text().strip())
        widget.setObjectName(".".join(path))
        widget.setAccessibleName(label)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        signal.connect(lambda *_: self.put(path, read()))
        self.bindings.append((widget, path, default, checked_values))
        if path[0] == "run":
            self.fields[(path[1], path[2])] = widget
        if kind == "check":
            form.addRow(widget)
        else:
            form.addRow(label, widget)
        return widget

    def set_documents(self, documents, metadata=None, on_change=None, *, read_only=False):
        return self.set_case(InputSession(deepcopy(documents), deepcopy(metadata or {}), on_change), read_only=read_only)

    def set_case(self, case, *, read_only=False):
        if not self.save():
            return False
        self.restore_controls()
        self.case = case
        self.read_only = read_only
        self.dirty = False
        self.loading = True
        for widget, path, default, checked_values in self.bindings:
            value = self.get(path, default)
            widget.setEnabled(isinstance(self.get(path[:-1], {}), dict))
            if isinstance(widget, QComboBox):
                select_value(widget, value)
            elif isinstance(widget, QSpinBox):
                widget.setValue(value if type(value) is int else widget.minimum())
                # Do not silently replace out-of-range imported draft values with the displayed limit.
                if type(value) is not int or not widget.minimum() <= value <= widget.maximum():
                    widget.lineEdit().setText(display(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(value == checked_values[1] if checked_values else value is True)
            else:
                widget.setText(display(value))
        self.general.load()
        self.chemistry.load()
        self.bed.load()
        self.program.load()
        self.loading = False
        if self.read_only:
            self.freeze_controls()
        self.tabs.setCurrentIndex(0)
        self.refresh()
        if self.dirty:
            self.debounce.start()
        return True

    def restore_controls(self):
        for restore, value in reversed(self.frozen_controls):
            try:
                restore(value)
            except RuntimeError:
                pass  # A page rebuild already discarded this control.
        self.frozen_controls.clear()

    def freeze_controls(self):
        def freeze(setter, original, value):
            self.frozen_controls.append((setter, original))
            setter(value)

        def freeze_item(item):
            freeze(item.setFlags, item.flags(), item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable & ~Qt.ItemFlag.ItemIsEditable)
            for i in range(item.childCount()):
                freeze_item(item.child(i))
        for page in (self.general, self.chemistry, self.bed, self.program):
            for widget in page.findChildren(QWidget):
                if any(preview.isAncestorOf(widget) for preview in (self.bed.preview, self.program.preview)):
                    continue  # Plot navigation remains interactive during inspection.
                if widget in (channel.toggle for channel in self.program.channels.values()):
                    continue  # Collapsing channels only changes the layout.
                if isinstance(widget, (QLineEdit, QSpinBox)):
                    freeze(widget.setReadOnly, widget.isReadOnly(), True)
                elif isinstance(widget, (QComboBox, QAbstractButton)):
                    freeze(widget.setEnabled, widget.isEnabled(), False)
                if isinstance(widget, QAbstractItemView):
                    freeze(widget.setEditTriggers, widget.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
                if isinstance(widget, QTreeWidget):
                    for i in range(widget.topLevelItemCount()):
                        freeze_item(widget.topLevelItem(i))

    def queue_edit(self):
        if self.loading or self.read_only or self.case is None:
            return
        self.dirty = True
        self.debounce.start()
        self.validation.setText("Saving changes…")
        for preview in (self.program.preview, self.bed.preview):
            preview.message.setText("Updating preview…")
            preview.message.show()
        self.changed.emit()

    def save(self):
        self.debounce.stop()
        if self.read_only or self.case is None or not self.dirty:
            return True
        # The reactor endpoints remain tied to its geometry, even during invalid edits.
        self.bed.anchor_zones()
        try:
            self.case.save()
        except (OSError, ValueError) as exc:
            self.validation.setText(f"Draft could not be saved: {exc}")
            return False
        self.dirty = False
        self.refresh()
        self.changed.emit()
        return True

    def set_species(self, phase, values):
        if self.read_only:
            return
        path = ("chemistry", "gas_species") if phase == "gas" else ("solids", "solid_species")
        previous = self.get(path, [])
        removed, added = set(previous) - set(values), set(values) - set(previous)
        self.put(path, list(values))
        def update_composition(composition):
            if not isinstance(composition, dict):
                return
            for key in removed:
                composition.pop(key, None)
            for key in added:
                composition[key] = ""
        if phase == "solid":
            for zone in self.get(("solids", "initial_profile", "zones"), []):
                update_composition(zone.setdefault("values", {}))
            self.bed.load_zones()
        else:
            def update_program(program):
                for key in ("inlet_composition", "feed_stream"):
                    channel = program.get(key, {})
                    for index, state in enumerate([channel, *channel.get("steps", [])]):
                        name = "initial" if index == 0 else "target"
                        value = state.get(name)
                        if key == "feed_stream":
                            value = value.get("composition") if isinstance(value, dict) else None
                        update_composition(value)
            update_program(self.get(("program",), {}))
            for program in self.case.metadata.get("program_modes", {}).values():
                update_program(program)
            self.program.load()
        self.chemistry.load()
        self.queue_edit()

    def refresh(self):
        if self.case is None:
            return
        for preview in (self.program.preview, self.bed.preview):
            preview.clear()
        try:
            case = self.case.resolve()
            self._draw_preview(case, preview_case(case))
            self.case.validate_for_run()
        except (ValueError, OSError) as exc:
            message = str(exc)
            self.validation.setText("Draft — cannot run: " + message.splitlines()[0] + " · hover for details")
            self.validation.setToolTip(message)
            if not self.figures[0].axes:
                for preview in (self.program.preview, self.bed.preview):
                    preview.clear("Preview unavailable while inputs are incomplete or invalid. See the validation message below.")
        else:
            self.validation.setText("Inputs ready · read-only" if self.read_only else "Inputs ready · changes save automatically")
            self.validation.setToolTip("Structural checks passed. Previews use the engine's smoothing, feed mixing and cycle carry-over.")
        self.update_results()

    def update_results(self):
        self.general.plots.set_results_available(False)

    def show_plot(self, plot_id):
        pass

    def _draw_preview(self, case, preview):
        axes = self.figures[0].subplots(4, 1, sharex=True)
        for axis, values, label in zip((axes[0], axes[1], axes[3]), (
            preview.flow_mol_s, preview.temperature_k, preview.pressure_pa,
        ), ("Inlet flow\n(mol/s)", "Temperature\n(K)", "Outlet pressure\n(Pa)")):
            axis.plot(preview.time_s, values, color="#317c89")
            axis.set_ylabel(label, fontsize=9)
        axes[2].plot(preview.time_s, preview.mole_fractions, label=case.chemistry.gas_species)
        axes[2].legend(loc="upper right", fontsize=8, ncols=3)
        axes[2].set(ylabel="Mole fraction", ylim=(-0.02, 1.02))
        axes[3].set_xlabel("Time (s)")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.tick_params(labelsize=8)
        axes = self.figures[1].subplots(1, 3)
        for name, values in zip(case.solids.solid_species, preview.solid_concentrations_mol_m3_bed):
            axes[0].stairs(values, preview.face_positions_m, label=name)
        axes[0].set(ylabel="Concentration (mol/m³ bed)", title="Solid concentrations")
        if case.solids.solid_species:
            axes[0].legend(fontsize=8)
        axes[1].stairs(preview.interparticle_voidage, preview.face_positions_m, label="Interparticle")
        axes[1].stairs(preview.particle_voidage, preview.face_positions_m, label="Particle")
        axes[1].set(ylabel="Voidage (fraction)", ylim=(0, 1), title="Voidages")
        axes[1].legend(fontsize=8)
        axes[2].plot(preview.face_positions_m, preview.particle_diameter_m, marker=".")
        axes[2].set(ylabel="Particle diameter (m)", title="Particle size")
        for axis in axes:
            axis.set_xlabel("Axial position (m)")
            for boundary in preview.zone_edges_m:
                axis.axvline(boundary, color="grey", linestyle="--", alpha=0.4)
            axis.grid(alpha=0.2)
            axis.tick_params(labelsize=8)
        for widget in (self.program.preview, self.bed.preview):
            widget.draw()


class CaseEditor(InputEditor):
    """Project persistence and retained results around the shared input controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from .report import ReportPage
        self.report = ReportPage(self)
        self.tabs.addTab(self.report, "Report")

    def set_case(self, case):
        if not super().set_case(case, read_only=bool(case.metadata.get("study_id"))):
            return False
        self.report.set_case(case)
        return True

    def queue_edit(self):
        super().queue_edit()
        if self.dirty:
            self.checkpoint()

    def checkpoint(self):
        if self.case is None:
            return
        from .recovery import case_payload
        try:
            if self.dirty or self.report.dirty:
                if self.dirty:
                    self.bed.anchor_zones()
                report = self.report.definition if self.report.case is self.case and self.report.dirty else None
                self.case.project.drafts.write("case", self.case.id, case_payload(self.case, report))
            else:
                self.case.project.drafts.clear("case", self.case.id)
        except (OSError, ValueError) as exc:
            self.validation.setText(f"Recovery copy could not be saved: {exc}")

    def save(self):
        if not super().save():
            return False
        if self.report.needs_refresh and self.report.case is self.case:
            self.report.refresh_source()
        if not self.report.save():
            return False
        self.checkpoint()
        return True

    def update_results(self):
        if self.case is None:
            return
        available = (self.case.run_folder / "output" / RESULTS_FILENAME).is_file()
        self.general.plots.set_results_available(available)
        if self.report.case is self.case:
            self.report.invalidate()

    def show_plot(self, plot_id):
        if self.case is None or plot_id not in PLOT_REGISTRY:
            return
        path = self.case.run_folder / "output" / RESULTS_FILENAME
        if not path.is_file():
            return
        try:
            # Read retained data independently of the current draft, including stale results.
            import xarray as xr
            with xr.open_dataset(path, engine="scipy") as source:
                dataset = source.load()
            spec = PLOT_REGISTRY[plot_id]
            with TemporaryDirectory(prefix="multisolid-plot-") as folder:
                destination = Path(folder) / spec.filename
                spec.render(dataset, destination)
                svg = destination.read_bytes()
        except Exception as exc:
            QMessageBox.warning(self, "Plot unavailable", f"This plot could not be drawn from the retained results.\n{exc}")
            return
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle(f"{self.case.name} · {plot_id.replace('_', ' ')}")
        dialog.resize(950, 720)
        layout = QVBoxLayout(dialog)
        from .case_list import result_label
        status = QLabel("Retained run · " + result_label(self.case.state()))
        layout.addWidget(status)
        plot = QSvgWidget()
        plot.load(svg)
        layout.addWidget(plot, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.close)
        layout.addWidget(buttons)
        self.plot_windows.append(dialog)
        dialog.destroyed.connect(lambda: self.plot_windows.remove(dialog))
        dialog.show()
