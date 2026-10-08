"""Run, solver and output selections in three equal columns."""

from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QSpinBox, QVBoxLayout, QWidget,
)

from packed_bed.axial_schemes import SUPPORTED_SCHEMES
from packed_bed.config.models import SolverConfig
from packed_bed.plotting import PLOT_REGISTRY
from packed_bed.reports import REPORT_REGISTRY
from packed_bed.solver_support import DESKTOP_SOLVERS, SOLVER_LABELS, require_desktop_solver
from types import SimpleNamespace
from pydantic import ValidationError
from packed_bed.config.issues import model_issues
from .validation import messages_for, set_field_issue

from .theme import numeric

from .editor_widgets import SelectionList, action_button, display, number


def form_panel(title, *, panel_type=QGroupBox):
    panel = panel_type(title)
    form = QFormLayout(panel)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setVerticalSpacing(10)
    return panel, form


class GeneralPage(QWidget):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        run, form = form_panel("Simulation")
        self.horizon = editor.field(form, ("run", "simulation", "time_horizon_s"), "Time horizon (s)")
        editor.field(form, ("run", "simulation", "reporting_interval_s"), "Reporting interval (s)")
        editor.field(form, ("run", "model", "axial_cells"), "Number of cells", kind="spin", bounds=(3, 1000000))
        for key, label in (("mass_scheme", "Mass scheme"), ("heat_scheme", "Heat scheme")):
            editor.field(form, ("run", "simulation", key), label, options=SUPPORTED_SCHEMES)
        layout.addWidget(run, 1)

        solver, form = form_panel("Solver")
        self.backend = editor.field(form, ("run", "solver", "backend"), "Backend", options=[
            ("Standard", "daetools"), ("Compiled", "compiled"),
        ], default="daetools", binary=True)
        self.solver = editor.field(form, ("run", "solver", "name"), "Linear solver", options=[])
        self.solver_status = QLabel()
        self.solver_status.setWordWrap(True)
        self.solver_status.setProperty("state", "warning")
        form.addRow(self.solver_status)
        for widget, key in ((self.backend, "backend"), (self.solver, "name")):
            widget.currentIndexChanged.disconnect()
            widget.currentIndexChanged.connect(lambda _, widget=widget, key=key: self.select_solver(key, widget.currentData()))
        threads = editor.field(form, ("run", "solver", "threads"), "Number of threads (0 = environment)",
                     kind="spin", bounds=(0, 1024), default=0)
        threads.setToolTip("KLU factorization is serial. This setting still controls other applicable numerical work.")
        editor.field(form, ("run", "solver", "relative_tolerance"), "Relative tolerance")
        self.advanced_button = action_button("Advanced solver settings…", self.advanced, inspection=True)
        form.addRow(self.advanced_button)
        self.cache_status = QLabel("Cache checked when the run starts")
        self.cache_status.setWordWrap(True)
        form.addRow(self.cache_status)
        layout.addWidget(solver, 1)

        output = QWidget()
        output_layout = QVBoxLayout(output)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.reports = SelectionList({key: (key.replace("_", " ").capitalize(), spec.description)
                                      for key, spec in REPORT_REGISTRY.items()}, "reports")
        self.plots = SelectionList({key: (key.replace("_", " ").capitalize(), spec.description
                                         + "\nRequires reports: " + ", ".join(spec.required_reports))
                                    for key, spec in PLOT_REGISTRY.items()}, "plots", show=True)
        for title, widget, key, stretch in (("Requested reports", self.reports, "requested_reports", 162),
                                            ("Requested plots", self.plots, "requested_plots", 100)):
            group = QGroupBox(title)
            group_layout = QVBoxLayout(group)
            group_layout.addWidget(widget)
            output_layout.addWidget(group, stretch)
            widget.changed.connect(lambda values, key=key: editor.put(("run", "outputs", key), values))
        self.plots.show_requested.connect(editor.show_plot)
        layout.addWidget(output, 1)

    def load(self):
        self.update_solver_choices()
        self.reports.set_values(self.editor.get(("run", "outputs", "requested_reports"), []))
        self.plots.set_values(self.editor.get(("run", "outputs", "requested_plots"), []))

    def update_solver_choices(self):
        backend = self.editor.get(("run", "solver", "backend"), "daetools")
        current = self.editor.get(("run", "solver", "name"))
        self.cache_status.setVisible(backend == "compiled")
        supported = DESKTOP_SOLVERS.get(backend, ())
        with QSignalBlocker(self.solver):
            self.solver.clear()
            for name in supported:
                title = SOLVER_LABELS[name]
                reason = ""
                try:
                    require_desktop_solver(SimpleNamespace(run=SimpleNamespace(solver=SimpleNamespace(backend=backend, name=name))))
                except ValueError as exc:
                    reason = str(exc)
                    title += " — runtime unavailable"
                self.solver.addItem(title, name)
                item = self.solver.model().item(self.solver.count() - 1)
                item.setEnabled(not reason)
                item.setToolTip(reason)
            if current not in supported:
                # Retain invalid imports in the field without offering them in the popup.
                title = SOLVER_LABELS.get(current, str(current) if current is not None else "Select…")
                self.solver.addItem(title, current)
                index = self.solver.count() - 1
                item = self.solver.model().item(index)
                item.setEnabled(False)
                item.setToolTip(f"Imported solver {title} is unavailable for this backend. Select a supported solver.")
                self.solver.view().setRowHidden(index, True)
            self.solver.setCurrentIndex(self.solver.findData(current))
        selected = self.solver.model().item(self.solver.currentIndex())
        message = selected.toolTip() if selected and not selected.isEnabled() else ""
        self.solver_status.setText(message)
        self.solver_status.setVisible(bool(message))

    def select_solver(self, key, value):
        editor = self.editor
        if editor.loading or editor.read_only or editor.case is None:
            return
        settings = dict(editor.get(("run", "solver"), {}))
        settings[key] = value
        backend = settings.get("backend", "daetools")
        repairs = {}
        if settings.get("name") not in DESKTOP_SOLVERS.get(backend, ()):
            repairs[("run", "solver", "name")] = "superlu"
        for option in ("scale_residuals", "step_growth_threshold", "nonlinear_refresh_interval", "vector_exponentials", "band_reciprocals"):
            default = SolverConfig.model_fields[option].default
            incompatible = backend != "compiled" or (option == "band_reciprocals" and settings.get("name") != "band")
            if incompatible and settings.get(option, default) != default:
                repairs[("run", "solver", option)] = default
        if backend == "compiled":
            for path in (("run", "simulation", "report_time_derivatives"), ("run", "outputs", "solver_incidence_matrix")):
                if editor.get(path, False):
                    repairs[path] = False
        if repairs:
            changes = "\n".join(f"{path[-1].replace('_', ' ')} → {new}" for path, new in repairs.items())
            answer = QMessageBox.question(self, "Update solver settings", "This selection requires these changes:\n\n" + changes,
                                          QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                                          QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Ok:
                widget = self.backend if key == "backend" else self.solver
                widget.blockSignals(True)
                widget.setCurrentIndex(widget.findData(editor.get(("run", "solver", key))))
                widget.blockSignals(False)
                return
        # Commit the accepted changes before scheduling a single autosave.
        from .inputs import set_value
        for path, new in {("run", "solver", key): value, **repairs}.items():
            set_value(editor.case.documents, path, new)
        for widget, name in ((self.backend, "backend"), (self.solver, "name")):
            widget.blockSignals(True)
            widget.setCurrentIndex(widget.findData(editor.get(("run", "solver", name))))
            widget.blockSignals(False)
        self.update_solver_choices()
        editor.queue_edit()

    def advanced(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Advanced solver settings")
        form = QFormLayout(dialog)
        controls = {}
        for key, label in (
            ("suppress_algebraic_errors", "Suppress algebraic errors"),
            ("concentration_absolute_tolerance", "Concentration absolute tolerance (mol/m³)"),
            ("max_nonlinear_iterations", "Maximum nonlinear iterations"),
            ("nonlinear_convergence_coefficient", "Nonlinear convergence coefficient"),
            ("maximum_order", "Maximum integration order"),
            ("scale_residuals", "Scale residuals (compiled)"),
            ("step_growth_threshold", "Step growth threshold (compiled)"),
            ("nonlinear_refresh_interval", "Nonlinear refresh interval (compiled)"),
            ("vector_exponentials", "Vector exponentials (compiled)"),
            ("band_reciprocals", "Band reciprocals (compiled)"),
        ):
            definition = SolverConfig.model_fields[key]
            value = self.editor.get(("run", "solver", key), definition.default)
            if definition.annotation is bool:
                control = QCheckBox()
                control.setChecked(value is True)
            elif definition.annotation is int:
                control = QSpinBox()
                control.setRange(0 if key == "nonlinear_refresh_interval" else 1,
                                 5 if key == "maximum_order" else 1000000)
                control.setValue(value if isinstance(value, int) else control.minimum())
            else:
                control = numeric(QLineEdit(display(value)))
            control.setAccessibleName(label)
            control.setObjectName(key)
            compiled = self.editor.get(("run", "solver", "backend"), "daetools") == "compiled"
            control.setEnabled(("(compiled)" not in label or compiled) and
                               (key != "band_reciprocals" or self.editor.get(("run", "solver", "name")) == "band"))
            form.addRow(label, control)
            controls[key] = (control, value)

        def validate_fields():
            values = dict(self.editor.get(("run", "solver"), {}))
            for key, (control, _) in controls.items():
                if control.isEnabled():
                    values[key] = (control.isChecked() if isinstance(control, QCheckBox) else
                                   control.value() if isinstance(control, QSpinBox) else number(control.text()))
            try:
                SolverConfig.model_validate(values)
                issues = []
            except ValidationError as exc:
                issues = list(model_issues(exc))
            for key, (control, _) in controls.items():
                set_field_issue(control, messages_for(issues, (key,)))

        for control, _ in controls.values():
            signal = (control.toggled if isinstance(control, QCheckBox) else
                      control.valueChanged if isinstance(control, QSpinBox) else control.textChanged)
            signal.connect(validate_fields)
        validate_fields()
        if self.editor.read_only:
            for control, _ in controls.values():
                if isinstance(control, (QLineEdit, QSpinBox)):
                    control.setReadOnly(True)
                else:
                    control.setEnabled(False)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close if self.editor.read_only else
                                   QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted and not self.editor.read_only:
            for key, (control, original) in controls.items():
                if not control.isEnabled():
                    continue
                value = (control.isChecked() if isinstance(control, QCheckBox) else
                         control.value() if isinstance(control, QSpinBox) else number(control.text()))
                if value != original:
                    self.editor.put(("run", "solver", key), value)
