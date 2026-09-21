"""Hold/ramp program tables with initial states and structured feed targets."""

from copy import deepcopy
from uuid import uuid4
import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

from packed_bed.programs import NORMAL_MOLAR_DENSITY_MOL_PER_M3

from .editor_widgets import Preview, action_button, cell, choices, display, number, select_value, table


CHANNELS = ("inlet_flow", "inlet_temperature", "inlet_composition", "outlet_pressure")


from .inputs import program_duration, ensure_step_ids


class ChannelTable(QGroupBox):
    def __init__(self, page, key, title):
        super().__init__()
        self.page, self.editor, self.key = page, page.editor, key
        self.loading = False
        self.setStyleSheet("QGroupBox { margin-top: 0; padding-top: 0; }")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.toggle.setStyleSheet("QToolButton { font-weight: 600; border: none; text-align: left; }")
        self.toggle.setToolTip("Collapse channel")
        layout.addWidget(self.toggle)
        self.table = table(["Step", "Duration (s)", "Target", ""])
        self.table.setMinimumHeight(72)
        self.table.verticalHeader().setDefaultSectionSize(29)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 82)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 96)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(3, 32)
        self.table.itemChanged.connect(self.edit_item)
        layout.addWidget(self.table)
        self.toggle.toggled.connect(self.set_expanded)

    def set_expanded(self, expanded):
        self.table.setVisible(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle.setToolTip("Collapse channel" if expanded else "Expand channel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding if expanded else QSizePolicy.Policy.Fixed)
        self.page.update_channel_layout()

    def channel(self):
        return self.editor.get(("program", self.key), {})

    def load(self):
        self.loading = True
        channel = self.channel()
        steps = channel.get("steps", [])
        self.table.clearSpans()
        self.table.clearContents()
        self.table.setRowCount(len(steps) + 2)
        cell(self.table, 0, 0, "Initial", editable=False)
        cell(self.table, 0, 1, "—", editable=False)
        self.target_cell(0, channel.get("initial", ""))
        for index, step in enumerate(steps):
            row = index + 1
            kind = choices([("Hold", "hold"), ("Ramp", "ramp")])
            select_value(kind, step.get("kind"))
            kind.currentIndexChanged.connect(lambda _, index=index, kind=kind: self.change_kind(index, kind.currentData()))
            self.table.setCellWidget(row, 0, kind)
            cell(self.table, row, 1, step.get("duration_s", ""))
            self.target_cell(row, step.get("target", ""), hold=step.get("kind") == "hold")
            self.table.setCellWidget(row, 3, action_button("×", lambda _, index=index: self.remove_step(index),
                                                         tooltip=f"Remove step {row} from {self.toggle.text()}"))
        self.table.setSpan(len(steps) + 1, 0, 1, 4)
        self.table.setCellWidget(len(steps) + 1, 0, action_button("+ Add step", self.add_step))
        self.loading = False
        self.update_title()

    def update_title(self):
        flow_unit = "h⁻¹" if self.page.flow_basis.currentData() == "ghsv_per_h" else "mol/s"
        title, unit = {"inlet_flow": ("Inlet flow", flow_unit), "inlet_temperature": ("Inlet temperature", "K"),
                       "inlet_composition": ("Inlet composition", "mole fractions"), "outlet_pressure": ("Outlet pressure", "Pa"),
                       "feed_stream": ("Feed", f"{flow_unit}, K, mole fractions")}[self.key]
        duration = program_duration({self.key: self.channel()})
        self.toggle.setText(title + (f" · {duration:g} s" if duration is not None else " · unfinished timing"))
        self.table.horizontalHeaderItem(2).setText(f"Target ({unit})")

    def target_cell(self, row, value, hold=False):
        self.table.removeCellWidget(row, 2)
        if hold:
            cell(self.table, row, 2, "— retain previous value", editable=False)
        elif self.key in ("inlet_composition", "feed_stream"):
            value = value if isinstance(value, dict) else {}
            composition = value.get("composition", {}) if self.key == "feed_stream" else value
            try:
                summary = f"Σ {math.fsum(float(v) for v in composition.values()):g}" if composition else "retained composition" if row else "Set composition"
            except (ValueError, TypeError):
                summary = "Unfinished composition"
            if self.key == "feed_stream":
                summary = f"{display(value.get('flow', 'retain'))} · {display(value.get('temperature', 'retain'))} · {summary}"
            button = action_button(summary + "  …", lambda _, row=row: self.edit_state(row),
                                   tooltip="Edit feed values" if self.key == "feed_stream" else "Edit species mole fractions")
            self.table.setCellWidget(row, 2, button)
        else:
            cell(self.table, row, 2, value)

    def write(self, channel, *, rebuild=True):
        self.editor.put(("program", self.key), channel)
        self.page.update_horizon()
        if rebuild:
            self.load()
        else:
            self.update_title()

    def edit_item(self, item):
        if self.loading or item.column() not in (1, 2):
            return
        channel = deepcopy(self.channel())
        row, col = item.row(), item.column()
        if row == 0:
            if col != 2:
                return
            channel["initial"] = number(item.text())
        else:
            channel["steps"][row - 1]["duration_s" if col == 1 else "target"] = number(item.text())
        self.write(channel, rebuild=False)

    def add_step(self):
        if self.page.editor.read_only:
            return
        ids = ensure_step_ids(self.page.editor.case.documents["program"], self.page.editor.case.metadata)
        ids.setdefault(self.key, []).append(uuid4().hex)
        channel = deepcopy(self.channel())
        channel.setdefault("steps", []).append({"kind": "hold", "duration_s": ""})
        self.write(channel)
        self.table.scrollToItem(self.table.item(len(channel["steps"]), 1))
        self.table.editItem(self.table.item(len(channel["steps"]), 1))

    def remove_step(self, index):
        if self.page.editor.read_only:
            return
        ids = ensure_step_ids(self.page.editor.case.documents["program"], self.page.editor.case.metadata)
        ids[self.key].pop(index)
        channel = deepcopy(self.channel())
        channel["steps"].pop(index)
        self.write(channel)

    def change_kind(self, index, kind):
        channel = deepcopy(self.channel())
        step = channel["steps"][index]
        step["kind"] = kind
        if kind == "hold":
            step.pop("target", None)
        else:
            step["target"] = ({} if self.key == "feed_stream" else
                              {key: "" for key in self.editor.get(("chemistry", "gas_species"), [])}
                              if self.key == "inlet_composition" else "")
        self.write(channel)

    def edit_state(self, row):
        channel = deepcopy(self.channel())
        original = channel.get("initial", {}) if row == 0 else channel["steps"][row - 1].get("target", {})
        original = original if isinstance(original, dict) else {}
        feed, optional = self.key == "feed_stream", self.key == "feed_stream" and row > 0
        dialog = QDialog(self)
        dialog.setWindowTitle(("Initial " if row == 0 else f"Step {row} target · ") + ("feed" if feed else "composition"))
        dialog.resize(440, 480)
        outer = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QFormLayout(content)
        fields, enabled = {}, {}
        if optional:
            label = QLabel("Unticked values carry over from the preceding feed.")
            label.setWordWrap(True)
            form.addRow(label)
        if feed:
            for key, label in (("flow", "Flow (h⁻¹)" if self.page.flow_basis.currentData() == "ghsv_per_h" else "Flow (mol/s)"),
                               ("temperature", "Temperature (K)")):
                field = QLineEdit(display(original.get(key)))
                field.setObjectName(key)
                fields[key] = field
                if optional:
                    check = QCheckBox(label)
                    check.setChecked(key in original and original[key] is not None)
                    field.setEnabled(check.isChecked())
                    check.toggled.connect(field.setEnabled)
                    enabled[key] = check
                    form.addRow(check, field)
                else:
                    form.addRow(label, field)
        composition = original.get("composition", {}) if feed else original
        composition = composition if isinstance(composition, dict) else {}
        composition_enabled = QCheckBox("Set composition")
        composition_enabled.setChecked(not optional or "composition" in original and original["composition"] is not None)
        if optional:
            form.addRow(composition_enabled)
        species = self.editor.get(("chemistry", "gas_species"), [])
        composition_fields = {}
        total = QLabel()
        def update_total():
            try:
                value = math.fsum(float(field.text()) for field in composition_fields.values())
                valid = math.isclose(value, 1.0, rel_tol=0, abs_tol=1e-12)
                total.setText(f"Total: {value:g}" + ("" if valid else " · must sum to 1"))
            except ValueError:
                total.setText("Total: unfinished composition")
        for key in species:
            field = QLineEdit(display(composition.get(key)))
            field.setObjectName(key)
            field.setEnabled(composition_enabled.isChecked())
            composition_enabled.toggled.connect(field.setEnabled)
            composition_fields[key] = field
            field.textChanged.connect(update_total)
            form.addRow(key, field)
        form.addRow(total)
        update_total()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        outer.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        # Preserve unexpected imported keys for validation, unless explicitly editing those values.
        values = {**composition, **{key: number(field.text()) for key, field in composition_fields.items()}}
        if feed:
            value = deepcopy(original)
            for key, field in fields.items():
                if not optional or enabled[key].isChecked():
                    value[key] = number(field.text())
                else:
                    value.pop(key, None)
            if composition_enabled.isChecked():
                value["composition"] = values
            else:
                value.pop("composition", None)
        else:
            value = values
        if row == 0:
            channel["initial"] = value
        else:
            channel["steps"][row - 1]["target"] = value
        self.write(channel)


class ProgramPage(QWidget):
    def __init__(self, editor):
        super().__init__()
        self.editor, self.loading = editor, False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.preview = Preview()
        layout.addWidget(self.preview, 1)
        right = QWidget()
        controls = QVBoxLayout(right)
        controls.setContentsMargins(0, 0, 0, 0)
        options = QGroupBox("Program options")
        form = QGridLayout(options)
        self.flow_basis = choices([("mol/s", "mol_per_s"), ("GHSV (h⁻¹)", "ghsv_per_h")])
        self.mode = choices([("Independent channels", "separate_channels"), ("Feed", "feed_stream")])
        self.repeat = QCheckBox("Repeat program")
        form.addWidget(QLabel("Flow basis"), 0, 0)
        form.addWidget(QLabel("Mode"), 0, 1)
        form.addWidget(self.flow_basis, 1, 0)
        form.addWidget(self.mode, 1, 1)
        form.addWidget(self.repeat, 1, 2)
        self.note = QLabel("GHSV reference: 273.15 K, 100,000 Pa. Each mode keeps its own inlet steps.")
        self.note.setWordWrap(True)
        form.addWidget(self.note, 2, 0, 1, 3)
        controls.addWidget(options)
        self.channel_layout = QVBoxLayout()
        self.channels = {}
        for key in (*CHANNELS[:-1], "feed_stream", "outlet_pressure"):
            self.channels[key] = ChannelTable(self, key, key)
            self.channel_layout.addWidget(self.channels[key], 1)
        self.channel_layout.addStretch(0)
        controls.addLayout(self.channel_layout, 1)
        self.timing = QLabel()
        self.timing.setWordWrap(True)
        controls.addWidget(self.timing)
        layout.addWidget(right, 1)
        self.mode.currentIndexChanged.connect(self.change_mode)
        self.flow_basis.currentIndexChanged.connect(self.change_basis)
        self.repeat.toggled.connect(self.change_repeat)

    def update_channel_layout(self):
        active = ("feed_stream", "outlet_pressure") if self.mode.currentData() == "feed_stream" else CHANNELS
        expanded = False
        for index, (key, channel) in enumerate(self.channels.items()):
            channel.setVisible(key in active)
            stretch = (3 if key == "feed_stream" else 1) if key in active and channel.toggle.isChecked() else 0
            self.channel_layout.setStretch(index, stretch)
            expanded = expanded or bool(stretch)
        self.channel_layout.setStretch(len(self.channels), 0 if expanded else 1)

    def load(self):
        self.loading = True
        mode = self.editor.get(("run", "simulation", "program_mode"), "separate_channels")
        select_value(self.mode, mode)
        key = "feed_stream" if mode == "feed_stream" else "inlet_flow"
        select_value(self.flow_basis, self.editor.get(("program", key, "basis"), "mol_per_s"))
        self.repeat.setChecked(self.editor.get(("run", "simulation", "repeat_program"), False) is True)
        for channel in self.channels.values():
            channel.load()
        self.update_channel_layout()
        self.loading = False
        self.update_horizon(update_documents=False)

    def change_repeat(self, checked):
        if self.loading or self.editor.read_only:
            return
        self.editor.put(("run", "simulation", "repeat_program"), checked)
        self.update_horizon()

    def update_horizon(self, *, update_documents=True):
        if self.editor.case is None:
            return
        repeat = self.editor.get(("run", "simulation", "repeat_program"), False)
        field = self.editor.general.horizon
        field.setEnabled(bool(repeat))
        field.setToolTip("Duration calculated from the longest channel. Non-empty channels must have matching durations."
                         if not repeat else "Total simulation time across repeating cycles.")
        duration = program_duration(self.editor.get(("program",), {}))
        if not repeat:
            value = duration if duration is not None else ""
            if update_documents and not self.editor.loading and value != self.editor.get(("run", "simulation", "time_horizon_s")):
                self.editor.put(("run", "simulation", "time_horizon_s"), value)
            if not update_documents or self.editor.read_only:
                value = self.editor.get(("run", "simulation", "time_horizon_s"), "")
            field.blockSignals(True)
            field.setText(display(value))
            field.blockSignals(False)
        timings = {key: program_duration({key: self.channels[key].channel()})
                   for key in (("feed_stream", "outlet_pressure") if self.mode.currentData() == "feed_stream" else CHANNELS)}
        mismatched = not repeat and duration is not None and any(
            value not in (None, 0) and not math.isclose(value, duration, rel_tol=0, abs_tol=1e-9)
            for value in timings.values())
        self.timing.setText("Channel durations must match for a non-repeating program." if mismatched else
                            "Enter a positive duration for every step." if duration is None else
                            "Add a timed hold or ramp to define the non-repeating horizon." if not repeat and duration == 0 else "")
        self.timing.setVisible(bool(self.timing.text()))

    def change_mode(self):
        if self.loading or self.editor.read_only or self.editor.case is None:
            return
        mode = self.mode.currentData()
        previous = self.editor.get(("run", "simulation", "program_mode"), "separate_channels")
        if mode == previous:
            return
        program = deepcopy(self.editor.get(("program",), {}))
        saved = self.editor.case.metadata.setdefault("program_modes", {})
        mode_ids = self.editor.case.metadata.setdefault("program_step_ids", {})
        mode_ids[previous] = deepcopy(ensure_step_ids(program, self.editor.case.metadata))
        saved[previous] = {key: value for key, value in program.items() if key != "outlet_pressure"}
        if mode in saved:
            replacement = deepcopy(saved[mode])
        elif mode == "feed_stream":
            replacement = {"feed_stream": {"basis": program.get("inlet_flow", {}).get("basis", "mol_per_s"), "initial": {
                "flow": program.get("inlet_flow", {}).get("initial", ""),
                "temperature": program.get("inlet_temperature", {}).get("initial", ""),
                "composition": program.get("inlet_composition", {}).get("initial", {}),
            }, "steps": []}}
        else:
            feed = program.get("feed_stream", {})
            initial = feed.get("initial", {})
            replacement = {key: {"initial": deepcopy(initial.get(value, {} if value == "composition" else "")), "steps": []}
                           for key, value in (("inlet_flow", "flow"), ("inlet_temperature", "temperature"), ("inlet_composition", "composition"))}
            replacement["inlet_flow"]["basis"] = feed.get("basis", "mol_per_s")
        replacement["outlet_pressure"] = program.get("outlet_pressure", {"initial": "", "steps": []})
        self.editor.put(("program",), replacement)
        pressure_ids = self.editor.case.metadata.get("step_ids", {}).get("outlet_pressure", [])
        self.editor.case.metadata["step_ids"] = deepcopy(mode_ids.get(mode, {}))
        self.editor.case.metadata["step_ids"]["outlet_pressure"] = pressure_ids
        ensure_step_ids(replacement, self.editor.case.metadata)
        self.editor.put(("run", "simulation", "program_mode"), mode)
        self.load()

    def change_basis(self):
        if self.loading or self.editor.read_only or self.editor.case is None:
            return
        key = "feed_stream" if self.mode.currentData() == "feed_stream" else "inlet_flow"
        channel = deepcopy(self.editor.get(("program", key), {}))
        previous = channel.get("basis", "mol_per_s")
        basis = self.flow_basis.currentData()
        if basis == previous:
            return
        targets = [channel, *[step for step in channel.get("steps", []) if step.get("kind") == "ramp"]]
        values = []
        for index, target in enumerate(targets):
            name = "initial" if index == 0 else "target"
            if key == "feed_stream":
                value = target.get(name)
                if isinstance(value, dict) and "flow" in value:
                    values.append((value, "flow"))
            elif name in target:
                values.append((target, name))
        numeric = [(value, name) for value, name in values if isinstance(value[name], (int, float))]
        try:
            radius = float(self.editor.get(("run", "model", "bed_radius_m")))
            length = float(self.editor.get(("run", "model", "bed_length_m")))
            if not (math.isfinite(radius) and math.isfinite(length) and radius > 0 and length > 0):
                raise ValueError
            scale = math.pi * radius ** 2 * length * NORMAL_MOLAR_DENSITY_MOL_PER_M3 / 3600
            if basis == "ghsv_per_h":
                scale = 1 / scale
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            if numeric:
                self.loading = True
                select_value(self.flow_basis, previous)
                self.loading = False
                self.note.setText("Set a positive bed radius and length before converting existing flow values.")
                return
            scale = 1
        for value, name in numeric:
            value[name] *= scale
        channel["basis"] = basis
        self.editor.put(("program", key), channel)
        self.note.setText("GHSV reference: 273.15 K, 100,000 Pa. Each mode keeps its own inlet steps.")
        self.channels[key].load()
