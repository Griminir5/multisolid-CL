"""Material zones and geometry, above the full-width numerical bed preview."""

from copy import deepcopy

from PyQt6.QtWidgets import QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QVBoxLayout, QWidget

from .editor_widgets import Preview, action_button, cell, display, number, table
from .general import form_panel


class BedPage(QWidget):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.loading = False
        self.previous_length = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        group = QGroupBox("Material zones")
        zones_layout = QVBoxLayout(group)
        self.units = QLabel()
        self.units.setWordWrap(True)
        zones_layout.addWidget(self.units)
        self.zones = table([])
        self.zones.itemChanged.connect(self.edit_zone)
        zones_layout.addWidget(self.zones, 1)
        zones_layout.addWidget(action_button("+ Add zone", self.add_zone))
        top.addWidget(group, 162)
        options, form = form_panel("Bed settings")
        # Compact labels beside fields leave room for the bed preview at ordinary window sizes.
        form.setRowWrapPolicy(form.RowWrapPolicy.WrapLongRows)
        editor.field(form, ("run", "model", "bed_radius_m"), "Radius (m)")
        self.length = editor.field(form, ("run", "model", "bed_length_m"), "Length (m)")
        self.length.editingFinished.connect(self.resize_zones)
        self.basis = editor.field(form, ("solids", "initial_profile", "basis"), "Concentration basis", options=[
            ("Bed volume", "bed"), ("Solid volume", "solid"),
        ])
        self.basis.currentIndexChanged.connect(lambda: self.update_units())
        editor.field(form, ("run", "model", "gas_voidage_mode"), "Gas voidage", options=[
            ("Bed only · e_b", "bed_only"), ("Bed + particle", "bed_and_particle"),
        ], default="bed_and_particle")
        editor.field(form, ("run", "model", "ambient_temperature_k"), "Ambient temperature (K)", default=873.15)
        editor.field(form, ("run", "model", "heat_transfer_coefficient_w_per_m2_k"), "Heat transfer (W/m²/K)", default=100.0)
        editor.field(form, ("run", "simulation", "interior_flow_mode"), "Reversible flow", kind="check",
                     default="forward_only", checked_values=("forward_only", "reversible"))
        top.addWidget(options, 100)
        layout.addLayout(top, 1)
        self.preview = Preview()
        layout.addWidget(self.preview, 1)

    def load(self):
        self.previous_length = self.editor.get(("run", "model", "bed_length_m"))
        self.load_zones()

    def update_units(self):
        basis = self.editor.get(("solids", "initial_profile", "basis"), "bed")
        self.units.setText(f"x_start, x_end and d_p in m · voidages as fractions · concentrations in mol/m³ {basis}")

    def load_zones(self):
        self.loading = True
        self.update_units()
        self.species = self.editor.get(("solids", "solid_species"), [])
        self.columns = ["x_start_m", "x_end_m", "e_b", "e_p", "d_p"] + list(self.species)
        self.zones.clear()
        self.zones.setColumnCount(len(self.columns) + 1)
        self.zones.setHorizontalHeaderLabels(["x_start", "x_end", "e_b", "e_p", "d_p", *self.species, ""])
        self.zones.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for column in range(len(self.columns)):
            self.zones.setColumnWidth(column, 78)
        self.zones.horizontalHeader().setStretchLastSection(False)
        self.zones.horizontalHeader().setSectionResizeMode(len(self.columns), QHeaderView.ResizeMode.Fixed)
        self.zones.setColumnWidth(len(self.columns), 34)
        zones = self.editor.get(("solids", "initial_profile", "zones"), [])
        self.zones.setRowCount(len(zones))
        for row, zone in enumerate(zones):
            for column, key in enumerate(self.columns):
                locked = (row == 0 and column == 0) or (row == len(zones) - 1 and column == 1)
                value = zone.get(key, "") if column < 5 else zone.get("values", {}).get(key, "")
                cell(self.zones, row, column, value, editable=not locked,
                     tooltip="Tied to the reactor boundary" if locked else "")
            self.zones.setCellWidget(row, len(self.columns), action_button("×", lambda _, row=row: self.remove_zone(row),
                                                                         tooltip=f"Remove zone {row + 1}"))
        self.loading = False

    def anchor_zones(self):
        zones = self.editor.get(("solids", "initial_profile", "zones"), [])
        if zones:
            zones[0]["x_start_m"] = 0.0
            zones[-1]["x_end_m"] = self.editor.get(("run", "model", "bed_length_m"), "")
            if self.zones.rowCount() == len(zones):
                loading, self.loading = self.loading, True
                for row, column, value in ((0, 0, 0.0), (len(zones) - 1, 1, zones[-1]["x_end_m"])):
                    item = self.zones.item(row, column)
                    if item is not None:
                        item.setText(display(value))
                self.loading = loading

    def edit_zone(self, item):
        if self.loading:
            return
        zones = deepcopy(self.editor.get(("solids", "initial_profile", "zones"), []))
        key = self.columns[item.column()]
        target = zones[item.row()] if item.column() < 5 else zones[item.row()].setdefault("values", {})
        target[key] = number(item.text())
        self.editor.put(("solids", "initial_profile", "zones"), zones)

    def add_zone(self):
        zones = deepcopy(self.editor.get(("solids", "initial_profile", "zones"), []))
        end = self.editor.get(("run", "model", "bed_length_m"), "")
        start = 0.0
        if zones:
            # Split the final zone's extent, leaving the new material properties unfinished.
            previous_start = zones[-1].get("x_start_m")
            if isinstance(previous_start, (int, float)) and isinstance(end, (int, float)) and end > previous_start:
                start = (previous_start + end) / 2
                zones[-1]["x_end_m"] = start
            else:
                start = ""
        zones.append({"x_start_m": start, "x_end_m": end, "e_b": "", "e_p": "", "d_p": "",
                      "values": {key: "" for key in self.editor.get(("solids", "solid_species"), [])}})
        self.editor.put(("solids", "initial_profile", "zones"), zones)
        self.anchor_zones()
        self.load_zones()

    def remove_zone(self, row):
        zones = deepcopy(self.editor.get(("solids", "initial_profile", "zones"), []))
        removed = zones.pop(row)
        if row < len(zones):
            zones[row]["x_start_m"] = removed.get("x_start_m", "")
        self.editor.put(("solids", "initial_profile", "zones"), zones)
        self.anchor_zones()
        self.load_zones()

    def resize_zones(self):
        if self.editor.loading:
            return
        length = self.editor.get(("run", "model", "bed_length_m"))
        previous = self.previous_length
        if length == previous:
            return
        zones = self.editor.get(("solids", "initial_profile", "zones"), [])
        if (len(zones) > 1 and isinstance(length, (int, float)) and length > 0
                and isinstance(previous, (int, float)) and previous > 0):
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Change bed length")
            dialog.setText("How should the internal zone boundaries change?")
            keep = dialog.addButton("Keep positions", QMessageBox.ButtonRole.AcceptRole)
            scale = dialog.addButton("Scale proportionally", QMessageBox.ButtonRole.ActionRole)
            dialog.setDefaultButton(keep)
            dialog.exec()
            if dialog.clickedButton() == scale:
                for zone in zones:
                    for key in ("x_start_m", "x_end_m"):
                        if isinstance(zone.get(key), (int, float)):
                            zone[key] *= length / previous
        self.previous_length = length
        self.anchor_zones()
        self.editor.queue_edit()
        self.load_zones()
