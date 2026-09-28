"""Material zones and geometry, above the full-width numerical bed preview."""

from copy import deepcopy

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSplitter, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QVBoxLayout, QWidget

from .editor_widgets import Preview, action_button, cell, display, number, table, table_action
from .general import form_panel


class BedSettings(QGroupBox):
    """QSplitter does not negotiate QFormLayout's wrapped height automatically."""
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.layout():
            height = self.layout().totalHeightForWidth(self.width())
            if height >= 0 and height != self.minimumHeight():
                self.setMinimumHeight(height)


class BedPage(QWidget):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.loading = False
        self.previous_length = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.vertical_split = QSplitter(Qt.Orientation.Vertical)
        self.vertical_split.setChildrenCollapsible(False)
        self.settings_split = QSplitter()
        self.settings_split.setChildrenCollapsible(False)
        group = QGroupBox("Material zones")
        zones_layout = QVBoxLayout(group)
        self.units = QLabel()
        self.units.setWordWrap(True)
        zones_layout.addWidget(self.units)
        self.zones = table([])
        self.zones.itemChanged.connect(self.edit_zone)
        zones_layout.addWidget(self.zones, 1)
        options, form = form_panel("Bed settings", panel_type=BedSettings)
        options.setMinimumWidth(310)
        # Compact labels beside fields leave room for the bed preview at ordinary window sizes.
        form.setRowWrapPolicy(form.RowWrapPolicy.WrapLongRows)
        editor.field(form, ("run", "model", "bed_radius_m"), "Radius (m)")
        self.length = editor.field(form, ("run", "model", "bed_length_m"), "Length (m)")
        self.length.editingFinished.connect(self.resize_zones)
        self.basis = editor.field(form, ("solids", "initial_profile", "basis"), "Concentration basis", options=[
            ("Bed volume", "bed"), ("Solid volume", "solid"),
        ], binary=True)
        self.basis.currentIndexChanged.connect(lambda: self.update_units())
        editor.field(form, ("run", "model", "gas_voidage_mode"), "Gas voidage", options=[
            ("Interparticle only", "bed_only"), ("Interparticle + intraparticle", "bed_and_particle"),
        ], default="bed_and_particle", binary=True, vertical=True)
        editor.field(form, ("run", "model", "ambient_temperature_k"), "Ambient temperature (K)", default=873.15)
        editor.field(form, ("run", "model", "heat_transfer_coefficient_w_per_m2_k"), "Heat transfer (W/m²/K)", default=100.0)
        editor.field(form, ("run", "simulation", "interior_flow_mode"), "Reversible flow", kind="check",
                     default="forward_only", checked_values=("forward_only", "reversible"))
        self.settings_split.addWidget(options)
        self.settings_split.addWidget(group)
        self.settings_split.setStretchFactor(0, 1)
        self.settings_split.setStretchFactor(1, 3)
        self.settings_split.setSizes([300, 900])
        self.vertical_split.addWidget(self.settings_split)
        self.preview = Preview()
        self.vertical_split.addWidget(self.preview)
        self.vertical_split.setSizes([330, 280])
        layout.addWidget(self.vertical_split)

    def load(self):
        self.previous_length = self.editor.get(("run", "model", "bed_length_m"))
        self.load_zones()

    def update_units(self):
        basis = self.editor.get(("solids", "initial_profile", "basis"), "bed")
        self.units.setText(f"Positions and particle diameter in m · voidages as fractions · concentrations in mol/m³ {basis}")

    def load_zones(self):
        self.loading = True
        self.update_units()
        self.species = self.editor.get(("solids", "solid_species"), [])
        self.columns = ["x_start_m", "x_end_m", "e_b", "e_p", "d_p"] + list(self.species)
        self.zones.clearSpans()
        self.zones.clear()
        self.zones.setColumnCount(len(self.columns) + 1)
        self.zones.setHorizontalHeaderLabels(["Start\n(m)", "End\n(m)", "Interparticle\nvoidage", "Intraparticle\nvoidage", "Particle\ndiameter (m)", *[self.editor.species_label(key, formula_only=True) for key in self.species], ""])
        self.zones.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for column in range(len(self.columns)):
            self.zones.setColumnWidth(column, 120 if column in (2, 3, 4) else 78)
        self.zones.horizontalHeader().setStretchLastSection(False)
        self.zones.horizontalHeader().setSectionResizeMode(len(self.columns), QHeaderView.ResizeMode.Fixed)
        self.zones.setColumnWidth(len(self.columns), 34)
        zones = self.editor.get(("solids", "initial_profile", "zones"), [])
        self.zones.setRowCount(len(zones) + 1)
        for row, zone in enumerate(zones):
            self.zones.setRowHeight(row, self.zones.verticalHeader().defaultSectionSize())
            for column, key in enumerate(self.columns):
                locked = (row == 0 and column == 0) or (row == len(zones) - 1 and column == 1)
                value = zone.get(key, "") if column < 5 else zone.get("values", {}).get(key, "")
                cell(self.zones, row, column, value, editable=not locked, numeric_value=True,
                     tooltip="Tied to the reactor boundary" if locked else "")
            self.zones.setCellWidget(row, len(self.columns), action_button("×", lambda _, row=row: self.remove_zone(row),
                                                                         tooltip=f"Remove zone {row + 1}"))
        self.add_button = table_action(self.zones, len(zones), "+ Add zone", self.add_zone)
        self.add_button.setEnabled(not self.editor.read_only)
        self.loading = False

    def anchor_zones(self):
        if self.editor.read_only:
            return
        zones = self.editor.get(("solids", "initial_profile", "zones"), [])
        if zones:
            zones[0]["x_start_m"] = 0.0
            zones[-1]["x_end_m"] = self.editor.get(("run", "model", "bed_length_m"), "")
            if self.zones.rowCount() == len(zones) + 1:
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
        if item.row() >= len(zones) or item.column() >= len(self.columns):
            return  # The trailing action is not a material zone.
        key = self.columns[item.column()]
        target = zones[item.row()] if item.column() < 5 else zones[item.row()].setdefault("values", {})
        target[key] = number(item.text())
        self.editor.put(("solids", "initial_profile", "zones"), zones)

    def add_zone(self):
        if self.editor.read_only:
            return
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
        self.zones.scrollToItem(self.zones.item(len(zones) - 1, 2))

    def remove_zone(self, row):
        zones = deepcopy(self.editor.get(("solids", "initial_profile", "zones"), []))
        removed = zones.pop(row)
        if row < len(zones):
            zones[row]["x_start_m"] = removed.get("x_start_m", "")
        self.editor.put(("solids", "initial_profile", "zones"), zones)
        self.anchor_zones()
        self.load_zones()

    def resize_zones(self):
        if self.editor.loading or self.editor.read_only:
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
