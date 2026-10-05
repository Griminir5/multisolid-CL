"""Present structured validation issues without changing or rejecting drafts."""

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap

from packed_bed.config.load import input_field_issues

from .theme import colors


ISSUE_ROLE = Qt.ItemDataRole.UserRole + 74


def _with_message(text, previous, message):
    suffix = "\n\n" + previous if previous else ""
    if previous and text == previous:
        text = ""
    elif suffix and text.endswith(suffix):
        text = text[:-len(suffix)]
    return "\n\n".join(part for part in (text, message) if part)


def set_field_issue(widget, message):
    previous = widget.property("inputIssue") or ""
    widget.setToolTip(_with_message(widget.toolTip(), previous, message))
    widget.setAccessibleDescription(_with_message(widget.accessibleDescription(), previous, message))
    widget.setProperty("inputIssue", message)
    if bool(previous) != bool(message):
        widget.setProperty("invalidInput", bool(message))
        # Binary choices render their border on the child buttons.
        for control in (widget, *getattr(widget, "buttons", ())):
            control.style().unpolish(control)
            control.style().polish(control)
            control.update()


def set_cell_issue(item, message):
    previous = item.data(ISSUE_ROLE) or ""
    item.setToolTip(_with_message(item.toolTip(), previous, message))
    description = item.data(Qt.ItemDataRole.AccessibleDescriptionRole) or ""
    item.setData(Qt.ItemDataRole.AccessibleDescriptionRole, _with_message(description, previous, message))
    item.setData(ISSUE_ROLE, message)


def related(first, second):
    return first[:len(second)] == second or second[:len(first)] == first


def messages_for(issues, *paths):
    messages = []
    for issue in issues:
        if any(related(location, path) for path in paths for location in issue.paths):
            if issue.message not in messages:
                messages.append(issue.message)
    return "\n".join(messages)


def issue_icon():
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(colors()["error"]))
    painter.drawEllipse(1, 1, 14, 14)
    painter.setPen(QColor(colors()["surface"]))
    font = painter.font()
    font.setPixelSize(12)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "!")
    painter.end()
    return QIcon(pixmap)


def refresh_field_issues(editor):
    if editor.case is None:
        return []
    issues = input_field_issues(editor.case.documents)
    tab_messages = {page: [] for page in (editor.general, editor.chemistry, editor.bed, editor.program)}

    def mark(widget, *paths):
        message = messages_for(issues, *paths)
        set_field_issue(widget, message)
        if message:
            for owner in tab_messages:
                if owner.isAncestorOf(widget):
                    tab_messages[owner].append(message)

    for widget, path, *_ in editor.bindings:
        mark(widget, path)
    for phase, path in (("gas", ("chemistry", "gas_species")), ("solid", ("solids", "solid_species"))):
        mark(getattr(editor.chemistry, phase + "_list").table, path)
        mark(editor.chemistry.sections[phase].toggle, path)
    mark(editor.chemistry.families, ("chemistry", "reaction_ids"), ("chemistry", "reaction_families"),
         ("chemistry", "mechanisms"))
    mark(editor.general.reports.table, ("run", "outputs", "requested_reports"))
    mark(editor.general.plots.table, ("run", "outputs", "requested_plots"))
    from packed_bed.config.models import SolverConfig
    advanced = [("run", "solver", key) for key in SolverConfig.model_fields
                if key not in ("backend", "name", "threads", "relative_tolerance")]
    mark(editor.general.advanced_button, *advanced)
    mark(editor.program.mode, ("run", "simulation", "program_mode"))
    mark(editor.program.repeat, ("run", "simulation", "repeat_program"))
    channel_key = "feed_stream" if editor.program.mode.currentData() == "feed_stream" else "inlet_flow"
    mark(editor.program.flow_basis, ("program", channel_key, "basis"))

    zones_path = ("solids", "initial_profile", "zones")
    zones = editor.bed.zones
    with QSignalBlocker(zones):
        for row in range(zones.rowCount() - 1):
            for col, key in enumerate(editor.bed.columns):
                item = zones.item(row, col)
                if item is not None:
                    path = zones_path + (row,) + ((key,) if col < 5 else ("values", key))
                    message = messages_for(issues, path)
                    set_cell_issue(item, message)
                    if message:
                        tab_messages[editor.bed].append(message)
    # There is no editable cell to mark until the first zone exists.
    empty_message = editor.bed.update_empty_state(zones.rowCount() == 1)
    if empty_message:
        tab_messages[editor.bed].append(empty_message)
    zones.viewport().update()

    active = ("feed_stream", "outlet_pressure") if channel_key == "feed_stream" else (
        "inlet_flow", "inlet_temperature", "inlet_composition", "outlet_pressure")
    for key, channel in editor.program.channels.items():
        channel_issues = issues if key in active else []
        prefix = ("program", key)
        if key in active:
            mark(channel.toggle, prefix)
        else:
            set_field_issue(channel.toggle, "")
        with QSignalBlocker(channel.table):
            for row in range(channel.table.rowCount() - 1):
                for col in range(3):
                    if row == 0 and col != 2:
                        continue
                    path = prefix + (("initial",) if row == 0 else
                                     ("steps", row - 1, ("kind", "duration_s", "target")[col]))
                    message = messages_for(channel_issues, path)
                    widget, item = channel.table.cellWidget(row, col), channel.table.item(row, col)
                    if widget is not None:
                        set_field_issue(widget, message)
                    if item is not None:
                        set_cell_issue(item, message)
                    if message:
                        tab_messages[editor.program].append(message)
        channel.table.viewport().update()

    icon = issue_icon()
    for page, messages in tab_messages.items():
        index = editor.tabs.indexOf(page)
        editor.tabs.setTabIcon(index, icon if messages else QIcon())
        editor.tabs.setTabToolTip(index, "\n".join(dict.fromkeys(messages)))
    return issues
