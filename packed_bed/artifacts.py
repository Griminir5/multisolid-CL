"""Explicit pre-run diagrams built only from a resolved case."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np

from .config import Case
from .plotting.definitions import pyplot, save_figure
from .programs import DEFAULT_SMOOTH_RAMP_WIDTH_S
from .preview import _smoothed_program_sample_times, _series_from_smoothed_program
from .solid_profiles import (
    build_cell_profiles,
    build_face_scalar_profile,
    build_uniform_axial_grid,
    gas_fraction_from_voidages,
    solid_fraction_from_voidages,
    zone_edges,
)


def _draw_zone_boundaries(axes, boundaries):
    if len(boundaries) <= 2:
        return
    for axis in axes:
        for boundary in boundaries[1:-1]:
            axis.axvline(float(boundary), color="#adb5bd", linestyle="--", linewidth=1.0, alpha=0.7, zorder=0)


def _finalize_series_axes(axes, *, x_min, x_max):
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.set_xlim(float(x_min), float(x_max))
        axis.margins(x=0.0)


def _render_system_graph(case: Case, output_dir: Path, property_registry) -> dict[str, Path]:
    import pygraphviz
    from .reactions import reaction_catalog

    graph = pygraphviz.AGraph(name="system_graph", strict=False, directed=True)
    graph.graph_attr.update(
        bgcolor="white",
        pad="0.35",
        outputorder="edgesfirst",
        overlap="false",
        splines="true",
        labelloc="t",
        labeljust="c",
        fontname="Arial",
        fontsize="20",
        label="Species and Reaction System Graph",
        rankdir="LR",
        nodesep="0.55",
        ranksep="0.85",
        newrank="true",
    )

    graph.node_attr.update(
        fontname="Arial",
        fontsize="13",
        penwidth="1.6",
        margin="0.18,0.12",
    )
    graph.edge_attr.update(
        fontname="Arial",
        fontsize="13",
        penwidth="2.0",
        arrowsize="0.95",
    )

    for species_ids, fill_color in (
        (case.chemistry.gas_species, "#81b29a"),
        (case.solids.solid_species, "#f2cc8f"),
    ):
        for species_id in species_ids:
            graph.add_node(
                species_id,
                label=f"{species_id}\n{property_registry.get_record(species_id).name}",
                shape="ellipse",
                style="filled",
                fillcolor=fill_color,
                color="#2f3e46",
                fontcolor="#1f2933",
            )

    def add_edge(source, target, coefficient, reversible, label):
        edge_color = (
            "#355070"
            if coefficient < 0.0
            else "#bc6c25" if coefficient > 0.0 else "#6d597a"
        )
        graph.add_edge(
            source,
            target,
            label=label,
            color=edge_color,
            fontcolor=edge_color,
            dir="both" if reversible else "forward",
            arrowhead="normal",
            arrowtail="normal" if reversible else "none",
            style="dashed" if coefficient == 0.0 else "solid",
            constraint="false" if coefficient == 0.0 else "true",
            penwidth="1.8" if coefficient == 0.0 else "2.2",
            arrowsize="0.85" if coefficient == 0.0 else "0.95",
        )

    reactions = reaction_catalog(case.reaction_families)
    for reaction_id in case.chemistry.reaction_ids:
        reaction = reactions[reaction_id]
        reaction_node = f"reaction:{reaction.id}"
        label = [reaction.id]
        if reaction.reversible:
            label.append("reversible")
        if reaction.catalyst_species:
            label.append(f"cat: {', '.join(reaction.catalyst_species)}")
        graph.add_node(
            reaction_node,
            label="\n".join(label),
            shape="box",
            style="rounded,filled",
            fillcolor="#6d597a",
            color="#3d405b",
            fontcolor="white",
            margin="0.22,0.14",
        )
        for species_id, coefficient in reaction.stoichiometry.items():
            magnitude = abs(coefficient)
            rounded = round(magnitude)
            coefficient_label = (
                str(int(rounded))
                if abs(magnitude - rounded) < 1.0e-9
                else f"{magnitude:g}"
            )
            source, target = (
                (species_id, reaction_node)
                if coefficient < 0.0
                else (reaction_node, species_id)
            )
            add_edge(
                source,
                target,
                coefficient,
                reaction.reversible,
                coefficient_label,
            )
        for species_id in reaction.catalyst_species:
            add_edge(species_id, reaction_node, 0.0, reaction.reversible, "cat")

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "system_graph.svg"
    graph.draw(str(svg_path), prog="neato")
    return {"system_graph_svg": svg_path}


def render_operating_program(
    case: Case,
    output_dir,
    *,
    smooth_ramp_width_s: float = DEFAULT_SMOOTH_RAMP_WIDTH_S,
) -> dict[str, Path]:
    plt = pyplot()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    simulation = case.run.simulation
    time_horizon = simulation.time_horizon_s
    gas_species = case.chemistry.gas_species
    inlet_flow_program = case.inlet_flow_program
    inlet_temperature_program = case.inlet_temperature_program
    outlet_pressure_program = case.outlet_pressure_program
    inlet_composition_program = case.inlet_composition_program
    sample_times = _smoothed_program_sample_times(
        (
            inlet_flow_program,
            inlet_temperature_program,
            outlet_pressure_program,
            inlet_composition_program,
        ),
        final_time=time_horizon,
        smooth_ramp_width_s=smooth_ramp_width_s,
    )

    figure, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    for axis, program, color, unit, title in (
        (axes[0], inlet_flow_program, "#1d3557", "mol/s", "Inlet Flow"),
        (axes[1], inlet_temperature_program, "#e76f51", "K", "Inlet Temperature"),
        (axes[2], outlet_pressure_program, "#264653", "Pa", "Outlet Pressure"),
    ):
        values = _series_from_smoothed_program(
            program, sample_times, smooth_ramp_width_s=smooth_ramp_width_s,
        )
        axis.plot(sample_times, values, color=color, linewidth=2)
        axis.set(ylabel=unit, title=title)

    composition_values = _series_from_smoothed_program(
        inlet_composition_program,
        sample_times,
        smooth_ramp_width_s=smooth_ramp_width_s,
    )
    for species_idx, species_id in enumerate(gas_species):
        axes[3].plot(sample_times, composition_values[:, species_idx], linewidth=2, label=species_id)
    axes[3].set_ylabel("Mole fraction")
    axes[3].set_title("Inlet Composition")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylim(-0.02, 1.02)
    if gas_species:
        axes[3].legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=8,
        )

    axes[0].set_ylim(bottom=0.0)
    axes[2].set_ylim(bottom=0.0)
    _finalize_series_axes(axes, x_min=0.0, x_max=time_horizon)

    figure.tight_layout()

    svg_path = output_dir / "operating_program.svg"
    save_figure(figure, svg_path)

    return {"operating_program_svg": svg_path}


def render_initial_solid_profile(case: Case, output_dir) -> dict[str, Path]:
    plt = pyplot()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zones = case.solids.initial_profile.zones
    cell_centers, face_positions = build_uniform_axial_grid(
        case.run.model.bed_length_m,
        case.run.model.axial_cells,
    )
    authored_edges = zone_edges(case.solids)
    e_b, e_p, bed_basis_concentration = build_cell_profiles(case.solids, cell_centers)
    gas_fraction = gas_fraction_from_voidages(e_b, e_p, mode=case.run.model.gas_voidage_mode)
    solid_fraction = solid_fraction_from_voidages(e_b, e_p)
    d_p = build_face_scalar_profile(case.solids, face_positions, "d_p")

    unit_label = {
        "solid": "mol/m^3 solid",
        "bed": "mol/m^3 bed",
    }.get(case.solids.initial_profile.basis, case.solids.initial_profile.basis)

    figure, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True)

    for species_id in case.solids.solid_species:
        zone_values = [float(zone.values[species_id]) for zone in zones]
        axes[0].stairs(zone_values, authored_edges, label=species_id, linewidth=2)
    axes[0].set_title("Initial Solid Concentration Input")
    axes[0].set_ylabel(unit_label)

    for species_index, species_id in enumerate(case.solids.solid_species):
        axes[1].stairs(
            bed_basis_concentration[species_index],
            face_positions,
            label=species_id,
            linewidth=2,
        )
    axes[1].set_title("Initial Solid Concentration on Bed-Volume Basis")
    axes[1].set_ylabel("mol/m^3 bed")

    axes[2].stairs(e_b, face_positions, linewidth=2, label="e_b")
    axes[2].stairs(e_p, face_positions, linewidth=2, label="e_p")
    axes[2].stairs(gas_fraction, face_positions, linewidth=1.5, linestyle="--", label="gasfrac")
    axes[2].stairs(solid_fraction, face_positions, linewidth=1.5, linestyle=":", label="solfrac")
    axes[2].set_title("Voidages and Volume Fractions")
    axes[2].set_ylabel("Fraction")
    axes[2].set_ylim(0.0, 1.05)

    axes[3].plot(face_positions, d_p, marker="o", markersize=4, linewidth=2, color="#7f5539")
    axes[3].set_title("Particle Characteristic Length on Face Domain")
    axes[3].set_ylabel("d_p [m]")
    axes[3].set_xlabel("Axial position [m]")

    axes[0].set_ylim(bottom=0.0)
    axes[1].set_ylim(bottom=0.0)
    axes[3].set_ylim(bottom=0.0)
    _draw_zone_boundaries(axes, authored_edges)
    _finalize_series_axes(axes, x_min=0.0, x_max=case.run.model.bed_length_m)

    if case.solids.solid_species:
        axes[0].legend(loc="upper right", fontsize=8)
        axes[1].legend(loc="upper right", fontsize=8)
    axes[2].legend(loc="upper right", fontsize=8)

    figure.tight_layout()

    svg_path = output_dir / "initial_solid_profile.svg"
    save_figure(figure, svg_path)

    return {"initial_solid_profile_svg": svg_path}


def generate_artifacts(case: Case) -> dict[str, Path]:
    """Generate the explicitly requested pre-run diagrams."""

    from .properties import PROPERTY_REGISTRY

    case.output_directory.mkdir(parents=True, exist_ok=True)
    case.artifacts_directory.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    if find_spec("pygraphviz") is not None:
        artifacts.update(
            _render_system_graph(
                case,
                case.artifacts_directory,
                PROPERTY_REGISTRY,
            )
        )
    artifacts.update(render_operating_program(case, case.artifacts_directory))
    artifacts.update(render_initial_solid_profile(case, case.artifacts_directory))
    return artifacts


__all__ = ("generate_artifacts",)
