from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .config import RunBundle
from .solid_profiles import (
    build_cell_scalar_profile,
    build_face_scalar_profile,
    build_uniform_axial_grid,
    convert_solid_profile_to_bed_volume,
    gas_fraction_from_voidages,
    solid_fraction_from_voidages,
    zone_edges,
)


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    kind: str
    phase: str
    color: str


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    label: str
    coefficient: float
    reversible: bool


@dataclass(frozen=True)
class SystemGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


def is_pygraphviz_available() -> bool:
    return find_spec("pygraphviz") is not None


def build_system_graph(
    run_bundle: RunBundle,
    *,
    property_registry,
    reaction_catalog
) -> SystemGraph:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for species_id in run_bundle.chemistry.gas_species:
        record = property_registry.get_record(species_id)
        nodes.append(
            GraphNode(
                id=species_id,
                label=f"{species_id}\n{record.name}",
                kind="species",
                phase="gas",
                color="#81b29a",
            )
        )

    for species_id in run_bundle.solids.solid_species:
        record = property_registry.get_record(species_id)
        nodes.append(
            GraphNode(
                id=species_id,
                label=f"{species_id}\n{record.name}",
                kind="species",
                phase="solid",
                color="#f2cc8f",
            )
        )

    for reaction_id in run_bundle.chemistry.reaction_ids:
        reaction = reaction_catalog[reaction_id]
        reaction_node_id = f"reaction:{reaction.id}"
        label_lines = [reaction.id]
        if reaction.reversible:
            label_lines.append("reversible")
        if reaction.catalyst_species:
            label_lines.append(f"cat: {', '.join(reaction.catalyst_species)}")
        nodes.append(
            GraphNode(
                id=reaction_node_id,
                label="\n".join(label_lines),
                kind="reaction",
                phase=reaction.phase,
                color="#6d597a",
            )
        )
        for species_id, coefficient in reaction.stoichiometry.items():
            if coefficient < 0.0:
                edges.append(
                    GraphEdge(
                        source=species_id,
                        target=reaction_node_id,
                        label=str(abs(coefficient)),
                        coefficient=coefficient,
                        reversible=reaction.reversible,
                    )
                )
            elif coefficient > 0.0:
                edges.append(
                    GraphEdge(
                        source=reaction_node_id,
                        target=species_id,
                        label=str(abs(coefficient)),
                        coefficient=coefficient,
                        reversible=reaction.reversible,
                    )
                )
        for catalyst_species_id in reaction.catalyst_species:
            edges.append(
                GraphEdge(
                    source=catalyst_species_id,
                    target=reaction_node_id,
                    label="cat",
                    coefficient=0.0,
                    reversible=reaction.reversible,
                )
            )

    return SystemGraph(nodes=tuple(nodes), edges=tuple(edges))


def _series_from_segments(segments, initial_value, *, final_time: float | None = None):
    if not segments:
        if final_time is not None and final_time > 0.0:
            return [0.0, float(final_time)], [initial_value, initial_value]
        return [0.0], [initial_value]

    times = [0.0]
    values = [initial_value]
    for segment in segments:
        if times[-1] != segment.start_time or values[-1] != segment.start_value:
            times.append(float(segment.start_time))
            values.append(float(segment.start_value))
        times.append(float(segment.end_time))
        values.append(float(segment.end_value))
    if final_time is not None and final_time > times[-1]:
        times.append(float(final_time))
        values.append(values[-1])
    return times, values


def _vector_series_from_segments(segments, initial_value, *, final_time: float | None = None):
    initial_components = tuple(float(value) for value in initial_value)
    if not segments:
        if final_time is not None and final_time > 0.0:
            return [0.0, float(final_time)], [[value, value] for value in initial_components]
        return [0.0], [[value] for value in initial_components]

    times = [0.0]
    values_by_component = [[value] for value in initial_components]
    for segment in segments:
        start_values = tuple(float(value) for value in segment.start_value)
        end_values = tuple(float(value) for value in segment.end_value)
        if times[-1] != segment.start_time or any(
            component_values[-1] != start_value
            for component_values, start_value in zip(values_by_component, start_values)
        ):
            times.append(float(segment.start_time))
            for component_values, start_value in zip(values_by_component, start_values):
                component_values.append(start_value)
        times.append(float(segment.end_time))
        for component_values, end_value in zip(values_by_component, end_values):
            component_values.append(end_value)
    if final_time is not None and final_time > times[-1]:
        times.append(float(final_time))
        for component_values in values_by_component:
            component_values.append(component_values[-1])
    return times, values_by_component


def _format_edge_label(edge: GraphEdge) -> str:
    if edge.coefficient == 0.0:
        return edge.label

    magnitude = abs(edge.coefficient)
    rounded = round(magnitude)
    if abs(magnitude - rounded) < 1e-9:
        return str(int(rounded))
    return f"{magnitude:g}"


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


def _save_figure(figure, path: Path) -> None:
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _build_system_agraph(system_graph: SystemGraph):
    import pygraphviz

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

    for node in system_graph.nodes:
        if node.kind == "reaction":
            attributes = {
                "label": node.label,
                "shape": "box",
                "style": "rounded,filled",
                "fillcolor": node.color,
                "color": "#3d405b",
                "fontcolor": "white",
                "margin": "0.22,0.14",
            }
        else:
            attributes = {
                "label": node.label,
                "shape": "ellipse",
                "style": "filled",
                "fillcolor": node.color,
                "color": "#2f3e46",
                "fontcolor": "#1f2933",
            }
        graph.add_node(node.id, **attributes)

    for edge in system_graph.edges:
        if edge.coefficient < 0.0:
            edge_color = "#355070"
        elif edge.coefficient > 0.0:
            edge_color = "#bc6c25"
        else:
            edge_color = "#6d597a"

        attributes = {
            "label": _format_edge_label(edge),
            "color": edge_color,
            "fontcolor": edge_color,
            "dir": "both" if edge.reversible else "forward",
            "arrowhead": "normal",
            "arrowtail": "normal" if edge.reversible else "none",
            "style": "dashed" if edge.coefficient == 0.0 else "solid",
            "constraint": "false" if edge.coefficient == 0.0 else "true",
            "penwidth": "1.8" if edge.coefficient == 0.0 else "2.2",
            "arrowsize": "0.85" if edge.coefficient == 0.0 else "0.95",
        }
        graph.add_edge(edge.source, edge.target, **attributes)

    return graph


def render_system_graph(system_graph: SystemGraph, output_dir) -> dict[str, Path]:
    if not is_pygraphviz_available():
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "system_graph.svg"
    graph = _build_system_agraph(system_graph)
    graph.draw(str(svg_path), prog="neato")

    return {"system_graph_svg": svg_path}


def render_operating_program(run_bundle: RunBundle, output_dir) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_horizon = run_bundle.run.time_horizon_s
    gas_species = run_bundle.chemistry.gas_species

    inlet_flow_program = run_bundle.program.inlet_flow.compile_program(
        repeat=run_bundle.run.repeat_program,
        time_horizon=time_horizon,
    )
    inlet_temperature_program = run_bundle.program.inlet_temperature.compile_program(
        repeat=run_bundle.run.repeat_program,
        time_horizon=time_horizon,
    )
    outlet_pressure_program = run_bundle.program.outlet_pressure.compile_program(
        repeat=run_bundle.run.repeat_program,
        time_horizon=time_horizon,
    )
    inlet_composition_program = run_bundle.program.inlet_composition.compile_program(
        gas_species,
        repeat=run_bundle.run.repeat_program,
        time_horizon=time_horizon,
    )

    figure, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    flow_times, flow_values = _series_from_segments(
        inlet_flow_program.build_segments(),
        inlet_flow_program.initial_value,
        final_time=time_horizon,
    )
    axes[0].plot(flow_times, flow_values, color="#1d3557", linewidth=2)
    axes[0].set_ylabel("mol/s")
    axes[0].set_title("Inlet Flow")

    temp_times, temp_values = _series_from_segments(
        inlet_temperature_program.build_segments(),
        inlet_temperature_program.initial_value,
        final_time=time_horizon,
    )
    axes[1].plot(temp_times, temp_values, color="#e76f51", linewidth=2)
    axes[1].set_ylabel("K")
    axes[1].set_title("Inlet Temperature")

    pressure_times, pressure_values = _series_from_segments(
        outlet_pressure_program.build_segments(),
        outlet_pressure_program.initial_value,
        final_time=time_horizon,
    )
    axes[2].plot(pressure_times, pressure_values, color="#264653", linewidth=2)
    axes[2].set_ylabel("Pa")
    axes[2].set_title("Outlet Pressure")

    composition_times, series_by_species = _vector_series_from_segments(
        inlet_composition_program.build_segments(),
        inlet_composition_program.initial_value,
        final_time=time_horizon,
    )
    for species_id, series in zip(gas_species, series_by_species):
        axes[3].plot(composition_times, series, linewidth=2, label=species_id)
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
    _save_figure(figure, svg_path)

    return {"operating_program_svg": svg_path}


def render_initial_solid_profile(run_bundle: RunBundle, output_dir) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zones = run_bundle.solids.initial_profile_zones
    cell_centers, face_positions = build_uniform_axial_grid(
        run_bundle.run.model.bed_length_m,
        run_bundle.run.model.axial_cells,
    )
    authored_edges = zone_edges(run_bundle.solids)
    e_b = build_cell_scalar_profile(run_bundle.solids, cell_centers, "e_b")
    e_p = build_cell_scalar_profile(run_bundle.solids, cell_centers, "e_p")
    gas_fraction = gas_fraction_from_voidages(e_b, e_p)
    solid_fraction = solid_fraction_from_voidages(e_b, e_p)
    bed_basis_concentration = convert_solid_profile_to_bed_volume(
        run_bundle.solids,
        cell_centers,
        solid_fraction,
        run_bundle.solids.solid_species,
    )
    d_p = build_face_scalar_profile(run_bundle.solids, face_positions, "d_p")

    unit_label = {
        "mol_per_m3_solid": "mol/m^3 solid",
        "mol_per_m3_bed": "mol/m^3 bed",
    }.get(run_bundle.solids.concentration_unit, run_bundle.solids.concentration_unit)

    figure, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True)

    for species_id in run_bundle.solids.solid_species:
        zone_values = [float(zone.values_mol_per_m3[species_id]) for zone in zones]
        axes[0].stairs(zone_values, authored_edges, label=species_id, linewidth=2)
    axes[0].set_title("Initial Solid Concentration Input")
    axes[0].set_ylabel(unit_label)

    for species_index, species_id in enumerate(run_bundle.solids.solid_species):
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
    _finalize_series_axes(axes, x_min=0.0, x_max=run_bundle.run.model.bed_length_m)

    if run_bundle.solids.solid_species:
        axes[0].legend(loc="upper right", fontsize=8)
        axes[1].legend(loc="upper right", fontsize=8)
    axes[2].legend(loc="upper right", fontsize=8)

    figure.tight_layout()

    svg_path = output_dir / "initial_solid_profile.svg"
    _save_figure(figure, svg_path)

    return {"initial_solid_profile_svg": svg_path}
