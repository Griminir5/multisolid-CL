from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx

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


@dataclass(frozen=True)
class SystemGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


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
        nodes.append(
            GraphNode(
                id=reaction_node_id,
                label=reaction.id,
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
                    )
                )
            elif coefficient > 0.0:
                edges.append(
                    GraphEdge(
                        source=reaction_node_id,
                        target=species_id,
                        label=str(abs(coefficient)),
                        coefficient=coefficient,
                    )
                )

    return SystemGraph(nodes=tuple(nodes), edges=tuple(edges))


def _series_from_segments(segments, initial_value):
    if not segments:
        return [0.0], [initial_value]

    times = [0.0]
    values = [initial_value]
    for segment in segments:
        if times[-1] != segment.start_time or values[-1] != segment.start_value:
            times.append(float(segment.start_time))
            values.append(float(segment.start_value))
        times.append(float(segment.end_time))
        values.append(float(segment.end_value))
    return times, values


def render_system_graph(system_graph: SystemGraph, output_dir) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = nx.DiGraph()
    for node in system_graph.nodes:
        graph.add_node(node.id, label=node.label, color=node.color, kind=node.kind, phase=node.phase)
    for edge in system_graph.edges:
        graph.add_edge(edge.source, edge.target, label=edge.label)

    species_nodes = [node.id for node in system_graph.nodes if node.kind == "species"]
    reaction_nodes = [node.id for node in system_graph.nodes if node.kind == "reaction"]

    positions = {}
    for index, node_id in enumerate(species_nodes):
        positions[node_id] = (0.0, -index)
    for index, node_id in enumerate(reaction_nodes):
        positions[node_id] = (1.0, -index)

    figure, axis = plt.subplots(figsize=(10, max(4, 0.8 * max(1, len(system_graph.nodes)))))
    axis.set_title("Species and Reaction System Graph")
    axis.axis("off")

    node_colors = [graph.nodes[node_id]["color"] for node_id in graph.nodes]
    nx.draw_networkx(
        graph,
        pos=positions,
        ax=axis,
        labels={node.id: node.label for node in system_graph.nodes},
        node_color=node_colors,
        node_size=2600,
        font_size=8,
        arrows=True,
        arrowsize=18,
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos=positions,
        edge_labels={(edge.source, edge.target): edge.label for edge in system_graph.edges},
        font_size=8,
        ax=axis,
    )
    figure.tight_layout()

    svg_path = output_dir / "system_graph.svg"
    figure.savefig(svg_path)
    plt.close(figure)

    return {"system_graph_svg": svg_path}


def render_operating_program(run_bundle: RunBundle, output_dir) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_horizon = run_bundle.run.time_horizon_s
    gas_species = run_bundle.chemistry.gas_species

    inlet_flow_program = run_bundle.program.inlet_flow.compile_program()
    inlet_temperature_program = run_bundle.program.inlet_temperature.compile_program()
    outlet_pressure_program = run_bundle.program.outlet_pressure.compile_program()
    inlet_composition_program = run_bundle.program.inlet_composition.compile_program(gas_species)

    figure, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    flow_times, flow_values = _series_from_segments(
        inlet_flow_program.build_segments(),
        inlet_flow_program.initial_value,
    )
    axes[0].plot(flow_times, flow_values, color="#1d3557", linewidth=2)
    axes[0].set_ylabel("mol/s")
    axes[0].set_title("Inlet Flow")

    temp_times, temp_values = _series_from_segments(
        inlet_temperature_program.build_segments(),
        inlet_temperature_program.initial_value,
    )
    axes[1].plot(temp_times, temp_values, color="#e76f51", linewidth=2)
    axes[1].set_ylabel("K")
    axes[1].set_title("Inlet Temperature")

    pressure_times, pressure_values = _series_from_segments(
        outlet_pressure_program.build_segments(),
        outlet_pressure_program.initial_value,
    )
    axes[2].plot(pressure_times, pressure_values, color="#264653", linewidth=2)
    axes[2].set_ylabel("Pa")
    axes[2].set_title("Outlet Pressure")

    composition_initial = inlet_composition_program.initial_value
    composition_segments = inlet_composition_program.build_segments()

    if composition_segments:
        times = [0.0]
        series_by_species = [[float(value)] for value in composition_initial]
        for segment in composition_segments:
            times.append(float(segment.end_time))
            for species_index, species_series in enumerate(series_by_species):
                species_series.append(float(segment.end_value[species_index]))
    else:
        times = [0.0]
        series_by_species = [[float(value)] for value in composition_initial]

    for species_id, series in zip(gas_species, series_by_species):
        axes[3].plot(times, series, linewidth=2, label=species_id)
    axes[3].set_ylabel("Mole fraction")
    axes[3].set_title("Inlet Composition")
    axes[3].set_xlabel("Time [s]")
    if gas_species:
        axes[3].legend(loc="upper right", fontsize=8)

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.set_xlim(0.0, time_horizon)

    figure.tight_layout()

    svg_path = output_dir / "operating_program.svg"
    figure.savefig(svg_path)
    plt.close(figure)

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

    figure, axes = plt.subplots(4, 1, figsize=(11, 15), sharex=False)

    for species_id in run_bundle.solids.solid_species:
        zone_values = [float(zone.values_mol_per_m3[species_id]) for zone in zones]
        axes[0].stairs(zone_values, authored_edges, label=species_id, linewidth=2)
    axes[0].set_title("Initial Solid Concentration Input")
    axes[0].set_ylabel(unit_label)

    for species_index, species_id in enumerate(run_bundle.solids.solid_species):
        axes[1].step(
            cell_centers,
            bed_basis_concentration[species_index],
            where="mid",
            label=species_id,
            linewidth=2,
        )
    axes[1].set_title("Initial Solid Concentration on Bed-Volume Basis")
    axes[1].set_ylabel("mol/m^3 bed")

    axes[2].step(cell_centers, e_b, where="mid", linewidth=2, label="e_b")
    axes[2].step(cell_centers, e_p, where="mid", linewidth=2, label="e_p")
    axes[2].step(cell_centers, gas_fraction, where="mid", linewidth=1.5, linestyle="--", label="gasfrac")
    axes[2].step(cell_centers, solid_fraction, where="mid", linewidth=1.5, linestyle=":", label="solfrac")
    axes[2].set_title("Voidages and Volume Fractions")
    axes[2].set_ylabel("Fraction")
    axes[2].set_ylim(0.0, 1.05)

    axes[3].plot(face_positions, d_p, marker="o", linewidth=2, color="#7f5539")
    axes[3].set_title("Particle Characteristic Length on Face Domain")
    axes[3].set_ylabel("d_p [m]")
    axes[3].set_xlabel("Axial position [m]")

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.set_xlim(0.0, run_bundle.run.model.bed_length_m)

    if run_bundle.solids.solid_species:
        axes[0].legend(loc="upper right", fontsize=8)
        axes[1].legend(loc="upper right", fontsize=8)
    axes[2].legend(loc="upper right", fontsize=8)

    figure.tight_layout()

    svg_path = output_dir / "initial_solid_profile.svg"
    figure.savefig(svg_path)
    plt.close(figure)

    return {"initial_solid_profile_svg": svg_path}
