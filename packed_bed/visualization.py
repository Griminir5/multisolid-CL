from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx

from .config import RunBundle
from .programs import default_inlet_composition
from .properties import DEFAULT_PROPERTY_REGISTRY
from .reactions import DEFAULT_REACTION_CATALOG


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
    property_registry=DEFAULT_PROPERTY_REGISTRY,
    reaction_catalog=DEFAULT_REACTION_CATALOG,
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

    for species_id in run_bundle.chemistry.solid_species:
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

    png_path = output_dir / "system_graph.png"
    svg_path = output_dir / "system_graph.svg"
    figure.savefig(png_path, dpi=200)
    figure.savefig(svg_path)
    plt.close(figure)

    return {
        "system_graph_png": png_path,
        "system_graph_svg": svg_path,
    }


def render_operating_program(run_bundle: RunBundle, output_dir) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_horizon = run_bundle.run.time_horizon_s
    gas_species = run_bundle.chemistry.gas_species

    inlet_flow_program = (
        None if run_bundle.program.inlet_flow is None else run_bundle.program.inlet_flow.compile_program()
    )
    inlet_temperature_program = (
        None
        if run_bundle.program.inlet_temperature is None
        else run_bundle.program.inlet_temperature.compile_program()
    )
    outlet_pressure_program = (
        None if run_bundle.program.outlet_pressure is None else run_bundle.program.outlet_pressure.compile_program()
    )
    inlet_composition_program = (
        None
        if run_bundle.program.inlet_composition is None
        else run_bundle.program.inlet_composition.compile_program(gas_species)
    )

    figure, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    flow_times, flow_values = _series_from_segments(
        [] if inlet_flow_program is None else inlet_flow_program.build_segments(time_horizon=time_horizon),
        0.785 if inlet_flow_program is None else inlet_flow_program.initial_value,
    )
    axes[0].plot(flow_times, flow_values, color="#1d3557", linewidth=2)
    axes[0].set_ylabel("mol/s")
    axes[0].set_title("Inlet Flow")

    temp_times, temp_values = _series_from_segments(
        [] if inlet_temperature_program is None else inlet_temperature_program.build_segments(time_horizon=time_horizon),
        500.0 if inlet_temperature_program is None else inlet_temperature_program.initial_value,
    )
    axes[1].plot(temp_times, temp_values, color="#e76f51", linewidth=2)
    axes[1].set_ylabel("K")
    axes[1].set_title("Inlet Temperature")

    pressure_times, pressure_values = _series_from_segments(
        [] if outlet_pressure_program is None else outlet_pressure_program.build_segments(time_horizon=time_horizon),
        1.01325e5 if outlet_pressure_program is None else outlet_pressure_program.initial_value,
    )
    axes[2].plot(pressure_times, pressure_values, color="#264653", linewidth=2)
    axes[2].set_ylabel("Pa")
    axes[2].set_title("Outlet Pressure")

    if inlet_composition_program is None:
        composition_initial = default_inlet_composition(gas_species)
        composition_segments = []
    else:
        composition_initial = inlet_composition_program.initial_value
        composition_segments = inlet_composition_program.build_segments(time_horizon=time_horizon)

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

    png_path = output_dir / "operating_program.png"
    svg_path = output_dir / "operating_program.svg"
    figure.savefig(png_path, dpi=200)
    figure.savefig(svg_path)
    plt.close(figure)

    return {
        "operating_program_png": png_path,
        "operating_program_svg": svg_path,
    }
