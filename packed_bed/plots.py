"""Static pre-run diagrams and post-run plots."""

from __future__ import annotations

from importlib.util import find_spec
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import Case
from .programs import DEFAULT_SMOOTH_RAMP_WIDTH_S
from .reports import RunResult, load_dataset
from .solid_profiles import (
    build_cell_scalar_profile,
    build_face_scalar_profile,
    build_uniform_axial_grid,
    convert_solid_profile_to_bed_volume,
    gas_fraction_from_voidages,
    solid_fraction_from_voidages,
    zone_edges,
)


def _segment_changes_value(segment) -> bool:
    start_value = np.asarray(segment.start_value, dtype=float)
    end_value = np.asarray(segment.end_value, dtype=float)
    return not np.allclose(start_value, end_value, rtol=0.0, atol=1e-12)


def _smoothed_program_sample_times(programs, *, final_time: float, smooth_ramp_width_s: float) -> np.ndarray:
    final_time = float(final_time)
    if final_time <= 0.0:
        return np.array([0.0], dtype=float)

    times = {0.0, final_time}
    total_segments = sum(len(program.segments) for program in programs)
    baseline_count = max(400, min(2500, 20 * total_segments + 400))
    times.update(float(value) for value in np.linspace(0.0, final_time, baseline_count))

    width = float(smooth_ramp_width_s)
    if width <= 0.0:
        raise ValueError("smooth_ramp_width_s must be positive.")
    edge_offsets = width * np.array(
        [-8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    )
    for program in programs:
        for segment in program.segments:
            start_time = float(segment.start_time)
            end_time = float(segment.end_time)
            for edge_time in (start_time, end_time):
                for sample_time in edge_time + edge_offsets:
                    if 0.0 <= sample_time <= final_time:
                        times.add(float(sample_time))

            if _segment_changes_value(segment):
                duration_s = max(end_time - start_time, 0.0)
                interior_count = max(
                    8,
                    min(80, int(math.ceil(duration_s / width)) * 4),
                )
                times.update(float(value) for value in np.linspace(start_time, end_time, interior_count))

    return np.asarray(sorted(times), dtype=float)


def _series_from_smoothed_program(
    program,
    times: np.ndarray,
    *,
    smooth_ramp_width_s: float,
) -> np.ndarray:
    return np.asarray(
        [
            program.value_at(float(time_s), smooth_ramp_width_s=smooth_ramp_width_s)
            for time_s in times
        ],
        dtype=float,
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


def _save_figure(figure, path: Path) -> None:
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


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

    flow_values = _series_from_smoothed_program(
        inlet_flow_program,
        sample_times,
        smooth_ramp_width_s=smooth_ramp_width_s,
    )
    axes[0].plot(sample_times, flow_values, color="#1d3557", linewidth=2)
    axes[0].set_ylabel("mol/s")
    axes[0].set_title("Inlet Flow")

    temp_values = _series_from_smoothed_program(
        inlet_temperature_program,
        sample_times,
        smooth_ramp_width_s=smooth_ramp_width_s,
    )
    axes[1].plot(sample_times, temp_values, color="#e76f51", linewidth=2)
    axes[1].set_ylabel("K")
    axes[1].set_title("Inlet Temperature")

    pressure_values = _series_from_smoothed_program(
        outlet_pressure_program,
        sample_times,
        smooth_ramp_width_s=smooth_ramp_width_s,
    )
    axes[2].plot(sample_times, pressure_values, color="#264653", linewidth=2)
    axes[2].set_ylabel("Pa")
    axes[2].set_title("Outlet Pressure")

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
    _save_figure(figure, svg_path)

    return {"operating_program_svg": svg_path}


def render_initial_solid_profile(case: Case, output_dir) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zones = case.solids.initial_profile.zones
    cell_centers, face_positions = build_uniform_axial_grid(
        case.run.model.bed_length_m,
        case.run.model.axial_cells,
    )
    authored_edges = zone_edges(case.solids)
    e_b = build_cell_scalar_profile(case.solids, cell_centers, "e_b")
    e_p = build_cell_scalar_profile(case.solids, cell_centers, "e_p")
    gas_fraction = gas_fraction_from_voidages(e_b, e_p)
    solid_fraction = solid_fraction_from_voidages(e_b, e_p)
    bed_basis_concentration = convert_solid_profile_to_bed_volume(
        case.solids,
        cell_centers,
        solid_fraction,
        case.solids.solid_species,
    )
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
    _save_figure(figure, svg_path)

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


ACTIVE_SPECIES_ATOL = 1.0e-10


def _result_dataset(result: RunResult):
    if result.dataset is not None:
        return result.dataset
    if result.results_path is None:
        raise ValueError("RunResult does not contain a dataset or results path.")
    return load_dataset(result.results_path)


def _edges(centers) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    differences = np.diff(centers)
    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * differences[0]
    edges[-1] = centers[-1] + 0.5 * differences[-1]
    return edges


def _save(figure, path: Path) -> Path:
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _composition_plot(dataset, output_directory: Path, image_format: str) -> Path:
    required = ("inlet_composition", "outlet_composition")
    if any(name not in dataset for name in required):
        raise ValueError("Composition plotting requires gas mole fraction and flux reports.")
    inlet = dataset.inlet_composition.values
    outlet = dataset.outlet_composition.values
    active = [
        index for index in range(dataset.sizes["gas_species"])
        if max(np.nanmax(abs(inlet[:, index])), np.nanmax(abs(outlet[:, index])))
        > ACTIVE_SPECIES_ATOL
    ] or list(range(dataset.sizes["gas_species"]))

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.get_cmap("tab10", max(len(active), 1))
    for axis, title, values in (
        (axes[0], "Inlet Composition Program", inlet),
        (axes[1], "Outlet Composition", outlet),
    ):
        for color_index, species_index in enumerate(active):
            axis.plot(
                dataset.time,
                values[:, species_index],
                color=colors(color_index),
                label=str(dataset.gas_species.values[species_index]),
            )
        axis.set(title=title, ylabel="Mole fraction [-]", ylim=(-0.02, 1.02))
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    axes[1].set_xlabel("Time [s]")
    figure.tight_layout()
    return _save(figure, output_directory / f"outlet_composition_vs_time.{image_format}")


def _outlet_plot(dataset, output_directory: Path, image_format: str) -> Path:
    required = ("outlet_temperature", "pressure_drop", "outlet_flow")
    if any(name not in dataset for name in required):
        raise ValueError("Outlet plotting requires temperature, pressure, and gas flux reports.")
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for axis, variable, color, label, title in (
        (axes[0], "outlet_temperature", "#d94841", "Temperature [K]", "Outlet Temperature"),
        (axes[1], "pressure_drop", "#1d3557", "Delta P [Pa]", "Pressure Drop"),
        (axes[2], "outlet_flow", "#2a9d8f", "Flowrate [mol/s]", "Outlet Flowrate"),
    ):
        axis.plot(dataset.time, dataset[variable], color=color, linewidth=2)
        axis.set(ylabel=label, title=title)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    figure.tight_layout()
    return _save(figure, output_directory / f"outlet_conditions_vs_time.{image_format}")


def _profile_plot(dataset, output_directory: Path, image_format: str) -> Path:
    if "temperature" not in dataset or "pressure" not in dataset:
        raise ValueError("Profile plotting requires temperature and pressure reports.")
    figure, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for axis, variable, color_map, label, title in (
        (axes[0], "temperature", "inferno", "Temperature [K]", "Temperature Profile"),
        (axes[1], "pressure", "viridis", "Pressure [Pa]", "Pressure Profile"),
    ):
        mesh = axis.pcolormesh(
            _edges(dataset.time),
            _edges(dataset.x_cell),
            dataset[variable].transpose("x_cell", "time"),
            shading="auto",
            cmap=color_map,
            rasterized=True,
        )
        figure.colorbar(mesh, ax=axis, label=label)
        axis.set(title=title, ylabel="Axial position [m]")
    axes[-1].set_xlabel("Time [s]")
    figure.tight_layout()
    return _save(figure, output_directory / f"temperature_profile_vs_time.{image_format}")


def render_run_result_plots(
    result: RunResult,
    output_directory: str | Path | None = None,
    *,
    image_format: str = "svg",
) -> dict[str, Path]:
    dataset = _result_dataset(result)
    output = Path(output_directory or result.case.artifacts_directory)
    output.mkdir(parents=True, exist_ok=True)
    return {
        f"outlet_composition_{image_format}": _composition_plot(dataset, output, image_format),
        f"outlet_conditions_{image_format}": _outlet_plot(dataset, output, image_format),
        f"temperature_profile_{image_format}": _profile_plot(dataset, output, image_format),
    }


__all__ = ("generate_artifacts", "render_run_result_plots")
