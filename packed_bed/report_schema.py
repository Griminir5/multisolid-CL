"""Import-safe descriptions of expected and recorded result dimensions."""

import numpy as np
import json

from .reports import REPORT_REGISTRY, reporting_times
from .solid_profiles import build_uniform_axial_grid


AXIS_LABELS = {
    "time": ("Time", "s"), "x_cell": ("Cell position", "m"),
    "x_face": ("Face position", "m"), "gas_species": ("Gas species", ""),
    "solid_species": ("Solid species", ""), "reaction": ("Reaction", ""),
}


def quantity_catalog():
    return {
        field.name: {"label": field.name.replace("_", " ").capitalize(),
                     "unit": field.unit, "dimensions": field.dimensions, "report": report}
        for report, spec in REPORT_REGISTRY.items() for field in spec.outputs
    }


def axis_description(name, values=None, *, attrs=None, error=""):
    attrs = attrs or {}
    label, unit = AXIS_LABELS.get(name, (name.replace("_", " ").capitalize(), ""))
    values = None if values is None else np.asarray(values)
    precision = 15
    if values is not None and values.dtype.kind == "f" and values.ndim == 1:
        ordered = np.unique(values)
        scale = np.maximum(np.abs(ordered[:-1]), np.abs(ordered[1:]))
        if np.any(np.diff(ordered) <= 1e-13 * scale):
            precision = 17  # Keep even adjacent floating-point coordinates distinguishable.
    return {"label": attrs.get("long_name", label), "unit": attrs.get("units", unit),
            "values": values, "error": error, "precision": precision}


def describe_dataset(dataset):
    """Discover all recorded variables/axes, including unknown future dimensions."""
    catalog = quantity_catalog()
    quantities = {}
    for name, variable in dataset.data_vars.items():
        known = catalog.get(name, {})
        unit = variable.attrs.get("units", "")
        if not unit and known.get("unit") == "1":
            unit = "1"
        quantities[name] = {
            "label": variable.attrs.get("long_name", known.get("label", name.replace("_", " ").capitalize())),
            "unit": unit or "unit unspecified", "dimensions": variable.dims,
            "report": known.get("report"),
        }
    axes = {}
    for dim in dict.fromkeys(d for q in quantities.values() for d in q["dimensions"]):
        coord = dataset.coords[dim] if dim in dataset.coords else None
        if coord is None or coord.dims != (dim,):
            axes[dim] = axis_description(dim, error=f"No one-dimensional coordinate for {dim}.")
        else:
            axes[dim] = axis_description(dim, coord.values.copy(), attrs=coord.attrs)
    labels = json.loads(dataset.attrs.get('definition_labels', '{}'))
    for dim in ('gas_species', 'solid_species', 'reaction'):
        if dim in axes:
            axes[dim]['labels'] = labels
    return {"quantities": quantities, "axes": axes}


def describe_inputs(documents, *, variables=None, axis_resolvers=None, catalogue=None):
    """Describe requested outputs without resolving a draft or creating a solver.

    Optional configured variables are keyed by the reporter source variable name.
    Additional scientific axes can supply a resolver accepting these documents.
    """
    def get(*path):
        value = documents
        for key in path:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        return value

    selected = get("run", "outputs", "requested_reports")
    selected = selected if isinstance(selected, (list, tuple)) else []
    catalog = quantity_catalog()
    quantities = {name: spec for name, spec in catalog.items() if spec["report"] in selected}
    axes = {d: axis_description(d, error=f"Coordinates for {d} are not yet defined.")
            for spec in quantities.values() for d in spec["dimensions"]}

    def resolve(name, callback):
        if name not in axes:
            return
        try:
            values = callback()
            if isinstance(values, dict):
                axes[name] = axis_description(name, values["values"], attrs=values)
            else:
                axes[name] = axis_description(name, values)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            axes[name]["error"] = f"{axes[name]['label']}: {exc}"

    def times():
        horizon = float(get("run", "simulation", "time_horizon_s"))
        interval = float(get("run", "simulation", "reporting_interval_s"))
        if interval > 0 and horizon / interval > 1_048_576:
            raise ValueError("Too many expected reporting times to list; increase the reporting interval.")
        return reporting_times(horizon, interval)

    def grid(index):
        length = float(get("run", "model", "bed_length_m"))
        cells = get("run", "model", "axial_cells")
        if not np.isfinite(length) or length <= 0 or type(cells) is not int or not 1 <= cells <= 1_000_000:
            raise ValueError("Enter a positive bed length and cell count.")
        return build_uniform_axial_grid(length, cells)[index]

    resolve("time", times)
    resolve("x_cell", lambda: grid(0))
    resolve("x_face", lambda: grid(1))
    for name, document, key in (("gas_species", "chemistry", "gas_species"),
                                ("solid_species", "solids", "solid_species"),
                                ("reaction", "chemistry", "reaction_ids")):
        def labels(document=document, key=key):
            values = get(document, key)
            if not isinstance(values, (list, tuple)) or not all(isinstance(v, str) for v in values):
                raise ValueError("Select the case's species/reactions.")
            return values
        resolve(name, labels)
    resolvers = {name: resolver for report in selected if report in REPORT_REGISTRY
                 for name, resolver in REPORT_REGISTRY[report].axis_resolvers.items()}
    resolvers.update(axis_resolvers or {})
    for name, resolver in resolvers.items():
        resolve(name, lambda resolver=resolver: resolver(documents))

    # Numeric domain points are authoritative when configured domains exist.
    # Categorical domains use scientific labels, not DAE Tools' integer indices.
    for report in selected:
        spec = REPORT_REGISTRY.get(report)
        if spec is None:
            continue
        for field in (*spec.fields, *spec.support):
            variable = (variables or {}).get(field.source)
            for name, domain in zip(field.dimensions, getattr(variable, "Domains", ())):
                if name in axes and name not in ("gas_species", "solid_species", "reaction"):
                    points = getattr(domain, "Points", ())
                    if len(points):
                        unit = str(getattr(domain, "Units", "")) or axes[name]["unit"]
                        axes[name] = axis_description(name, points, attrs={"units": unit})
    if catalogue is not None:
        chemistry = documents.get('chemistry', {})
        mappings = chemistry.get('species_definitions', {})
        for axis in ('gas_species', 'solid_species'):
            if axis in axes and axes[axis]['values'] is not None:
                labels = {}
                for key in axes[axis]['values']:
                    ref = mappings.get(key, 'builtin:' + key)
                    try:
                        labels[key] = key + ' · ' + catalogue.label('species', ref)
                    except ValueError:
                        labels[key] = key + ' — Missing ' + ref
                axes[axis]['labels'] = labels
        if 'reaction' in axes:
            from .definitions import reaction_instance_id
            from .plugins.catalogue import split_ref
            labels = {}
            for instance in chemistry.get('reaction_families', []):
                ref = chemistry.get('mechanisms', {}).get(instance, {}).get('definition', 'builtin:' + instance)
                try:
                    definition = catalogue.get('mechanisms', ref)
                    source = catalogue.manifest(split_ref(ref)[0]).name
                    labels.update({reaction_instance_id(instance, r.id, ref): r.name + ' — ' + source for r in definition.reactions})
                except ValueError:
                    pass
            axes['reaction']['labels'] = labels
    return {"quantities": quantities, "axes": axes}
