"""Small reactive cases for backend agreement checks and reproducible timings."""

from copy import deepcopy
from pathlib import Path
import yaml


def cases():
    from packed_bed.kinetics import FAMILY_REGISTRY

    root = Path(__file__).resolve().parents[1] / "packed_bed/examples/default_case"
    base = {name: yaml.safe_load((root / (name + ".yaml")).read_text())
            for name in ("run", "program", "solids", "chemistry")}
    result = {}
    for name, family_name, support in (("copper_sio2_san_pio", "copper_sio2_san_pio", None),
                                       ("copper_al2o3_san_pio", "copper_al2o3_san_pio", "Al2O3"),
                                       ("iron_he_mixed_feed", "iron_he", "Al2O3"),
                                       ("mixed_solid_zones", "nickel_medrano", "CaAl2O4")):
        documents = deepcopy(base)
        family = FAMILY_REGISTRY[family_name]
        gas = sorted(set(family.required_gas_species) | {"N2"})
        solids = sorted(set(family.required_solid_species) | ({support} if support else set()))
        composition = {species: .03 for species in gas}
        composition.update(H2=.3, H2O=.1, O2=.001)
        composition["N2"] = 1 - sum(value for key, value in composition.items() if key != "N2")
        documents["chemistry"] = {"gas_species": gas, "reaction_families": [family_name],
                                  "reaction_ids": [reaction.id for reaction in family.reactions]}
        values = {species: 200. for species in solids}
        if support:
            values[support] = 4000.
        zone = {"x_start_m": 0., "x_end_m": 1., "e_b": .5, "e_p": .5, "d_p": .002, "values": values}
        zones = [zone]
        if name == "mixed_solid_zones":
            zone["x_end_m"] = .5
            second = deepcopy(zone)
            second.update(x_start_m=.5, x_end_m=1.)
            second["values"]["Ni"] = 500.
            zones.append(second)
        documents["solids"] = {"solid_species": solids, "initial_profile": {"basis": "bed", "zones": zones}}
        documents["run"]["model"].update(bed_length_m=1., bed_radius_m=.05, axial_cells=3,
                                          ambient_temperature_k=973.15, heat_transfer_coefficient_w_per_m2_k=0.)
        documents["run"]["simulation"].update(time_horizon_s=20., reporting_interval_s=.7,
                                               repeat_program=True, program_mode="separate_channels")
        documents["program"] = {key: {"initial": value, "steps": [{"kind": "hold", "duration_s": 20.}]}
                                for key, value in (("inlet_flow", .05), ("inlet_composition", composition),
                                                   ("inlet_temperature", 973.15), ("outlet_pressure", 101325.))}
        result[name] = (documents, {"description": "Small reactive backend comparison"})
    return result
