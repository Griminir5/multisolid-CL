from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ReactionDefinition:
    id: str
    name: str
    phase: str
    stoichiometry: Mapping[str, float]
    required_species: tuple[str, ...]
    source_reference: str
    kinetics_hook: str | None = None
    notes: str = ""


DEFAULT_REACTION_CATALOG = {
    "ni_reduction_h2_medrano": ReactionDefinition(
        id="ni_reduction_h2_medrano",
        name="NiO reduction by H2",
        phase="gas_solid",
        stoichiometry={
            "H2": -1.0,
            "NiO": -1.0,
            "Ni": 1.0,
            "H2O": 1.0,
        },
        required_species=("H2", "H2O", "Ni", "NiO"),
        source_reference="Medrano et al., Applied Energy 2015, https://doi.org/10.1016/j.apenergy.2015.08.078",
        kinetics_hook=None,
        notes="Catalogued as metadata only in v1; assembly must reject it until kinetics are implemented.",
    ),
    "ni_reduction_co_medrano": ReactionDefinition(
        id="ni_reduction_co_medrano",
        name="NiO reduction by CO",
        phase="gas_solid",
        stoichiometry={
            "CO": -1.0,
            "NiO": -1.0,
            "Ni": 1.0,
            "CO2": 1.0,
        },
        required_species=("CO", "CO2", "Ni", "NiO"),
        source_reference="Medrano et al., Applied Energy 2015, https://doi.org/10.1016/j.apenergy.2015.08.078",
        kinetics_hook=None,
        notes="Catalogued as metadata only in v1; assembly must reject it until kinetics are implemented.",
    ),
    "ni_oxidation_o2_medrano": ReactionDefinition(
        id="ni_oxidation_o2_medrano",
        name="Ni oxidation by O2",
        phase="gas_solid",
        stoichiometry={
            "O2": -0.5,
            "Ni": -1.0,
            "NiO": 1.0,
        },
        required_species=("O2", "Ni", "NiO"),
        source_reference="Medrano et al., Applied Energy 2015, https://doi.org/10.1016/j.apenergy.2015.08.078",
        kinetics_hook=None,
        notes="Catalogued as metadata only in v1; assembly must reject it until kinetics are implemented.",
    ),
}
