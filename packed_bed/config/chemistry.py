from __future__ import annotations

from .validators import FrozenConfigModel, NonEmptyUniqueStringTuple, UniqueStringTuple


class ChemistryConfig(FrozenConfigModel):
    gas_species: NonEmptyUniqueStringTuple
    reaction_ids: UniqueStringTuple
