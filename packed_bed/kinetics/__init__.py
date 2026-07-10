from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ..reactions import KineticsHook, ReactionFamily, ReactionNetwork


@dataclass(frozen=True)
class KineticsContext:
    model: Any
    idx_cell: Any
    gas_species_index: Mapping[str, int]
    solid_species_index: Mapping[str, int]
    reaction_index: Mapping[str, int]

    def gas_index(self, species_id: str) -> int:
        return self.gas_species_index[species_id]

    def solid_index(self, species_id: str) -> int:
        return self.solid_species_index[species_id]

    def reaction_lookup(self, reaction_id: str) -> int:
        return self.reaction_index[reaction_id]


from .coper_redox import FAMILY as COPPER_SAN_PIO_FAMILY  # noqa: E402
from .fe_redox import FAMILY as IRON_HE_FAMILY  # noqa: E402
from .medrano import FAMILY as NICKEL_MEDRANO_FAMILY  # noqa: E402
from .numaguchi import FAMILY as REFORMING_NUMAGUCHI_FAMILY  # noqa: E402
from .xu_froment import FAMILY as REFORMING_XU_FROMENT_FAMILY  # noqa: E402


FAMILY_REGISTRY: Mapping[str, ReactionFamily] = MappingProxyType({
    NICKEL_MEDRANO_FAMILY.name: NICKEL_MEDRANO_FAMILY,
    REFORMING_XU_FROMENT_FAMILY.name: REFORMING_XU_FROMENT_FAMILY,
    REFORMING_NUMAGUCHI_FAMILY.name: REFORMING_NUMAGUCHI_FAMILY,
    COPPER_SAN_PIO_FAMILY.name: COPPER_SAN_PIO_FAMILY,
    IRON_HE_FAMILY.name: IRON_HE_FAMILY,
})


def load_reaction_families(names: tuple[str, ...]) -> tuple[ReactionFamily, ...]:
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate reaction families: {', '.join(duplicates)}.")
    unknown = sorted(set(names) - set(FAMILY_REGISTRY))
    if unknown:
        raise ValueError(
            f"Unknown reaction families: {', '.join(unknown)}. Available families: "
            f"{', '.join(FAMILY_REGISTRY)}."
        )
    return tuple(FAMILY_REGISTRY[name] for name in names)


def resolve_kinetics_hooks(
    reaction_network: ReactionNetwork,
    families: tuple[ReactionFamily, ...],
) -> tuple[KineticsHook, ...]:
    hooks = {
        reaction_id: hook
        for family in families
        for reaction_id, hook in family.kinetics_hooks.items()
    }
    missing = [reaction.id for reaction in reaction_network.reactions if reaction.id not in hooks]
    if missing:
        raise NotImplementedError(
            "Selected reactions do not have kinetics implementations: " + ", ".join(missing)
        )
    return tuple(hooks[reaction.id] for reaction in reaction_network.reactions)


__all__ = (
    "FAMILY_REGISTRY",
    "KineticsContext",
    "load_reaction_families",
    "resolve_kinetics_hooks",
)
