from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ..reactions import KineticsHook, ReactionFamily, ReactionNetwork
from .copper_al2o3 import FAMILY as COPPER_AL2O3_SAN_PIO_FAMILY
from .copper_sio2 import FAMILY as COPPER_SIO2_SAN_PIO_FAMILY
from .iron_he import FAMILY as IRON_HE_FAMILY
from .nickel_medrano import FAMILY as NICKEL_MEDRANO_FAMILY
from .reforming_numaguchi import FAMILY as REFORMING_NUMAGUCHI_FAMILY
from .reforming_xu_froment import FAMILY as REFORMING_XU_FROMENT_FAMILY


FAMILY_REGISTRY: Mapping[str, ReactionFamily] = MappingProxyType({
    NICKEL_MEDRANO_FAMILY.name: NICKEL_MEDRANO_FAMILY,
    REFORMING_XU_FROMENT_FAMILY.name: REFORMING_XU_FROMENT_FAMILY,
    REFORMING_NUMAGUCHI_FAMILY.name: REFORMING_NUMAGUCHI_FAMILY,
    COPPER_SIO2_SAN_PIO_FAMILY.name: COPPER_SIO2_SAN_PIO_FAMILY,
    COPPER_AL2O3_SAN_PIO_FAMILY.name: COPPER_AL2O3_SAN_PIO_FAMILY,
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
    "load_reaction_families",
    "resolve_kinetics_hooks",
)
