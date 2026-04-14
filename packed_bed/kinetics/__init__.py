from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..reactions import ReactionNetwork


KineticsHook = Callable[["KineticsContext"], Any]


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


# Contributor workflow:
# 1. add a ReactionDefinition in packed_bed.reactions
# 2. register one hook function here or in a submodule
# 3. point ReactionDefinition.kinetics_hook to that registry key
# 4. add numeric tests for the hook helpers
KINETICS_HOOK_REGISTRY: dict[str, KineticsHook] = {}


def register_kinetics_hook(name: str):
    def decorator(func: KineticsHook) -> KineticsHook:
        if name in KINETICS_HOOK_REGISTRY:
            raise ValueError(f"Kinetics hook '{name}' is already registered.")
        KINETICS_HOOK_REGISTRY[name] = func
        return func

    return decorator


def resolve_kinetics_hooks(
    reaction_network: ReactionNetwork,
    *,
    hook_registry: Mapping[str, KineticsHook] | None = None,
) -> tuple[KineticsHook, ...]:
    registry = KINETICS_HOOK_REGISTRY if hook_registry is None else hook_registry
    missing_reactions: list[str] = []
    unknown_hooks: list[str] = []
    resolved_hooks: list[KineticsHook] = []

    for reaction in reaction_network.reactions:
        if reaction.kinetics_hook is None:
            missing_reactions.append(reaction.id)
            continue

        hook = registry.get(reaction.kinetics_hook)
        if hook is None:
            unknown_hooks.append(f"{reaction.id} -> {reaction.kinetics_hook}")
            continue

        resolved_hooks.append(hook)

    messages: list[str] = []
    if missing_reactions:
        messages.append(
            "Selected reactions do not have kinetics implementations: " + ", ".join(missing_reactions)
        )
    if unknown_hooks:
        messages.append("Unknown kinetics hooks: " + ", ".join(unknown_hooks))
    if messages:
        raise NotImplementedError(" ".join(messages))

    return tuple(resolved_hooks)


from . import numaguchi_an, xu_froment  # noqa: E402,F401


__all__ = [
    "KINETICS_HOOK_REGISTRY",
    "KineticsContext",
    "KineticsHook",
    "register_kinetics_hook",
    "resolve_kinetics_hooks",
]
