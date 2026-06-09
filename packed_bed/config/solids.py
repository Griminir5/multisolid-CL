from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, model_validator

from .validators import (
    ConfigString,
    FrozenConfigModel,
    NonEmptyUniqueStringTuple,
    NonNegativeFloat,
    PositiveFloat,
    UnitFraction,
    _as_tuple,
    _require_exact_keys,
)


class SolidZoneConfig(FrozenConfigModel):
    x_start_m: NonNegativeFloat
    x_end_m: PositiveFloat
    e_b: UnitFraction
    e_p: UnitFraction
    d_p: PositiveFloat
    values: dict[ConfigString, NonNegativeFloat]

    @model_validator(mode="after")
    def validate_bounds(self) -> "SolidZoneConfig":
        if self.x_end_m <= self.x_start_m:
            raise ValueError("x_end_m must be greater than x_start_m.")
        if not self.values:
            raise ValueError("values must not be empty.")
        return self


SolidZones = Annotated[
    tuple[SolidZoneConfig, ...],
    BeforeValidator(_as_tuple),
    Field(min_length=1),
]


class SolidProfileConfig(FrozenConfigModel):
    basis: Literal["solid", "bed"]
    zones: SolidZones


class SolidConfig(FrozenConfigModel):
    solid_species: NonEmptyUniqueStringTuple
    initial_profile: SolidProfileConfig

    @model_validator(mode="after")
    def validate_zone_species(self) -> "SolidConfig":
        for zone_index, zone in enumerate(self.initial_profile.zones):
            _require_exact_keys(
                set(zone.values),
                self.solid_species,
                f"solids.initial_profile.zones[{zone_index}].values",
            )
        return self
