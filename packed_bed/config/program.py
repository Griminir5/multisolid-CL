from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BeforeValidator, Field

from .validators import FractionMapping, FrozenConfigModel, PositiveFloat, _as_tuple


class HoldStep(FrozenConfigModel):
    kind: Literal["hold"]
    duration_s: PositiveFloat


class ScalarRampStep(FrozenConfigModel):
    kind: Literal["ramp"]
    duration_s: PositiveFloat
    target: PositiveFloat


class CompositionRampStep(FrozenConfigModel):
    kind: Literal["ramp"]
    duration_s: PositiveFloat
    target: FractionMapping


ScalarStep = Annotated[HoldStep | ScalarRampStep, Field(discriminator="kind")]
CompositionStep = Annotated[HoldStep | CompositionRampStep, Field(discriminator="kind")]
ScalarSteps = Annotated[tuple[ScalarStep, ...], BeforeValidator(_as_tuple)]
CompositionSteps = Annotated[tuple[CompositionStep, ...], BeforeValidator(_as_tuple)]


class ScalarChannelConfig(FrozenConfigModel):
    initial: PositiveFloat
    steps: ScalarSteps = Field(default_factory=tuple)


class CompositionChannelConfig(FrozenConfigModel):
    initial: FractionMapping
    steps: CompositionSteps = Field(default_factory=tuple)


class ProgramConfig(FrozenConfigModel):
    inlet_flow: ScalarChannelConfig
    inlet_temperature: ScalarChannelConfig
    outlet_pressure: ScalarChannelConfig
    inlet_composition: CompositionChannelConfig
