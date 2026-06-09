from __future__ import annotations

import math
from typing import Any, Annotated

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field


def _require_string(value: str) -> str:
    if value == "" or value != value.strip():
        raise ValueError("must not be blank or padded with whitespace.")
    return value


def _require_unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"contains duplicates: {', '.join(duplicates)}.")
    return values


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("must be provided as a YAML sequence.")
    return tuple(value)


def _require_fraction_mapping(mapping: dict[str, float]) -> dict[str, float]:
    total = sum(mapping.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"must sum to 1.0 exactly, got {total:.16g}.")
    return mapping


def _require_exact_keys(actual: set[str], expected: tuple[str, ...], label: str) -> None:
    expected_keys = set(expected)
    if actual == expected_keys:
        return
    missing = sorted(expected_keys - actual)
    extra = sorted(actual - expected_keys)
    parts: list[str] = []
    if missing:
        parts.append(f"missing {', '.join(missing)}")
    if extra:
        parts.append(f"unexpected {', '.join(extra)}")
    raise ValueError(f"{label} species mismatch: {'; '.join(parts)}.")


class FrozenConfigModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

ConfigString = Annotated[str, AfterValidator(_require_string)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
UnitFraction = Annotated[float, Field(gt=0.0, lt=1.0, allow_inf_nan=False)]
UniqueStringTuple = Annotated[
    tuple[ConfigString, ...],
    BeforeValidator(_as_tuple),
    AfterValidator(_require_unique_strings),
]
NonEmptyUniqueStringTuple = Annotated[
    tuple[ConfigString, ...],
    BeforeValidator(_as_tuple),
    Field(min_length=1),
    AfterValidator(_require_unique_strings),
]
FractionMapping = Annotated[
    dict[ConfigString, NonNegativeFloat],
    Field(min_length=1),
    AfterValidator(_require_fraction_mapping),
]
