"""Small, import-safe parameter declarations shared by built-ins and extensions."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import math
import re
from collections.abc import Mapping
from types import MappingProxyType


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def plain(value):
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


@dataclass(frozen=True)
class ParameterSpec:
    default: float
    unit: str
    description: str = ''
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self):
        if not self.unit.strip():
            raise ValueError('Every editable parameter must state its unit or dimensionless basis.')
        if any(value is not None and not math.isfinite(value) for value in (self.minimum, self.maximum)):
            raise ValueError('Parameter bounds must be finite.')
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError('Parameter bounds must be increasing.')
        self.validate(self.default)

    def validate(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("must be a finite number")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"must be at least {self.minimum:g}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"must be at most {self.maximum:g}")
        return float(value)


def parameter_group(name, values, unit, description, minimum=None, maximum=None):
    """Explicitly expose a scalar or a named group of scalar constants."""
    if isinstance(values, Mapping):
        result = {}
        for key, value in values.items():
            result.update(parameter_group(f"{name}.{key}", value, unit, description, minimum, maximum))
        return result
    return {name: ParameterSpec(float(values), unit, description, minimum, maximum)}


def resolve_parameters(specs, overrides=None):
    overrides = overrides or {}
    unknown = set(overrides) - set(specs)
    if unknown:
        raise ValueError("Parameters are not editable: " + ", ".join(sorted(unknown)))
    values = {}
    for key, spec in specs.items():
        if not re.fullmatch(r'[A-Za-z_]\w*(\.[A-Za-z_]\w*)*', key):
            raise ValueError(f'Invalid editable parameter key: {key}')
        try:
            value = spec.validate(overrides.get(key, spec.default))
        except ValueError as exc:
            raise ValueError(f"Parameter {key}: {exc}") from exc
        target = values
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f'Parameter groups overlap scalar keys: {key}')
        if parts[-1] in target:
            raise ValueError(f'Parameter groups overlap scalar keys: {key}')
        target[parts[-1]] = value
    return freeze(values)

