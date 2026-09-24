"""Data-only manifest schema. Importing this module never imports plugin code."""
from __future__ import annotations

import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from packed_bed.parameters import ParameterSpec
from packed_bed.reactions import ReactionDefinition


class DataModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class Correlation(DataModel):
    model: str
    parameters: dict[str, float | list[float]] = Field(default_factory=dict)


class Species(DataModel):
    name: str
    chemical_key: str
    phase: Literal['gas', 'solid']
    mw: float = Field(gt=0, allow_inf_nan=False)
    enthalpy: Correlation
    viscosity: Correlation | None = None
    source: str = ''
    notes: str = ''
    temperature_range: tuple[float, float] = (298.15, 1200.0)

    @model_validator(mode='after')
    def check_phase(self):
        import math
        if not self.name.strip() or not self.chemical_key.strip():
            raise ValueError('Species name and chemical key are required.')
        if (self.phase == 'gas') != (self.viscosity is not None):
            raise ValueError('Gas species require viscosity; solids must not define it.')
        low, high = self.temperature_range
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
            raise ValueError('Temperature range must be finite, positive and increasing.')
        return self


class Mechanism(DataModel):
    name: str
    implementation: str
    gases: tuple[str, ...]
    solids: tuple[str, ...]
    reactions: tuple[ReactionDefinition, ...]
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    values: dict[str, float] = Field(default_factory=dict)
    source: str = ''
    notes: str = ''

    @model_validator(mode='after')
    def check_roles(self):
        roles = set(self.gases) | set(self.solids)
        if set(self.gases) & set(self.solids):
            raise ValueError('Species roles cannot occur in both phases.')
        if len({r.id for r in self.reactions}) != len(self.reactions):
            raise ValueError('Duplicate reaction IDs.')
        for reaction in self.reactions:
            if not set(reaction.all_species) <= roles:
                raise ValueError(f'{reaction.id} has undeclared species roles.')
        if set(self.values) - set(self.parameters):
            raise ValueError('Overrides contain undeclared parameters.')
        return self


class CorrelationType(DataModel):
    kind: Literal['enthalpy', 'viscosity']
    implementation: str
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    name: str = ''


class Manifest(DataModel):
    format_version: Literal[1] = 1
    api_version: Literal[1] = 1
    id: str
    name: str
    version: str = '1'
    description: str = ''
    source: str = ''
    derived_from: str = ''
    species: dict[str, Species] = Field(default_factory=dict)
    mechanisms: dict[str, Mechanism] = Field(default_factory=dict)
    correlations: dict[str, CorrelationType] = Field(default_factory=dict)

    @field_validator('id')
    @classmethod
    def valid_id(cls, value):
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]*', value) or value.lower() == 'builtin':
            raise ValueError('Plugin ID must start with a letter and contain only letters, digits, _, - or .; builtin is reserved.')
        if value.endswith('.') or re.match(r'(?i)^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.|$)', value):
            raise ValueError('Plugin ID must be portable to Windows.')
        return value

    @model_validator(mode='after')
    def valid_local_ids(self):
        if not self.name.strip() or not self.version.strip():
            raise ValueError('Plugin name and version must not be empty.')
        for mapping in (self.species, self.mechanisms, self.correlations):
            if any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]*', key) for key in mapping):
                raise ValueError('Invalid local definition ID.')
        return self
