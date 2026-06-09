from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Annotated, Literal

import yaml
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, field_validator, model_validator
from yaml.resolver import BaseResolver

from .axial_schemes import SUPPORTED_SCHEMES
from .properties import PROPERTY_REGISTRY
from .reactions import REACTION_CATALOG, build_reaction_network
from .reporting import REPORT_VARIABLE_REGISTRY










