from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from yaml.resolver import BaseResolver

from .errors import PackedBedValidationError

def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.load(handle, Loader=_UniqueKeyLoader)
    except PackedBedValidationError:
        raise
    except FileNotFoundError as exc:
        raise PackedBedValidationError(f"{label} was not found: {path}") from exc
    except OSError as exc:
        raise PackedBedValidationError(f"Could not read {label}: {path}") from exc
    except yaml.YAMLError as exc:
        raise PackedBedValidationError(f"{label} contains invalid YAML: {path}") from exc

    if not isinstance(data, dict):
        raise PackedBedValidationError(f"{label} must contain a top-level mapping: {path}")
    return data


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line = key_node.start_mark.line + 1
            raise PackedBedValidationError(f"Duplicate key {key!r} at line {line}.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)
