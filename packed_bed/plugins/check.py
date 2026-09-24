"""Shared metadata contracts and normal engine construction for plugin checks."""
from pathlib import Path
import sys

import yaml

from packed_bed.parameters import ParameterSpec, plain, resolve_parameters
from .catalogue import Catalogue
from .schema import Manifest
from .storage import inspect_package, load_factory, package_hash


def declarations(factory):
    module = sys.modules[factory.__module__]
    specs = getattr(module, 'PARAMETERS', {})
    if not isinstance(specs, dict) or not all(isinstance(spec, ParameterSpec) for spec in specs.values()):
        raise ValueError('PARAMETERS must map parameter keys to ParameterSpec objects.')
    resolve_parameters(specs)
    return plain(specs)


def family_metadata(family):
    from packed_bed.reactions import ReactionFamily
    if not isinstance(family, ReactionFamily):
        raise ValueError('A kinetics factory must return ReactionFamily.')
    return {'gases': list(family.required_gas_species), 'solids': list(family.required_solid_species),
            'reactions': plain(family.reactions)}


def verify_implementation(factory, definition, parameters, *, kinetics=True):
    if declarations(factory) != plain(definition['parameters']):
        raise ValueError('Python parameter declarations differ from the manifest. Update the manifest to match the implementation.')
    result = factory(parameters)
    if kinetics:
        actual = family_metadata(result)
        if any(actual[key] != plain(definition[key]) for key in actual):
            raise ValueError('Python reaction metadata differs from the manifest. Update the manifest to match the implementation.')
    else:
        from packed_bed.properties import BaseCorrelation
        if not isinstance(result, BaseCorrelation):
            raise ValueError('A correlation factory must return BaseCorrelation.')
    return result


def generate_metadata(folder):
    """Explicit authoring action: execute local factories, then write their declarations."""
    from packed_bed.config.load import read_yaml_mapping
    folder = Path(folder)
    from .storage import validate_imports
    validate_imports(folder)
    data = read_yaml_mapping(folder / 'manifest.yaml', 'plugin manifest')
    digest = package_hash(folder)
    ident = data['id']
    catalogue = Catalogue(paths={ident: folder}, hashes={ident: digest})
    for kind in ('mechanisms', 'correlations'):
        for name, item in data.get(kind, {}).items():
            item.setdefault('name', name)
            if item['implementation'].startswith('builtin:'):
                continue
            factory = load_factory(ident, item['implementation'], catalogue, (digest,))
            item['parameters'] = declarations(factory)
            if kind == 'mechanisms':
                specs = getattr(sys.modules[factory.__module__], 'PARAMETERS', {})
                item.update(family_metadata(factory(resolve_parameters(specs, item.get('values', {})))))
    updated = Manifest.model_validate(data)
    temporary = folder / '.manifest.tmp'
    try:
        temporary.write_text(yaml.safe_dump(updated.model_dump(mode='json', exclude_defaults=True), sort_keys=False, allow_unicode=True), encoding='utf-8')
        temporary.replace(folder / 'manifest.yaml')
    finally:
        temporary.unlink(missing_ok=True)
    return updated


def check_package(source, *, approved=()):
    """Use the same constructors and contracts as normal model assembly."""
    from packed_bed.definitions import species_properties, mechanism_family
    from packed_bed.properties import PropertyRegistry
    with inspect_package(source) as package:
        manifest = package.manifest
        catalogue = Catalogue({manifest.id: manifest}, {manifest.id: package.folder}, {manifest.id: package.digest})
        PropertyRegistry({key: species_properties(spec.model_dump(), manifest.id, catalogue, approved)
                          for key, spec in manifest.species.items()})
        for definition in manifest.mechanisms.values():
            parameters = resolve_parameters(definition.parameters, definition.values)
            mechanism_family(definition, manifest.id, catalogue, parameters, approved)
        for definition in manifest.correlations.values():
            factory = load_factory(manifest.id, definition.implementation, catalogue, approved)
            verify_implementation(factory, definition.model_dump(), resolve_parameters(definition.parameters), kinetics=False)
