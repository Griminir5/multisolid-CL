"""Static scientific catalogue shared by case pickers, checking and resolution."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import hashlib
import importlib
from pathlib import Path
from typing import Mapping

from packed_bed.properties import PROPERTY_REGISTRY, PolynomialHeatCapacity, ShomateHeatCapacity, QuadraticViscosity
from .schema import Species, Mechanism, Correlation, Manifest


BUILTIN_CORRELATIONS = {'polynomial': PolynomialHeatCapacity, 'shomate': ShomateHeatCapacity,
                        'quadratic_viscosity': QuadraticViscosity}


def correlation_data(correlation):
    for name, cls in BUILTIN_CORRELATIONS.items():
        if isinstance(correlation, cls):
            return Correlation(model='builtin:' + name, parameters=asdict(correlation))
    raise ValueError('Unknown correlation type.')


@lru_cache(maxsize=1)
def builtin_manifest():
    from packed_bed.kinetics import FAMILY_REGISTRY
    species = {key: Species(name=r.name, chemical_key=key, phase=r.phase, mw=r.mw,
                           enthalpy=correlation_data(r.enthalpy),
                           viscosity=correlation_data(r.viscosity) if r.viscosity else None,
                           source='MultiSolid built-in property data')
               for key, r in PROPERTY_REGISTRY.records.items()}
    mechanisms = {}
    for key, family in FAMILY_REGISTRY.items():
        module = importlib.import_module(next(iter(family.kinetics_hooks.values())).__module__)
        reactions = family.reactions
        # Shared terms in these implementations access the complete family state.
        if key in ('reforming_xu_froment', 'reforming_numaguchi', 'iron_he'):
            required = family.required_gas_species + family.required_solid_species
            reactions = tuple(replace(r, required_species=required) for r in reactions)
        mechanisms[key] = Mechanism(name=key.replace('_', ' ').title(), implementation='builtin:' + key,
                                    gases=family.required_gas_species, solids=family.required_solid_species,
                                    reactions=reactions, parameters=module.PARAMETERS)
    # Built-in is a reserved provider, constructed internally after regular schema validation.
    return Manifest(id='multisolid', name='Built-in', species=species, mechanisms=mechanisms).model_copy(update={'id': 'builtin'})


@lru_cache(maxsize=1)
def builtin_fingerprint():
    digest = hashlib.sha256(builtin_manifest().model_dump_json().encode())
    root = Path(__file__).parents[1]
    for path in sorted([root / 'properties.py', root / 'reactions.py', root / 'parameters.py',
                        *list((root / 'kinetics').glob('*.py'))]):
        digest.update(path.name.encode())
        digest.update(path.read_text(encoding='utf-8').encode('utf-8'))
    return digest.hexdigest()


def split_ref(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Definition references must be non-empty strings.')
    if ':' not in value:
        return 'builtin', value
    provider, local = value.split(':', 1)
    if not provider or not local or ':' in local:
        raise ValueError(f'Invalid definition reference: {value}')
    return provider, local


@dataclass(frozen=True)
class Catalogue:
    manifests: Mapping[str, Manifest] = field(default_factory=dict)
    paths: Mapping[str, Path] = field(default_factory=dict)
    hashes: Mapping[str, str] = field(default_factory=dict)

    def manifest(self, provider):
        if provider == 'builtin':
            return builtin_manifest()
        try:
            return self.manifests[provider]
        except KeyError as exc:
            raise ValueError(f'Plugin {provider} is missing or disabled.') from exc

    def get(self, kind, reference):
        provider, local = split_ref(reference)
        try:
            return getattr(self.manifest(provider), kind)[local]
        except KeyError as exc:
            raise ValueError(f'Unknown {kind} definition {reference}.') from exc

    def entries(self, kind):
        return {f'{provider}:{key}': definition
                for provider in ('builtin', *self.manifests)
                for key, definition in getattr(self.manifest(provider), kind).items()}

    def label(self, kind, reference):
        provider, _ = split_ref(reference)
        definition = self.get(kind, reference)
        return f'{definition.name} — {self.manifest(provider).name}'

    def lock(self, providers):
        providers = set(providers) - {'builtin'}
        return {'builtin': builtin_fingerprint(), 'plugins': {key: self.hashes[key] for key in sorted(providers)}}
