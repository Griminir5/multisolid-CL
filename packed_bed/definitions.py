"""One metadata selection and one authoritative, case-local engine handoff."""
from __future__ import annotations
from dataclasses import dataclass, fields, replace
from collections.abc import Mapping
from typing import Any

from .parameters import freeze, plain, resolve_parameters
from .properties import PropertyRegistry, SpeciesProperties
from .reactions import ReactionDefinition, ReactionFamily, ReactionNetwork, KineticsContext, build_reaction_network
from .plugins.catalogue import Catalogue, BUILTIN_CORRELATIONS, split_ref


@dataclass(frozen=True)
class DefinitionSelection:
    gas_species: tuple[str, ...]
    solid_species: tuple[str, ...]
    reaction_ids: tuple[str, ...]
    species: Mapping[str, Any]
    mechanisms: Mapping[str, Any]
    lock: Mapping[str, Any]

    def __post_init__(self):
        for key in ('species', 'mechanisms', 'lock'):
            object.__setattr__(self, key, freeze(getattr(self, key)))
        for key in ('gas_species', 'solid_species', 'reaction_ids'):
            object.__setattr__(self, key, tuple(getattr(self, key)))

    def to_dict(self):
        return {key: plain(getattr(self, key)) for key in ('gas_species', 'solid_species', 'reaction_ids',
                                                        'species', 'mechanisms', 'lock')}

    @property
    def labels(self):
        labels = {key: value['label'] for key, value in self.species.items()}
        for mechanism in self.mechanisms.values():
            labels.update({r['id']: f"{r['name']} — {mechanism['source']}" for r in mechanism['reactions']})
        return labels

    @property
    def cautions(self):
        seen, cautions = {}, []
        for instance, item in self.mechanisms.items():
            for reaction in item['reactions']:
                if reaction['id'] not in self.reaction_ids:
                    continue
                key = tuple(sorted(reaction['stoichiometry'].items()))
                if key in seen and seen[key] != instance:
                    cautions.append(f'{seen[key]} and {instance} contain overlapping reactions; their rates add.')
                seen[key] = instance
        return tuple(dict.fromkeys(cautions))


@dataclass(frozen=True)
class DefinitionEnvironment:
    properties: PropertyRegistry
    reaction_network: ReactionNetwork
    rate_hooks: tuple
    selection: DefinitionSelection

    def __post_init__(self):
        network, selection = self.reaction_network, self.selection
        if (network.gas_species, network.solid_species, network.reaction_ids) != (
                selection.gas_species, selection.solid_species, selection.reaction_ids):
            raise ValueError('Resolved network does not match its definition selection.')
        if len(self.rate_hooks) != network.reaction_count or not all(callable(hook) for hook in self.rate_hooks):
            raise ValueError('Every selected reaction must have one bound rate hook.')
        if set(self.properties.records) != set(selection.species):
            raise ValueError('Resolved properties do not match selected components.')
        for component, item in selection.species.items():
            record, spec = self.properties.get_record(component), item['definition']
            if (record.phase, record.mw) != (spec['phase'], spec['mw']) or record.enthalpy is None or (spec['phase'] == 'gas' and record.viscosity is None):
                raise ValueError(f'Resolved properties for {component} differ from its selection.')
            for name in ('enthalpy', 'viscosity'):
                correlation = spec[name]
                if correlation and correlation['model'].startswith('builtin:'):
                    expected = _correlation(correlation, 'builtin', Catalogue(), ())
                    if getattr(record, name) != expected:
                        raise ValueError(f'Resolved {name} for {component} differs from its selection.')
        selected = {r['id']: (r, item) for item in selection.mechanisms.values() for r in item['reactions']}
        for reaction, hook in zip(network.reactions, self.rate_hooks):
            if getattr(hook, '_reaction_id', None) != reaction.id:
                raise ValueError('Bound rate hooks are not in the selected reaction order.')
            metadata, owner = selected[reaction.id]
            if {field.name: plain(getattr(reaction, field.name)) for field in fields(reaction)} != plain(metadata):
                raise ValueError('Resolved reactions differ from their metadata selection.')
            if (getattr(hook, '_properties', None) is not self.properties
                    or getattr(hook, '_parameters', None) is not owner['parameters']
                    or getattr(hook, '_bindings', None) is not owner['bindings']):
                raise ValueError('Rate hooks must reference this environment’s properties, parameters and bindings.')
        for ids, matrix in ((network.gas_species, network.gas_source_matrix), (network.solid_species, network.solid_source_matrix)):
            if matrix != tuple(tuple(r.source_coefficient(key) for r in network.reactions) for key in ids):
                raise ValueError('Reaction source matrices differ from the selected stoichiometry.')

    def matches(self, chemistry, solids):
        if set(chemistry.species_definitions) - set(self.selection.species) or set(chemistry.mechanisms) - set(self.selection.mechanisms):
            raise ValueError('Case contains selections outside the supplied environment.')
        if (tuple(chemistry.gas_species), tuple(solids.solid_species), tuple(chemistry.reaction_ids)) != (
                self.selection.gas_species, self.selection.solid_species, self.selection.reaction_ids):
            raise ValueError('Case species/reactions differ from the supplied environment.')
        for component, value in self.selection.species.items():
            ref = chemistry.species_definitions.get(component, 'builtin:' + component)
            if split_ref(ref) != split_ref(value['reference']):
                raise ValueError(f'Component {component} differs from the supplied environment.')
        if set(chemistry.reaction_families) != set(self.selection.mechanisms):
            raise ValueError('Case mechanisms differ from the supplied environment.')
        for instance, value in self.selection.mechanisms.items():
            declared = chemistry.mechanisms.get(instance)
            ref = declared.definition if declared else 'builtin:' + instance
            bindings = declared.bindings if declared else {}
            if split_ref(ref) != split_ref(value['reference']) or any(value['bindings'].get(k) != v for k,v in bindings.items()):
                raise ValueError(f'Mechanism {instance} differs from the supplied environment.')
            from .plugins.schema import Mechanism
            definition = Mechanism.model_validate(plain(value['definition']))
            resolved, errors = resolve_bindings(instance, definition, bindings, self.selection.species, chemistry.reaction_ids, ref)
            if errors or resolved != value['bindings']:
                raise ValueError(f'Mechanism {instance} bindings differ from the supplied environment.')


def reaction_instance_id(instance, local, reference):
    from .plugins.catalogue import builtin_manifest
    return local if instance in builtin_manifest().mechanisms else f'{instance}/{local}'


def binding_candidates(definition, species):
    return {role: tuple(key for key, spec in species.items() if (spec['chemical_key'], spec['phase']) == (role, phase))
            for phase, roles in (('gas', definition.gases), ('solid', definition.solids)) for role in roles}


def resolve_bindings(instance, definition, configured, species, selected_ids, reference, *, repair=False):
    bindings, errors = dict(configured), []
    candidates = binding_candidates(definition, {key: item['definition'] for key, item in species.items()})
    needed = {key for r in definition.reactions if reaction_instance_id(instance, r.id, reference) in selected_ids
              for key in r.required_species}
    if set(bindings) - candidates.keys():
        errors.append(f'{instance} has undeclared species bindings.')
    for role, matches in candidates.items():
        if repair and bindings.get(role) not in matches:
            bindings.pop(role, None)
        if role not in bindings:
            if role in matches:
                bindings[role] = role
            elif matches and (repair or len(matches) == 1):
                bindings[role] = matches[0]
            elif role in needed:
                errors.append(f'{instance}: bind {role} explicitly; ' + ('several components match.' if matches else f'requires unselected species: {role}.'))
        if role in bindings and bindings[role] not in matches:
            errors.append(f'{instance}: {role} must bind to a selected matching component.')
    return bindings, errors


def bound_reactions(instance, definition, bindings, reference):
    return tuple(replace(reaction, id=reaction_instance_id(instance, reaction.id, reference),
        stoichiometry=freeze({bindings.get(key, key): v for key, v in reaction.stoichiometry.items()}),
        required_species=tuple(bindings.get(key, key) for key in reaction.required_species),
        catalyst_species=tuple(bindings.get(key, key) for key in reaction.catalyst_species)) for reaction in definition.reactions)


def select_definitions(chemistry, solids, catalogue=None):
    """Resolve metadata only; safe for unapproved code plugins and GUI browsing."""
    catalogue = catalogue or Catalogue()
    gases, solids_ids = tuple(chemistry.gas_species), tuple(solids.solid_species)
    species, mechanisms, providers, errors = {}, {}, set(), []
    for phase, ids in (('gas', gases), ('solid', solids_ids)):
        for component in ids:
            reference = chemistry.species_definitions.get(component, 'builtin:' + component)
            try:
                definition = catalogue.get('species', reference)
                if definition.phase != phase:
                    raise ValueError(f"Species '{component}' is phase '{definition.phase}', not {phase}.")
                provider, _ = split_ref(reference)
                providers.add(provider)
                species[component] = {'reference': reference, 'definition': definition.model_dump(mode='json'),
                                      'label': f'{component} · {definition.name} — {catalogue.manifest(provider).name}'}
            except ValueError as exc:
                errors.append(f"Unknown {phase} species '{component}': {exc}")
    if set(chemistry.species_definitions) - set(gases + solids_ids):
        errors.append('Species definition mappings contain unselected component IDs.')
    if set(chemistry.mechanisms) - set(chemistry.reaction_families):
        errors.append('Mechanism mappings contain unselected instances.')
    catalog_ids = set()
    for instance in chemistry.reaction_families:
        config = chemistry.mechanisms.get(instance)
        reference = config.definition if config else 'builtin:' + instance
        try:
            definition = catalogue.get('mechanisms', reference)
        except ValueError as exc:
            errors.append(f'chemistry.reaction_families: Unknown reaction families: {instance}. {exc}')
            continue
        provider, _ = split_ref(reference)
        providers.add(provider)
        bindings, binding_errors = resolve_bindings(instance, definition, config.bindings if config else {},
                                                   species, chemistry.reaction_ids, reference)
        errors.extend(binding_errors)
        parameters = resolve_parameters(definition.parameters, definition.values)
        reactions = plain(bound_reactions(instance, definition, bindings, reference))
        for reaction in reactions:
            if reaction['id'] in catalog_ids:
                errors.append(f'Duplicate reaction instance ID: {reaction["id"]}. Use distinct mechanism instance names.')
            catalog_ids.add(reaction['id'])
        mechanisms[instance] = {'reference': reference, 'definition': definition.model_dump(mode='json'),
                                'bindings': bindings, 'parameters': plain(parameters), 'reactions': reactions,
                                'source': catalogue.manifest(provider).name}
    for reaction_id in chemistry.reaction_ids:
        if reaction_id not in catalog_ids:
            errors.append(f"chemistry.reaction_ids contains unknown id '{reaction_id}'.")
    if not errors:
        from .reactions import _validate_reaction_phase_membership
        for item in mechanisms.values():
            for record in item['reactions']:
                if record['id'] in chemistry.reaction_ids:
                    try:
                        _validate_reaction_phase_membership(ReactionDefinition(**record), gases, solids_ids)
                    except ValueError as exc:
                        errors.append(str(exc))
    if errors:
        raise ValueError('\n'.join(errors))
    return DefinitionSelection(gases, solids_ids, tuple(chemistry.reaction_ids), freeze(species),
                               freeze(mechanisms), freeze(catalogue.lock(providers)))


def _correlation(spec, provider, catalogue, approved):
    model = spec['model']
    parameters = plain(spec['parameters'])
    if model.startswith('builtin:'):
        name = model.split(':', 1)[1]
        if name not in BUILTIN_CORRELATIONS:
            raise ValueError(f'Unknown built-in correlation {name}.')
        if 'coefficients' in parameters:
            parameters['coefficients'] = tuple(parameters['coefficients'])
        return BUILTIN_CORRELATIONS[name](**parameters)
    local = catalogue.manifest(provider).correlations.get(model)
    if local is None:
        raise ValueError(f'Correlation {model} must be included in plugin {provider}.')
    from .plugins.storage import load_factory
    from .plugins.check import verify_implementation
    values = resolve_parameters(local.parameters, parameters)
    factory = load_factory(provider, local.implementation, catalogue, approved)
    return verify_implementation(factory, local.model_dump(), values, kinetics=False)


def species_properties(spec, provider, catalogue, approved=()):
    return SpeciesProperties(spec['name'], spec['phase'], spec['mw'],
        _correlation(spec['enthalpy'], provider, catalogue, approved),
        _correlation(spec['viscosity'], provider, catalogue, approved) if spec['viscosity'] else None)


def mechanism_family(definition, provider, catalogue, parameters, approved=()):
    from .kinetics import FAMILY_REGISTRY
    from .plugins.catalogue import builtin_manifest
    from .plugins.storage import load_factory
    from .plugins.check import verify_implementation
    if definition.implementation.startswith('builtin:'):
        name = definition.implementation.split(':', 1)[1]
        base = builtin_manifest().mechanisms.get(name)
        if base is None or any(getattr(definition, key) != getattr(base, key) for key in ('gases', 'solids', 'reactions', 'parameters')):
            raise ValueError('Data variants must preserve their built-in implementation contract.')
        return FAMILY_REGISTRY[name]
    factory = load_factory(provider, definition.implementation, catalogue, approved)
    return verify_implementation(factory, definition.model_dump(), parameters)


def materialize_definitions(selection, catalogue=None, *, approved=()):
    catalogue = catalogue or Catalogue()
    expected = catalogue.lock(selection.lock['plugins'])
    if plain(selection.lock) != expected:
        raise ValueError('Definitions differ from the prepared selection.')
    records = {}
    for component, item in selection.species.items():
        provider, _ = split_ref(item['reference'])
        spec = item['definition']
        if plain(spec) != catalogue.get('species', item['reference']).model_dump(mode='json'):
            raise ValueError(f'Selected species metadata differs from {item["reference"]}.')
        records[component] = species_properties(spec, provider, catalogue, approved)
    properties = PropertyRegistry(freeze(records))
    families = []
    hooks = {}
    for instance, item in selection.mechanisms.items():
        provider, _ = split_ref(item['reference'])
        definition = item['definition']
        selected_definition = catalogue.get('mechanisms', item['reference'])
        if plain(definition) != selected_definition.model_dump(mode='json') or plain(item['parameters']) != plain(
                resolve_parameters(selected_definition.parameters, selected_definition.values)):
            raise ValueError(f'Selected mechanism metadata/parameters differ from {item["reference"]}.')
        original = mechanism_family(selected_definition, provider, catalogue, item['parameters'], approved)
        reactions = bound_reactions(instance, selected_definition, item['bindings'], item['reference'])
        bound_hooks = {}
        for local, reaction in zip(original.reactions, reactions):
            original_hook = original.kinetics_hooks[local.id]
            def hook(context, function=original_hook, values=item):
                gas_indices = {role: context.gas_species_index[target] for role,target in values['bindings'].items()
                               if target in context.gas_species_index}
                solid_indices = {role: context.solid_species_index[target] for role,target in values['bindings'].items()
                                 if target in context.solid_species_index}
                bound = KineticsContext(context.model, context.idx_cell, freeze(gas_indices), freeze(solid_indices),
                                        values['parameters'], properties, values['bindings'])
                return function(bound)
            hook._reaction_id = reaction.id
            hook._properties = properties
            hook._parameters = item['parameters']
            hook._bindings = item['bindings']
            bound_hooks[reaction.id] = hook
        hooks.update(bound_hooks)
        gas_roles = tuple(dict.fromkeys(item['bindings'].get(role, role) for role in definition['gases']))
        solid_roles = tuple(dict.fromkeys(item['bindings'].get(role, role) for role in definition['solids']))
        families.append(ReactionFamily(instance, reactions, gas_roles, solid_roles, freeze(bound_hooks)))
    network = build_reaction_network(selection.reaction_ids, selection.gas_species, selection.solid_species,
                                     families=tuple(families))
    return DefinitionEnvironment(properties, network, tuple(hooks[key] for key in network.reaction_ids), selection)
