"""Shared, metadata-only catalogue labels and scientific details."""
import yaml
from packed_bed.parameters import plain
from PyQt6.QtWidgets import QPlainTextEdit


class DefinitionDetails(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setAccessibleName('Definition details and source')

    def show_definition(self, catalogue, kind, reference, reaction_id=None):
        self.setPlainText(definition_details(catalogue, kind, reference, reaction_id))


def definition_details(catalogue, kind, reference, reaction_id=None):
    from packed_bed.plugins.catalogue import split_ref
    provider, _ = split_ref(reference)
    manifest = catalogue.manifest(provider)
    definition = catalogue.get(kind, reference)
    title = catalogue.label(kind, reference)
    parameters = ''
    if reaction_id is not None:
        reaction = next(item for item in definition.reactions if item.id == reaction_id)
        title = f'{reaction.name} — {manifest.name}\nFamily: {definition.name}'
        parameters = reaction_parameter_details(definition, reaction_id)
        definition = reaction
    return (f'{title}\n{reference}\n\n'
            + parameters + yaml.safe_dump(plain(definition) if reaction_id is not None else definition.model_dump(mode='json'), sort_keys=False, allow_unicode=True))


def reaction_parameter_details(definition, reaction_id):
    from packed_bed.plugins.parameter_details import reaction_parameter_groups
    groups = reaction_parameter_groups(definition)
    keys = groups[reaction_id] if groups is not None else tuple(definition.parameters)
    lines = ['Editable kinetic parameters']
    if not keys:
        return 'Editable kinetic parameters: none declared for this reaction.\n\n'
    if groups is None and len(definition.reactions) > 1:
        lines.append('Family parameters: this implementation does not declare per-reaction groups.')
    for key in keys:
        spec = definition.parameters[key]
        value = definition.values.get(key, spec.default)
        shared = groups is not None and sum(key in group for group in groups.values()) > 1
        lines.append(f'{key}: {value:g} {spec.unit}' + (' [shared]' if shared else ''))
        limits = []
        if spec.minimum is not None:
            limits.append(f'min {spec.minimum:g}')
        if spec.maximum is not None:
            limits.append(f'max {spec.maximum:g}')
        lines.append(f'  Default: {spec.default:g}' + ('; ' + ', '.join(limits) if limits else ''))
        if spec.description:
            lines.append('  ' + spec.description)
    if groups is not None and any(sum(key in group for group in groups.values()) > 1 for key in keys):
        lines.append('Shared parameters also affect other reactions in this family.')
    lines.append('Choose Edit to change these values, or Create variant for a separate copy.')
    return '\n'.join(lines) + '\n\n'


def component_key(reference):
    provider, local = reference.split(':', 1)
    return local if provider == 'builtin' else provider + '.' + local


def species_choices(catalogue, phase, mappings=None):
    # Keep case IDs stable while exposing displaced definitions under unused IDs.
    references = dict(mappings or {})
    entries = catalogue.entries('species')
    reserved = {component_key(ref) for ref in entries}
    for reference, item in entries.items():
        if item.phase != phase or reference in references.values():
            continue
        component = base = component_key(reference)
        suffix = 2
        while component in references or (component != base and component in reserved):
            component = f'{base}_{suffix}'
            suffix += 1
        references[component] = reference
    choices = {}
    for component, reference in references.items():
        try:
            if catalogue.get('species', reference).phase == phase:
                choices[component] = (f'{component} · {catalogue.label("species", reference)}',
                                      definition_details(catalogue, 'species', reference))
        except ValueError:
            choices[component] = (f'{component} — Missing {reference}', 'Definition is missing or disabled.')
    return choices, references
