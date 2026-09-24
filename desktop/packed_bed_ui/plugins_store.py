"""One current copy of each project plugin, saved with existing transactions."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from packed_bed.plugins.catalogue import Catalogue, builtin_fingerprint, split_ref
from packed_bed.plugins.storage import (catalogue_from_project, inspect_package, package_files,
                                        content_hash, has_code, code_approved, pack_plugin)


def references(documents):
    chemistry = documents.get('chemistry', documents)
    species = chemistry.get('species_definitions', {})
    result = [('species', ref) for ref in species.values()]
    if 'chemistry' in documents:
        components = chemistry.get('gas_species', []) + documents.get('solids', {}).get('solid_species', [])
        result += [('species', species.get(key, 'builtin:' + key)) for key in components]
    for instance in chemistry.get('reaction_families', []):
        result.append(('mechanisms', chemistry.get('mechanisms', {}).get(instance, {}).get('definition', 'builtin:' + instance)))
    return result


class PluginStore:
    def __init__(self, project):
        self.project = project

    @property
    def entries(self):
        return self.project.metadata.get('plugins', [])

    def catalogue(self, lock=None, *, include_disabled=False):
        entries = [{**entry, 'enabled': True} for entry in self.entries] if include_disabled else self.entries
        return catalogue_from_project(self.project.root, entries, lock)

    def browse_catalogue(self, *, include_disabled=False):
        """Inspect providers individually so one broken package remains removable.

        Runtime resolution still requires every selected plugin to be valid.
        """
        manifests, paths, hashes, errors = {}, {}, {}, {}
        for entry in self.entries:
            if not include_disabled and not entry['enabled']:
                continue
            try:
                catalogue = catalogue_from_project(self.project.root, [{**entry, 'enabled': True}])
                manifests.update(catalogue.manifests)
                paths.update(catalogue.paths)
                hashes.update(catalogue.hashes)
            except (ValueError, OSError) as exc:
                errors[entry['id']] = str(exc)
        return Catalogue(manifests, paths, hashes), errors

    def lock_for(self, documents):
        providers = {split_ref(ref)[0] for _, ref in references(documents)} - {'builtin'}
        catalogue, _ = self.browse_catalogue()
        return {'builtin': builtin_fingerprint(),
                'plugins': {key: catalogue.hashes.get(key, 'missing-or-disabled') for key in sorted(providers)}}

    def users(self, provider, *, reference=None, kind=None, ordinary_only=False):
        def uses(documents):
            return any(split_ref(ref)[0] == provider and (reference is None or split_ref(ref) == split_ref(reference))
                       and (kind is None or item_kind == kind) for item_kind, ref in references(documents))
        result = []
        for case in self.project.cases:
            if not (ordinary_only and case.metadata.get('study_id')) and uses(case.documents):
                result.append('Case: ' + case.name)
        for definition in self.project.study_store.definitions.values():
            if uses(definition.payload):
                result.append('Reusable definition: ' + definition.name)
        if not ordinary_only:
            for study in self.project.study_store.studies.values():
                if uses(study.baseline):
                    result.append('Study baseline: ' + study.name)
        return result

    def add(self, source):
        self.project.study_store._require_idle()
        with inspect_package(source) as package:
            previous = next((entry for entry in self.entries if entry['id'].casefold() == package.manifest.id.casefold()), None)
            if previous and previous['id'] != package.manifest.id:
                raise ValueError('Plugin identities must also be unique on Windows.')
            if not has_code(package.manifest):
                from packed_bed.plugins.check import check_package
                check_package(package.folder)
            entry = {'id': package.manifest.id, 'hash': package.digest,
                     'enabled': previous['enabled'] if previous else not has_code(package.manifest)}
            if entry['enabled'] and has_code(package.manifest) and not code_approved(package.folder):
                raise ValueError('Allow this plugin’s code before saving it to an enabled plugin.')
            files = {name: path.read_bytes() for name, path in package_files(package.folder)}
            if content_hash(files.items()) != package.digest:
                raise ValueError('Plugin changed while saving. Try again.')
        metadata = deepcopy(self.project.metadata)
        entries = metadata.setdefault('plugins', [])
        entries[:] = [item for item in entries if item['id'] != entry['id']]
        entries.append(entry)
        self.project.study_store._commit({f'plugins/{entry["id"]}/current': files}, metadata)
        return entry

    def create(self, name):
        from .project import write_text
        import yaml
        with TemporaryDirectory(prefix='multisolid-plugin-') as folder:
            write_text(Path(folder) / 'manifest.yaml', yaml.safe_dump({'id': 'plugin_' + uuid4().hex[:8], 'name': name}))
            return self.add(folder)

    def set_enabled(self, ident, enabled):
        metadata = deepcopy(self.project.metadata)
        entry = next(item for item in metadata.get('plugins', []) if item['id'] == ident)
        if not enabled:
            self._unused(ident)
        else:
            catalogue = catalogue_from_project(self.project.root, [{**entry, 'enabled': True}])
            if has_code(catalogue.manifest(ident)) and not code_approved(catalogue.paths[ident]):
                raise ValueError('Allow this plugin’s code before enabling it.')
        entry['enabled'] = bool(enabled)
        self.project.study_store._commit({}, metadata)

    def _unused(self, ident):
        users = self.users(ident)
        if users:
            raise ValueError('Plugin is still used by:\n' + '\n'.join(users))

    def remove(self, ident):
        self._unused(ident)
        metadata = deepcopy(self.project.metadata)
        metadata['plugins'] = [entry for entry in self.entries if entry['id'] != ident]
        self.project.study_store._commit({f'plugins/{ident}': None}, metadata)

    def export(self, ident, destination):
        entry = next(item for item in self.entries if item['id'] == ident)
        catalogue = catalogue_from_project(self.project.root, [{**entry, 'enabled': True}])
        return pack_plugin(catalogue.paths[ident], destination)

    def replacement_uses(self, kind, source, target):
        catalogue, _ = self.browse_catalogue()
        original, replacement = catalogue.get(kind, source), catalogue.get(kind, target)
        if kind == 'species':
            compatible = (original.chemical_key, original.phase) == (replacement.chemical_key, replacement.phase)
        elif kind == 'mechanisms':
            compatible = (original.gases, original.solids, tuple(r.id for r in original.reactions)) == (
                replacement.gases, replacement.solids, tuple(r.id for r in replacement.reactions))
        else:
            raise ValueError('Replace a species or reaction family.')
        if not compatible:
            raise ValueError('Replacement must preserve chemical key/phase or reaction IDs/species roles.')
        return self.users(split_ref(source)[0], reference=source, kind=kind, ordinary_only=True)

    def replace_uses(self, kind, source, target):
        self.replacement_uses(kind, source, target)
        def rewrite(documents):
            chemistry = documents.get('chemistry', documents)
            if kind == 'species':
                definitions = chemistry.get('species_definitions', {}).copy()
                keys = set(definitions)
                if 'chemistry' in documents:
                    keys.update(chemistry.get('gas_species', []) + documents.get('solids', {}).get('solid_species', []))
                for component in keys:
                    if split_ref(definitions.get(component, 'builtin:' + component)) == split_ref(source):
                        definitions[component] = target
                        chemistry['species_definitions'] = definitions
            else:
                definitions = deepcopy(chemistry.get('mechanisms', {}))
                for instance in chemistry.get('reaction_families', []):
                    item = definitions.get(instance, {})
                    if split_ref(item.get('definition', 'builtin:' + instance)) == split_ref(source):
                        definitions[instance] = {**item, 'definition': target}
                        chemistry['mechanisms'] = definitions
        from .study_store import _documents
        folders, changed = {}, {}
        for case in self.project.cases:
            if case.metadata.get('study_id'):
                continue
            documents = deepcopy(case.documents)
            rewrite(documents)
            if documents != case.documents:
                folders[f'cases/{case.id}/inputs'] = _documents(documents)
                changed[case.id] = documents
        for definition in self.project.study_store.definitions.values():
            replacement = deepcopy(definition)
            rewrite(replacement.payload)
            if replacement != definition:
                folders[f'definitions/{definition.id}'] = self.project.study_store._definition_files(replacement)
        self.project.study_store._commit(folders, self.project.metadata)
        for case in self.project.cases:
            if case.id in changed:
                case.documents = changed[case.id]
        return len(changed)
