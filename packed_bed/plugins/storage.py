"""Portable current plugin files, run snapshots and local code approval."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace

import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import unicodedata
import zipfile


from .catalogue import Catalogue
from .schema import Manifest

MAX_PACKAGE_BYTES = 100 * 1024 * 1024
MAX_PACKAGE_FILES = 5000


def _relative(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or path.is_absolute() or any(part in ('', '.', '..') for part in name.split('/')):
        raise ValueError(f'Unsafe plugin path: {name}')
    for part in path.parts:
        if re.search(r'[<>:"|?*\x00-\x1f]', part) or part.endswith((' ', '.')) or re.match(r'(?i)^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.|$)', part):
            raise ValueError(f'Plugin path is not portable: {name}')
    return path


def package_files(folder):
    folder = Path(folder)
    if folder.is_symlink():
        raise ValueError('Plugin folders cannot be symbolic links.')
    files, seen = [], set()
    spellings = {}
    total = 0
    for path in sorted(folder.rglob('*')):
        relative = path.relative_to(folder)
        if '__pycache__' in relative.parts or '.git' in relative.parts or path.suffix in ('.pyc', '.pyo'):
            continue
        _relative(relative.as_posix())
        for count in range(1, len(relative.parts) + 1):
            prefix = '/'.join(relative.parts[:count])
            key = unicodedata.normalize('NFC', prefix).casefold()
            if key in spellings and spellings[key] != prefix:
                raise ValueError('Plugin filenames collide on a supported platform.')
            spellings[key] = prefix
        if path.is_symlink():
            raise ValueError('Plugins cannot contain symbolic links.')
        if not path.is_file():
            if not path.is_dir():
                raise ValueError(f'Unsupported plugin file: {relative}')
            continue
        if path.suffix.lower() in ('.so', '.dll', '.pyd', '.exe', '.whl'):
            raise ValueError('Plugins cannot bundle additional native libraries or runtimes.')
        key = relative.as_posix().casefold()
        if key in seen:
            raise ValueError('Plugin filenames collide on a supported platform.')
        seen.add(key)
        total += path.stat().st_size
        files.append((relative.as_posix(), path))
        if total > MAX_PACKAGE_BYTES or len(files) > MAX_PACKAGE_FILES:
            raise ValueError('Plugin exceeds package size/file limits.')
    return files


def content_hash(files):
    digest = hashlib.sha256()
    for name, data in files:
        digest.update(name.encode() + b'\0' + str(len(data)).encode() + b'\0' + data)
    return digest.hexdigest()


def package_hash(folder):
    return content_hash((name, path.read_bytes()) for name, path in package_files(folder))


def code_hash(folder):
    """Approval follows Python/resources; changing form values needs no new consent."""
    return content_hash([('approval-scope', b'plugin-code-and-resources'),
                         *((name, path.read_bytes()) for name, path in package_files(folder) if name != 'manifest.yaml')])


def code_approved(folder):
    approved = approved_hashes()
    return code_hash(folder) in approved


def read_manifest(folder):
    from packed_bed.config.load import read_yaml_mapping
    return Manifest.model_validate(read_yaml_mapping(Path(folder) / 'manifest.yaml', 'plugin manifest'))


def has_code(manifest):
    return bool(manifest.correlations or any(not m.implementation.startswith('builtin:') for m in manifest.mechanisms.values()))


def _validate_files(folder, manifest):
    files = dict(package_files(folder))
    validate_imports(folder)
    entries = [m.implementation for m in manifest.mechanisms.values() if not m.implementation.startswith('builtin:')]
    entries += [c.implementation for c in manifest.correlations.values()]
    for entry in entries:
        module, _, factory = entry.partition(':')
        if not re.fullmatch(r'[A-Za-z_]\w*(\.[A-Za-z_]\w*)*', module) or not re.fullmatch(r'[A-Za-z_]\w*', factory):
            raise ValueError(f'Invalid local implementation reference {entry}.')
        if module.replace('.', '/') + '.py' not in files:
            raise ValueError(f'Plugin must contain implementation {entry}.')
    for species in manifest.species.values():
        for correlation in (species.enthalpy, species.viscosity):
            expected = 'viscosity' if correlation is species.viscosity else 'enthalpy'
            if correlation and correlation.model.startswith('builtin:'):
                allowed = ('builtin:quadratic_viscosity',) if expected == 'viscosity' else ('builtin:polynomial', 'builtin:shomate')
                if correlation.model not in allowed:
                    raise ValueError(f'Unknown or incompatible built-in {expected} correlation {correlation.model}.')
            if correlation and not correlation.model.startswith('builtin:') and correlation.model not in manifest.correlations:
                raise ValueError(f'Correlation {correlation.model} must be included in this plugin.')
            if correlation and not correlation.model.startswith('builtin:'):
                if manifest.correlations[correlation.model].kind != expected:
                    raise ValueError(f'Correlation {correlation.model} has the wrong property kind.')
    from .catalogue import builtin_manifest
    keys = {(s.chemical_key, s.phase) for s in (*manifest.species.values(), *builtin_manifest().species.values())}
    for mechanism in manifest.mechanisms.values():
        for phase, roles in (('gas', mechanism.gases), ('solid', mechanism.solids)):
            for role in roles:
                if (role, phase) not in keys:
                    raise ValueError(f'Plugin must include required custom {phase} species {role}.')


def validate_imports(folder):
    """Check ordinary imports against the executable's supported library surface.

    This is a portability check, not a sandbox for approved Python code.
    Package-local imports must be relative.
    """
    import ast
    supported = set(sys.stdlib_module_names) | {'packed_bed', 'numpy', 'scipy', 'yaml', 'pydantic',
                                               'xarray', 'matplotlib', 'daetools', 'pyUnits'}
    for name, path in package_files(folder):
        if path.suffix != '.py':
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=name)
        except (SyntaxError, UnicodeError) as exc:
            raise ValueError(f'{name}: invalid Python source: {exc}') from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [item.name.split('.')[0] for item in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                roots = [(node.module or '').split('.')[0]]
            else:
                continue
            missing = set(roots) - supported
            if missing:
                raise ValueError(f'{name}: external dependencies are not supported: {", ".join(sorted(missing))}. Use relative imports for included files.')


@contextmanager
def inspect_package(source):
    """Read metadata only, with automatic cleanup of extracted archives."""
    with ExitStack() as stack:
        folder = Path(source)
        if not folder.is_dir():
            folder = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='multisolid-plugin-')))
            try:
                with zipfile.ZipFile(source) as archive:
                    entries = archive.infolist()
                    if len(entries) > MAX_PACKAGE_FILES or sum(item.file_size for item in entries) > MAX_PACKAGE_BYTES:
                        raise ValueError('Plugin archive exceeds size/file limits.')
                    seen = set()
                    for item in entries:
                        name = item.filename.rstrip('/') if item.is_dir() else item.filename
                        _relative(name)
                        if name.casefold() in seen or stat.S_ISLNK(item.external_attr >> 16):
                            raise ValueError('Duplicate paths or symbolic links in plugin archive.')
                        seen.add(name.casefold())
                    archive.extractall(folder)
            except zipfile.BadZipFile as exc:
                raise ValueError('Plugin archive is not a valid ZIP file.') from exc
        manifest = read_manifest(folder)
        _validate_files(folder, manifest)
        yield SimpleNamespace(folder=folder, manifest=manifest, digest=package_hash(folder))


def copy_package(source, destination):
    """Copy into generated inputs; the caller owns their staging/transaction."""
    with inspect_package(source) as package:
        files = {name: path.read_bytes() for name, path in package_files(package.folder)}
        if content_hash(files.items()) != package.digest:
            raise ValueError('Plugin changed while copying.')
        target = Path(destination) / 'plugins' / package.manifest.id / 'current'
        if not target.resolve().is_relative_to(Path(destination).resolve()):
            raise ValueError('Plugin files must stay within the destination.')
        if target.exists():
            shutil.rmtree(target)
        for name, data in files.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return {'id': package.manifest.id, 'hash': package.digest, 'enabled': not has_code(package.manifest)}


def pack_plugin(folder, destination):
    folder, destination = Path(folder).resolve(), Path(destination).resolve()
    if destination.is_relative_to(folder):
        raise ValueError('Write the archive outside its source plugin folder.')
    with inspect_package(folder) as package:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name('.' + destination.name + '.tmp')
        try:
            with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for name, path in package_files(package.folder):
                    info = zipfile.ZipInfo(name)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, path.read_bytes())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def catalogue_from_project(root, entries, lock=None):
    manifests, paths, hashes = {}, {}, {}
    if lock is not None:
        if not isinstance(lock, Mapping) or set(lock) != {'builtin', 'plugins'} or not isinstance(lock['plugins'], Mapping):
            raise ValueError('An exact definition lock must contain a built-in fingerprint and a plugin content map.')
        contents = lock['plugins']
    else:
        if not isinstance(entries, list):
            raise ValueError('Project plugins must be a list.')
        contents, identities = {}, set()
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get('id'), str) or type(entry.get('enabled')) is not bool:
                raise ValueError('Each project plugin must have an identity, content hash and enabled flag.')
            ident = entry['id']
            if ident.casefold() in identities:
                raise ValueError(f'Project has duplicate entries for plugin {ident}.')
            identities.add(ident.casefold())
            if entry['enabled']:
                contents[ident] = entry.get('hash')
    for ident, digest in contents.items():
        if not isinstance(ident, str) or not isinstance(digest, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]*', ident) or not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError('Missing, disabled or invalid plugin reference.')
        path = Path(root) / 'plugins' / ident / 'current'
        if not path.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError('Installed plugin files must stay within the project.')
        with inspect_package(path) as package:
            if package.manifest.id != ident or (lock is not None and package.digest != digest):
                raise ValueError(f'Plugin {ident} no longer matches its recorded content hash.')
            manifests[ident], paths[ident], hashes[ident] = package.manifest, path, package.digest
    return Catalogue(manifests, paths, hashes)


def catalogue_for_case(run_path):
    """Discover a project's current plugins or an immutable run snapshot.

    Standalone inputs may put a definitions.json beside run.yaml with a lock and
    a project-relative root. This same descriptor travels with CLI batch copies.
    """
    run_path = Path(run_path).resolve()
    descriptor = run_path.parent / 'definitions.json'
    if descriptor.is_file():
        data = json.loads(descriptor.read_text(encoding='utf-8'))
        return catalogue_from_project((descriptor.parent / data.get('root', '.')).resolve(), [], data['lock'])
    for parent in run_path.parents:
        if (parent / 'project.json').is_file():
            project = json.loads((parent / 'project.json').read_text(encoding='utf-8'))
            if project.get('extensions'):
                raise ValueError('This project requires extensions in an unsupported legacy format.')
            return catalogue_from_project(parent, project.get('plugins', []))
    return Catalogue()


def require_approval(catalogue, lock):
    """Readiness check only; a project can never convey local code approval."""
    for ident in lock['plugins']:
        if has_code(catalogue.manifest(ident)) and not code_approved(catalogue.paths[ident]):
            raise ValueError(f'Plugin {ident} needs local code approval. Open Plugins and enable it.')


def approval_file():
    override = os.environ.get('MULTISOLID_PLUGIN_APPROVALS')
    if override:
        return Path(override)
    base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local')) if os.name == 'nt' else Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
    return base / 'multisolid' / 'plugin-approvals.json'


def approved_hashes():
    try:
        return frozenset(json.loads(approval_file().read_text()))
    except FileNotFoundError:
        return frozenset()


def approve_hash(digest):
    if not re.fullmatch(r'[a-f0-9]{64}', digest):
        raise ValueError('Invalid plugin content hash.')
    path = approval_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = sorted(approved_hashes() | {digest})
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data) + '\n')
    temporary.replace(path)


def load_factory(provider, entry, catalogue, approved=()):
    digest = catalogue.hashes[provider]
    if digest not in approved and not code_approved(catalogue.paths[provider]):
        raise ValueError(f'Enable code plugin {provider} on this machine before execution.')
    folder = catalogue.paths[provider].resolve()
    if package_hash(folder) != digest:
        raise ValueError(f'Plugin {provider} content has changed.')
    module_name, factory = entry.split(':')
    # A temporary archive inspection and an installed copy may have identical
    # contents but different resource paths. Keep their module namespaces apart.
    namespace = '_multisolid_plugin_' + digest + '_' + hashlib.sha256(str(folder).encode()).hexdigest()[:12]
    import importlib
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        if namespace not in sys.modules:
            initializer = folder / '__init__.py'
            spec = importlib.util.spec_from_file_location(namespace, initializer, submodule_search_locations=[str(folder)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = package
            try:
                if initializer.is_file():
                    spec.loader.exec_module(package)
            except Exception:
                sys.modules.pop(namespace, None)
                raise
        module = importlib.import_module(namespace + '.' + module_name)
        result = getattr(module, factory)
    finally:
        sys.dont_write_bytecode = old
    if not callable(result):
        raise ValueError(f'Implementation {entry} is not callable.')
    return result
