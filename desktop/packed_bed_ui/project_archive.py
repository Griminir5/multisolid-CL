"""Inputs-only project ZIPs. Inspection never loads plugin implementations."""

from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import stat
from tempfile import TemporaryDirectory
import unicodedata
from uuid import uuid4
from zipfile import ZipFile, BadZipFile, ZIP_DEFLATED, ZIP_STORED

import yaml

from packed_bed.config.load import read_yaml_mapping
from packed_bed.plugins.catalogue import builtin_fingerprint, split_ref
from packed_bed.plugins.storage import _relative, inspect_package, package_files

from .plugins_store import references
from .project import DOCUMENTS, PROJECT_VERSION, REFERENCES, portable_documents, read_json
from .studies import ReusableDefinition, Study, generation_signature


MAX_FILES = 100_000
MAX_BYTES = 1024 ** 3
CHUNK = 1024 ** 2
ARCHIVE = {"kind": "multisolid-project", "format_version": 1}


def _check(cancelled):
    if cancelled():
        raise InterruptedError("Project transfer cancelled.")


def _json(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _file(path):
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"Required input must be a regular file: {path}")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"Input exceeds archive size limit: {path}")
    return path


def _names(names):
    """Check portable spelling and collisions, including directory prefixes."""
    seen, spellings, directories = set(), {}, set()
    for name in names:
        parts = _relative(name).parts
        if name in seen:
            raise ValueError(f"Duplicate archive path: {name}")
        seen.add(name)
        for count in range(1, len(parts) + 1):
            prefix = "/".join(parts[:count])
            key = unicodedata.normalize("NFC", prefix).casefold()
            if key in spellings and spellings[key] != prefix:
                raise ValueError(f"Archive paths collide: {prefix}")
            spellings[key] = prefix
            if count < len(parts):
                directories.add(prefix)
    if seen & directories:
        raise ValueError("Archive file and directory paths collide.")


def _batch_refs(spec):
    yield spec, "base_case"
    for key in spec.get("programs", {}):
        yield spec["programs"], key
    for item in spec.get("geometries", {}).values():
        if item.get("solids_file"):
            yield item, "solids_file"


def _payload(root, *, exporting=False, cancelled=lambda: False):
    """Enumerate only owned inputs; paths point to files and bytes hold rewritten data."""
    project_file = _file(root / "project.json")
    if project_file.stat().st_size > 16 * CHUNK:
        raise ValueError("Project metadata is too large.")
    metadata = deepcopy(read_json(project_file))
    if metadata.get("format_version") != PROJECT_VERSION:
        raise ValueError("Unsupported project format.")
    if not isinstance(metadata.get("name"), str) or metadata.get("extensions"):
        raise ValueError("Invalid project name or unsupported legacy extensions.")
    if not isinstance(metadata.get("cases"), list):
        raise ValueError("Project cases must be a list.")
    workers = metadata.get("max_workers", 1)
    if type(workers) is not int or not 1 <= workers <= 1024:
        raise ValueError("Invalid project worker limit.")
    metadata.pop("study_transaction", None)
    files, documents = {}, []

    def add(name, value):
        if name in files and files[name] != value:
            raise ValueError(f"Conflicting input paths: {name}")
        files[name] = value

    def mapping(path):
        return read_yaml_mapping(_file(path), str(path))

    def inputs(folder, target, run_name="run.yaml"):
        run = mapping(folder / run_name)
        if not exporting and run.get("references") != REFERENCES:
            raise ValueError(f"Nonportable input references: {target}/run.yaml")
        data = {"run": run}
        for key in DOCUMENTS[1:]:
            ref = run.get("references", {}).get(f"{key}_file", f"{key}.yaml")
            data[key] = mapping(folder / ref)
        portable = portable_documents(data)
        if not exporting and portable != data:
            raise ValueError(f"Nonportable output destinations: {target}/run.yaml")
        for key, value in portable.items():
            add(f"{target}/{key}.yaml", yaml.safe_dump(value, sort_keys=False).encode())
        documents.append(data)
        return portable

    entries = {}
    for kind in ("cases", "studies", "definitions", "plugins"):
        values = metadata.get(kind, [])
        if not isinstance(values, list):
            raise ValueError(f"Project {kind} must be a list.")
        entries[kind] = {}
        seen = set()
        for entry in values:
            _check(cancelled)
            ident = entry.get("id") if isinstance(entry, dict) else None
            pattern = r"[A-Za-z][A-Za-z0-9_.-]*" if kind == "plugins" else r"[A-Za-z0-9_-]+"
            if not isinstance(ident, str) or not re.fullmatch(pattern, ident):
                raise ValueError(f"Invalid {kind} identity: {ident}")
            _relative(ident)
            if ident.casefold() in seen:
                raise ValueError(f"Duplicate {kind} identity: {ident}")
            seen.add(ident.casefold())
            if kind != "plugins" and not isinstance(entry.get("name"), str):
                raise ValueError(f"Missing {kind} name: {ident}")
            entries[kind][ident] = entry

    for ident, entry in entries["cases"].items():
        _check(cancelled)
        if entry.get("study_id") and entry["study_id"] not in entries["studies"]:
            raise ValueError(f"Case {ident} refers to a missing study.")
        inputs(root / f"cases/{ident}/inputs", f"cases/{ident}/inputs")

    definitions = {}
    for ident, entry in entries["definitions"].items():
        _check(cancelled)
        folder = root / f"definitions/{ident}"
        data = read_json(_file(folder / "definition.json"))
        if (data.get("id") != ident or not isinstance(data.get("name"), str)
                or data.get("kind") not in ("program", "bed")
                or data["kind"] != entry.get("kind")):
            raise ValueError(f"Invalid reusable definition: {ident}")
        document = "program" if data["kind"] == "program" else "solids"
        payload = deepcopy(data)
        payload["payload"][document] = mapping(folder / f"{document}.yaml")
        definition = definitions[ident] = ReusableDefinition(**payload)
        documents.append(definition.payload)
        add(f"definitions/{ident}/definition.json", _json(data))
        add(f"definitions/{ident}/{document}.yaml", folder / f"{document}.yaml")

    for ident in entries["studies"]:
        _check(cancelled)
        target = f"studies/{ident}"
        folder = root / target
        rule = read_json(_file(folder / "study.json"))
        study = Study.from_rule(rule, inputs(folder / "baseline", f"{target}/baseline"))
        if (study.id != ident or not isinstance(study.name, str)
                or not isinstance(study.provenance, dict) or not isinstance(study.editor_metadata, dict)):
            raise ValueError(f"Invalid study metadata: {ident}")
        if study.legacy or (folder / "batch.yaml").exists():
            batch = mapping(folder / "batch.yaml")
            relocated = {}
            def source_ref(ref, fallback):
                try:
                    portable = str(_relative(ref))
                except ValueError:
                    if not exporting:
                        raise
                    portable = fallback
                relocated[ref] = portable
                return portable
            base_ref = batch["base_case"]
            base = Path(source_ref(base_ref, "sources/base/run.yaml"))
            if base.name != "run.yaml" or len(base.parts) < 2:
                if not exporting:
                    raise ValueError(f"Invalid study base input path: {base}")
                base = Path("sources/base/run.yaml")
                relocated[base_ref] = base.as_posix()
            source = folder / base_ref
            inputs(source.parent, f"{target}/{base.parent.as_posix()}", source.name)
            batch["base_case"] = base.as_posix()
            for index, (container, key) in enumerate(list(_batch_refs(batch))[1:]):
                ref = container[key]
                container[key] = source_ref(ref, f"sources/input-{index}.yaml")
                mapping(folder / ref)
                add(f"{target}/{container[key]}", folder / ref)
            if not exporting and batch.get("output_directory") != "unused-output":
                raise ValueError("Nonportable study output directory.")
            batch["output_directory"] = "unused-output"
            add(f"{target}/batch.yaml", yaml.safe_dump(batch, sort_keys=False).encode())
            if study.legacy and any(a != b for a, b in relocated.items()):
                # File relocation alone must not turn an established study into Needs rebuild.
                spec = rule["legacy"]["spec"]
                for container, key in _batch_refs(spec):
                    container[key] = relocated.get(container[key], container[key])
                lock = study.provenance.get("extensions", [])
                entry = entries["studies"][ident]
                if entry.get("generation_signature") == generation_signature(study, definitions, lock):
                    entry["generation_signature"] = generation_signature(Study.from_rule(rule, study.baseline), definitions, lock)
        add(f"{target}/study.json", _json(rule))

    for ident, entry in entries["plugins"].items():
        _check(cancelled)
        if type(entry.get("enabled")) is not bool:
            raise ValueError(f"Invalid plugin enablement: {ident}")
        folder = root / f"plugins/{ident}/current"
        _file(folder / "manifest.yaml")
        with inspect_package(folder) as package:
            if package.manifest.id != ident:
                raise ValueError(f"Plugin identity does not match its folder: {ident}")
            if not exporting and entry.get("hash") != package.digest:
                raise ValueError(f"Plugin content hash does not match: {ident}")
            entry["hash"] = package.digest
            for name, path in package_files(folder):
                add(f"plugins/{ident}/current/{name}", path)
    for data in documents:
        for _, ref in references(data):
            provider, _ = split_ref(ref)
            if provider != "builtin" and provider not in entries["plugins"]:
                raise ValueError(f"Required plugin is missing: {provider}")
    files["project.json"] = _json(metadata)
    _names(files)
    if len(files) + 1 > MAX_FILES or sum(len(v) if isinstance(v, bytes) else v.stat().st_size
                                       for v in files.values()) > MAX_BYTES:
        raise ValueError("Project exceeds archive size/file limits.")
    return metadata, files


def _copy(source, target, cancelled):
    total = 0
    while True:
        _check(cancelled)
        block = source.read(CHUNK)
        if not block:
            return
        total += len(block)
        if total > MAX_BYTES:
            raise ValueError("Archive member exceeds size limit.")
        target.write(block)


def _write_files(folder, files, cancelled):
    # The project marker is published last, after all other inputs are complete.
    for name in sorted(files, key=lambda name: name == "project.json"):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        value = files[name]
        with BytesIO(value) if isinstance(value, bytes) else _file(value).open("rb") as source:
            with path.open("xb") as target:
                _copy(source, target, cancelled)


@contextmanager
def _open_archive(source):
    try:
        with ZipFile(source) as archive:
            members = archive.infolist()
            if len(members) > MAX_FILES or sum(m.file_size for m in members) > MAX_BYTES:
                raise ValueError("Archive exceeds size/file limits.")
            _names(m.filename for m in members)
            for member in members:
                mode = stat.S_IFMT(member.external_attr >> 16)
                if (mode not in (0, stat.S_IFREG) or member.is_dir() or member.orig_filename != member.filename
                        or member.flag_bits & 1 or member.compress_type not in (ZIP_STORED, ZIP_DEFLATED)):
                    raise ValueError(f"Unsupported archive member: {member.filename}")
            if archive.getinfo("archive.json").file_size > CHUNK:
                raise ValueError("Archive descriptor is too large.")
            descriptor = json.loads(archive.read("archive.json"))
            if not isinstance(descriptor, dict) or any(descriptor.get(k) != v for k, v in ARCHIVE.items()):
                raise ValueError("Unsupported project archive format.")
            yield archive
    except (BadZipFile, KeyError, TypeError, AttributeError, UnicodeError, RecursionError) as exc:
        raise ValueError(f"Invalid project archive: {exc}") from exc


def archive_name(source):
    """Read the suggested name; import subsequently revalidates the entire ZIP."""
    with _open_archive(source) as archive:
        if archive.getinfo("project.json").file_size > 16 * CHUNK:
            raise ValueError("Project metadata is too large.")
        data = json.loads(archive.read("project.json"))
        name = data.get("name")
        if not isinstance(name, str):
            raise ValueError("Missing project name.")
        return name


def export_project(project, destination, *, cancelled=lambda: False):
    """Export saved, idle project inputs without changing the source project."""
    root, destination = project.root.resolve(), Path(destination).absolute()
    if getattr(project, "executing", False) or (root / ".solver.lock").exists():
        raise ValueError("Wait for execution to finish before exporting.")
    if destination.is_relative_to(root) or destination.resolve().is_relative_to(root):
        raise ValueError("Save the archive outside the project folder.")
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with TemporaryDirectory(prefix="multisolid-export-") as staging:
            staging = Path(staging)
            _, files = _payload(root, exporting=True, cancelled=cancelled)
            _write_files(staging, files, cancelled)
            _, checked = _payload(staging, cancelled=cancelled)
            try:
                application_version = version("multisolid-cl-ui")
            except PackageNotFoundError:
                application_version = "unknown"  # Descriptive only; folder formats determine compatibility.
            descriptor = {**ARCHIVE, "application_version": application_version,
                          "builtin": builtin_fingerprint()}
            (staging / "archive.json").write_bytes(_json(descriptor))
            with ZipFile(temporary, "x", compression=ZIP_DEFLATED) as archive:
                for name in ["archive.json", *checked]:
                    with (staging / name).open("rb") as source, archive.open(name, "w") as target:
                        _copy(source, target, cancelled)
            with _open_archive(temporary):
                pass  # Include the descriptor in the final size/file checks.
            _check(cancelled)
            temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def import_project(source, destination, name, *, cancelled=lambda: False):
    """Validate in isolation, then publish into an exclusively created folder."""
    destination = Path(destination).absolute()
    if not name.strip():
        raise ValueError("Enter a project name.")
    with TemporaryDirectory(prefix="multisolid-import-") as staging, _open_archive(source) as archive:
        staging = Path(staging)
        for member in archive.infolist():
            path = staging / member.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, path.open("xb") as target:
                _copy(src, target, cancelled)
        metadata, files = _payload(staging, cancelled=cancelled)
        if set(archive.namelist()) != {*files, "archive.json"}:
            raise ValueError("Archive contains undeclared files or execution state.")
        metadata["name"] = name.strip()
        files["project.json"] = _json(metadata)
        _check(cancelled)
        destination.mkdir(parents=True, exist_ok=False)
        try:
            _write_files(destination, files, cancelled)
            _check(cancelled)
        except BaseException:
            shutil.rmtree(destination)
            raise
    return destination
