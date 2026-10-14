"""Download hash-pinned build inputs and safely extract them into a cache."""
import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen
import zipfile


def fetch(name, spec, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / name
    if not archive.is_file():
        print(f"Downloading {name}", flush=True)
        temporary = archive.with_name(archive.name + ".part")
        with urlopen(spec["url"], timeout=120) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != spec["sha256"]:
        raise ValueError(f"Checksum mismatch: {archive}. Remove it and download the pinned input again.")
    destination = directory / (name + ".extracted")
    marker = destination / ".archive-sha256"
    if marker.is_file() and marker.read_text() == digest:
        return destination
    if destination.exists():
        raise FileExistsError(f"Incomplete input extraction: {destination}; remove it before retrying.")
    with tempfile.TemporaryDirectory(prefix="extract-", dir=directory) as temporary:
        target = Path(temporary) / "input"
        target.mkdir()
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    if not (target / member.filename).resolve().is_relative_to(target):
                        raise ValueError("Unsafe archive member: " + member.filename)
                source.extractall(target)
        else:
            with tarfile.open(archive) as source:
                source.extractall(target, filter="data")
        (target / ".archive-sha256").write_text(digest)
        target.rename(destination)
    return destination


def find_root(directory, filename):
    matches = list(Path(directory).rglob(filename))
    if len(matches) != 1:
        raise ValueError(f"Expected one {filename} under {directory}; found {len(matches)}")
    return matches[0].parent


def archive_root(directory):
    folders = [path for path in Path(directory).iterdir() if path.is_dir()]
    if len(folders) != 1:
        raise ValueError(f"Expected one top-level source directory in {directory}")
    return folders[0]
