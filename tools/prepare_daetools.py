"""Copy a DAE Tools release tree for wheel building without creating OS shortcuts."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def prepare(source: Path, destination: Path):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    setup = source / "setup.py"
    original = setup.read_text(encoding="utf-8")
    marker = "\nif platform.system() == 'Windows':\n    try:\n        script_folder"
    if original.count(marker) != 1:
        raise ValueError("Unrecognized DAE Tools setup.py; inspect its shortcut installation first.")
    if not (source / "daetools/licence.txt").is_file():
        raise ValueError("The DAE Tools distribution is missing its license.")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__"))
    (destination / "setup.py").write_text(original.split(marker)[0] + "\n", encoding="utf-8")
    (destination / "MULTISOLID-BUILD-PATCH.txt").write_text(
        "MultiSolid build patch: setup.py's final Windows shortcut-creation block\n"
        "is removed. Building a dependency wheel must not install Start menu links.\n"
        "No solver or native binaries are changed by this patch.\n", encoding="utf-8")
    files = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                files[path.relative_to(destination).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    (destination / "multisolid-input-hashes.json").write_text(json.dumps(files, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(prepare(args.source, args.destination))
