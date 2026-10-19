"""Collect exactly four verified application downloads for a version-tag release."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tomllib


def prepare(artifacts, output, tag):
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match application version v{version}.")
    artifacts, output = Path(artifacts), Path(output)
    names = {
        "windows": ("setup.exe", "portable.zip"),
        "linux": ("setup.deb", "portable.tar.gz"),
    }
    verified = []
    for platform, suffixes in names.items():
        folder = artifacts / f"MultiSolid-{platform}"
        expected = {f"MultiSolid-{version}-{platform}-x64-{suffix}" for suffix in suffixes}
        if {p.name for p in folder.iterdir()} != expected | {"SHA256SUMS.txt"}:
            raise ValueError(f"Expected exactly the {platform} installer, portable archive, and checksums.")
        checksums = dict(line.split("  ", 1)[::-1] for line in (folder / "SHA256SUMS.txt").read_text().splitlines())
        if set(checksums) != expected:
            raise ValueError(f"Incomplete {platform} checksums.")
        for name in sorted(expected):
            path = folder / name
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != checksums[name]:
                raise ValueError(f"Download checksum mismatch: {name}")
            verified.append((path, digest))
    output.mkdir(parents=True, exist_ok=False)
    for path, _ in verified:
        shutil.copy2(path, output / path.name)
    (output / "SHA256SUMS.txt").write_text("".join(f"{digest}  {path.name}\n" for path, digest in verified))
    (output.parent / "release-notes.md").write_text(
        f"MultiSolid {version}\n\nChoose one application download:\n\n"
        "| Platform | Installer | Portable |\n| --- | --- | --- |\n"
        f"| Windows 11 x64 | `MultiSolid-{version}-windows-x64-setup.exe` | `MultiSolid-{version}-windows-x64-portable.zip` |\n"
        f"| Ubuntu 24.04 x86_64 | `MultiSolid-{version}-linux-x64-setup.deb` | `MultiSolid-{version}-linux-x64-portable.tar.gz` |\n\n"
        "Install the Linux package with `sudo apt install ./MultiSolid-" + version + "-linux-x64-setup.deb`. "
        "For portable downloads, extract the entire archive and run MultiSolid inside it. "
        "All four include Python and the solver dependencies. Keep projects outside the application folder.\n\n"
        "SHA256SUMS.txt verifies the four downloads. GitHub's source archives are for source development.\n",
        encoding="utf-8")
    return [output / path.name for path, _ in verified]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    for path in prepare(args.artifacts, args.output, args.tag):
        print(path)
