"""Package a validated Linux desktop, preserving executable modes and symlinks."""
import argparse
import json
from pathlib import Path
import tarfile

from check_desktop_release import verify_inventory
from release_metadata import sha256


def package(bundle, output, validation):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    release = json.loads((bundle / "release.json").read_text())
    result = json.loads(Path(validation).read_text())
    if (release.get("platform") != "linux" or result.get("passed") is not True
            or result.get("executable_sha256") != sha256(bundle / "MultiSolid")
            or result.get("inventory_sha256") != sha256(bundle / "bundle-files.json")):
        raise ValueError("Run check_desktop_release.py against this exact Linux bundle before packaging.")
    verify_inventory(bundle)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"MultiSolid-{release['version']}-linux-x64-portable.tar.gz"
    temporary = archive.with_name(archive.name + ".part")
    with tarfile.open(temporary, "w:gz", dereference=False) as target:
        target.add(bundle, arcname="MultiSolid")
    temporary.replace(archive)
    (output / "SHA256SUMS.txt").write_text(f"{sha256(archive)}  {archive.name}\n")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dist/releases"))
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    print(package(args.bundle, args.output, args.validation))
