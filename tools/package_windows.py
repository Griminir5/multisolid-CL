"""Create an installer and a single-folder portable ZIP from a verified bundle."""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile

from release_metadata import sha256


def package(bundle, output, iscc, validation):
    bundle, output, iscc = (Path(p).resolve() for p in (bundle, output, iscc))
    release = json.loads((bundle / "release.json").read_text())
    inventory = json.loads((bundle / "bundle-files.json").read_text())
    result = json.loads(Path(validation).read_text())
    if (result.get("passed") is not True or result.get("executable_sha256") != sha256(bundle / "MultiSolid.exe")
            or result.get("inventory_sha256") != sha256(bundle / "bundle-files.json")):
        raise ValueError("Run check_windows_release.py against this exact bundle before packaging.")
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != set(inventory) | {"bundle-files.json"}:
        raise ValueError("The bundle has missing or unrecorded files; rebuild and validate it.")
    for name, expected in inventory.items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or sha256(path) != expected:
            raise ValueError(f"Bundle changed since validation: {name}")
    output.mkdir(parents=True, exist_ok=True)
    version = release["version"]
    root = Path(__file__).resolve().parents[1]
    subprocess.run([str(iscc), f"/DBundleDir={bundle}", f"/DAppVersion={version}",
                    f"/DBuildId={release['build_id']}", f"/O{output}",
                    str(root / "desktop/installer/MultiSolid.iss")], check=True)
    archive = output / f"MultiSolid-{version}-windows-x64-portable.zip"
    temporary = archive.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                target.write(path, "MultiSolid/" + path.relative_to(bundle).as_posix())
    temporary.replace(archive)
    artifacts = [output / f"MultiSolid-{version}-windows-x64-setup.exe", archive]
    (output / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in artifacts))
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("dist/desktop/MultiSolid"))
    parser.add_argument("--output", type=Path, default=Path("dist/releases"))
    parser.add_argument("--iscc", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    for path in package(args.bundle, args.output, args.iscc, args.validation):
        print(path)
