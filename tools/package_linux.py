"""Package a validated Linux desktop, preserving executable modes and symlinks."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from packaging.version import Version

from check_desktop_release import verify_inventory
from release_metadata import sha256


def debian_version(value):
    """Keep prereleases ordered before the corresponding Debian final version."""
    version = Version(value)
    text = ".".join(map(str, version.release))
    if version.epoch:
        text = f"{version.epoch}:" + text
    if version.pre:
        text += "~" + "".join(map(str, version.pre))
    if version.post is not None:
        text += f"+post{version.post}"
    if version.dev is not None:
        text += f"~~dev{version.dev}"
    if version.local:
        text += "+" + version.local
    return text


def build_deb(bundle, output, release):
    root = Path(__file__).resolve().parents[1]
    installer = output / f"MultiSolid-{release['version']}-linux-x64-setup.deb"
    temporary = installer.with_suffix(".deb.part")
    with tempfile.TemporaryDirectory(prefix="deb-", dir=output.parent) as scratch:
        stage = Path(scratch) / "package"
        application = stage / "opt/multisolid"
        shutil.copytree(bundle, application, symlinks=True)
        launcher = stage / "usr/bin/multisolid"
        launcher.parent.mkdir(parents=True)
        launcher.write_text('#!/bin/sh\nexec /opt/multisolid/MultiSolid "$@"\n', encoding="utf-8")
        launcher.chmod(0o755)
        for source, relative in (
            (root / "desktop/installer/multisolid.desktop", "usr/share/applications/multisolid.desktop"),
            (root / "desktop/packed_bed_ui/assets/multisolid.svg", "usr/share/icons/hicolor/scalable/apps/multisolid.svg"),
            (root / "LICENSE", "usr/share/doc/multisolid/copyright"),
        ):
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        # System installation files must be readable and not writable by other users.
        for path in stage.rglob("*"):
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644)
        control = stage / "DEBIAN/control"
        control.parent.mkdir()
        size = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file() and not p.is_symlink())
        glibc = release.get("minimum_glibc")
        if not glibc:
            raise ValueError("The Linux bundle must record its minimum glibc version.")
        control.write_text(
            f"Package: multisolid\nVersion: {debian_version(release['version'])}\n"
            "Architecture: amd64\nMaintainer: MultiSolid contributors\n"
            "Section: science\nPriority: optional\n"
            f"Installed-Size: {(size + 1023) // 1024}\nDepends: libc6 (>= {glibc})\n"
            "Homepage: https://github.com/Griminir5/multisolid-CL\n"
            "Description: MultiSolid packed-bed reactor simulation desktop\n"
            " Includes Python, DAETools, Graphviz, and the compiled solver toolchain.\n"
            " Built and tested for Ubuntu 24.04 x86_64.\n", encoding="utf-8")
        subprocess.run(["dpkg-deb", "--build", "--root-owner-group", "-Zxz", "-z3", "--threads-max=2",
                        str(stage), str(temporary)], check=True)
    temporary.replace(installer)
    return installer


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
    installer = build_deb(bundle, output, release)
    artifacts = [installer, archive]
    (output / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in artifacts))
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dist/releases"))
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    for path in package(args.bundle, args.output, args.validation):
        print(path)
