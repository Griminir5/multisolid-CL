"""Stage a separate Graphviz runtime for development or a folder-based release.

On Debian/Ubuntu: python tools/bundle_graphviz.py --system
For a portable Windows/Linux prefix: --source PREFIX --source-url URL
Use --destination APP/graphviz to stage beside a frozen application executable.
The output is generated, platform-specific data, never a Python extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from packed_bed.reaction_graph import GraphvizCommand, build_reaction_graph, render_svg


def _run(*args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, **kwargs).stdout.strip()


def _system_bundle(target):
    if not sys.platform.startswith("linux") or not shutil.which("dpkg-query"):
        raise RuntimeError("--system needs Debian/Ubuntu; use --source for a portable distribution.")
    executable = shutil.which("neato")
    if not executable:
        raise RuntimeError("Install Graphviz in the build environment first.")
    # Locate plugins beside Graphviz's core library, including multiarch prefixes.
    libraries = _dependencies(Path(executable))
    plugin_dirs = {p.parent / "graphviz" for p in libraries if p.name.startswith("libgvc.so")}
    plugin_dir = next((p for p in sorted(plugin_dirs) if list(p.glob("libgvplugin_neato_layout.so.*"))), None)
    if plugin_dir is None:
        raise RuntimeError("The Graphviz neato layout plugin was not found.")
    sources = set()

    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sources.add(source.resolve())

    copy(Path(executable), target / "bin" / "neato")
    for plugin in ("core", "neato_layout", "pango"):
        candidates = sorted(plugin_dir.glob(f"libgvplugin_{plugin}.so.*"))
        if not candidates:
            raise RuntimeError(f"Missing Graphviz {plugin} plugin.")
        for path in candidates:
            copy(path, target / "lib" / "graphviz" / path.name)
            libraries.update(_dependencies(path))
    # The platform C runtime comes from the supported OS baseline. All other
    # dynamically linked dependencies of neato and its plugins travel together.
    baseline = re.compile(r"(?:ld-linux.*|lib(?:c|m|mvec|dl|pthread|rt|resolv|util)\.so\..*)$")
    for path in sorted(libraries):
        if not baseline.fullmatch(path.name):
            copy(path, target / "lib" / path.name)
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    copy(font, target / "share" / "fonts" / font.name)
    config = target / "etc" / "fonts" / "fonts.conf"
    config.parent.mkdir(parents=True)
    config.write_text('''<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">
<fontconfig>
  <dir prefix="relative">../../share/fonts</dir>
  <cachedir prefix="xdg">fontconfig</cachedir>
</fontconfig>
''', encoding="utf-8")
    packages = {}
    for path in sorted(sources):
        # Account for merged-/usr systems in dpkg's historical file list.
        candidates = [path, Path(str(path).removeprefix("/usr"))] if str(path).startswith("/usr/lib/") else [path]
        owner = None
        for candidate in candidates:
            result = subprocess.run(["dpkg-query", "-S", str(candidate)], capture_output=True, text=True)
            if result.returncode == 0:
                owner = result.stdout.split(": /", 1)[0].strip()
                break
        if not owner:
            raise RuntimeError(f"Cannot identify the package and notices for {path}")
        if owner in packages:
            continue
        metadata = _run("dpkg-query", "-W", "-f=${Package}\t${Version}\t${source:Package}\t${source:Version}", owner).split("\t")
        package = metadata[0]
        copyright_file = Path("/usr/share/doc") / package / "copyright"
        if not copyright_file.is_file():
            raise RuntimeError(f"Missing copyright notice for {package}")
        copy_notice = target / "licenses" / f"{package}.copyright"
        copy_notice.parent.mkdir(exist_ok=True)
        shutil.copy2(copyright_file, copy_notice)
        packages[owner] = dict(zip(("package", "version", "source_package", "source_version"), metadata))
    # Debian copyright files may reference these common full licence texts.
    shutil.copytree("/usr/share/common-licenses", target / "licenses" / "common-licenses")
    command = GraphvizCommand(target / "bin" / "neato", target)
    _run(str(command.executable), "-c", env=command.environment())
    return {"origin": "Debian/Ubuntu build environment", "packages": packages}


def _dependencies(path):
    output = _run("ldd", str(path))
    if "not found" in output:
        raise RuntimeError(f"Missing native dependencies for {path}:\n{output}")
    return {Path(match) for match in re.findall(r"=>\s+(/\S+)", output)}


def stage(destination, *, source=None, source_url=None):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="graphviz-stage-", dir=destination.parent) as temporary:
        target = Path(temporary) / "graphviz"
        if source is not None:
            if not source_url:
                raise ValueError("Provide --source-url for the exact Graphviz source release.")
            shutil.copytree(source, target, symlinks=False)
            notices = [p for p in target.rglob("*") if p.is_file()
                       and any(word in p.name.lower() for word in ("license", "copying", "copyright"))]
            if not notices:
                raise ValueError("The source distribution must include its licences and third-party notices.")
            provenance = {"source_url": source_url}
        else:
            target.mkdir()
            provenance = _system_bundle(target)
        name = "neato.exe" if sys.platform == "win32" else "neato"
        command = GraphvizCommand(target / "bin" / name, target)
        if not command.executable.is_file():
            raise ValueError(f"The runtime must contain bin/{name}.")
        # Remove host executable and library discovery for the smoke check.
        environment = command.environment()
        environment["PATH"] = str(target / "bin")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join((str(target / "lib"), str(target / "lib" / "graphviz")))
        environment["XDG_CACHE_HOME"] = str(Path(temporary) / "font-cache")
        result = subprocess.run([str(command.executable), "-Tsvg"],
                                input='digraph { overlap=false; node [fontname="DejaVu Sans"]; a -> b [dir=both]; c -> b [style=dashed]; }',
                                capture_output=True, text=True, env=environment, timeout=20, check=True)
        if "<svg" not in result.stdout or result.stderr.strip():
            raise RuntimeError(f"Graphviz bundle smoke check failed: {result.stderr}")
        version = subprocess.run([str(command.executable), "-V"], capture_output=True, text=True,
                                 env=environment, check=True).stderr.strip()
        provenance.update(version=version, platform=sys.platform, files={
            str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(target.rglob("*")) if p.is_file() and p != target / "bundle.json"
        })
        (target / "bundle.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        target.rename(destination)
    # Recheck after relocation: plugin/font configuration must remain relative.
    render_svg(build_reaction_graph(["N2", "H2"], [], []), GraphvizCommand(destination / "bin" / name, destination))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--system", action="store_true", help="Collect Debian/Ubuntu Graphviz, plugins, native libraries, fonts and notices")
    choice.add_argument("--source", type=Path, help="A complete portable Graphviz distribution with bin/neato[.exe]")
    parser.add_argument("--source-url", help="Corresponding source release URL for --source")
    parser.add_argument("--destination", type=Path, default=REPO / "desktop" / "vendor" / "graphviz")
    args = parser.parse_args()
    print(stage(args.destination, source=args.source, source_url=args.source_url))


if __name__ == "__main__":
    main()
