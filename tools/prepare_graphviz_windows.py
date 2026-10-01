"""Add the matching upstream notices to the official Graphviz Windows ZIP."""
import argparse
from pathlib import Path
import os
import re
import shutil
import sys


def prepare(binary, source, dependencies, destination):
    binary, source, dependencies, destination = map(Path, (binary, source, dependencies, destination))
    import pefile
    from importlib.util import find_spec
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "bin").mkdir()
    qt = Path(find_spec("PyQt6").origin).parent / "Qt6/bin"
    search = [binary / "bin", Path(sys.base_prefix), qt]
    system = Path(os.environ["SystemRoot"]) / "System32"
    plugins = ("gvplugin_core.dll", "gvplugin_neato_layout.dll", "gvplugin_gdiplus.dll")
    todo, seen = ["neato.exe", *plugins], set()
    while todo:
        name = todo.pop()
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        path = next((folder / name for folder in search if (folder / name).is_file()), None)
        if path is None:
            if name.lower().startswith(("api-ms-", "ext-ms-")) or (system / name).is_file():
                continue
            raise FileNotFoundError(f"Missing Graphviz runtime dependency: {name}")
        shutil.copy2(path, destination / "bin" / path.name)
        with pefile.PE(str(path)) as pe:
            todo.extend(entry.dll.decode() for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", ()))
    notices = destination / "licenses"
    notices.mkdir()
    # The Windows Pango build expects installed fonts. Prefer the native GDI+
    # font adapter, which supports Windows font substitution without setup.
    plugin_config = destination / "bin/config6"
    original = (binary / "bin/config6").read_text()
    blocks = re.findall(r"(?ms)^(gvplugin_\w+\.dll .*?^\})", original)
    original = "\n".join(block for block in blocks if block.split()[0] in plugins) + "\n"
    if "textlayout 8" not in original:
        raise ValueError("Expected Graphviz 12.2.1 GDI+ font adapter configuration")
    plugin_config.write_text(original.replace("textlayout 8", "textlayout 11"))
    for name in ("LICENSE", "COPYING"):
        shutil.copy2(source / name, notices / ("Graphviz-" + name))
    for path in source.rglob("*"):
        if path.is_file() and path.name.lower() in {"license", "copying", "copyright"}:
            target = notices / "graphviz-source" / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    # This is the exact dependencies submodule revision of the Graphviz release.
    # Preserve both licence texts and vcpkg provenance for its distributed DLLs.
    share = dependencies / "vcpkg/installed/x64-windows/share"
    for folder in share.iterdir():
        if folder.is_dir():
            for name in ("copyright", "vcpkg.spdx.json", "vcpkg_abi_info.txt"):
                path = folder / name
                if path.is_file():
                    target = notices / "dependencies" / folder.name / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
    for path in dependencies.glob("x64/*"):
        if path.is_file() and any(s in path.name.lower() for s in ("license", "copying", "copyright")):
            shutil.copy2(path, notices / path.name)
    (notices / "README.txt").write_text(
        "Graphviz 12.2.1 official Windows x64 portable distribution.\n"
        "Dependency notices/provenance: Graphviz windows/dependencies submodule\n"
        "ff985525b23a1a72ddb1a89482ea12233c3cbe85.\n"
        "The matching upstream archives and application build scripts are retained\n"
        "in the accompanying source/build-assets archive.\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "source", "dependencies", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(prepare(**vars(parser.parse_args())))
