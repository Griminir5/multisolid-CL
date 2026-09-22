"""Smoke-test relocation of the separately staged native runtime."""

import hashlib
import json
from pathlib import Path

import pytest

from packed_bed.reaction_graph import GraphvizCommand, build_reaction_graph, render_svg
from tools.bundle_graphviz import stage


def test_staging_refuses_to_replace_existing_files(tmp_path):
    destination = tmp_path / "graphviz"
    destination.mkdir()
    marker = destination / "keep"
    marker.write_text("existing runtime")
    with pytest.raises(FileExistsError):
        stage(destination)
    assert marker.read_text() == "existing runtime"


def test_incomplete_source_distribution_is_not_published(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="source-url"):
        stage(tmp_path / "runtime", source=source)
    with pytest.raises(ValueError, match="licences"):
        stage(tmp_path / "runtime", source=source, source_url="https://example.org/source")
    assert not (tmp_path / "runtime").exists()


def test_staged_bundle_relocates_without_host_executables(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "desktop" / "vendor" / "graphviz"
    if not source.is_dir():
        pytest.skip("Run tools/bundle_graphviz.py to exercise native bundle relocation")
    destination = stage(tmp_path / "runtime with spaces", source=source,
                        source_url="https://graphviz.org/download/")
    manifest = json.loads((destination / "bundle.json").read_text())
    assert "bundle.json" not in manifest["files"]
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
    # Exercise a second relocation and override host plugin/font settings.
    relocated = tmp_path / "relocated"
    destination.rename(relocated)
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("GVBINDIR", str(tmp_path / "missing-plugins"))
    monkeypatch.setenv("FONTCONFIG_FILE", str(tmp_path / "missing-fonts.conf"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    name = "neato.exe" if (relocated / "bin" / "neato.exe").exists() else "neato"
    command = GraphvizCommand(relocated / "bin" / name, relocated)
    graph = build_reaction_graph(["H2", "N2"], [], [])
    svg = render_svg(graph, command)
    assert b"<svg" in svg
    assert b"N2" in svg and b"H2" in svg
