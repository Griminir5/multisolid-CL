# Reaction graph runtime

The desktop and CLI share `packed_bed/reaction_graph.py` to generate DOT and invoke
the separate `neato -Tsvg` executable. The only layout adjustment is
`overlap=false`, agreed to keep labels from overlapping. Graphviz chooses node
positions and edge routes. No solver or valid operating program is needed for
the desktop graph.

Gas species stay green, solids sand, reactions violet, and missing species red.
Species are ellipses and reactions boxes. Reversible stoichiometric links have
arrows at both ends; catalysts and rate-only dependencies have dashed arrows into
the reaction. Edges have no coefficient labels. Scientific equations elsewhere
in the Chemistry editor retain their coefficients.

Qt displays the SVG elements directly. Clicking a node highlights that node,
its direct neighbours and the connecting links by fading unrelated elements.
Click again, click empty space, or press Escape to clear. Scroll or use +/− to
zoom; drag to pan; Fit or 0 resets the view. New layouts and window resizes fit
the whole graph. A 150 ms debounce and a cancellable QProcess keep edits responsive;
obsolete results cannot replace a newer graph. Rendering times out after 20 seconds.
If a previous graph is retained during an update or error, a visible status says
so. Retry runs the latest selection again.

## Development and executable discovery

Use one of these setups:

- Stage a runtime into `desktop/vendor/graphviz` using the tool below.
- Install Graphviz and put `neato` on `PATH`.
- Set `MULTISOLID_GRAPHVIZ` to a `neato` executable, including its filename.

The staged bundle takes precedence. A frozen application looks for `graphviz`
beside its executable and under PyInstaller's internal runtime directory before
considering the explicit override. Frozen applications do not search the host
`PATH`. Library, font and plugin paths are set only in the Graphviz child process.
The CLI uses the same discovery and rendering; `--artifacts` warns and skips the
graph if rendering fails, removing any stale `system_graph.svg` export.

## Stage the separate runtime

For a Debian/Ubuntu Linux build environment with Graphviz and DejaVu Sans installed:

```sh
python tools/bundle_graphviz.py --system
```

The collector copies neato, the core/neato/Pango plugins, their native-library
dependencies, a font, relative font configuration, plugin configuration and
package licence notices. The platform C runtime is supplied by the target OS:
build on the oldest supported distribution and verify against the release's OS
baseline. This does not make arbitrary Linux binaries universally portable.

For a complete portable Windows or Linux Graphviz distribution:

```sh
python tools/bundle_graphviz.py --source /path/to/graphviz-prefix \
  --source-url https://example.org/exact-corresponding-source-release
```

Supply the real corresponding source-release URL. The prefix must contain
`bin/neato` or `bin/neato.exe`, all required libraries/plugins, plugin configuration,
fonts where needed, and its licence/third-party notices. The full prefix is copied
to preserve its distribution layout. Windows distributions normally keep DLLs,
plugins and configuration in `bin`; Linux uses `lib/graphviz` for plugins.

Both modes refuse to overwrite an existing destination, run a minimal SVG smoke
check with host executable/library search paths removed, and record the runtime
version and SHA-256 file hashes in `bundle.json`. They also recheck after moving
the runtime to its destination, to detect non-relocatable configuration. Generated
runtime files are ignored by Git; source and platform-specific release builds
stage them explicitly. Ordinary Python wheels do not carry these native binaries.

For a PyInstaller folder release, use `--destination dist/MultiSolid/graphviz`
after building the application folder, before creating the installer or archive:

```text
MultiSolid/
  MultiSolid[.exe]
  _internal/...
  graphviz/
    bin/neato[.exe]
    bin/*.dll                     (Windows distribution)
    lib/graphviz/                 (Linux plugins and config)
    lib/*.so.*                    (Linux native dependencies)
    share/fonts/                  (Linux collector's bundled font)
    etc/fonts/fonts.conf          (relative font configuration)
    licenses/                     (or the supplied distribution's notices)
    bundle.json
```

Keep the Graphviz runtime separate from Python extensions and include it unchanged
in the installer/archive. Retain the original notices and corresponding sources
required by each actual distribution. The Debian collector records exact source
package names/versions for this purpose; it does not download source archives.
Pin the Graphviz distribution/build inputs in release automation and archive its
manifest alongside the application. Full Windows/Linux installer qualification
remains part of the packaging stage in `PLAN.md`.

## Verification

```sh
QT_QPA_PLATFORM=offscreen python -m pytest \
  tests/test_reaction_graph.py desktop/tests/test_graph_view.py
```

Coverage includes the example and all built-in reaction families, missing and
isolated species, catalysts/rate dependencies, reversibility, escaping, shared
CLI export, fitting, clicking, zoom/pan, rapid edits, obsolete results, empty
graphs, missing executables, malformed SVG, timeout and view destruction.
Release qualification additionally runs offline on each supported OS with no
host Graphviz installed; the Linux staging smoke check is not Windows validation.
