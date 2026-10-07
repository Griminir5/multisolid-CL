# Building the Windows distribution

The release uses CPython **3.11.9 x64**, DAE Tools **2.6.0** with its `py311`
extensions, PyInstaller **6.22.0**, Graphviz **12.2.1**, Zig **0.16.0** and
Inno Setup **6.7.3**. Build on Windows x64; this is not a cross-compilation recipe.
The installer targets Windows 11 x64 and installs for the current user.

Users need neither Python nor a compiler installation. The installer and portable
ZIP contain the same complete application folder. The ZIP has one `MultiSolid/`
directory: extract it fully and launch `MultiSolid.exe`. The managed compiler is
included because Compiled execution generates native models on the user's machine.

## Inputs and isolation

Run the commands below from the repository root. Keep the release inputs and logs
under `build/release/`. They are generated data, not Git source files.
[`tools/windows-release-inputs.json`](../tools/windows-release-inputs.json) records
upstream download URLs and SHA-256 hashes.
[`tools/compiled_sources.json`](../tools/compiled_sources.json) pins the native runtime.
`tools/windows-requirements.lock` pins the exact Windows/Python wheels, including
build tools; `windows-requirements.txt` is the readable version list.

Use a fresh, ordinary venv **without** `--system-site-packages`. Do not use the
developer `.venv`, editable installs, or the user's global packages:

```powershell
py -3.11 -m venv build/release/venv
$py = (Resolve-Path build/release/venv/Scripts/python.exe).Path
& $py -m pip download --no-cache-dir --only-binary=:all: --require-hashes -r tools/windows-requirements.lock --dest build/release/wheels
& $py -m pip install --no-index --find-links build/release/wheels --require-hashes -r tools/windows-requirements.lock
```

Prepare an extracted DAE Tools Windows distribution. Its upstream `setup.py`
creates shortcuts even during wheel builds, so the helper copies the tree and
removes that final block. It records the patch and input hashes and leaves the
original tree untouched:

```powershell
& $py tools/prepare_daetools.py daetools-2.6.0-win64 build/release/daetools
& $py -m pip wheel --no-cache-dir --no-deps --no-build-isolation --wheel-dir build/release/wheels build/release/daetools
& $py -m pip install --no-index --no-deps build/release/wheels/daetools-2.6.0-cp311-cp311-win_amd64.whl
```

Keep the prepared DAE Tools input tree and its hash manifest with the release
build assets. The C++ SDK is a separate upstream input; do not describe that SDK
alone as the complete source of DAE Tools. Its source repository is
<https://svn.code.sf.net/p/daetools/code/>.

DAE Tools must match the release Python version and include its SuperLU and
Trilinos/Amesos adapters. These supply Standard solvers and initialize models
before Compiled integration. The compiled runtime does not replace DAE Tools.

## Native bundles

Both native bundles are generated, platform-specific directories under
`desktop/vendor/`, ignored by Git. Ordinary application wheels do not contain
their binaries. Stage both before freezing; the freezer copies them beside the
application executable, preserving their manifests, libraries and notices.

### Compiled runtime

The managed runtime contains Zig **0.16.0**, SUNDIALS **7.5.0** with double
precision and 32-bit indices, SuiteSparse **7.7.0**, SuperLU_MT **4.0.1**, and
OpenBLAS **0.3.30**. SUNDIALS and its dependencies form a separate bundle from
DAE Tools' native libraries. The supported Compiled solvers are `superlu`, `klu`
and `band`; KLU factorization is serial.

Use MSYS2 UCRT64 GCC/CMake/Ninja, with CMake **3.22+** and
**libgomp explicitly installed**. Current MSYS2 splits the OpenMP runtime from
GCC. A build without it fails SuperLU_MT's
OpenMP check. Save `pacman -Q` and the downloaded package archives with the build.

```text
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-libgomp mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
```

In PowerShell, prepend that toolchain's `ucrt64/bin` directory to PATH and build:

```powershell
$env:CMAKE_GENERATOR = 'Ninja'
& $py tools/build_compiled_runtime.py --work build/native --prefix build/native-prefix --jobs 8
& $py -c "from pathlib import Path; from tools.build_compiled_runtime import fetch; print(fetch('zig_windows', Path('build/native')))"
& $py tools/bundle_compiled.py --compiler build/native/zig_windows-source/zig-x86_64-windows-0.16.0 --runtime build/native-prefix --destination desktop/vendor/compiled
```

The first build downloads the pinned source archives and verifies their SHA-256
hashes. The archives can be supplied in advance or reused for offline builds.
Keep them with the build recipe, patches and toolchain package records.

Staging verifies the SUNDIALS ABI, collects required runtime DLLs from the release
toolchain, and copies their notices. It records compiler/runtime identities and
file hashes in `manifest.json`, upstream inputs in `sources.json`, and copied
DLL hashes in `native-dependencies.json`. Toolchain notices must be available
under the MSYS2 prefix's `share/licenses` directory.

Staging checks all three Compiled solvers plus threaded SuperLU and cache reuse
with only the bundled compiler on PATH. It refuses to replace an existing vendor
folder. Native smoke checks run in a child process so Windows releases loaded
DLLs before the parent removes their scratch directory.

The removed `superlu_mt`, `trilinos_umfpack`, `intel_pardiso` and `trilinos_klu`
configuration choices stay unavailable. `klu` remains. Compiled `superlu` uses
SUNDIALS' SuperLU_MT internally; that native dependency is required, but the
DAE Tools `pySuperLU_MT` adapter is not.

### Graphviz runtime

The desktop and CLI invoke the separate `neato -Tsvg` executable through
`packed_bed/reaction_graph.py`. No Python Graphviz binding is needed. Keep this
runtime separate from Python extensions and include its complete staged layout
in the installer and portable ZIP.

For Graphviz, extract the three pinned archives (Windows binary ZIP, source
archive and exact Windows dependencies submodule). Then:

```powershell
& $py tools/prepare_graphviz_windows.py --binary build/release/graphviz/Graphviz-12.2.1-win64 --source build/release/graphviz-source/graphviz-12.2.1 --dependencies build/release/graphviz-dependencies/graphviz-windows-dependencies-ff985525b23a1a72ddb1a89482ea12233c3cbe85 --destination build/release/graphviz-ready
& $py tools/bundle_graphviz.py --source build/release/graphviz-ready --source-url https://gitlab.com/graphviz/graphviz/-/archive/12.2.1/graphviz-12.2.1.tar.gz
```

Preparation adds the matching licence texts and dependency provenance. It keeps
only neato, the core/layout/GDI+ modules and their required DLLs, using the native
Windows GDI+ font adapter. Windows DLLs, plugins and `config6` stay together in
`bin/`. The prepared prefix must contain `bin/neato.exe`, all required DLLs and
plugins, plugin configuration, any required fonts, and licence/third-party notices.

`bundle_graphviz.py --source` copies the full prepared prefix and requires the
exact corresponding source-release URL. Preparation and staging require new
destination directories. Staging checks SVG rendering with host executable and
library search paths removed, then repeats rendering after relocation to detect
absolute plugin/font paths. `bundle.json` records the Graphviz version, platform,
source URL and SHA-256 file hashes. Archive that manifest with the matching
upstream source and dependency notices. The tool's `--system` collector is for
Debian/Ubuntu builds and cannot supply the Windows runtime.

### Runtime discovery and compilation caches

Source execution detects `desktop/vendor/compiled`;
`MULTISOLID_COMPILED_BUNDLE` can select another staged bundle. Without a managed
bundle, source/CLI execution can use its native compiler discovery and the
optional scikit-SUNDAE wheel. That wheel does not supply KLU. Frozen applications
require their bundled compiler and runtime and do not search for a host compiler.

For Graphviz, the staged runtime takes precedence over the
`MULTISOLID_GRAPHVIZ` override, which must name the executable itself. Source
development can also find `neato` on `PATH`. Frozen applications look beside
their executable and under PyInstaller's internal runtime directory before the
explicit override; they do not search the host `PATH`. Library, plugin and font
paths are set in the Graphviz child process without changing the parent process.

The first Compiled run builds model code and Zig's C++ support libraries, which
can take several minutes. Desktop projects store reusable artifacts in their own
`.packed_bed_cache/`; cases and studies share it, while project archives omit it.
Platform, compiler, runtime, engine, CPU capability or model changes can require
recompilation. Projects need writable cache storage even when the application
installation is read-only. Cache reuse after relocation and compiler cancellation
are release checks, alongside successful first-run compilation.

## Freeze, validate and package

Install Inno Setup on the build machine. Given the isolated Python environment
and both staged vendor folders:

```powershell
./tools/build_windows_release.ps1 -ReleasePython build/release/venv/Scripts/python.exe -ISCC build/release/inno/ISCC.exe
```

This rebuilds and installs the two application wheels, generates the icon and
Windows version resources, freezes the app, checks the actual executable, and
creates the installer and ZIP under `dist/releases/`. Choose a new `-Validation`
directory for each attempt. Keep one current engine/UI wheel per wheelhouse.

The installer and portable ZIP exclude repository tools, tests, examples, sample
plugins and Property_Estimation. Only explicitly listed application data is
collected. A content audit rejects these unwanted files and modules. Existing
plugin support remains part of the engine; no plugin packages are preinstalled.
Zig's non-Windows C libraries and documentation are omitted. Three small pcurve
sources under Zig's own `tests` directory must remain: Zig 0.16 imports them when
building its C++ runtime. They are compiler inputs, not the repository test suite.

The freezer runs from a neutral working directory to prevent PyInstaller hooks
from collecting source-checkout results or compiler caches. The DAE Tools hook
collects the supported adapters and their DLL dependencies. Native vendor assets
remain outside PyInstaller's `_internal/` directory for reliable discovery.
Keep the engine's compiled helper `.cpp`/`.hpp` sources and notices in the wheels
and frozen application; model compilation reads these files at runtime.

The resulting application folder has this layout:

```text
MultiSolid/
  MultiSolid.exe
  _internal/                  Python, Qt, application and DAE Tools modules
  compiled/
    compiler/                 Zig executable and required compiler libraries
    lib/                      SUNDIALS and dependency DLLs
    licenses/
    manifest.json
    sources.json
    native-dependencies.json
  graphviz/
    bin/                      neato.exe, DLLs, plugins and config6
    licenses/
    bundle.json
  licenses/                   Application dependency notices
  LICENSE.txt
  README.txt
  BUILD-INFO.json
  CONTENTS.json
  release.json
  bundle-files.json
```

To diagnose a staged native runtime, run
`& $py -m packed_bed.compiled.smoke` from the repository root. It checks generated
model code, solver callbacks, vector exponentials, Band and nonlinear helpers,
and cache reuse. The frozen executable exposes the same check through
`MultiSolid.exe --check-compiled`, which also loads every supported Standard
solver adapter. These focused checks complement the full executable acceptance
check below.

The external `tools/release_check.py` diagnostic script checks native solver imports, a real Qt
window, graphs, both program modes, Standard and all three Compiled simulations,
two concurrent workers, repeat-run cache hits, NetCDF reads, Excel export,
project archives, cancellation and paths containing spaces/Unicode. The external
check also runs an external Python plugin from its separate test workspace. It clears development
search paths and isolates temporary files, application data and approvals.
The diagnostic script and its test inputs are not shipped. The executable loads
that script only when its path is explicitly supplied through the hidden
`--diagnostics-script` command-line argument together with `--self-test`.

Packaging requires a passing validation record for the exact executable and
bundle inventory. It refuses modified, missing or unrecorded bundle files.
`SHA256SUMS.txt` identifies the deliverables. Keep the validation logs alongside
the release, not inside the application folder.

After packaging, extract the final ZIP to a new directory containing spaces and
Unicode characters and run the external checker against that extracted folder:

```powershell
& $py tools/check_windows_release.py --bundle 'build/release/Portable check α/MultiSolid' --output build/release/portable-check
```

Also check per-user installation, same-version reinstallation and the installed
executable. In the release environment, run `check_windows_release.py` against
the installed application folder, then uninstall and verify that a project
stored outside the application survives. Record the final artifact hashes and
validation results together; `dist/releases/VALIDATION.json` accompanies the
current build. Clean-machine qualification is described below.

The installer uses a stable application ID and version/build-specific application
directories so upgrades cannot mix DLL versions. Shortcuts point to the current
build. Older application directories remain until uninstall. Projects, settings
and project-local caches are outside installer ownership; uninstallation does
not target them. Save projects outside the installation/extracted folder.

## Release qualification

These scripts create **unsigned** builds. Public releases still need an
Authenticode certificate and signing of the application before validation, then
the completed installer. Regenerate the bundle inventory after signing and run
validation again before packaging. Signing a final artifact changes its hash;
regenerate `SHA256SUMS.txt` last.

Restricted-PATH checks on a development machine do not replace an offline clean
Windows 11 VM test. Test the extracted ZIP and installed executable on a machine
without Python, Graphviz, Visual Studio, MSYS2 or Zig, including upgrade/uninstall
and preservation of a project stored outside the app. Include read-only and
relocated installations, writable projects with spaces/Unicode, concurrent cases,
compiler cancellation, cache reuse, reports, archive round trips and plugin
expressions. Windows Sandbox was not available on this build host. Native CI
builds and Linux development checks do not establish Windows desktop acceptance;
this Windows work does not qualify a Linux release either.

Keep the application source, dependency notices, exact build inputs, patches,
upstream source provenance and validation records with each distributed build.
The optional source/build-assets archive is for maintainers; ordinary users need
only one of the installer or portable ZIP.
