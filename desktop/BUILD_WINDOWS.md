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

Keep the release inputs and logs under `build/release/`. They are generated data,
not Git source files. `tools/windows-release-inputs.json` records upstream download
URLs and SHA-256 hashes. `tools/compiled_sources.json` pins the native runtime.
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

## Native bundles

Use MSYS2 UCRT64 GCC/CMake/Ninja, with **libgomp explicitly installed**. Current
MSYS2 splits the OpenMP runtime from GCC. A build without it fails SuperLU_MT's
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

Staging checks all three Compiled solvers plus threaded SuperLU and cache reuse
with only the bundled compiler on PATH. It refuses to replace an existing vendor
folder. Native smoke checks run in a child process so Windows releases loaded
DLLs before the parent removes their scratch directory.

For Graphviz, extract the three pinned archives (Windows binary ZIP, source
archive and exact Windows dependencies submodule). Then:

```powershell
& $py tools/prepare_graphviz_windows.py --binary build/release/graphviz/Graphviz-12.2.1-win64 --source build/release/graphviz-source/graphviz-12.2.1 --dependencies build/release/graphviz-dependencies/graphviz-windows-dependencies-ff985525b23a1a72ddb1a89482ea12233c3cbe85 --destination build/release/graphviz-ready
& $py tools/bundle_graphviz.py --source build/release/graphviz-ready --source-url https://gitlab.com/graphviz/graphviz/-/archive/12.2.1/graphviz-12.2.1.tar.gz
```

Preparation adds the matching licence texts and dependency provenance. It keeps
only neato, the core/layout/GDI+ modules and their required DLLs, using the native
Windows GDI+ font adapter.
Staging checks rendering without host executables and again after relocation.

The removed `superlu_mt`, `trilinos_umfpack`, `intel_pardiso` and `trilinos_klu`
configuration choices stay unavailable. `klu` remains. Compiled `superlu` uses
SUNDIALS' SuperLU_MT internally; that native dependency is required.

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
./tools/test_windows_installer.ps1 -Installer dist/releases/MultiSolid-0.1.0-windows-x64-setup.exe -ReleasePython $py -WorkArea build/release/installer-check
```

The installer test requires a new workspace directory and no existing registered
MultiSolid installation. It temporarily registers a per-user installation, tests
reinstallation and the installed executable, then uninstalls it and checks that
an external user file survives. Record the final artifact hashes and validation
results together; `dist/releases/VALIDATION.json` accompanies the current build.

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
and preservation of a project stored outside the app. Windows Sandbox was not
available on this build host. This Windows work does not qualify the separate
Linux release described in PLAN 1.md.

Keep the application source, dependency notices, exact build inputs, patches,
upstream source provenance and validation records with each distributed build.
The optional source/build-assets archive is for maintainers; ordinary users need
only one of the installer or portable ZIP.
