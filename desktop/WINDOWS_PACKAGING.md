# Windows standalone packaging

The Windows build implementation now has pinned release dependencies, isolated
wheel-based freezing, managed native bundles, executable resources, an Inno Setup
installer recipe, portable ZIP packaging and checks of the packaged executable.
See [BUILD_WINDOWS.md](BUILD_WINDOWS.md) for the build commands, artifact layout
and remaining signing/clean-machine qualification steps.

The assessment below records the starting state; its missing-tool and
missing-recipe observations are historical.

## Initial assessment

Assessed on 2026-09-28 against [PLAN 1.md](../PLAN%201.md), the current
packaging code, and the local Windows development environment.

MultiSolid can be distributed as a normal Windows application without asking
users to install Python, pip, Graphviz, or a compiler. The existing architecture
supports this. A release bundle and installer still need to be built and
qualified; the working development environment is not yet that bundle.

## Recommended deliverable

Build a Windows x64 **PyInstaller folder bundle**, then wrap the entire folder
in a per-user **Inno Setup installer**, for example
`MultiSolid-0.1.0-windows-x64-setup.exe`.

PyInstaller includes the interpreter used to build the application along with
its collected dependencies. End users do not install Python separately.
[PyInstaller operating model](https://pyinstaller.org/en/stable/operating-mode.html)

Use `PrivilegesRequired=lowest` for an installer that does not request elevation,
install under the user's application directory, create a Start menu shortcut,
and register an uninstaller.
[Inno Setup install privileges](https://jrsoftware.org/ishelp/topic_setup_privilegesrequired.htm)

The single download installs a directory resembling:

```text
MultiSolid/
  MultiSolid.exe
  _internal/             Python, Qt, Python packages, DAE Tools and native DLLs
  graphviz/              neato, plugins, dependencies and notices
  compiled/              managed compiler, solver runtime and notices
  licenses/              application and dependency notices
```

Keep projects, user settings and project compilation caches outside this
directory. The installer must not run pip, download runtime components, or
modify the user's Python installation. Build tools are needed on the release
machine; users receive their required runtime components already packaged.

## What already exists

| Area | Existing implementation |
| --- | --- |
| Desktop application | Separate `packed_bed_ui` package and graphical entry point. |
| Freezing | `tools/freeze_desktop.py` invokes PyInstaller with `--onedir --windowed`, collects application packages, and adds DAE Tools extension imports. |
| Worker processes | `packed_bed_ui.__main__` calls `freeze_support()` before argument parsing; execution and plugin checking relaunch the frozen executable. |
| Graphviz | `tools/bundle_graphviz.py` stages and checks a relocatable runtime; frozen execution discovers the adjacent bundle. |
| Compiled execution | Runtime build/staging scripts, pinned native source hashes, bundle discovery, integrity checks and project-local caches already exist. |
| Native CI | `.github/workflows/compiled-runtime.yml` builds Windows and Linux compiled runtimes. It does not build the complete desktop installer. |
| Results | NetCDF reads/writes explicitly use SciPy; the absence of `netCDF4` and `h5netcdf` here is not a missing requirement for these paths. |

## Gaps observed on this machine

1. **The Python environment is not isolated.** `.venv/pyvenv.cfg` sets
   `include-system-site-packages = true`. Python is 3.11.9 x64; DAE Tools,
   PyQt6, NumPy, SciPy, xarray and openpyxl resolve to the user's global Python
   installation. The application packages are editable installs. Rebuild in a
   separate environment without global site packages and install built wheels.

2. **The build tools and staged runtime assets are incomplete.** PyInstaller
   is absent. Neither `desktop/vendor/graphviz` nor `desktop/vendor/compiled`
   exists. Inno Setup was not found on PATH or in the inspected Program Files
   location, and no installer recipe exists in the repository.

3. **Solver choices have been narrowed after this assessment.** The initially
   missing Standard SuperLU_MT adapter and unsupported UMFPACK implementation
   are no longer release requirements. The `superlu_mt`, `trilinos_umfpack`,
   `intel_pardiso` and legacy `trilinos_klu` choices have been removed from
   configuration and execution. Existing cases require an explicit replacement;
   ordinary KLU remains available as `klu`.

   The remaining Standard choices are SuperLU, KLU, LAPACK, the three AztecOO
   variants and SUNDIALS GMRES/Ifpack. They constructed successfully on this
   machine. Construction is a native-loading check, not numerical qualification.
   Compiled SuperLU still needs SUNDIALS' SuperLU_MT library internally, but uses
   the ordinary DAE Tools SuperLU adapter for initialization. Keep that internal
   library in the managed runtime; it is distinct from the removed adapter.

4. **Compiled execution currently depends on host development tools.** No
   managed bundle is detected. Compiler discovery selects the installed Visual
   Studio 2022 Build Tools. The local scikit-sundae installation is a development
   fallback, not the planned managed distribution. The freezer excludes
   `sksundae`, so shipping the managed native runtime is essential for the full
   release. Follow [COMPILED.md](COMPILED.md) to build and stage it.

5. **Graphviz currently comes from the host installation.** `neato.exe` resolves
   under `C:\Program Files\Graphviz`. Stage a complete portable distribution
   with its DLLs, plugins, configuration and notices using
   [GRAPHVIZ.md](GRAPHVIZ.md). Copying only `neato.exe` is insufficient.

6. **The release inputs are not fully pinned.** Most Python dependencies are
   unconstrained; no release dependency lock was found. The native build guide
   and CI use Python 3.12, while this installed DAE Tools tree exposes only
   `py311` extension modules. Select an interpreter version with matching DAE
   Tools binaries and test it explicitly. DLL filenames mentioning other Python
   versions are not evidence that the corresponding extension modules exist.

7. **The installer and full release pipeline are missing.** Add installation,
   upgrade and uninstall behavior, version metadata, release artifact inventory,
   notices/source assets, and the signing step required by the plan for public
   releases. The top-level notice file still describes separately installed
   dependencies and is not an inventory of a finished standalone distribution.

## Build sequence

1. **Establish a reproducible Windows build environment.** Choose the Python/
   DAE Tools combination; record exact Python package versions and hashes,
   DAE Tools artifact provenance and any local patches. Build and install the
   engine/UI wheels without editable installs or global site packages. Verify
   the release's required solver capabilities before freezing.

2. **Stage and validate native assets.** Build the managed compiled runtime and
   stage the pinned compiler with `tools/build_compiled_runtime.py` and
   `tools/bundle_compiled.py`. Stage Graphviz with `tools/bundle_graphviz.py`.
   Preserve component provenance and notices alongside the assets.

3. **Freeze and inspect the application.** Run
   `python tools/freeze_desktop.py --output dist/desktop` in that prepared
   environment. Audit the resulting files, PyInstaller warnings, Qt platform
   and SVG plugins, DAE Tools dynamically loaded modules, examples, C++ source/
   header assets, and xarray's SciPy backend discovery. Check native DLL
   dependencies and include permitted runtime libraries needed on clean Windows.
   The broad `--collect-all daetools` is a starting point: verify that the
   resulting distribution excludes unsupported/local-only components, including
   the Pardiso runtime excluded by the plan.

   The current freezer **requires both staged runtimes**, even for Standard
   execution. To make the plan's Standard-only packaging proof independently,
   first add an explicit proof-build option and verify that compiled choices
   report unavailable. Otherwise stage both runtimes before the first build.
   Do not present a reduced proof build as the complete release.

4. **Create the installer.** Add a versioned Inno Setup recipe consuming the
   tested application folder. Use a stable application ID for upgrades, a
   per-user location and a shortcut. Preserve projects/settings/extensions;
   handle obsolete application-owned files during upgrades. Attach the
   application licence and collected component notices. Start with an unsigned
   internal test installer; configure the plan's Authenticode signing stage
   before public distribution.

5. **Qualify the installed artifact on a clean Windows 11 x64 VM.** Disconnect
   networking and use a standard account with no Python, Graphviz, Visual
   Studio, MSYS2, Zig or separately installed solver packages. Test the actual
   installed executable, with no access to the checkout or developer caches.

   Acceptance includes first launch; a Standard simulation; concurrent cases;
   cancellation; NetCDF results and Excel export; reaction graphs; project
   archive transfer; an approved code plugin; first and cached Compiled runs;
   paths containing spaces/Unicode; relocation of the bundle; and upgrade/
   uninstall preserving user data. A GUI launch alone is not sufficient.
   Exercise subprocesses in the frozen environment, where library search paths
   and multiprocessing differ from source execution.
   [PyInstaller runtime pitfalls](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html)

6. **Automate the proven recipe.** Extend CI from the native runtime job to
   wheel builds, dependency staging, freezing, packaged smoke checks, installer
   creation, checksums and release assets. Keep clean-machine acceptance as an
   explicit release gate. Linux packaging remains a separate build and
   qualification task under the original plan.

The immediate milestone is a relocatable Windows application folder that runs
without developer installations. Wrapping that verified folder in an installer
is the next step. No rewrite into another language is needed for this approach.

## Verification performed for the initial assessment

- Imported the desktop window and attempted to construct every then-advertised
  Standard solver choice. SuperLU_MT and UMFPACK failed; those choices have
  subsequently been removed as described above.
- Confirmed the interpreter/package locations, absent staged bundles and
  PyInstaller, host Graphviz path, and Visual Studio compiler fallback.
- Ran `python -m pytest tests/test_graphviz_bundle.py
  tests/compiled/test_managed.py desktop/tests/test_worker.py -q` using `.venv`:
  **18 passed, 1 skipped**. The skipped test requires a staged Graphviz bundle
  and checks its relocation without host executables. The passing tests include
  source-environment worker execution and compiled cache coordination; they do
  not prove standalone installation.
- No frozen executable or installer was produced during this assessment, and
  clean-machine acceptance remains outstanding.
