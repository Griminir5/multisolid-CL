# Building the Linux distribution

Start with the [shared build instructions](BUILDING.md). The supported build
baseline is **Ubuntu 24.04 x86_64 / Python 3.12**. The automated command is:

```sh
python3.12 tools/build_desktop.py --headless
```

It uses `tools/linux-requirements.lock` for Python wheels and
`tools/desktop-inputs.json` / `tools/compiled_sources.json` for native downloads.
The readable version list is `tools/linux-requirements.txt`. The DAETools input
is its full Linux release, not the C++ SDK alone. The selected Python's native
extensions are installed into the dedicated build environment.

## Native portability

The runtime contains SUNDIALS 7.5.0, SuiteSparse 7.7.0, SuperLU_MT 4.0.1,
OpenBLAS 0.3.30, and Zig 0.16.0. The SUNDIALS build applies a recorded source
patch because upstream overwrites the requested CMake install RPATH. Installed
libraries use `$ORIGIN`. Staging rejects absolute search paths, dependencies
resolved outside the bundle (except the platform C runtime), and missing
libraries. Its smoke check runs without `LD_LIBRARY_PATH` and with only the
bundled compiler on `PATH`.

The generated `release.json` records the **actual minimum glibc version required
by shipped ELF files**. The compiler's `gnu.2.35` target alone does not determine
the application's minimum OS: Python, Qt and copied system libraries can require
a newer glibc. Build on the oldest OS you intend to support. Ubuntu 24.04 builds
are not promised to run on Ubuntu 22.04. Supporting an older base requires a
matching Python/Qt/dependency environment and clean-machine validation there.

The Qt X11 plugin is checked before freezing. Its required dependencies must be
available to PyInstaller. TIFF is not an application input/output format, so the
unused optional Qt TIFF plugin is omitted. The DAETools hook collects only the
supported adapters and their dependency closure. A physical `daetools/pyDAE`
directory is included so DAETools can find its configuration on POSIX systems.

## Manual build and diagnostic commands

After creating an ordinary isolated Python 3.12 venv and installing the pinned
requirements and a matching DAETools 2.6.0 wheel, the underlying steps are:

```sh
python tools/build_help.py --check
python tools/build_app_wheels.py --output build/release-linux/application-wheels
python -m pip install --no-deps --force-reinstall build/release-linux/application-wheels/*.whl
python tools/build_compiled_runtime.py --work build/native --prefix build/native-prefix --jobs 4
python -c "from pathlib import Path; from tools.build_compiled_runtime import fetch; print(fetch('zig_linux', Path('build/native')))"
python tools/bundle_compiled.py --compiler build/native/zig_linux-source/zig-x86_64-linux-0.16.0 --runtime build/native-prefix --destination desktop/vendor/compiled
python tools/bundle_graphviz.py --system
python tools/freeze_desktop.py --output dist/linux
xvfb-run -a python tools/check_desktop_release.py --bundle dist/linux/MultiSolid --output build/linux-validation --qt-platform xcb
python tools/package_linux.py --bundle dist/linux/MultiSolid --output dist/linux/releases --validation build/linux-validation/validation.json
```

Use new staging and validation destinations; the individual staging tools refuse
to overwrite existing bundles. The automated builder manages its own generated
staging directories. With custom native bundle paths, pass `--compiled-bundle`
and `--graphviz-bundle` to the freezer.

The validator copies the application into a path containing spaces and Unicode,
clears developer search paths, runs the frozen application's checks, and verifies
that no bundled files changed. Packaging requires this passing record and checks
the inventory again. The tar archive preserves permissions and symlinks. Test
the extracted archive with the same validator before publishing, including a
clean Linux machine or VM and a real graphical desktop.
