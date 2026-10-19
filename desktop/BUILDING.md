# Build MultiSolid from a checkout

Build on the operating system you want to run: **Ubuntu 24.04 x86_64 with
Python 3.12**, or **Windows 11 x64 with Python 3.11**. These are the pinned
desktop build environments; this is not a cross-compilation recipe. Source
development still supports Python 3.11 and 3.12.

Clone the repository, install the prerequisites below, then run one command
from the repository root. You do not need to activate a virtual environment.
The builder creates its own environment, installs hash-pinned Python packages,
downloads verified DAETools and native inputs, builds the compiler/solver bundle,
stages Graphviz, builds application wheels, freezes the desktop, tests a relocated
copy, and creates an installer and portable archive. Allow several GB of disk space and time for downloads and
the first native build. No administrator privileges are needed after installing
the system prerequisites.

## Linux

On Ubuntu 24.04, install the build and GUI dependencies:

```sh
sudo apt update
sudo apt install -y python3.12 python3.12-venv git build-essential gfortran cmake binutils \
  graphviz fonts-dejavu-core libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 \
  libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0 libxcb-randr0 \
  libxcb-shape0 libxcb-xfixes0 libxcb-sync1 libxcb-xkb1 libgl1 libegl1 \
  libfontconfig1 libdbus-1-3 xvfb xauth
python3.12 tools/build_desktop.py --headless
```

`--headless` uses Xvfb with Qt's real X11 backend. From a graphical desktop you
can omit it to see the acceptance-test windows. The build fails early if a
required Qt library is missing; `libxcb-cursor0` is required even for headless
X11 validation. Other distributions need equivalent packages and their own
qualification. The automatic Graphviz collector currently supports Debian/Ubuntu.

The finished downloads are under `dist/linux/releases/`: a
`MultiSolid-<version>-linux-x64-setup.deb` installer and a
`MultiSolid-<version>-linux-x64-portable.tar.gz` archive, with `SHA256SUMS.txt`.
The installer adds `/opt/multisolid`, an applications-menu entry, and the
`multisolid` command. Building it uses `dpkg-deb` without root privileges.
Both downloads include Python, DAETools, Qt, Graphviz, the compiler and solver
libraries. See [Linux details](BUILD_LINUX.md) for compatibility and manual steps.

## Windows

Install CPython **3.11.9 x64**, Git, and MSYS2. In the **MSYS2 UCRT64** terminal,
install the native build tools (including the separately packaged OpenMP runtime):

```text
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-libgomp mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
```

Then use **PowerShell**, from the repository root:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
py -3.11 tools/build_desktop.py
```

Adjust `C:\msys64` if you installed MSYS2 elsewhere. Use ordinary Windows Python,
not MSYS2 Python. Keep MSYS2's package licence files installed: staging includes
them with the native runtime. DAETools, Zig, and the pinned Windows Graphviz
archives are downloaded automatically.

The finished downloads under `dist/windows/releases/` are
`MultiSolid-<version>-windows-x64-setup.exe` and
`MultiSolid-<version>-windows-x64-portable.zip`, with `SHA256SUMS.txt`.
The builder downloads and verifies Inno Setup 6.7.3 and prepares its compiler
in portable mode inside the build workspace. Installer generation is automatic.
To use an existing compiler instead, supply its path:

```powershell
py -3.11 tools/build_desktop.py --iscc 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
```

The existing [Windows release guide](BUILD_WINDOWS.md) describes manual staging,
installer qualification, and release records. Its PowerShell release wrapper
continues to work for environments staged by hand.

## GitHub downloads and releases

Every push runs **Check application** on Linux and Windows. Each platform's
artifact contains its installer, portable archive, and checksums. The workflow
also installs, reinstalls, launches, and uninstalls each installer on a fresh
runner, checking that the external simulation project survives removal. The
Windows job uses Server 2025 so the installer can enforce its Windows 11-era
minimum kernel version; this still requires separate Windows 11 qualification.

Pushing a version tag such as `v1.0.0` runs the same checks, then prepares a
**draft GitHub Release** with all four application downloads and one combined
`SHA256SUMS.txt`. The tag must match the package version. Open **Releases**,
review the draft, and publish it when ready. A failed build or installer check
prevents draft creation. Rerunning a tag can refresh a draft; published releases
are never overwritten by this workflow.

## Source setup and packaging checks

Use the [repository installation and development instructions](../README.md#development).
Opening projects and generating numerical previews with built-in definitions do
not require DAETools. Python plugins may require it when their implementations
load; both execution backends require it. From an activated source environment,
`packed-bed-ui` and `python -m packed_bed_ui` launch the same application.
An optional argument may be a project folder or its `project.json`.

For relevant engine or packaging changes, run these from the repository root:

```sh
python -m packed_bed examples/simulations/default_case/run.yaml --validate-only
python -m packed_bed.compiled.smoke
python -m pip wheel --no-deps --no-build-isolation --wheel-dir build/wheels ./desktop
```

The compiled smoke check requires a working compiler and a managed native runtime
including KLU; the scikit-SUNDAE wheel alone does not supply KLU.
Wheel building checks desktop packaging separately. See
[Building the Windows distribution](BUILD_WINDOWS.md) for release
isolation, runtime staging, and installer/portable builds.

## Caches, inputs and troubleshooting

- Work files and logs are under `build/release-linux/` or `build/release-windows/`.
  `--work PATH` and `--output PATH` choose another build cache and distribution
  directory. Use a new work directory when changing Python versions or toolchains.
- `--jobs 4` controls native build parallelism. Use fewer jobs on small machines.
- `--daetools PATH` accepts an extracted DAETools 2.6.0 release or a matching
  wheel when you already have it. Native extensions must match the build Python
  version and include SuperLU and Trilinos. The official releases are linked at
  [DAETools downloads](https://daetools.sourceforge.io/downloads.html).
- `--compiled-bundle PATH` and `--graphviz-bundle PATH` reuse prepared bundles.
  The freezer still checks Linux runtime relocation and GUI dependencies.
- Downloads are cached and hash checked. For a checksum mismatch, remove the
  named archive and retry; do not bypass verification. A failed or interrupted
  build can be rerun. Application wheels are rebuilt each time.
- Keep the build environment separate from your development environment. Extra
  Python packages can affect PyInstaller discovery. The supplied command owns
  its dedicated venv; use the low-level tools for custom environments.
- `acceptance-*/check/validation.json` records validation for the exact bundle.
  A failure leaves logs beside that file and prevents packaging. The acceptance
  test covers simulations, concurrent workers, caches, reports, project archives,
  cancellation, external plugins, and paths with spaces/Unicode.
- `example-check/run-*/result.json` records the full reacting example check,
  including complete reporting times, finite results, solid conversion, plots,
  and mass/heat balance limits. Its `simulation.log` and copied case outputs are
  retained even when the check fails. A failure prevents packaging.
- CI's `installer-check/validation.json` records the exact download hashes,
  portable extraction, installed desktop checks, reinstall, and uninstall.
  Its logs are retained in the check-results artifact even after failure.
- Release checks remove developer compiler, Python, Graphviz and library paths.
  Linux runs must not need an `LD_LIBRARY_PATH` workaround. A missing SUNDIALS
  library usually means an old native bundle needs rebuilding/restaging.

`SHA256SUMS.txt`, `release.json`, `BUILD-INFO.json`, and `bundle-files.json`
identify the artifacts. Retain the sources, downloaded inputs and validation
logs when distributing a build. Always qualify the final extracted archive on
a clean target machine; a passing build-host check does not establish support
for every Linux distribution or Windows configuration.

## Release version and source identity

Update the engine version in `pyproject.toml`, the UI version in
`desktop/pyproject.toml`, and its exact `multisolid-cl[compiled]` dependency
together. Run `python tools/build_help.py`, then `python tools/build_help.py --check`.
A Git tag alone does not change the application version. Use `1.0.0rc1` while
qualifying a release candidate; change all three inputs to `1.0.0` for the final
release and rebuild. Windows resources and the installer keep the full display
version and use its numeric release components separately.

`tools/build_app_wheels.py` stamps both wheels with the source commit and dirty
state before copying the source. The freezer requires these stamps to match the
release inputs, then writes the shared application identity into `release.json`.
The content-derived build identifier includes the shipped source stamps. Run
manifests, desktop snapshots, and About diagnostics retain this identity without
requiring Git. Build distributable releases from a clean committed checkout;
local changes are recorded as `source_dirty: true`, not as a new commit.
