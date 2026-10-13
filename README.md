# MultiSolid-CL

MultiSolid-CL simulates gas flow, heat transfer, as well as gas-gas and gas-solid reactions in packed-bed
reactors. It provides a Python simulation engine, a command-line interface for
YAML-defined cases and batches, and a desktop application for projects, parameter studies,
plots, and Excel reports. Simulations use DAETools, with an optional compiled
SUNDIALS backend. Run data are saved as NetCDF with a JSON provenance manifest.

## Documentation

The four guides under `docs/` are outlines for the planned documentation. The
existing plugin and desktop references remain available while their content is
migrated.

| Document | Contents | Status |
| --- | --- | --- |
| [Plugin guide](docs/plugins.md) | Installing, using, developing, checking, and sharing plugins | Outline; links to the existing guide |
| [Model equations](docs/equations.md) | Governing equations, assumptions, boundary conditions, and source terms | Outline |
| [GUI guide](docs/gui-guide.md) | Projects, cases, studies, plots, and Excel reports | Outline; links to the existing guide |
| [CLI guide](docs/cli-guide.md) | YAML inputs, single runs, batches, solver options, and outputs | Outline |
| [Windows build guide](desktop/BUILD_WINDOWS.md) | Building, validating, and packaging the desktop distribution | Available |
| [Plugin examples](examples/plugins/README.md) | Worked species, property, and kinetics examples | Available |
| [Waste iron plugin](plugins/waste_iron/README.md) | Installation, kinetics, property definitions, and provenance | Available |
| [Kinetics sources](packed_bed/kinetics/KINETICS_SOURCES.md) | References for the built-in reaction families | Available |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | Dependency attribution and accompanying licences | Available |

## Install the desktop application

The first public desktop release is pending. Once published, download it from
[GitHub Releases](https://github.com/Griminir5/multisolid-CL/releases). Until then,
use the source installation below, or follow the
[Windows build guide](desktop/BUILD_WINDOWS.md) to build a distribution.

The Windows distribution targets **Windows 11 x64**. Choose one release asset:

| Asset | Installation |
| --- | --- |
| `MultiSolid-<version>-windows-x64-setup.exe` | Run the installer, then open MultiSolid from the Start menu. Installation is for the current user. |
| `MultiSolid-<version>-windows-x64-portable.zip` | Extract the entire ZIP into a new folder, then run `MultiSolid/MultiSolid.exe`. Keep its files and subfolders together. |

Both distributions include Python, DAETools, Graphviz, and the compiler and
libraries needed for Compiled execution. End users do not need to install these
separately. GitHub's automatically generated source archives are source code;
choose the installer or portable ZIP for the ready-to-run application.

Create projects in a writable folder outside the application directory. The first
Compiled run may take several minutes to build its model; compatible later runs
reuse the project's cache. To update a portable installation, extract the new
version into a new folder and open your existing project. The
[current desktop reference](desktop/README.md) explains the project workflow.

## Install the full repository from source

Use **Python 3.11 or 3.12** and Git. These steps install the engine, CLI, desktop
application, compiled-backend Python dependencies, and DAETools plotter support
in one environment. Windows and Linux need a matching DAETools distribution and
its native dependencies.

### 1. Get the source and create an environment

The desktop application currently lives on the `application-and-ui` branch;
the repository's default branch does not contain this layout.

```sh
git clone --branch application-and-ui https://github.com/Griminir5/multisolid-CL.git
cd multisolid-CL
python -m venv .venv
```

Activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or in a Linux shell:

```sh
source .venv/bin/activate
```

Use `python3.11` or `python3.12` to create the environment if your default `python`
is a different version. Run the remaining repository commands from its root.

### 2. Install the Python packages

```sh
python -m pip install -e ".[compiled,plotter]" -e ./desktop
```

Editable installation makes local Python changes available without reinstalling.
The desktop package includes Excel export dependencies. Release builds use a
separate, pinned environment described in the Windows build guide.

### 3. Install DAETools

Download [DAETools 2.6.0](https://daetools.sourceforge.io/downloads.html) for your
operating system and Python version. Extract it, then, with the same virtual
environment active, install from the extracted DAETools directory:

```sh
python -m pip install .
```

Follow the DAETools installation instructions for its native dependencies. The
distribution must include the solver adapters you intend to use; the supplied
single-case example uses SuperLU. Compiled execution also uses DAETools to build
the model and solve its initial conditions. Return to the MultiSolid repository
root after installation.

### 4. Install the external tools

- **Graphviz:** install the native Graphviz distribution with `neato` on `PATH`,
  or set `MULTISOLID_GRAPHVIZ` to the full path of its executable. This supplies
  the chemistry graph and CLI system diagram; no Python Graphviz binding is needed.
- **C++ compiler:** for source Compiled execution, install GCC or Clang on Linux
  and make it available on `PATH`; `CXX` can select its executable. On Windows,
  install Visual Studio Build Tools with C++ x64 support; the application locates
  its toolchain through the Visual Studio installer.

Without a managed native bundle, Compiled execution uses the pinned
`scikit-sundae==1.1.3` wheel's SUNDIALS runtime. Arbitrary system SUNDIALS libraries
are not interchangeable with that runtime. Compiled KLU requires the managed
bundle. See [native bundles](desktop/BUILD_WINDOWS.md#native-bundles) for staging
the Windows compiler, solver libraries, and Graphviz runtime. Source execution
also detects staged bundles under `desktop/vendor/`.

### 5. Check the installation and launch

```sh
python -m packed_bed examples/simulations/default_case/run.yaml --validate-only
packed-bed-ui
```

`python -m packed_bed_ui` also launches the desktop application. To run the
supplied case from the terminal:

```sh
python -m packed_bed examples/simulations/default_case/run.yaml
```

Input validation does not need DAETools. Execution requires the selected solver
and its dependencies. Each case run can replace files in its output directory;
use a separate directory to retain earlier results.

## Repository structure

| Path | Purpose |
| --- | --- |
| [`packed_bed/`](packed_bed/) | Simulation engine, CLI, configuration, scientific definitions, and reports |
| [`packed_bed/compiled/`](packed_bed/compiled/) | Native code generation, compiler integration, and SUNDIALS runtime support |
| [`desktop/`](desktop/) | Desktop package metadata, Windows build instructions, and installer definition |
| [`desktop/packed_bed_ui/`](desktop/packed_bed_ui/) | PyQt6 application, project storage, editors, studies, and Excel export |
| [`docs/`](docs/) | Plugin, equation, GUI, and CLI guide outlines |
| [`examples/simulations/`](examples/simulations/) | Complete single-case and batch YAML inputs |
| [`examples/plugins/`](examples/plugins/) | Illustrative plugin packages |
| [`plugins/`](plugins/) | Additional scientific plugins, including waste iron |
| [`tools/`](tools/) | Runtime staging, wheel builds, desktop freezing, and release checks |
| [`.github/workflows/`](.github/workflows/) | Native runtime CI |

Generated runtimes, build outputs, and caches are excluded from Git. Licences and
scientific provenance notes stay beside the components they describe.

## Development

Use the editable source installation above. Keep release builds separate from
the development environment; the
[Windows build guide](desktop/BUILD_WINDOWS.md) defines the pinned release inputs
and acceptance checks.

Useful starting points are [configuration and validation](packed_bed/config/),
[model equations](packed_bed/model.py), [initialization](packed_bed/initialization.py),
[operating programs](packed_bed/programs.py), [simulation execution](packed_bed/simulation.py),
and [reports](packed_bed/reports.py). Desktop navigation starts in
[window.py](desktop/packed_bed_ui/window.py); project persistence is in
[project.py](desktop/packed_bed_ui/project.py).

Use [scientific plugins](docs/plugins.md) for new species, property correlations,
or reaction families that can be distributed independently. Changes to built-in
definitions should retain their scientific references, units, and concentration
bases. Keep DAETools imports lazy where possible so input validation remains
usable without the solver. When changing the model, preserve equation names,
ordering, and solver incidence unless the change deliberately requires otherwise.

Check the example inputs after configuration changes:

```sh
python -m packed_bed examples/simulations/default_case/run.yaml --validate-only
python -m packed_bed batch examples/simulations/default_batch_case/batch.yaml --validate-only
```

For native runtime changes, run the smoke check with a working compiler and
runtime. On Linux, expose the staged runtime's `lib` directory through
`LD_LIBRARY_PATH` if needed:

```sh
python -m packed_bed.compiled.smoke
```

For packaging changes, build both application wheels with the configured build
dependencies installed:

```sh
python -m pip wheel --no-deps --no-build-isolation --wheel-dir build/wheels . ./desktop
```

Use complete reactor runs, balance diagnostics, and grid/tolerance comparisons
when changing numerical behaviour. Input validation and native smoke checks
alone do not validate a scientific model. Update the relevant documentation with
changes to interfaces, equations, or defaults.

## Attribution and citation

A **Zenodo DOI is planned for the first release**. Until it is issued, cite the
[repository](https://github.com/Griminir5/multisolid-CL) and record the version,
Git commit, and access date used for your work. For example, replace the fields
in this provisional citation:

> MultiSolid-CL contributors. *MultiSolid-CL* [Computer software]. Version
> VERSION, commit COMMIT. https://github.com/Griminir5/multisolid-CL
> (accessed DATE).

Once the first release is archived, this section will provide its Zenodo citation
and DOI. Cite the scientific sources for the kinetics and property data you use
as well: see the [built-in kinetics references](packed_bed/kinetics/KINETICS_SOURCES.md)
and the documentation shipped with each plugin. Run manifests record source and
configuration provenance to help identify the implementation behind results.

MultiSolid-CL is licensed under [GPL-3.0-only](LICENSE). Dependency attribution
and notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); bundled
components retain their accompanying licence texts.
