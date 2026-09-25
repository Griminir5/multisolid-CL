# Managed compiled execution

Select **Compiled** and a linear solver in General, then Run normally. Standard
and Compiled both expose KLU. Standard/SuperLU remains the new-case default.

Standard also offers SuperLU_MT, UMFPACK, dense LAPACK, AztecOO with ILUT,
Ifpack ILU or ML, and SUNDIALS GMRES with Ifpack ILU. These use the existing
DAE Tools SuperLU, SuperLU_MT and Trilinos adapters on both platforms. Missing
adapter modules disable their choices; native loading is checked in the worker
before replacing retained results. Pardiso is not enabled. The packaged
`--check-compiled` check loads every Standard adapter as well as the compiled runtime.

AztecOO retains its bundled linear convergence/preconditioner defaults and keeps
factorizations for DAE Tools' reuse requests. Its variants can still fail on
reactive cases: the 100-second default-chemistry check failed with ILUT and ML,
and ML failed the inert case when balance equations were included. AztecOO/Ifpack
and SUNDIALS GMRES/Ifpack have reactive agreement tests; the former differed from
SuperLU by at most 0.000073 K and `9.04e-7` in gas mole fraction in this check.

Compilation and reuse are automatic; no compiler paths or cache settings are
scientific inputs. The first build includes Zig's C++ support libraries and can
take several minutes. The status display identifies compilation and cache reuse.

Every desktop project owns `.packed_bed_cache/`. Cases and studies share it,
reruns preserve it, and archives omit it. Moving a project preserves compatible
entries; a different platform, compiler, runtime, engine, CPU capability, or model
rebuilds them. Deleting the cache while idle is safe. An unwritable cache blocks
Compiled instead of redirecting artifacts outside the project.

`solver.name: klu` works for both backends; `trilinos_klu` remains an alias.
DAE Tools' Amesos KLU initializes the model, and Compiled KLU integrates through
SUNDIALS `SUNLinSol_KLU`. KLU factorization is serial. The manifest records the
requested, initialization, and integration implementations, build identities,
case size, cache decision, and phase timings. `status.json` additionally records
full worker elapsed time after writing the results and manifest.

## Prepare a release runtime

End users do not run these commands. Build independently on Linux x86_64 and
Windows x64. Linux release builds use Ubuntu 22.04 for the glibc 2.35 baseline.
Windows builds use MSYS2 UCRT64 GCC, CMake, and Ninja with Python 3.12. DAE Tools
must already be installed for that Python/platform, including its SuperLU,
SuperLU_MT, and Trilinos/Amesos adapters. Install the root and desktop packages
into the release environment, plus PyInstaller, pytest, and pefile.

1. Run `python tools/build_compiled_runtime.py --work build/native --prefix build/native-prefix`.
2. Fetch the pinned Zig distribution using `fetch('zig_linux', Path('build/native'))`
   or `fetch('zig_windows', Path('build/native'))` from `tools.build_compiled_runtime`.
3. Run `python tools/bundle_compiled.py --compiler <extracted-zig-folder> --runtime build/native-prefix --destination desktop/vendor/compiled`.
4. Stage Graphviz following [GRAPHVIZ.md](GRAPHVIZ.md), then run
   `python tools/freeze_desktop.py --output dist/desktop`.

Use `CMAKE_GENERATOR=Ninja` in the Windows build environment. The build downloads
verified source archives on its first invocation; retain them with release build
assets. Subsequent builds can use these archives offline. Staging requires a new
destination, copies the complete compiler and native dependency set, creates a
hashed manifest and component notices, and runs mandatory native smoke checks
with external compilers removed from PATH. Keep the native source archives,
build recipe, hashes, and toolchain runtime notices alongside release artifacts.
Audit the notice/source inventory before redistribution.

Pinned components are Zig 0.16.0, SUNDIALS 7.5.0 (double/32-bit indices),
SuiteSparse 7.7.0, SuperLU_MT 4.0.1, and OpenBLAS 0.3.30. Their URLs and SHA-256
hashes are in [compiled_sources.json](../tools/compiled_sources.json). SUNDIALS and
its dependencies load as one bundle, separate from DAE Tools' native libraries.
Windows DLL dependencies are copied from the release toolchain; Linux copies
non-glibc dependencies. The frozen application requires its bundled compiler and
never searches for a system compiler.

Source execution detects `desktop/vendor/compiled`; `MULTISOLID_COMPILED_BUNDLE`
can select another staged bundle. Without one, the existing source/CLI compiler
discovery and scikit-SUNDAE wheel remain available. That wheel omits KLU, so
Compiled KLU is unavailable until a managed bundle is supplied.

## Verification and release gates

`python -m pytest` covers UI transactions, legacy settings, project snapshots,
cache coordination/recovery, cancellation, native callbacks, and reactor agreement.
`python -m packed_bed.compiled.smoke` exercises every exposed compiled solver and
the model, CPU, Band, and nonlinear helpers. Missing managed components fail the
check instead of becoming optional skips. A frozen executable supports the same
check through `MultiSolid --check-compiled` (`MultiSolid.exe` on Windows).

[The native workflow](../.github/workflows/compiled-runtime.yml) builds both
platform runtimes and runs their native checks. It does not establish clean-machine
desktop acceptance by itself. Run actual frozen bundles offline on Windows 11
and Ubuntu 22.04/24.04 without Python or development tools. Include read-only and
relocated installations, writable projects with spaces/Unicode, concurrent cases,
compiler cancellation, reports, archive round trips, and plugin expressions.
Windows behavior cannot be qualified from Linux tests.

Use `python -m tools.benchmark_compiled --cases <run.yaml> [<another-run.yaml>] --output <new-folder>`
to compare Standard, cold Compiled, and cached Compiled, including KLU, at one
and two workers. Keep the raw measurements and numerical comparisons. Cold
compiler-cache cost is part of the result; no universal speedup is assumed.

## Development validation (25 September 2026)

On Ubuntu 24.04 with Python 3.12 and DAE Tools 2.6:

- The expanded Standard selection passed 70 focused engine/UI tests, including
  real inert runs for every exposed identifier and reactive agreement for both
  Ifpack paths. The known missing `run_iterative.yaml` example test was deselected.
  ILUT/ML convergence limitations are described above; this follow-up has not
  been qualified on Windows or in a newly frozen bundle.
- All 118 compiled tests and six focused desktop tests passed. Numerical checks
  used GCC; the bundled Zig compiler was separately exercised by the mandatory
  native smoke check and real frozen desktop workers.
- Thirty scheduler/process tests passed; five Windows-only locking tests were
  skipped. Worker termination and coordinator failure stopped compiler children.
- The pinned native build recipe completed, and its staged runtime passed
  SuperLU, SuperLU_MT, KLU, Band, CPU-probe, and nonlinear-helper checks.
- A frozen application, with external tools absent from PATH and its installation
  read-only, completed a mixed Standard/Compiled batch. Identical KLU cases shared
  a compiled model. After relocating the installation to a path containing spaces
  and Unicode, all four Compiled cases reused their cache. The GUI also opened
  with Qt's offscreen platform. An archive round trip retained all five case solver
  selections and omitted results/cache. The wheel contains all helper sources,
  headers, and notices.
- The broad repository run passed 532 tests before reaching its failure limit.
  Its timing-dependent concurrency test was corrected and passes with explicit
  synchronization. Two unrelated existing failures remain: the missing
  `run_iterative.yaml` example and the stale `packed_bed.artifacts.find_spec` mock.

The smoke benchmark used copper/silica and heterogeneous nickel cases, each with
three cells, a 20-second horizon, relative tolerance `1e-6`, and concentration
absolute tolerance `1e-9`. Each batch had two distinct model kernels. These are
single-repeat measurements on a shared development host, not performance guarantees.

| Compiled solver | Workers | Standard batch (s) | Cold Compiled (s) | Cached Compiled (s) |
|---|---:|---:|---:|---:|
| KLU | 1 | 3.04 | 33.34 | 3.01 |
| KLU | 2 | 3.22 | 43.87 | 2.48 |
| SuperLU | 1 | 2.37 | 29.25 | 3.94 |
| SuperLU | 2 | 1.87 | 31.00 | 2.67 |
| Band | 1 | 3.08 | 86.86 | 10.68 |
| Band | 2 | 2.03 | 63.40 | 4.04 |

All repeats hit the cache. Maximum differences from their Standard reference were
0.00553 K in temperature, `5.04e-5` in gas mole fractions, and `1.01e-5` in solid
mole fractions. Reporting coordinates agreed within floating-point roundoff.
Raw measurements include worker startup, initialization, cache waiting, building,
integration, and output costs; cold CPU-helper compilation is included in compile time.

For KLU, the first sequential case spent 26.89 seconds compiling the CPU helper
within 27.76 seconds of total compilation. Cached KLU integration took about
3.06 ms and 1.18 ms for the two cases, compared with Standard's 101.93 ms and
32.08 ms. Startup, initialization and output dominate the full elapsed time for
such small cases, so the integration improvement barely changes the sequential
batch total. Standard already uses native numerical code; Compiled generates
model-specific code and reduces the system (139 to 36 and 175 to 39 unknowns in
these cases). It does not make KLU's factorization itself newly compiled.

These checks do not qualify a graphical release on clean machines. The development
host's freeze reported missing Qt X11 (`libxcb-cursor.so.0`) and TIFF plugin
dependencies; provision those on the release build host and check the resulting
GUI bundle. Windows 11 and clean Ubuntu 22.04/24.04 acceptance, including their
graphical runtime dependencies, remain outstanding.
