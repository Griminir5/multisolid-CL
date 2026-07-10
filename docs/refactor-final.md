# Refactor final comparison

Measured on the Phase 7 candidate on 2026-07-10, against the Phase 0 baseline
in [`refactor-baseline.md`](refactor-baseline.md).

## Size and import boundary

| Metric | Phase 0 | Phase 7 | Change |
|---|---:|---:|---:|
| Production Python modules | 40 | 26 | -14 |
| Production physical lines (`wc -l`) | 12,011 | 7,322 | -4,689 (-39.0%) |
| Top-level private helpers | 277 | 139 | -138 (-49.8%) |
| Top-level public functions | 166 | 87 | -79 (-47.6%) |
| Classes | 76 | 61 | -15 (-19.7%) |
| `import packed_bed` maximum RSS | 9 MB | 9 MB | unchanged |
| `import packed_bed.config` maximum RSS | 96 MB | 32 MB | -64 MB |
| DAETools modules loaded by `packed_bed.config` | 25 | 0 | -25 |

The measured import times were approximately 0.01 s for `packed_bed` and
0.13 s for `packed_bed.config`. Neither the configuration import nor
`--validate-only` loads DAETools, `pyUnits`, or xarray.

## Runtime and output comparison

The three-cell N2/Ni inert check still completes with Amesos KLU and preserves
the baseline sequence of 43 declared equations when both integral balances are
requested. The initialized model has 64 expanded solver equation rows in that
configuration; omitting the mass and heat reports removes their three and four
accounting rows respectively.

The equivalent all-report inert run still records two time samples and the
baseline cell, face, gas-species, and solid-species shapes. Those values now
have labelled dataset names and dimensions in `results.nc`; derived program,
outlet, mole-fraction, and balance-error fields share that same dataset. Every
successful run also writes `manifest.json`. The incidence diagnostic is now a
labelled CSV edge list plus a static PNG rather than HTML and solver-dependent
XPM variants.

## Verification

The final suite passes 86 tests, including the real inert solver run, declared
equation ordering, report-controlled DAE size, dataset/manifest round trips,
batch timeout and path behavior, pre- and post-run plots, and incidence
artifacts. The only warnings are DAETools imports of the Python 3.12-deprecated
standard-library `cgi` and `cgitb` modules.
