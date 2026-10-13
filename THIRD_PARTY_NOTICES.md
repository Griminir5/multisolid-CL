# Third-Party Notices

This project is licensed under `GPL-3.0-only`. Third-party packages,
installations, generated files, and optional tools retain their own licenses.
This file is a practical notice summary, not a replacement for the licenses of
those projects.

## Runtime Requirements

- DAETools: GPLv3. Source installations obtain it separately from the DAETools
  project or a package source you are licensed to use. The Windows desktop
  distribution bundles its required runtime and notices.
- PyQt6 and PyQt6-WebEngine: GPL-3.0-only unless used under Riverbank's
  commercial license.
- OpenCS: LGPLv3. Install or obtain separately when needed by the DAETools
  workflow.
- NumPy: BSD-3-Clause and other permissive notices in the NumPy distribution.
- xarray: Apache-2.0.
- pandas: BSD-3-Clause and bundled permissive third-party notices. It is an
  xarray dependency and supports conversion of labelled results to tables.
- SciPy: BSD-3-Clause and bundled permissive third-party notices.
- matplotlib: matplotlib license and bundled third-party notices.
- Maratype by Artyom Galinov: bundled desktop display font, freely
  redistributable with no commercial restrictions.
- pydantic: MIT.
- PyYAML: MIT.

## Optional Compiled Solver

- scikit-sundae supplies the SUNDIALS runtime for source Compiled execution
  without a managed bundle. Its wheel and bundled libraries retain their own
  notices. Managed desktop builds include a separate native runtime with its
  own dependency notices under `compiled/licenses/`.
- The native Newton helper adapts SUNDIALS control flow under BSD-3-Clause.
  The bundled [license](packed_bed/compiled/licenses/SUNDIALS-LICENSE.txt) and
  [notice](packed_bed/compiled/licenses/SUNDIALS-NOTICE.txt) accompany the source
  and are included in package distributions.
- The optional vector exponential uses code extracted from SLEEF 3.9.0,
  copyright Naoki Shibata and contributors, under the
  [Boost Software License 1.0](packed_bed/compiled/licenses/SLEEF-LICENSE.txt).

## Optional Tools

- Graphviz is invoked as a separate executable; no pygraphviz binding is used.
  Its applicable licence and all bundled native-library/font notices travel with
  the staged distribution. `tools/bundle_graphviz.py` retains these notices and
  records versions, source metadata and file hashes in `graphviz/bundle.json`.
  See [Graphviz bundling](desktop/BUILD_WINDOWS.md#graphviz-runtime) for Windows staging
  and the [release layout and checks](desktop/BUILD_WINDOWS.md#freeze-validate-and-package).
- VTK: BSD-3-Clause.
