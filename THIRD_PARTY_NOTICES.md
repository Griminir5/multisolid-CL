# Third-Party Notices

This project is licensed under `GPL-3.0-only`. Third-party packages,
installations, generated files, and optional tools retain their own licenses.
This file is a practical notice summary, not a replacement for the licenses of
those projects.

## Runtime Requirements

- DAETools: GPLv3. Install separately from the DAETools project or a package
  source you are licensed to use.
- PyQt6 and PyQt6-WebEngine: GPL-3.0-only unless used under Riverbank's
  commercial license.
- OpenCS: LGPLv3. Install or obtain separately when needed by the DAETools
  workflow.
- NumPy: BSD-3-Clause and other permissive notices in the NumPy distribution.
- xarray: Apache-2.0.
- pandas: BSD-3-Clause and bundled permissive third-party notices. It is an
  xarray dependency and is used by the optional ML matrix conversion tool.
- SciPy: BSD-3-Clause and bundled permissive third-party notices.
- matplotlib: matplotlib license and bundled third-party notices.
- pydantic: MIT.
- PyYAML: MIT.

## Optional Compiled Solver

- scikit-sundae supplies the separately installed SUNDIALS runtime used by the
  compiled backend. Its wheel and bundled libraries retain their own notices.
- The native Newton helper adapts SUNDIALS control flow under BSD-3-Clause.
  The bundled [license](packed_bed/compiled/licenses/SUNDIALS-LICENSE.txt) and
  [notice](packed_bed/compiled/licenses/SUNDIALS-NOTICE.txt) accompany the source
  and are included in package distributions.
- The optional vector exponential uses code extracted from SLEEF 3.9.0,
  copyright Naoki Shibata and contributors, under the
  [Boost Software License 1.0](packed_bed/compiled/licenses/SLEEF-LICENSE.txt).

## Optional Tools

- pygraphviz: BSD-3-Clause. PyPI binary wheels may bundle Graphviz components
  under separate licenses.
- Graphviz: Eclipse Public License 2.0 for current Graphviz distributions.
- VTK: BSD-3-Clause.
