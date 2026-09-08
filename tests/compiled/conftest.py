from __future__ import annotations
from importlib.util import find_spec
import pytest


@pytest.fixture
def native_tools():
    if find_spec("sksundae") is None:
        pytest.skip("Compiled tests need the compiled optional dependency")
    from packed_bed.compiled.compiler import find_toolchain
    from packed_bed.compiled.runtime import check_runtime

    try:
        find_toolchain()
        check_runtime()
    except RuntimeError as exc:
        pytest.skip(str(exc))
