from __future__ import annotations
import platform
from importlib.util import find_spec
import pytest


@pytest.fixture
def native_tools():
    if (
        platform.system() != "Windows"
        or find_spec("sksundae") is None
        or find_spec("daetools") is None
    ):
        pytest.skip(
            "Compiled tests need Windows, DAETools and the compiled optional dependency"
        )
    from packed_bed.compiled.compiler import find_toolchain

    try:
        find_toolchain()
    except RuntimeError as exc:
        pytest.skip(str(exc))
