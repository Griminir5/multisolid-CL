from pathlib import Path
import os

import pytest

from packed_bed.compiled import _export_stack_files


@pytest.mark.skipif(os.name != "nt", reason="DAE Tools Windows narrow-path exporter")
@pytest.mark.parametrize("fail", [False, True])
def test_native_export_in_unicode_folder_restores_working_directory(tmp_path, fail):
    directory = tmp_path / "Project with spaces \u03b1"
    directory.mkdir()
    previous = Path.cwd()

    class NativeExporter:
        def ExportComputeStackStructs(self, equations, indexes):
            # Mimic a native library accepting only narrow ASCII file names.
            equations.encode("ascii")
            indexes.encode("ascii")
            Path(equations).write_bytes(b"equations")
            if fail:
                raise RuntimeError("native export failed")
            Path(indexes).write_bytes(b"indexes")

    if fail:
        with pytest.raises(RuntimeError, match="native export failed"):
            _export_stack_files(NativeExporter(), directory)
    else:
        equations, indexes = _export_stack_files(NativeExporter(), directory)
        assert equations.read_bytes() == b"equations"
        assert indexes.read_bytes() == b"indexes"
    assert Path.cwd() == previous
