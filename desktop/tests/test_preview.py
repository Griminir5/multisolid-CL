from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

from packed_bed.config import load_case
from packed_bed.preview import preview_case
from packed_bed.programs import DEFAULT_SMOOTH_RAMP_WIDTH_S


@pytest.mark.parametrize("filename", ["run.yaml", "run_feed_stream.yaml"])
def test_preview_matches_compiled_programs_across_horizon(filename):
    source = Path(__file__).resolve().parents[2] / "packed_bed/examples/default_case" / filename
    case = load_case(source)
    preview = preview_case(case)
    assert preview.time_s[0] == 0
    assert preview.time_s[-1] == case.run.simulation.time_horizon_s
    assert np.all(np.diff(preview.time_s) > 0)
    for samples, program in (
        (preview.flow_mol_s, case.inlet_flow_program),
        (preview.temperature_k, case.inlet_temperature_program),
        (preview.pressure_pa, case.outlet_pressure_program),
        (preview.mole_fractions, case.inlet_composition_program),
    ):
        for index in np.linspace(0, len(preview.time_s) - 1, 13, dtype=int):
            np.testing.assert_allclose(samples[index], program.value_at(
                preview.time_s[index], smooth_ramp_width_s=DEFAULT_SMOOTH_RAMP_WIDTH_S,
            ))
    np.testing.assert_allclose(preview.mole_fractions.sum(axis=1), 1)


def test_preview_keeps_cell_and_face_coordinates_and_concentration_basis(source_case):
    preview = preview_case(load_case(source_case))
    np.testing.assert_allclose(preview.cell_positions_m, [1/6, 1/2, 5/6])
    np.testing.assert_allclose(preview.face_positions_m, [0, 1/3, 2/3, 1])
    np.testing.assert_allclose(preview.solid_concentrations_mol_m3_bed, [[0.3, 0.3, 0.3]])
    np.testing.assert_allclose(preview.particle_diameter_m, [0.001] * 4)


def test_headless_services_do_not_import_qt_daetools_or_pyplot(source_case):
    code = '''
import sys
class Guard:
    def find_spec(self, fullname, *args):
        if fullname.startswith(("daetools", "PyQt6", "matplotlib.pyplot")):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from packed_bed_ui import project, worker
from packed_bed.config import load_case
from packed_bed.preview import preview_case
preview_case(load_case(sys.argv[1]))
'''
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    subprocess.run([sys.executable, "-c", code, str(source_case)], env=environment, check=True, capture_output=True)
