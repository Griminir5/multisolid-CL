import pytest
import yaml


@pytest.fixture
def source_case(tmp_path):
    """A short, inert physical case suitable for real worker integration checks."""
    source = tmp_path / "source"
    source.mkdir()
    documents = {
        "run": {
            "references": {f"{name}_file": f"{name}.yaml" for name in ("chemistry", "program", "solids")},
            "simulation": {"system_name": "DesktopTest", "time_horizon_s": 0.01, "reporting_interval_s": 0.01,
                           "mass_scheme": "weno3", "heat_scheme": "weno3", "report_time_derivatives": False},
            "model": {"bed_length_m": 1.0, "bed_radius_m": 0.01, "axial_cells": 3,
                      "ambient_temperature_k": 300.0, "heat_transfer_coefficient_w_per_m2_k": 0.0},
            "solver": {"backend": "daetools", "name": "superlu", "threads": 1, "relative_tolerance": 1e-5},
            "outputs": {"directory": str(tmp_path / "original-output"),
                        "artifacts_directory": str(tmp_path / "original-artifacts"),
                        "requested_reports": ["temperature", "pressure", "gas_mole_fraction"], "requested_plots": []},
        },
        "chemistry": {"gas_species": ["N2"], "reaction_families": [], "reaction_ids": []},
        "program": {
            "inlet_flow": {"initial": 1e-8, "steps": []},
            "inlet_temperature": {"initial": 300.0, "steps": []},
            "outlet_pressure": {"initial": 100000.0, "steps": []},
            "inlet_composition": {"initial": {"N2": 1.0}, "steps": []},
        },
        "solids": {"solid_species": ["Ni"], "initial_profile": {"basis": "solid", "zones": [
            {"x_start_m": 0.0, "x_end_m": 1.0, "e_b": 0.4, "e_p": 0.5, "d_p": 0.001, "values": {"Ni": 1.0}},
        ]}},
    }
    for name, document in documents.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return source / "run.yaml"


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PyQt6.QtWidgets")
    app = widgets.QApplication.instance() or widgets.QApplication([])
    yield app
