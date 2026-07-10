from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from packed_bed.incidence_matrix import write_solver_incidence_artifacts


def test_incidence_artifacts_are_labelled_csv_and_static_png(tmp_path: Path) -> None:
    execution_info = SimpleNamespace(
        EquationIndex=0,
        Name="bed.species_balance_cell_0_N2",
        EquationType="daeEquationType.eDifferential",
        VariableIndexes=(0, 1),
        DiffVariableIndexes=(1,),
    )
    model = SimpleNamespace(
        Name="bed",
        Equations=(SimpleNamespace(EquationExecutionInfos=(execution_info,)),),
        OverallIndex_BlockIndex_VariableNameMap={
            0: (0, "bed.c_gas(N2,cell_0)"),
            1: (1, "bed.temp_bed(cell_0)"),
        },
    )

    artifacts = write_solver_incidence_artifacts(model=model, output_dir=tmp_path)

    assert set(artifacts) == {
        "solver_incidence_matrix_csv",
        "solver_incidence_matrix_png",
    }
    with artifacts["solver_incidence_matrix_csv"].open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "row_index": "0",
            "equation": "species_balance_cell_0_N2",
            "equation_type": "eDifferential",
            "column_index": "0",
            "variable": "c_gas(N2,cell_0)",
            "incidence_kind": "variable",
        },
        {
            "row_index": "0",
            "equation": "species_balance_cell_0_N2",
            "equation_type": "eDifferential",
            "column_index": "1",
            "variable": "temp_bed(cell_0)",
            "incidence_kind": "variable_and_time_derivative",
        },
    ]
    assert artifacts["solver_incidence_matrix_png"].read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
