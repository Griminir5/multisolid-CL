"""Package the supplied reactor case for the README's desktop walkthrough.

Run from a source installation with: python tools/create_example_project.py
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from packed_bed_ui.project import Project
from packed_bed_ui.project_archive import export_project, import_project


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "examples/projects/first-reactor.msproject"


def main():
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="multisolid-example-") as temporary:
        root = Path(temporary)
        project = Project.create(root / "source", "First reactor")
        case = project.add_case_from_files(
            ROOT / "examples/simulations/default_case/run.yaml", "Baseline"
        )
        case.resolve()
        export_project(project, DESTINATION)
        imported = import_project(DESTINATION, root / "imported", "First reactor")
        restored = Project.open(imported)
        assert len(restored.cases) == 1
        assert restored.cases[0].documents == case.documents
        restored.cases[0].resolve()
    print(DESTINATION)


if __name__ == "__main__":
    main()
