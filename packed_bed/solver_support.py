"""Shared desktop solver choices and inexpensive runtime checks."""

from importlib.util import find_spec

DESKTOP_SOLVERS = {
    "daetools": (
        "superlu", "superlu_mt", "klu", "trilinos_klu", "trilinos_umfpack",
        "trilinos_lapack", "trilinos_aztecoo", "trilinos_aztecoo_ifpack",
        "trilinos_aztecoo_ml", "sundials_gmres_ifpack",
    ),
    "compiled": ("superlu", "superlu_mt", "klu", "trilinos_klu", "band"),
}
SOLVER_LABELS = {
    "superlu": "SuperLU", "superlu_mt": "SuperLU_MT", "klu": "KLU",
    "trilinos_klu": "KLU (legacy identifier)", "band": "Band",
    "trilinos_umfpack": "UMFPACK", "trilinos_lapack": "LAPACK (dense)",
    "trilinos_aztecoo": "AztecOO / ILUT",
    "trilinos_aztecoo_ifpack": "AztecOO / Ifpack ILU",
    "trilinos_aztecoo_ml": "AztecOO / ML",
    "sundials_gmres_ifpack": "SUNDIALS GMRES / Ifpack ILU",
}


def require_desktop_solver(case, *, native=False):
    settings = case.run.solver
    if settings.name not in DESKTOP_SOLVERS.get(settings.backend, ()):
        raise ValueError("Select a supported desktop solver: " + ", ".join(
            SOLVER_LABELS[name] for name in DESKTOP_SOLVERS.get(settings.backend, ()) if name != "trilinos_klu"))
    try:
        if settings.backend == "compiled":
            from .compiled.compiler import find_toolchain
            from .compiled.runtime import check_runtime, load_runtime_library
            from .compiled.bundle import asset, bundle_root, manifest
            find_toolchain()
            folder = check_runtime()
            module = "sunlinsolklu" if settings.name in {"klu", "trilinos_klu"} else (
                "sunlinsolband" if settings.name == "band" else "sunlinsolsuperlumt")
            root = bundle_root()
            if root:
                asset(root, manifest(root)["runtime"]["libraries"][module])
            elif not any(folder.glob("*" + module + "*")):
                raise RuntimeError("KLU is absent from this runtime. Use the managed compiled bundle with KLU support.")
            if native:
                load_runtime_library(folder, module)
        initial_solver = "superlu" if settings.name == "band" else settings.name
        module = initial_solver if initial_solver in {"superlu", "superlu_mt"} else "trilinos"
        if find_spec("daetools.solvers." + module) is None:
            raise RuntimeError(f"The DAE Tools {SOLVER_LABELS[initial_solver]} component is missing.")
        if native:
            from .simulation import create_linear_solver
            create_linear_solver(initial_solver)
    except (ImportError, RuntimeError, OSError, KeyError) as exc:
        raise ValueError(f"{settings.backend.capitalize()} execution is unavailable: {exc}") from exc
