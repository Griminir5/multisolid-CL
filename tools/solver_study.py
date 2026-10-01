"""Reproducible experiments with existing solver controls; no solver modifications.

Prepare an immutable case corpus, run resumable isolated subprocesses, and compare
losslessly archived outputs. Use ``python -m tools.solver_study --help``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
import ctypes
from datetime import datetime
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
SEED = 20260929
BASE = dict(backend="compiled", name="band", threads=1, relative_tolerance=1e-5,
            concentration_absolute_tolerance=1e-6, max_nonlinear_iterations=20,
            nonlinear_convergence_coefficient=.1, maximum_order=5,
            scale_residuals=True, step_growth_threshold=1.25,
            nonlinear_refresh_interval=4, suppress_algebraic_errors=False,
            vector_exponentials=False, band_reciprocals=False)
TIGHT = dict(BASE, relative_tolerance=1e-6, concentration_absolute_tolerance=1e-8)
STOCK = dict(max_nonlinear_iterations=4, nonlinear_convergence_coefficient=.33,
             scale_residuals=False, step_growth_threshold=2., nonlinear_refresh_interval=0)
PROFILES = {
    "proposed": BASE,
    "tight": TIGHT,
    "rtol_only": dict(BASE, relative_tolerance=1e-6),
    "atol_only": dict(BASE, concentration_absolute_tolerance=1e-10),
    "trace": dict(TIGHT, concentration_absolute_tolerance=1e-11),
    "trace_1e10": dict(TIGHT, concentration_absolute_tolerance=1e-10),
    "trace_refresh0": dict(TIGHT, concentration_absolute_tolerance=1e-11, nonlinear_refresh_interval=0),
    "trace_newton8": dict(TIGHT, concentration_absolute_tolerance=1e-11, max_nonlinear_iterations=8),
    "trace_order3": dict(TIGHT, concentration_absolute_tolerance=1e-11, maximum_order=3),
    "trace_stock": dict(TIGHT, concentration_absolute_tolerance=1e-11, **STOCK),
    "trace_superlu": dict(TIGHT, concentration_absolute_tolerance=1e-11, name="superlu"),
    "trace_superlu4": dict(TIGHT, concentration_absolute_tolerance=1e-11, name="superlu", threads=4),
    "stock": dict(TIGHT, **STOCK),
    "no_scaling": dict(TIGHT, scale_residuals=False),
    "refresh_0": dict(TIGHT, nonlinear_refresh_interval=0),
    "refresh_2": dict(TIGHT, nonlinear_refresh_interval=2),
    "refresh_8": dict(TIGHT, nonlinear_refresh_interval=8),
    "growth_2": dict(TIGHT, step_growth_threshold=2.),
    "newton_8": dict(TIGHT, max_nonlinear_iterations=8),
    "newton_12": dict(TIGHT, max_nonlinear_iterations=12),
    "coefficient_033": dict(TIGHT, nonlinear_convergence_coefficient=.33),
    "suppress_algebraic": dict(TIGHT, suppress_algebraic_errors=True),
    "order_3": dict(TIGHT, maximum_order=3),
    "vector_exp": dict(TIGHT, vector_exponentials=True),
    "reciprocals": dict(TIGHT, band_reciprocals=True),
    "superlu": dict(TIGHT, name="superlu"),
    "klu": dict(TIGHT, name="klu"),
    "reference": dict(TIGHT, relative_tolerance=1e-7, concentration_absolute_tolerance=1e-11),
    "reference_order1": dict(TIGHT, relative_tolerance=1e-7, concentration_absolute_tolerance=1e-11, maximum_order=1),
    "reference_order2": dict(TIGHT, relative_tolerance=1e-7, concentration_absolute_tolerance=1e-11, maximum_order=2),
    "reference_refresh1": dict(TIGHT, relative_tolerance=1e-7, concentration_absolute_tolerance=1e-11, nonlinear_refresh_interval=1),
    "reference_newton50": dict(TIGHT, relative_tolerance=1e-7, concentration_absolute_tolerance=1e-11, max_nonlinear_iterations=50),
    "refined": dict(TIGHT, relative_tolerance=1e-8, concentration_absolute_tolerance=1e-12),
    "standard_reference": dict(TIGHT, backend="daetools", name="superlu", scale_residuals=False,
                               step_growth_threshold=2., nonlinear_refresh_interval=0,
                               relative_tolerance=1e-7, concentration_absolute_tolerance=1e-11),
    "standard": dict(TIGHT, backend="daetools", name="superlu", scale_residuals=False,
                     step_growth_threshold=2., nonlinear_refresh_interval=0),
    "standard_klu": dict(TIGHT, backend="daetools", name="klu", scale_residuals=False,
                         step_growth_threshold=2., nonlinear_refresh_interval=0),
}
REPORTS = ["temperature", "pressure", "gas_mole_fraction", "solid_mole_fraction",
           "gas_flux", "mass_balance", "heat_balance"]


class WindowsJob:
    """Contain a benchmark worker and native compiler children for bounded timeouts."""
    def __init__(self, process):
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [("process_time",ctypes.c_longlong),("job_time",ctypes.c_longlong),
                        ("flags",w.DWORD),("min_working",ctypes.c_size_t),("max_working",ctypes.c_size_t),
                        ("active",w.DWORD),("affinity",ctypes.c_size_t),("priority",w.DWORD),("scheduling",w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name,ctypes.c_ulonglong) for name in ("read_ops","write_ops","other_ops","read_bytes","write_bytes","other_bytes")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic",Basic),("io",IO),("process_memory",ctypes.c_size_t),("job_memory",ctypes.c_size_t),
                        ("peak_process",ctypes.c_size_t),("peak_job",ctypes.c_size_t)]
        self.api = ctypes.WinDLL("kernel32",use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p,w.LPCWSTR]
        self.api.CreateJobObjectW.restype = w.HANDLE
        self.api.SetInformationJobObject.argtypes = [w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD]
        self.api.AssignProcessToJobObject.argtypes = [w.HANDLE,w.HANDLE]
        self.api.TerminateJobObject.argtypes = [w.HANDLE,w.UINT]
        self.api.CloseHandle.argtypes = [w.HANDLE]
        self.handle = self.api.CreateJobObjectW(None,None)
        limits = Extended(); limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (not self.handle or not self.api.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits))
                or not self.api.AssignProcessToJobObject(self.handle,w.HANDLE(int(process._handle)))):
            error = ctypes.get_last_error()
            self.close()
            process.kill()
            raise OSError(error,"Cannot contain benchmark worker in a Windows job")

    def terminate(self):
        if not self.api.TerminateJobObject(self.handle,124):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_case(root, name, docs, metadata, reports=None):
    folder = root / "cases" / name
    folder.mkdir(parents=True, exist_ok=False)
    docs = deepcopy(docs)
    docs["run"]["references"] = {k + "_file": k + ".yaml" for k in ("chemistry", "program", "solids")}
    docs["run"]["simulation"]["system_name"] = "SolverStudy"
    docs["run"]["outputs"].update(directory="output", artifacts_directory="output/artifacts",
                                      requested_reports=REPORTS if reports is None else reports,
                                      requested_plots=[], solver_incidence_matrix=False)
    for key, document in docs.items():
        (folder / (key + ".yaml")).write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return dict(id=name, documents_sha256=digest(docs), **metadata)


def program_features(program):
    """Features for reproducible selection across the authored programs."""
    scalar_values, durations = [], []
    def visit(value, key=""):
        if isinstance(value, dict):
            for k, v in value.items():
                visit(v, k)
        elif isinstance(value, list):
            for v in value:
                visit(v, key)
        elif isinstance(value, (int, float)):
            if key == "duration_s": durations.append(float(value))
            elif key in ("initial", "temperature", "flow"): scalar_values.append(float(value))
    visit(program)
    return [len(durations), min(durations, default=0.), max(durations, default=0.),
            sum(durations), *sorted(scalar_values)[:3], *([0.] * max(0, 3-len(scalar_values)))][:7]


def prepare(root):
    from packed_bed.batch import load_batch_spec, expand_batch_cases
    from packed_bed.compiled.bundle import engine_fingerprint
    from tools.performance_cases import cases as chemistry_cases
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for family, source in (("small", "ml_batch_case"), ("large", "ml_batch_case_large")):
        source_root = REPO / "packed_bed/examples" / source
        expanded = expand_batch_cases(load_batch_spec(source_root / "batch.yaml"))
        old = list(csv.DictReader((source_root / "output/summary.csv").open(encoding="utf-8-sig")))
        slow = {r["axis_program"] for r in sorted(old, key=lambda r:float(r.get("runtime_s") or 0), reverse=True)[:4]}
        rng = random.Random(SEED)
        selected = set(rng.sample(range(len(expanded)), 12))
        features = np.asarray([program_features(c.program) for c in expanded])
        for col in range(features.shape[1]):
            selected.update((int(np.argmin(features[:, col])), int(np.argmax(features[:, col]))))
        selected.update(i for i,c in enumerate(expanded) if c.selections["program"] in slow)
        remaining = sorted(set(range(len(expanded))) - selected)
        holdout = set(rng.sample(remaining, 24))
        for i, case in enumerate(expanded):
            docs = {key: deepcopy(getattr(case, key)) for key in ("run", "chemistry", "program", "solids")}
            tags = ["full", family]
            if i in selected: tags.append("screen")
            if i in holdout: tags.append("holdout")
            records.append(save_case(root, family + "__" + case.case_id, docs,
                                     dict(source=str(source_root / "batch.yaml"), selection=case.selections,
                                          tags=tags, features=features[i].tolist())))
        # Independent mesh/transport checks use three different authored programs.
        for i in (0, len(expanded)//2, len(expanded)-1):
            case = expanded[i]
            for cells in (3, 60, 150):
                docs = {key: deepcopy(getattr(case, key)) for key in ("run", "chemistry", "program", "solids")}
                docs["run"]["model"]["axial_cells"] = cells
                records.append(save_case(root, f"mesh_{family}_{i}_{cells}", docs,
                                         dict(tags=["supplement", "mesh"], source=str(source_root / "batch.yaml"), selection=case.selections)))
    for name, (docs, meta) in chemistry_cases().items():
        for cells in (3, 30, 100):
            changed = deepcopy(docs)
            changed["run"]["model"]["axial_cells"] = cells
            # Preserve the existing small comparison case's physical scenario.
            records.append(save_case(root, f"chem_{name}_{cells}", changed,
                                     dict(tags=["supplement", "chemistry"], source="tools/performance_cases.py", **meta)))
    # Built-in examples add many cycles and an alternative reforming mechanism.
    default = REPO / "packed_bed/examples/default_case"
    docs = {key: yaml.safe_load((default/(key+".yaml")).read_text()) for key in ("run", "chemistry", "program", "solids")}
    for scheme in ("upwind1", "weno3", "weno5"):
        changed = deepcopy(docs)
        changed["run"]["simulation"].update(mass_scheme=scheme, heat_scheme=scheme)
        records.append(save_case(root, "default_" + scheme, changed, dict(tags=["supplement", "scheme"], source=str(default))))
    configuration = dict(version=1, seed=SEED, engine_sha256=engine_fingerprint(),
                         git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                         python=sys.version, platform=platform.platform(), processor=platform.processor(),
                         logical_cpus=os.cpu_count(), profiles=PROFILES, cases=records,
                         notes="Full authored horizons/reporting grids; consistent added balance reports; plots disabled.")
    write_json(root / "study.json", configuration)
    print(json.dumps({"cases":len(records), "tags":{tag:sum(tag in c['tags'] for c in records) for tag in sorted({t for c in records for t in c['tags']})}}))


def dataset_metrics(path):
    import xarray as xr
    with xr.open_dataset(path, engine="scipy") as dataset:
        finite = all(np.isfinite(v.values).all() for v in dataset.data_vars.values())
        result = dict(finite=finite, time_end=float(dataset.time[-1]), reports=len(dataset.time),
                      variables=list(dataset.data_vars))
        if not finite:
            return result
        for name in ("gas_mole_fraction", "solid_mole_fraction", "temperature", "pressure"):
            if name in dataset:
                result[name] = {"min":float(dataset[name].min()), "max":float(dataset[name].max())}
        for phase, dimension in (("gas", "gas_species"), ("solid", "solid_species")):
            name = phase + "_mole_fraction"
            if name in dataset:
                result[phase + "_fraction_sum_error"] = float(abs(dataset[name].sum(dimension)-1.).max())
        for quantity in ("mass", "heat"):
            name = quantity + "_balance_error"
            if name in dataset:
                error = dataset[name].values
                scales = [float(abs(dataset[quantity + suffix]).max()) for suffix in ("_bed_total", "_in_total", "_out_total")]
                scale = max(*scales, 1e-30)
                result[quantity + "_balance"] = dict(max_absolute=float(abs(error).max()),
                    max_normalized=float(abs(error).max()/scale), final_normalized=float(error[-1]/scale), scale=scale)
        if "outlet_species_flow" in dataset:
            from scipy.integrate import trapezoid
            result["outlet_species_integrals"] = trapezoid(dataset.outlet_species_flow.values, dataset.time.values, axis=0).tolist()
        return result


def augment(root):
    """Add reproducible cycles for every built-in family and structural variants."""
    from packed_bed.kinetics import FAMILY_REGISTRY
    config = json.loads((root / "study.json").read_text())
    existing = {c["id"] for c in config["cases"]}
    template = REPO / "packed_bed/examples/default_case"
    base = {key:yaml.safe_load((template/(key+".yaml")).read_text()) for key in ("run","program","chemistry","solids")}
    added = []
    for name, family in FAMILY_REGISTRY.items():
        for cells, length, radius in ((5,.4,.0175),(30,2.,.1),(120,6.,.5)):
            case_id = f"cycle_{name}_{cells}"
            if case_id in existing: continue
            docs = deepcopy(base)
            gases = sorted(set(family.required_gas_species) | {"N2"})
            support = "Al2O3" if name.startswith(("iron","copper")) else "CaAl2O4"
            solids = sorted(set(family.required_solid_species) | {support})
            fuel = {g:0. for g in gases}
            fuel.update({g:v for g,v in {"H2":.25,"H2O":.25,"CO":.05,"CO2":.05,"CH4":.1}.items() if g in gases})
            fuel["N2"] = 1.-sum(fuel.values())
            purge = {g:float(g=="N2") for g in gases}
            oxidation = {g:0. for g in gases}
            oxidation.update({"N2":.79,"O2":.21} if "O2" in gases else {"N2":.5,"H2O":.5})
            docs["chemistry"] = dict(gas_species=gases,reaction_families=[name],reaction_ids=[r.id for r in family.reactions])
            docs["solids"] = dict(solid_species=solids,initial_profile=dict(basis="bed",zones=[dict(x_start_m=0.,x_end_m=length,
                    e_b=.5,e_p=.4,d_p=.0012 if cells==5 else .008,values={s:4000. if s==support else 300. for s in solids})]))
            docs["run"]["model"].update(bed_length_m=length,bed_radius_m=radius,axial_cells=cells,ambient_temperature_k=973.15,
                                           heat_transfer_coefficient_w_per_m2_k=0.,gas_voidage_mode="bed_only" if cells==30 else "bed_and_particle")
            docs["run"]["simulation"].update(time_horizon_s=2400.,reporting_interval_s=1.,repeat_program=True,
                    program_mode="separate_channels",interior_flow_mode="reversible",mass_scheme="weno3",heat_scheme="weno3")
            docs["program"] = dict(inlet_flow=dict(basis="ghsv_per_h",initial=1200.),inlet_temperature=dict(initial=973.15),
                outlet_pressure=dict(initial=1e5 if cells==5 else 3e6),inlet_composition=dict(initial=purge,steps=[
                    dict(kind="hold",duration_s=20.),dict(kind="ramp",duration_s=2.,target=fuel),dict(kind="hold",duration_s=150.),
                    dict(kind="ramp",duration_s=2.,target=purge),dict(kind="hold",duration_s=20.),
                    dict(kind="ramp",duration_s=2.,target=oxidation),dict(kind="hold",duration_s=180.),
                    dict(kind="ramp",duration_s=2.,target=purge),dict(kind="hold",duration_s=22.)]))
            added.append(save_case(root,case_id,docs,dict(tags=["supplement","cycles","chemistry"],source="generated from built-in reaction families")))
    original = {k:yaml.safe_load((root/"cases/small__program-prog-0__bed-small"/(k+".yaml")).read_text()) for k in ("run","program","chemistry","solids")}
    for variant in ("reordered","inert","two_zones","heat_loss","long_cycles"):
        case_id = "structure_" + variant
        if case_id in existing: continue
        docs = deepcopy(original)
        if variant == "reordered":
            docs["chemistry"]["gas_species"].reverse()
            docs["chemistry"]["reaction_ids"].reverse()
            docs["solids"]["solid_species"].reverse()
        elif variant == "inert":
            docs["chemistry"]["reaction_families"] = []
            docs["chemistry"]["reaction_ids"] = []
        elif variant == "two_zones":
            zone = docs["solids"]["initial_profile"]["zones"][0]
            second = deepcopy(zone)
            zone["x_end_m"] = second["x_start_m"] = .2
            second["values"]["Ni"] *= .25
            second["values"]["NiO"] = 1000.
            docs["solids"]["initial_profile"]["zones"].append(second)
        elif variant == "heat_loss": docs["run"]["model"]["heat_transfer_coefficient_w_per_m2_k"] = 100.
        elif variant == "long_cycles": docs["run"]["simulation"]["time_horizon_s"] = 48000.
        added.append(save_case(root,case_id,docs,dict(tags=["supplement","structure"],source="small program 0 transformation")))
    config["cases"].extend(added)
    write_json(root/"study.json",config)
    print(f"Added {len(added)} cases; total {len(config['cases'])}")


def register_profiles(root):
    config = json.loads((root/"study.json").read_text())
    for name, settings in PROFILES.items():
        if name in config["profiles"] and config["profiles"][name] != settings:
            raise ValueError(f"Cannot change recorded profile {name}")
        config["profiles"][name] = settings
    write_json(root/"study.json",config)
    print(f"Registered {len(config['profiles'])} profiles")


def diagnostic_cases(root):
    config = json.loads((root/"study.json").read_text())
    existing = {c["id"] for c in config["cases"]}
    for number in (301,987,1228):
        source = f"small__program-prog-{number}__bed-small"
        name = f"diagnostic_reversible_{number}"
        if name in existing: continue
        docs = {k:yaml.safe_load((root/"cases"/source/(k+".yaml")).read_text()) for k in ("run","chemistry","program","solids")}
        docs["run"]["simulation"]["interior_flow_mode"] = "reversible"
        config["cases"].append(save_case(root,name,docs,dict(tags=["diagnostic"],source=source,
                 note="Changes the transport formulation. Does not count as success on the original case.")))
    for case in config["cases"]:
        if re.search(r"^(small|large)__program-(prog-(749|595|301|687)|css-prog-05|test-prog-00)(__bed-small)?$",case["id"]) or case["id"] in (
                "chem_copper_al2o3_san_pio_30","chem_copper_sio2_san_pio_30","cycle_copper_al2o3_san_pio_30","cycle_iron_he_120"):
            if "ablation" not in case["tags"]: case["tags"].append("ablation")
    write_json(root/"study.json",config)
    print("Ablation cases",sum("ablation" in c["tags"] for c in config["cases"]))


def extended_cases(root):
    from packed_bed.kinetics import FAMILY_REGISTRY
    config = json.loads((root/"study.json").read_text())
    existing = {c["id"] for c in config["cases"]}
    for cells in (5,30,100):
        name = f"mixed_four_families_{cells}"
        if name in existing: continue
        source = "cycle_iron_he_30"
        docs = {k:yaml.safe_load((root/"cases"/source/(k+".yaml")).read_text()) for k in ("run","program","chemistry","solids")}
        families = [FAMILY_REGISTRY[f] for f in ("nickel_medrano","reforming_xu_froment","iron_he","copper_sio2_san_pio")]
        gases = sorted({g for f in families for g in f.required_gas_species} | {"N2"})
        solids = sorted({s for f in families for s in f.required_solid_species} | {"Al2O3"})
        docs["chemistry"] = dict(gas_species=gases,reaction_families=[f.name for f in families],reaction_ids=[r.id for f in families for r in f.reactions])
        docs["solids"]["solid_species"] = solids
        docs["solids"]["initial_profile"]["zones"][0]["values"] = {s:4000. if s=="Al2O3" else 100. for s in solids}
        docs["run"]["model"]["axial_cells"] = cells
        config["cases"].append(save_case(root,name,docs,dict(tags=["extended","chemistry"],source=source)))
    for cells in (200,):
        name = f"mesh_large_0_{cells}"
        if name in existing: continue
        source = "large__program-prog-0"
        docs = {k:yaml.safe_load((root/"cases"/source/(k+".yaml")).read_text()) for k in ("run","program","chemistry","solids")}
        docs["run"]["model"]["axial_cells"] = cells
        config["cases"].append(save_case(root,name,docs,dict(tags=["supplement","mesh","boundary_200"],source=source,
                  axial_cells=cells,note="One boundary check at the user-specified maximum; full original horizon retained.")))
    write_json(root/"study.json",config)
    print("Extended cases",sum("extended" in c["tags"] for c in config["cases"]))


def finish_dataset(folder, path):
    path = Path(path).resolve()
    metrics = dataset_metrics(path)
    archive = Path(str(path) + ".gz")
    with path.open("rb") as src, gzip.open(archive, "wb", compresslevel=1) as dst:
        shutil.copyfileobj(src, dst)
    if path.parent != (folder / "output").resolve():
        raise ValueError("Unexpected output dataset location")
    path.unlink()
    return dict(metrics=metrics, dataset=str(archive))


def worker(folder):
    study_root = folder.parents[3]
    configuration = json.loads((study_root/"study.json").read_text())
    if configuration.get("engine_snapshot"):
        sys.path.insert(0,configuration["engine_snapshot"])
    from packed_bed.compiled.bundle import engine_fingerprint
    from packed_bed.config import load_case
    from packed_bed.simulation import run_case
    engine_hash = engine_fingerprint()
    expected = configuration.get("snapshot_engine_sha256",configuration["engine_sha256"])
    result = {"engine_sha256":engine_hash}
    stage = "validation"
    started = perf_counter()
    try:
        if engine_hash != expected:
            raise ValueError("Worker solver source does not match the study snapshot")
        write_json(folder / "worker_started.json", dict(engine_sha256=engine_hash, pid=os.getpid()))
        case = load_case(folder / "run.yaml")
        stage = "simulation"
        run = run_case(case)
        result.update(status="success", runtime_s=run.runtime_s, solver_stats=run.solver_stats)
        stage = "postprocessing"
        result.update(finish_dataset(folder, run.results_path))
        if not result["metrics"]["finite"]:
            result["status"] = "nonfinite"
        elif not np.isclose(result["metrics"]["time_end"],case.run.simulation.time_horizon_s,rtol=0,atol=1e-8):
            result["status"] = "incomplete_horizon"
    except Exception as exc:
        result.update(status="failed", stage=stage, error=str(exc), traceback=traceback.format_exc())
        print(result["traceback"], flush=True)
    result["worker_elapsed_s"] = perf_counter() - started
    write_json(folder / "worker_result.json", result)
    return 0 if result["status"] == "success" else 1


def recover_postprocessing(root):
    repaired = 0
    for result_path in (root / "runs").glob("*/*/*/result.json"):
        record = json.loads(result_path.read_text())
        folder = result_path.parent
        manifest_path = folder / "output/manifest.json"
        if record["status"] != "failed" or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        dataset = folder / "output/results.nc"
        if manifest.get("status") != "success" or not dataset.exists():
            continue
        # Simulation success is authoritative in its manifest; retain the harness error.
        record["postprocessing_error_recovered"] = record.pop("error", "")
        record.update(status="success", runtime_s=manifest["runtime_s"], solver_stats=manifest["solver_stats"])
        record.update(finish_dataset(folder, dataset))
        if not record["metrics"]["finite"]: record["status"] = "nonfinite"
        write_json(result_path, record)
        repaired += 1
    print(f"Recovered postprocessing for {repaired} completed simulations")


def execute(root, case, profile_name, settings, iteration, timeout, concurrency):
    folder = root / "runs" / case["id"] / profile_name / str(iteration)
    signature = digest(dict(case=case["documents_sha256"], solver=settings, iteration=iteration))
    terminal = folder / "result.json"
    if terminal.exists():
        record = json.loads(terminal.read_text())
        if record["signature"] != signature:
            raise ValueError(f"Changed configuration at {folder}; use a new profile or study")
        if record["status"] == "success" and not Path(record["dataset"]).exists():
            raise ValueError(f"Completed dataset missing at {folder}")
        return record, True
    if (folder / "running.json").exists():
        raise RuntimeError(f"Unresolved worker at {folder}; verify its process before resuming")
    folder.mkdir(parents=True, exist_ok=True)
    for key in ("run", "chemistry", "program", "solids"):
        doc = yaml.safe_load((root / "cases" / case["id"] / (key+".yaml")).read_text())
        if key == "run": doc["solver"] = settings
        (folder/(key+".yaml")).write_text(yaml.safe_dump(doc, sort_keys=False))
    env = dict(os.environ, PACKED_BED_COMPILED_CACHE=str(root / "cache"))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = str(max(1, settings["threads"]))
    env.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    start = perf_counter()
    (folder / "worker_started.json").unlink(missing_ok=True)
    with (folder / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "tools.solver_study", "worker", str(folder)],
                                   cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        job = WindowsJob(process) if os.name == "nt" else None
        write_json(folder / "running.json", dict(pid=process.pid, parent_pid=os.getpid()))
        try:
            code = process.wait(timeout=timeout)
            result_file = folder / "worker_result.json"
            record = json.loads(result_file.read_text()) if result_file.exists() else dict(status="crashed", returncode=code)
        except subprocess.TimeoutExpired:
            if job is not None:
                job.terminate()
            else:
                process.kill()
            process.wait()
            record = dict(status="timeout", timeout_s=timeout)
        finally:
            if job is not None: job.close()
    started_path = folder / "worker_started.json"
    if started_path.exists():
        record.setdefault("engine_sha256", json.loads(started_path.read_text())["engine_sha256"])
    record.update(case=case["id"], profile=profile_name, iteration=iteration, signature=signature,
                  elapsed_s=perf_counter()-start, concurrency=concurrency, solver=settings)
    if record["status"] != "success":
        record["log_tail"] = (folder / "worker.log").read_text(errors="replace")[-3000:]
    write_json(terminal, record)
    (folder / "running.json").unlink(missing_ok=True)
    return record, False


def run(args):
    config = json.loads((args.root / "study.json").read_text())
    if config.get("engine_snapshot"):
        sys.path.insert(0,config["engine_snapshot"])
    from packed_bed.compiled.bundle import engine_fingerprint
    if config.get("snapshot_engine_sha256",config["engine_sha256"]) != engine_fingerprint():
        raise ValueError("Solver source changed since study preparation; create a new study")
    selected = [c for c in config["cases"] if "out_of_scope" not in c["tags"]
                and (not args.tag or args.tag in c["tags"])
                and (not args.exclude_tag or args.exclude_tag not in c["tags"]) and re.search(args.cases, c["id"])]
    if args.limit: selected = selected[:args.limit]
    jobs = [(c,p) for c in selected for p in args.profiles]
    random.Random(SEED + args.iteration).shuffle(jobs)
    print(f"{len(jobs)} experiments; {len(selected)} cases; {args.workers} workers", flush=True)
    completed = failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute,args.root,c,p,config["profiles"][p],args.iteration,args.timeout,args.workers) for c,p in jobs]
        for future in as_completed(futures):
            record, reused = future.result()
            completed += 1
            failures += record["status"] != "success"
            print(json.dumps(dict(completed=completed,total=len(jobs),failures=failures,case=record["case"],
                                  profile=record["profile"],status=record["status"],reused=reused,
                                  elapsed_s=round(record["elapsed_s"],2))), flush=True)


def repair_provenance(args):
    """Preserve and rerun terminal measurements made during concurrent edits."""
    config = json.loads((args.root/"study.json").read_text())
    cutoff = datetime.fromisoformat(config["concurrent_edits_first_seen"]).timestamp()
    cases = {c["id"]:c for c in config["cases"]}
    jobs = []
    for path in (args.root/"runs").glob("*/*/*/result.json"):
        record = json.loads(path.read_text())
        if "out_of_scope" in cases[record["case"]]["tags"]: continue
        if "engine_sha256" in record: continue
        if path.stat().st_mtime < cutoff:
            record["engine_sha256"] = config["engine_sha256"]
            record["provenance"] = "Completed before concurrent source edits; initial revision recorded by study"
            write_json(path,record)
            continue
        if (path.parent/"running.json").exists(): continue
        if record.get("dataset"):
            source = Path(record["dataset"]).resolve()
            destination = source.with_name("results.unpinned.nc.gz")
            if not source.is_relative_to(args.root) or not destination.is_relative_to(args.root):
                raise ValueError("Provenance archive outside study")
            source.replace(destination)
            record["dataset"] = str(destination)
        record["provenance"] = "Concurrent source edit window; excluded pending pinned rerun"
        write_json(path.with_name("result.unpinned.json"),record)
        path.unlink()
        (path.parent/"worker_result.json").unlink(missing_ok=True)
        jobs.append(record)
    # Resume earlier archived repairs if a controller was interrupted.
    keys={(r["case"],r["profile"],r["iteration"]) for r in jobs}
    for path in (args.root/"runs").glob("*/*/*/result.unpinned.json"):
        if not path.with_name("result.json").exists() and not path.with_name("running.json").exists():
            record=json.loads(path.read_text());key=(record["case"],record["profile"],record["iteration"])
            if "out_of_scope" not in cases[record["case"]]["tags"] and key not in keys: jobs.append(record);keys.add(key)
    print(f"Rerunning {len(jobs)} measurements with pinned source",flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(execute,args.root,cases[r["case"]],r["profile"],r["solver"],r["iteration"],args.timeout,args.workers) for r in jobs]
        for i,future in enumerate(as_completed(futures),1):
            record,_=future.result()
            print(json.dumps(dict(completed=i,total=len(jobs),case=record["case"],profile=record["profile"],status=record["status"])),flush=True)


def compare_datasets(expected, actual):
    """Compare aligned finite arrays without hiding trace values or output shifts."""
    if set(expected.variables) != set(actual.variables):
        raise ValueError("Dataset variables differ")
    for name in expected.coords:
        if np.issubdtype(expected[name].dtype, np.number):
            np.testing.assert_allclose(expected[name],actual[name],rtol=1e-12,atol=1e-12)
        else:
            np.testing.assert_array_equal(expected[name],actual[name])
    errors = {}
    for name in expected.data_vars:
        a,b = expected[name].values,actual[name].values
        if expected[name].dims != actual[name].dims or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Nonfinite or mismatched {name}")
        delta = abs(a-b)
        errors[name] = dict(max_abs=float(delta.max()), rms=float(np.sqrt(np.mean(delta**2))), reference_scale=float(abs(a).max()))
    if "outlet_species_flow" in expected:
        from scipy.integrate import trapezoid
        times = expected.time.values
        a,b = expected.outlet_species_flow.values,actual.outlet_species_flow.values
        windows = [(0,len(times)-1)] + list(zip(np.linspace(0,len(times)-1,5,dtype=int)[:-1],np.linspace(0,len(times)-1,5,dtype=int)[1:]))
        absolute, normalized, mixed_scores, stream_errors, species_errors = [], [], [], [], []
        for start,end in windows:
            ref = trapezoid(a[start:end+1],times[start:end+1],axis=0)
            value = trapezoid(b[start:end+1],times[start:end+1],axis=0)
            throughput = trapezoid(abs(a[start:end+1]),times[start:end+1],axis=0)
            delta = abs(ref-value)
            absolute.append(float(delta.max()))
            normalized.append(float(np.max(delta/np.maximum(throughput,1e-8))))
            stream_amount = float(throughput.sum())
            # A vanishing species needs an absolute allowance tied to the stream,
            # rather than an arbitrary molar floor that depends on reactor size.
            allowance = .001*throughput + 1e-6*stream_amount + 1e-12
            mixed_scores.append(float(np.max(delta/allowance)))
            stream_errors.append(float(delta.max()/max(stream_amount,1e-12)))
            species_errors.append(dict(reference=ref.tolist(),actual=value.tolist(),
                                       absolute_throughput=throughput.tolist(),absolute_error=delta.tolist()))
        errors["net_outlet_amount"] = dict(max_abs=max(absolute),normalized_error=max(normalized),
                                          mixed_tolerance_score=max(mixed_scores),max_stream_fraction_error=max(stream_errors),
                                          species=[str(s) for s in expected.gas_species.values],window_species=species_errors,
                                          total_and_quarter_max_absolute=absolute,total_and_quarter_max_relative=normalized,
                                          note="Total and four reporting-grid windows; gate is 0.1% of species throughput plus 1 ppm of total stream throughput plus 1e-12 mol. Original species-relative metric retained.")
    return errors


def assess_errors(errors, metrics, temperature_limit=.1, fraction_limit=.001):
    """Explicit provisional scientific gates; raw errors remain in the record."""
    scores = {}
    for name, value in errors.items():
        if name == "net_outlet_amount":
            scores[name] = value["mixed_tolerance_score"]
            continue
        scale = value["reference_scale"]
        if "temperature" in name: allowance = temperature_limit
        elif "mole_fraction" in name or "composition" in name: allowance = fraction_limit
        elif "pressure" in name: allowance = 1. + .001*scale
        elif name in ("gas_flux", "outlet_species_flow", "outlet_flow", "inlet_flow"): allowance = 1e-7 + .001*scale
        else: continue
        scores[name] = value["max_abs"]/allowance
    checks = {name:value <= 1. for name,value in scores.items()}
    for kind in ("mass","heat"):
        value = metrics.get(kind+"_balance")
        if value: checks[kind+"_balance"] = value["max_normalized"] <= 1e-4
    return dict(passed=bool(checks) and all(checks.values()),checks=checks,scores=scores)


def compare(args):
    import xarray as xr
    records = [json.loads(p.read_text()) for p in (args.root / "runs").glob("*/*/*/result.json")]
    config = json.loads((args.root / "study.json").read_text())
    excluded = {c["id"] for c in config["cases"] if "out_of_scope" in c["tags"]}
    records = [r for r in records if r["case"] not in excluded]
    references = {r["case"]:r for r in records if r["profile"]==args.reference and r["status"]=="success" and r["iteration"]==0}
    destination = args.root / ("comparisons_" + args.reference + ".json")
    previous = json.loads(destination.read_text()) if destination.exists() else []
    cache = {(r["case"],r["profile"],r["iteration"]):r for r in previous}
    groups = {}
    for record in sorted(records,key=lambda r:(r["case"],r["profile"],r["iteration"])):
        if record["status"] == "success" and record["case"] in references and record["profile"] != args.reference:
            groups.setdefault(record["case"],[]).append(record)

    def compare_case(case, candidates):
        reference = references[case]
        comparisons = []
        loaded_reference = None
        try:
            for record in candidates:
                key = (record["case"],record["profile"],record["iteration"])
                signature = digest([reference["signature"],reference.get("engine_sha256"),record["signature"],record.get("engine_sha256"),4])
                cached = cache.get(key)
                if cached and cached.get("comparison_signature") == signature and "error" not in cached:
                    comparisons.append(cached)
                    continue
                try:
                    # Each case owns its arrays. Decode its reference once, then release it.
                    if loaded_reference is None:
                        with gzip.open(reference["dataset"],"rb") as stream:
                            loaded_reference = xr.load_dataset(io.BytesIO(stream.read()),engine="scipy")
                    with gzip.open(record["dataset"],"rb") as stream:
                        b = xr.load_dataset(io.BytesIO(stream.read()),engine="scipy")
                    with b:
                        errors = compare_datasets(loaded_reference,b)
                    comparison = dict(case=record["case"],profile=record["profile"],iteration=record["iteration"],reference=args.reference,errors=errors,
                                      assessment=assess_errors(errors,record["metrics"]),
                                      strict_assessment=assess_errors(errors,record["metrics"],.01,.0001))
                except Exception as exc:
                    comparison = dict(case=record["case"],profile=record["profile"],iteration=record["iteration"],reference=args.reference,error=str(exc))
                comparison["comparison_signature"] = signature
                comparisons.append(comparison)
        finally:
            if loaded_reference is not None:
                loaded_reference.close()
        return comparisons

    comparisons = []
    workers = max(1,getattr(args,"workers",4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(compare_case,case,group) for case,group in groups.items()]
        checkpoint = 0
        for future in as_completed(futures):
            comparisons.extend(future.result())
            if len(comparisons) - checkpoint >= 100:
                print(f"Compared {len(comparisons)} results",flush=True)
                checkpoint = len(comparisons)
    comparisons.sort(key=lambda r:(r["case"],r["profile"],r["iteration"]))
    write_json(destination, comparisons)
    print(f"Saved {len(comparisons)} comparisons across {len(references)} reference cases ({workers} comparison workers)")


def summarize(root):
    records = [json.loads(p.read_text()) for p in (root/"runs").glob("*/*/*/result.json")]
    config = json.loads((root / "study.json").read_text())
    excluded = {c["id"] for c in config["cases"] if "out_of_scope" in c["tags"]}
    records = [r for r in records if r["case"] not in excluded]
    output = {}
    comparison_path = root / "comparisons_reference.json"
    comparisons = json.loads(comparison_path.read_text()) if comparison_path.exists() else []
    for profile in sorted({r["profile"] for r in records}):
        rows = [r for r in records if r["profile"]==profile]
        good = [r for r in rows if r["status"]=="success"]
        times = [r["solver_stats"]["integration_s"] for r in good]
        evaluated = [c for c in comparisons if c['profile']==profile and c['case'] not in excluded]
        output[profile] = dict(completed=len(rows),success=len(good),compared=len(evaluated),
                               accuracy_pass=sum(c.get('assessment',{}).get('passed',False) for c in evaluated),
                               failures=[dict(case=r["case"],status=r["status"],error=r.get("error",r.get("log_tail",""))[-400:]) for r in rows if r["status"]!="success"],
                               integration_median_s=float(np.median(times)) if times else None,
                               integration_p95_s=float(np.percentile(times,95)) if times else None,
                               max_mass_balance=max((r["metrics"].get("mass_balance",{}).get("max_normalized",0) for r in good),default=None),
                               max_heat_balance=max((r["metrics"].get("heat_balance",{}).get("max_normalized",0) for r in good),default=None))
    write_json(root/"summary.json",output)
    print(json.dumps(output,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("prepare"); p.add_argument("root",type=Path)
    p = sub.add_parser("worker"); p.add_argument("root",type=Path)
    p = sub.add_parser("recover-postprocessing"); p.add_argument("root",type=Path)
    p = sub.add_parser("augment"); p.add_argument("root",type=Path)
    p = sub.add_parser("summarize"); p.add_argument("root",type=Path)
    p = sub.add_parser("register-profiles"); p.add_argument("root",type=Path)
    p = sub.add_parser("diagnostic-cases"); p.add_argument("root",type=Path)
    p = sub.add_parser("extended-cases"); p.add_argument("root",type=Path)
    p = sub.add_parser("repair-provenance"); p.add_argument("root",type=Path)
    p.add_argument("--workers",type=int,default=4);p.add_argument("--timeout",type=float,default=300)
    p = sub.add_parser("run"); p.add_argument("root",type=Path)
    p.add_argument("--profiles",nargs="+",required=True)
    p.add_argument("--tag"); p.add_argument("--cases",default=".")
    p.add_argument("--exclude-tag")
    p.add_argument("--limit",type=int); p.add_argument("--workers",type=int,default=6)
    p.add_argument("--timeout",type=float,default=240); p.add_argument("--iteration",type=int,default=0)
    p = sub.add_parser("compare"); p.add_argument("root",type=Path); p.add_argument("--reference",default="reference")
    p.add_argument("--workers",type=int,default=4)
    args = parser.parse_args(); args.root = args.root.resolve()
    if args.command == "prepare": prepare(args.root)
    elif args.command == "worker": return worker(args.root)
    elif args.command == "recover-postprocessing": recover_postprocessing(args.root)
    elif args.command == "augment": augment(args.root)
    elif args.command == "summarize": summarize(args.root)
    elif args.command == "register-profiles": register_profiles(args.root)
    elif args.command == "diagnostic-cases": diagnostic_cases(args.root)
    elif args.command == "extended-cases": extended_cases(args.root)
    elif args.command == "repair-provenance": repair_provenance(args)
    elif args.command == "run": run(args)
    elif args.command == "compare": compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
