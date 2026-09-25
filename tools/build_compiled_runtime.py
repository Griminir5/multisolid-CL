"""Build the pinned native runtime. Only release/development machines run this.

Needs CMake 3.22+, a C/C++ toolchain with OpenMP, and network access on the
first build. On Windows use MSYS2 UCRT64 GCC with Ninja.
Downloads are verified against compiled_sources.json and may be pre-populated.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
from urllib.request import urlopen
import zipfile

SOURCES = json.loads(Path(__file__).with_name("compiled_sources.json").read_text())


def fetch(name, work):
    spec = SOURCES[name]
    archive = work / (name + (".zip" if spec["url"].endswith(".zip") else ".tar"))
    if not archive.exists():
        with urlopen(spec["url"], timeout=120) as response:
            archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Checksum mismatch: {archive}")
    folder = work / (name + "-source")
    if not folder.exists():
        folder.mkdir()
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    if not (folder / member.filename).resolve().is_relative_to(folder.resolve()):
                        raise ValueError("Unsafe compiler archive path")
                source.extractall(folder)
        else:
            with tarfile.open(archive) as source:
                source.extractall(folder, filter="data")
    return next(path for path in folder.iterdir() if path.is_dir())


def build(work, prefix, jobs):
    sources = {name: fetch(name, work) for name in ("openblas", "suitesparse", "superlu_mt", "sundials")}
    common = [f"-DCMAKE_INSTALL_PREFIX={prefix}", f"-DCMAKE_PREFIX_PATH={prefix}",
              "-DCMAKE_INSTALL_LIBDIR=lib", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
              "-DCMAKE_POSITION_INDEPENDENT_CODE=ON", "-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON",
              "-DCMAKE_INSTALL_RPATH=$ORIGIN", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF"]

    def cmake(name, options):
        directory = work / (name + "-build")
        subprocess.run(["cmake", "-S", str(sources[name]), "-B", str(directory), *common, *options], check=True)
        subprocess.run(["cmake", "--build", str(directory), "--config", "Release", "--parallel", str(jobs)], check=True)
        subprocess.run(["cmake", "--install", str(directory), "--config", "Release"], check=True)

    cmake("openblas", ["-DNOFORTRAN=ON", "-DBUILD_WITHOUT_LAPACK=ON", "-DBUILD_TESTING=OFF",
                      "-DUSE_THREAD=OFF", "-DUSE_LOCKING=ON", "-DTARGET=CORE2"])
    patterns = ("*openblas*.dll.a", "*openblas*.lib") if os.name == "nt" else ("libopenblas.so",)
    libraries = [path for pattern in patterns for path in (prefix / "lib").glob(pattern)]
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one OpenBLAS link library, found {libraries}")
    blas = str(libraries[0])
    cmake("suitesparse", ["-DSUITESPARSE_ENABLE_PROJECTS=suitesparse_config;amd;colamd;btf;klu", "-DKLU_USE_CHOLMOD=OFF",
                         "-DSUITESPARSE_USE_CUDA=OFF", "-DSUITESPARSE_DEMOS=OFF",
                         "-DBUILD_STATIC_LIBS=OFF", f"-DBLAS_LIBRARIES={blas}"])
    cmake("superlu_mt", ["-DPLAT=_OPENMP", "-Denable_single=OFF", "-Denable_complex=OFF",
                         "-Denable_complex16=OFF", "-Denable_examples=OFF", "-Denable_tests=OFF",
                         "-Denable_fortran=OFF", "-Denable_internal_blaslib=OFF", f"-DTPL_BLAS_LIBRARIES={blas}"])
    cmake("sundials", ["-DBUILD_ARKODE=OFF", "-DBUILD_CVODE=OFF", "-DBUILD_CVODES=OFF",
                       "-DBUILD_IDAS=OFF", "-DBUILD_KINSOL=OFF", "-DBUILD_STATIC_LIBS=OFF",
                       "-DEXAMPLES_ENABLE_C=OFF", "-DEXAMPLES_ENABLE_CXX=OFF", "-DEXAMPLES_INSTALL=OFF",
                       "-DSUNDIALS_PRECISION=double", "-DSUNDIALS_INDEX_SIZE=32", "-DENABLE_KLU=ON",
                       f"-DKLU_ROOT={prefix}", "-DENABLE_SUPERLUMT=ON", "-DSUPERLUMT_THREAD_TYPE=OPENMP",
                       f"-DSUPERLUMT_INCLUDE_DIR={prefix / 'include/superlu_mt'}",
                       f"-DSUPERLUMT_LIBRARY_DIR={prefix / 'lib'}"])
    (prefix / "sources.json").write_text(json.dumps(SOURCES, indent=2))
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    work, prefix = args.work.resolve(), args.prefix.resolve()
    work.mkdir(parents=True, exist_ok=True)
    sources = build(work, prefix, args.jobs)
    notices = prefix / "licenses"
    for name, folder in sources.items():
        for path in folder.rglob("*"):
            if path.is_file() and any(token in path.name.lower() for token in ("license", "copying", "copyright")):
                target = notices / name / path.relative_to(folder)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())


if __name__ == "__main__":
    main()
