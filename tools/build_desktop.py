"""Build, validate and package MultiSolid in an isolated Windows/Linux environment."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, help="Build cache and isolated environment (default: build/release-PLATFORM)")
    parser.add_argument("--output", type=Path, help="Distribution folder (default: dist/PLATFORM)")
    parser.add_argument("--daetools", type=Path, help="Use a local DAETools 2.6.0 wheel or extracted release instead of downloading")
    native = parser.add_mutually_exclusive_group()
    native.add_argument("--compiled-bundle", type=Path, help="Use an existing validated native bundle")
    native.add_argument("--native-runtime", type=Path, help="Restage an existing native install prefix with the current engine and rerun its smoke tests")
    parser.add_argument("--graphviz-bundle", type=Path, help="Use an existing staged Graphviz bundle")
    parser.add_argument("--iscc", type=Path, help="Windows: use this Inno Setup compiler instead of preparing the pinned version")
    parser.add_argument("--headless", action="store_true", help="Linux: validate the real X11 backend under xvfb-run")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    system = platform.system().lower()
    if system not in {"linux", "windows"} or platform.machine().lower() not in {"x86_64", "amd64"}:
        parser.error("Build on Linux x86_64 or Windows x64; cross-compilation is not supported.")
    minor = 12 if system == "linux" else 11
    if sys.version_info[:2] != (3, minor):
        parser.error(f"The pinned {system} build uses Python 3.{minor}; rerun with that interpreter.")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.headless and (system != "linux" or not shutil.which("xvfb-run")):
        parser.error("--headless requires Linux with xvfb and xauth installed.")
    if system == "linux" and not args.headless and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        parser.error("No graphical session detected. Install xvfb/xauth and use --headless.")
    build_native = not (args.compiled_bundle or args.native_runtime)
    required = ["cmake", "gcc", "g++"] if build_native else []
    required.append("git")
    if system == "windows" and build_native:
        required.append("ninja")
    if args.headless:
        required.append("xauth")
    if system == "linux":
        required += ["readelf", "ldd", "dpkg-deb"] + ([] if args.graphviz_bundle else ["neato", "dpkg-query"])
    missing = [name for name in required if not shutil.which(name)]
    if missing:
        parser.error("Missing build tools: " + ", ".join(missing) + ". See desktop/BUILDING.md for prerequisites.")
    work = (args.work or ROOT / f"build/release-{system}").resolve()
    output = (args.output or ROOT / f"dist/{system}").resolve()
    work.mkdir(parents=True, exist_ok=True)
    iscc = None
    if system == "windows":
        if args.iscc:
            iscc = args.iscc.resolve()
            if not iscc.is_file():
                parser.error(f"Inno Setup compiler not found: {iscc}")
        else:
            from prepare_inno_setup import prepare
            iscc = prepare(work)
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "MULTISOLID_COMPILED_BUNDLE", "MULTISOLID_GRAPHVIZ"):
        environment.pop(key, None)
    environment.update(PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8",
                       PIP_CACHE_DIR=str(work / "pip-cache"), MPLCONFIGDIR=str(work / "matplotlib"),
                       PYINSTALLER_CONFIG_DIR=str(work / "pyinstaller-cache"))
    if system == "windows":
        environment["CMAKE_GENERATOR"] = "Ninja"
    release_venv = work / "venv"
    python = release_venv / ("Scripts/python.exe" if system == "windows" else "bin/python")
    # A dedicated environment keeps optional developer modules out of the freezer.
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(release_venv)
    def run(*command):
        print("Running:", *map(str, command), flush=True)
        subprocess.run(list(map(str, command)), cwd=ROOT, env=environment, check=True)
    def py(*command):
        run(python, *command)
    lock = ROOT / f"tools/{system}-requirements.lock"
    py("-m", "pip", "--isolated", "--cache-dir", work / "pip-cache", "install", "--require-hashes", "--only-binary=:all:", "-r", lock,
       "-r", ROOT / "tools/help-requirements.txt")
    if system == "linux":
        py("-c", "import sys; sys.path.insert(0, sys.argv[1]); from linux_bundle import check_qt; check_qt()", ROOT / "tools")
    from download_inputs import fetch, find_root, archive_root
    inputs = json.loads((ROOT / "tools/desktop-inputs.json").read_text())
    downloads = work / "downloads"
    source = args.daetools.resolve() if args.daetools else find_root(
        fetch(f"daetools-{system}.zip", inputs[f"daetools_{system}"], downloads), "setup.py")
    if source.suffix == ".whl":
        py("-m", "pip", "install", "--no-deps", "--force-reinstall", source)
    else:
        prepared = work / "daetools"
        # Recreate this generated copy so switching local inputs cannot retain
        # stale extensions from the previous dependency distribution.
        if prepared.exists():
            shutil.rmtree(prepared)
        py(ROOT / "tools/prepare_daetools.py", source, prepared)
        py("-m", "pip", "install", "--no-deps", "--no-build-isolation", "--force-reinstall", prepared)
    wheels = work / "application-wheels"
    if wheels.exists():
        shutil.rmtree(wheels)
    py(ROOT / "tools/build_app_wheels.py", "--output", wheels)
    py("-m", "pip", "install", "--no-deps", "--force-reinstall", *sorted(wheels.glob("*.whl")))
    py("-m", "pip", "check")
    py("-m", "unittest", "discover", "-s", ROOT / "tests", "-v")
    # Fail on incompatible upstream binaries before spending time building solvers.
    py("-c", "from importlib.metadata import version; assert version('daetools') == '2.6.0'; "
       "from packed_bed.simulation import create_linear_solver; "
       "from packed_bed.solver_support import DESKTOP_SOLVERS; "
       "[create_linear_solver(name) for name in DESKTOP_SOLVERS['daetools']]")
    compiled = args.compiled_bundle.resolve() if args.compiled_bundle else work / "vendor/compiled"
    if not args.compiled_bundle:
        runtime = args.native_runtime.resolve() if args.native_runtime else work / "native-prefix"
        from build_compiled_runtime import SOURCES, fetch as fetch_native
        if args.native_runtime:
            if json.loads((runtime / "sources.json").read_text()) != SOURCES:
                raise ValueError("The cached native runtime does not match the pinned sources; rebuild it.")
            print(f"Reusing native libraries: {runtime}; restaging and testing the current engine.", flush=True)
        else:
            py(ROOT / "tools/build_compiled_runtime.py", "--work", work / "native", "--prefix", runtime, "--jobs", args.jobs)
        (work / "native").mkdir(parents=True, exist_ok=True)
        compiler = fetch_native("zig_" + system, work / "native")
        if compiled.exists():
            shutil.rmtree(compiled)
        py(ROOT / "tools/bundle_compiled.py", "--compiler", compiler, "--runtime", runtime, "--destination", compiled)
    # Exercise the installed engine with the shipped reacting example before
    # freezing or publishing any application archive. Keep each run's evidence.
    py(ROOT / "tools/check_example.py", "--output", work / "example-check", "--compiled-bundle", compiled)
    graphviz = args.graphviz_bundle.resolve() if args.graphviz_bundle else work / "vendor/graphviz"
    if not args.graphviz_bundle:
        if graphviz.exists():
            shutil.rmtree(graphviz)
        if system == "linux":
            py(ROOT / "tools/bundle_graphviz.py", "--system", "--destination", graphviz)
        else:
            windows = json.loads((ROOT / "tools/windows-release-inputs.json").read_text())
            binary = find_root(fetch("graphviz-12.2.1.zip", windows["graphviz-12.2.1.zip"], downloads), "neato.exe").parent
            source = archive_root(fetch("graphviz-source.tar.gz", windows["graphviz-source.tar.gz"], downloads))
            dependencies = fetch("graphviz-dependencies.tar.gz", windows["graphviz-dependencies.tar.gz"], downloads)
            dependencies = next(p for p in dependencies.iterdir() if p.is_dir())
            ready = work / "graphviz-ready"
            if ready.exists():
                shutil.rmtree(ready)
            py(ROOT / "tools/prepare_graphviz_windows.py", "--binary", binary, "--source", source,
               "--dependencies", dependencies, "--destination", ready)
            py(ROOT / "tools/bundle_graphviz.py", "--source", ready, "--source-url", windows["graphviz-source.tar.gz"]["url"], "--destination", graphviz)
    py(ROOT / "tools/freeze_desktop.py", "--output", output, "--work", work / "freeze",
       "--compiled-bundle", compiled, "--graphviz-bundle", graphviz)
    import tempfile
    validation = Path(tempfile.mkdtemp(prefix="acceptance-", dir=work)) / "check"
    command = [python, ROOT / "tools/check_desktop_release.py", "--bundle", output / "MultiSolid", "--output", validation]
    if args.headless:
        command = ["xvfb-run", "-a", *command, "--qt-platform", "xcb"]
    run(*command)
    packaging = [ROOT / f"tools/package_{system}.py", "--bundle", output / "MultiSolid", "--output", output / "releases",
                 "--validation", validation / "validation.json"]
    if system == "windows":
        packaging += ["--iscc", iscc]
    py(*packaging)
    print(f"Finished. Downloads: {output / 'releases'}\nValidation: {validation / 'validation.json'}")


if __name__ == "__main__":
    main()
