"""Reject development content and record exactly which runtime modules ship."""
import ast
import json
from pathlib import Path


def audit(bundle, analysis):
    bundle = Path(bundle).resolve()
    toc = ast.literal_eval(Path(analysis).read_text(encoding="utf-8"))
    modules = []
    for section in toc:
        if isinstance(section, list):
            for row in section:
                if isinstance(row, tuple) and len(row) == 3 and row[2] in {"PYMODULE", "EXTENSION"}:
                    modules.append(row[0])
    forbidden = {"examples", "tests", "pytest", "_pytest", "Property_Estimation"}
    for name in modules:
        parts = name.replace("\\", ".").split(".")
        if (any(part in forbidden for part in parts) or parts[0] == "tools"
                or name == "packed_bed_ui.release_check"):
            raise ValueError(f"Development module in release: {name}")
    for path in bundle.rglob("*"):
        relative = path.relative_to(bundle)
        if relative.as_posix() in {
            "compiled/compiler/lib/std/crypto/pcurves/tests/p256.zig",
            "compiled/compiler/lib/std/crypto/pcurves/tests/p384.zig",
            "compiled/compiler/lib/std/crypto/pcurves/tests/secp256k1.zig",
        }:
            continue  # Required by Zig's compiler_rt/libzigc, not application tests.
        if relative.parts[0] == "licenses" or "licenses" in relative.parts:
            continue
        if path.is_file() and any(part in forbidden for part in relative.parts):
            raise ValueError(f"Development files in release: {relative}")
        if relative.parts[0] == "plugins":
            raise ValueError("Sample plugin directory in release")
        if relative.parts[0] == "tools" or relative.parts[:2] == ("_internal", "tools"):
            raise ValueError("Repository build tools in release")
        if path.suffix.lower() in {".dll", ".pyd"} and any(
                token in path.name.lower() for token in ("pardiso", "pysuperlu_mt", "umfpack")):
            raise ValueError(f"Removed solver in release: {relative}")
    data = {"policy": "Runtime application and dependency files only; no examples, sample plugins, tests, Property_Estimation or build tools.",
            "python_modules": sorted(set(modules))}
    (bundle / "CONTENTS.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data
