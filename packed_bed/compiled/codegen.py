"""Share exact residual/Jacobian expression graphs across equivalent cells."""

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix

from .graph import Graph
from .sharing import shared_inputs


def emit_model(
    graph,
    keep,
    residuals,
    jacobian,
    reconstruction,
    sparsity,
    locations,
    owners,
    cells,
    *,
    vectorize=False,
    vector_exponentials=False,
):
    """Share cell functions and expressions crossing cell boundaries.

    Output buffers must be separate from the state and derivative inputs.
    """
    row_cells = [
        min(locations[keep[col]][2], cells - 1)
        if locations[keep[col]][2] is not None
        else None
        for col in owners
    ]
    plans, helpers, shared_metadata = {}, [], {}
    for name, selected, selected_cells in (
        ("evaluate", residuals, row_cells),
        ("jacobian", jacobian, [row_cells[row] for row in sparsity.indices]),
    ):
        plan = shared_inputs(graph, selected, selected_cells, keep, locations)
        if plan is None:
            continue
        plans[name] = plan
        helper, metadata = _emit_model(
            graph,
            plan.extended_keep,
            plan.nodes,
            [],
            [],
            csc_matrix((len(plan.nodes), len(plan.extended_keep))),
            plan.locations,
            np.arange(len(keep), len(plan.extended_keep)),
            cells,
            # Jacobian intermediates can contain derivative branch operators.
            vectorize=vectorize and name == "evaluate",
        )
        helper = helper.replace('extern "C" __declspec(dllexport)', "static")
        helper = re.sub(r"^#include.*\n", "", helper, flags=re.MULTILINE)
        helpers.append(
            f"namespace shared_{name} {{\n"
            "using ::fabs;using ::sqrt;using ::exp;using ::log;using ::log10;"
            "using ::pow;using ::fmin;using ::fmax;\n" + helper + "\n}\n"
        )
        shared_metadata[name] = metadata
    source, metadata = _emit_model(
        graph,
        keep,
        residuals,
        jacobian,
        reconstruction,
        sparsity,
        locations,
        owners,
        cells,
        vectorize=vectorize,
        plans=plans,
        helpers="".join(helpers),
    )
    metadata["shared_expressions"] = {
        name: len(plan.nodes) for name, plan in plans.items()
    }
    metadata["shared_kernel_metadata"] = shared_metadata
    if shared_metadata.get("evaluate", {}).get("residual_lanes") == 4:
        source = "#include <immintrin.h>\n" + source
        # The helper can use AVX2 even if the remaining cell functions cannot.
        metadata["residual_lanes"] = 4
    metadata["vector_exponentials"] = "scalar libm"
    if vector_exponentials and metadata["residual_lanes"] == 4:
        # Only the four-lane exp changes; scalar rows, Jacobian and report math
        # keep their existing functions. The caller verifies AVX2 and FMA3.
        source = source.replace(
            "CELL_UNARY(exp)",
            "static __forceinline CellPacket exp(CellPacket a) {"
            "return ::packed_bed_sleef::Sleef_expd4_u10avx2(a.value);}",
        )
        header = Path(__file__).with_name("sleef_exp.hpp").read_text(encoding="utf-8")
        source = header + "\n" + source
        metadata["vector_exponentials"] = "SLEEF 3.9.0 AVX2/FMA3 exp_u10"
    return source, metadata


def _emit_model(
    graph,
    keep,
    residuals,
    jacobian,
    reconstruction,
    sparsity,
    locations,
    owners,
    cells,
    *,
    vectorize=False,
    plans=None,
    helpers="",
):
    mapping = {old: new for new, old in enumerate(keep)}
    original_graph, original_mapping, original_locations = graph, mapping, locations
    roots = {"evaluate": residuals, "jacobian": jacobian, "reconstruct": reconstruction}
    row_cells = [
        min(locations[keep[col]][2], cells - 1)
        if locations[keep[col]][2] is not None
        else None
        for col in owners
    ]
    jac_rows = sparsity.indices
    jac_cols = np.repeat(np.arange(len(keep)), np.diff(sparsity.indptr))

    def variable_key(old, cell):
        name, component, position = locations[old]
        return (
            name,
            component,
            position - cell if position is not None and cell is not None else position,
        )

    @lru_cache(None)
    def time_only(node):
        op, *args = graph.nodes[node]
        if op in ("var", "dot", "cj"):
            return False
        if op in ("const", "time"):
            return True
        return all(time_only(arg) for arg in args)

    active = set()

    def visit(node):
        if node in active:
            return
        active.add(node)
        op, *args = graph.nodes[node]
        if op not in ("const", "var", "dot", "time", "cj"):
            for arg in args:
                visit(arg)

    for selected in roots.values():
        for node in selected:
            visit(node)
    time_nodes = {
        node
        for node in active
        if time_only(node) and graph.nodes[node][0] not in ("const", "time")
    }
    frontier = set()
    for node in active:
        op, *args = graph.nodes[node]
        if not time_only(node) and op not in ("var", "dot", "cj"):
            frontier.update(arg for arg in args if arg in time_nodes)
    frontier.update(
        node for selected in roots.values() for node in selected if node in time_nodes
    )
    frontier = sorted(frontier)
    positions = {node: i for i, node in enumerate(frontier)}

    def use_cached_time(source):
        source = re.sub(
            r"^const double z(\d+) = .*;\n",
            lambda match: "" if int(match[1]) in time_nodes else match[0],
            source,
            flags=re.MULTILINE,
        )
        return re.sub(
            r"\bz(\d+)\b",
            lambda match: (
                f"time_values[{positions[int(match[1])]}]"
                if int(match[1]) in positions
                else match[0]
            ),
            source,
        )

    source = "#include <cmath>\n#include <cstring>\n"
    source += (
        "static thread_local bool cached_valid=false;\n"
        "static thread_local double cached_t=0;\n"
        f"static thread_local double time_values[{max(1, len(frontier))}];\n"
    )
    source += graph.emit("update_time", frontier, mapping).replace(
        'extern "C" __declspec(dllexport)', "static"
    )
    source += (
        "static void ensure_time(double t) {\n"
        "if(!cached_valid || t!=cached_t) {\n"
        "update_time(t,nullptr,nullptr,0,time_values);cached_t=t;cached_valid=true;\n"
        "}}\n"
    )
    source += helpers
    canonical = {}
    constants = defaultdict(dict)
    fixed_constants = {0.0, 1.0, -1.0, 2.0, 0.5}

    @lru_cache(None)
    def normalized(node, cell):
        op, *args = graph.nodes[node]
        if node in positions:
            # A cached expression belongs to this exact time function. Treating
            # its constants as cell parameters could incorrectly share a cache
            # slot between spatially different time-dependent coefficients.
            key = ("cached_time", node)
        elif op in ("var", "dot"):
            key = (op, variable_key(args[0], cell))
        elif op == "const" and args[0] not in fixed_constants:
            bindings = constants[cell]
            if args[0] not in bindings:
                bindings[args[0]] = len(bindings)
            key = ("parameter", bindings[args[0]])
        elif op in ("const", "time", "cj"):
            key = graph.nodes[node]
        else:
            key = (op, *(normalized(arg, cell) for arg in args))
        if key not in canonical:
            canonical[key] = len(canonical)
        return canonical[key]

    metadata = {
        "kernel_layout": "shared cells",
        "cached_time_expressions": len(frontier),
        "vectorized_residual_cells": 0,
        "residual_lanes": 1,
    }
    for name, selected in roots.items():
        plan = plans.get(name) if plans else None
        graph = plan.graph if plan else original_graph
        mapping = plan.mapping if plan else original_mapping
        locations = plan.locations if plan else original_locations
        if name == "reconstruct":
            code = use_cached_time(graph.emit(name, selected, mapping))
            source += code.replace("double* out) {", "double* out) {\nensure_time(t);")
            continue
        normalized.cache_clear()
        constants.clear()
        grouped = defaultdict(list)
        for output, node in enumerate(selected):
            row = output if name == "evaluate" else int(jac_rows[output])
            cell = row_cells[row]
            row_key = variable_key(keep[owners[row]], cell)
            key = (
                (row_key,)
                if name == "evaluate"
                else (row_key, variable_key(keep[jac_cols[output]], cell))
            )
            grouped[cell].append((key, output, node))
        templates = defaultdict(list)
        for cell, entries in grouped.items():
            entries.sort(key=lambda entry: repr(entry[0]))
            signature = tuple(normalized(node, cell) for _, _, node in entries)
            templates[signature].append((cell, entries))
        calls = []
        metadata[name + "_templates"] = len(templates)
        for number, groups in enumerate(templates.values()):
            cell, entries = groups[0]
            nodes = [node for _, _, node in entries]
            dependencies = sorted(
                set().union(*(graph.dependencies(node) for node in nodes))
            )
            keys = [variable_key(old, cell) for old in dependencies]
            function = f"{name}_cell{number}"
            variable_map = {old: f"ix[{i}]" for i, old in enumerate(dependencies)}
            emitter = Graph()
            emitter.nodes = graph.nodes.copy()
            parameter_start = max(locations) + 1
            for node, expression in enumerate(graph.nodes):
                if expression[0] == "const" and expression[1] in constants[cell]:
                    i = constants[cell][expression[1]]
                    emitter.nodes[node] = ("var", parameter_start + i)
                    variable_map[parameter_start + i] = f"PARAM_{i}"
            code = emitter.emit(function, nodes, variable_map)
            code = re.sub(r"y\[PARAM_(\d+)\]", r"par[\1]", code)
            code = (
                use_cached_time(code)
                .replace(
                    'extern "C" __declspec(dllexport)', "static __declspec(noinline)"
                )
                .replace(
                    "double* out) {", "const int* ix,const double* par,double* out) {"
                )
            )
            # Each cell writes its assigned outputs directly. The offset table
            # retains arbitrary equation/CSC order without a temporary copy.
            scalar_code = code.replace(
                "double* out)", "double* __restrict out,const int* offsets)"
            )
            source += re.sub(r"out\[(\d+)\] =", r"out[offsets[\1]] =", scalar_code)
            metadata["direct_scalar_output_functions"] = (
                metadata.get("direct_scalar_output_functions", 0) + 1
            )
            vectorized = vectorize and name == "evaluate" and len(groups) >= 4
            if vectorized:
                # Keep the same graph, constants and scalar operation order in
                # every lane. Only independent cells are evaluated together.
                vector_code = code.replace(function + "(", function + "_vec(")
                vector_code = vector_code.replace(
                    "double* out)", "double* __restrict out,const int* offsets)"
                )
                input_stride = max(1, len(dependencies))
                parameter_stride = max(1, len(constants[cell]))
                vector_code = re.sub(
                    r"\b(yp|y)\[ix\[(\d+)\]\]",
                    rf"gather_cells(\1,ix+\2,{input_stride})",
                    vector_code,
                )
                vector_code = re.sub(
                    r"\bpar\[(\d+)\]",
                    rf"cell_parameter(par,\1,{parameter_stride})",
                    vector_code,
                )
                vector_code = re.sub(
                    r"const double z(\d+) =", r"const CellPacket z\1 =", vector_code
                )
                vector_code = re.sub(
                    r"out\[(\d+)\] = ([^\n]+);",
                    rf"scatter_cells(out,offsets+\1,{len(entries)},\2);",
                    vector_code,
                )
                metadata["vectorized_residual_cells"] += len(groups) // 4 * 4
                metadata["residual_lanes"] = 4
                metadata["direct_vector_output_functions"] = (
                    metadata.get("direct_vector_output_functions", 0) + 1
                )
            inputs, outputs, parameters = [], [], []
            for cell, group in groups:
                local = set().union(*(graph.dependencies(node) for _, _, node in group))
                local_map = {variable_key(old, cell): old for old in local}
                if set(local_map) != set(keys):
                    raise RuntimeError(
                        "Cell template has inconsistent variable bindings."
                    )
                inputs.append([mapping[local_map[key]] for key in keys] or [0])
                outputs.append([output for _, output, _ in group])
                parameters.append(list(constants[cell]) or [0.0])
            if any(len(row) != len(parameters[0]) for row in parameters):
                raise RuntimeError("Cell template has inconsistent parameter bindings.")
            if vectorized:
                for j in range(len(parameters[0])):
                    # A parameter may differ between batches. Broadcast it only
                    # when every four-cell batch has equal lane values.
                    same = all(
                        repr(parameters[i + lane][j]) == repr(parameters[i][j])
                        for i in range(0, len(groups) - 3, 4)
                        for lane in range(4)
                    )
                    token = f"cell_parameter(par,{j},{parameter_stride})"
                    if same and token in vector_code:
                        vector_code = vector_code.replace(
                            token, f"CellPacket(par[{j}])"
                        )
                        metadata["broadcast_cell_parameters"] = (
                            metadata.get("broadcast_cell_parameters", 0) + 1
                        )
                source += vector_code

            def table(rows):
                return (
                    "{"
                    + ",".join("{" + ",".join(map(str, row)) + "}" for row in rows)
                    + "}"
                )

            count, width = len(groups), len(entries)
            batch = ""
            if vectorized:
                batch = (
                    f"for(;i+4<={count};i+=4) {{\n"
                    f"{function}_vec(t,y,yp,cj,ix[i],parameters[i],out,offsets[i]);\n"
                    "}\n"
                )
            calls.append(
                "{\n"
                f"static const int ix[{count}][{max(1, len(dependencies))}]={table(inputs)};\n"
                f"static const int offsets[{count}][{width}]={table(outputs)};\n"
                f"static const double parameters[{count}][{len(parameters[0])}]={table(parameters)};\n"
                "int i=0;\n" + batch + f"for(;i<{count};i++) {{\n"
                f"{function}(t,y,yp,cj,ix[i],parameters[i],out,offsets[i]);\n"
                "}}\n"
            )
        preparation = ""
        if plan:
            # Thread-local storage avoids a stack allocation proportional to
            # mesh size. Every entry is refreshed from this callback's inputs.
            preparation = (
                f"static thread_local double extended[{len(plan.extended_keep)}];\n"
                f"memcpy(extended,y,{len(keep)}*sizeof(double));\n"
                f"shared_{name}::evaluate(t,y,yp,cj,extended+{len(keep)});\ny=extended;\n"
            )
        source += (
            f'extern "C" __declspec(dllexport) void {name}(double t,const double* y,'
            "const double* yp,double cj,double* out) {\nensure_time(t);\n"
            + preparation
            + "".join(calls)
            + "}\n"
        )
    if metadata["residual_lanes"] == 4:
        source = (
            Path(__file__).with_name("cell.hpp").read_text(encoding="utf-8")
            + "\n"
            + source
        )
    return source, metadata
