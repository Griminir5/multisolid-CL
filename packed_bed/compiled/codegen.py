"""Share exact residual/Jacobian expression graphs across equivalent cells."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix

from .graph import Graph, LEAVES, postorder
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
    runtime_programs=False,
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
            linkage="static",
            include_headers=False,
        )
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
        linkage="static" if runtime_programs else "PB_EXPORT",
        function_suffix="_values" if runtime_programs else "",
    )
    metadata["shared_expressions"] = {
        name: len(plan.nodes) for name, plan in plans.items()
    }
    metadata["shared_kernel_metadata"] = shared_metadata
    if shared_metadata.get("evaluate", {}).get("residual_lanes") == 4:
        if metadata["residual_lanes"] != 4:
            source = Path(__file__).with_name("cell.hpp").read_text(encoding="utf-8") + "\n" + source
        # The helper can use AVX2 even if the remaining cell functions cannot.
        metadata["residual_lanes"] = 4
    metadata["vector_exponentials"] = "scalar libm"
    if vector_exponentials and metadata["residual_lanes"] == 4:
        # Only the four-lane exp changes; scalar rows, Jacobian and report math
        # keep their existing functions. The caller verifies AVX2 and FMA3.
        source = source.replace(
            "CELL_UNARY(exp)",
            "static PB_INLINE CellPacket exp(CellPacket a) {"
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
    linkage="PB_EXPORT",
    function_suffix="",
    include_headers=True,
):
    runtime_parameters = any(node[0] == "param" for node in graph.nodes)
    runtime_arg = "runtime," if runtime_parameters else ""
    runtime_decl = "const double* runtime," if runtime_parameters else ""
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

    active = list(graph.postorder([node for selected in roots.values() for node in selected]))
    time_only = {}
    for node in active:
        op, *args = graph.nodes[node]
        time_only[node] = (op in ("const", "time") if op in LEAVES else
                           all(time_only[arg] for arg in args))

    time_nodes = {
        node
        for node in active
        if time_only[node] and graph.nodes[node][0] not in ("const", "time")
    }
    frontier = set()
    for node in active:
        op, *args = graph.nodes[node]
        if not time_only[node] and op not in ("var", "dot", "cj", "param"):
            frontier.update(arg for arg in args if arg in time_nodes)
    frontier.update(
        node for selected in roots.values() for node in selected if node in time_nodes
    )
    frontier = sorted(frontier)
    positions = {node: i for i, node in enumerate(frontier)}

    time_references = {node: f"time_values[{index}]" for node, index in positions.items()}

    source = "#include <cmath>\n#include <cstring>\n" if include_headers else ""
    source += (
        "static thread_local bool cached_valid=false;\n"
        "static thread_local double cached_t=0;\n"
        f"static thread_local double time_values[{max(1, len(frontier))}];\n"
    )
    source += graph.emit("update_time", frontier, mapping, linkage="static")
    source += (
        f"static void ensure_time(double t{',const double* runtime' if runtime_parameters else ''}) {{\n"
        "if(!cached_valid || t!=cached_t) {\n"
        f"update_time(t,nullptr,nullptr,0,{runtime_arg}time_values);cached_t=t;cached_valid=true;\n"
        "}}\n"
    )
    source += helpers
    canonical = {}
    constants = defaultdict(dict)
    fixed_constants = {0.0, 1.0, -1.0, 2.0, 0.5}

    normalized_nodes = defaultdict(dict)

    def normalized(node, cell):
        known = normalized_nodes[cell]
        children = lambda current: () if current in positions else graph.children(current)
        for current in postorder([node], children, known):
            op, *args = graph.nodes[current]
            if current in positions:
                # A cached expression belongs to this exact time function. Treating
                # its constants as cell parameters could incorrectly share a cache
                # slot between spatially different time-dependent coefficients.
                key = ("cached_time", current)
            elif op in ("var", "dot"):
                key = (op, variable_key(args[0], cell))
            elif op == "const" and args[0] not in fixed_constants:
                bindings = constants[cell]
                if args[0] not in bindings:
                    bindings[args[0]] = len(bindings)
                key = ("parameter", bindings[args[0]])
            elif op in ("const", "time", "cj", "param"):
                key = graph.nodes[current]
            else:
                key = (op, *(known[arg] for arg in args))
            if key not in canonical:
                canonical[key] = len(canonical)
            known[current] = canonical[key]
        return known[node]

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
            source += graph.emit(name + function_suffix, selected, mapping, references=time_references,
                                 linkage=linkage, prologue=f"ensure_time(t{',runtime' if runtime_parameters else ''});")
            continue
        normalized_nodes.clear()
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
            variable_map = {old: i for i, old in enumerate(dependencies)}
            parameter_nodes = {node: constants[cell][expression[1]]
                               for node, expression in enumerate(graph.nodes)
                               if expression[0] == "const" and expression[1] in constants[cell]}
            input_stride = max(1, len(dependencies))
            parameter_stride = max(1, len(constants[cell]))
            signature_args = (f"double t,const double* y,const double* yp,double cj,{runtime_decl}"
                              "const int* ix,const double* par,double* PB_RESTRICT out,const int* offsets")
            source += graph.emit(
                function, nodes, variable_map,
                references={**time_references, **{node: f"par[{index}]" for node, index in parameter_nodes.items()}},
                input_reference=lambda op, old: f"{'y' if op == 'var' else 'yp'}[ix[{variable_map[old]}]]",
                signature=f"static PB_NOINLINE void {function}({signature_args})",
                store=lambda index, value: f"out[offsets[{index}]] = {value};")
            metadata["direct_scalar_output_functions"] = metadata.get("direct_scalar_output_functions", 0) + 1
            vectorized = vectorize and name == "evaluate" and len(groups) >= 4
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
                broadcast = {j for j in range(len(parameters[0])) if all(
                    repr(parameters[i + lane][j]) == repr(parameters[i][j])
                    for i in range(0, len(groups) - 3, 4) for lane in range(4))}
                references = {**time_references, **{
                    node: f"CellPacket(par[{index}])" if index in broadcast else
                          f"cell_parameter(par,{index},{parameter_stride})"
                    for node, index in parameter_nodes.items()}}
                source += graph.emit(
                    function + "_vec", nodes, variable_map, references=references,
                    input_reference=lambda op, old: f"gather_cells({'y' if op == 'var' else 'yp'},ix+{variable_map[old]},{input_stride})",
                    temporary_type="CellPacket",
                    signature=f"static PB_NOINLINE void {function}_vec({signature_args})",
                    store=lambda index, value: f"scatter_cells(out,offsets+{index},{len(entries)},{value});")
                used = {parameter_nodes[node] for node in graph.postorder(nodes, time_references) if node in parameter_nodes}
                metadata["broadcast_cell_parameters"] = metadata.get("broadcast_cell_parameters", 0) + len(broadcast & used)
                metadata["vectorized_residual_cells"] += len(groups) // 4 * 4
                metadata["residual_lanes"] = 4
                metadata["direct_vector_output_functions"] = metadata.get("direct_vector_output_functions", 0) + 1

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
                    f"{function}_vec(t,y,yp,cj,{runtime_arg}ix[i],parameters[i],out,offsets[i]);\n"
                    "}\n"
                )
            calls.append(
                "{\n"
                f"static const int ix[{count}][{max(1, len(dependencies))}]={table(inputs)};\n"
                f"static const int offsets[{count}][{width}]={table(outputs)};\n"
                f"static const double parameters[{count}][{len(parameters[0])}]={table(parameters)};\n"
                "int i=0;\n" + batch + f"for(;i<{count};i++) {{\n"
                f"{function}(t,y,yp,cj,{runtime_arg}ix[i],parameters[i],out,offsets[i]);\n"
                "}}\n"
            )
        preparation = ""
        if plan:
            # Thread-local storage avoids a stack allocation proportional to
            # mesh size. Every entry is refreshed from this callback's inputs.
            preparation = (
                f"static thread_local double extended[{len(plan.extended_keep)}];\n"
                f"memcpy(extended,y,{len(keep)}*sizeof(double));\n"
                f"shared_{name}::evaluate(t,y,yp,cj,{runtime_arg}extended+{len(keep)});\ny=extended;\n"
            )
        source += (
            f'{linkage} void {name}{function_suffix}(double t,const double* y,'
            f"const double* yp,double cj,{runtime_decl}double* out) {{\nensure_time(t{',runtime' if runtime_parameters else ''});\n"
            + preparation
            + "".join(calls)
            + "}\n"
        )
    if metadata["residual_lanes"] == 4 and include_headers:
        source = (
            Path(__file__).with_name("cell.hpp").read_text(encoding="utf-8")
            + "\n"
            + source
        )
    return source, metadata
