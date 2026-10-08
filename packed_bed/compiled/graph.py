"""Reduce explicit DAETools closures and differentiate a shared expression graph.

The model remains the source of the equations. No kinetic or transport
correlations are duplicated here. Internal energy and species concentrations remain
differential variables; only named explicit algebraic definitions are removed.
"""

import math
import hashlib
import json


LEAVES = frozenset(("const", "var", "dot", "time", "cj", "param"))


def postorder(roots, children, known=()):
    """Depth-first, left-to-right postorder without using the Python call stack."""
    done, visiting = set(), set()
    for root in roots:
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if node in known or node in done:
                continue
            if expanded:
                visiting.remove(node)
                done.add(node)
                yield node
            else:
                if node in visiting:
                    raise ValueError("Cyclic elimination")
                visiting.add(node)
                stack.append((node, True))
                stack.extend((child, False) for child in reversed(children(node)))


class Graph:
    def __init__(self):
        self.nodes = []
        self.intern = {}
        self._dependencies = {}
        self._gradients = {}
        self.zero = self.make("const", 0.0)
        self.one = self.make("const", 1.0)

    def make(self, op, *args):
        if op == "neg" and self.nodes[args[0]][0] == "const":
            return self.constant(-self.nodes[args[0]][1])
        if op in ("min", "max") and all(self.nodes[a][0] == "const" for a in args):
            values = [self.nodes[a][1] for a in args]
            return self.constant(min(values) if op == "min" else max(values))
        if op in ("add", "sub", "mul", "div", "pow"):
            a, b = args
            av = self.nodes[a][1] if self.nodes[a][0] == "const" else None
            bv = self.nodes[b][1] if self.nodes[b][0] == "const" else None
            if av is not None and bv is not None:
                try:
                    v = {
                        "add": lambda: av + bv,
                        "sub": lambda: av - bv,
                        "mul": lambda: av * bv,
                        "div": lambda: av / bv,
                        "pow": lambda: av**bv,
                    }[op]()
                    if isinstance(v, (float, int)) and math.isfinite(v):
                        return self.make("const", float(v))
                except (ValueError, ZeroDivisionError, OverflowError):
                    pass
            if op == "add":
                if av == 0:
                    return b
                if bv == 0:
                    return a
            if op == "sub" and bv == 0:
                return a
            if op == "mul":
                if av == 0 or bv == 0:
                    return self.zero
                if av == 1:
                    return b
                if bv == 1:
                    return a
            if op == "div":
                if av == 0:
                    return self.zero
                if bv == 1:
                    return a
            if op == "pow":
                if bv == 0:
                    return self.one
                if bv == 1:
                    return a
                if bv == 2:
                    return self.make("mul", a, a)
                if bv == 3:
                    return self.make("mul", a, self.make("mul", a, a))
                if bv == 4:
                    square = self.make("mul", a, a)
                    return self.make("mul", square, square)
        key = (op, *args)
        if key not in self.intern:
            self.intern[key] = len(self.nodes)
            self.nodes.append(key)
        return self.intern[key]

    def constant(self, x):
        return self.make("const", float(x))

    def parameter(self, name, slot):
        """An exogenous input whose identity survives constant folding."""
        return self.make("param", name, slot)

    def children(self, node):
        op, *args = self.nodes[node]
        return () if op in LEAVES else args

    def postorder(self, roots, known=()):
        return postorder(roots, self.children, known)

    def fingerprint(self, roots):
        """Hash reachable instructions independently of discarded graph nodes."""
        nodes, mapping = [], {}
        for node in self.postorder(roots):
            op, *args = self.nodes[node]
            if op not in LEAVES:
                args = [mapping[arg] for arg in args]
            mapping[node] = len(nodes)
            nodes.append((op, *args))
        indices = [mapping[node] for node in roots]
        return hashlib.sha256(json.dumps([nodes, indices], allow_nan=False).encode()).hexdigest()

    def from_dae(self, node, index_map):
        import daetools.pyDAE as d

        unary = {d.eSign: "neg", d.eSqrt: "sqrt", d.eExp: "exp", d.eAbs: "abs",
                 d.eLn: "log", d.eLog: "log10"}
        binary = {d.ePlus: "add", d.eMinus: "sub", d.eMulti: "mul", d.eDivide: "div",
                  d.ePower: "pow", d.eMin: "min", d.eMax: "max"}
        stack, values = [(node, False)], []
        while stack:
            current, expanded = stack.pop()
            kind = type(current).__name__
            if kind in ("adUnaryNode", "adBinaryNode"):
                if not expanded:
                    stack.append((current, True))
                    if kind == "adBinaryNode":
                        stack.append((current.RNode, False))
                        stack.append((current.LNode, False))
                    else:
                        stack.append((current.Node, False))
                    continue
                if kind == "adBinaryNode":
                    right, left = values.pop(), values.pop()
                    value = self.make(binary[current.Function], left, right)
                else:
                    value = self.make(unary[current.Function], values.pop())
            elif kind == "adConstantNode":
                value = self.constant(current.Quantity.value)
            elif kind in ("adRuntimeParameterNode", "adDomainIndexNode"):
                value = self.constant(current.Value)
            elif kind == "adTimeNode":
                value = self.make("time")
            elif kind == "adRuntimeVariableNode":
                value = self.make("var", index_map[current.OverallIndex])
            elif kind == "adRuntimeTimeDerivativeNode":
                value = self.make("dot", index_map[current.OverallIndex])
            else:
                raise NotImplementedError(kind)
            values.append(value)
        return values[0]

    def dependencies(self, node):
        for n in self.postorder([node], self._dependencies):
            op, *args = self.nodes[n]
            self._dependencies[n] = (frozenset(args) if op in ("var", "dot") else
                                     frozenset() if op in LEAVES else
                                     frozenset().union(*(self._dependencies[a] for a in args)))
        return self._dependencies[node]

    def isolate(self, node, target):
        parts = {}

        def split(n):
            op, *a = self.nodes[n]
            if target not in self.dependencies(n):
                return self.zero, n
            if op == "var" and a[0] == target:
                return self.one, self.zero
            if op in ("add", "sub"):
                p, q = parts[a[0]]
                r, s = parts[a[1]]
                return self.make(op, p, r), self.make(op, q, s)
            if op == "neg":
                p, q = parts[a[0]]
                return self.make("neg", p), self.make("neg", q)
            if op == "mul":
                if target not in self.dependencies(a[0]):
                    p, q = parts[a[1]]
                    return self.make("mul", a[0], p), self.make("mul", a[0], q)
                if target not in self.dependencies(a[1]):
                    p, q = parts[a[0]]
                    return self.make("mul", a[1], p), self.make("mul", a[1], q)
            if op == "div" and target not in self.dependencies(a[1]):
                p, q = parts[a[0]]
                return self.make("div", p, a[1]), self.make("div", q, a[1])
            raise ValueError("Nonlinear elimination")

        def children(n):
            return self.children(n) if target in self.dependencies(n) else ()

        for n in postorder([node], children):
            parts[n] = split(n)
        coefficient, remainder = parts[node]
        return self.make("div", self.make("neg", remainder), coefficient)

    def substitute(self, roots, replacements):
        def children(n):
            op, *args = self.nodes[n]
            if op == "var" and args[0] in replacements:
                return (replacements[args[0]],)
            return self.children(n)

        substituted = {}
        for n in postorder(roots, children):
            op, *args = self.nodes[n]
            if op == "var" and args[0] in replacements:
                result = substituted[replacements[args[0]]]
            elif op in LEAVES:
                result = n
            else:
                result = self.make(op, *(substituted[a] for a in args))
            substituted[n] = result
        return [substituted[n] for n in roots]

    def gradient(self, n):
        for node in self.postorder([n], self._gradients):
            self._gradients[node] = self._gradient_for(node)
        return self._gradients[n]

    def _gradient_for(self, n):
        op, *a = self.nodes[n]
        if op == "var":
            return {a[0]: self.one}
        if op == "dot":
            return {a[0]: self.make("cj")}
        if op in ("const", "time", "cj", "param"):
            return {}
        ga = self._gradients[a[0]]
        gb = self._gradients[a[1]] if len(a) > 1 else {}
        out = {}
        for k in ga.keys() | gb.keys():
            da = ga.get(k, self.zero)
            db = gb.get(k, self.zero)
            if op in ("add", "sub"):
                v = self.make(op, da, db)
            elif op == "mul":
                v = self.make(
                    "add", self.make("mul", da, a[1]), self.make("mul", a[0], db)
                )
            elif op == "div":
                v = self.make(
                    "div",
                    self.make(
                        "sub", self.make("mul", da, a[1]), self.make("mul", a[0], db)
                    ),
                    self.make("mul", a[1], a[1]),
                )
            elif op == "neg":
                v = self.make("neg", da)
            elif op == "exp":
                v = self.make("mul", n, da)
            elif op == "sqrt":
                v = self.make("div", da, self.make("mul", self.constant(2), n))
            elif op == "log":
                v = self.make("div", da, a[0])
            elif op == "log10":
                v = self.make(
                    "div", da, self.make("mul", a[0], self.constant(math.log(10)))
                )
            elif op == "abs":
                v = self.make("absgrad", a[0], da)
            elif op in ("min", "max"):
                v = self.make(op + "grad", a[0], a[1], da, db)
            elif op == "pow":
                v = self.make(
                    "mul",
                    self.make(
                        "mul",
                        a[1],
                        self.make("pow", a[0], self.make("sub", a[1], self.one)),
                    ),
                    da,
                )
                if db != self.zero:
                    v = self.make(
                        "add",
                        v,
                        self.make(
                            "mul", self.make("mul", n, self.make("log", a[0])), db
                        ),
                    )
            else:
                raise NotImplementedError(op)
            if v != self.zero:
                out[k] = v
        return out

    def emit(self, name, roots, variable_map, *, references=None, input_reference=None,
             temporary_type="double", signature=None, store=None, linkage="PB_EXPORT", prologue=""):
        """Emit arithmetic with explicit addressing, types, signature and stores."""
        # Finish and store each output in dependency order, keeping common
        # expressions shared. This shortens temporary lifetimes in large kernels.
        # Output storage is separate from the input state/derivative arrays.
        refs = dict(references or {})
        lines = [prologue] if prologue else []
        if input_reference is None:
            input_reference = lambda op, old: f"{'y' if op == 'var' else 'yp'}[{variable_map[old]}]"
        if store is None:
            store = lambda index, value: f"out[{index}] = {value};"

        def visit(n):
            op, *args = self.nodes[n]
            if op == "const":
                refs[n] = repr(args[0])
                return
            if op == "param":
                refs[n] = f"runtime[{args[1]}]"
                return
            if op == "var":
                refs[n] = input_reference(op, args[0])
                return
            if op == "dot":
                refs[n] = input_reference(op, args[0])
                return
            if op in ("time", "cj"):
                refs[n] = "t" if op == "time" else "cj"
                return
            a = [refs[x] for x in args]
            if op in ("add", "sub", "mul", "div"):
                operator = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
                expression = f"({a[0]} {operator} {a[1]})"
            elif op == "neg":
                expression = f"(-({a[0]}))"
            elif op == "absgrad":
                expression = (
                    f"({a[0]}>0 ? {a[1]} : ({a[0]}<0 ? -({a[1]}) : fabs({a[1]})))"
                )
            elif op in ("mingrad", "maxgrad"):
                compare = "<" if op == "mingrad" else ">"
                fn = "fmin" if op == "mingrad" else "fmax"
                expression = f"({a[0]} {compare} {a[1]} ? {a[2]} : ({a[0]}=={a[1]} ? {fn}({a[2]},{a[3]}) : {a[3]}))"
            else:
                fn = {"abs": "fabs", "min": "fmin", "max": "fmax"}.get(op, op)
                expression = f"{fn}({','.join(a)})"
            refs[n] = f"z{n}"
            lines.append(f"const {temporary_type} z{n} = {expression};")

        for i, n in enumerate(roots):
            for node in self.postorder([n], refs):
                visit(node)
            lines.append(store(i, refs[n]))
        parameters = "const double* runtime, " if any(n[0] == "param" for n in self.nodes) else ""
        signature = signature or f'{linkage} void {name}(double t, const double* y, const double* yp, double cj, {parameters}double* out)'
        return signature + " {\n" + "\n".join(lines) + "\n}\n"


def export_model(simulation):
    g = Graph()
    mapping = simulation.IndexMappings
    infos = sorted(
        (x for eq in simulation.model.Equations for x in eq.EquationExecutionInfos),
        key=lambda x: x.EquationIndex,
    )
    roots = []
    boundaries = {}
    from .program_data import boundary_slots
    model = simulation.model
    for equation, variable, component, name, slot in boundary_slots(model.gas_species):
        old = mapping[getattr(model, variable).OverallIndex + component]
        boundaries[equation] = g.make("sub", g.make("var", old), g.parameter(name, slot))
    for info in infos:
        from .cache import check_cancelled
        check_cancelled()
        if info.Equation.Name in boundaries:
            roots.append(boundaries[info.Equation.Name])
            continue
        try:
            roots.append(g.from_dae(info.Node, mapping))
        except (KeyError, NotImplementedError) as exc:
            raise ValueError(f"Compiled execution does not support expression {exc} in equation "
                             f"{info.Equation.Name}. Check the selected property/kinetics definitions "
                             "or explicitly select Standard execution.") from exc
    variables = {
        mapping[v.OverallIndex + i]: v.Name
        for v in simulation.model.Variables
        for i in range(v.NumberOfPoints)
    }
    replacements = {}
    removed = set()
    for i, info in enumerate(infos):
        name = info.Equation.Name
        target = None
        for prefixes, var in [
            (("total_concentration_closure",), "c_gas_tot"),
            (("solid_total_concentration_closure",), "c_sol_tot"),
            (("molar_fraction_calc",), "y_gas"),
            (("lhs_boundary_flux", "rhs_boundary_flux", "face_flux"), "N_gas_face"),
            (("reaction_rate",), "R_rxn"),
            (("gas_component_enthalpy",), "h_gas"),
            (("solid_component_enthalpy",), "h_sol"),
            (
                ("lhs_boundary_enthalpy", "rhs_boundary_enthalpy", "face_enthalpy"),
                "J_gas_face",
            ),
            (("axial_dispersion_face",), "Dax"),
            (("gas_equation_of_state",), "pres_bed"),
            (("gas_mixture_viscosity",), "mu_g"),
            (("gas_density_closure",), "rho_g"),
            (("mass_bed_total_definition",), "mass_bed_total"),
            (("heat_bed_total_definition",), "heat_bed_total"),
            (("Active_inlet_flow",), "F_in"),
            (("Active_inlet_composition",), "y_in"),
            (("Active_inlet_temperature",), "T_in"),
            (("Active_outlet_pressure",), "P_out"),
        ]:
            if name.startswith(prefixes):
                target = var
                break
        if target:
            candidates = [j for j in g.dependencies(roots[i]) if variables[j] == target]
            if len(candidates) != 1:
                raise RuntimeError((name, candidates))
            j = candidates[0]
            replacements[j] = g.isolate(roots[i], j)
            removed.add(i)
    keep = [j for j in range(simulation.NumberOfEquations) if j not in replacements]
    reduced_roots = g.substitute(
        [r for i, r in enumerate(roots) if i not in removed], replacements
    )
    reconstructed = g.substitute(
        [g.make("var", j) for j in range(simulation.NumberOfEquations)], replacements
    )
    return g, keep, reduced_roots, reconstructed, variables
