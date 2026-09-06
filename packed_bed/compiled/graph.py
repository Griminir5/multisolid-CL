"""Reduce explicit DAETools closures and differentiate a shared expression graph.

The model remains the source of the equations. No kinetic or transport
correlations are duplicated here. Enthalpy and species concentrations remain
differential variables; only named explicit algebraic definitions are removed.
"""

import math
from functools import lru_cache


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

    def from_dae(self, node, index_map):
        import daetools.pyDAE as d

        kind = type(node).__name__
        if kind == "adConstantNode":
            return self.constant(node.Quantity.value)
        if kind in ("adRuntimeParameterNode", "adDomainIndexNode"):
            return self.constant(node.Value)
        if kind == "adTimeNode":
            return self.make("time")
        if kind == "adRuntimeVariableNode":
            return self.make("var", index_map[node.OverallIndex])
        if kind == "adRuntimeTimeDerivativeNode":
            return self.make("dot", index_map[node.OverallIndex])
        if kind == "adUnaryNode":
            mapping = {
                d.eSign: "neg",
                d.eSqrt: "sqrt",
                d.eExp: "exp",
                d.eAbs: "abs",
                d.eLn: "log",
                d.eLog: "log10",
            }
            return self.make(
                mapping[node.Function], self.from_dae(node.Node, index_map)
            )
        if kind == "adBinaryNode":
            mapping = {
                d.ePlus: "add",
                d.eMinus: "sub",
                d.eMulti: "mul",
                d.eDivide: "div",
                d.ePower: "pow",
                d.eMin: "min",
                d.eMax: "max",
            }
            return self.make(
                mapping[node.Function],
                self.from_dae(node.LNode, index_map),
                self.from_dae(node.RNode, index_map),
            )
        raise NotImplementedError(kind)

    def dependencies(self, node):
        if node not in self._dependencies:
            self._dependencies[node] = self._dependencies_for(node)
        return self._dependencies[node]

    def _dependencies_for(self, node):
        op, *args = self.nodes[node]
        if op in ("var", "dot"):
            return frozenset(args)
        if op in ("const", "time", "cj"):
            return frozenset()
        return frozenset().union(*(self.dependencies(a) for a in args))

    def isolate(self, node, target):
        @lru_cache(None)
        def parts(n):
            op, *a = self.nodes[n]
            if target not in self.dependencies(n):
                return self.zero, n
            if op == "var" and a[0] == target:
                return self.one, self.zero
            if op in ("add", "sub"):
                p, q = parts(a[0])
                r, s = parts(a[1])
                return self.make(op, p, r), self.make(op, q, s)
            if op == "neg":
                p, q = parts(a[0])
                return self.make("neg", p), self.make("neg", q)
            if op == "mul":
                if target not in self.dependencies(a[0]):
                    p, q = parts(a[1])
                    return self.make("mul", a[0], p), self.make("mul", a[0], q)
                if target not in self.dependencies(a[1]):
                    p, q = parts(a[0])
                    return self.make("mul", a[1], p), self.make("mul", a[1], q)
            if op == "div" and target not in self.dependencies(a[1]):
                p, q = parts(a[0])
                return self.make("div", p, a[1]), self.make("div", q, a[1])
            raise ValueError("Nonlinear elimination")

        coefficient, remainder = parts(node)
        return self.make("div", self.make("neg", remainder), coefficient)

    def substitute(self, roots, replacements):
        visiting = set()

        @lru_cache(None)
        def sub(n):
            if n in visiting:
                raise ValueError("Cyclic elimination")
            visiting.add(n)
            op, *a = self.nodes[n]
            if op == "var" and a[0] in replacements:
                result = sub(replacements[a[0]])
            elif op in ("const", "var", "dot", "time", "cj"):
                result = n
            else:
                result = self.make(op, *(sub(x) for x in a))
            visiting.remove(n)
            return result

        return [sub(n) for n in roots]

    def gradient(self, n):
        if n not in self._gradients:
            self._gradients[n] = self._gradient_for(n)
        return self._gradients[n]

    def _gradient_for(self, n):
        op, *a = self.nodes[n]
        if op == "var":
            return {a[0]: self.one}
        if op == "dot":
            return {a[0]: self.make("cj")}
        if op in ("const", "time", "cj"):
            return {}
        ga = self.gradient(a[0])
        gb = self.gradient(a[1]) if len(a) > 1 else {}
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

    def emit(self, name, roots, variable_map):
        # Finish and store each output in dependency order, keeping common
        # expressions shared. This shortens temporary lifetimes in large kernels.
        # Output storage is separate from the input state/derivative arrays.
        refs = {}
        lines = []

        def visit(n):
            if n in refs:
                return
            op, *args = self.nodes[n]
            if op not in ("const", "var", "dot", "time", "cj"):
                for arg in args:
                    visit(arg)
            if op == "const":
                refs[n] = repr(args[0])
                return
            if op == "var":
                refs[n] = f"y[{variable_map[args[0]]}]"
                return
            if op == "dot":
                refs[n] = f"yp[{variable_map[args[0]]}]"
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
            lines.append(f"const double z{n} = {expression};")

        for i, n in enumerate(roots):
            visit(n)
            lines.append(f"out[{i}] = {refs[n]};")
        return (
            f'extern "C" __declspec(dllexport) void {name}(double t, const double* y, const double* yp, double cj, double* out) {{\n'
            + "\n".join(lines)
            + "\n}\n"
        )


def export_model(simulation):
    g = Graph()
    mapping = simulation.IndexMappings
    infos = sorted(
        (x for eq in simulation.model.Equations for x in eq.EquationExecutionInfos),
        key=lambda x: x.EquationIndex,
    )
    roots = [g.from_dae(x.Node, mapping) for x in infos]
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
