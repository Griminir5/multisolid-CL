"""Pinned SUNDIALS 7.5 C ABI with native residual and Jacobian callbacks.

The scikit-sundae wheel provides the runtime. Its Python wrapper does
not expose algebraic error suppression or the Newton convergence coefficient;
this adapter calls the public IDA C API to preserve those case controls.
No private SUNDIALS structures are accessed.
"""

import ctypes as C
import math
import platform
from pathlib import Path

import numpy as np

from .structure import band_layout

P = C.c_void_p
D = C.c_double
I = C.c_int
L = C.c_long


class CallbackData(C.Structure):
    _fields_ = [
        (name, P)
        for name in (
            "vec",
            "data",
            "index_values",
            "index_pointers",
            "row_scale",
            "jac_values",
        )
    ]


def check_runtime():
    if platform.system() not in {"Windows", "Linux", "Darwin"} or C.sizeof(P) != 8:
        raise RuntimeError("The compiled backend requires 64-bit Windows, Linux or macOS.")
    try:
        import sksundae
    except ImportError as exc:
        raise RuntimeError(
            'Install the compiled backend with: python -m pip install -e ".[compiled]"'
        ) from exc
    if sksundae.__version__ != "1.1.3" or sksundae.SUNDIALS_VERSION != "7.5.0":
        raise RuntimeError(
            "The compiled backend requires a scikit-sundae 1.1.3 wheel (SUNDIALS 7.5.0)."
        )
    directory = Path(sksundae.__file__).resolve().parent
    try:
        config = (directory / "py_config.pxi").read_text()
    except OSError as exc:
        raise RuntimeError(
            "Cannot verify the installed SUNDIALS ABI: missing py_config.pxi."
        ) from exc
    if (
        'SUNDIALS_FLOAT_TYPE = "double"' not in config
        or 'SUNDIALS_INT_TYPE = "int"' not in config
    ):
        raise RuntimeError(
            "Unsupported SUNDIALS ABI: expected double precision and 32-bit indices."
        )
    folder = (
        directory / ".dylibs"
        if platform.system() == "Darwin"
        else directory.parent / "scikit_sundae.libs"
    )
    if not folder.is_dir():
        raise RuntimeError(
            f"Bundled SUNDIALS libraries not found in {folder}. "
            "Install scikit-sundae==1.1.3 using --only-binary=:all:."
        )
    return folder


def load_runtime_library(folder, module):
    """Load only the verified wheel's libraries, including its bundled dependencies."""
    system = platform.system()
    if system == "Windows":
        pattern = f"sundials_{module}-*.dll"
    elif system == "Darwin":
        pattern = f"libsundials_{module}.*.dylib"
    else:
        pattern = f"libsundials_{module}-*.so*"
    candidates = sorted(folder.glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one SUNDIALS {module} library matching {pattern} in {folder}; "
            f"found {len(candidates)}. Reinstall the scikit-sundae==1.1.3 wheel."
        )
    try:
        return C.CDLL(str(candidates[0]))
    except OSError as exc:
        raise RuntimeError(
            f"Cannot load SUNDIALS {module} from {candidates[0]}: {exc}"
        ) from exc


class NativeIDA:
    def __init__(
        self,
        kernel,
        y0,
        yp0,
        atol,
        differential,
        sparsity,
        rtol=1e-3,
        max_order=5,
        nonlinear_coef=1.0,
        max_nonlin_iters=12,
        colperm=3,
        suppress_algebraic_errors=True,
        threads=1,
        row_scale=None,
        linear_solver="superlu",
        band_library=None,
        step_growth_threshold=2.0,
        nonlinear_refresh_interval=0,
        nonlinear_library=None,
    ):
        if (
            isinstance(nonlinear_refresh_interval, bool)
            or not isinstance(nonlinear_refresh_interval, int)
            or nonlinear_refresh_interval < 0
        ):
            raise ValueError(
                "Nonlinear refresh interval must be a nonnegative integer."
            )
        if bool(nonlinear_refresh_interval) != (
            nonlinear_library is not None
        ):
            raise ValueError(
                "Custom Newton requires its helper and a positive refresh interval."
            )
        self.nonlinear_library = nonlinear_library
        self.nonlinear_refresh_interval = nonlinear_refresh_interval
        if not math.isfinite(step_growth_threshold) or step_growth_threshold < 1.0:
            raise ValueError("Step growth threshold must be finite and at least one.")
        if linear_solver not in {"superlu", "band"}:
            raise ValueError("Unknown native linear solver.")
        self.linear_solver = linear_solver
        if band_library is not None and linear_solver != "band":
            raise ValueError("A band library requires the band linear solver.")
        self.band_library = band_library
        folder = check_runtime()
        self.libs = {}
        matrix_modules = (
            ("sunmatrixband", "sunlinsolband")
            if linear_solver == "band"
            else ("sunmatrixsparse", "sunlinsolsuperlumt")
        )
        for key in ("core", "nvecserial", "ida", *matrix_modules):
            self.libs[key] = load_runtime_library(folder, key)

        def fn(lib, name, restype, *argtypes):
            f = getattr(self.libs[lib], name)
            f.restype = restype
            f.argtypes = argtypes
            setattr(self, name, f)
            return f

        fn("core", "SUNContext_Create", I, I, C.POINTER(P))
        fn("core", "SUNContext_Free", I, C.POINTER(P))
        fn("core", "N_VGetArrayPointer", C.POINTER(D), P)
        fn("core", "N_VDestroy", None, P)
        fn("core", "SUNMatDestroy", None, P)
        fn("core", "SUNLinSolFree", I, P)
        fn("nvecserial", "N_VNew_Serial", P, I, P)
        fn("ida", "IDACreate", P, P)
        fn("ida", "IDAInit", I, P, P, D, P, P)
        fn("ida", "IDASVtolerances", I, P, D, P)
        fn("ida", "IDASetId", I, P, P)
        fn("ida", "IDASetUserData", I, P, P)
        fn("ida", "IDASetSuppressAlg", I, P, I)
        fn("ida", "IDASetMaxNonlinIters", I, P, I)
        fn("ida", "IDASetNonlinConvCoef", I, P, D)
        fn("ida", "IDASetMaxOrd", I, P, I)
        fn("ida", "IDASetMaxNumSteps", I, P, L)
        fn("ida", "IDASetEtaFixedStepBounds", I, P, D, D)
        fn("ida", "IDASetStopTime", I, P, D)
        fn("ida", "IDASetLinearSolver", I, P, P, P)
        fn("ida", "IDASetNonlinearSolver", I, P, P)
        fn("ida", "IDASetJacFn", I, P, P)
        fn("ida", "IDASolve", I, P, D, C.POINTER(D), P, P, I)
        fn("ida", "IDAFree", None, C.POINTER(P))
        for name in (
            "NumSteps",
            "NumResEvals",
            "NumJacEvals",
            "NumNonlinSolvIters",
            "NumNonlinSolvConvFails",
            "NumErrTestFails",
        ):
            fn("ida", "IDAGet" + name, I, P, C.POINTER(L))
        if linear_solver == "band":
            fn("sunmatrixband", "SUNBandMatrixStorage", P, I, I, I, I, P)
            data = fn("sunmatrixband", "SUNBandMatrix_Data", C.POINTER(D), P)
            fn("sunlinsolband", "SUNLinSol_Band", P, P, P, P)
            indices = pointers = None
        else:
            fn("sunmatrixsparse", "SUNSparseMatrix", P, I, I, I, I, P)
            data = fn("sunmatrixsparse", "SUNSparseMatrix_Data", C.POINTER(D), P)
            indices = fn(
                "sunmatrixsparse", "SUNSparseMatrix_IndexValues", C.POINTER(I), P
            )
            pointers = fn(
                "sunmatrixsparse", "SUNSparseMatrix_IndexPointers", C.POINTER(I), P
            )
            fn("sunlinsolsuperlumt", "SUNLinSol_SuperLUMT", P, P, P, I, P)
            fn("sunlinsolsuperlumt", "SUNLinSol_SuperLUMTSetOrdering", I, P, I)
        self.kernel = kernel
        self.jac_values = np.empty(sparsity.nnz)
        if hasattr(kernel, "report_loop"):
            kernel.report_loop.argtypes = [P, P, P, P, P, P, I, I, P, P, P, P, P]
            kernel.report_loop.restype = I
        self.row_scale = np.ascontiguousarray(
            np.ones(len(y0)) if row_scale is None else row_scale,
            dtype=float,
        )
        if (
            self.row_scale.shape != (len(y0),)
            or not np.isfinite(self.row_scale).all()
            or np.any(self.row_scale <= 0)
        ):
            raise ValueError(
                "Row scales must be finite, positive and match the state dimension."
            )
        self.callback_data = CallbackData(
            *(
                C.cast(f, P)
                for f in (
                    self.N_VGetArrayPointer,
                    data,
                    indices,
                    pointers,
                )
            ),
            self.row_scale.ctypes.data,
            self.jac_values.ctypes.data,
        )
        self.ctx = P()
        self.mem = P()
        self.matrix = None
        self.linear = None
        self.nonlinear = None
        self.vectors = []
        try:
            self._initialize(
                y0,
                yp0,
                atol,
                differential,
                sparsity,
                rtol,
                max_order,
                nonlinear_coef,
                max_nonlin_iters,
                colperm,
                suppress_algebraic_errors,
                threads,
                step_growth_threshold,
            )
        except BaseException:
            self.close()
            raise

    def _initialize(
        self,
        y0,
        yp0,
        atol,
        differential,
        sparsity,
        rtol,
        max_order,
        nonlinear_coef,
        max_nonlin_iters,
        colperm,
        suppress_algebraic_errors,
        threads,
        step_growth_threshold,
    ):
        kernel = self.kernel
        self.ctx = P()
        self.check(self.SUNContext_Create(0, C.byref(self.ctx)))
        self.n = len(y0)
        self.vectors = []

        def vector(values):
            v = self.N_VNew_Serial(self.n, self.ctx)
            if not v:
                raise MemoryError("N_Vector allocation")
            self.vectors.append(v)
            a = np.ctypeslib.as_array(self.N_VGetArrayPointer(v), shape=(self.n,))
            a[:] = values
            return v, a

        self.y, self.values = vector(y0)
        self.yp, self.derivatives = vector(yp0)
        av, _ = vector(atol)
        ids, _ = vector(np.asarray(differential, dtype=float))
        self.mem = P(self.IDACreate(self.ctx))
        if not self.mem:
            raise MemoryError("IDA allocation")
        self.check(self.IDAInit(self.mem, kernel.residual_callback, 0, self.y, self.yp))
        self.check(self.IDASetUserData(self.mem, C.byref(self.callback_data)))
        self.check(self.IDASVtolerances(self.mem, rtol, av))
        self.check(self.IDASetId(self.mem, ids))
        if self.nonlinear_library is not None:
            from .nonlinear import make_nonlinear_solver

            self.nonlinear = make_nonlinear_solver(
                self,
                self.nonlinear_library,
                self.nonlinear_refresh_interval,
            )
            self.check(self.IDASetNonlinearSolver(self.mem, self.nonlinear))
        self.check(self.IDASetSuppressAlg(self.mem, int(suppress_algebraic_errors)))
        self.check(self.IDASetMaxNonlinIters(self.mem, max_nonlin_iters))
        self.check(self.IDASetNonlinConvCoef(self.mem, nonlinear_coef))
        self.check(self.IDASetMaxOrd(self.mem, max_order))
        self.check(self.IDASetMaxNumSteps(self.mem, 10000))
        # Retain the current step for predicted growth between 1 and this
        # threshold. This does not change IDA's maximum growth factor or weights.
        self.check(self.IDASetEtaFixedStepBounds(self.mem, 1.0, step_growth_threshold))
        if self.linear_solver == "band":
            upper, lower, stored_upper, _ = band_layout(sparsity)
            self.matrix = self.SUNBandMatrixStorage(
                self.n, upper, lower, stored_upper, self.ctx
            )
        else:
            self.matrix = self.SUNSparseMatrix(
                self.n, self.n, sparsity.nnz, 0, self.ctx
            )
        if not self.matrix:
            raise MemoryError("Jacobian matrix allocation")
        if self.linear_solver == "band":
            self.linear = self.SUNLinSol_Band(self.y, self.matrix, self.ctx)
        else:
            self.linear = self.SUNLinSol_SuperLUMT(
                self.y, self.matrix, max(1, threads), self.ctx
            )
        if not self.linear:
            raise RuntimeError("Native linear solver construction")
        if self.band_library is not None:
            from .band import make_band_solver

            prototype = self.linear
            self.linear = make_band_solver(self, self.band_library, prototype, sparsity)
            self.SUNLinSolFree(prototype)
        if self.linear_solver == "superlu":
            self.check(self.SUNLinSol_SuperLUMTSetOrdering(self.linear, colperm))
        self.check(self.IDASetLinearSolver(self.mem, self.linear, self.matrix))
        self.check(self.IDASetJacFn(self.mem, kernel.jacobian_callback))

    @staticmethod
    def check(flag):
        if flag < 0:
            raise RuntimeError(f"SUNDIALS returned {flag}")

    def solve(self, times):
        times = np.ascontiguousarray(times, dtype=float)
        if (
            times.ndim != 1
            or not len(times)
            or times[0] != 0
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(
                "Reporting times must start at zero and increase strictly."
            )
        if len(times) == 1:
            return self.values[None, :].copy(), self.derivatives[None, :].copy()
        self.check(self.IDASetStopTime(self.mem, float(times[-1])))
        values = np.empty((len(times), self.n))
        deriv = np.empty_like(values)
        values[0] = self.values
        deriv[0] = self.derivatives
        reached = D()
        if hasattr(self.kernel, "report_loop"):
            report = I()
            flag = self.kernel.report_loop(
                C.cast(self.IDASolve, P),
                self.mem,
                self.y,
                self.yp,
                self.values.ctypes.data,
                self.derivatives.ctypes.data,
                self.n,
                len(times),
                times.ctypes.data,
                values.ctypes.data,
                deriv.ctypes.data,
                C.byref(reached),
                C.byref(report),
            )
            if flag < 0:
                raise RuntimeError(
                    f"Compiled IDA failed with status {flag} while reporting "
                    f"t={times[report.value]:g} s; last reached {reached.value:g} s."
                )
            return values, deriv
        for i, t in enumerate(times[1:], 1):
            flag = self.IDASolve(self.mem, t, C.byref(reached), self.y, self.yp, 1)
            if flag < 0:
                raise RuntimeError(
                    f"Compiled IDA failed with status {flag} while reporting t={t:g} s; last reached {reached.value:g} s."
                )
            if not np.isfinite(reached.value) or abs(reached.value - t) > (
                16 * np.finfo(float).eps * max(1.0, abs(t))
            ):
                raise RuntimeError(
                    f"Compiled IDA stopped at t={reached.value:g} s "
                    f"before the requested report at t={t:g} s (status {flag})."
                )
            if not (np.isfinite(self.values).all() and np.isfinite(self.derivatives).all()):
                raise RuntimeError(
                    f"Compiled IDA produced nonfinite values at t={t:g} s."
                )
            values[i] = self.values
            deriv[i] = self.derivatives
        return values, deriv

    def stats(self):
        result = {}
        for name in (
            "NumSteps",
            "NumResEvals",
            "NumJacEvals",
            "NumNonlinSolvIters",
            "NumNonlinSolvConvFails",
            "NumErrTestFails",
        ):
            value = L()
            self.check(getattr(self, "IDAGet" + name)(self.mem, C.byref(value)))
            result[name] = value.value
        if self.nonlinear is not None:
            from .nonlinear import nonlinear_statistics

            result.update(
                nonlinear_statistics(self.nonlinear_library, self.nonlinear)
            )
        return result

    def close(self):
        if self.mem:
            self.IDAFree(C.byref(self.mem))
        if self.nonlinear is not None:
            self.nonlinear_library.free_solver(self.nonlinear)
            self.nonlinear = None
        if self.linear:
            self.SUNLinSolFree(self.linear)
            self.linear = None
        if self.matrix:
            self.SUNMatDestroy(self.matrix)
            self.matrix = None
        for v in self.vectors:
            self.N_VDestroy(v)
        self.vectors.clear()
        if self.ctx:
            self.SUNContext_Free(C.byref(self.ctx))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def callback_source(sparsity, linear_solver="superlu"):
    if linear_solver not in {"superlu", "band"}:
        raise ValueError("Unknown callback matrix format.")
    rows = ",".join(map(str, sparsity.indices))
    cols = ",".join(map(str, sparsity.indptr))
    source = (
        """
using Ptr = void*;
struct CallbackData {
    double* (*vec)(Ptr);
    double* (*data)(Ptr);
    int* (*index_values)(Ptr);
    int* (*index_pointers)(Ptr);
    const double* row_scale;
    double* jac_values;
};
PB_EXPORT int residual_callback(double t, Ptr y, Ptr yp, Ptr r, Ptr user) {
    auto& f=*static_cast<CallbackData*>(user);
    double* residual=f.vec(r);
    evaluate(t,f.vec(y),f.vec(yp),0,residual);
    for(int k=0;k<NSTATE;k++) {
        residual[k]*=f.row_scale[k];
        if(!std::isfinite(residual[k])) return 1;
    }
    return 0;
}
PB_EXPORT int jacobian_callback(double t,double cj,Ptr y,Ptr yp,Ptr r,Ptr mat,Ptr user,Ptr tmp1,Ptr tmp2,Ptr tmp3) {
    auto& f=*static_cast<CallbackData*>(user);
    static const int indices[] = {ROWS};
    static const int pointers[] = {COLS};
    memcpy(f.index_values(mat),indices,sizeof(indices));
    memcpy(f.index_pointers(mat),pointers,sizeof(pointers));
    double* values=f.data(mat);
    jacobian(t,f.vec(y),f.vec(yp),cj,values);
    for(int k=0;k<NNZ;k++) {
        values[k]*=f.row_scale[indices[k]];
        if(!std::isfinite(values[k])) return 1;
    }
    return 0;
}
""".replace("ROWS", rows)
        .replace("COLS", cols)
        .replace("NSTATE", str(sparsity.shape[0]))
        .replace("NNZ", str(sparsity.nnz))
    )
    if linear_solver == "band":
        _, _, stored_upper, leading = band_layout(sparsity)
        offsets = [
            col * leading + int(row) - col + stored_upper
            for col in range(sparsity.shape[0])
            for row in sparsity.indices[sparsity.indptr[col] : sparsity.indptr[col + 1]]
        ]
        source = source.replace(
            "memcpy(f.index_values(mat),indices,sizeof(indices));", ""
        )
        source = source.replace(
            "memcpy(f.index_pointers(mat),pointers,sizeof(pointers));", ""
        )
        source = source.replace(
            "double* values=f.data(mat);",
            "double* values=f.jac_values; double* destination=f.data(mat);\n"
            f"memset(destination,0,{sparsity.shape[0] * leading}*sizeof(double));\n"
            "static const int offsets[]={" + ",".join(map(str, offsets)) + "};",
        )
        source = source.replace(
            "if(!std::isfinite(values[k])) return 1;",
            "if(!std::isfinite(values[k])) return 1; destination[offsets[k]]=values[k];",
        )
    source += """
using Solve = int (*)(Ptr,double,double*,Ptr,Ptr,int);
PB_EXPORT int report_loop(Solve solve,Ptr mem,Ptr yvec,Ptr ypvec,
 const double* y,const double* yp,int n,int nt,const double* times,
 double* output,double* derivatives,double* reached,int* report) {
    for(int it=0;it<nt;it++) {
        *report=it;
        if(it) {
            int flag=solve(mem,times[it],reached,yvec,ypvec,1);
            if(flag<0)return flag;
            // Positive statuses can indicate a stop/root before the requested
            // output. Never label an earlier state with a later report time.
            double time_tolerance=16*2.2204460492503131e-16*(1+std::abs(times[it]));
            if(!std::isfinite(*reached) || std::abs(*reached-times[it])>time_tolerance) return -101;
        }
        for(int j=0;j<n;j++) if(!std::isfinite(y[j]) || !std::isfinite(yp[j])) return -100;
        memcpy(output+it*n,y,n*sizeof(double));
        memcpy(derivatives+it*n,yp,n*sizeof(double));
    }
    return 0;
}
"""
    return source
