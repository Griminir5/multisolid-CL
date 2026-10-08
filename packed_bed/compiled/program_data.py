"""Named boundary inputs and per-run storage for exact smoothed programs."""

import ctypes as C
import hashlib
import math
from pathlib import Path

import numpy as np

from ..programs import RatioProgram


def boundary_slots(species):
    yield "Active_inlet_flow_smooth", "F_in", 0, "inlet_flow", 0
    for i, name in enumerate(species):
        yield f"Active_inlet_composition_{i}_smooth", "y_in", i, f"inlet_composition.{name}", i + 1
    yield "Active_inlet_temperature_smooth", "T_in", 0, "inlet_temperature", len(species) + 1
    yield "Active_outlet_pressure_smooth", "P_out", 0, "outlet_pressure", len(species) + 2


class ProgramData(C.Structure):
    # Keep in sync with program_data.hpp. All pointed-to arrays belong to one run.
    _fields_ = [("channels", C.c_int), ("outputs", C.c_int), ("valid", C.c_int),
                ("width", C.c_double), ("time", C.c_double)] + [
        (name, C.c_void_p) for name in (
            "initial", "offsets", "repeat_offsets", "period", "start", "end", "base", "delta",
            "numerator", "denominator", "raw", "values")]


class RuntimePrograms:
    """Two compact cycles per channel and private native evaluation scratch space."""

    def __init__(self, programs, width):
        if not math.isfinite(width) or width <= 0:
            raise ValueError("Program smoothing width must be finite and positive.")
        arrays = {name: [] for name in ("initial", "repeat_offsets", "period", "start", "end", "base", "delta",
                                       "numerator", "denominator")}
        arrays["offsets"] = [0]
        channels = {}

        def channel(program, component):
            key = (program, component)
            if key in channels:
                return channels[key]
            index = len(channels)
            channels[key] = index
            def scalar(value):
                return float(value if component is None else value[component])
            arrays["initial"].append(scalar(program.initial_value))
            arrays["period"].append(program.duration_s if program.repeat_segments else 0.)
            for cycle, segments in enumerate((program.segments, program.repeat_segments)):
                if cycle:
                    arrays["repeat_offsets"].append(len(arrays["start"]))
                for segment in segments:
                    if not (math.isfinite(segment.start_time) and math.isfinite(segment.end_time)
                            and segment.end_time > segment.start_time):
                        raise ValueError("Program segments must have positive finite duration.")
                    arrays["start"].append(segment.start_time)
                    arrays["end"].append(segment.end_time)
                    arrays["base"].append(scalar(segment.start_value))
                    arrays["delta"].append(scalar(segment.end_value) - scalar(segment.start_value))
            arrays["offsets"].append(len(arrays["start"]))
            return index

        for program, component in programs:
            if isinstance(program, RatioProgram):
                arrays["numerator"].append(channel(program.numerator, component))
                arrays["denominator"].append(channel(program.denominator, None))
            else:
                arrays["numerator"].append(channel(program, component))
                arrays["denominator"].append(-1)
        integers = {"offsets", "repeat_offsets", "numerator", "denominator"}
        self.arrays = {name: np.ascontiguousarray(values, dtype=np.int32 if name in integers else np.float64)
                       for name, values in arrays.items()}
        fingerprint = hashlib.sha256(b"compact-support-program-v2" + np.float64(width).tobytes())
        for name, array in self.arrays.items():
            fingerprint.update(name.encode())
            fingerprint.update(str(array.shape).encode())
            fingerprint.update(array.tobytes())
            array.flags.writeable = False
        self.fingerprint = fingerprint.hexdigest()
        self.arrays.update(raw=np.empty(len(channels)), values=np.empty(len(programs)))
        self.data = ProgramData(len(channels), len(programs), 0, width, 0,
                                *(self.arrays[name].ctypes.data for name, _ in ProgramData._fields_[5:]))
        self.pointer = C.cast(C.pointer(self.data), C.c_void_p)

    @classmethod
    def from_simulation(cls, simulation):
        model = simulation.model
        return cls([(model.inlet_flow_program, None),
                    *((model.inlet_composition_program, i) for i in range(len(model.gas_species))),
                    (model.inlet_temperature_program, None), (model.outlet_pressure_program, None)],
                   model.smooth_ramp_width_s)


def program_source():
    return Path(__file__).with_suffix(".hpp").read_text(encoding="utf-8")


def program_wrappers():
    return "\n".join(
        f"PB_EXPORT void {name}(double t,const double* y,const double* yp,double cj,double* out,ProgramData* programs) {{\n"
        f"{name}_values(t,y,yp,cj,program_values(t,programs),out);\n}}"
        for name in ("evaluate", "jacobian", "reconstruct"))
