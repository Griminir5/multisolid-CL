from dataclasses import dataclass
from typing import Dict, Literal

VALID_GAS_SPECIES = ["AR", "CH4", "CO", "CO2", "H2", "H2O", "HE", "N2", "O2"]
#VALID_SOLID_SPECIES = ["CaAl:A-01", "Ni", "NiO"]


@dataclass(frozen=True)
class InletProgramStep:
    duration: float
    kind: Literal["hold", "ramp"] = "ramp"
    F_target: float = None
    y_target: Dict[str, float] = None


class InletProgram:
    """Piecewise-linear inlet program made of ramps and holds."""

    def __init__(self, initial_F: float, initial_y: Dict[str, float], composition_tolerance=1e-9, gas_species=None):
        self.composition_tolerance = composition_tolerance
        self.gas_species = tuple(initial_y.keys()) if gas_species is None else tuple(gas_species)
        self.initial_F = initial_F
        self.initial_y = self.validate_composition(initial_y)
        self.steps = []

    def add_ramp(self, duration, F=None, y=None):
        self.steps.append(InletProgramStep(duration=duration, kind="ramp", F_target=F, y_target=y))

    def add_hold(self, duration):
        self.steps.append(InletProgramStep(duration=duration, kind="hold"))

    def validate_composition(self, composition, tolerance=None):
        """Require a complete species->fraction mapping and only enforce sum-to-one."""
        tolerance = self.composition_tolerance if tolerance is None else tolerance
        species_set = set(self.gas_species)

        missing_species = species_set - set(composition)
        if missing_species:
            raise ValueError(f"Missing gas species in step definition: {missing_species}")

        unknown_species = set(composition) - species_set
        if unknown_species:
            raise ValueError(f"Unknown gas species in step definition: {unknown_species}")

        total = sum(composition[species] for species in self.gas_species)
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"Inlet composition must sum to 1.0 within {tolerance:g}; received {total:.12g}.")

        return {species: composition[species] for species in self.gas_species}

    def build(self, repeat=False, time_horizon=None):
        if repeat and time_horizon is None:
            raise ValueError("time_horizon must be provided when repeat=True.")

        times = [0.0]
        F_profile = [self.initial_F]
        y_profiles = {species: [self.initial_y[species]] for species in self.gas_species}

        current_time = 0.0
        current_F = self.initial_F
        current_y = dict(self.initial_y)

        while True:
            for step in self.steps:
                current_time += step.duration

                next_F = current_F if step.F_target is None else step.F_target
                next_y = dict(current_y) if step.y_target is None else self.validate_composition(step.y_target)

                times.append(current_time)
                F_profile.append(next_F)
                for species in self.gas_species:
                    y_profiles[species].append(next_y[species])

                current_F = next_F
                current_y = next_y

                if repeat and current_time >= time_horizon:
                    break

            if not repeat or current_time >= time_horizon or not self.steps:
                break

        if time_horizon is not None and times[-1] < time_horizon:
            times.append(time_horizon)
            F_profile.append(current_F)
            for species in self.gas_species:
                y_profiles[species].append(current_y[species])

        return {
            "times": times,
            "F_in": F_profile,
            "y_in": y_profiles,
            "end_time": times[-1],
        }
    
