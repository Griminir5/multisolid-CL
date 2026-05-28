import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import copy
'''
End goal: produce a 12000 rows x 42 columns .npy file which is ready for normalization/standardization and then training.
Rows are timesteps, columns are features, going in the following order:
0: Inlet temperature (K)
1: Outlet pressure (Pa)
2: Inlet flowrate (mol/s)

3-11: Inlet mole fraction of 9 species (Ar, CH4, CO, CO2, H2, H2O, He, N2, O2)

12-20: Temperature at 10% to 90% of the bed length (K)
21: Temperature at the outlet of the bed (K)

22-30: Pressure drop at 10% to 90% of the bed length (Pa)
31: Pressure drop at the outlet of the bed (Pa)

32: Flowrate at the outlet of the bed (mol/s)

33-41: Outlet mole fraction of 9 species (Ar, CH4, CO, CO2, H2, H2O, He, N2, O2)
'''
def interp_rows(source_x, values, target_x):
    return np.vstack([
        np.interp(target_x, source_x, row)
        for row in values
    ])

def rescale_matrix(arr):
    scaled = copy.deepcopy(arr)
    scaled[:, 0] /= 1e3 # temperature
    scaled[:, 1] /= 1e6 # pressure
    scaled[:, 2] /= 2e-3 # flowrate

    scaled[:, 12:22] /= 1e3 # temperature
    scaled[:, 22:32] /= 1e3 # pressure drop
    scaled[:, 32] /= 2e-3 # flowrate

    return scaled

SPECIES = ("Ar", "CH4", "CO", "CO2", "H2", "H2O", "He", "N2", "O2")
FRACTIONS = np.arange(0.1, 1.0, 0.1)
BED_LENGTH = 0.4
SAMPLE_X = BED_LENGTH * FRACTIONS

print(SAMPLE_X)

ALL_CASES = r"C:\MyRepos\multisolid-CL\packed_bed\examples\opt5_batch_case\output\cases"
OUTPUT_DIR = r"C:\MyRepos\multisolid-CL\active_learning_optimization\extracted_opt5"

for folder in os.listdir(ALL_CASES):

    reports = pd.read_pickle(os.path.join(ALL_CASES, folder, "output", "reports.pkl")).iloc[1:]
    gas = pd.read_pickle(os.path.join(ALL_CASES, folder, "output", "gas_mole_fraction.pkl")).iloc[1:]

    assert len(reports) == 12000, f"{folder}: has {len(reports)} reports, expected 12000"
    assert len(gas) == 12000, f"{folder}: has {len(gas)} gas mole fraction entries, expected 12000"

    out = np.empty((12000, 42), dtype=np.float64)

    out[:, 0] = reports[("inlet_temperature_k", 0.0)]
    out[:, 1] = reports[("outlet_pressure_pa", BED_LENGTH)]
    out[:, 2] = reports[("inlet_flowrate_mol_s", 0.0)]

    out[:, 3:12] = np.column_stack([gas[(s, 0.0)] for s in SPECIES])

    temp = reports.xs("temperature_k", axis=1, level="feature")
    temp_x = temp.columns.to_numpy(dtype=float)
    temp_values = temp.to_numpy(dtype=float)

    out[:, 12:21] = interp_rows(temp_x, temp_values, SAMPLE_X)
    out[:, 21] = interp_rows(temp_x, temp_values, [BED_LENGTH])[:, 0]
    
    pressure = reports.xs("pressure_pa", axis=1, level="feature")
    p_x = pressure.columns.to_numpy(dtype=float)
    p_values = pressure.to_numpy(dtype=float)

    p_in = reports[("inlet_pressure_pa", 0.0)].to_numpy(dtype=float)
    p_out = reports[("outlet_pressure_pa", BED_LENGTH)].to_numpy(dtype=float)

    full_p_x = np.r_[0.0, p_x, BED_LENGTH]
    full_p_values = np.column_stack([p_in, p_values, p_out])
    p_at_x = interp_rows(full_p_x, full_p_values, SAMPLE_X)

    out[:, 22:31] = p_in[:, None] - p_at_x
    out[:, 31] = p_in - p_out

    out[:, 32] = reports[("outlet_flowrate_mol_s", BED_LENGTH)]
    out[:, 33:42] = np.column_stack([gas[(s, BED_LENGTH)] for s in SPECIES])

    if out.shape[1] != 42 or np.isnan(out).any():
        raise ValueError(f"{folder}: invalid extracted array {out.shape}")

    np.save(os.path.join(OUTPUT_DIR, f"{folder}.npy"), out)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(rescale_matrix(out), interpolation="none")
    ax.set_aspect(0.005)
    fig.suptitle(folder)
    plt.show()


