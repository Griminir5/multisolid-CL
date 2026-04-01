import numpy as np
import matplotlib.pyplot as plt

T_data = np.array([300,
360,
420,
480,
540,
600,
660,
720,
780,
840,
900,
960,
1020,
1080,
1140,
1200,
1260,
1320,
1380,
1440,
1500,], dtype=float) # K

h_data = np.array([48.2536,
1658.47,
3353.76,
5125.58,
6998.52,
9008.39,
11078.9,
12942.2,
14796.6,
16668.6,
18569.3,
20505.8,
22478,
24484.7,
26526.1,
28602.3,
30711.8,
32850.6,
35011.2,
37182.1,
39353.6,], dtype=float) # J/mol

cp_data = np.array([26.0996,
27.604,
28.8683,
30.2697,
32.2707,
34.7735,
31.5529,
30.8363,
31.0179,
31.4127,
31.9641,
32.5784,
33.1572,
33.7333,
34.315,
34.8877,
35.4176,
35.8544,
36.1349,
36.1916,
36.1916,], dtype=float) # J/(K*mol)

h_ref = 0 # J/mol
t_ref = 298.15 # K

h_weight = 1
cp_weight = 1

def fit_enthcp_poly(T_data, h_data, cp_data, t_ref, href, h_weight, cp_weight, order: int):
    """
    Enthalpy correlation defined by a Cp polynomial about t_ref.

    cp(T) = a0 + a1*dT + a2*dT^2 + a3*dT^3 + a4*dT^4
    h(T)  = h_ref + integral(cp dT)
    h(T)  = h_ref + a0*dT+ 0.50*a1*dT^2 + 0.33*a2*dT^3 + 0.25*a3*dT^4 + 0.20*a4*dT^5
    where dT = T - t_ref
    """
    if order < 0:
        raise ValueError('Polynomial order too low')
    
    dT = T_data - t_ref
    h_scale = max(np.std(h_data), 1.0)
    cp_scale = max(np.std(cp_data), 1.0)
    
    A_cp = np.column_stack([dT**i for i in range(order + 1)])
    b_cp = cp_data

    A_h  = np.column_stack([dT**(i + 1)/(i + 1) for i in range(order + 1)])
    b_h = h_data - href

    A_cp_scaled = (cp_weight / cp_scale) * A_cp
    b_cp_scaled = (cp_weight / cp_scale) * b_cp

    A_h_scaled = (h_weight / h_scale) * A_h
    b_h_scaled = (h_weight / h_scale) * b_h

    A = np.vstack([A_cp_scaled, A_h_scaled])
    b = np.concatenate([b_cp_scaled, b_h_scaled])

    theta, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    print('fitted parameters = ',theta)
    print('residual = ',residuals)
    print('rank = ',rank)
    print('singular values = ',s)
    print('condition number = ',s[0]/s[-1])

    return theta


def cp_poly_model(T, params):
    dT = T - t_ref

    res = np.zeros_like(dT)
    for i, param in enumerate(params):
        res += param*dT**i

    return res

def h_poly_model(T, params):
    dT = T - t_ref

    res = np.zeros_like(dT) + h_ref
    for i, param in enumerate(params):
        res += param*dT**(i+1)/(i+1)

    return res


params = fit_enthcp_poly(T_data, h_data, cp_data, t_ref, h_ref, h_weight, cp_weight, 1)

cp_calc = cp_poly_model(T_data, params)
h_calc = h_poly_model(T_data, params)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

ax1.scatter(T_data, cp_data, label = "Original")
ax1.plot(T_data, cp_calc, label = "Fitted")


ax2.scatter(T_data, h_data, label = "Original")
ax2.plot(T_data, h_calc, label = "Fitted")

plt.show()