from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt

POWER = 0.9
DOMAIN = np.concat((np.linspace(0.0, 0.1, 100), np.linspace(0.1, 0.99, 1000),np.linspace(0.9, 1, 100)))
TARGET = DOMAIN**POWER

print(DOMAIN.shape)

def approx(params, x):
    a, b, c, d = params
    return a * x / (1 + b * np.abs(x)) + c * x / (1 + d * np.abs(x))

def residual(params):
    return TARGET - approx(params, DOMAIN)


result = least_squares(
    residual,
    x0=[1.0, 1.0, 1.0, 10.0],
    bounds=([-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
)

print(result.x)
print("max abs error:", np.max(np.abs(residual(result.x))))
print("rms error:", np.sqrt(np.mean(residual(result.x) ** 2)))

plt.plot(DOMAIN, TARGET)
plt.plot(DOMAIN, approx(result.x, DOMAIN))
plt.show()