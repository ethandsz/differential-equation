
import numpy as np
import matplotlib.pyplot as plt

# Custom function
def f(t):
    return t**2

# Parameters
L = np.pi
N = 10                          # Number of Fourier terms
t = np.linspace(-L, L, 1000)    # Time vector
dt = t[1] - t[0]                # Time step for integration

# Compute a0
a0 = (1 / L) * np.sum(f(t) * dt)

print(a0)
# Compute an and bn numerically
a_n = []
b_n = []
for n in range(1, N + 1):
    an = (1 / L) * np.sum(f(t) * np.cos(n * np.pi * t / L) * dt)
    bn = (1 / L) * np.sum(f(t) * np.sin(n * np.pi * t / L) * dt)
    a_n.append(an)
    b_n.append(bn)

# Build Fourier series approximation
f_series = np.full_like(t, a0 / 2)
for n in range(1, N + 1):
    f_series += a_n[n - 1] * np.cos(n * np.pi * t / L) + b_n[n - 1] * np.sin(n * np.pi * t / L)

# Plotting
plt.plot(t, f(t), label='Original $f(t) = t^2$')
plt.plot(t, f_series, '--', label=f'Fourier Approx (N={N})')
plt.title('Fourier Series Approximation (No SciPy)')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid(True)
plt.legend()
plt.show()
