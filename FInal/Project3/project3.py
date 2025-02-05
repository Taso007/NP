import numpy as np
import matplotlib.pyplot as plt
import math

# Euler method for solving ODEs
def euler_method(f, t_span, y0, args, h=0.01):
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        y_values[i] = y_values[i - 1] + h * np.array(f(t_values[i - 1], y_values[i - 1], *args))
    
    return t_values, y_values

# Taylor series expansion for singularity handling
def taylor_series_central(f, x, n=1, h=1e-5):
    def nth_derivative(f, x, h, order):
        if order == 1:
            return (f(x + h) - f(x - h)) / (2 * h)
        return (nth_derivative(f, x + h, h, order - 1) - nth_derivative(f, x - h, h, order - 1)) / (2 * h)

    taylor_approximation = f(x)
    for i in range(1, n + 1):
        derivative = nth_derivative(f, x, h, i)
        taylor_approximation += (derivative * (h**i)) / math.factorial(i)
    return taylor_approximation

# Shooting method implementation
def shooting_method(px, qx, wx, a, b, max_eigenvalues=8, h=0.01):
    def first_order_system(t, y, lambda_val):
        u, v = y
        du_dt = v
        dv_dt = (p_prime(t) * v - qx(t) * u + lambda_val * wx(t) * u) / px(t)
        return [du_dt, dv_dt]

    lambdas = []
    eigenfunctions = []
    lambda_guess = 0  # Initial guess for lambda

    while len(lambdas) < max_eigenvalues:
        # Solve for u1
        t1, sol1 = euler_method(first_order_system, [a, b], [0, 1], [lambda_guess], h)
        u1_b = sol1[-1, 0]  # Value of u at x = b

        # Check the boundary condition at x = b
        if abs(u1_b) < 1e-6:  # Converged eigenvalue
            lambdas.append(lambda_guess)
            eigenfunctions.append(lambda x, sol=sol1: np.interp(x, t1, sol[:, 0]))
            lambda_guess += 1  # Increment for the next eigenvalue
        else:
            lambda_guess += 0.1  # Adjust lambda_guess

    return lambdas, eigenfunctions

# Define p(x), q(x), w(x), and boundaries with Taylor handling
def px(x):
    cos_x = np.cos(x)
    if np.isclose(cos_x, 0):  # Handle singularity
        cos_x = taylor_series_central(np.cos, x, n=1)
    return 0.5 * cos_x**4

def p_prime(x):
    return -2 * np.cos(x)**3 * np.sin(x)

def qx(x, m=100):
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    if np.isclose(sin_x, 0):  # Handle singularity for sin(x)
        sin_x = taylor_series_central(np.sin, x, n=1)
    if np.isclose(cos_x, 0):  # Handle singularity for cos(x)
        cos_x = taylor_series_central(np.cos, x, n=1)
    return (m**2 * cos_x**2) / (2 * sin_x**2) - cos_x / sin_x

def wx(x):
    return 1

# Define boundaries
a, b = 0, np.pi / 2  # Boundary values

# Compute eigenvalues and eigenfunctions
lambdas, eigenfunctions = shooting_method(px, qx, wx, a, b, max_eigenvalues=8, h=0.01)

# Plot the first 8 eigenfunctions
x_vals = np.linspace(a, b, 500)
plt.figure(figsize=(12, 8))
for i, eigenfunc in enumerate(eigenfunctions):
    y_vals = eigenfunc(x_vals)
    plt.plot(x_vals, y_vals, label=f"Eigenfunction {i + 1} (λ = {lambdas[i]:.2f})")
plt.xlabel("x")
plt.ylabel("Eigenfunction")
plt.title("First 8 Eigenfunctions (Boundary Conditions u(0) = 0, u(π/2) = 0)")
plt.legend()
plt.grid(True)
plt.show()
