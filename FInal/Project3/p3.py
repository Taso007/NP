import numpy as np
import matplotlib.pyplot as plt
import math

# Functions for p(x), q(x), and w(x)
def p(x):
    cos_x = np.cos(x)
    if np.isclose(cos_x, 0):
        cos_x = taylor_series_central(np.cos, x, 1)
    return 0.5 * cos_x**4

def p_prime(x):
    return -2 * np.cos(x)**3 * np.sin(x)

def q(x, m=100):
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    if np.isclose(sin_x, 0):
        sin_x = taylor_series_central(np.sin, x, 1)
    if np.isclose(cos_x, 0):
        cos_x = taylor_series_central(np.cos, x, 1)
    return (m**2 * cos_x**2) / (2 * sin_x**2) - cos_x / sin_x

def w(x):
    return 1

# First-order system
def first_order_system(x, u, v, lam):
    u_prime = v
    v_prime = (p_prime(x) * v - q(x) * u + lam * w(x) * u)/p(x)
    return u_prime, v_prime

# Euler's method
def euler_method(u0, v0, x_start, x_end, h, lam):
    x_values = np.arange(x_start, x_end + h, h)
    u_values = np.zeros(len(x_values))
    v_values = np.zeros(len(x_values))
    
    u_values[0], v_values[0] = u0, v0
    
    for i in range(len(x_values) - 1):
        u_prime, v_prime = first_order_system(x_values[i], u_values[i], v_values[i], lam)
        u_values[i+1] = u_values[i] + h * u_prime
        v_values[i+1] = v_values[i] + h * v_prime
    
    return x_values, u_values, v_values

# Central difference method to approximate derivative
def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# Taylor series expansion for singularity handling
def taylor_series_central(f, x, n, h=1e-5):
    def nth_derivative(f, x, h, order):
        if order == 1:
            return central_difference(f, x, h)
        return (nth_derivative(f, x + h, h, order - 1) - nth_derivative(f, x - h, h, order - 1)) / (2 * h)

    taylor_approximation = f(x)
    for i in range(1, n + 1):
        derivative = nth_derivative(f, x, h, i)
        taylor_approximation += (derivative * (h**i)) / math.factorial(i)
    return taylor_approximation

# Shooting method to solve for eigenvalues and eigenfunctions
def shooting_method(x_start, x_end, h, lam_start, lam_end, tol):
    lam_values = np.linspace(lam_start, lam_end, 1000)
    eigenvalues = []
    eigenfunctions = []

    for lam in lam_values:
        # Solve first IVP: u1(0) = 0, v1(0) = 1
        x_vals, u1, _ = euler_method(0, 1, x_start, x_end, h, lam)
        
        # Check the boundary condition at x = pi/2
        if abs(u1[-1]) < tol:
            eigenvalues.append(lam)
            eigenfunctions.append((x_vals, u1))
        
        # Stop after finding 8 eigenvalues
        if len(eigenvalues) == 8:
            break
    
    return eigenvalues, eigenfunctions

# Parameters
x_start = 0
x_end = np.pi / 2
h = 0.01
lam_start = 0  # Start of lambda search range
lam_end = 100  # End of lambda search range
tol = 1e-6

# Find eigenvalues and eigenfunctions
eigenvalues, eigenfunctions = shooting_method(x_start, x_end, h, lam_start, lam_end, tol)

# Display eigenvalues
print("Eigenvalues:")
for i, ev in enumerate(eigenvalues, 1):
    print(f"λ{i} = {ev:.6f}")

# Plot the first 8 eigenfunctions
plt.figure(figsize=(10, 8))
for i, (x_vals, u_vals) in enumerate(eigenfunctions):
    if len(x_vals) > 0 and len(u_vals) > 0:  # Ensure data exists before plotting
        plt.plot(x_vals, u_vals, label=f"Eigenfunction {i+1} (λ = {eigenvalues[i]:.6f})")

plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("First 8 Eigenfunctions")
plt.legend()  # Ensure labels are added to the legend
plt.grid()
plt.show()
