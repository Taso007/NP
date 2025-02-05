import numpy as np
import matplotlib.pyplot as plt
import math

x_range = (0, np.pi / 2)  # This is part of the boundary conditions
x_start, x_end = x_range
boundary_conditions = (0, 0)  # y(0) = 0, y(pi/2) = 0
m = 1

h = 0.01  
x_values = np.arange(x_start, x_end + h, h)  # Generate x values with step size h

def compute_y2_prime(x, u, y2, m):
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    cos_2x = np.cos(2 * x)
    
    if np.isclose(sin_x, 0):
        sin_x = taylor_series_central(np.sin, x, 1)
    if np.isclose(cos_x, 0):
        cos_x = taylor_series_central(np.cos, x, 1)
    
    term1 = -x + (m**2 * cos_x**2 / (2 * sin_x**2)) - cos_x / sin_x
    term2 = (cos_x**3 * cos_2x) / 2 * sin_x

    y2_prime = (2 / cos_x**4) * (term1 * u - term2 * y2)
    return y2_prime

def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def newtons_method(initial_guess, f, tol=1e-6, max_iter=100):
    x = initial_guess
    for _ in range(max_iter):
        f_val = f(x)
        df_val = central_difference(f, x)
        if np.abs(f_val) < tol:
            return x
        x -= f_val / df_val
    return x

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

def shooting_method_newton(y2_prime_func, boundary_conditions, m, u_guess, tol=1e-6, max_iter=100):
    y_start, y_end = boundary_conditions

    def solve_ivp(u):
        y_values = np.zeros_like(x_values)
        y2_values = np.zeros_like(x_values)

        y_values[0] = y_start
        y2_values[0] = u

        for i in range(1, len(x_values)):
            y2_prime = y2_prime_func(x_values[i - 1], u, y2_values[i - 1], m)
            y2_values[i] = y2_values[i - 1] + h * y2_prime
            y_values[i] = y_values[i - 1] + h * y2_values[i - 1]

        return y_values, y_values[-1] - y_end

    def boundary_residual(u):
        _, residual = solve_ivp(u)
        return residual

    u_corrected = newtons_method(u_guess, boundary_residual, tol, max_iter)
    y_values, _ = solve_ivp(u_corrected)
    return y_values, u_corrected

def find_eigenvalues_and_eigenfunctions(boundary_conditions, m, n_eigenvalues=8):
    eigenvalues = []
    eigenfunctions = []

    for n in range(1, n_eigenvalues + 1):
        u_guess = n / 100
        y_values, u_correct = shooting_method_newton(compute_y2_prime, boundary_conditions, m, u_guess)
        eigenvalues.append(u_correct)
        eigenfunctions.append(y_values)  

    return eigenvalues, eigenfunctions

eigenvalues, eigenfunctions = find_eigenvalues_and_eigenfunctions(boundary_conditions, m)

sorted_indices = np.argsort(eigenvalues)
eigenvalues = [eigenvalues[i] for i in sorted_indices]
eigenfunctions = [eigenfunctions[i] for i in sorted_indices]


plt.figure(figsize=(10, 6))
for i, (eigenvalue, y_values) in enumerate(zip(eigenvalues, eigenfunctions), 1):
    print(f"Eigenvalue {i}: {eigenvalue}")
    plt.plot(x_values, y_values, label=f"Eigenvalue {eigenvalue}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Eigenvalues and Eigenfunctions")
plt.legend()
plt.grid()
plt.show()
