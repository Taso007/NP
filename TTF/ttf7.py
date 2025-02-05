import numpy as np
import matplotlib.pyplot as plt

# Function to interpolate
def f(x):
    return 1 / (1 + x**2)

# Generate Chebyshev points in the interval [-5, 5)
def generate_chebyshev_points(n, a=-5, b=5):
    chebyshev_nodes = np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))
    return (b - a) * 0.5  * (chebyshev_nodes + 1) + a 

# linear interpolation 
def piecewise_linear_interpolation(x, y, x_test):
    y_interp = []
    for xt in x_test:
        if xt < x[0]:
            y_interp.append(y[0])
        elif xt > x[-1]:
            y_interp.append(y[-1])
        else:
            for i in range(len(x) - 1):
                if x[i] <= xt < x[i + 1]:
                    slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                    intercept = y[i] - slope * x[i]
                    y_interp.append(slope * xt + intercept)
                    break
    return np.array(y_interp)

# Piecewise polynomial interpolation
def piecewise_polynomial_interpolation(x, y, x_test):
    y_interp = []
    for xt in x_test:
        for i in range(0, len(x) - 2, 2):
            if x[i] <= xt < x[i + 2]:
                coeffs = np.polyfit(x[i:i + 3], y[i:i + 3], 2)
                y_interp.append(np.polyval(coeffs, xt))
                break
    return np.array(y_interp)

# Cubic spline interpolation
def cubic_spline_interpolation(x, y, x_test):
    n = len(x) - 1
    h = np.diff(x)
    alpha = [0] + [(3 / h[i] * (y[i + 1] - y[i]) - 3 / h[i - 1] * (y[i] - y[i - 1])) for i in range(1, n)]

    l = np.ones(n + 1)
    mu = np.zeros(n)
    z = np.zeros(n + 1)

    for i in range(1, n):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    b = np.zeros(n)
    c = np.zeros(n + 1)
    d = np.zeros(n)

    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    def spline_eval(xt):
        for i in range(n):
            if x[i] <= xt < x[i + 1]:
                dx = xt - x[i]
                return y[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
        return y[-1]

    return np.array([spline_eval(xt) for xt in x_test])

n = 15  
a, b = -5, 5
chebyshev_points = generate_chebyshev_points(n, a, b)
y_values = f(chebyshev_points)

x_test = np.linspace(a, b, 1000)
y_actual = f(x_test)

y_linear = piecewise_linear_interpolation(chebyshev_points, y_values, x_test)
y_poly = piecewise_polynomial_interpolation(chebyshev_points, y_values, x_test)
y_spline = cubic_spline_interpolation(chebyshev_points, y_values, x_test)

mse_linear = np.mean((y_actual - y_linear)**2)
mse_poly = np.mean((y_actual[:len(y_poly)] - y_poly)**2)
mse_spline = np.mean((y_actual - y_spline)**2)

print(f'MSE of Piecewise Linear Interpolation: {mse_linear:.6f}')
print(f'MSE of Piecewise Polynomial Interpolation: {mse_poly:.6f}')
print(f'MSE of Cubic Spline Interpolation: {mse_spline:.6f}')

errors = {'Linear': mse_linear, 'Polynomial': mse_poly, 'Cubic Spline': mse_spline}
best_method = min(errors, key=errors.get)
print(f'The method with the smallest error is: {best_method}')

# Plot the results  
plt.figure(figsize=(14, 8))
plt.plot(x_test, y_actual, label='Actual f(x)', color='black', linewidth=2)
plt.plot(x_test, y_linear, label='Piecewise Linear', linestyle='--')
plt.plot(x_test[:len(y_poly)], y_poly, label='Piecewise Polynomial', linestyle='-.')
plt.plot(x_test, y_spline, label='Cubic Spline', linestyle=':')
plt.scatter(chebyshev_points, y_values, color='red', label='Chebyshev Points')
plt.legend()
plt.title('Interpolation Comparison')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
