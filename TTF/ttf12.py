import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x**5 - 2*x**4 + 3*x**3 - 4*x**2 + 5*x - 6

def df(x):
    return 5*x**4 - 8*x**3 + 9*x**2 - 8*x + 5

def newtons_method(f, df, x0, tolerance, max_iter=100):
    x_vals = [x0]  
    for _ in range(max_iter):
        x1 = x0 - f(x0) / df(x0)  
        x_vals.append(x1)
        if abs(x1 - x0) < tolerance:  
            break
        x0 = x1
    return x1, x_vals

x0 = 1.5
tolerance = 1e-6

root, approximations = newtons_method(f, df, x0, tolerance)

print("Root:", root)
print("Approximations:", approximations)

x = np.linspace(0, 2, 400)  
y = f(x)

plt.figure(figsize=(10, 6))

plt.plot(x, y, label="f(x)", color="blue")

plt.scatter(root, f(root), color="red", label=f"Root: {root:.6f}")

for i, xi in enumerate(approximations[:5]): 
    plt.scatter(xi, f(xi), label=f"x{i} = {xi:.6f}")

plt.axhline(0, color="black", linestyle="--") 
plt.title("Newton's Method Visualization")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.show()
