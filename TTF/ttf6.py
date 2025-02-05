#given the data points: 
# (x0, y0)= (1, 2), (x1, y1)= (2, 3), (x2, y2)= (3, 5), (x3, y3)= (4, 4)
#compute lagrange polynomial and plot it together with the data points


import numpy as np 
import matplotlib.pyplot as plt

x_points = np.array([1, 2, 3, 4])
y_points = np.array([2, 3, 5, 4])

def lagrange_polynomial(x, x_points, y_points) :
  n = len(x_points)
  p = 0
  for i in range(n) :
    l = 1
    for j in range(n) :
      if i != j :
        l *= (x - x_points[j]) / (x_points[i] - x_points[j])
    p += y_points[i] * l
  return p

x_values = np.linspace(min(x_points) - 1, max(x_points) + 1, 100)
y_values = [lagrange_polynomial(x, x_points, y_points) for x in x_values]


plt.plot(x_values, y_values, label="Lagrange Polynomial", color="blue")
plt.scatter(x_points, y_points, color="black", label="Data Points")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Lagrange Polynomial Interpolation")
plt.legend()
plt.grid(True)
plt.show()