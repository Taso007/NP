import numpy as np
import matplotlib.pyplot as plt


# exercise 1

def f(x):
   return 3*x**3 - 2*x**2 + x

def forward_difference(f, x, h) :
   return (f(x + h) - f(x))/h

def backward_difference(f, x, h) :
   return (f(x) - f( x - h))/h

def central_difference(f, x, h) :
   return (f(x + h) - f(x - h))/(2 * h)

def exact_derivative(x):
   return 9 * x**2 - 4 * x + 1

# exercise 2

h = 0.1
h_input = input("Enter the step size h, or it will be default and equal to 0.1 : ")

if(h_input) :
   h = float(h_input)
else :
   print("h is default and is equal to 0.1")

# exercise 3

x_values = np.linspace(-2, 2, 100)

forward_diff = [forward_difference(f, x, h) for x in x_values]
backward_diff = [backward_difference(f, x, h) for x in x_values]
central_diff = [central_difference(f, x, h) for x in x_values]
exact_diff = [exact_derivative(x) for x in x_values]

plt.figure(figsize=(10, 6))

plt.plot(x_values, forward_diff, label="Forward Difference")
plt.plot(x_values, backward_diff, label="Backward Difference")
plt.plot(x_values, central_diff, label="Central Difference")
plt.plot(x_values, exact_diff, label="Exact Derivative")

plt.title(f"Derivative Approximations (h = {h})")
plt.xlabel("x")
plt.ylabel("Derivative")
plt.legend()
plt.grid(True)
plt.show()
