import numpy as np
import matplotlib.pyplot as plt

# driving function
def driving_function(t):
    return 2 * t

# Derivative function
def dg_dt(t, g):
    return driving_function(t) 

# Runge-Kutta 4th order method
def runge_kutta_4(g0, t0, t_end, N):
    h = (t_end - t0) / N  # Step size
    t_values = np.linspace(t0, t_end, N + 1)
    g_values = np.zeros(N + 1, dtype=complex)
    
    g_values[0] = g0
    
    for i in range(N):
        t = t_values[i]
        g = g_values[i]
        
        k1 = h * dg_dt(t, g)
        k2 = h * dg_dt(t + h / 2, g + k1 / 2)
        k3 = h * dg_dt(t + h / 2, g + k2 / 2)
        k4 = h * dg_dt(t + h, g + k3)
        
        g_values[i + 1] = g + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    return t_values, g_values

# Initial conditions
t0 = 0
g0 = 1 + 1j  

# Time interval and step size
t_end = 1
N = 100  

# call RK4
time, g_values = runge_kutta_4(g0, t0, t_end, N)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(time, np.abs(g_values), label='|g(t)|')
plt.title('Solution of the Loewner Differential Equation using RK4')
plt.xlabel('Time t')
plt.ylabel('|g(t)|')
plt.legend()
plt.grid(True)
plt.show()
