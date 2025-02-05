import numpy as np
import matplotlib.pyplot as plt

# Parameters for the ODE
r = 0.05 
I = 100  
C0 = 0    
t = 100   
h = 0.1   

# Time points
t = np.arange(0, t + h, h)
C = np.zeros(len(t))
C[0] = C0

# Euler's method to solve the ODE
for i in range(1, len(t)):
    C[i] = C[i-1] + h * (r * C[i-1] - I)

# Plotting the result
plt.figure(figsize=(10, 6))
plt.plot(t, C, label="C(t) using Euler's Method")
plt.axhline(I/r, color='red', linestyle='--', label=f"Steady State C = {I/r}")
plt.title("Investment-Consumption Model: Numerical Solution")
plt.xlabel("Time (t)")
plt.ylabel("Consumption (C)")
plt.legend()
plt.grid()
plt.show()

## If r > 0, the exponential term e^rt grows indefinetly unless C is properly balanced,
## solution depends on whether r>0, r=0, or r<0.