import numpy as np
import matplotlib.pyplot as plt

# Define the system of ODEs for the spread of mold
def mold_model(t, y, beta0, gamma, delta, xi):
    S, I, R, H = y  
    beta = beta0 * H  
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    dH_dt = -delta * H + xi
    return [dS_dt, dI_dt, dR_dt, dH_dt]

# Parameters
beta0 = 0.4  
gamma = 0.1  
delta = 0.05 
xi = 0.01    

# Initial conditions
S0 = 0.9
I0 = 0.1
R0 = 0.0
H0 = 1.0
y0 = [S0, I0, R0, H0]

# Time setup
t_start, t_end, dt = 0, 50, 0.1
steps = int((t_end - t_start) / dt)

# Arrays to store results
t = np.linspace(t_start, t_end, steps)
S, I, R, H = np.zeros(steps), np.zeros(steps), np.zeros(steps), np.zeros(steps)

# Initialize
S[0], I[0], R[0], H[0] = S0, I0, R0, H0

# Euler method
for i in range(1, steps):
    dS, dI, dR, dH = mold_model(t[i-1], [S[i-1], I[i-1], R[i-1], H[i-1]], beta0, gamma, delta, xi)
    S[i], I[i], R[i], H[i] = S[i-1] + dS * dt, I[i-1] + dI * dt, R[i-1] + dR * dt, H[i-1] + dH * dt

# Plot
plt.figure(figsize=(12, 8))
plt.plot(t, S, label="Susceptible (Clean surfaces)", color="blue")
plt.plot(t, I, label="Infected (Mold-covered)", color="red")
plt.plot(t, R, label="Recovered (Treated/exhausted)", color="green")
plt.plot(t, H, label="Humidity (Environmental factor)", color="purple", linestyle="--")
plt.xlabel("Time (days)")
plt.ylabel("Proportion / Humidity Level")
plt.title("Spread of Mold with Environmental Factor (4-Equation Model)")
plt.legend()
plt.grid()
plt.show()
