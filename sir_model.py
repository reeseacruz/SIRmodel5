import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- 1. Define the SIR Model Equations ---
def sir_model(y, t, beta, gamma):
    """
    Returns the derivatives for the SIR model.
    y: [S, I, R] at time t
    beta: transmission rate
    gamma: recovery rate
    """
    S, I, R = y
    N = S + I + R # Total population, assumed constant
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# --- 2. Set Parameters and Initial Conditions ---
# Parameters
beta = 0.3   # Average number of contacts per person per time
gamma = 1/5  # Recovery rate (1/5 means average infectious period of 5 days)

# Initial conditions: 999 susceptible, 1 infected, 0 recovered
N = 1000
S0 = N - 1
I0 = 1
R0 = 0
initial_conditions = [S0, I0, R0]

# Time vector (simulate for 160 days)
t = np.linspace(0, 160, 160)

# --- 3. Solve the ODEs ---
# The odeint function integrates the equations
solution = odeint(sir_model, initial_conditions, t, args=(beta, gamma))

# Extract the results
S, I, R = solution.T

# --- 4. Plot the Results ---
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible', linewidth=2)
plt.plot(t, I, label='Infected', linewidth=2)
plt.plot(t, R, label='Recovered', linewidth=2)
plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.title('SIR Model Simulation')
plt.legend()
plt.grid(True)
plt.show()
