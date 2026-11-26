import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sid_model(y, t, beta, mu, nu):
    S, I, D = y
    dSdt = -beta * S * I + nu * S
    dIdt = beta * S * I - mu * I
    dDdt = mu * I
    return [dSdt, dIdt, dDdt]

beta = 0.003
mu = 0.02
nu = 0.01

N = 100
S0 = 99
I0 = 1
D0 = 0
initial_conditions = [S0, I0, D0]

t = np.linspace(0, 200, 200)

solution = odeint(sid_model, initial_conditions, t, args=(beta, mu, nu))
S, I, D = solution.T

plt.figure(figsize=(12, 6))
plt.plot(t, S, label='Susceptible', linewidth=2, color='blue')
plt.plot(t, I, label='Infected', linewidth=2, color='orange')
plt.plot(t, D, label='Deaths', linewidth=2, color='red', linestyle='--')
plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.title('SID Model Simulation with P[S] = νS')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sid_model_output.png', dpi=150, bbox_inches='tight')
print("Plot saved to sid_model_output.png")

peak_infected = np.max(I)
peak_day = t[np.argmax(I)]
final_deaths = D[-1]
final_susceptible = S[-1]
total_population = S[-1] + I[-1] + D[-1]

print("\n=== Model Summary ===")
print(f"Peak infected: {peak_infected:.1f} people on day {peak_day:.1f}")
print(f"Final deaths: {final_deaths:.1f} people")
print(f"Final susceptible: {final_susceptible:.1f} people")
print(f"Total population (final): {total_population:.1f}")
