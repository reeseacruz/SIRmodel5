import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def evolutionary_sid_model(y, t, beta, mu, nu):
    # y is now a 5-element vector: [S, I1, I2, D1, D2]
    S, I1, I2, D1, D2 = y
    
    # Strain 1 (Slow Replication) parameters
    beta1, mu1 = beta[0], mu[0]
    
    # Strain 2 (Fast Replication) parameters
    beta2, mu2 = beta[1], mu[1]
    
    # Force of infection for each strain
    force1 = beta1 * S * I1
    force2 = beta2 * S * I2
    
    # Differential Equations
    dSdt = -(force1 + force2) + nu * S
    dI1dt = force1 - mu1 * I1
    dI2dt = force2 - mu2 * I2
    dD1dt = mu1 * I1
    dD2dt = mu2 * I2
    
    return [dSdt, dI1dt, dI2dt, dD1dt, dD2dt]

# Parameters
nu = 0.01 # Regeneration rate (P[S] = vS)

# Viral Replication Trade-offs
# Fast strain (I2) has higher transmission (beta) but also higher death rate (mu)
beta_slow, mu_slow = 0.002, 0.01   # Low replication: low transmission, low death
beta_fast, mu_fast = 0.004, 0.05   # High replication: high transmission, high death

beta = [beta_slow, beta_fast]
mu = [mu_slow, mu_fast]

# Initial Conditions
N = 100
S0 = 99
I1_0 = 1 # Start with the slow strain
I2_0 = 0 # No fast strain yet
D1_0 = 0
D2_0 = 0
initial_conditions = [S0, I1_0, I2_0, D1_0, D2_0]

t = np.linspace(0, 150, 150)

# Solve ODEs
solution = odeint(evolutionary_sid_model, initial_conditions, t, args=(beta, mu, nu))
S, I1, I2, D1, D2 = solution.T

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(t, S, label='Susceptible', linewidth=2, color='blue')
plt.plot(t, I1, label='Infected (Slow Strain)', linewidth=2, color='green')
plt.plot(t, I2, label='Infected (Fast Mutant)', linewidth=2, color='orange')
plt.plot(t, D1, label='Deaths (Slow)', linewidth=1, color='green', linestyle='--')
plt.plot(t, D2, label='Deaths (Fast)', linewidth=1, color='red', linestyle='--')

plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.title('Evolutionary Dynamics: Slow vs. Fast Replicating Strains')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('evolutionary_sid_output.png', dpi=150, bbox_inches='tight')
plt.show()
