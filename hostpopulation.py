import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def metapopulation_model(y, t, beta, mu, nu, m):
    # Unpack 10 variables (S, I1, I2, D1, D2 for Patch A and Patch B)
    Sa, I1a, I2a, D1a, D2a, Sb, I1b, I2b, D1b, D2b = y
    
    beta1, mu1 = beta[0], mu[0]
    beta2, mu2 = beta[1], mu[1]
    
    # Patch A Dynamics
    force1a = beta1 * Sa * I1a
    force2a = beta2 * Sa * I2a
    dSadt = -(force1a + force2a) + nu * Sa + (m * Sb) - (m * Sa)
    dI1adt = force1a - mu1 * I1a + (m * I1b) - (m * I1a)
    dI2adt = force2a - mu2 * I2a + (m * I2b) - (m * I2a)
    dD1adt = mu1 * I1a
    dD2adt = mu2 * I2a
    
    # Patch B Dynamics
    force1b = beta1 * Sb * I1b
    force2b = beta2 * Sb * I2b
    dSbdt = -(force1b + force2b) + nu * Sb + (m * Sa) - (m * Sb)
    dI1bdt = force1b - mu1 * I1b + (m * I1a) - (m * I1b)
    dI2bdt = force2b - mu2 * I2b + (m * I2a) - (m * I2b)
    dD1bdt = mu1 * I1b
    dD2bdt = mu2 * I2b
    
    return [dSadt, dI1adt, dI2adt, dD1adt, dD2adt, dSbdt, dI1bdt, dI2bdt, dD1bdt, dD2bdt]

# Parameters
nu = 0.03       # Birth rate
m = 0.01        # Migration rate (1% of people move between patches daily)

beta_slow, mu_slow = 0.0005, 0.01
beta_fast, mu_fast = 0.0015, 0.02
beta = [beta_slow, beta_fast]
mu = [mu_slow, mu_fast]

# Initial Conditions
# Patch A starts with the outbreak. Patch B starts completely healthy.
initial_conditions = [
    499, 1.0, 0.1, 0, 0,   # Patch A (500 people total)
    500, 0.0, 0.0, 0, 0    # Patch B (500 people total)
]

t = np.linspace(0, 150, 150)
solution = odeint(metapopulation_model, initial_conditions, t, args=(beta, mu, nu, m))

# Unpack results
Sa, I1a, I2a, D1a, D2a, Sb, I1b, I2b, D1b, D2b = solution.T

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot Patch A
ax1.plot(t, Sa, label='Susceptible', color='blue', linewidth=2)
ax1.plot(t, I1a, label='Infected (Slow)', color='green', linewidth=2)
ax1.plot(t, I2a, label='Infected (Fast)', color='orange', linewidth=2)
ax1.set_title('Patch A (Origin of Outbreak)')
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Number of Individuals')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot Patch B
ax2.plot(t, Sb, label='Susceptible', color='blue', linewidth=2)
ax2.plot(t, I1b, label='Infected (Slow)', color='green', linewidth=2)
ax2.plot(t, I2b, label='Infected (Fast)', color='orange', linewidth=2)
ax2.set_title('Patch B (Healthy Host Population)')
ax2.set_xlabel('Time (days)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('metapopulation_output.png', dpi=150, bbox_inches='tight')
plt.show()
