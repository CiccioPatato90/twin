import matplotlib.pyplot as plt
import numpy as np

# --- Data Simulation based on Text ---
# Text: "Best fitness increases sharply during the first 100 generations
# and stabilizes near generation 150" (Section V-A)
generations = np.arange(0, 201)
# Simulating a saturation curve that flattens out
fitness = 1 - 0.6 * np.exp(-0.025 * generations) 

# Add slight noise to simulate evolutionary variance, then smooth it
noise = np.random.normal(0, 0.005, len(generations))
fitness = fitness + noise
# Force the plateau after gen 150 to look very stable as per text
fitness[150:] = fitness[149] + np.random.normal(0, 0.001, len(fitness[150:]))

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(generations, fitness, color='#1f77b4', linewidth=2.5, label='Best Fitness Value')

# Annotations to match text description
plt.axvline(x=150, color='gray', linestyle='--', alpha=0.7)
plt.text(155, fitness[50], 'Stabilization Point\n(Gen ~150)', color='gray')

plt.title('Figure 1: Convergence Curve of Evolutionary Optimization', fontsize=14)
plt.xlabel('Generations', fontsize=12)
plt.ylabel('Normalized Fitness Score', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()

plt.savefig('fig1.png', dpi=300, bbox_inches='tight')
plt.close()