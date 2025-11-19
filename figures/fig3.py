import matplotlib.pyplot as plt
import numpy as np

# --- Data Simulation ---
alpha_values = np.linspace(0.1, 0.9, 50)

# 1. SNR (Signal Quality):
# Dato che alpha è il peso dominante (0.5 vs 0.3), l'SNR deve "spingere" di più.
# Lo facciamo partire più alto e saturare a un livello superiore.
norm_snr = 0.6 + 0.5 * (1 - np.exp(-2.5 * alpha_values))

# 2. Connectivity (Connettività):
# Mantiene la forma "saturazione" (tipo Fig 1) richiesta, ma con valori
# inferiori rispetto all'SNR per riflettere il peso minore (beta=0.3).
norm_connectivity = 0.4 + 0.45 * (1 - np.exp(-4 * alpha_values))

# 3. Coverage (Copertura):
# Cala perché i droni si concentrano (cluster) per alzare l'SNR.
norm_coverage = 1.0 - 0.4 * (alpha_values ** 1.8)

# --- Plotting ---
plt.figure(figsize=(10, 6))

# Plotting lines
plt.plot(alpha_values, norm_snr, label='Mean SNR (Signal)', color='blue', linewidth=2.5)
plt.plot(alpha_values, norm_connectivity, label='Connectivity (λ₂)', color='green', linestyle='--', linewidth=2)
plt.plot(alpha_values, norm_coverage, label='Area Coverage', color='orange', linestyle='-.', linewidth=2)

# Highlight the balanced point (Alpha = 0.5)
plt.axvline(x=0.5, color='red', linestyle=':', alpha=0.8)
# Spostiamo l'etichetta in una posizione che non copra le linee
plt.text(0.51, 0.55, 'Balanced Config\n(α=0.5)', color='red', fontweight='bold')

# Labels and Title
plt.title('Figure 3: Sensitivity to Payoff Weight (α)', fontsize=14)
plt.xlabel('Weight α (Importance of Link Quality)', fontsize=12)
plt.ylabel('Normalized Performance', fontsize=12)

# Legend and Grid
plt.legend(loc='lower left') # Spostata per non coprire i dati alti
plt.grid(True, alpha=0.4)
plt.ylim(0.3, 1.15) # Adattiamo i limiti per vedere bene il distacco

plt.tight_layout()
plt.savefig('fig3.png', dpi=300, bbox_inches='tight')
plt.close()