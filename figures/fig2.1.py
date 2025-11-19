import matplotlib.pyplot as plt
import numpy as np

# --- Data Based Strictly on Section V-B ---
metrics = ['Mean SNR', 'Connectivity (λ₂)', 'Coverage']

# Grid is the baseline (100%)
grid_values = [100, 100, 100]

# Optimized values based on reported percentages:
# "improves mean SNR by approximately 22%" -> 122
# "increases connectivity... by 35%" -> 135
# "maintaining similar coverage" -> ~100
optimized_values = [122, 135, 100] 

x = np.arange(len(metrics))
width = 0.35 

# --- Plotting ---
plt.figure(figsize=(9, 6))

# Create the grouped bars
rects1 = plt.bar(x - width/2, grid_values, width, label='Grid (Standard)', color='gray', alpha=0.6)
rects2 = plt.bar(x + width/2, optimized_values, width, label='Optimized (Evolutionary)', color='#1f77b4', alpha=0.9)

# Formatting
plt.ylabel('Normalized Performance (%)', fontsize=12)
plt.title('Figure 2: Quantified Improvement over Grid Formation', fontsize=14)
plt.xticks(x, metrics, fontsize=11)
plt.ylim(0, 160) # Room for labels
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# --- Add Text Annotations (The Quantitative Conclusions) ---
def autolabel(rects, is_baseline=False):
    for rect in rects:
        height = rect.get_height()
        if not is_baseline and height > 100:
            # Calculate and display the exact Delta from the paper
            percent_change = int(height - 100)
            plt.annotate(f'+{percent_change}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', color='black')
        elif not is_baseline and height == 100:
             plt.annotate('Similar',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', color='black', fontsize=9)

autolabel(rects2)

plt.tight_layout()
plt.savefig('fig2.1.png', dpi=300, bbox_inches='tight')
plt.close()