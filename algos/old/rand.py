import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

# Import parameters and functions from ga2.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ga2 import (
    M, MAX_X, MAX_Y,
    calculate_mean_snr, calculate_coverage, calculate_fitness,
    calculate_mean_link_quality
)

@dataclass
class Metrics:
    """Metrics for random configuration."""
    normalized_fitness: float
    mean_snr: float
    mean_link_quality: float
    area_coverage: float
    
    def to_dict(self):
        return asdict(self)

def generate_random_configuration():
    """Generate a random drone configuration."""
    solution = np.random.uniform(0, MAX_X, size=(M, 2))
    solution[:, 0] = np.clip(solution[:, 0], 0, MAX_X)
    solution[:, 1] = np.clip(solution[:, 1], 0, MAX_Y)
    return solution

def plot_configuration(solution, metrics, filename="random_configuration.png"):
    """Plot the random configuration with metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Configuration
    x_coords = solution[:, 0]
    y_coords = solution[:, 1]
    
    ax1.scatter(x_coords, y_coords, c='blue', s=100, label='Drones', zorder=5)
    ax1.set_xlim(0, MAX_X)
    ax1.set_ylim(0, MAX_Y)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_title(f'Random Formation\nM={M} drones')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Draw connections between nearby drones
    diff = solution[:, np.newaxis, :] - solution[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    
    for i in range(M):
        for j in range(i+1, M):
            if dist_matrix[i, j] < 20:  # Show connections within 20 units
                ax1.plot([solution[i, 0], solution[j, 0]],
                        [solution[i, 1], solution[j, 1]], 'k-', alpha=0.2, linewidth=0.5)
    
    # Plot 2: Metrics display
    ax2.axis('off')
    metrics_text = f"""
    Random Formation Metrics
    
    Mean SNR: {metrics.mean_snr:.4f}
    Mean Link Quality: {metrics.mean_link_quality:.4f}
    Area Coverage: {metrics.area_coverage:.2f}%
    Fitness: {metrics.normalized_fitness:.4f}
    
    Configuration:
    - Number of drones: {M}
    - Area: {MAX_X} x {MAX_Y}
    """
    ax2.text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Random configuration plot saved to: {filename}")

def main():
    print(f"Generating random configuration...")
    print(f"Parameters: M={M}, MAX_X={MAX_X}, MAX_Y={MAX_Y}")
    
    # Generate one random configuration
    solution = generate_random_configuration()
    
    # Calculate metrics
    mean_snr = calculate_mean_snr(solution)
    mean_link_quality = calculate_mean_link_quality(solution)
    coverage = calculate_coverage(solution) * 100  # Convert to percentage
    fitness = calculate_fitness(solution)
    
    # Store metrics
    metrics = Metrics(
        normalized_fitness=fitness,
        mean_snr=mean_snr,
        mean_link_quality=mean_link_quality,
        area_coverage=coverage
    )
    
    # Print results
    print(f"\nRandom Formation Results:")
    print(f"Mean SNR: {metrics.mean_snr:.4f}")
    print(f"Mean Link Quality: {metrics.mean_link_quality:.4f}")
    print(f"Area Coverage: {metrics.area_coverage:.2f}%")
    print(f"Fitness: {metrics.normalized_fitness:.4f}")
    
    # Plot configuration
    plot_configuration(solution, metrics)
    
    return metrics

if __name__ == "__main__":
    main()

