import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

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
    """Metrics for grid formation."""
    normalized_fitness: float
    mean_snr: float
    mean_link_quality: float
    area_coverage: float
    
    def to_dict(self):
        return asdict(self)

def generate_grid_configuration():
    """Generate a grid formation of drones."""
    # Calculate grid dimensions (as square as possible)
    # Find factors of M that are close to sqrt(M)
    best_rows = int(np.sqrt(M))
    best_cols = M // best_rows
    
    # Adjust if we can't fit all drones
    while best_rows * best_cols < M:
        if best_cols <= best_rows:
            best_cols += 1
        else:
            best_rows += 1
    
    # Calculate spacing
    if best_rows > 1:
        x_spacing = MAX_X / (best_rows - 1) if best_rows > 1 else MAX_X
    else:
        x_spacing = MAX_X / 2
    
    if best_cols > 1:
        y_spacing = MAX_Y / (best_cols - 1) if best_cols > 1 else MAX_Y
    else:
        y_spacing = MAX_Y / 2
    
    # Generate grid positions
    solution = []
    drone_count = 0
    
    for i in range(best_rows):
        for j in range(best_cols):
            if drone_count >= M:
                break
            
            x = i * x_spacing if best_rows > 1 else MAX_X / 2
            y = j * y_spacing if best_cols > 1 else MAX_Y / 2
            
            # Clamp to bounds
            x = np.clip(x, 0, MAX_X)
            y = np.clip(y, 0, MAX_Y)
            
            solution.append([x, y])
            drone_count += 1
        
        if drone_count >= M:
            break
    
    return np.array(solution)

def plot_configuration(solution, metrics, filename="grid_configuration.png"):
    """Plot the grid configuration with metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Configuration
    x_coords = solution[:, 0]
    y_coords = solution[:, 1]
    
    ax1.scatter(x_coords, y_coords, c='blue', s=100, label='Drones', zorder=5)
    ax1.set_xlim(0, MAX_X)
    ax1.set_ylim(0, MAX_Y)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_title(f'Grid Formation\nM={M} drones, Grid: {int(np.sqrt(M))}x{M//int(np.sqrt(M))}')
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
    Grid Formation Metrics
    
    Mean SNR: {metrics.mean_snr:.4f}
    Mean Link Quality: {metrics.mean_link_quality:.4f}
    Area Coverage: {metrics.area_coverage:.2f}%
    Fitness: {metrics.normalized_fitness:.4f}
    
    Configuration:
    - Number of drones: {M}
    - Area: {MAX_X} x {MAX_Y}
    - Grid layout: {int(np.sqrt(M))} rows x {M//int(np.sqrt(M))} cols
    """
    ax2.text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Grid configuration plot saved to: {filename}")

def main():
    print(f"Generating grid formation configuration...")
    print(f"Parameters: M={M}, MAX_X={MAX_X}, MAX_Y={MAX_Y}")
    
    # Generate grid configuration
    solution = generate_grid_configuration()
    
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
    print(f"\nGrid Formation Results:")
    print(f"Mean SNR: {metrics.mean_snr:.4f}")
    print(f"Mean Link Quality: {metrics.mean_link_quality:.4f}")
    print(f"Area Coverage: {metrics.area_coverage:.2f}%")
    print(f"Fitness: {metrics.normalized_fitness:.4f}")
    
    # Plot configuration
    plot_configuration(solution, metrics)
    
    return metrics

if __name__ == "__main__":
    main()

