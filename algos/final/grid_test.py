import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass
from lib2 import (
    Position, 
    calc_connectivity, 
    calculate_coverage, 
    calculate_mean_link_quality,
    M, MAX_X, MAX_Y, r
)

# ---------------------------------------------------------
# GRID GENERATION
# ---------------------------------------------------------

def generate_grid_positions(num_drones, max_x, max_y):
    """
    Generate a regular grid of drone positions.
    """
    positions = []
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_drones)))
    
    # Calculate spacing
    if grid_size > 1:
        x_spacing = max_x / (grid_size + 1)
        y_spacing = max_y / (grid_size + 1)
    else:
        x_spacing = max_x / 2
        y_spacing = max_y / 2
    
    # Generate positions in a grid pattern
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= num_drones:
                break
            x = (i + 1) * x_spacing
            y = (j + 1) * y_spacing
            positions.append(Position(x, y))
            count += 1
        if count >= num_drones:
            break
    
    return positions

def generate_random_positions(num_drones, max_x, max_y):
    """
    Generate random drone positions.
    """
    positions = []
    for _ in range(num_drones):
        x = np.random.uniform(0, max_x)
        y = np.random.uniform(0, max_y)
        positions.append(Position(x, y))
    return positions

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("DRONE GRID METRICS TEST")
    print("="*60)
    print(f"Number of drones: {M}")
    print(f"Grid size: {MAX_X} x {MAX_Y}")
    print(f"Sensing radius: 10")
    print("="*60)
    print()
    
    # Generate grid positions
    print("Generating grid positions...")
    solution = generate_grid_positions(M, MAX_X, MAX_Y)
    
    # Print positions
    print("\nDrone Positions:")
    print("-" * 40)
    for i, pos in enumerate(solution):
        print(f"Drone {i+1}: ({pos.x:.2f}, {pos.y:.2f})")
    print()
    
    # Calculate metrics
    print("Calculating metrics...")
    print("-" * 40)
    
    connectivity = calc_connectivity(solution)
    coverage = calculate_coverage(solution)
    link_quality = calculate_mean_link_quality(solution)
    
    # Display results in a table
    print("\n" + "="*60)
    print("RESULTS TABLE")
    print("="*60)
    print(f"{'Metric':<20} {'Value':<15} {'Description'}")
    print("-"*60)
    print(f"{'Connectivity':<20} {connectivity:<15.6f} {'Algebraic connectivity (2nd smallest eigenvalue)'}")
    print(f"{'Coverage':<20} {coverage:<15.6f} {'Fraction of area covered (0-1)'}")
    print(f"{'Link Quality':<20} {link_quality:<15.6f} {'Mean log(1+SNR) across all links'}")
    print("="*60)
    
    # Summary statistics
    print("\nSummary:")
    print(f"  • Connectivity: {connectivity:.4f} (higher is better)")
    print(f"  • Coverage: {coverage*100:.2f}% of area covered")
    print(f"  • Link Quality: {link_quality:.4f} (higher is better)")
    print()
    
    # Create visualization
    print("Generating visualization...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw the area boundary
    ax.add_patch(plt.Rectangle((0, 0), MAX_X, MAX_Y, fill=False, edgecolor='black', linewidth=2, linestyle='--'))
    
    # Draw sensing radius circles for each drone
    colors = plt.cm.tab10(np.linspace(0, 1, M))
    for i, pos in enumerate(solution):
        circle = Circle((pos.x, pos.y), r, fill=True, alpha=0.2, color=colors[i], edgecolor=colors[i], linewidth=1.5)
        ax.add_patch(circle)
    
    # Draw drone positions
    for i, pos in enumerate(solution):
        ax.plot(pos.x, pos.y, 'o', markersize=12, color=colors[i], markeredgecolor='black', markeredgewidth=2, label=f'Drone {i+1}')
    
    # Draw connections between drones (optional - can be commented out if too cluttered)
    for i in range(M):
        for j in range(i+1, M):
            dist = math.sqrt((solution[i].x - solution[j].x)**2 + (solution[i].y - solution[j].y)**2)
            if dist < r * 2:  # Only draw if drones are relatively close
                ax.plot([solution[i].x, solution[j].x], [solution[i].y, solution[j].y], 
                       'k-', alpha=0.3, linewidth=0.5)
    
    # Set axis properties
    ax.set_xlim(-2, MAX_X + 2)
    ax.set_ylim(-2, MAX_Y + 2)
    ax.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax.set_title('Drone Grid Configuration with Sensing Radii', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add statistics text box
    stats_text = f"""METRICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Connectivity:  {connectivity:.6f}
Coverage:      {coverage*100:.2f}%
Link Quality:  {link_quality:.6f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Drones:        {M}
Grid Size:     {MAX_X} × {MAX_Y}
Sensing Radius: {r}
"""
    
    # Position text box in upper right
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5),
            family='monospace')
    
    # Add legend (only show first few drones to avoid clutter)
    if M <= 10:
        ax.legend(loc='upper left', fontsize=8, ncol=2)
    else:
        ax.text(0.02, 0.98, f'{M} drones', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the figure
    output_file = 'drone_grid_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    plt.show()

