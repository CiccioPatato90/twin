import pygad
import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys

# ---------------------------------------------------------
# 1. PARAMETERS & CONSTANTS
# ---------------------------------------------------------
M = 7      # Number of drones
MAX_X = 50        # bounds
MAX_Y = 50        
POP_SIZE = 30     # Population size


# three curves on paper is when 

# 1 plot lq y value, x 
    # 4 subplot: alpha varies, rest fixed 
# 1 plot connectivity
# 1 plot coverage 
# 1 plot repulsion
# 1 plot fitness

global alpha, beta, gamma, delta
# Weights
# alpha = 0       # Link Quality -> y is payoff function. 
# beta = 0        # Connectivity
# gamma = 0     # Coverage (lattice parameter. if higher, more geometric)
# delta = 0       # Repulsion

# Physics Constants
P_T = 5           
G_T = 1           
G_R = 1           
spreading_factor = 2 
f = 1             
B = 1             
r = 10            # sensing radius
d_0 = 20         # minimum safe distance 
d_min = d_0
rho = 0         # penalty coefficient

# ---------------------------------------------------------
# 2. HELPER CLASSES & PHYSICS (Kept mostly original)
# ---------------------------------------------------------

@dataclass
class Position:
    x: float
    y: float

def norm(i, j):
    return math.sqrt((i.x - j.x)**2 + (i.y - j.y)**2)

def calculate_received_power(i, j, f):
    distance = norm(i, j)
    if distance == 0: return P_T * G_T * G_R
    nom = P_T * G_T * G_R
    # Absorption model
    absorb = 0.001 * f
    val = nom / ((max(distance, 0.1) ** spreading_factor) * (math.exp(-absorb * distance)))
    return val

def calculate_snr(i, j, f):
    received = calculate_received_power(i, j, f)
    noise = 0.01 * f * B
    return received / noise

def calculate_mean_link_quality(solution):
    total = 0
    count = 0
    for i in range(M):
        for j in range(M):
            if i != j:
                total += math.log(1 + calculate_snr(solution[i], solution[j], f))
                count += 1
    return total / count if count > 0 else 0

def laplacian(weights):
    degrees = np.sum(weights, axis=1)
    return np.diag(degrees) - weights

def calc_connectivity(solution):
    weight_list = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i != j:
                dist = norm(solution[i], solution[j])
                weight_list[i][j] = 1.0 / max(dist, 0.1)
    
    L = laplacian(weight_list)
    eigenvals = np.sort(np.real(np.linalg.eigvals(L)))
    return max(0.0, eigenvals[1]) # Algebraic connectivity

def is_covered(x, y, solution):
    for drone in solution:
        if (drone.x - x)**2 + (drone.y - y)**2 <= r**2:
            return True
    return False

def calculate_coverage(solution):
    # Reduced sampling for speed (Step 2 instead of 1)
    # If you need exact precision, change step to 1
    covered_pts = 0
    total_pts = 0
    step = 2 
    for i in range(0, MAX_X, step):
        for j in range(0, MAX_Y, step):
            total_pts += 1
            if is_covered(i, j, solution):
                covered_pts += 1
    return covered_pts / total_pts

def calculate_repulsion_penalty(solution):
    val = 0
    for i in range(M):
        for j in range(i+1, M):
            val += math.exp(-(norm(solution[i], solution[j])/d_0))
    return val

def calculate_penalty_function(solution):
    val = 0
    for i in range(M):
        for j in range(i+1, M):
            val += max(0, d_min - norm(solution[i], solution[j]))
    return val

# ---------------------------------------------------------
# 3. PyGAD BRIDGE FUNCTIONS
# ---------------------------------------------------------

def decode_solution(flat_solution):
    """Converts [x1, y1, x2, y2...] to [Position(x1,y1), ...]"""
    positions = []
    for i in range(0, len(flat_solution), 2):
        positions.append(Position(flat_solution[i], flat_solution[i+1]))
    return positions

def fitness_func(ga_instance, solution, solution_idx):
    """
    The function PyGAD calls to evaluate a genome.
    """
    # 1. Decode generic array to Drone Positions
    decoded = decode_solution(solution)
    
    # 2. Calculate Sub-metrics
    lq = alpha * calculate_mean_link_quality(decoded)
    conn = beta * calc_connectivity(decoded)
    cov = gamma * calculate_coverage(decoded)
    rep = delta * calculate_repulsion_penalty(decoded)
    
    # 3. Calculate Fitness
    # Maximize Payoff - Penalty
    payoff = lq + conn + cov - rep
    penalty = rho * calculate_penalty_function(decoded)
    
    return payoff - penalty

# Store history for plotting
history = {
    "fitness": [],
    "connectivity": [],
    "coverage": [],
    "link_quality": [],
    "repulsion": []
}

# def on_generation(ga_instance):
    # """
    # Callback running at the end of every generation.
    # We use this to log the specific sub-metrics of the BEST solution.
    # """
    # best_sol, best_fit, _ = ga_instance.best_solution()
    # decoded = decode_solution(best_sol)
    
    # # Re-calc specific stats for logging
    # history["fitness"].append(best_fit)
    # history["connectivity"].append(calc_connectivity(decoded))
    # history["coverage"].append(calculate_coverage(decoded))
    # history["link_quality"].append(calculate_mean_link_quality(decoded))
    # history["repulsion"].append(calculate_repulsion_penalty(decoded))
    
    # print(f"Gen {ga_instance.generations_completed} | Fitness: {best_fit:.4f}")

# ---------------------------------------------------------
# 4. VISUALIZATION FUNCTIONS
# ---------------------------------------------------------

def plot_solution(solution, title="Drone Formation Solution"):
    """
    Plot the drone positions with sensing radius and connections.
    
    Args:
        solution: List of Position objects representing drone positions
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract x and y coordinates
    x_coords = [pos.x for pos in solution]
    y_coords = [pos.y for pos in solution]
    
    # Plot sensing radius circles for each drone
    for i, pos in enumerate(solution):
        circle = plt.Circle((pos.x, pos.y), r, color='lightblue', 
                           alpha=0.3, fill=True, label='Sensing Radius' if i == 0 else '')
        ax.add_patch(circle)
    
    # Plot connections between drones (if within reasonable distance)
    connection_threshold = 30  # Only show connections within this distance
    for i in range(M):
        for j in range(i+1, M):
            dist = norm(solution[i], solution[j])
            if dist < connection_threshold:
                ax.plot([solution[i].x, solution[j].x], 
                       [solution[i].y, solution[j].y], 
                       'gray', alpha=0.3, linewidth=1, linestyle='--')
    
    # Plot drone positions
    ax.scatter(x_coords, y_coords, c='red', s=200, marker='o', 
              edgecolors='black', linewidths=2, zorder=5, label='Drones')
    
    # Label drones
    for i, pos in enumerate(solution):
        ax.annotate(f'D{i+1}', (pos.x, pos.y), xytext=(5, 5), 
                   textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Set plot properties
    ax.set_xlim(-5, MAX_X + 5)
    ax.set_ylim(-5, MAX_Y + 5)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    
    # Add metrics text box
    conn = calc_connectivity(solution)
    cov = calculate_coverage(solution)
    lq = calculate_mean_link_quality(solution)
    metrics_text = f'Connectivity: {conn:.4f}\nCoverage: {cov:.4f}\nLink Quality: {lq:.4f}'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 5. MAIN EXECUTION
# ---------------------------------------------------------

def main():
    # Define gene space: [0, 50] for every gene
    # This prevents drones from flying out of the area
    gene_space = [{'low': 0, 'high': MAX_X}] * (M * 2)

    ga_instance = pygad.GA(
        num_generations=100,
        num_parents_mating=10,          # Number of parents to select
        fitness_func=fitness_func,
        sol_per_pop=POP_SIZE,
        num_genes=M * 2,                # 2 genes (x,y) per drone
        gene_space=gene_space,          # Hard bounds
        parent_selection_type="tournament",
        keep_parents=2,                 # Elitism: keep 2 best parents
        crossover_type="uniform",
        mutation_type="random",
        mutation_percent_genes=10,      # Mutate 10% of genes
        # on_generation=on_generation,
        suppress_warnings=True,
    )

    print("Starting PyGAD Optimization...")
    ga_instance.run()

    # ---------------------------------------------------------
    # 5. VISUALIZATION
    # ---------------------------------------------------------
    
    best_sol, best_fitness, _ = ga_instance.best_solution()

    final_pos = decode_solution(best_sol)
    
    # Plot the solution
    plot_solution(final_pos, title=f"Best Solution (Fitness: {best_fitness:.4f})")

    history = {
        "connectivity": calc_connectivity(final_pos),
        "coverage": calculate_coverage(final_pos),
        "link_quality": calculate_mean_link_quality(final_pos)
    }

    return [history["connectivity"], history["coverage"], history["link_quality"]]

if __name__ == "__main__":
    global alpha, beta, gamma, delta
    alpha = float(sys.argv[1])
    beta = float(sys.argv[2])
    gamma = float(sys.argv[3])
    delta = float(sys.argv[4])
    print(f"Running with alpha: {alpha}, beta: {beta}, gamma: {gamma}, delta: {delta}")
    main()


    