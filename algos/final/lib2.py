import pygad
import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------
# 1. PARAMETERS & CONSTANTS
# ---------------------------------------------------------
M = 6     # Number of drones
MAX_X = 50        # bounds
MAX_Y = 50        
POP_SIZE = 30     # Population size

# global alpha, beta, gamma, delta
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

def compute_distance_matrix(solution):
    """Vectorized distance matrix computation - much faster than nested loops"""
    coords = np.array([[p.x, p.y] for p in solution])
    # Compute pairwise distances using broadcasting
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    # Avoid division by zero
    distances = np.maximum(distances, 0.1)
    return distances

def calculate_mean_link_quality(solution):
    """Optimized using vectorized distance matrix"""
    distances = compute_distance_matrix(solution)
    return calculate_mean_link_quality_optimized(solution, distances)

def calculate_mean_link_quality_optimized(solution, distances):
    """Optimized version that reuses precomputed distance matrix"""
    # Vectorized power and SNR calculations
    nom = P_T * G_T * G_R
    absorb = 0.001 * f
    received_power = nom / ((distances ** spreading_factor) * np.exp(-absorb * distances))
    noise = 0.01 * f * B
    snr = received_power / noise
    
    # Mask diagonal (self-connections)
    mask = ~np.eye(M, dtype=bool)
    snr_masked = snr[mask]
    
    return np.mean(np.log(1 + snr_masked))

def laplacian(weights):
    degrees = np.sum(weights, axis=1)
    return np.diag(degrees) - weights

def calc_connectivity(solution):
    """Optimized using vectorized distance matrix"""
    distances = compute_distance_matrix(solution)
    return calc_connectivity_optimized(solution, distances)

def calc_connectivity_optimized(solution, distances):
    """Optimized version that reuses precomputed distance matrix"""
    weight_list = 1.0 / distances
    # Set diagonal to 0 (no self-connections)
    np.fill_diagonal(weight_list, 0)
    
    L = laplacian(weight_list)
    eigenvals = np.sort(np.real(np.linalg.eigvals(L)))
    return max(0.0, eigenvals[1]) # Algebraic connectivity

def calculate_coverage(solution):
    """Optimized using vectorized operations"""
    # Reduced sampling for speed (Step 2 instead of 1)
    # If you need exact precision, change step to 1
    step = 2
    x_coords = np.arange(0, MAX_X, step)
    y_coords = np.arange(0, MAX_Y, step)
    
    # Create grid of test points
    X, Y = np.meshgrid(x_coords, y_coords)
    test_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    # Get drone positions
    drone_pos = np.array([[p.x, p.y] for p in solution])
    
    # Vectorized distance calculation: (n_test_points, n_drones)
    diff = test_points[:, np.newaxis, :] - drone_pos[np.newaxis, :, :]
    distances_sq = np.sum(diff**2, axis=2)
    
    # Check if any drone covers each point
    covered = np.any(distances_sq <= r**2, axis=1)
    
    return np.mean(covered)

def calculate_repulsion_penalty(solution):
    """Optimized using vectorized distance matrix"""
    distances = compute_distance_matrix(solution)
    return calculate_repulsion_penalty_optimized(solution, distances)

def calculate_repulsion_penalty_optimized(solution, distances):
    """Optimized version that reuses precomputed distance matrix"""
    # Get upper triangle (i < j) to avoid double counting
    mask = np.triu(np.ones((M, M), dtype=bool), k=1)
    repulsion = np.exp(-(distances[mask] / d_0))
    return np.sum(repulsion)

def calculate_penalty_function(solution):
    """Optimized using vectorized distance matrix"""
    distances = compute_distance_matrix(solution)
    return calculate_penalty_function_optimized(solution, distances)

def calculate_penalty_function_optimized(solution, distances):
    """Optimized version that reuses precomputed distance matrix"""
    # Get upper triangle (i < j) to avoid double counting
    mask = np.triu(np.ones((M, M), dtype=bool), k=1)
    penalties = np.maximum(0, d_min - distances[mask])
    return np.sum(penalties)

# ---------------------------------------------------------
# 3. PyGAD BRIDGE FUNCTIONS
# ---------------------------------------------------------

def decode_solution(flat_solution):
    """Converts [x1, y1, x2, y2...] to [Position(x1,y1), ...]"""
    positions = []
    for i in range(0, len(flat_solution), 2):
        positions.append(Position(flat_solution[i], flat_solution[i+1]))
    return positions

def create_fitness_func(alpha, beta, gamma, delta):
    """
    Factory function that creates a fitness function with captured parameters.
    This allows each GA instance to have its own parameters without using globals.
    """
    def fitness_func(ga_instance, solution, solution_idx):
        """
        The function PyGAD calls to evaluate a genome.
        Optimized to compute distance matrix once and reuse it.
        """
        # 1. Decode generic array to Drone Positions
        decoded = decode_solution(solution)
        
        # 2. Compute distance matrix once (used by multiple metrics)
        distances = compute_distance_matrix(decoded)
        
        # 3. Calculate Sub-metrics (optimized versions)
        lq = alpha * calculate_mean_link_quality_optimized(decoded, distances)
        conn = beta * calc_connectivity_optimized(decoded, distances)
        cov = gamma * calculate_coverage(decoded)  # Coverage doesn't use distance matrix
        rep = delta * calculate_repulsion_penalty_optimized(decoded, distances)
        
        # 4. Calculate Fitness
        # Maximize Payoff - Penalty
        payoff = lq + conn + cov - rep
        penalty = rho * calculate_penalty_function_optimized(decoded, distances)
        
        return payoff - penalty
    
    return fitness_func

# Store history for plotting
history = {
    "fitness": [],
    "connectivity": [],
    "coverage": [],
    "link_quality": [],
    "repulsion": []
}

# def on_generation(ga_instance):
# handler to pygda library

# ---------------------------------------------------------
# 3.5. DATA LOADING/SAVING HELPERS
# ---------------------------------------------------------

def load_results(filename="results.npz"):
    """
    Load saved results from a .npz file.
    
    Returns:
        dict: Dictionary with keys 'results_conn', 'results_coverage', 'results_lq', 'test_values'
    
    Example:
        data = load_results("results.npz")
        results_conn = data['results_conn']
        results_coverage = data['results_coverage']
        results_lq = data['results_lq']
        test_values = data['test_values']
    """
    data = np.load(filename)
    return {
        'results_conn': data['results_conn'],
        'results_coverage': data['results_coverage'],
        'results_lq': data['results_lq'],
        'test_values': data['test_values']
    }

# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------

def main(alpha, beta, gamma, delta):
    """
    Run GA optimization with specific parameter values.
    Now accepts parameters instead of using globals to enable parallel execution.
    """
    # Define gene space: [0, 50] for every gene
    # This prevents drones from flying out of the area
    gene_space = [{'low': 0, 'high': MAX_X}] * (M * 2)

    # Create fitness function with captured parameters
    fitness_func = create_fitness_func(alpha, beta, gamma, delta)

    ga_instance = pygad.GA(
        num_generations=50,             # Reduced from 100 for faster runs
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
        parallel_processing=['thread', 8],  # Use 8 threads for parallel fitness evaluation
        # on_generation=on_generation,
        suppress_warnings=True,
    )
    ga_instance.run()

    # ---------------------------------------------------------
    # 5. VISUALIZATION
    # ---------------------------------------------------------
    
    best_sol, _, _ = ga_instance.best_solution()

    final_pos = decode_solution(best_sol)

    history = {
        "connectivity": calc_connectivity(final_pos),
        "coverage": calculate_coverage(final_pos),
        "link_quality": calculate_mean_link_quality(final_pos)
    }

    return [history["connectivity"], history["coverage"], history["link_quality"]]

async def run_single_optimization(i, j, k, l, val, val2, val3, val4, executor):
    """
    Run a single optimization and return the results with indices.
    This function is designed to be run concurrently.
    """
    alpha = float(val)
    beta = float(val2)
    gamma = float(val3)
    delta = float(val4)
    
    print(f"Running with alpha: {alpha}, beta: {beta}, gamma: {gamma}, delta: {delta}")
    
    # Run CPU-bound main() in thread pool executor
    loop = asyncio.get_event_loop()
    history = await loop.run_in_executor(executor, main, alpha, beta, gamma, delta)
    
    print(f"  -> conn: {history[0]:.4f}, cov: {history[1]:.4f}, lq: {history[2]:.4f}")
    
    return i, j, k, l, history

async def run_parallel_optimizations(test_values, results_conn, results_coverage, results_lq):
    """
    Run all optimizations in parallel for the val4 loop.
    Each iteration of the outer loops (i, j, k) runs all val4 iterations concurrently.
    """
    # Create thread pool executor for CPU-bound tasks
    # Limit threads to avoid overwhelming the system (PyGAD already uses 8 threads internally)
    executor = ThreadPoolExecutor(max_workers=8)  # Adjust based on your CPU cores
    
    try:
        for i, val in enumerate(test_values):
            for j, val2 in enumerate(test_values):
                for k, val3 in enumerate(test_values):
                    # Create tasks for all val4 iterations at once
                    tasks = [
                        run_single_optimization(i, j, k, l, val, val2, val3, val4, executor)
                        for l, val4 in enumerate(test_values)
                    ]
                    
                    # Run all val4 iterations concurrently
                    results = await asyncio.gather(*tasks)
                    
                    # Store results (each result has unique indices, so no race condition)
                    for i_res, j_res, k_res, l_res, history in results:
                        results_conn[i_res, j_res, k_res, l_res] = history[0]
                        results_coverage[i_res, j_res, k_res, l_res] = history[1]
                        results_lq[i_res, j_res, k_res, l_res] = history[2]
    finally:
        executor.shutdown(wait=True)

if __name__ == "__main__":
    
    test_values = np.linspace(0, 100, 10).astype(int) 

    results_conn = np.zeros((len(test_values), len(test_values), len(test_values), len(test_values)))
    results_coverage = np.zeros((len(test_values), len(test_values), len(test_values), len(test_values)))
    results_lq = np.zeros((len(test_values), len(test_values), len(test_values), len(test_values)))

    print("results.shape: ", results_conn.shape)
    print(f"Total iterations: {len(test_values)**4}")
    print(f"Parallelizing val4 loop: {len(test_values)} concurrent runs per (i,j,k) combination")

    # Run async optimization loop
    asyncio.run(run_parallel_optimizations(test_values, results_conn, results_coverage, results_lq))

    # Save results in NumPy compressed format for easy reloading
    np.savez_compressed(
        "results.npz",
        results_conn=results_conn,
        results_coverage=results_coverage,
        results_lq=results_lq,
        test_values=test_values  # Also save test_values for reference
    )

    print("Results saved to results.npz")
    print("To reload: data = np.load('results.npz'); results_conn = data['results_conn']; ...")
    
    # Plotting: For each metric, plot how it varies with each coefficient
    # Data structure: results[i, j, k, l] where:
    # i = alpha index, j = beta index, k = gamma index, l = delta index
    
    metrics = {
        'Connectivity': results_conn,
        'Coverage': results_coverage,
        'Link Quality': results_lq
    }
    
    coefficient_names = ['Alpha', 'Beta', 'Gamma', 'Delta']
    coefficient_axes = [0, 1, 2, 3]  # Which axis to vary (i, j, k, l)
    
    # Create a figure for each metric
    for metric_name, metric_data in metrics.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{metric_name} vs Coefficient Values', fontsize=14, fontweight='bold')
        axes = axes.flatten()
        
        for idx, (coeff_name, axis_idx) in enumerate(zip(coefficient_names, coefficient_axes)):
            # Calculate mean for each value of this coefficient
            # When varying axis_idx, average over all other axes
            means = []
            for val_idx in range(len(test_values)):
                # Create a slice that fixes this coefficient and averages over others
                if axis_idx == 0:  # Vary alpha (i)
                    mean_val = np.mean(metric_data[val_idx, :, :, :])
                elif axis_idx == 1:  # Vary beta (j)
                    mean_val = np.mean(metric_data[:, val_idx, :, :])
                elif axis_idx == 2:  # Vary gamma (k)
                    mean_val = np.mean(metric_data[:, :, val_idx, :])
                else:  # Vary delta (l)
                    mean_val = np.mean(metric_data[:, :, :, val_idx])
                means.append(mean_val)
            
            # Plot
            axes[idx].plot(test_values, means, marker='o', linewidth=2, markersize=8)
            axes[idx].set_xlabel(f'{coeff_name} Value', fontsize=11)
            axes[idx].set_ylabel(f'Mean {metric_name}', fontsize=11)
            axes[idx].set_title(f'{metric_name} vs {coeff_name}', fontsize=12)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xticks(test_values)
        
        plt.tight_layout()
        plt.savefig(f'{metric_name.lower().replace(" ", "_")}_plots.png', dpi=150, bbox_inches='tight')
        print(f"Saved plot: {metric_name.lower().replace(' ', '_')}_plots.png")
    
    plt.show()
    