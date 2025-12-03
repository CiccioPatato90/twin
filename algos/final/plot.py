import pygad
import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys
import logging

SCALING = 10

# ---------------------------------------------------------
# 1. PARAMETERS & CONSTANTS (UNCHANGED)
# ---------------------------------------------------------
M = 7      # Number of drones
MAX_X = int(50/SCALING)        # bounds
MAX_Y = int(50/SCALING)        
POP_SIZE = 30     # Population size

# Weights (Will be set dynamically in the loop)
global alpha, beta, gamma, delta
alpha = 0       
beta = 0        
gamma = 0     
delta = 0       

# Physics Constants
P_T = 5           
G_T = 1           
G_R = 1           
spreading_factor = 2 
f = 1             
B = 1             
r = int(10/SCALING)            # sensing radius
d_0 = int(20/SCALING)          # minimum safe distance 
d_min = d_0
rho = 0           # penalty coefficient

# ---------------------------------------------------------
# 2. HELPER CLASSES & PHYSICS (UNCHANGED)
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
# 3. PyGAD BRIDGE FUNCTIONS (UNCHANGED)
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

# ---------------------------------------------------------
# 4. EXPERIMENT RUNNER (MODIFIED LOGIC)
# ---------------------------------------------------------

def run_single_optimization(set_alpha, set_beta, set_gamma, set_delta):
    """
    Runs one instance of the GA with specific weights and returns
    the specific raw metric values (not weighted) of the best solution.
    """
    global alpha, beta, gamma, delta
    alpha = set_alpha
    beta = set_beta
    gamma = set_gamma
    delta = set_delta

    gene_space = [{'low': 0, 'high': MAX_X}] * (M * 2)

    ga_instance = pygad.GA(
        num_generations=50,             # Reduced slightly for loop speed, adjust if needed
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=POP_SIZE,
        num_genes=M * 2,
        gene_space=gene_space,
        parent_selection_type="tournament",
        keep_parents=2,
        crossover_type="uniform",
        mutation_type="random",
        mutation_percent_genes=10,
        suppress_warnings=True
    )

    ga_instance.run()

    # Extract best solution
    best_sol, best_fit, _ = ga_instance.best_solution()
    decoded = decode_solution(best_sol)

    # Return RAW metrics (unweighted) to see the physical effect
    return {
        "lq": calculate_mean_link_quality(decoded),
        "conn": calc_connectivity(decoded),
        "cov": calculate_coverage(decoded),
        "rep": calculate_repulsion_penalty(decoded),
        "fitness": best_fit
    }

def main():
    print("Starting Multi-Objective Sweep...")
    
    # Define range of values to test for weights
    # Adjust 'num=5' to 'num=10' for smoother curves (but takes longer)
    test_values = np.linspace(int(0/SCALING), int(1000/SCALING), int(10*SCALING)) 
    
    # Baseline weights for the "fixed" parameters
    BASE = int(1.0/SCALING)

    # Storage for plotting
    plot_data = {
        "alpha": {"x": [], "y": []}, # Vary Alpha -> Measure LQ
        "beta":  {"x": [], "y": []}, # Vary Beta  -> Measure Connectivity
        "gamma": {"x": [], "y": []}, # Vary Gamma -> Measure Coverage
        "delta": {"x": [], "y": []}  # Vary Delta -> Measure Repulsion
    }

    # -----------------------------------------------------
    # LOOP 1: Vary Alpha (Link Quality Weight)
    # -----------------------------------------------------
    print("1. Varying Alpha (Link Quality)...")
    for val in test_values:
        # alpha varies, others fixed at BASE
        res = run_single_optimization(val, BASE, BASE, BASE)
        plot_data["alpha"]["x"].append(val)
        plot_data["alpha"]["y"].append(res["lq"])

    # -----------------------------------------------------
    # LOOP 2: Vary Beta (Connectivity Weight)
    # -----------------------------------------------------
    print("2. Varying Beta (Connectivity)...")
    for val in test_values:
        # beta varies, others fixed at BASE
        res = run_single_optimization(BASE, val, BASE, BASE)
        plot_data["beta"]["x"].append(val)
        plot_data["beta"]["y"].append(res["conn"])

    # -----------------------------------------------------
    # LOOP 3: Vary Gamma (Coverage Weight)
    # -----------------------------------------------------
    print("3. Varying Gamma (Coverage)...")
    for val in test_values:
        # gamma varies, others fixed at BASE
        res = run_single_optimization(BASE, BASE, val, BASE)
        plot_data["gamma"]["x"].append(val)
        plot_data["gamma"]["y"].append(res["cov"])

    # -----------------------------------------------------
    # LOOP 4: Vary Delta (Repulsion Weight)
    # -----------------------------------------------------
    print("4. Varying Delta (Repulsion)...")
    for val in test_values:
        # delta varies, others fixed at BASE
        res = run_single_optimization(BASE, BASE, BASE, val)
        plot_data["delta"]["x"].append(val)
        plot_data["delta"]["y"].append(res["rep"])

    # -----------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Impact of Weights on Drone Swarm Metrics\n(Fixed Params = {BASE}, Drones = {M})')

    # 1. Alpha vs LQ
    axs[0, 0].plot(plot_data["alpha"]["x"], plot_data["alpha"]["y"], 'b-o')
    axs[0, 0].set_title('Varying Alpha')
    axs[0, 0].set_xlabel('Alpha (Weight)')
    axs[0, 0].set_ylabel('Resulting Link Quality')
    axs[0, 0].grid(True)

    # 2. Beta vs Connectivity
    axs[0, 1].plot(plot_data["beta"]["x"], plot_data["beta"]["y"], 'g-o')
    axs[0, 1].set_title('Varying Beta')
    axs[0, 1].set_xlabel('Beta (Weight)')
    axs[0, 1].set_ylabel('Resulting Connectivity')
    axs[0, 1].grid(True)

    # 3. Gamma vs Coverage
    axs[1, 0].plot(plot_data["gamma"]["x"], plot_data["gamma"]["y"], 'r-o')
    axs[1, 0].set_title('Varying Gamma')
    axs[1, 0].set_xlabel('Gamma (Weight)')
    axs[1, 0].set_ylabel('Resulting Coverage (%)')
    axs[1, 0].grid(True)

    # 4. Delta vs Repulsion
    axs[1, 1].plot(plot_data["delta"]["x"], plot_data["delta"]["y"], 'k-o')
    axs[1, 1].set_title('Varying Delta')
    axs[1, 1].set_xlabel('Delta (Weight)')
    axs[1, 1].set_ylabel('Resulting Repulsion (Lower is better)')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()