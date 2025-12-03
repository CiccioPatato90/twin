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

def on_generation(ga_instance):
    """
    Callback running at the end of every generation.
    We use this to log the specific sub-metrics of the BEST solution.
    """
    best_sol, best_fit, _ = ga_instance.best_solution()
    decoded = decode_solution(best_sol)
    
    # Re-calc specific stats for logging
    history["fitness"].append(best_fit)
    history["connectivity"].append(calc_connectivity(decoded))
    history["coverage"].append(calculate_coverage(decoded))
    history["link_quality"].append(calculate_mean_link_quality(decoded))
    history["repulsion"].append(calculate_repulsion_penalty(decoded))
    
    # print(f"Gen {ga_instance.generations_completed} | Fitness: {best_fit:.4f}")

# ---------------------------------------------------------
# 4. MAIN EXECUTION
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
        on_generation=on_generation,
        suppress_warnings=True,
    )

    print("Starting PyGAD Optimization...")
    ga_instance.run()

    # ---------------------------------------------------------
    # 5. VISUALIZATION
    # ---------------------------------------------------------
    # ga_instance.summary()


    # 1. Metric Evolution
    # plt.figure(figsize=(10, 6))
    
    # Normalize for clean plotting
    # def norm_data(d):
    #     return (d - np.min(d)) / (np.max(d) - np.min(d) + 1e-6)

    # plt.plot(norm_data(history['connectivity']), label='Connectivity')
    # plt.plot(norm_data(history['link_quality']), label='Link Quality')
    # plt.plot(norm_data(history['coverage']), label='Coverage')
    # plt.plot(norm_data(history['repulsion']), label='Repulsion')
    # plt.plot(norm_data(history['fitness']), 'k--', linewidth=2, label='Fitness')
    
    # plt.title("Normalized Metrics over Generations")
    # plt.xlabel("Generation")
    # plt.legend()
    # plt.show()

    # 2. Final Drone Positions
    best_sol, _, _ = ga_instance.best_solution()
    final_pos = decode_solution(best_sol)



    # plt.figure(figsize=(6, 6))
    # plt.xlim(0, MAX_X)
    # plt.ylim(0, MAX_Y)
    # plt.grid(True, alpha=0.3)
    # xs = [p.x for p in final_pos]
    # ys = [p.y for p in final_pos]
    # plt.scatter(xs, ys, s=100, c='red', label='Drones')
    # # Draw connections
    # for i in range(M):
    #     for j in range(i+1, M):
    #         plt.plot([final_pos[i].x, final_pos[j].x], 
    #                  [final_pos[i].y, final_pos[j].y], 'b-', alpha=0.1)
    # # Draw sensing radius
    # ax = plt.gca()
    # for p in final_pos:
    #     circle = plt.Circle((p.x, p.y), r, color='green', fill=False, linestyle='--', alpha=0.5)
    #     ax.add_patch(circle)
    # plt.title("Final Drone Placement")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    
    test_values = np.linspace(0, 10, 3).astype(int) 
    for val in test_values:
        for val2 in test_values:
            for val3 in test_values:
                for val4 in test_values:
                    global alpha, beta, gamma, delta
                    alpha = float(val)
                    beta = float(val2)
                    gamma = float(val3)
                    delta = float(val4)
                    print(f"Running with alpha: {alpha}, beta: {beta}, gamma: {gamma}, delta: {delta}")
                    # main()
                    # main(val, val2, val3, val4)
                    # valore di lq medio per alpha = 0 
                    # valore di lq medio per alpha = 10    

    sys.exit(0)
    

    