import math
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

# params
M = 15 # number of drones per solution
POP_SIZE = 100 # population size (number of solutions)
MAX_X = 50 # maximum x coordinate
MAX_Y = 50 # maximum y coordinate
alpha = 0.5 # mutation rate (weight)
beta = 0.3 # crossover rate (weight)
gamma = 0.2 # selection rate (weight)
delta = 0.5 # elitism rate (weight)
G_MAX = 200 # maximum number of generations
k = 0.02 # decadimento varianza di mutazione
theta_0 = 10 # varianza mutazione generazione 0
rho = 0.5 # penalty coefficient

# CONSTS
P_T = 5 # transmitted power
G_T = 1 # transmitter gain
G_R = 1 # receiver gain
spreading_factor = 2 # spreading factor (2 = free space path loss)
f = 2.4e9 # frequency
B = 1e6 # bandwidth in decibel. diffusione di banda
r = 7 # sensing radius
d_0 = 2.0 # minimum safe distance (reduced to allow closer spacing for better SNR)
d_min = d_0

# --- Physics / Helper Functions ---

def calculate_overall_noise_power_spectral_density(f):
    return 1.5

def calculate_absorption_coefficient(frequency):
    # Normalize frequency to GHz to get reasonable absorption coefficient
    return 0.0001 * (frequency / 1e9)

def calculate_received_power(dist_matrix, f):
    # Handle distance = 0 safely
    dist_safe = dist_matrix.copy()
    dist_safe[dist_safe == 0] = 1e-9 
    
    nom = P_T * G_T * G_R
    denom = (dist_safe ** spreading_factor) * (np.exp(-calculate_absorption_coefficient(f) * dist_safe))
    
    received = nom / denom
    return received

def calculate_snr(dist_matrix, f):
    received = calculate_received_power(dist_matrix, f)
    noise = calculate_overall_noise_power_spectral_density(f) * B
    return received / noise

def calculate_mean_link_quality(solution):
    # solution: (M, 2) array
    # dist_matrix: (M, M)
    diff = solution[:, np.newaxis, :] - solution[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    
    snr_matrix = calculate_snr(dist_matrix, f)
    
    # mean_link_quality += math.log(1 + snr) for i != j
    log_snr = np.log(1 + snr_matrix)
    np.fill_diagonal(log_snr, 0) # exclude self-loops
    
    total_lq = np.sum(log_snr)
    num_pairs = M * (M - 1)
    return (total_lq / num_pairs) if num_pairs > 0 else 0

def calculate_coverage(solution):
    # Vectorized coverage check
    # Create grid (MAX_X, MAX_Y) - cell centers
    grid_x, grid_y = np.meshgrid(np.arange(MAX_X), np.arange(MAX_Y), indexing='ij')
    
    # Add axis for broadcasting against drones (M)
    gx = grid_x[:, :, np.newaxis] # (MAX_X, MAX_Y, 1)
    gy = grid_y[:, :, np.newaxis]
    
    dx = solution[:, 0] # (M,)
    dy = solution[:, 1]
    
    # Use Euclidean distance from drone to cell center
    # Distance from each drone to each cell center
    dist_x = dx - gx  # (MAX_X, MAX_Y, M)
    dist_y = dy - gy
    dist_sq = dist_x**2 + dist_y**2
    
    # Check if any drone covers the cell (within sensing radius)
    covered_mask = dist_sq <= (r**2)
    cell_covered = np.any(covered_mask, axis=2) # (MAX_X, MAX_Y)
    
    covered_cells = np.sum(cell_covered)
    return covered_cells / (MAX_X * MAX_Y)

def calculate_repulsion_penalty(solution):
    diff = solution[:, np.newaxis, :] - solution[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    
    # Sum for i < j
    mask = np.triu(np.ones((M, M), dtype=bool), k=1)
    dists = dist_matrix[mask]
    
    penalties = np.exp(-dists / d_0)
    return np.sum(penalties)

def calculate_penalty_function(solution):
    diff = solution[:, np.newaxis, :] - solution[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((M, M), dtype=bool), k=1)
    dists = dist_matrix[mask]
    
    # Part 1: sum max(0, d_min - dist) for i < j
    sum1 = np.sum(np.maximum(0, d_min - dists))
    
    # Part 2: Count of non-zero distances between all pairs
    # (original logic: if w_i_j > 0 -> +1, where w = 1/norm)
    sum2 = np.sum(dist_matrix > 0)
    
    return sum1 + sum2

def global_payoff(solution):
    connectivity = 0
    mean_lq = calculate_mean_link_quality(solution)
    lq = alpha * mean_lq
    cov_ratio = calculate_coverage(solution)
    coverage = gamma * cov_ratio
    rep_penalty = calculate_repulsion_penalty(solution)
    repulsion = delta * rep_penalty

    # Debug: uncomment to see component values
    # print(f"LQ: {mean_lq:.4f} (weighted: {lq:.4f}), Coverage: {cov_ratio:.4f} (weighted: {coverage:.4f}), Repulsion: {rep_penalty:.4f} (weighted: {repulsion:.4f})")
    return lq + beta * connectivity + coverage - repulsion

def calculate_fitness(solution):
    return global_payoff(solution) - rho * calculate_penalty_function(solution)

def calculate_mean_snr(solution):
    """Calculate mean SNR for all drone pairs (excluding self-loops)."""
    diff = solution[:, np.newaxis, :] - solution[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    
    snr_matrix = calculate_snr(dist_matrix, f)
    # Exclude diagonal (self-loops)
    mask = ~np.eye(M, dtype=bool)
    snr_values = snr_matrix[mask]
    
    # Handle any NaN or inf values
    snr_values = snr_values[np.isfinite(snr_values)]
    
    return np.mean(snr_values) if len(snr_values) > 0 else 0.0

@dataclass
class Metrics:
    """Metrics collected at each generation."""
    generation: int
    normalized_fitness: float
    mean_snr: float
    mean_link_quality: float
    area_coverage: float
    
    def to_dict(self):
        return asdict(self)

# --- GA Logic ---

mating_pool = []
best_solution = None # (solution_array, fitness)
gen_number = 1

def init():
    global best_solution
    # Initialize POP_SIZE solutions (each with M drones)
    # Use integers for initial positions as in original (random.randint)
    for _ in range(POP_SIZE):
        solution = np.random.randint(0, MAX_X, size=(M, 2)).astype(float)
        solution[:, 1] = np.random.randint(0, MAX_Y, size=M)
        fitness = calculate_fitness(solution)
        mating_pool.append((solution, fitness))
        
        if best_solution is None or fitness > best_solution[1]:
            best_solution = (solution, fitness)

def selection():
    # Use tournament selection or fitness-proportional with scaling
    # For fitness-proportional, we need to handle negative fitnesses
    fitnesses = np.array([ind[1] for ind in mating_pool])
    
    # Shift fitnesses to be positive (min fitness -> 0)
    min_fitness = np.min(fitnesses)
    shifted_fitnesses = fitnesses - min_fitness + 1e-6  # Add small epsilon to avoid zeros
    
    total_fitness = np.sum(shifted_fitnesses)
    
    if total_fitness == 0 or np.isnan(total_fitness):
        p = np.ones(len(mating_pool)) / len(mating_pool)
    else:
        p = shifted_fitnesses / total_fitness
        
    indices = np.random.choice(len(mating_pool), size=2, p=p, replace=True)
    return [mating_pool[indices[0]][0].copy(), mating_pool[indices[1]][0].copy()]

def cross_over(parents):
    p1, p2 = parents
    # Beta per drone (same range as original: -0.25 to 1.25)
    betas = np.random.uniform(-0.25, 1.25, size=(M, 1))
    new_solution = p1 * betas + p2 * (1 - betas)
    # Clamp to bounds
    new_solution[:, 0] = np.clip(new_solution[:, 0], 0, MAX_X)
    new_solution[:, 1] = np.clip(new_solution[:, 1], 0, MAX_Y)
    return new_solution

def mutation(solution):
    sol = solution.copy()
    # Variance decreases exponentially with generation number
    variance = theta_0 * math.exp(-k * gen_number)
    # variance variable is used as sigma (std dev)
    noise = np.random.normal(0, variance, size=(M, 2))
    sol += noise
    # Clamp to bounds
    sol[:, 0] = np.clip(sol[:, 0], 0, MAX_X)
    sol[:, 1] = np.clip(sol[:, 1], 0, MAX_Y)
    return sol

def evaluate(solution):
    global best_solution
    fitness = calculate_fitness(solution)

    if best_solution is None or fitness > best_solution[1]:
        best_solution = (solution.copy(), fitness)
    
    return fitness

def plot_solution(ax, solution, generation, connection_threshold=150):
    if solution is None:
        return
    
    fitness = calculate_fitness(solution)
    x_coords = solution[:, 0]
    y_coords = solution[:, 1]
    
    ax.clear()
    ax.set_xlim(0, MAX_X)
    ax.set_ylim(0, MAX_Y)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f"Generation: {generation} | Fitness: {fitness:.2f}")
    ax.grid(True, alpha=0.3)
    
    # Draw connections
    # We can compute distances efficiently
    diff = solution[:, np.newaxis, :] - solution[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    
    for i in range(M):
        for j in range(i+1, M):
            if dist_matrix[i, j] < connection_threshold:
                ax.plot([solution[i, 0], solution[j, 0]],
                        [solution[i, 1], solution[j, 1]], 'k-', alpha=0.1, linewidth=0.5)
    
    ax.scatter(x_coords, y_coords, c='blue', s=50, label='Drones', zorder=5)
    ax.legend(loc='upper right')

def main():
    # Interactive mode
    # plt.ion() 
    # fig, ax = plt.subplots(figsize=(8, 8))
    
    init()

    fitness_history = []
    best_fitness_history = []
    metrics_history = []  # Store all metrics
    all_fitnesses = []  # Track all fitnesses for normalization
    
    global gen_number
    # Loop
    for t in tqdm(range(G_MAX)):
        # Create new generation
        new_population = []
        
        # Elitism: keep best solution
        if best_solution is not None:
            new_population.append((best_solution[0].copy(), best_solution[1]))
        
        # Generate remaining population
        while len(new_population) < POP_SIZE:
            parents = selection()
            new_chromosome = cross_over(parents)
            mutated_chromosome = mutation(new_chromosome)
            new_fitness = evaluate(mutated_chromosome)
            new_population.append((mutated_chromosome, new_fitness))
        
        # Replace old population with new one
        mating_pool.clear()
        mating_pool.extend(new_population)
        
        # Track fitness
        avg_fitness = np.mean([ind[1] for ind in mating_pool])
        fitness_history.append(avg_fitness)
        best_fitness_history.append(best_solution[1] if best_solution else 0)
        all_fitnesses.extend([ind[1] for ind in mating_pool])
        
        # Calculate metrics for best solution
        if best_solution is not None:
            best_sol = best_solution[0]
            
            # Calculate metrics
            mean_snr = calculate_mean_snr(best_sol)
            mean_link_quality = calculate_mean_link_quality(best_sol)
            coverage = calculate_coverage(best_sol) * 100  # Convert to percentage
            avg_fitness_current = avg_fitness
            
            # Normalize fitness: normalize average fitness across all generations
            # Use min-max normalization based on all fitnesses seen so far
            if len(all_fitnesses) > 1:
                min_fitness = min(all_fitnesses)
                max_fitness = max(all_fitnesses)
                if max_fitness != min_fitness:
                    normalized_fitness = (avg_fitness_current - min_fitness) / (max_fitness - min_fitness)
                else:
                    normalized_fitness = 0.0
            else:
                normalized_fitness = 0.0
            
            # Store metrics
            metrics = Metrics(
                generation=gen_number,
                normalized_fitness=normalized_fitness,
                mean_snr=mean_snr,
                mean_link_quality=mean_link_quality,
                area_coverage=coverage
            )
            metrics_history.append(metrics)
        
        gen_number += 1
        
        # Visualization
        # if gen_number % 2 == 0:
        #   plot_solution(ax, best_solution[0], gen_number)
        #   plt.pause(0.01)

    # plt.ioff()
    # plt.show()

    # Plot fitness evolution
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, label='Average Fitness')
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    fitness_filename = "fitness_evolution.png"
    plt.savefig(fitness_filename, dpi=300, bbox_inches='tight')
    
    print(f"Fitness evolution plot saved to: {fitness_filename}")
    
    # Plot metrics
    if metrics_history:
        generations = [m.generation for m in metrics_history]
        normalized_fitnesses = [m.normalized_fitness for m in metrics_history]
        mean_snrs = [m.mean_snr for m in metrics_history]
        mean_link_qualities = [m.mean_link_quality for m in metrics_history]
        coverages = [m.area_coverage for m in metrics_history]
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 16))
        
        # Normalized fitness
        axes[0].plot(generations, normalized_fitnesses, 'b-', linewidth=2)
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Normalized Fitness')
        axes[0].set_title('Normalized Fitness Score of Swarm')
        axes[0].grid(True, alpha=0.3)
        
        # Mean SNR
        axes[1].plot(generations, mean_snrs, 'g-', linewidth=2)
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Mean SNR')
        axes[1].set_title('Mean Signal-to-Noise Ratio')
        axes[1].grid(True, alpha=0.3)
        
        # Mean Link Quality
        axes[2].plot(generations, mean_link_qualities, 'm-', linewidth=2)
        axes[2].set_xlabel('Generation')
        axes[2].set_ylabel('Mean Link Quality')
        axes[2].set_title('Mean Link Quality (log(1+SNR))')
        axes[2].grid(True, alpha=0.3)
        
        # Area coverage
        axes[3].plot(generations, coverages, 'r-', linewidth=2)
        axes[3].set_xlabel('Generation')
        axes[3].set_ylabel('Coverage (%)')
        axes[3].set_title('Area Coverage')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim(0, 100)
        
        plt.tight_layout()
        metrics_filename = "metrics.png"
        plt.savefig(metrics_filename, dpi=300, bbox_inches='tight')
        
        
        print(f"Metrics plot saved to: {metrics_filename}")
        print(f"Total generations: {len(metrics_history)}")
        print(f"Final normalized fitness: {metrics_history[-1].normalized_fitness:.4f}")
        print(f"Final mean SNR: {metrics_history[-1].mean_snr:.4f}")
        print(f"Final mean link quality: {metrics_history[-1].mean_link_quality:.4f}")
        print(f"Final area coverage: {metrics_history[-1].area_coverage:.4f}")
    
    return best_solution

if __name__ == "__main__":
    main()
    print("final best fitness: ", best_solution[1])
    print("\n")
