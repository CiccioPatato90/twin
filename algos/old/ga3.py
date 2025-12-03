from dataclasses import dataclass
import math
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

# params
M = 5             # Number of drones (Genes per chromosome)
POP_SIZE = 30     # Population size (Number of solutions per generation) <-- ADDED
MAX_X = 50        # maximum x coordinate
MAX_Y = 50        # maximum y coordinate

# Weights
alpha = 0.5       # Link Quality weight
beta = 0.3        # Connectivity weight
gamma = 30       # Coverage weight
delta = 100       # Repulsion weight

G_MAX = 200       # maximum number of generations
k = 0.02          # decay of mutation variance
theta_0 = 10      # variance of mutation generation 0
rho = 0.5         # penalty coefficient

# CONSTS
P_T = 5           # transmitted power
G_T = 1           # transmitter gain
G_R = 1           # receiver gain
spreading_factor = 2 
f = 1             # frequency
B = 1             # bandwidth
r = 10            # sensing radius
d_0 = 5.0         # minimum safe distance 
d_min = d_0

@dataclass
class Position:
    """Struct to keep track of position."""
    x: float
    y: float
    @classmethod
    def init(cls):
      return Position(random.uniform(0, MAX_X), random.uniform(0, MAX_Y))
    @classmethod
    def new(cls, x, y):
      return Position(x,y)

def norm(i,j):
  return math.sqrt((i.x-j.x)**2 + (i.y-j.y)**2)

def calculate_overall_noise_power_spectral_density(f):
  return 0.01 * f

def calculate_absorption_coefficient(frequency):
  return 0.001 * frequency

def calculate_received_power(i,j,f):
  distance = norm(i,j)
  if distance == 0: return P_T * G_T * G_R # Max power if on top of each other
  
  nom = P_T * G_T * G_R
  # Added max(distance, 0.1) to prevent div by zero
  received = nom / ((max(distance, 0.1) ** spreading_factor) * (math.exp(-calculate_absorption_coefficient(f) * distance)))
  return received

def calculate_snr(i,j,f):
  received = calculate_received_power(i,j,f)
  noise = calculate_overall_noise_power_spectral_density(f) * B
  return received / noise

def calculate_mean_link_quality(solution):
  mean_link_quality = 0
  count = 0
  for i in range(M):
    for j in range(M):
      if i != j:
        val = calculate_snr(solution[i], solution[j], f)
        mean_link_quality += math.log(1 + val)
        count += 1
  if count == 0: return 0
  return (mean_link_quality / count)

def is_covered(x, y, solution):
    # CORRECTED: Standard Euclidean distance check
    for drone in solution:
        dist_sq = (drone.x - x)**2 + (drone.y - y)**2
        if dist_sq <= r**2:
            return True
    return False

def calculate_coverage(solution):
  # Note: This is computationally expensive (50x50 loop). 
  # For POP_SIZE=30, this runs 30 times per gen.
  covered_cells = 0
  # Optimization: Use step size or sampling if too slow, currently kept exact.
  for i in range(MAX_X):
    for j in range(MAX_Y):
      if is_covered(i, j, solution):
        covered_cells += 1
  return covered_cells / (MAX_X * MAX_Y)

def laplacian(weights, num_nodes):
  # CORRECTED: Combinatorial Laplacian (D - A)
  # This ensures eigenvalues represent algebraic connectivity correctly
  degrees = np.sum(weights, axis=1)
  D = np.diag(degrees)
  L = D - weights
  return L

def calc_connectivity(solution):
  weight_list = np.zeros((M, M))
  for i in range(M):
    for j in range(M):
      if i != j:
        d = norm(solution[i], solution[j])
        weight_list[i][j] = 1.0 / max(d, 0.1) # Avoid div/0
      else:
        weight_list[i][j] = 0
        
  matrix = laplacian(weight_list, M)
  eigenvals = np.linalg.eigvals(matrix)
  sorted_eigenvals = np.sort(np.real(eigenvals)) # specific real part
  
  # The second smallest eigenvalue is algebraic connectivity
  algebraic_connectivity = sorted_eigenvals[1]
  # Clamp negative float errors to 0
  return max(0.0, algebraic_connectivity)

def calculate_repulsion_penalty(solution):
  sum_val = 0
  for i in range(M):
    for j in range(M):
      if i < j:
        sum_val += math.exp(-(norm(solution[i], solution[j])/d_0))
  return sum_val

def calculate_penalty_function(solution):
  sum_val = 0
  for i in range(M):
    for j in range(M):
      if i < j:
        sum_val += max(0, d_min - norm(solution[i], solution[j]))
  return sum_val

def global_payoff(solution, record=False):
  # Calculate raw metrics
  lq = calculate_mean_link_quality(solution)
  connectivity = calc_connectivity(solution)
  coverage = calculate_coverage(solution)
  repulsion = calculate_repulsion_penalty(solution)

  # Only record metrics if specifically asked (usually for the best solution of generation)
  if record:
      metrics.connectivity.append(connectivity)
      metrics.link_quality.append(lq)
      metrics.coverage.append(coverage)
      metrics.repulsion.append(repulsion)

  # Weighted Sum
  # Note: connectivity often results in large numbers compared to coverage (0-1).
  # You might need to normalize these weights, but I kept your logic.
  obj = (alpha * lq) + (beta * connectivity) + (gamma * coverage) - (delta * repulsion)
  return obj

def calculate_fitness(solution, record=False):
  return global_payoff(solution, record) - rho * calculate_penalty_function(solution)

# ---------------------------------------------------------
# GA OPERATIONS
# ---------------------------------------------------------

def selection(current_population, fitnesses):
  # Tournament Selection (More stable than roulette for un-normalized fitness)
  tournament_size = 3
  indices = np.random.choice(len(current_population), size=tournament_size, replace=False)
  best_idx = indices[0]
  best_fit = fitnesses[best_idx]
  
  for idx in indices[1:]:
      if fitnesses[idx] > best_fit:
          best_fit = fitnesses[idx]
          best_idx = idx
          
  return copy.deepcopy(current_population[best_idx])

def cross_over(p1, p2):
  # Arithmetic Crossover
  new_chromosome = []
  # Random mix vector
  alpha_mix = np.random.uniform(-0.25, 1.25, size=M)
  
  for i in range(M):
    new_x = p1[i].x * alpha_mix[i] + p2[i].x * (1 - alpha_mix[i])
    new_y = p1[i].y * alpha_mix[i] + p2[i].y * (1 - alpha_mix[i])
    
    # Clip immediately
    new_x = np.clip(new_x, 0, MAX_X)
    new_y = np.clip(new_y, 0, MAX_Y)
    
    new_chromosome.append(Position.new(new_x, new_y))
  return new_chromosome

def mutation(chromosome, current_gen):
  mutated = copy.deepcopy(chromosome)
  variance = theta_0 * math.exp(-k * current_gen)
  
  for i in range(M):
    if random.random() < 0.2: # Probability of mutating a specific gene
        pos = mutated[i]
        pos.x += random.gauss(0, variance)
        pos.y += random.gauss(0, variance)
        pos.x = np.clip(pos.x, 0, MAX_X)
        pos.y = np.clip(pos.y, 0, MAX_Y)
  return mutated

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

@dataclass
class Metrics:
  connectivity: list[float]
  link_quality: list[float]
  coverage: list[float]
  repulsion: list[float]
  fitness: list[float]

metrics = Metrics([], [], [], [], [])
PLOT_SOLUTION = False 

def main():
  global metrics
  
  # 1. Initialization
  population = []
  for _ in range(POP_SIZE):
      sol = [Position.init() for _ in range(M)]
      population.append(sol)

  best_global_solution = None
  best_global_fitness = -float('inf')

  # 2. Main Loop
  for t in tqdm(range(G_MAX)):
    
    # Evaluate current population
    fitnesses = []
    for ind in population:
        fit = calculate_fitness(ind, record=False)
        fitnesses.append(fit)
    
    # Find best of this generation
    gen_best_val = max(fitnesses)
    gen_best_idx = fitnesses.index(gen_best_val)
    gen_best_sol = population[gen_best_idx]

    # Log metrics ONLY for the best solution of this generation
    # Recalculate with record=True
    calculate_fitness(gen_best_sol, record=True)
    metrics.fitness.append(gen_best_val)
    
    # Update global best
    if gen_best_val > best_global_fitness:
        best_global_fitness = gen_best_val
        best_global_solution = copy.deepcopy(gen_best_sol)

    # Elitism: Keep the single best
    next_population = [copy.deepcopy(gen_best_sol)]

    # Create rest of new population
    while len(next_population) < POP_SIZE:
        p1 = selection(population, fitnesses)
        p2 = selection(population, fitnesses)
        
        child = cross_over(p1, p2)
        child = mutation(child, t)
        
        next_population.append(child)
        
    population = next_population

  return best_global_solution, best_global_fitness

# Run
best_sol, best_fit = main()

# Plotting
plt.figure(figsize=(10, 6))

# Normalize metrics for plotting comparison
def normalize(data):
    return data
    # d_min = min(data)
    # d_max = max(data)
    # if d_max - d_min == 0: return [0.5 for _ in data]
    # return [(x - d_min) / (d_max - d_min) for x in data]

plt.plot(normalize(metrics.connectivity), label='Connectivity', alpha=0.7)
plt.plot(normalize(metrics.link_quality), label='Link Quality', alpha=0.7)
plt.plot(normalize(metrics.coverage), label='Coverage', alpha=0.7)
plt.plot(normalize(metrics.repulsion), label='Repulsion', alpha=0.7)
plt.plot(normalize(metrics.fitness), label='Fitness', linewidth=2, color='black')

plt.xlabel('Generation')
plt.ylabel('Normalized Metric')
plt.title('Optimization Progress (Best of Generation)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Final Best Fitness: {best_fit}")