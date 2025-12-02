from dataclasses import dataclass
import math
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# params
M = 5 #population size
MAX_X = 50 #maximum x coordinate
MAX_Y = 50 #maximum y coordinate
alpha = 0.5 # mutation rate (weight)
beta = 0.3 # crossover rate (weight)
gamma = 0.2 # selection rate (weight)
delta = 0.5 # elitism rate (weight)
G_MAX = 200 #maximum number of generations
k = 0.02 # decay of mutation variance
theta_0 = 10 # variance of mutation generation 0
rho = 0.5 #penalty coefficient

# CONSTS
P_T = 5 # transmitted power
G_T = 1 # transmitter gain
G_R = 1 # receiver gain
spreading_factor = 2 # spreading factor
f = 1 # frequency
B = 1 # bandwidth in decibel. diffusione di banda
r = 10 # sensing radius
d_0 = 2.0 # minimum safe distance (reduced to allow closer spacing for better SNR)
d_min = d_0

@dataclass
class Area:
  cells: list

  def get_cell(self, x, y):
    return self.cells[x][y]

@dataclass
class Position:
    """Struct to keep track of position."""
    x: float
    y: float
    @classmethod
    def init(cls):
      return Position(random.randint(0, MAX_X), random.randint(0, MAX_Y))
    def new(x,y):
      return Position(x,y)

def norm(i,j):
  return math.sqrt((i.x-j.x)**2 + (i.y-j.y)**2)

def calculate_overall_noise_power_spectral_density(f):
  # Simple thermal noise model: N0 = k*T (approximately constant)
  # For simplicity, use a small constant that scales with frequency
  return 0.01 * f

def calculate_absorption_coefficient(frequency):
  # Simple absorption model: increases with frequency
  # In free space, absorption is very small; in other media it increases
  # Returns absorption per unit distance
  return 0.001 * frequency

def calculate_snr(i,j,f):
  received = calculate_received_power(i,j,f)
  noise = calculate_overall_noise_power_spectral_density(f) * B
  return received / noise

def calculate_received_power(i,j,f):
  distance = norm(i,j)
  assert distance > 0, "Distance is 0 at gen_number: " + str(gen_number)
  nom = P_T * G_T * G_R
  assert nom > 0, "Nom is 0 at gen_number: " + str(gen_number)
  
  received = nom / ((distance ** spreading_factor) * (math.exp(-calculate_absorption_coefficient(f) * distance)))
  assert received > 0, "Received is 0 at gen_number: " + str(gen_number)
  return received

def calculate_mean_link_quality(solution):
  mean_link_quality = 0
  for i in range(M):
    for j in range(M):
      if i != j:
        mean_link_quality += math.log(1 + calculate_snr(solution[i], solution[j],f))
  return (mean_link_quality / (M**2 - M))


def is_covered(i,j,solution):
  for drone in range(M):
    # sensing radius
    if((max(abs(solution[drone].x - i),abs(solution[drone].x - i - 1)))**2 + (max(abs(solution[drone].y - j),abs(solution[drone].y - j - 1)))**2 <= (r**2)):
      return True
  return False


def laplacian(weights, num_nodes):
  L = np.zeros((num_nodes, num_nodes))
  for i in range(num_nodes):
    for j in range(num_nodes):
      if i != j:
        # non diagonal elements
        L[i][j] = (-weights[i][j] / sum(weights[i]))
      else:
        # diagonal elements
        L[i][i] = (sum(weights[i]) / sum(weights[i]))
  return L


def calc_connectivity(solution):
  weight_list = np.zeros((M, M))
  for i in range(M):
    for j in range(M):
      if i != j:
        weight_list[i][j] = 1/(norm(solution[i], solution[j]))
      else:
        weight_list[i][j] = 0
  matrix = laplacian(weight_list, M)
  eigenvals = np.linalg.eigvals(matrix)
  sorted_eigenvals = np.sort(eigenvals)
  # For a connected graph, the Laplacian has exactly one zero eigenvalue
  # The second smallest eigenvalue (algebraic connectivity) should be > 0 for connected graphs
  # sorted_eigenvals[0] should be ~0 (smallest), sorted_eigenvals[1] is the algebraic connectivity
  algebraic_connectivity = sorted_eigenvals[1]
  assert algebraic_connectivity > 1e-10, f"Graph is not connected (algebraic connectivity = {algebraic_connectivity})"
  return algebraic_connectivity

def calculate_coverage(solution):
  covered_cells = 0
  for i in range(MAX_X):
    for j in range(MAX_Y):
      # check if there's a drone in the cell
      # if present
      if(is_covered(i,j,solution)):
        covered_cells += 1

  return covered_cells/(MAX_X*MAX_Y)


def calculate_repulsion_penalty(solution):
  sum = 0
  for i in range(M):
    for j in range(M):
      if(i<j):
        sum += math.exp(-(norm(solution[i], solution[j])/d_0))
  return sum

def global_payoff(solution):
  lq = alpha * calculate_mean_link_quality(solution)
  connectivity = beta*calc_connectivity(solution)
  coverage = gamma*calculate_coverage(solution)
  repulsion = delta*calculate_repulsion_penalty(solution)

  global metrics
  metrics.connectivity.append(connectivity)
  metrics.link_quality.append(lq)
  metrics.coverage.append(coverage)
  metrics.repulsion.append(repulsion)

  print("lq: ", lq, "\ncoverage: ", coverage, "\nrepulsion: ", repulsion, "\nconnectivity: ", connectivity)
  print("--------------------------------")
  return lq + connectivity + coverage - repulsion


def calculate_penalty_function(solution):
  sum = 0
  for i in range(M):
    for j in range(M):
      if(i<j):
        sum += max(0, d_min - norm(solution[i], solution[j]))
  for i in range(M):
    internal_sum = 0
    for j in range(M):
      norm_i_j = norm(solution[i], solution[j])
      if(norm_i_j == 0):
        continue
      w_i_j = 1 / norm_i_j
      if(w_i_j > 0):
        internal_sum += 1
    sum += internal_sum

  return sum

def calculate_fitness(solution):
  return global_payoff(solution) - rho*calculate_penalty_function(solution)

def init():
  # init best_solution and mating_pool
  initial_solution = []
  for i in range(M):
    initial_solution.append(Position.init())
  global best_solution
  best_solution = (initial_solution, calculate_fitness(initial_solution))
  global mating_pool
  mating_pool.append(best_solution)


# def selection():
#   p = []
#   total_fitness = 0
#   # dopo N interations we will have N items
#   global mating_pool
#   for i in mating_pool:
#     fitness = i[1]
#     total_fitness += fitness
#   for i in mating_pool:
#     p.append(i[1] / total_fitness)
  
#   # print("probs: ",p)

#   choice1 = np.random.choice(len(mating_pool), p=p)
#   choice2 = np.random.choice(len(mating_pool), p=p)
#   return [mating_pool[choice1][0], mating_pool[choice2][0]]
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
  return [mating_pool[indices[0]][0][:], mating_pool[indices[1]][0][:]]


# def cross_over(parents):
#   # return a new chromosome (list of genes)
#   new_chromosome = []
#   for i in range(M):
#     beta = random.uniform(-(1/4), (5/4))
#     new_chromosome.append(Position.new(parents[0][i].x * beta + parents[1][i].x * (1-beta), parents[0][i].y * beta + parents[1][i].y * (1-beta)))
#   return new_chromosome
def cross_over(parents):
  p1, p2 = parents
  # Beta per drone (same range as original: -0.25 to 1.25)
  betas = np.random.uniform(-0.25, 1.25, size=(M, 1))
  # Convert Position objects to numpy arrays
  p1_arr = np.array([[pos.x, pos.y] for pos in p1])
  p2_arr = np.array([[pos.x, pos.y] for pos in p2])
  # new_chromosome.append(Position.new(parents[0][i].x * beta + parents[1][i].x * (1-beta), parents[0][i].y * beta + parents[1][i].y * (1-beta)))
  new_solution = p1_arr * betas + p2_arr * (1 - betas)
  # Clamp to bounds
  new_solution[:, 0] = np.clip(new_solution[:, 0], 0, MAX_X)
  new_solution[:, 1] = np.clip(new_solution[:, 1], 0, MAX_Y)
  # Convert back to Position objects
  new_chromosome = [Position.new(new_solution[i, 0], new_solution[i, 1]) for i in range(M)]
  return new_chromosome


def mutation(chromosome):
  for i in range(M):
    position = chromosome[i]
    variance = theta_0 * math.exp(-k * gen_number)
    position.x = position.x + random.gauss(0, variance)
    position.y = position.y + random.gauss(0, variance)
    position.x = np.clip(position.x, 0, MAX_X)
    position.y = np.clip(position.y, 0, MAX_Y)
    chromosome[i] = position
  return chromosome

def evaluate(chromosome):
  # calculate chromosome fitness
  fitness = calculate_fitness(chromosome)
  global mating_pool
  mating_pool.append((chromosome, fitness))

  global best_solution
  if(fitness >= best_solution[1]):
    best_solution = (chromosome, fitness)
  
  return fitness


def plot_solution(ax, solution, generation, connection_threshold=150):
  """Plot a solution showing drone positions and connections."""
  if solution is None or len(solution) == 0:
    return
  
  # Calculate fitness for the current solution
  fitness = calculate_fitness(solution)
  
  # Extract x and y coordinates from Position objects
  x_coords = [pos.x for pos in solution]
  y_coords = [pos.y for pos in solution]
  
  # Clear and set up the plot
  ax.clear()
  ax.set_xlim(0, MAX_X)
  ax.set_ylim(0, MAX_Y)
  ax.set_xlabel('X coordinate')
  ax.set_ylabel('Y coordinate')
  ax.set_title(f"Generation: {generation} | Fitness: {fitness:.2f}")
  ax.grid(True, alpha=0.3)
  
  # Draw connections between nearby drones
  for i in range(M):
    for j in range(i+1, M):
      distance = norm(solution[i], solution[j])
      if distance < connection_threshold:
        ax.plot([solution[i].x, solution[j].x],
                [solution[i].y, solution[j].y], 'k-', alpha=0.1, linewidth=0.5)
  
  # Draw drones
  ax.scatter(x_coords, y_coords, c='blue', s=50, label='Drones', zorder=5)
  
  ax.legend(loc='upper right')


@dataclass
class Metrics:
  connectivity: list[float]
  link_quality: list[float]
  coverage: list[float]
  repulsion: list[float]
  fitness: list[float]

metrics = Metrics(connectivity=[], link_quality=[], coverage=[], repulsion=[], fitness=[])
mating_pool = []
best_solution = None
gen_number = 1
PLOT_SOLUTION = False
def main():
  if PLOT_SOLUTION:
    plt.ion() # Interactive mode on
    fig, ax = plt.subplots(figsize=(8, 8))
  # main algorithm loop
  init()

  for t in range(G_MAX):
    parents = selection()
    new_chromosome = cross_over(parents)
    mutated_chromosome = mutation(new_chromosome)
    # mutated_chromosome = mutation(best_solution[0])
    new_fitness = evaluate(mutated_chromosome)

    global metrics
    metrics.fitness.append(new_fitness)

    global gen_number
    gen_number += 1

    # Visualization
    if gen_number % 2 == 0 and PLOT_SOLUTION:
      plot_solution(ax, mutated_chromosome, gen_number)
      plt.pause(0.01)

  if PLOT_SOLUTION:
    plt.ioff()
    plt.show()


  return best_solution

main()

# plot metrics
plt.figure(figsize=(10, 6))
plt.plot(metrics.connectivity, label='Connectivity')
plt.plot(metrics.link_quality, label='Link Quality')
plt.plot(metrics.coverage, label='Coverage')
plt.plot(metrics.repulsion, label='Repulsion')

normalized_fitness = (metrics.fitness - min(metrics.fitness)) / (max(metrics.fitness) - min(metrics.fitness))
plt.plot(normalized_fitness, label='Fitness')
plt.xlabel('Generation')
plt.ylabel('Normalized Metric')
plt.title('Normalized Metrics')
plt.legend()
plt.show()





print("final best fitness: ", best_solution[1])
print("\n")
