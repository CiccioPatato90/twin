from dataclasses import dataclass
import math
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# params
M = 20 #population size
MAX_X = 100 #maximum x coordinate
MAX_Y = 100 #maximum y coordinate
alpha = 0.5 #mutation rate
beta = 0.5 #crossover rate
gamma = 0.5 #selection rate
delta = 0.5 #elitism rate
G_MAX = 100 #maximum number of generations
k = 2 #decadimento varianza di mutazione
theta_0 = 100 #varianza mutazione generazione 0
rho = 0.5 #penalty coefficient

# CONSTS
P_T = 5 # transmitted power
G_T = 1 # transmitter gain
G_R = 1 # receiver gain
spreading_factor = 2 # spreading factor
f = 2.4e9 # frequency
B = 1e6 # bandwidth in decibel. diffusione di banda
r = 1.5 # sensing radius
d_0 = 0.5 # minimum safe distance
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
  return 1.5

def calculate_absorption_coefficient(frequency):
  # Normalize frequency to GHz to get reasonable absorption coefficient
  return 0.0001 * (frequency / 1e9)

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
  # TODO: symmetric operation optimization
  mean_link_quality = 0
  for i in range(M):
    for j in range(M):
      if i != j:
        mean_link_quality += math.log(1 + calculate_snr(solution[i], solution[j],f))
  return (mean_link_quality / (M**2 - M))


# TODO: write this function
def is_covered(i,j,solution):
  for drone in range(M):
    # sensing radius
    if((max(abs(solution[drone].x - i),abs(solution[drone].x - i - 1)))**2 + (max(abs(solution[drone].y - j),abs(solution[drone].y - j - 1)))**2 <= (r**2)):
      return True
  return False

def calculate_coverage(solution):
  covered_cells = 0
  for i in range(MAX_X):
    for j in range(MAX_Y):
      # here we access the cell
      # clamp
      

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
  # TODO
  connectivity = 0
  lq = alpha * calculate_mean_link_quality(solution)
  coverage = gamma*calculate_coverage(solution)
  repulsion = delta*calculate_repulsion_penalty(solution)


  print("lq: ", lq, "\ncoverage: ", coverage, "\nrepulsion: ", repulsion)
  print("--------------------------------")
  return lq + beta * connectivity + coverage - repulsion


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
  mating_pool.append(best_solution)


def selection():
  p = []
  total_fitness = 0
  # dopo N interations we will have N items
  for i in mating_pool:
    fitness = i[1]
    total_fitness += fitness
  for i in mating_pool:
    p.append(i[1] / total_fitness)
  
  # print("probs: ",p)

  choice1 = np.random.choice(len(mating_pool), p=p)
  choice2 = np.random.choice(len(mating_pool), p=p)

  return [mating_pool[choice1][0], mating_pool[choice2][0]]


def cross_over(parents):
  # return a new chromosome (list of genes)
  new_chromosome = []
  for i in range(M):
    beta = random.uniform(-(1/4), (5/4))
    new_chromosome.append(Position.new(parents[0][i].x * beta + parents[1][i].x * (1-beta), parents[0][i].y * beta + parents[1][i].y * (1-beta)))
  return new_chromosome


def mutation(chromosome):
  for i in range(M):
    position = chromosome[i]
    variance = theta_0 * math.exp(-k * gen_number)
    position.x = position.x + random.gauss(0, variance)
    position.y = position.y + random.gauss(0, variance)
    chromosome[i] = position
  return chromosome

def evaluate(chromosome):
  # calculate chromosome fitness
  fitness = calculate_fitness(chromosome)
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

mating_pool = []
best_solution = None
gen_number = 1
def main():
  plt.ion() # Interactive mode on
  fig, ax = plt.subplots(figsize=(8, 8))
  # main algorithm loop
  init()

  for t in range(G_MAX):
    parents = selection()
    new_chromosome = cross_over(parents)
    mutated_chromosome = mutation(new_chromosome)
    evaluate(mutated_chromosome)

    global gen_number
    gen_number += 1

    # Visualization
    if gen_number % 2 == 0:
      plot_solution(ax, mutated_chromosome, gen_number)
      plt.pause(0.01)

  plt.ioff()
  plt.show()


  # print("starting best fitness: ", best_solution[1])
  # print("\n")
  # for g in tqdm(range(G_MAX)):
  #   parents = selection()
  #   new_chromosome = cross_over(parents)
  #   mutated_chromosome = mutation(new_chromosome)
  #   evaluate(mutated_chromosome)

  #   global gen_number
  #   gen_number += 1


  return best_solution

main()





print("final best fitness: ", best_solution[1])
print("\n")
