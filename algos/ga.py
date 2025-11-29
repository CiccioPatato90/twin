from dataclasses import dataclass
import math
import random

# params
M = 100 #population size
MAX_X = 100 #maximum x coordinate
MAX_Y = 100 #maximum y coordinate
alpha = 0.5 #mutation rate
beta = 0.5 #crossover rate
gamma = 0.5 #selection rate
delta = 0.5 #elitism rate
G_MAX = 100 #maximum number of generations
k = 2 #decadimento varianza di mutazione
theta_0 = 0.5 #varianza mutazione generazione 0
rho = 0.5 #penalty coefficient

# CONSTS
P_T = 5 # transmitted power
G_T = 1 # transmitter gain
G_R = 1 # receiver gain
spreading_factor = 2 # spreading factor
f = 2.4e9 # frequency
e = 2
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

def norm(i,j):
  return math.sqrt((i.x-j.x)**2 + (i.y-j.y)**2)

def calculate_overall_noise_power_spectral_density(f):
  return 1.5

def calculate_absorption_coefficient(frequency):
  return 0.0001 * frequency

def calculate_snr(i,j,f):
  return calculate_received_power(i,j,f) / (calculate_overall_noise_power_spectral_density(f) * B)

def calculate_received_power(i,j,f):
  distance = norm(i,j)
  print("distance", distance, "i", i, "j", j)
  print("-----")
  if(distance == 0):
    return 0
  received = P_T * G_T * G_R / (distance ** spreading_factor) * (e ** (-calculate_absorption_coefficient(f) * distance))
  return received

def calculate_mean_link_quality(solution):
  # TODO: symmetric operation optimization
  mean_link_quality = 0
  for i in range(M):
    for j in range(M):
      if i == j:
        continue
      mean_link_quality += math.log(1 + calculate_snr(solution[i], solution[j],f))
  # TODO: CHANGE WHEN OPTIMIZING
  return (mean_link_quality / (M**2 - M))


# TODO: write this function
# def is_covered(i,j,solution):
#   for drone in range(M):
#     if(solution[drone].x >= i-r and solution[drone].x <= i+r and solution[drone].y >= j-r and solution[drone].y <= j+r):
#       return True
#   return False

def calculate_coverage(solution):
  covered_cells = 0
  for i in range(MAX_X):
    for j in range(MAX_Y):
      # here we access the cell
      # clamp
      i_min = i- r
      i_max = i+ r

      j_min = j- r
      j_max = j+ r

      # check if there's a drone in the cell
      # if present
      for drone in range(M):
        if(solution[drone].x >= i_min and solution[drone].x <= i_max and solution[drone].y >= j_min and solution[drone].y <= j_max):
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
  return alpha * calculate_mean_link_quality(solution) + beta * connectivity + gamma*calculate_coverage(solution) - delta*calculate_repulsion_penalty(solution)


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
  best_solution = initial_solution
  mating_pool.append((initial_solution, calculate_fitness(initial_solution)))


def selection():
  p = []
  total_fitness = 0
  # dopo N interations we will have N items
  for i in mating_pool:
    fitness = i[1]
    total_fitness += fitness
  




mating_pool = []
best_solution = None
def main():
  # main algorithm loop
  init()
  print(mating_pool)
  for g in range(G_MAX):
    selection()
    cross_over()
    mutation()
    evaluate()
  return best_solution

main()
