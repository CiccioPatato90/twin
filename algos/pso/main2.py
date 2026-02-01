import copy
import math
import random
import sys

import numpy as np
from common2 import Logger
from simplex import nm

# ------- Setup ---------
ENABLE_NM = True
logger = Logger("hybrid_grie.txt")


def fitness_rastrigin(position):
    fitnessVal = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitnessVal += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    logger.log_best(fitnessVal, "SWARM")
    return fitnessVal


def fitness_griewank(position):
    sum_part = 0.0
    prod_part = 1.0

    for i in range(len(position)):
        xi = position[i]
        # Part 1: Summation term (x^2 / 4000)
        sum_part += (xi * xi) / 4000.0
        # Part 2: Product term (cos(x / sqrt(i)))
        # We use (i + 1) because Python index starts at 0, but math index starts at 1
        prod_part *= math.cos(xi / math.sqrt(i + 1))

    # Combine the parts: Sum - Product + 1
    fitnessVal = sum_part - prod_part + 1

    # Log and return
    logger.log_best(fitnessVal, "SWARM")
    return fitnessVal


# -------------------------


class Swarm:
    def __init__(self, n_particles, dim, fitness, minx, maxx):
        self.swarm = [Particle(fitness, dim, minx, maxx, i) for i in range(n_particles)]
        self.n_particles = n_particles
        self.dim = dim
        self.minx = minx
        self.maxx = maxx
        self.best_swarm_pos = [0.0 for i in range(dim)]
        self.best_swarm_fitnessVal = sys.float_info.max

    def best_solution(self):
        return self.best_swarm_pos, self.best_swarm_fitnessVal


class Particle:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(dim)]
        self.velocity = [0.0 for i in range(dim)]
        self.best_part_pos = [0.0 for i in range(dim)]
        self.best_part_fitnessVal = sys.float_info.max
        self.is_apprentice = False

        for i in range(dim):
            self.position[i] = (maxx - minx) * self.rnd.random() + minx
            self.velocity[i] = (maxx - minx) * self.rnd.random() + minx

        self.fitness = fitness(self.position)
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness

    def get_simplex(self):
        simplex = []
        # NOTE: If velocity is 0 (after a reset), we need a fallback radius
        # otherwise simplex will be a single point.
        vel_norm = np.linalg.norm(self.velocity)
        radius = vel_norm if vel_norm > 1e-9 else 0.1

        start_angle = random.uniform(0, 2 * math.pi)

        # FIX: Include the current position!
        # Otherwise we lose the particle's current location and "shoot out"
        # to the perimeter of the simplex.
        simplex.append(np.array(self.position))

        for i in range(2):
            theta = start_angle + i * (math.pi / 2)
            p = np.array(
                [
                    self.position[0] + radius * math.cos(theta),
                    self.position[1] + radius * math.sin(theta),
                ]
            )
            simplex.append(p)
        return simplex


def pso(swarm_ext, max_iter):
    # FIX: Deepcopy ensures we don't accidentally modify the previous swarm state
    swarm = copy.deepcopy(swarm_ext)

    # Sync logger to ensure we don't log spikes
    logger.sync_best(swarm.best_swarm_fitnessVal)

    w = 0.729
    c1 = 1.49445
    c2 = 1.49445
    rnd = random.Random(0)

    Iter = 0
    while Iter < max_iter:
        for i in range(swarm.n_particles):
            # Velocity Update
            for k in range(swarm.dim):
                r1 = rnd.random()
                r2 = rnd.random()

                swarm.swarm[i].velocity[k] = (
                    (w * swarm.swarm[i].velocity[k])
                    + (
                        c1
                        * r1
                        * (swarm.swarm[i].best_part_pos[k] - swarm.swarm[i].position[k])
                    )
                    + (c2 * r2 * (swarm.best_swarm_pos[k] - swarm.swarm[i].position[k]))
                )

            # Position Update
            for k in range(swarm.dim):
                swarm.swarm[i].position[k] += swarm.swarm[i].velocity[k]

            # Fitness
            swarm.swarm[i].fitness = fitness(swarm.swarm[i].position)

            # Update Personal Best
            if swarm.swarm[i].fitness < swarm.swarm[i].best_part_fitnessVal:
                swarm.swarm[i].best_part_fitnessVal = swarm.swarm[i].fitness
                swarm.swarm[i].best_part_pos = copy.copy(swarm.swarm[i].position)

            # Update Global Best
            if swarm.swarm[i].fitness < swarm.best_swarm_fitnessVal:
                swarm.best_swarm_fitnessVal = swarm.swarm[i].fitness
                swarm.best_swarm_pos = copy.copy(swarm.swarm[i].position)

        # Select Apprentice (Worst particle gets help, or best?)
        # Logic: Pick the best one to refine? Or the worst to rescue?
        # Your previous logic picked the MINIMUM (Best)
        current_best_val = float("inf")
        current_idx = -1
        for idx, p in enumerate(swarm.swarm):
            p.is_apprentice = False
            if p.fitness < current_best_val:
                current_best_val = p.fitness
                current_idx = idx

        if current_idx != -1:
            swarm.swarm[current_idx].is_apprentice = True

        Iter += 1
    return swarm


# ----------------------------
# Driver code

dim = 2
fitness = fitness_griewank

num_particles = 2
pso_iter = 10
nm_iter = 3

swarm = Swarm(num_particles, dim, fitness, -100.0, 100.0)
outer_iterations = 100

for outer in range(outer_iterations):
    new_swarm = pso(swarm, pso_iter)

    if ENABLE_NM:
        for i in range(new_swarm.n_particles):
            if new_swarm.swarm[i].is_apprentice:
                simplex = new_swarm.swarm[i].get_simplex()
                opt_position = nm(simplex, nm_iter, new_swarm.best_swarm_fitnessVal)

                # Update Position
                new_swarm.swarm[i].position = opt_position

                # FIX: ZERO VELOCITY to prevent slingshot
                new_swarm.swarm[i].velocity = [0.0 for _ in range(dim)]

                # Recalculate Fitness
                new_swarm.swarm[i].fitness = fitness(new_swarm.swarm[i].position)

                # Update Personal Best
                if new_swarm.swarm[i].fitness < new_swarm.swarm[i].best_part_fitnessVal:
                    new_swarm.swarm[i].best_part_fitnessVal = new_swarm.swarm[i].fitness
                    new_swarm.swarm[i].best_part_pos = copy.copy(
                        new_swarm.swarm[i].position
                    )

                # Update Global Best
                if new_swarm.swarm[i].fitness < new_swarm.best_swarm_fitnessVal:
                    new_swarm.best_swarm_fitnessVal = new_swarm.swarm[i].fitness
                    new_swarm.best_swarm_pos = copy.copy(new_swarm.swarm[i].position)
    swarm = new_swarm

print("\nBest solution found:", swarm.best_swarm_pos)
print("Fitness:", swarm.best_swarm_fitnessVal)
