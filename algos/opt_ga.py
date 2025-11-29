import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import connected_components

# --- CONFIGURATION ---
# Environment
MAX_X, MAX_Y = 500.0, 500.0
AREA_SIZE = np.array([MAX_X, MAX_Y])
CENTER = AREA_SIZE / 2.0

# Drones (Online Constraints)
N_DRONES = 12
MAX_SPEED = 2.0         # m/s (Drones can't teleport)
DT = 1.0                # Time step duration (seconds)

# Physics (Acoustics)
FREQ = 25e3             # 25 kHz
P_T = 10.0              # 10 Watts
NOISE_FLOOR = 1e-9      # Baseline noise
SNR_THRESHOLD = 10.0    # 10 dB for a valid link

# GA Parameters (The "Brain")
POP_SIZE = 200
MUTATION_RATE = 0.7
MUTATION_SIGMA = 15.0   # Meters deviation
ALPHA = 0.6             # Weight: Link Quality
BETA = 0.4              # Weight: Connectivity (Lambda2)
GAMMA = 0.3             # Weight: Coverage
DELTA = 3.0             # Weight: Repulsion
PENALTY_DISCONNECT = 5.0 # Strong penalty if graph is broken

class OnlineSwarmController:
    def __init__(self):
        # 1. Initialize Swarm roughly in the center (Cluster start to ensure initial connectivity)
        # If we start random, we might start disconnected.
        self.drones = CENTER + np.random.normal(0, 50, (N_DRONES, 2))

        # 2. Initialize GA Population around the current physical drones
        # The population represents "Potential Future Configurations"
        self.population = np.tile(self.drones, (POP_SIZE, 1, 1))
        # Add some initial variance to the thoughts
        self.population += np.random.normal(0, 10, (POP_SIZE, N_DRONES, 2))

        # Grid for coverage
        self.grid = np.mgrid[0:MAX_X:20, 0:MAX_Y:20].reshape(2, -1).T

        # History
        self.history_l2 = []
        self.history_snr = []

    def get_physics_metrics(self, config, noise_level):
        """
        Calculates physics for a specific configuration.
        """
        # Distance Matrix
        d_ij = distance_matrix(config, config)
        np.fill_diagonal(d_ij, np.inf)

        # Acoustic Path Loss (Thorp + Spreading)
        f_khz = FREQ / 1000.0
        alpha_db_km = 0.11*f_khz**2/(1+f_khz**2) + 44*f_khz**2/(4100+f_khz**2) + 0.003
        alpha_np_m = (alpha_db_km * 0.1151) / 1000.0

        numerator = P_T
        denominator = (d_ij ** 1.5) * np.exp(alpha_np_m * d_ij)
        P_rx = numerator / denominator

        # SNR
        snr = P_rx / (noise_level * 4000.0)
        snr_db = 10 * np.log10(snr + 1e-20)

        # 1. Connectivity (Lambda 2 + Components)
        adj_matrix = (snr_db > SNR_THRESHOLD).astype(int)
        degrees = np.sum(adj_matrix, axis=1)
        laplacian = np.diag(degrees) - adj_matrix
        eig = np.linalg.eigvalsh(laplacian)
        lambda_2 = eig[1] if len(eig) > 1 else 0.0

        # Check disconnected components (to fix the "Constant 0" issue)
        n_components, labels = connected_components(adj_matrix, directed=False)

        # 2. Coverage (Sampled)
        d_grid = distance_matrix(config, self.grid)
        covered = np.sum(np.min(d_grid, axis=0) < 80.0) / len(self.grid)

        # 3. Quality
        mean_snr = np.mean(snr_db[snr_db > SNR_THRESHOLD]) if np.any(snr_db > SNR_THRESHOLD) else 0

        # 4. Repulsion
        repulsion = np.sum(np.exp(-d_ij / 5.0))

        return lambda_2, n_components, covered, mean_snr, repulsion

    def optimize_one_step(self, noise_level):
        """
        Runs ONE generation of GA to find the best immediate move.
        This is the 'Online' part.
        """
        fitness = []

        for ind in self.population:
            l2, n_comp, cov, q, rep = self.get_physics_metrics(ind, noise_level)

            # FITNESS FUNCTION
            # If broken (n_comp > 1), we penalize heavily so GA prioritizes reconnecting
            score = (ALPHA * q/100) + (BETA * l2) + (GAMMA * cov) - (DELTA * rep)
            if n_comp > 1:
                score -= (n_comp * PENALTY_DISCONNECT)

            fitness.append(score)

        fitness = np.array(fitness)

        # Selection (Tournament)
        best_idx = np.argmax(fitness)
        target_config = self.population[best_idx].copy()

        # Log stats of the chosen target
        l2_best, _, _, q_best, _ = self.get_physics_metrics(target_config, noise_level)
        self.history_l2.append(l2_best)

        # Evolve Population for Next Time Step
        # (This keeps the "brain" running)
        p1 = np.random.randint(0, POP_SIZE, POP_SIZE)
        p2 = np.random.randint(0, POP_SIZE, POP_SIZE)
        winners = np.where((fitness[p1] > fitness[p2])[:, None, None],
                           self.population[p1], self.population[p2])

        # Crossover
        children = (winners + np.roll(winners, 1, axis=0)) / 2.0

        # Mutation: We mutate around the WINNERS, but we must also
        # keep the population somewhat tethered to the PHYSICAL reality.
        # We inject the current physical drone positions to keep the search local.
        children[-1] = self.drones # Inject current reality

        noise = np.random.normal(0, MUTATION_SIGMA, (POP_SIZE, N_DRONES, 2))
        children += noise
        self.population = np.clip(children, 0, MAX_X)

        return target_config

    def update_physics(self, target_pos):
        """
        Moves the physical drones toward the target calculated by GA.
        Limited by MAX_SPEED.
        """
        direction = target_pos - self.drones
        dist = np.linalg.norm(direction, axis=1)

        # Normalize and scale by max speed
        with np.errstate(divide='ignore', invalid='ignore'):
            move_vector = (direction / dist[:, None]) * MAX_SPEED

        # If close to target, just snap to it (avoid jitter)
        move_vector = np.where(dist[:, None] < MAX_SPEED, direction, move_vector)

        self.drones += move_vector
        self.drones = np.clip(self.drones, 0, MAX_X)

# --- RUNNING THE ONLINE SIMULATION ---
sim = OnlineSwarmController()
noise_scenario = NOISE_FLOOR

print("Starting Online Simulation...")
print("(Drones are physically moving towards GA targets)")

plt.ion() # Interactive mode on
fig, ax = plt.subplots(figsize=(6, 6))

for t in range(100):
    # SCENARIO: At t=50, Noise spikes! (Simulating a storm or jammer)
    # The swarm should react by clustering tighter.
    if t == 50:
        print("\n!!! NOISE EVENT DETECTED (Jammer) !!!")
        noise_scenario = NOISE_FLOOR * 1000.0

    # 1. PLAN: GA finds best configuration for *current* noise
    target = sim.optimize_one_step(noise_scenario)

    # 2. ACT: Drones move physically
    sim.update_physics(target)

    # Visualization
    if t % 2 == 0:
        ax.clear()
        ax.set_xlim(0, MAX_X)
        ax.set_ylim(0, MAX_Y)
        ax.set_title(f"Time: {t}s | Noise: {'HIGH' if t>=50 else 'LOW'}\n$\lambda_2$: {sim.history_l2[-1]:.2f}")

        # Draw Drones
        ax.scatter(sim.drones[:,0], sim.drones[:,1], c='blue', s=50, label='Drones')

        # Draw Target Ghost (Where they want to go)
        ax.scatter(target[:,0], target[:,1], c='green', s=10, alpha=0.5, label='GA Target')

        # Draw Connections
        l2, _, _, _, _ = sim.get_physics_metrics(sim.drones, noise_scenario)
        # Re-calc adjacency for plotting
        dists = distance_matrix(sim.drones, sim.drones)
        # Note: Just for vis, rough check
        for i in range(N_DRONES):
            for j in range(i+1, N_DRONES):
                if dists[i,j] < 150: # Rough visual line
                    ax.plot([sim.drones[i,0], sim.drones[j,0]],
                            [sim.drones[i,1], sim.drones[j,1]], 'k-', alpha=0.1)

        ax.legend(loc='upper right')
        plt.pause(0.01)

plt.ioff()
plt.show()
