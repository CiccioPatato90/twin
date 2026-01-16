import math

import numpy as np


def cost(position):
    # note: const = -fitness
    fitnessVal = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitnessVal += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitnessVal


def nm(simplex, max_iterations):
    evaluations = 0

    # Need a loop to iterate
    for iteration in range(max_iterations):
        # 1. Sort [cite: 11]
        simplex.sort(key=cost)
        evaluations += 1

        print(simplex)

        u = simplex[0]  # Best
        v = simplex[-2]  # Next-to-worst
        w = simplex[-1]  # Worst

        # 2. Reflect [cite: 12]
        c = np.mean(simplex[:-1], axis=0)  # Centroid of u and v
        r = 2 * c - w
        cost_r = cost(r)

        # Logic: f(u) <= f(r) < f(v)
        if cost(u) <= cost_r < cost(v):
            simplex[-1] = r
            print("Used Reflected point")

        # 3. Extend: f(r) < f(u)
        elif cost_r < cost(u):  # <--- FIXED THIS SIGN (was >)
            e = c + 2 * (c - w)
            if cost(e) < cost_r:  # [cite: 15]
                simplex[-1] = e
                print("Used Extended point")
            else:
                simplex[-1] = r  # [cite: 16]
                print("Used Reflected (extension failed)")

        # 4. Contract/Shrink
        else:
            # We are here because f(r) >= f(v) [cite: 17]
            path_vector = r - w
            c_i = w + 0.25 * path_vector
            c_o = w + 0.75 * path_vector

            f_ci = cost(c_i)
            f_co = cost(c_o)

            # Pick better contraction point
            if f_ci < f_co:
                best_c = c_i
                best_c_cost = f_ci
            else:
                best_c = c_o
                best_c_cost = f_co

            # 4a. Check against next-to-worst [cite: 23]
            if best_c_cost < cost(v):
                simplex[-1] = best_c
                print("Used Contraction point")
            else:
                # 4b. Shrink [cite: 24]
                # "Shrink the simplex into the best point u"
                # Note: You need to update ALL points except u
                print("Shrinking simplex")
                for i in range(1, len(simplex)):
                    simplex[i] = u + 0.5 * (simplex[i] - u)  # [cite: 87]

        # ---------------------------------------------------------
        # 5. Check Convergence [cite: 25]
        # ---------------------------------------------------------
        epsilon_x = 1e-4  # Tolerance for position
        epsilon_f = 1e-4  # Tolerance for cost

        # Re-sort to ensure u, v, w are current for the check
        simplex.sort(key=cost)
        u = np.array(simplex[0])  # Best
        v = np.array(simplex[-2])  # Next-to-worst
        w = np.array(simplex[-1])  # Worst

        # Check 1: Simplex Size
        # "max |[v,w] - [u,v]| < epsilon_x"
        # We calculate the distance of v and w from u
        dist_v = np.linalg.norm(v - u)
        dist_w = np.linalg.norm(w - u)
        max_dist = max(dist_v, dist_w)

        # Check 2: Function Value Difference
        # "max |f(u) - [f(v),f(w)]| < epsilon_f"
        diff_fv = abs(cost(v) - cost(u))
        diff_fw = abs(cost(w) - cost(u))
        max_cost_diff = max(diff_fv, diff_fw)

        if max_dist < epsilon_x and max_cost_diff < epsilon_f:
            # best_cost = cost(simplex[0])
            # best_pos = None
            # for pos in simplex:
            #     print("pos: ", pos)
            #     new_cost = cost(pos)
            #     if new_cost > best_cost:
            #         best_pos = pos
            #         best_cost = new_cost
            # return best_pos, evaluations, cost(best_cost)
            return simplex, evaluations
    return simplex, evaluations


# simplex = [
#     np.array([-1.2, 1.0]),  # Start Point
#     np.array([-1.0, 1.0]),  # Neighbor 1
#     np.array([-1.2, 1.2]),  # Neighbor 2
# ]
# result = nm(simplex, 100)
# print(f"Converged! reuslt: {result}")
