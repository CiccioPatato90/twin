import numpy as np
data = np.load("results.npz")


# extract the results_conn, results_coverage, results_lq
results_conn = data['results_conn']
results_coverage = data['results_coverage']
results_lq = data['results_lq']
test_values = data['test_values']

# normalize the results_conn, results_coverage, results_lq
results_conn = results_conn / np.max(results_conn)
results_coverage = results_coverage / np.max(results_coverage)
results_lq = results_lq / np.max(results_lq)

# total_matrix = sum the three matrices
total_matrix = results_conn + results_coverage + results_lq

# for the max in matrix, return the index
max_index = np.argmax(total_matrix)
max_index = np.unravel_index(max_index, total_matrix.shape)
print("max_index: ", max_index)

# you effectively found the weights for the "best solution"
alpha = test_values[max_index[0]]
beta = test_values[max_index[1]]
gamma = test_values[max_index[2]]
delta = test_values[max_index[3]]
print("alpha: ", alpha)
print("beta: ", beta)
print("gamma: ", gamma)
print("delta: ", delta)