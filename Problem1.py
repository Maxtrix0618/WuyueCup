# 2023 APMCM Wuyue Cup - Problem 1
import numpy as np
import kaiwu as kw

# the relative coordinates of neighbors (including themselves)
dx = [0, -1, 1, 0, 0]
dy = [0, 0, 0, -1, 1]
# number of bits
n = 16
# penalty for incorrect number of servers
L = 10 ** 3


# Marks conversion for bits
def n_(i, j):
    return 4 * i + j


def p_(z):
    return [int(z / 4), z % 4]


# Add v to the position(n(i1,j1),n(i2,j2)) of input H
def Hv(H, i1, j1, i2, j2, v):
    if 0 <= i1 < 4 and 0 <= j1 < 4 and 0 <= i2 < 4 and 0 <= j2 < 4:
        H[n_(i1, j1)][n_(i2, j2)] += v


# demand map
A = np.array([[36, 51, 54, 48],
              [20, 46, 34, 35],
              [63, 74,  5, 46],
              [54, 44, 27, 38]])

# 'negative demand satisfaction matrix', B
MB = [[0 for j in range(n)] for i in range(n)]

for i in range(4):
    for j in range(4):
        d = A[i][j]
        # diagonal element
        for k in range(5):
            Hv(MB, i + dx[k], j + dy[k], i + dx[k], j + dy[k], -d)
        # non-diagonal element
        for a in range(0, 5):
            for b in range(a+1, 5):
                Hv(MB, i + dx[a], j + dy[a], i + dx[b], j + dy[b], d)


# Create qubo variable
x = kw.qubo.ndarray(n, "x", kw.qubo.binary)


# Server limit penalty function
HA = L * (2 - kw.qubo.sum(x[a] for a in range(n))) ** 2
# Negative demand function
HB = kw.qubo.sum(kw.qubo.sum(x[a] * x[b] * MB[a][b] for b in range(n)) for a in range(n))
# Objective function (H)
obj = HA + HB


# Parse QUBO
obj = kw.qubo.make(obj)
# Convert to Ising model
obj_ising = kw.qubo.cim_ising_model(obj)
# Extract the Ising matrix
matrix = obj_ising.get_ising()["ising"]


# Perform calculation using CIM simulator
output = kw.cim.simulator(
                matrix,
                pump = 1.3,
                noise = 0.2,
                laps = 5000,
                dt = 0.05,
                normalization = 0.3,
                iterations = 50)

# Sort the results
opt = kw.sampler.optimal_sampler(matrix, output, bias = 0, negtail_ff = False)
# Select the best solution
cim_best = opt[0][0]
# If the linear term variable is -1, perform a flip
cim_best = cim_best * cim_best[-1]

# Print the spin value
print("spin: {}".format(cim_best))


# Get the list of variable names
var_s = obj_ising.get_variables()
# Substitute the spin vector and obtain the result dictionary
sol_dict = kw.qubo.get_sol_dict(cim_best, var_s)

snl_val = kw.qubo.get_val(HA, sol_dict)
print('servers_num_limit(HA): {}'.format(snl_val))

negative_demand_val = kw.qubo.get_val(HB, sol_dict)
print('negative_demand(-HB): {}'.format(negative_demand_val))


if snl_val != 0:
    print('\nInvalid path.')
    print('\nPlease try again.')
else:
    print('\nValid path.')
    print('Satisfied demand: ', end='')
    print(int(-negative_demand_val))

    # Get the numerical value matrix of x
    x_val = kw.qubo.get_array_val(x, sol_dict)
    # Find the indices of non-zero items
    nonzero_index = np.array(np.nonzero(x_val)).T
    orders = nonzero_index[:].flatten()

    # Print the path order
    print('The coordinates of the edge servers:')
    print('Number | (X-axis, Y-axis)')

    for o in range(len(orders)):
        co = p_(orders[o])
        print(o+1, end=' | (')
        print(co[1]+1, end=', ')
        print(co[0]+1, end=')\n')
