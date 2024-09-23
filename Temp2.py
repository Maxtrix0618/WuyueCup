# 2023 APMCM Wuyue Cup - Problem 2
import numpy as np
import kaiwu as kw


def euclidean_distance(i1, j1, i2, j2):
    return round(((i1 - i2) ** 2 + (j1 - j2) ** 2) ** 0.5, 2)


# determines whether (i, j) is covered by the scope of edge server r
def isCoveredBy(r, i, j):
    return euclidean_distance(Ei[r], Ej[r], i, j) <= 3


# return 1 if (i, j) is covered by the scope of r, else 0
def Q(r, i, j):
    return 1 if isCoveredBy(r, i, j) else 0


# penalty coefficients
L1 = 10 ** 6
L2 = 10 ** 6
L3 = 10 ** 3

# transmission cost and calculation cost
kT_C = 2
kT_E = 1
kT_CE = 1
kC_C = 1
kC_E = 2

# edge server capacity
Cap = 12

# cloud server's coordinate
Cc = [4, 0]
Ci = Cc[1] - 1
Cj = Cc[0] - 1

# edge servers' coordinates and fixed cost
F = [70, 52, 56, 64, 40]
Ex = [6, 2, 4, 6, 2]
Ey = [1, 3, 5, 5, 1]
Ei = [0 for i in range(5)]
Ej = [0 for j in range(5)]
for i in range(5):
    Ei[i] = Ey[i] - 1
for j in range(5):
    Ej[j] = Ex[j] - 1

# distance between users and cloud server
Dc = [[0 for j in range(6)] for i in range(6)]
# distance between users and edge servers
D = [[[0 for j in range(6)] for i in range(6)] for r in range(5)]
# distance between edge servers and cloud server
Dce = [0 for r in range(5)]
for i in range(6):
    for j in range(6):
        Dc[i][j] = euclidean_distance(i, j, Ci, Cj)
        for r in range(5):
            D[r][i][j] = euclidean_distance(i, j, Ei[r], Ej[r])
            Dce[r] = euclidean_distance(Ei[r], Ej[r], Ci, Cj)

# demand map
A = np.array([[0, 0, 3, 0, 0, 0],
              [0, 0, 0, 4, 6, 0],
              [0, 0, 4, 0, 0, 7],
              [4, 0, 0, 11, 0, 0],
              [0, 0, 8, 0, 3, 0],
              [0, 5, 0, 0, 0, 5]])

# the number of bits which are used for indicate connectivity to the servers, x_ijr, count from 0
NB = -1
NBc = -1
# the distribution network of bits x_ijc
Bc = [[-1 for j in range(6)] for i in range(6)]
for i in range(6):
    for j in range(6):
        if A[i][j] > 0:
            NB += 1
            NBc += 1
            Bc[i][j] = NB

# the distribution network of bits x_ijr
Br = [[[-1 for j in range(6)] for i in range(6)] for r in range(5)]
for r in range(5):
    for i in range(6):
        for j in range(6):
            if A[i][j] > 0:
                if isCoveredBy(r, i, j):
                    NB += 1
                    Br[r][i][j] = NB


# return the coordinates of the n-th bit in the distribution network
def coordinateOfBit(z):
    for i in range(6):
        for j in range(6):
            if Bc[i][j] == z:
                return [i, j]
    for i in range(6):
        for j in range(6):
            for r in range(5):
                if Br[r][i][j] == z:
                    return [r, i, j]


# from now on NB count from 1
NB += 1
NBc += 1

# Create qubo variable
x = kw.qubo.ndarray(NB + 5, "x", kw.qubo.binary)


# return 1 if (i, j) has demand(a_ij > 0), else 0
def M(i, j):
    return 1 if A[i][j] > 0 else 0


# bits that indicate the connectivity to the cloud servers, select non-zero bits to be summed here
def Xc(i, j):
    return x[0] * 0 if Bc[i][j] < 0 else x[Bc[i][j]]


# bits that indicate the connectivity to the edge servers, the principle is same as above
def Xr(r, i, j):
    return x[0] * 0 if Br[r][i][j] < 0 else x[Br[r][i][j]]


# bits that indicate whether to build the edge server r
def Sr(r):
    return x[NB + r]


# bits that indicates whether edge server r needs to connect to the cloud server, will update in iterations
Y = [0 for r in range(5)]


# cost of cloud server
Hc = kw.qubo.sum(kw.qubo.sum((Dc[i][j] * kT_C + kC_C) * A[i][j] * Xc(i, j) for j in range(6)) for i in range(6))


# cost of all edge server
He = kw.qubo.sum(
    (kw.qubo.sum(kw.qubo.sum((D[r][i][j] * kT_E + kC_E) * A[i][j] * Xr(r, i, j) * Q(r, i, j) for j in range(6))
                 for i in range(6)) + Sr(r) * F[r]) for r in range(5))

# compensation cost
C = kw.qubo.sum((Dce[r] * kT_CE + kC_C - kC_E) * (
            kw.qubo.sum(kw.qubo.sum(A[i][j] * Xr(r, i, j) * Q(r, i, j) for j in range(6)) for i in range(6)) - 12) * Y[r] for r in range(5))

# Constraint 1: Each location can only use one of all servers, including the cloud
G1 = L1 * kw.qubo.sum(
    kw.qubo.sum(((Xc(i, j) + kw.qubo.sum(Xr(r, i, j) for r in range(5)) - 1) ** 2) * M(i, j) for j in range(6)) for i in range(6))

# Constraint 2: Edge servers can only be used after they are constructed
G2 = L2 * kw.qubo.sum(
    kw.qubo.sum(kw.qubo.sum(Xr(r, i, j) * Q(r, i, j) for j in range(6)) for i in range(6)) * (1 - Sr(r))
    for r in range(5))

# Constraint 3: Edge server r must not exceed the computation capacity limit if it completes calculation independently
# G3 = L3 * kw.qubo.sum(
#     (kw.qubo.sum(kw.qubo.sum(A[i][j] * Xr(r, i, j) * Q(r, i, j) for j in range(6)) for i in range(6)) - Cap) * (1 - Yr(r))
#     for r in range(5))


# Total Cost
HT = Hc + He + C
# Objective function (H)
obj = HT + G1 + G2


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
opt = kw.sampler.optimal_sampler(matrix, output, bias=0, negtail_ff=False)
# Select the best solution
cim_best = opt[0][0]
# If the linear term variable is -1, perform a flip
cim_best = cim_best * cim_best[-1]





# Print the spin value
# print("spin: {}".format(cim_best))

# Get the list of variable names
var_s = obj_ising.get_variables()
# Substitute the spin vector and obtain the result dictionary
sol_dict = kw.qubo.get_sol_dict(cim_best, var_s)
g1_val = kw.qubo.get_val(G1, sol_dict)
g2_val = kw.qubo.get_val(G2, sol_dict)
total_cost_val = kw.qubo.get_val(HT, sol_dict)

print('[{}|'.format(g1_val), end='')
print('{}] '.format(g2_val), end='')


# Get the numerical value matrix of x
x_val = kw.qubo.get_array_val(x, sol_dict)
# Find the indices of non-zero items
nonzero_index = np.array(np.nonzero(x_val)).T
orders = nonzero_index[:].flatten()

UA = [[0 for j in range(6)] for i in range(6)]
UC = [[0 for j in range(6)] for i in range(6)]
UE = [[[0 for j in range(6)] for i in range(6)] for r in range(5)]
Srs = [0 for r in range(5)]
Yrs = [0 for r in range(5)]

for o in range(len(orders)):
    n = orders[o]
    if n < NB:
        co = coordinateOfBit(n)
        if n < NBc:
            UC[co[0]][co[1]] = 1
        else:
            UE[co[0]][co[1]][co[2]] = 1
    else:
        if n < NB + 5:
            Srs[n-NB] = 1
        else:
            Yrs[n-NB-5] = 1

for i in range(6):
    for j in range(6):
        UA[i][j] += UC[i][j]
        for r in range(5):
            UA[i][j] += UE[r][i][j]


# print('Users who connect directly to the cloud server:')
# for i in range(6):
#     print(UC[i])
# for r in range(5):
#     print('Users who connect to the edge server r{}:'.format(r))
#     for i in range(6):
#         print(UE[r][i])
print()
print('Sr: {}'.format(Srs))
print('Yr: {}'.format(Yrs))


# if g1_val != 0 or g2_val != 0 or g3_val <= 0:
#     print('\nInvalid path.')
# else:
#     print('\nValid path.')
#     print('Total Cost: ', end='')
#     print(int(HT))
