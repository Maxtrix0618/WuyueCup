# 2023 APMCM Wuyue Cup - Problem 2
import numpy as np
import kaiwu as kw


def euclidean_distance(i1, j1, i2, j2):
    return round(((i1 - i2) ** 2 + (j1 - j2) ** 2) ** 0.5, 2)


# determines whether (i, j) is covered by the scope of edge server r
def isCoveredBy(r, i, j):
    return euclidean_distance(Ei[r], Ej[r], i, j) <= 3


# return 1 if (i, j) is covered by the scope of r, else 0
def R(r, i, j):
    return 1 if isCoveredBy(r, i, j) else 0


# penalty coefficients
L1 = 10 ** 9
L2 = 10 ** 8
L3 = 10 ** 3
L4 = 10 ** 4
L5 = 10 ** 8

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
            NBc += 1
            NB += 1
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


def printC(n, i, j):
    print(n + 1, end=' | (')
    print(i + 1, end=', ')
    print(j + 1, end=')\n')


def count(array, v):
    c = 0
    for i in range(len(array)):
        if v == array[i]:
            c += 1
    return c


# Semi-classical iteration, correcting y based on the result of delta_r at each round
def IterationY(t, y, final):
    OB = count(y, 1) * 6
    od = 0
    ods = [-1 for _ in range(5)]
    for rc in range(5):
        if y[rc] == 1:
            ods[rc] = od
            od += 1

    # Create qubo variable
    x = kw.qubo.ndarray(NB + 5 , "x", kw.qubo.binary)

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

    # the difference between capacity limit and the total demand among the range of edge server r
    def delta_r(r):
        return kw.qubo.sum(kw.qubo.sum(A[i][j] * Xr(r, i, j) * R(r, i, j) for j in range(6)) for i in range(6)) - Cap

    # Classical bits, updated in iterations, indicate whether edge server r needs to connect to the cloud
    # Y = [0 for r in range(5)]

    def Hr(r):
        if y[r] == 0:
            return x[0] * 0
        else:
            return kw.qubo.sum((2 ** p) * x[NB + 5 + (6 * ods[r]) + p] for p in range(6))

    # cost of cloud server
    Hc = kw.qubo.sum(kw.qubo.sum((Dc[i][j] * kT_C + kC_C) * A[i][j] * Xc(i, j) for j in range(6)) for i in range(6))

    # cost of all edge server
    He = kw.qubo.sum(
        (kw.qubo.sum(kw.qubo.sum((D[r][i][j] * kT_E + kC_E) * A[i][j] * Xr(r, i, j) * R(r, i, j) for j in range(6))
                     for i in range(6)) + Sr(r) * F[r]) for r in range(5))

    # compensation cost
    C = kw.qubo.sum((Dce[r] * kT_CE + kC_C - kC_E) * (
            kw.qubo.sum(kw.qubo.sum(A[i][j] * Xr(r, i, j) * R(r, i, j) for j in range(6)) for i in range(6)) - Cap) *
                    y[r] for r in range(5))

    # Constraint 1: Each location can only use one of all servers, including the cloud
    G1 = L1 * kw.qubo.sum(
        kw.qubo.sum(((Xc(i, j) + kw.qubo.sum(Xr(r, i, j) for r in range(5)) - 1) ** 2) * M(i, j) for j in range(6)) for
        i in range(6))

    # Constraint 2: Edge servers can only be used after they are constructed
    G2 = L2 * kw.qubo.sum(
        kw.qubo.sum(kw.qubo.sum(Xr(r, i, j) * R(r, i, j) for j in range(6)) for i in range(6)) * (1 - Sr(r))
        for r in range(5))

    G3 = L3 * kw.qubo.sum(((Hr(r) - delta_r(r)) ** 2) * y[r] for r in range(5))
    G4 = L4 * kw.qubo.sum(((Hr(r) + delta_r(r)) ** 2) * (1 - y[r]) for r in range(5))
    G5 = L5 * kw.qubo.sum(y[r] * (1 - Sr(r)) for r in range(5))

    # Total Cost
    HT = Hc + He + C
    # Objective function (H)
    obj = HT + G1 + G2 + G3 + G4 + G5

    # Parse QUBO
    obj = kw.qubo.make(obj)
    # Convert to Ising model



    
    obj_ising = kw.qubo.cim_ising_model(obj)
    # Extract the Ising matrix
    matrix = obj_ising.get_ising()["ising"]

    # Perform calculation using CIM simulator
    output = kw.cim.simulator(
        matrix,
        pump=1.3,
        noise=0.2,
        laps=5000,
        dt=0.05,
        normalization=0.3,
        iterations=50)

    # Sort the results
    opt = kw.sampler.optimal_sampler(matrix, output, bias=0, negtail_ff=False)
    # Select the best solution
    cim_best = opt[0][0]
    # If the linear term variable is -1, perform a flip
    cim_best = cim_best * cim_best[-1]

    # Get the list of variable names
    var_s = obj_ising.get_variables()
    # Substitute the spin vector and obtain the result dictionary
    sol_dict = kw.qubo.get_sol_dict(cim_best, var_s)
    g1_val = kw.qubo.get_val(G1, sol_dict)
    g2_val = kw.qubo.get_val(G2, sol_dict)
    # g3_val = kw.qubo.get_val(G3, sol_dict)
    # g4_val = kw.qubo.get_val(G4, sol_dict)
    g5_val = kw.qubo.get_val(G5, sol_dict)
    c_val = kw.qubo.get_val(C, sol_dict)
    cost_val = kw.qubo.get_val(HT, sol_dict)

    if g1_val != 0 or g2_val != 0 or g5_val != 0:
        print('x')
        return False

    print('{}'.format(t), end=' ')
    print('[{}'.format(g1_val), end=' | ')
    print('{}'.format(g2_val), end=' | ')
    # print('{}'.format(g3_val), end=' | ')
    # print('{}'.format(g4_val), end=' | ')
    print('{}]'.format(g5_val), end=' ')
    print('{}'.format(Y), end=' ')

    # Get the numerical value matrix of x
    x_val = kw.qubo.get_array_val(x, sol_dict)
    # Find the indices of non-zero items
    nonzero_index = np.array(np.nonzero(x_val)).T
    orders = nonzero_index[:].flatten()

    Srs = [0 for _ in range(5)]
    for o in range(len(orders)):
        n = orders[o]
        if NB <= n < NB + 5:
            Srs[n - NB] = 1
    print('{}'.format(Srs), end=' ')
    print('{}'.format(round(c_val), 2), end=' ')
    print('{}'.format(round(cost_val), 2), end=' [')

    for r in range(5):
        dr = kw.qubo.get_val(delta_r(r), sol_dict)
        print('{}'.format(dr), end=(']\n' if r == 4 else ', '))

    if not final:
        return True

    Cis = 0
    UC = [[0 for _ in range(6)] for _ in range(6)]
    UE = [[[0 for _ in range(6)] for _ in range(6)] for _ in range(5)]
    for o in range(len(orders)):
        n = orders[o]
        if n < NB:
            co = coordinateOfBit(n)
            if n < NBc:
                UC[co[0]][co[1]] = 1
                Cis += kC_C * A[co[0]][co[1]]
            else:
                UE[co[0]][co[1]][co[2]] = 1
                Cis += kC_E * A[co[0]][co[1]]
        else:
            if NB <= n < NB + 5:
                Srs[n - NB] = 1

    print('Users who connect directly to the cloud server:')
    for i in range(6):
        print(UC[i])
    for r in range(5):
        print('Users who connect to the edge server r{}:'.format(r + 1))
        for i in range(6):
            print(UE[r][i])
    print()

    Con = 0
    for r in range(5):
        Con += Srs[r] * F[r]

    Tis = cost_val - Con - Cis

    print('-Final Result-')
    print('Total Cost: {}'.format(round(cost_val, 2)))
    print('Construction Cost: {}'.format(round(Con, 2)))
    print('Transmission Cost: {}'.format(round(Tis, 2)))
    print('Calculation Cost: {}'.format(round(Cis, 2)))

    print('The coordinates of edge servers constructed:')
    for r in range(5):
        print('', end='(')
        print(Ex[r], end=', ')
        print(Ey[r], end=')\n')

    print('The coordinates of the users who connect directly to the cloud server:')
    print('Number | (X-axis, Y-axis)')
    n = 0
    for i in range(6):
        for j in range(6):
            if UC[i][j] == 1:
                printC(n, i, j)
                n += 1
    print('The coordinates of the users who connect directly to the edge server:')
    for r in range(5):
        print('Edge server {}, or '.format(r+1), end='(')
        print(Ex[r], end=', ')
        print(Ey[r], end=')\n')
        print('Number | (X-axis, Y-axis)')
        n = 0
        for i in range(6):
            for j in range(6):
                if UE[r][i][j] == 1:
                    printC(n, i, j)
                    n += 1

    return True


T = 0
print('T [ G1 | G2  |  G3  |  G4 | G5 ]        Yr            Sr        C        Cost                delta_r')
for y1 in range(2):
    for y2 in range(2):
        for y3 in range(2):
            for y4 in range(2):
                T += 1
                Y = [y1, y2, y3, y4, 0]
                while True:
                    if IterationY(T, Y, False):
                        break


T = 0
Y = [0, 0, 1, 1, 0]
while True:
    if IterationY(T, Y, True):
        break
