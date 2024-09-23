# 计算Problem2的所需比特数
import numpy as np


# demand map
Demand = np.array([[0, 0, 3, 0, 0, 0],
                   [0, 0, 0, 4, 6, 0],
                   [0, 0, 4, 0, 0, 7],
                   [4, 0, 0, 11, 0, 0],
                   [0, 0, 8, 0, 3, 0],
                   [0, 5, 0, 0, 0, 5]])


def Hv(H, X, Y, U):
    if 0 <= X < 6 and 0 <= Y < 6:
        H[X][Y] += U


BN = [[1 for j in range(6)] for i in range(6)]

Cs = np.array([[5, 0],
               [1, 2],
               [3, 4],
               [5, 4],
               [1, 0]])

for c in range(5):
    C = Cs[c]
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            Hv(BN, C[0] + dx, C[1] + dy, 1)
    Hv(BN, C[0] - 3, C[1], 1)
    Hv(BN, C[0] + 3, C[1], 1)
    Hv(BN, C[0], C[1] - 3, 1)
    Hv(BN, C[0], C[1] + 3, 1)

for i in range(6):
    for j in range(6):
        if Demand[i][j] == 0:
            BN[i][j] = 0

t = 0
for i in range(6):
    print(BN[i])
    for j in range(6):
        t += BN[i][j]

print(t)
