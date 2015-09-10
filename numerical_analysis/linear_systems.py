import numpy as np
import math
from exceptions import ValueError

def swapRows(A, i, j):
    k = A.shape[0]
    tempRow = np.array([])
    for p in range(0, k):
        tempRow[p] = A[i][p]
    for p in range(0, k):
        A[i][p] = A[j][p]
    for p in range(0, k):
        A[j][p] = tempRow[p]
    return A

def gaussian_with_backward_substitution(A):
    n = A.shape[0]
    if n + 1 != A.shape[1]:
        raise ValueError('Invalid Matrix Size')
    m = np.matrix([[],[]])
    m.reshape((n,n))
    for i in range(0, n):
        p = -1
        for j in range(i, n):
            if A[j][i] != 0:
                p = j
                break
            if p == -1:
                raise ValueError('no unique solution exists')
        if p != i:
            A = swapRows(A, i, p)
        for j in range(i + 1, n + 1):
            m[j][i] = A[j][i] / A[i][i]
            A[j] = A[j] - m[j][i] * A[i]
    if A[0][0] == 0:
        raise ValueError('no unique solution exists')
    x = np.array([])
    x[n-1] = A[n-1][n]/A[n-1][n-1]
    for i in range(n-2, -1, -1):
        s = 0
        for j in range(i + 1, n - 1):
            s = s + A[i][j] * x[j]
        x[i] = (A[i][n+1] - s) / A[i][i]
    return x
    
