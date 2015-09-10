import math
import numpy as np
from exceptions import ValueError 

def simpson(f, a, b, n):
    h = (b - a) / n
    XI0 = f(a) + f(b)
    XI1 = 0
    XI2 = 0
    for i in range(1, n, 1):
        X = a + i * h
        if i % 2 == 0:
            XI2 = XI2 + f(X)
        else:
            XI1 = XI1 + f(X)
    XI = h * (XI0 + 2 * XI2 + 4 * XI1) / 3.
    return XI

def adaptive_quadrature(f, a, b, TOL, N):
    APP = 0
    i = 1
    TOLi = np.array([0, 10. * TOL])
    ai = np.array([0, a])
    h = np.array([0, (b - a) / 2.])
    FA = np.array([0, f(a)])
    FC = np.array([0, f(a + hi[i])])
    FB = np.array([0, f(b)])
    S = np.array([0, h[i] * (FA[i] + 4. * FC[i] + FB[i]) / 2. ])
    L = np.array([0, 1])
    
    while i > 0:
        FD = f(ai[i] + h[i] / 2.)
        FE = f(ai[i] + 3. * h[i] / 2.)
        S1 = h[i] * (FA[i] + 4. * FD + FC[i]) / 6.
        S2 = h[i] * (FC[i] + 4. * FE + FB[i]) / 6.
        v1 = ai[i]
        v2 = FA[i]
        v3 = FC[i]
        v4 = FB[i]
        v5 = h[i]
        v6 = TOLi[i]
        v7 = S[i]
        v8 = L[i]
        i -= 1
        if abs(S1 + S2 - v7) < v6:
            APP += S1 + S2
        else:
            if v8 >= N:
                raise ValueError('LEVEL EXCEEDED')
            else:
                i += 1
                ai[i] = v1 + v5
                FA[i] = v3
                FC[i] = FE
                FB[i] = v4
                h[i] = v5 / 2.
                TOLi[i] = v6 / 2.
                S[i] = S2
                L[i] = v8 + 1
                i += 1
                ai[i] = v1
                FA[i] = v2
                FC[i] = FD
                FB[i] = v3
                h[i] = h[i - 1]
                TOLi[i] = TOL[i - 1]
                S[i] = S1
                L[i] = L[i - 1]
    return APP

def simpsons_double(f, a, b, c, d, m, n):
    h = (b - a) / n
    J1 = 0
    J2 = 0
    J3 = 0
    for i in range(0, n + 1, 1):
        x = a + i * h
        HX = (d(x) - c(x)) / m
        K1 = f(x, c(x)) + f(x, d(x))
        K2 = 0
        K3 = 0
        for j in range(1, m, 1):
            y = c(x) + j * HX
            Q = f(x, y)
            if j % 2 == 0:
                K2 = K2 + Q
            else:
                K3 = K3 + Q
        L = (K1 + 2. * K2 + 4. * K3) * HX / 3.
        if i == 0 or i == n:
            J1 += L
        elif i % 2 == 0:
            J2 += L
        else:
            J3 += L
    J = h * (J1 + 2. * J2 + 4. * J3) / 3.
    return J
