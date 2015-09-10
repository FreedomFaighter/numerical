# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 01:28:15 2016

@author: johnfreeman
"""

import numpy as np
from exceptions import ValueError
import math

def derivative(f, x, i, h = 1e-05):
    tempX = x
    tempX[i] = x[i] + h
    return (f(tempX) - f(x)) / h
    
def JacobiMatrix(F, x, h = 1e-05):
    J = np.matrix([[],[]])
    np.resize((len(F), len(F)))
    for i in range(0, len(F)):
        for j in range(0, len(F)):
            J[i][j] = derivative(F[i], x, j, h)
    return J


def newton_nonlinear(F, x, N, TOL = 1e-08):
    sizeF = F.shape[0]
    J = Jacobi(F)
    k = 0
    X = x
    while k < N:
        tempF = np.array([])
        tempF.reshape((sizeF))
        for i in range(0, sizeF):
            tempF[i] = F[i](x)
            tempJ = np.matrix([[],[]])
            tempJ.reshape((sizeF,sizeF))
            for k in range(0, sizeF):
                for j in range(0, sizeF):
                    tempJ[k][j] = J[k][j](x)
            y = np.linalg.solve(tempJ, -tempF)
            X = X + y
            if max(y) < TOL:
                return X
            k = k + 1
    raise ValueError('maximum number of iterations exceeded-the procedure was unsuccessful')


def Broyden(F, x, N, TOL = 1e-05):
    J = Jacobi(F)
    j = np.matrix([[],[]])
    j.reshape((len(F),len(F)))
    for i in range(0, len(F)):
        for j in range(0, len(F)):
            j[i][j] = J[i][j](x)
    A0 = j
    v = np.array([])
    v.reshape((len(F)))
    for i in range(0, len(F)):
        v[i] = F[i](x)
    A = np.linalg.inv(A0)
    s = -A * v
    X = x
    X = X + s
    k = 2
    while k <= N:
        w = v
        for i in range(0, len(F)):
            v[i] = F[i](X)
        y = v - w
        z = -A * y
        p = -s.T * z
        u = s.T * A
        A = A + (s + z) * u / p
        s = -A * v
        X = X + s
        if max([math.abs(i) for i in s]) < TOL:
            return x
        k = k + 1
    raise ValueError('Maximum number of iterations exceeded')
    
def Continuation(F, N, x):
    h = 1 / N
    f = []
    for i in range(0, len(F)):
        f[i] = F[i](x)
    b = -h * f
    J = Jacobi(F)
    for i in range(0, N):
        k1 = np.linalg.solve(J(x),b)
        k2 = np.linalg.solve(J(x + 0.5 *k1),b)
        k3 = np.linalg.solve(J(x + 0.5 *k2),b)
        k4 = np.linalg.solve(J(x + k3),b)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.
    return x
    
def derivative(f, i, h = 1e-05):
    def deri(f,x):
        xminus2h = x
        xminus2h[i] = x[i] - 2 * h
        xminush = x
        xminush[i] = x[i] - h
        xaddh = x
        xaddh[i] = x[i] + h
        xadd2h = x
        xadd2h[i] = x[i] + 2 * h
        return (f(xminus2h) + 8 * (f(xaddh) - f(xminush)) - f(xadd2h)) / 12 / h
    return deri
    
def Jacobi(F, h = 1e-05):
    def J(x):
        j = [[]]
        for i in range(0, len(F)):
            for k in range(0, len(F)):
                j[k][i] = derivative(F[k], i, h)
        return j
    return J
