from exceptions import ValueError 
import numpy as np
import math

def bisection(f, a, b, TOL, N0):
    i = 1
    FA = f(a)
    while i <= N0:
        p = a + (b - a)/2.
        FP = f(p)
        if FP == 0. or (b - a)/2. < TOL:
            return p
        i += 1
        if np.sign(FA) * np.sign(FP) > 0.:
            a = p
            FA = FP
        else:
            b = p
    raise ValueError('Method failed after N0 iterations, N0 = {N}'.format(N=N0))

def fixedpoint_iteration(g, p0, TOL, N0):
    i = 1
    while i <= N0:
        p = g(p0)
        if math.abs(p - p0) < TOL:
            return p
        i += 1
        p0 = p
    raise ValueError('Method failed after N0 iterations, N0 = {N}'.format(N=N0))
    
def newtons(f, p0, TOL, N0):
    i = 1
    while i <= N0:
        p = p0 - f(p0)*TOL/(f(p0+TOL)-f(p0))
        if math.abs(p - p0) < TOL:
            return p
        i += 1
        p0 = p
    raise ValueError('Method failed after N0 iterations, N0 = {N}'.format(N=N0))
    
def secant(f, p0, p1, TOL, N0):
    i = 2
    q0 = f(p0)
    q1 = f(p1)
    while i <= N0:
        p = p1 -q1*(p1 - p0)/(q1 - q0)
        if math.abs(p - p1) < TOL:
            return p
        i += 1
        p0 = p1
        q0 = q1
        p1 = p
        q1 = f(p)
    raise ValueError('Method failed after N0 iterations, N0 = {N}'.format(N=N0))
    
def false_position(f, p0, p1, TOL, N0):
    i = 2
    q0 = f(p0)
    q1 = f(p1)
    while i <= N0:
        p = p1 - math.exp(math.log(q1) + math.log(p1 - p0) - math.log(q1 - q0))
        if abs(p - p1) < TOL:
            return p
        i += 1
        q = f(p)
        if np.sign(q) * np.sign(q1) < 0:
            p0 = p1
            q0 = q1
            p1 = p
            q1 = q
    raise ValueError('Method failed after N0 iterations, N0 = {N}'.format(N=N0))
    
def steffensens(g, p0, TOL, N0):
    i = 1
    while i <= N0:
        p1 = g(p0)
        p2 = g(p1)
        p = p0 - math.exp(math.log(p1 - p0) + math.log(p1 - p0) - math.log(p2 - 2. * p1 + p0))
        if abs(p - p0) < TOL:
            return p
        i += 1
        p0 = p
    raise ValueError('Method failed after N0 iterations, N0 = {N}'.format(N=N0))
    
def horners(P = np.poly1d([0]), x0):
    y = P.c[-1]
    z = P.c[-1]
    for j in range(P.order, 0, -1):
        y = x0 * y + P.c[j]
        z = x0 * z + y
    y = x0 * y + P.c[0]
    return y, z

def mullers(f, p0, p1, p2, TOL, N0):
    h1 = p1 - p0
    h2 = p2 - p1
    delta1 = (f(p1) - f(p0)) / h1
    delta2 = (f(p2) - f(p1)) / h2
    d = (delta2 - delta1) / (h2 + h1)
    i = 3
    
    while i <= N0:
        b = delta2 + h2 * d
        D = (b * b - 4. * f(p2) * d) ** 0.5
        E = 0
        if abs(b - D) < abs(b + D): 
            E = b + D
        else:
            E = b - D
            
        h = -2. * f(p2) / E
        p = p2 + h
        
        if abs(h) < TOL:
            return p
        
        p0 = p1
        p1 = p2
        p2 = p
        h1 = p1 - p0
        h2 = p2 - p1
        delta1 = (f(p1) - f(p0)) / h1
        delta2 = (f(p2) - f(p1)) / h2
        d = (delta2 - delta1) / (h2 + h1)
        i += 1
    raise ValueError('Method failed after N0 iterations, N0 = {N}'.format(N=N0))
    

    