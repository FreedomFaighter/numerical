import numpy as np
import * from exceptions

def nevills(xf = [(,)]):
    Q = np.array([[]])
    Q.resize((len(xf),len(xf)))
    for i in range(0:len(xf)):
        Q[i,0] = lambda x: xf[i][1]
    for i in range(1:len(xf)):
        for j in range(1:i+1):
            Q[i,j] = lambda x: math.exp(math.log((x - xf[i-j][0]) * Q[i,j-1](x) - (x - xf[i][0]) * Q[i - 1, j - 1](x)) - 
                                        math.log(xf[i][0] - xf[i - j][0]))
    return Q[len(xf) - 1, len(xf) - 1]

def newtons_divideddifference(xf = [(,)]):
    F = np.array([[]])
    F.resize((len(xf),len(xf)))
    for i in range(0:len(xf)):
        F[i,0] = lambda x: xf[i][1]
    
    for i in range(1:len(xf)):
        for j in range(1:i+1):
            F[i,j] = lambda x: math.exp(math.log(F[i, j - 1](x) - F[i - 1, j - 1](x)) - math.log(xf[i][0] - x[i - j][0]))
    for i in range(0:len(xf)):
        F0 = F[i,i]
        
    return F0

def natural_cubic_spline(xa = [(,)]):
    S = np.array([])
    S.resize((len(xa) - 1))
    hi = np.array([])
    for i in range(0, len(xa) - 1, 1):
        hi[i] = xa[i + 1][0] - xa[i][0]
    alpha = np.array([0])
    for i in range(1, len(xa) - 1, 1):
        alpha[i] = 3. * ((1 / hi[i]) * (xa[i+1][1] - xa[i][1]) - (1 / hi[i - 1]) * (xa[i][1] - xa[i - 1][1]))
    l0 = 1
    mu = np.array([0])
    z = np.array([0])
    for i in range(1, len(xa) - 1, 1): 
        li = 2 * (xa[i+1][0] - xa[i - 1][0]) - hi[i - 1] * mu[i - 1]
        mu[i] = hi[i] / li
        z[i] = (alpha[i] - hi[i - 1] * z[i - 1]) / li
        
    l = 1
    z[len(xa) - 1] = 0
    c = np.array([0])
    c.resize((len(xa)))
    c[len(xa) - 1] = 0
    b = c
    d = c
    for j in range(len(xa) - 2, -1 , -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (xa[j + 1][1] - xa[j][1]) / hi[j] - hi[j] * (c[j + 1] + 2. * c[j]) / 3.
        d[j] = (c[j + 1] - c[j]) / (3. * hi[j])
    
    for j in range(0, len(xa) - 1, 1):
        S = lambda x: a[j] + b[j] * (x - xa[j][0]) + c[j] * (x - xa[j][0]) ** 2 + d[j] * (x - xa[j][0]) ** 3
    
    return S

def bezier_curve(xy = np.array([(,)]), xy_plus = np.array([(,)]), xy_minus = np.array([(,)])):
    a0 = np.array([0])
    b0 = np.array([0])
    a1 = np.array([0])
    b1 = np.array([0])
    a2 = np.array([0])
    b2 = np.array([0])
    a3 = np.array([0])
    b3 = np.array([0])
    C = np.array([lambda x: 0])
    for i in range(0, len(xy) - 1, 1):
        (a0[i], b0[i]) = xy[i]
        (a1[i], b1[i]) = 3. * (xy_plus[i] - xy[i])
        (a2[i], b2[i]) = 3. * (xy[i] + xy_minus[i] - 2. * xy_plus[i])
        (a3[i], b3[i]) = xy[i + 1] - xy[i] + 3. (xy_plus[i] - xy_minus[i])
        C[i] = lambda t: return (a0[i] + a1[i] * t + a2[i] * t ** 2 + a3[i] * t ** 3, 
                                 b0[i] + b1[i] * t + b2[i] * t ** 2 + b3[i] * t ** 3)
    
    return C

