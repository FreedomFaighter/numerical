import math
import ValueError from exceptions
import numpy as np

def eulers(f, a, b, N, alpha):
    h = (b - a) / N
    t = a
    w = alpha
    yield (t, w)
    for i in range(1, N + 1, 1):
        w += h * f(t, w)
        t = a + i * h
        yield (t, w)

def rungekutta(f, a, b, N, alpha):
    h = (b - a) / N
    t = a
    w = alpha
    yield (t, w)
    for i in range(1, N + 1, 1):
        K1 = h * f(t, w)
        K2 = h * f(t + h / 2., w + K1 / 2.)
        K3 = h * f(t + h / 2., w + K2 / 2.)
        K4 = h * f(t + h, w + K3)
        
        w += (K1 + 2. * K2 + 2. * K3 + K4) / 6.
        t = a + i * h
        yield (t, w)
        
def rungekuttafehlberg(f, a, b, alpha, TOL, hmax, hmin):
    t = a
    w = alpha
    h = hmax
    FLAG = 1
    yield (t, w, h)
    quarter = 1. / 4.
    eigth = 1. / 8.
    thirtysecondth = 1. / 32.
    twothousandonehundredninetyseventh = 1. / 2197.
    twohundredsixteenth = 1. / 216.
    fivehundredthirteenth = 1. / 513.
    fourthousandonehundredfourth = 1. / 4104.
    
    while FLAG == 1:
        K1 = h * f(t, w)
        K2 = h * f(t + h * quarter, w + K1 * quarter)
        K3 = h * f(t + 3. * h / 8., w + 3. * K1 * thirtysecondth + 9 * K2 * thirtysecondth)
        K4 = h * f(t + 12. * h / 13., w + 1932. * K1 * twothousandonehundredninetyseventh - 7200. * K2 * twothousandonehundredninetyseventh + 7296. * K3 * twothousandonehundredninetyseventh)
        K5 = h * f(t + h, w + 439. * K1 * twohundredsixteenth - 8. * K2 + 3680. * K3 * fivehundredthirteenth - 845. * K4 * fourthousandonehundredfourth)
        K6 = h * f(t + h / 2., w - 8 * K1 / 27. + 2. * K2 - 3544 * K3 / 2565. + 1859 * K4 * fourthousandonehundredfourth - 11 * K5 / 40.)
        R = (1. / h) * abs(K1 / 360. - 128. * K3 / 4275. - 2197. * K4 / 75240. + K5 / 50. + 2 * K6 / 55.)
        if R <= TOL:
            t += h
            w += 25 * K1 * twohundredsixteenth + 1408 * K3 / 2565. + 2197. * K4 * fourthousandonehundredfourth - K5 / 5.
            yield (t, w, h)
        delta = 0.84 * (TOL / R) ** 0.25
        if delta <= 0.1:
            h = 0.1 * h
        elif delta >= 4:
            h = 4 * h
        else:
            h = delta * h
        if h > hmax:
            h = hmax
        if t >= b:
            FLAG = 0
        elif t + h > b:
            h = b - t
        elif h < hmin:
            FLAG = 0
            raise ValueError('minimum h exceeded, hmin = {hm}'.format(hm=hmin)))
        
            
def adams_fourthorder_predictorcorrector(f, a, b, N, alpha):
    h = (b - a) / N
    t = np.array([a])
    w = np.array([alpha])
    yield (t[0], w[0])
    for i in range(1, 4, 1):
        K1 = h * f(t[i - 1], w[i - 1])
        K2 = h * f(t[i - 1] + h / 2., w[i - 1] + K1 / 2.)
        K3 = h * f(t[i - 1] + h / 2., w[i - 1] + K2 / 2.)
        K4 = h * f(t[i - 1] + h, w[i - 1] + K3)
        
        w[i] = w[i - 1] + (K1 + 2. * K2 + 2. * K3 + K4) / 6.
        t[i] = a + i * h
        yield (t[i], w[i])
    for i in range(4, N + 1, 1):
        ti = a + i * h
        wi = w[3] + h * (55. * f(t[3], w[3]) - 59. * f(t[2], w[2]) + 37. * f(t[1], w[1]) - 9. * f(t[0], w[0])) / 24.
        wi = w[3] + h * (9 * f(ti, wi) + 19. * f(t[3], w[3]) - 5 * f(t[2], w[2]) + f(t[1], w[1])) / 24.
        yield (ti, wi)
        for j in range(0, 3, 1):
            t[j] = t[j + 1]
            w[j] = w[j + 1]
        t[3] = ti
        w[3] = wi

        
def extrapolation(f, a, b, alpha, TOL, hmax, hmin):
    NK = np.array([2, 4, 6, 8, 12, 16, 24, 32])
    TO = a
    WO = alpha
    h = hmax
    FLAG = 1
    Q = np.array([[]])
    Q.resize((7, 7))
    for i in range(0, 7, 1):
        for j in range(0, i + 1, 1):
            Q[i, j] = (NK[i + 1] / NK[j]) ** 2
    while FLAG == 1:
        k = 0
        NFLAG = 0
        y = np.array([])
        y.resize((8))
        while k < 8 and NFLAGS == 0:
            HK = h / NK[k]
            T = TO
            W2 = WO
            W3 = W2 + HK * f(T, W2)
            for j in range(1, NK[k] - 1):
                W1 = W2 
                W2 = W3 
                W3 = W1 + 2. * HK * f(T, W2)
                T = TO + (j + 1) * HK
            y[k] = (W3 + W2 + HK * f(T, W3)) / 2.
            if k >= 1:
                j = k
                v = y[0]
                while j >= 1:
                    y[j - 1] = y[j] + (y[j] - y[j - 1]) / (Q[k - 1, j - 1] - 1)
                    j -= 1
                if abs(y[0] - v) <= TOL:
                    NFLAG = 1
            k += 1
        k -= 1
        if NFLAG == 0:
            h = h / 2.
            if h < hmin:
                FLAG = 0
        else:
            WO = y[0]
            TO += h
            yield (TO, WO, h)
            if TO >= b:
                FLAG = 0
            elif TO + h > b:
                h = b - TO
            elif k <= 3 and h < 0.5 * hmax:
                h = 2. * h

                
def rungekutta_systems(f, a, b, m, N, alpha):
    w = np.array([])
    w.resize((m))
    k = np.array([[]])
    k.resize((4,m))
    h = (b - a) / N
    t = a
    for j in range(0, m, 1):
        w[j] = alpha[j]
    yield (t, w)
    for i in range(0, N, 1):
        for j in range(0, m, 1):
            k[0, j] = h * f[j](t, w)
        for j in range(0, m, 1):
            k[1, j] = h * f[j](t + 0.5 * h, w + 0.5 * k[0, :])
        for j in range(0, m, 1):
            k[2, j] = h * f[j](t + 0.5 * h, w + 0.5 * k[1, :])
        for j in range(0, m, 1):
            k[3, j] = h * f[j](t + h, w + k[2,:])
        for j in range(0, m, 1):
            w += (k[0,:] + 2. * k[1,:] + 2. * k[2,:] + k[3,:]) / 6.
        t = a + i * h
        yield (t, w)
        
