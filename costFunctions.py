import math
import numpy as np

#@np.vectorize
def deJong5(x):
    a1 = [-32.0, -16.0, 0.0, 16.0, 32.0]*5
    a2 = [-32.0]*5 + [-16.0]*5 + [0.0]*5 + [16.0]*5 + [32.0]*5
    a = [a1, a2]

    sum = 0.002
    for i in range(1, 26):
        sum += ( 1 / (i + (x[0] - a[0][i-1])**6 + (x[1] - a[1][i-1])**6) )
    
    return 1.0/sum

#@np.vectorize
def langermann(x):
    m = 5
    c = [1, 2, 5, 2, 3]
    A = [[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]]

    outer = 0.0
    for i in range(m):
        inner = 0.0

        inner += (x[0] - A[i][0]) ** 2
        
        inner += (x[1] - A[i][1]) ** 2
        
        newTerm = c[i] * math.exp(-inner / math.pi) * math.cos(math.pi * inner)
        
        outer += newTerm
    return outer


def costFunction(x):
    #return langermann(x[0], x[1])
    #return deJong5(x[0], x[1])#
    pass