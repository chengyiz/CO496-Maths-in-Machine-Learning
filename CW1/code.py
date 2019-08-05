import numpy as np
a = np.array([1,0])
b = np.array([0,-1])
B = np.array([[3,-1],[-1,3]])

def grad_f1(x):
    A = B + np.identity(2)
    c = np.array([1/6, 1/6])
    return np.dot(A, np.subtract(x, c)) * 2

def grad_f2(x):
    return 2*np.cos(np.dot(x-a, x-a))*(x-a) + 2 * np.dot(B, (x-b))
