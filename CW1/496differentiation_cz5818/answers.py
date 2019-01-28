# -*- coding: utf-8 -*-

"""
Use this file for your answers. 

This file should been in the root of the repository 
(do not move it or change the file name) 

"""

# NB this is tested on python 2.7. Watch out for integer division

import numpy as np
    
def grad_f1(x):
    """
    4 marks

    :param x: input array with shape (2, )
    :return: the gradient of f1, with shape (2, )
    """
    a = np.array([1,0])
    b = np.array([0,-1])
    B = np.array([[3,-1],[-1,3]])    
    A = B + np.identity(2)
    c = np.array([1.0/6, 1.0/6])
    return np.dot(A, np.subtract(x, c)) * 2

def grad_f2(x):
    """
    6 marks

    :param x: input array with shape (2, )
    :return: the gradient of f2, with shape (2, )
    """
    a = np.array([1,0])
    b = np.array([0,-1])
    B = np.array([[3,-1],[-1,3]])
    return 2*np.cos(np.dot(x-a, x-a))*(x-a) + 2 * np.dot(B, (x-b))

def grad_f3(x):
    """
    This question is optional. The test will still run (so you can see if you are correct by
    looking at the testResults.txt file), but the marks are for grad_f1 and grad_f2 only.

    Do not delete this function.

    :param x: input array with shape (2, )
    :return: the gradient of f3, with shape (2, )
    """
    B = np.array([[3,-1],[-1,3]])
    a = np.array([1,0])
    b = np.array([0,-1])    
    tmp1 = 2 * (x-a) * np.exp(-np.dot(x-a, x-a))
    tmp2 = 2 * np.dot(B, x-b) * np.exp(-np.dot(x-b, np.dot(B, x-b)))
    tmp3 = 0.02 * 0.1 * x / (0.01 * x[0]**2 + 0.01 * x[1] **2 + 0.0001)
    return tmp1 + tmp2 + tmp3


