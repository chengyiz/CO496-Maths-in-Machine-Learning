# -*- coding: utf-8 -*-

"""
Use this file for your answers. 

This file should been in the root of the repository 
(do not move it or change the file name) 

"""

import numpy as np

def lml(alpha, beta, Phi, Y):
    """
    4 marks

    :param alpha: float
    :param beta: float
    :param Phi: array of shape (N, M)
    :param Y: array of shape (N, 1)
    :return: the log marginal likelihood, a scalar
    """
    N = Phi.shape[0]
    mat =  alpha * np.dot(Phi, Phi.T) + beta * np.eye(N)
    inv = np.linalg.inv(mat)
    det = np.linalg.det(mat)
    return -N * 0.5 * np.log(2*np.pi) - 0.5 * np.log(det) - 0.5 * np.dot(Y[:,0], np.dot(inv, Y[:,0]))
    

def grad_lml(alpha, beta, Phi, Y):
    """
    8 marks (4 for each component)

    :param alpha: float
    :param beta: float
    :param Phi: array of shape (N, M)
    :param Y: array of shape (N, 1)
    :return: array of shape (2,). The components of this array are the gradients
    (d_lml_d_alpha, d_lml_d_beta), the gradients of lml with respect to alpha and beta respectively.
    """
    N = Phi.shape[0]
    #epsilon = 10**-12
    #gradient_a = (lml(alpha+epsilon, beta, Phi, Y) - lml(alpha-epsilon,beta,Phi,Y))/ epsilon * 0.5
    #gradient_b = (lml(alpha, beta+epsilon, Phi, Y) - lml(alpha, beta-epsilon, Phi, Y))/ epsilon * 0.5
    #return np.array([gradient_a, gradient_b])
    PPt = np.dot(Phi, Phi.T)
    X = alpha * PPt + beta * np.eye(N)
    inv = np.linalg.inv(X)
    gradient_a = -0.5 * np.trace(np.dot(inv, PPt)) + 0.5 * np.dot(Y[:,0], np.dot(inv, np.dot(inv, np.dot(PPt, Y[:,0]))))
    gradient_b = -0.5 * np.trace(inv) + 0.5 * np.dot(Y[:,0], np.dot(inv, np.dot(inv, Y[:,0])))
    return np.array([gradient_a, gradient_b])
