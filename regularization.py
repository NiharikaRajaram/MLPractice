# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:04:12 2019

@author: Rajaram
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

data = pd.read_csv("ex2data1.txt")
X = data.iloc[:, :-1]
y = data.iloc[:, 2]
m = len(X)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def mapFeature(X1, X2):
    degree = 6
    out = np.ones(X.shape[0])[:,np.newaxis]
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))[:,np.newaxis]))
    return out
X = mapFeature(X.iloc[:,0], X.iloc[:,1])

def costFunction(theta_t, X_t, y_t, lambda_t):
    m = len(y_t)
    J = (-1/m) * (y_t.T @ np.log(sigmoid(X_t @ theta_t)) + (1 - y_t.T) @ np.log(1 - sigmoid(X_t @ theta_t)))
    reg = (lambda_t/(2*m)) * (theta_t[1:].T @ theta_t[1:])
    J = J + reg
    return J

def lrGradientDescent(theta, X, y, lambda_t):
    m = len(y)
    grad = np.zeros([m,1])
    grad = (1/m) * X.T @ (sigmoid(X @ theta) - y)
    grad[1:] = grad[1:] + (lambda_t / m) * theta[1:]
    return grad

(m, n) = X.shape
y = y[:, np.newaxis]
theta = np.zeros((n,1))
lmbda = 1
J = costFunction(theta, X, y, lmbda)
print(J)


output = opt.fmin_tnc(func = costFunction, x0 = theta.flatten(), fprime = lrGradientDescent, \
                         args = (X, y.flatten(), lmbda))
theta = output[0]
print(theta) 


pred = [sigmoid(np.dot(X, theta)) >= 0.5]
print(np.mean(pred == y.flatten()) * 100)

