# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:56:02 2019

@author: Rajaram
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data.csv')
X = data.iloc[:,0]
Y = data.iloc[:,1]
plt.figure(figsize=(10,6))
plt.scatter(X, Y)
plt.show()

m = 0
c = 0
l = 0.0001
epochs = 1000

n = float(len(X))

for i in range(epochs):
    y_pred = (m*X) + c
    d_m = (-2/n) * sum(X * (Y - y_pred))
    d_c = (-2/n) * sum(Y - y_pred)
    m = m - l * d_m  # Update m
    c = c - l * d_c  # Update c
    
print(m,c)

y_pred = m*X + c

plt.figure(figsize=(10,6))
plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red')  # regression line
plt.show()
