# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:14:51 2019

@author: Rajaram
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("ex2data2.txt", header=None)

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
pos , neg = (y==1).reshape(118,1) , (y==0).reshape(118,1)

def mapFeature(x1,x2,degree):
    out = np.ones(len(x1)).reshape(len(x1),1)
    for i in range(1,degree+1):
        for j in range(i+1):
            terms= (x1**(i-j) * x2**j).reshape(len(x1),1)
            out= np.hstack((out,terms))
    return out

X = mapFeature(X[:,0], X[:,1],6)

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
def costFunctionReg(theta, X, y ,Lambda):
    m=len(y)
    y=y[:,np.newaxis]
    predictions = sigmoid(X @ theta)
    error = (-y * np.log(predictions)) - ((1-y)*np.log(1-predictions))
    cost = 1/m * sum(error)
    regCost= cost + Lambda/(2*m) * sum(theta**2)
    
    j_0= 1/m * (X.transpose() @ (predictions - y))[0]
    j_1 = 1/m * (X.transpose() @ (predictions - y))[1:] + (Lambda/m)* theta[1:]
    grad= np.vstack((j_0[:,np.newaxis],j_1))
    return cost[0], grad

initial_theta = np.zeros((X.shape[1], 1))
Lambda = 1
cost, grad=costFunctionReg(initial_theta, X, y, Lambda)
print("Cost at initial theta (zeros): ",cost,"\n")

def gradientDescent(X,y,theta,alpha,num_iters,Lambda):
    m=len(y)
    J_history =[]
    
    for i in range(num_iters):
        cost, grad = costFunctionReg(theta,X,y,Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)
    return theta , J_history

theta , J_history = gradientDescent(X,y,initial_theta,1,800,0.8)
#print("Regularized theta:\n",theta,"\n")

def classifierPredict(theta,X):
    predictions = X.dot(theta)
    return predictions>0

p=classifierPredict(theta,X)
print("Train Accuracy:", (sum(p==y[:,np.newaxis])/len(y) *100)[0],"%")

def mapFeaturePlot(x1,x2,degree):
    out = np.ones(1)
    for i in range(1,degree+1):
        for j in range(i+1):
            terms= (x1**(i-j) * x2**j)
            out= np.hstack((out,terms))
    return out

plt.figure(figsize=(10,6))
plt.scatter(X[pos[:,0],1],X[pos[:,0],2],c="orange",marker="*",label="Admitted")
plt.scatter(X[neg[:,0],1],X[neg[:,0],2],c="red",marker="x",label="Not Admitted")

u_vals = np.linspace(-1,1.5,50)
v_vals= np.linspace(-1,1.5,50)
z=np.zeros((len(u_vals),len(v_vals)))
for i in range(len(u_vals)):
    for j in range(len(v_vals)):
        z[i,j] =mapFeaturePlot(u_vals[i],v_vals[j],6) @ theta 
plt.contour(u_vals,v_vals,z.T,0)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()