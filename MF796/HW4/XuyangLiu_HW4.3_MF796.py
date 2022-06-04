# 
# Homework4-problem3 - MF796
# Name: Xuyang Liu
# Email address: xyangliu@bu.edu
#

import pandas as pd
import numpy as np
from scipy import optimize

## Problem 3: Portfolio Stability
#(a)

file = '/Users/liuxuyang/Desktop/BU SPRING 2022/MF796/HW4/DataForProblem3.csv'
df = pd.read_csv(file,index_col=0)

def minvar_w():
    covmat = df.cov()
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    weights = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    weights = pd.DataFrame(weights)
    weights = weights.values
    bounds = tuple((0, 1) for w in weights)
    def portsd(weights):
        return np.dot(np.dot(weights.T,covmat),weights)
        
    minw = optimize.minimize(fun = portsd, x0 = weights,bounds= bounds,method = 'SLSQP',constraints = constraints,tol=1e-100)
    w = minw["x"]
    return w

minvar_w = minvar_w()



#(b) 
# def optimal_w1():
#     a = 0.5
#     C = df.cov()
#     print(C)
#     R = df.mean()
#     print(R)
#     return 1/2/a * np.dot(np.linalg.inv(C),R)


def optimal_w(R):
    a = 0.5
    covmat = df.cov()
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    weights = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    # weights = pd.DataFrame(weights)
    bounds = tuple((0, 1) for w in weights)
    def portsd(weights):
        return -(np.dot(weights.T,R)-a * np.dot(np.dot(weights.T,covmat),weights))
        
    minw = optimize.minimize(fun = portsd, x0 = weights,bounds= bounds,method = 'SLSQP',constraints = constraints,tol=1e-100)
    w = minw["x"]
    return w    
    
optw = optimal_w(df.mean())


#(c)
def max_w(R):
    a = 0.5
    
    covmat = df.cov()
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    weights = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    # weights = pd.DataFrame(weights)
    bounds = tuple((0, 1) for w in weights)
    def portsd(weights):
        return -(np.dot(weights.T,R))
        
    minw = optimize.minimize(fun = portsd, x0 = weights,bounds= bounds,method = 'SLSQP',constraints = constraints,tol=1e-100)
    w = minw["x"]
    return w    

R = df.mean()
maxw = max_w(R)
R2 = R +.01
R3 = R +0.0001
optw2 = optimal_w(R2)
optw3 = optimal_w(R3)

R4 = R
R4[1] = R4[1] +0.0001
optw4 = optimal_w(R4)



 
    
    
    













