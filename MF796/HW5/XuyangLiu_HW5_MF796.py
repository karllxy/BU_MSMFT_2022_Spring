# 
# Homework5 - MF796
# Name: Xuyang Liu
# Email address: xyangliu@bu.edu
#

import yfinance as yf
import numpy as np
import pandas as pd
import datetime

def get_sigma():
    SPY = yf.Ticker("SPY")
    adj = SPY.history(period = "1y",interval="1d")['Close']
    returns = np.log(adj.pct_change().dropna()+1)
    sigma = np.std(returns) * 252**0.5
    return sigma

S0 = 440
K1 = 445
K2 = 450
sigma = get_sigma()
smax = 550
M = 275
N = 1000
r = 0.05/100 # we got it from FRED
T =np.busday_count(datetime.date(2022,3,23),datetime.date(2022,9,16)) / 365
hs = smax/M
ht = T/N

S = np.arange(hs, smax+hs, hs)
si = S[:-1]
ai = 1 - (sigma**2) * (si**2) * (ht/hs**2) - r * ht
li = (sigma**2) * (si**2)/2 * (ht/hs**2) - r * si * ht/2/hs
ui = (sigma**2) * (si**2)/2 * (ht/hs**2) + r * si * ht/2/hs



A = np.diag(ai)
l = li[1:]
u = ui[:-1]

for i in range(M-2):
    A[i][i+1] = u[i]
    A[i+1][i] = l[i]


egv,egl = np.linalg.eig(A)


def cprice():
    C = np.zeros([M-1, N])
    pl = np.maximum(si-K1,0)
    ps = np.maximum(si-K2,0)
    C[:,-1] = pl-ps
    for j in range(N, 1, -1):
        tj = ht*j
        bj = ui[-1]*(K2-K1)*np.exp(-r*(T-tj))  # use to formular to minus
        C[:,j-2] = A.dot(C[:,j-1])
        C[-1,j-2] = C[-1,j-2] + bj
    return np.interp(S0, si, C[:,0])

p1 = cprice()


def cprice2():
    C = np.zeros([M-1, N])
    pl = np.maximum(si-K1,0)
    ps = np.maximum(si-K2,0)
    C[:,-1] = pl-ps
    for j in range(N, 1, -1):
        
        tj = ht*j
        bj = ui[-1]*(K2-K1)*np.exp(-r*(T-tj))  # use to formular to minus
        C[:,j-2] = A.dot(C[:,j-1])
        C[-1,j-2] = C[-1,j-2] + bj       
        C[:,j-2] = np.max([C[:,j-2], pl-ps], axis=0)
    return np.interp(S0, si, C[:,0])

p2 = cprice2()

pre = p2 - p1












        
        
        
        
        


    
    