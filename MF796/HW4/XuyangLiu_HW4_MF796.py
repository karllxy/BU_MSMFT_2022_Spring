# 
# Homework4-problem1,2 - MF796
# Name: Xuyang Liu
# Email address: xyangliu@bu.edu
#

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import yfinance as yf
import bs4 as bs
import pickle
import requests
from scipy import optimize
# I get this function from Internet to collect the symbols of SPY 500 constitutents
# I make some changes to the function so that it will fit the problems.
def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    for i in range(len(tickers)):
        tickers[i] = tickers[i].replace('\n','')
        
    return tickers




# Now start Problem1
# Problem 1: Covariance Matrix Decomposition
#(a)
def collect_data(data):
    dt = yf.download(data,start="2018-03-11",end="2022-03-11")
    dt = dt.fillna(method='ffill')
    dt = dt.fillna(method='bfill')

    
    return dt


#(c)
def covmat_decom(returns):
    covmat = returns.cov()
    egv, egd = np.linalg.eig(covmat)
    
    return covmat,egv,egd
  
  
#(d)
def egv_portion(egv):
    base = np.sum(egv)
    cumbase = np.cumsum(egv)
    portions = cumbase / base
    port_50 = len(portions[portions<=0.5])
    port_90 = len(portions[portions<=0.9])
    return port_50, port_90


#(e)
def residuals(port90,egv,egd,returns):
    vectors = egd[:,:port90-1]
    return returns - np.dot(np.dot(returns.values,vectors),vectors.T)



#Problem2: Portfolio Construction
#(a)
def find_inv(covmat,returns):
    G = np.zeros([2,len(returns.columns)])
    G[0,:] = 1
    G[1,:17] = 1
    C_inv = np.linalg.inv(covmat)
    gcg = np.dot(np.dot(G,C_inv),G.T)
    
    return np.linalg.inv(gcg),G,C_inv


#(b)
def weight(invgcg,covmat,returns,G,C_inv):
    a = 1
    c = np.array([1,0.1])
    R = returns.mean()
    lamda = np.dot(np.dot(np.dot(G,C_inv),R)-2*a*c,invgcg)

    w = 1/2/a * np.dot(C_inv,R-np.dot(G.T , lamda))
    
    return lamda, w



if __name__ == '__main__':

## Problem1
# (a):
    data = save_sp500_tickers()
    adj = get = collect_data(data[100:200])['Adj Close']
    
#(b)
    returns = np.log(adj.pct_change().dropna()+1)
    
#(c)
    covmat,egv,egd = covmat_decom(returns)
    
#(d)
    port50,port90 = egv_portion(egv)

#(e)
    residual = residuals(port90,egv,egd,returns)
    returns.plot(legend=None)
    residual.plot(legend=None)
    

## Problem2
# (a):
    invgcg,G,C_inv = find_inv(covmat,returns)

#(b):
    lamda , w = weight(invgcg,covmat,returns,G,C_inv) 
    plt.figure()
    plt.plot(range(len(w)), w)
    plt.xlabel('Assets')
    plt.ylabel('Weights')

    plt.show()




    a = 1
    R = returns.mean()
    covmat = returns.cov()
    G = np.zeros(100)
    G[:17] = 1
    cons = ({'type': 'eq', 'fun': lambda w: G.dot(w) - 0.1},
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    weights = (np.zeros(100)+1)/100
    weights = np.random.random(100)
    # weights = pd.DataFrame(weights)
    bounds = tuple((0, 1) for w in weights)
    def portsd(weights):
        return -(np.dot(weights.T,R)-a * np.dot(np.dot(weights.T,covmat),weights))
        
    minw = optimize.minimize(fun = portsd, x0 = weights,method = 'SLSQP',constraints = cons,tol=1e-1000000000)
    w2 = minw["x"]





    
    