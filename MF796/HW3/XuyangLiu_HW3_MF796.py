# 
# HM3_1.py - MF796
# Name: Xuyang Liu
# Email address: xyangliu@bu.edu
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.stats import norm
from scipy import optimize
from scipy.optimize import fsolve
from XuyangLiu_HM2 import *


#(a)extract a table of strikes corresponding to each option.
def exstrikes(df,s0=100,r=0):
    
    sigma1 = df.iloc[:,1]
    sigma2 = df.iloc[:,2]
    m1k = []
    m3k = []
    for i  in range(len(df.iloc[:,0])):
        if df.iloc[i,3] == 'call':
            c = norm.ppf(df.iloc[i,0])
            k = s0 * np.exp(0.5 * sigma1[i] ** 2 * 1/12 - sigma1[i] * np.sqrt(1/12) * c)
            
            m1k +=[k]
            
        else:
            p = norm.ppf(df.iloc[i,0])
            
            k = s0 * np.exp(0.5 * sigma1[i] ** 2 * 1/12 + sigma1[i] * np.sqrt(1/12) * p)
            
            m1k += [k]
            
    for i  in range(len(df.iloc[:,0])):
        if df.iloc[i,3] == 'call':
            c = norm.ppf(df.iloc[i,0])
            k = s0 * np.exp(0.5 * sigma2[i] ** 2 * 3/12 - sigma2[i] * np.sqrt(3/12) * c)
            
            m3k +=[k]
            
        else:
            p = norm.ppf(df.iloc[i,0])
            
            k = s0 * np.exp(0.5 * sigma2[i] ** 2 * 3/12 + sigma2[i] * np.sqrt(3/12) * p)
            
            m3k += [k]
            
    df['1M_Strike'] = m1k
    df['3M_Strike'] = m3k
        
    
    return df


#(b) Choose an interpolation scheme that defines the volatility function for all strikes, σ(K)

def interpk(df):
    sigma1 = df.iloc[:,1]
    sigma3 = df.iloc[:,2]
    k1 = df.iloc[:,4]
    k3 = df.iloc[:,5]
    
    f1 = np.polyfit(k1,sigma1,3)
    f2 = np.polyfit(k3,sigma3,3)
    n = np.arange(84,108,0.01)
    
    plt.figure(num=3,figsize=(8,5))
    plt.plot(n,np.polyval(f1,n),label='1M')
    plt.plot(n,np.polyval(f2,n),label='3M')
    plt.legend()
    plt.xlabel('Strike Price')
    plt.ylabel('Volatility')
    plt.show() 
    
    return f1, f2
    
    
#(c)Extract the risk neutral density for 1 & 3 month options.

def bsc(s0,k,t,r,sigma):
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = (np.log(s0 / k) + (r - 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    
    c = s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    
    return c

def ntden(sigma1,sigma2,s0=100,r=0):
    t1 = 1/12
    t2 = 3/12
    h = 0.01
    k = list(np.arange(65,120,0.01))
    d1 = []
    d2 = []
    
    
    for i in range(len(k)):
        
        d = bsc(s0,k[i]-h,t1,r,max(0.00001,np.polyval(sigma1,k[i]))) - 2 * bsc(s0,k[i],t1,r,max(0.00001,np.polyval(sigma1,k[i]))) + bsc(s0,k[i]+h,t1,r,max(0.00001,np.polyval(sigma1,k[i])))
        d1 += [d/h**2]
        
    for i in range(len(k)):
        d = bsc(s0,k[i]-h,t2,r,np.polyval(sigma2,k[i])) - 2 * bsc(s0,k[i],t2,r,np.polyval(sigma2,k[i])) + bsc(s0,k[i]+h,t2,r,np.polyval(sigma2,k[i]))
        d2 += [d/h**2] 

    plt.figure(num=3,figsize=(8,5))
    plt.plot(k,d1,label='1M')
    plt.plot(k,d2,label='3M')
    plt.legend()
    plt.xlabel('Strike Price')
    plt.ylabel('Density')
    plt.show()         
    
    
    
    return d1,d2
    

#(d) Extract the risk neutral density for 1 & 3 month options using a constant volatility    
def consden(sigma1,sigma2,s0=100,r=0):
    t1 = 1/12
    t2 = 3/12
    h = 0.1
    k1 = list(np.arange(65,120,0.05))
    k2 = list(np.arange(65,120,0.05))
    d1 = []
    d2 = []
    
    
    for i in range(len(k1)):
        
        d = bsc(s0,k1[i]-h,t1,r,sigma1) - 2 * bsc(s0,k1[i],t1,r,sigma1) + bsc(s0,k1[i]+h,t1,r,sigma1)
        d1 += [d/h**2]
        
    for i in range(len(k2)):
        d = bsc(s0,k2[i]-h,t2,r,sigma2) - 2 * bsc(s0,k2[i],t2,r,sigma2) + bsc(s0,k2[i]+h,t2,r,sigma2)
        d2 += [d/h**2] 

    plt.figure(num=3,figsize=(8,5))
    plt.plot(k1,d1,label='1M')
    plt.plot(k2,d2,label='3M')
    plt.legend()
    plt.xlabel('Strike Price using Constant Delta')
    plt.ylabel('Density')
    plt.show()         
    
    
    
    return d1,d2   
    
def e1(d1):
    p = 0
    
    for i in range(4500):
        p += d1[i] * 0.01
        
    
    return p

def e2(d2):
    c = 0
    
    for i in range(4000,5500):
        c += d2[i] * 0.01
        
    return c
        

def e3(d1,d2):
    c = 0
    p = 0
    k = list(np.arange(65,120,0.01))
    for i in range(3500):
        p += d1[i] * 0.01 * (100-k[i])
        
    for i in range(3500,5500):
        c += d2[i] * 0.01 * (k[i]-100)
        
    return (p+c)/2
        
        
        

















if __name__ =='__main__':
    
    m1 = [0.3225,0.2473,0.2012,0.1824,0.1574,0.1370,0.1148]
    m3 = [0.2836,0.2178,0.1818,0.1645,0.1462,0.1256,0.1094]
    delta = [0.1,0.25,0.4,0.5,0.4,0.25,0.1]
    pc = ['put','put','put','call','call','call','call']
    df = pd.DataFrame({'Delta':delta,'1M':m1,'3M':m3,'Put or Call':pc})
#(a)    
    df2 = exstrikes(df,s0=100,r=0)

#(b)
    sigma1, sigma2 = interpk(df2)    

#(c)
    d1, d2 =ntden(sigma1,sigma2,s0=100,r=0)

#(d)
    cd1,cd2 = consden(0.1824,0.1645,s0=100,r=0)
    
    p1 = e1(d1)
    c1 = e2(d2)
    e3 = e3(d1,d2)
    
    