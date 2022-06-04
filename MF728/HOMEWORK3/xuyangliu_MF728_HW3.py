# 
# HW3.py - MF728
# Name: Xuyang Liu
# Email address: xyangliu@bu.edu
#
import numpy as np
from scipy.stats import norm
import math
from scipy.optimize import root
import matplotlib.pyplot as plt

#1. Caplet Pricing in Different Models:
#(a)
def a_discountf():
    
    return np.exp(-1.25 * 0.0125)

#(b)
def bsmodel(f0,k,sigma,factor,t):
    d1 = (np.log(f0/k) + 0.5 * sigma**2 * t)/sigma/t**0.5
    d2 = (np.log(f0/k) - 0.5 * sigma**2 * t)/sigma/t**0.5
    p = 0.25 * factor * (k * norm.cdf(-d2) - f0 * norm.cdf(-d1) )
    return p

#(c)
# the sigma of normal model should be (sigma of log-normal model * f0)
def bcmodel(f0,k,sigma,factor,t):
    d = (k-f0)/sigma/t**0.5
    p = factor * 0.25 * sigma * t**0.5 * (d * norm.cdf(d) + norm.pdf(d))
    
    return p 


#(e)
def bs_greeks_put(f0,k,sigma,factor,t):    
# these formulars are calculated by sympy which I save in greeks.py   
    delta = 0.25*factor*(math.erf(math.sqrt(2)*t**(-0.5)*(0.5*sigma**2*t + math.log(f0/k))/(2*sigma))/2 - 1/2 + math.sqrt(2)*t**(-0.5)*math.exp(-t**(-1.0)*(0.5*sigma**2*t + math.log(f0/k))**2/(2*sigma**2))/(2*math.sqrt(math.pi)*sigma) - math.sqrt(2)*k*t**(-0.5)*math.exp(-t**(-1.0)*(-0.5*sigma**2*t + math.log(f0/k))**2/(2*sigma**2))/(2*math.sqrt(math.pi)*f0*sigma))
    gamma = 0.25*factor*(math.sqrt(2)*t**(-0.5)*math.exp(-t**(-1.0)*(0.5*sigma**2*t + math.log(f0/k))**2/(2*sigma**2))/(2*math.sqrt(math.pi)*f0*sigma) - math.sqrt(2)*t**(-1.5)*(0.5*sigma**2*t + math.log(f0/k))*math.exp(-t**(-1.0)*(0.5*sigma**2*t + math.log(f0/k))**2/(2*sigma**2))/(2*math.sqrt(math.pi)*f0*sigma**3) + math.sqrt(2)*k*t**(-0.5)*math.exp(-t**(-1.0)*(-0.5*sigma**2*t + math.log(f0/k))**2/(2*sigma**2))/(2*math.sqrt(math.pi)*f0**2*sigma) + math.sqrt(2)*k*t**(-1.5)*(-0.5*sigma**2*t + math.log(f0/k))*math.exp(-t**(-1.0)*(-0.5*sigma**2*t + math.log(f0/k))**2/(2*sigma**2))/(2*math.sqrt(math.pi)*f0**2*sigma**3))
    vega = 0.25*factor*(f0*(0.5*math.sqrt(2)*t**0.5 - math.sqrt(2)*t**(-0.5)*(0.5*sigma**2*t + math.log(f0/k))/(2*sigma**2))*math.exp(-t**(-1.0)*(0.5*sigma**2*t + math.log(f0/k))**2/(2*sigma**2))/math.sqrt(math.pi) - k*(-0.5*math.sqrt(2)*t**0.5 - math.sqrt(2)*t**(-0.5)*(-0.5*sigma**2*t + math.log(f0/k))/(2*sigma**2))*math.exp(-t**(-1.0)*(-0.5*sigma**2*t + math.log(f0/k))**2/(2*sigma**2))/math.sqrt(math.pi))
    theta = 0.25*factor*(f0*(0.25*math.sqrt(2)*sigma*t**(-0.5) - 0.25*math.sqrt(2)*t**(-1.5)*(0.5*sigma**2*t + math.log(f0/k))/sigma)*math.exp(-t**(-1.0)*(0.5*sigma**2*t + math.log(f0/k))**2/(2*sigma**2))/math.sqrt(math.pi) - k*(-0.25*math.sqrt(2)*sigma*t**(-0.5) - 0.25*math.sqrt(2)*t**(-1.5)*(-0.5*sigma**2*t + math.log(f0/k))/sigma)*math.exp(-t**(-1.0)*(-0.5*sigma**2*t + math.log(f0/k))**2/(2*sigma**2))/math.sqrt(math.pi))
    
    print('Delta: ',delta,' Gamma: ',gamma,' Vega: ',vega,' Theta: ',-theta)
    print()
    return delta,gamma, vega, -theta

    
    

# 2. Stripping Caplet Volatilities:
#(a)
def discount(begin,end,f):
    return np.exp(-(end-begin)*f)

def bsfac(f0,k,sigma,t):
    d1 = (np.log(f0/k) + 0.5 * sigma**2 * t)/sigma/t**0.5
    d2 = (np.log(f0/k) - 0.5 * sigma**2 * t)/sigma/t**0.5
    prep =  0.25 * (k * norm.cdf(-d2) - f0 * norm.cdf(-d1) )
    
    return prep



#(b)
def caplet_vol(nsigma,price,head,osigma):
    price1 = 0
    price2 = 0
    for i in range(4):
        price1 = price1 + discount(0,head-1+(i-1)*0.25,f) * bsfac(f0,k,osigma,head-1+i*0.25)
        price2 = price2 + discount(0,head+(i-1)*0.25,f) * bsfac(f0,k,nsigma,head+i*0.25)
        
    return price1 + price2 - price

def root_vol(fir_sigma,price):
    volst = [fir_sigma]
    sigma = fir_sigma
    for i in range(3,7):
        func = root(caplet_vol,sigma,args=(price[i-2],i,sigma))
        sigma = func.x[0]
        volst += [sigma]
    
    
    return volst
    
    







if __name__ == '__main__':


#1    
    f = 0.0125
    factor = a_discountf()
    f0 = (1 / 0.25) * ( np.exp(0.25 * f) - 1)
    sigma = 0.15
    k = 0.005
    t = 1
    pbs = bsmodel(f0=0.0125,k=0.0125,sigma=0.15,factor=factor,t=1)
    pb = bcmodel(f0=0.0125,k=0.0125,factor=factor,sigma=0.15*0.0125,t=1)
    delta1, gamma1, vega1,theta1 = bs_greeks_put(f0=0.0125,k=0.005,sigma=0.15,factor=factor,t=1)
    delta2, gamma2, vega2,theta2 = bs_greeks_put(f0=0.0125,k=0.0075,sigma=0.15,factor=factor,t=1)
    delta3, gamma3, vega3,theta3 = bs_greeks_put(f0=0.0125,k=0.01,sigma=0.15,factor=factor,t=1)
    delta4, gamma4, vega4,theta4 = bs_greeks_put(f0=0.0125,k=0.0125,sigma=0.15,factor=factor,t=1)


#2    
    gap = 0.25
    sigma = 0.15
    f = 0.01
    k = 0.01
    f0 = (1 / 0.25) * ( np.exp(0.25 * f) - 1)
    t = 1
    T = 1.25
# first cap
    pricf11 = bsfac(f0,k,sigma,t) * discount(0,T,f)
    pricf12 = pricf11 + bsfac(f0,k,sigma,t+gap) * discount(0,T+gap,f)
    pricf13 = pricf12 + bsfac(f0,k,sigma,t+gap*2) * discount(0,T+gap*2,f)
    pricf14 = pricf13 + bsfac(f0,k,sigma,t+gap*3) * discount(0,T+gap*3,f)
    pricf15 = pricf14 + bsfac(f0,k,sigma,t+gap*4) * discount(0,T+gap*4,f)
    pricf16 = pricf15 + bsfac(f0,k,sigma,t+gap*5) * discount(0,T+gap*5,f)
    pricf17 = pricf16 + bsfac(f0,k,sigma,t+gap*6) * discount(0,T+gap*6,f)
    pricf18 = pricf17 + bsfac(f0,k,sigma,t+gap*7) * discount(0,T+gap*7,f)
    
#second cap
    sigma = 0.2
    t = 2
    T = 2.25
    pricf21 = bsfac(f0,k,sigma,t) * discount(0,T+2,f)
    pricf22 = pricf21 + bsfac(f0,k,sigma,t+gap) * discount(0,T+gap,f)
    pricf23 = pricf22 + bsfac(f0,k,sigma,t+gap*2) * discount(0,T+gap*2,f)
    pricf24 = pricf23 + bsfac(f0,k,sigma,t+gap*3) * discount(0,T+gap*3,f)
    pricf25 = pricf24 + bsfac(f0,k,sigma,t+gap*4) * discount(0,T+gap*4,f)
    pricf26 = pricf25 + bsfac(f0,k,sigma,t+gap*5) * discount(0,T+gap*5,f)
    pricf27 = pricf26 + bsfac(f0,k,sigma,t+gap*6) * discount(0,T+gap*6,f)
    pricf28 = pricf27 + bsfac(f0,k,sigma,t+gap*7) * discount(0,T+gap*7,f)
    
#third cap    
    sigma = 0.225
    t = 3
    T = 3.25
    pricf31 = bsfac(f0,k,sigma,t) * discount(0,T+3,f)
    pricf32 = pricf31 + bsfac(f0,k,sigma,t+gap) * discount(0,T+gap,f)
    pricf33 = pricf32 + bsfac(f0,k,sigma,t+gap*2) * discount(0,T+gap*2,f)
    pricf34 = pricf33 + bsfac(f0,k,sigma,t+gap*3) * discount(0,T+gap*3,f)
    pricf35 = pricf34 + bsfac(f0,k,sigma,t+gap*4) * discount(0,T+gap*4,f)
    pricf36 = pricf35 + bsfac(f0,k,sigma,t+gap*5) * discount(0,T+gap*5,f)
    pricf37 = pricf36+ bsfac(f0,k,sigma,t+gap*6) * discount(0,T+gap*6,f)
    pricf38 = pricf37 + bsfac(f0,k,sigma,t+gap*7) * discount(0,T+gap*7,f)
    
#forth cap
    sigma = 0.225
    t = 4
    T = 4.25
    pricf41 = bsfac(f0,k,sigma,t) * discount(0,T+3,f)
    pricf42 = pricf41 + bsfac(f0,k,sigma,t+gap) * discount(0,T+gap,f)
    pricf43 = pricf42 + bsfac(f0,k,sigma,t+gap*2) * discount(0,T+gap*2,f)
    pricf44 = pricf43 + bsfac(f0,k,sigma,t+gap*3) * discount(0,T+gap*3,f)
    pricf45 = pricf44 + bsfac(f0,k,sigma,t+gap*4) * discount(0,T+gap*4,f)
    pricf46 = pricf45 + bsfac(f0,k,sigma,t+gap*5) * discount(0,T+gap*5,f)
    pricf47 = pricf46 + bsfac(f0,k,sigma,t+gap*6) * discount(0,T+gap*6,f)
    pricf48 = pricf47 + bsfac(f0,k,sigma,t+gap*7) * discount(0,T+gap*7,f)
    
# fifth cap

    sigma = 0.250
    t = 5
    T = 5.25
    pricf51 = bsfac(f0,k,sigma,t) * discount(0,T+3,f)
    pricf52 = pricf51 + bsfac(f0,k,sigma,t+gap) * discount(0,T+gap,f)
    pricf53 = pricf52 + bsfac(f0,k,sigma,t+gap*2) * discount(0,T+gap*2,f)
    pricf54 = pricf53 + bsfac(f0,k,sigma,t+gap*3) * discount(0,T+gap*3,f)
    pricf55 = pricf54 + bsfac(f0,k,sigma,t+gap*4) * discount(0,T+gap*4,f)
    pricf56 = pricf55 + bsfac(f0,k,sigma,t+gap*5) * discount(0,T+gap*5,f)
    pricf57 = pricf56 + bsfac(f0,k,sigma,t+gap*6) * discount(0,T+gap*6,f)
    pricf58 = pricf57 + bsfac(f0,k,sigma,t+gap*7) * discount(0,T+gap*7,f)
    
    
    price = [pricf18,pricf28,pricf38,pricf48,pricf58]
    volst = root_vol(0.15,price)
    
#visualize
    t = np.linspace(1, 7, 24, endpoint = False)
    caplet = np.repeat(np.array(volst), [8, 4, 4, 4, 4], axis = 0)
    caps = [0.15,0.2,0.225,0.225,0.25]
    
    plt.title("Shape of caplet vs. cap volatilities")
    plt.xlabel("Times")
    plt.ylabel("Volatility")
    plt.plot(t, caplet, label = "caplet")
    plt.hlines(caps[0], xmin = 1, xmax = 2.75, color = 'r',label = "1Y")
    plt.hlines(caps[1], xmin = 2, xmax = 3.75, color = 'm',label = "2Y")
    plt.hlines(caps[2], xmin = 3, xmax = 4.75, color= 'y',label = "3Y")
    plt.hlines(caps[3], xmin = 4, xmax = 5.75, color = 'g',label = "4Y")
    plt.hlines(caps[4], xmin = 5, xmax = 6.75, color = 'c',label = "5Y")
    plt.legend()

    
    





    
    