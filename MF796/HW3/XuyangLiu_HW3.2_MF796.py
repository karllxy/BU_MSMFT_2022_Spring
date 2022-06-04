# 
# HM3_2.py - MF796
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
from scipy.optimize import minimize
import warnings

#(a)
def clean_data(name):
    data = pd.read_excel(name)
    data['call_mid'] = (data.call_bid + data.call_ask) / 2
    data['put_mid'] = (data.put_bid + data.put_ask) / 2

    data_call = data[['expDays', 'expT', 'K',
                      'call_mid', 'call_ask', 'call_bid']]
    data_put = data[['expDays', 'expT', 'K', 'put_mid', 'put_ask', 'put_bid']]
    data
    return data_call, data_put, data


   
# def check_abr(dfc,dfp,df): 
# Whether Call (put) prices are monotonically decreasing (increasing) in strike.
def check_mono(df, opt_type):
    mid_col = df.columns[df.columns.str.contains(
        'mid')][0]  # find the column containing 'mid'
    if opt_type == 'c':
        return any(df[mid_col].pct_change().dropna() >= 0)
    else:
        return any(df[mid_col].pct_change().dropna() <= 0)


def check_delta(df, opt_type):
    mid_col = df.columns[df.columns.str.contains('mid')][0]
    df['delta'] = (df[mid_col] - df[mid_col].shift(1)) / (df.K - df.K.shift(1))
    if opt_type == 'c':
        return any(df.delta >= 0) or any(df.delta < -1)
    else:
        return any(df.delta > 1) or any(df.delta <= 0)


def check_convex(df):
    mid_col = df.columns[df.columns.str.contains('mid')][0]
    df['convex'] = df[mid_col] - 2 * \
        df[mid_col].shift(1) + df[mid_col].shift(2)
    return any(df.convex < 0)

def check_arb(df, opt_type):
    r1 = check_mono(df, opt_type)
    r2 = check_delta(df, opt_type)
    r3 = check_convex(df)
    return pd.Series([r1, r2, r3], index=['Monotonic', 'Delta', 'Convexity'])   
    
    





#(b) find the values of κ, θ, σ, ρ and ν0 that minimize the equal weight least squared pricing error
class FFT:
    
    def __init__(self,param,t,s0=267.15,r=0.015):
        self.s0 = s0
        self.ka = param[0]
        self.v0 = param[4]
        self.sigma = param[2]
        self.rho = param[3]
        self.theta = param[1]
        self.r = r
        self.t = t
        # self.n = n
        # self.B = B
        # self.k = k
    
    def characterfunc(self,u,t):
        s0 = self.s0
        ka = self.ka
        v0 = self.v0
        sigma = self.sigma
        rho = self.rho
        theta = self.theta
        r = self.r
        
        j = complex(0, 1)
        
        lamda = np.sqrt(sigma**2 * (u**2 + u*j)+(ka-rho*sigma*u*j)**2)
        w1 = np.exp(u*j * np.log(s0) + u*j * (r-0.0177) * t + ka * theta *(ka-sigma * rho * u*j)*t/ sigma**2 )
        w2 = (np.cosh(lamda*t/2) + np.sinh(lamda * t/2) * (ka-sigma * rho *u*j)/lamda)**(2*ka*theta/sigma**2)
        w = w1/w2
        
        fi = w * np.exp(v0 * -(u**2 +u*j)/(lamda / np.tanh(lamda*t/2)+ka-sigma*rho*u*j))
        
        return fi
    
    def compute_fft(self,alpha,n,B,k,t):
        
        r = self.r
        j = complex(0, 1)
        N = 2**n
        dv = B/N
        dk = 2 * np.pi / N / dv
        beta = np.log(self.s0) - dk * N/2
        km = beta + np.arange(N) * dk
        vj = np.arange(0, N, dtype=complex) * dv
        
        delta = np.zeros(N)
        delta[0] = 1
        xlst = np.zeros(N, dtype=complex)
        for i in range(N):
            u = vj[i] - (alpha+1)*j
            f1 =  np.exp(beta * vj[i] * -j) * self.characterfunc(u,t)
            f2 = 2 * (alpha + vj[i] * j) * (alpha + vj[i] * j +1) 
            f = f1/f2
            xlst[i] = f
        
        x = dv * xlst  * (2-delta)   
        y = np.fft.fft(x)

        cs = np.exp(-alpha * np.array(km)) / np.pi
        y2 = cs * np.array(y).real
        
        k_List = list(km)
        Kt = np.exp(np.array(k_List))
        
        nk = []
        y3 = []
        for i in range(len(Kt)):
            if( Kt[i]>1e-16 )&(Kt[i] < 1e16)& ( Kt[i] != float("inf"))&( Kt[i] != float("-inf")) &( y2[i] != float("inf"))&(y2[i] != float("-inf")) & (y2[i] is not  float("nan")):
                nk += [Kt[i]]
                y3 += [y2[i]]
        tck = interpolate.splrep(nk , np.real(y3))
        price =  np.exp(-r*t)*interpolate.splev(k, tck).real

        
        return price



def price(params, alpha, n, B, k,t):
    
    price = FFT(params,t).compute_fft(alpha,n,B,k,t)
    return price

def spreadsum(params,dfc,dfp,alpha,n,B):
    r1 = 0
    for t in dfc['expT'].unique():
        d = dfc[dfc.expT==t]
        k = d['K']
        prices = price(params, alpha, n, B, k, t)
        r1 += sum((prices - d['call_mid'])**2)
    r2 = 0
    for t in dfp['expT'].unique():
        d = dfp[dfp.expT==t]
        k = d['K']
        prices = price(params, -alpha, n, B, k, t)
        r2 += sum((prices - d['put_mid'])**2)
     
        
    return r1+r2



def callback1(params):
    global times
    if times % 5 == 0:
        print('{}: {}'.format(times, spreadsum(params,dfc, dfp, alpha, n, B)))
    times += 1



def spreadsum2(params,dfc,dfp,alpha,n,B):
    r1 = 0
    for t in dfc['expT'].unique():
        d = dfc[dfc.expT==t]
        w = 1/(d['call_ask']-d['call_bid'])
        k = d['K']
        prices = price(params, alpha, n, B, k, t)
        r1 += sum(w * (prices - d['call_mid'])**2)
    r2 = 0
    for t in dfp['expT'].unique():
        d = dfp[dfp.expT==t]
        w = 1/(d['put_ask']-d['put_bid'])
        k = d['K']
        prices = price(params, -alpha, n, B, k, t)
        r2 += sum(w * (prices - d['put_mid'])**2)
     
        
    return r1+r2


def callback2(params):
    global times
    if times % 5 == 0:
        print('{}: {}'.format(times, spreadsum2(params,dfc, dfp, alpha, n, B)))
    times += 1



#problem3
def bsmodel(K, s0, sigma, r, q, T):
    d1 = (np.log(s0/K)+(r-q+sigma**2/2)*T)/(sigma*T**0.5)
    d2 = d1- sigma*T**0.5
    return(norm.cdf(d1)*s0-norm.cdf(d2)*K*np.exp(-r*T))




def bsvol(price,k,s0,r,t):
        '''Using binary search to find the implied vol'''  
        vol = 0.5
        upper=1
        lower=0
        d1 = (np.log(s0/k)+(r + vol**2 / 2)*t)/(vol * t**0.5)
        d2 = d1 - vol * t**0.5
        c = s0 * norm.cdf(d1) - k * np.exp(-r*t)*norm.cdf(d2)
        while abs(c-price) >= 0.0001:
            if c - price <0:
                lower = vol
            if c - price > 0:
                upper = vol
            if c == price:
                break
            vol = (lower + upper)/2
            d1 = (np.log(s0/k)+(r + vol**2 / 2)*t)/(vol * t**0.5)
            d2 = d1 - vol * t**0.5
            c = s0 * norm.cdf(d1) - k * np.exp(-r*t)*norm.cdf(d2)
        
        return vol



def greekbs(k,s0,vol,r,q,t):
    d1 = (np.log(s0/k)+(r-q+vol**2/2)*t)/(vol*(t)**0.5)
    delta = np.exp(-q*t)*norm.cdf(d1)
    vega = np.exp(-q*t)*s0*(t**0.5)*norm.cdf(d1)
    
    return delta,vega,d1




if __name__ == '__main__':
    name = 'mf796-hw3-opt-data.xlsx'
    dfc,dfp,df = clean_data(name)
    checkc= dfc.groupby('expDays').apply(check_arb, opt_type='c')
    checkp =dfp.groupby('expDays').apply(check_arb, opt_type='p')


#(b) (c)  
    alpha=1.5
    n=14
    B=250
    args = (dfc,dfp,alpha,n,B)
    warnings.filterwarnings('ignore')


## first try
    times = 1
    x0 = [0, 0.2, 0.2, 0, 0.2]
    lower = [0.01, 0.01, 0.0, -1, 0]
    upper = [2.5, 1, 1, 0.5, 0.5]
    bounds = tuple(zip(lower, upper))
    opt1 = minimize(spreadsum, x0, args=args, method='SLSQP', bounds=bounds, callback=callback1)
    print('The result is :',opt1.x,' and the square sum is about: ' , opt1.fun)
# ## second try (changing start value without changing bounds)   
#     times = 1
#     x0 = [0.2, 0.3, 0.3, 0.1, 0.3]
#     lower = [0.01, 0.01, 0.0, -1, 0]
#     upper = [2.5, 1, 1, 0.5, 0.5]
#     bounds = tuple(zip(lower, upper))
#     opt2 = minimize(spreadsum, x0, args=args, method='SLSQP', bounds=bounds, callback=callback1)
#     print('The result is :',opt2.x,' and the square sum is about: ' , opt2.fun)
# ## third try (changing bounds without start value)   
#     times = 1
#     x0 = [0.2, 0.3, 0.3, 0.1, 0.3]
#     lower = [0.001, 0, 0.0, -1, 0]
#     upper = [5, 2, 2, 1, 1]
#     bounds = tuple(zip(lower, upper))
#     opt3 = minimize(spreadsum, x0, args=args, method='SLSQP', bounds=bounds, callback=callback1) 
#     print('The result is :',opt3.x,' and the square sum is about: ' , opt3.fun)
    
    

# #(d)
# ## first try
#     times = 1
#     x0 = [0, 0.2, 0.2, 0, 0.2]
#     lower = [0.01, 0.01, 0.0, -1, 0]
#     upper = [2.5, 1, 1, 0.5, 0.5]
#     bounds = tuple(zip(lower, upper))
#     optw1 = minimize(spreadsum2, x0, args=args, method='SLSQP', bounds=bounds, callback=callback2)
#     print('The result is :',optw1.x,' and the square sum is about: ' , optw1.fun)
# ## second try (changing start value without changing bounds)   
#     times = 1
#     x0 = [0.2, 0.3, 0.3, 0.1, 0.3]
#     lower = [0.01, 0.01, 0.0, -1, 0]
#     upper = [2.5, 1, 1, 0.5, 0.5]
#     bounds = tuple(zip(lower, upper))
#     optw2 = minimize(spreadsum2, x0, args=args, method='SLSQP', bounds=bounds, callback=callback2)
#     print('The result is :',optw2.x,' and the square sum is about: ' , optw2.fun)
# ## third try (changing bounds without start value)   
#     times = 1
#     x0 = [0.2, 0.3, 0.3, 0.1, 0.3]
#     lower = [0.001, 0, 0.0, -1, 0]
#     upper = [5, 2, 2, 1, 1]
#     bounds = tuple(zip(lower, upper))
#     optw3 = minimize(spreadsum2, x0, args=args, method='SLSQP', bounds=bounds, callback=callback2) 
#     print('The result is :',optw3.x,' and the square sum is about: ' , optw3.fun)
    



#Problem3 (a)
    ka = 3.333
    theta = 0.054
    sigma = 1.161
    rho = -0.776
    v0 = 0.034
    params = [ka,theta,sigma,rho,v0]
    s0 = 267.15
    q = 0.0177
    r = 0.015
    t = 3/12
    k = 275
    alpha = 1
    n = 15
    B = 1000
    priceH = FFT(params,t).compute_fft(alpha,n,B,k,t)
    vol = bsvol(priceH,k,s0,r,t)
    
    bdelta,bvega,d1 = greekbs(k,s0,vol,r,q,t)
    
    dv = 0.01 * v0
    sup = s0 + 0.01
    sdo = s0 - 0.01
    hdelta = (FFT(params,t,s0=sup).compute_fft(alpha,n,B,k,t) - FFT(params,t,s0=sdo).compute_fft(alpha,n,B,k,t))/2/0.01
    
    pup = [ka,theta+dv,sigma,rho,v0+dv]
    pdo = [ka,theta-dv,sigma,rho,v0-dv]
    hvega = (FFT(pup,t).compute_fft(alpha,n,B,k,t) - FFT(pdo,t).compute_fft(alpha,n,B,k,t))/2/dv

    
    
    



    