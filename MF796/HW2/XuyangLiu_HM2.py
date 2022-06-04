# 
# HM2.py - MF796
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



class FFT:
    
    def __init__(self,s0,ka,v0,sigma,rho,theta,r,t):
        self.s0 = s0
        self.ka = ka
        self.v0 = v0
        self.sigma = sigma
        self.rho = rho
        self.theta = theta
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
        w1 = np.exp(u*j * np.log(s0) + u*j * (r-0) * t + ka * theta *(ka-sigma * rho * u*j)*t/ sigma**2 )
        w2 = (np.cosh(lamda*t/2) + np.sinh(lamda * t/2) * (ka-sigma * rho *u*j)/lamda)**(2*ka*theta/sigma**2)
        w = w1/w2
        
        fi = w * np.exp(v0 * -(u**2 +u*j)/(lamda / np.tanh(lamda*t/2)+ka-sigma*rho*u*j))
        
        return fi
    
    def compute_fft(self,alpha,n,B,k,t):
        
        begin = time.time()
      
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
        end = time.time()
        deltatime = end - begin

        
        return price, deltatime
 
        
 
    def relation_plot(self,n_list,B_list,K):
        zz = np.zeros((len(n_list),len(B_list)))
        ee = np.zeros((len(n_list),len(B_list)))
        xx, yy = np.meshgrid(n_list, B_list)
        for i in range(len(n_list)):
            for j in range(len(B_list)):
                temp = self.compute_fft(1,n_list[i],B_list[j],K,t)
                zz[i][j] = temp[0]
                ee[i][j] = 1/((temp[0]-16.7348)**2*temp[1])
                
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(xx, yy, zz.T, rstride=1, cstride=1,cmap='rainbow')
        plt.title("Euro Call Option Price with respct to N and B")
        ax.set_xlabel("N_value")
        ax.set_ylabel("B_value")
        ax.set_zlabel("Call Option Price Using Different B N")
        plt.show()
        
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(xx, yy, ee.T, rstride=1, cstride=1,cmap='rainbow')
        plt.title("Efficiency with respect to N and B")
        ax.set_xlabel("N_value")
        ax.set_ylabel("B_value")
        ax.set_zlabel("Efficiency Using different B N")
        plt.show()
        
        return ee
 
    
 
    def voplot(self,klst,alpha,n,B):
        r = self.r
        t = self.t
        s0 = self.s0
        pricelist = []
        for i in range(len(klst)):
            pricelist += [self.compute_fft(alpha,n,B,klst[i],t)[0]]
        
        volist = []
        for i in range(len(pricelist)):
            volist += [Bsvol(pricelist[i],klst[i],s0,r,t)]
            
        plt.plot(klst,volist)
        plt.xlabel('Strike Price')
        plt.ylabel('Implied volatility')
        return volist,pricelist


    
    def tplot(self,tlst,alpha,n,B,k=150):
        r = self.r
        s0 = self.s0
        pricelist = []
        for i in range(len(tlst)):
            pricelist += [self.compute_fft(alpha,n,B,k,tlst[i])[0]]
            
        volist = []
        for i in range(len(tlst)):
            volist += [Bsvol(pricelist[i],k,s0,r,tlst[i])]
            
        plt.plot(tlst,volist)
        plt.xlabel('Expiry')
        plt.ylabel('Implied volatility')
        return volist,pricelist
            
        
            
            
def Bsvol(price,k,s0,r,t):
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



        
        
        
        

    # def findarelation(self,alphalist):
    #     pricelist = []
    #     for i in range(len(alphalist)):
    #         price = self.compute_fft(alphalist[i])
    #         pricelist +=[price]
            
        
    #     plt.plot(alphalist,pricelist)
        
    #     return pricelist
        
    












if __name__ == '__main__':

#(a)    
    sigma = 0.2
    v0 = 0.08
    ka = 0.7
    rho = -0.4
    theta = 0.1
    s0 = 250
    r = 0.02
    t = 0.5
    n = 14
    B = 250 
    k = 260 #(21.2688,16.7348)
    
    
    alpha = 1
    a = FFT(s0,ka,v0,sigma,rho,theta,r,t)
    price,deltatime= a.compute_fft(alpha,n,B,k,t)
    print(a.compute_fft(alpha,n,B,k,t)[0])
    # alphalist = np.arange(0.01,20,0.01)
    # alphalist = list(alphalist)
    # pricelist = a.findarelation(alphalist)
    a.compute_fft(alpha,n,B,k,t)
    Bs = np.linspace(50,250,100)
    ns = np.array([7,8,9,10,11,12,13,14])
    efficiency = a.relation_plot(ns,Bs,k)

# #(b)
#     sigma = 0.4
#     v0 = 0.09
#     ka = 0.5
#     rho = 0.25
#     theta = 0.12
#     r = 0.025
#     s0 = 150
#     t = 0.25 
#     k = 150
#     B = 250
#     n = 14
#     alpha = 1
#     b = FFT(s0,ka,v0,sigma,rho,theta,r,t)
    # price= b.compute_fft(alpha,n,B,k,t)[0]
    # vol = Bsvol(price,k,s0,r,t)
 
    
    # '''changing ka'''
    # klst = list(np.arange(100,180))
    # tlst = list(np.arange(0.1,2.1,0.02))
    # volstv1,pricelistv1 = b.voplot(klst,alpha,n,B)
    # volstt1,pricelistt1 = b.tplot(tlst,alpha,n,B,k)
    # ka = 0.1
    # c = FFT(s0,ka,v0,sigma,rho,theta,r,t)
    # volstv2,pricelistv2 = c.voplot(klst,alpha,n,B)
    # volstt2,pricelistt2 = c.tplot(tlst,alpha,n,B,k)
    # ka = 0.9
    # d = FFT(s0,ka,v0,sigma,rho,theta,r,t)
    # volstv3,pricelistv3 = d.voplot(klst,alpha,n,B)
    # volstt3,pricelistt3 = d.tplot(tlst,alpha,n,B,k)
    
    # plt.figure(num=3,figsize=(8,5))
    # plt.plot(klst,volstv2,label='ka=0.1')
    # plt.plot(klst,volstv1,label='ka=0.5')
    # plt.plot(klst,volstv3,label='ka=0.9')
    # plt.legend()
    # plt.xlabel('Strike Price')
    # plt.ylabel('Implied volatility')
    # plt.show()
    
    
    

    # '''changing sigma'''
    # klst = list(np.arange(100,180))
    # tlst = list(np.arange(0.1,2.1,0.02))
    # volstv1,pricelistv1 = b.voplot(klst,alpha,n,B)
    # volstt1,pricelistt1 = b.tplot(tlst,alpha,n,B,k)
    # sigma = 0.5
    # c = FFT(s0,ka,v0,sigma,rho,theta,r,t)
    # volstv2,pricelistv2 = c.voplot(klst,alpha,n,B)
    # volstt2,pricelistt2 = c.tplot(tlst,alpha,n,B,k)
    # sigma = 0.6
    # d = FFT(s0,ka,v0,sigma,rho,theta,r,t)
    # volstv3,pricelistv3 = d.voplot(klst,alpha,n,B)
    # volstt3,pricelistt3 = d.tplot(tlst,alpha,n,B,k)
    
    # plt.figure(num=3,figsize=(8,5))
    # plt.plot(klst,volstv1,label='sigma=0.4')
    # plt.plot(klst,volstv2,label='sigma=0.5')
    # plt.plot(klst,volstv3,label='sigma=0.6')
    # plt.legend()
    # plt.xlabel('Strike Price')
    # plt.ylabel('Implied volatility')
    # plt.show()
    
    # plt.figure(num=3,figsize=(8,5))
    # plt.plot(tlst,volstt1,label='sigma=0.4')
    # plt.plot(tlst,volstt2,label='sigma=0.5')
    # plt.plot(tlst,volstt3,label='sigma=0.6')
    # plt.legend()
    # plt.xlabel('Expiry')
    # plt.ylabel('Implied volatility')
    # plt.show()
    
    
    # '''changing rho'''
    # klst = list(np.arange(100,180))
    # tlst = list(np.arange(0.1,2.1,0.02))
    # volstv1,pricelistv1 = b.voplot(klst,alpha,n,B)
    # volstt1,pricelistt1 = b.tplot(tlst,alpha,n,B,k)
    # rho = 0.35
    # c = FFT(s0,ka,v0,sigma,rho,theta,r,t)
    # volstv2,pricelistv2 = c.voplot(klst,alpha,n,B)
    # volstt2,pricelistt2 = c.tplot(tlst,alpha,n,B,k)
    # rho = 0.45
    # d = FFT(s0,ka,v0,sigma,rho,theta,r,t)
    # volstv3,pricelistv3 = d.voplot(klst,alpha,n,B)
    # volstt3,pricelistt3 = d.tplot(tlst,alpha,n,B,k)
    
    # plt.figure(num=3,figsize=(8,5))
    # plt.plot(klst,volstv1,label='rho=0.25')
    # plt.plot(klst,volstv2,label='rho=0.35')
    # plt.plot(klst,volstv3,label='rho=0.45')
    # plt.legend()
    # plt.xlabel('Strike Price')
    # plt.ylabel('Implied volatility')
    # plt.show()
    
    # plt.figure(num=3,figsize=(8,5))
    # plt.plot(tlst,volstt1,label='rho=0.25')
    # plt.plot(tlst,volstt2,label='rho=0.35')
    # plt.plot(tlst,volstt3,label='rho=0.45')
    # plt.legend()
    # plt.xlabel('Expiry')
    # plt.ylabel('Implied volatility')
    # plt.show()
    
    
    # '''changing theta'''
    # klst = list(np.arange(100,180))
    # tlst = list(np.arange(0.1,2.1,0.02))
    # volstv1,pricelistv1 = b.voplot(klst,alpha,n,B)
    # volstt1,pricelistt1 = b.tplot(tlst,alpha,n,B,k)
    # theta = 0.14
    # c = FFT(s0,ka,v0,sigma,rho,theta,r,t)
    # volstv2,pricelistv2 = c.voplot(klst,alpha,n,B)
    # volstt2,pricelistt2 = c.tplot(tlst,alpha,n,B,k)
    # theta = 0.16
    # d = FFT(s0,ka,v0,sigma,rho,theta,r,t)
    # volstv3,pricelistv3 = d.voplot(klst,alpha,n,B)
    # volstt3,pricelistt3 = d.tplot(tlst,alpha,n,B,k)
    
    # plt.figure(num=3,figsize=(8,5))
    # plt.plot(klst,volstv1,label='theta=0.12')
    # plt.plot(klst,volstv2,label='theta=0.14')
    # plt.plot(klst,volstv3,label='theta=0.16')
    # plt.legend()
    # plt.xlabel('Strike Price')
    # plt.ylabel('Implied volatility')
    # plt.show()
    
    # plt.figure(num=3,figsize=(8,5))
    # plt.plot(tlst,volstt1,label='theta=0.12')
    # plt.plot(tlst,volstt2,label='theta=0.14')
    # plt.plot(tlst,volstt3,label='theta=0.16')
    # plt.legend()
    # plt.xlabel('Expiry')
    # plt.ylabel('Implied volatility')
    # plt.show()



    # '''changing v0'''
    # klst = list(np.arange(100,180))
    # tlst = list(np.arange(0.1,2.1,0.02))
    # volstv1,pricelistv1 = b.voplot(klst,alpha,n,B)
    # volstt1,pricelistt1 = b.tplot(tlst,alpha,n,B,k)
    # v0 = 0.11
    # c = FFT(s0,ka,v0,sigma,rho,theta,r,t)
    # volstv2,pricelistv2 = c.voplot(klst,alpha,n,B)
    # volstt2,pricelistt2 = c.tplot(tlst,alpha,n,B,k)
    # v0 = 0.13
    # d = FFT(s0,ka,v0,sigma,rho,theta,r,t)
    # volstv3,pricelistv3 = d.voplot(klst,alpha,n,B)
    # volstt3,pricelistt3 = d.tplot(tlst,alpha,n,B,k)
    
    # plt.figure(num=3,figsize=(8,5))
    # plt.plot(klst,volstv1,label='v0=0.09')
    # plt.plot(klst,volstv2,label='v0=0.11')
    # plt.plot(klst,volstv3,label='v0=0.13')
    # plt.legend()
    # plt.xlabel('Strike Price')
    # plt.ylabel('Implied volatility')
    # plt.show()
    
    # plt.figure(num=3,figsize=(8,5))
    # plt.plot(tlst,volstt1,label='v0=0.09')
    # plt.plot(tlst,volstt2,label='v0=0.11')
    # plt.plot(tlst,volstt3,label='v0=0.13')
    # plt.legend()
    # plt.xlabel('Expiry')
    # plt.ylabel('Implied volatility')
    # plt.show()
    
    
    
    
    
    
    
    
    
    
    