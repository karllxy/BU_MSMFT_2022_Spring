# 
# Homework4 - MF728
# Name: Xuyang Liu
# Email address: xyangliu@bu.edu
#
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize , root


#(b)
def ann_swap(F0,Y,instr,expiry):
    ann = np.zeros(5)
    for i in range(10):
        ann +=  np.exp(-instr/2 * (expiry + i +1)) / 2
    return ann
        
#(c)
def bachelir(ann,T,sigma,F0,K):
    d1 = (F0-K)/(sigma * T**0.5)
    r= ann * sigma * T**0.5 * (norm.cdf(d1) * d1 + norm.pdf(d1))

    return r

def premuim(sigma,F0,ann,change):
    K = np.array(6 * F0.tolist()).reshape(6,5).T + change
    pre = np.zeros([5,6])
    for i in range(5):
        for j in range(6):
            pre[i][j] = bachelir(ann[i],5,sigma[i,j],F0[i],K[i,j])
            
    return pre,K
    
    
#(d)
def asymp_form(T,K,F0,sigma0,alpha,beta,rho): 
    Fmid = (F0+K)/2
    y1 = beta / Fmid
    y2 = beta * (beta-1)/Fmid/Fmid
    
    ep = alpha/(sigma0 * (1-beta)) * (F0**(1-beta) - K**(1-beta))
    e = T * alpha **2
    delta = np.log(((1-2*rho*ep+ep**2)**0.5 + ep - rho)/(1-rho))
    
    
    a = alpha * (F0-K) / (delta)
    
    b = 1 + (2/24*(y2-y1**2) * (sigma0 * Fmid**beta / alpha)**2 + rho * y1/4 * sigma0 * Fmid**beta/alpha + (2-3*rho**2)/24)*e
    
    return a * b


def obj(params,T,K,F0,sigma,beta):
    obj = 0
    sigma0,alpha,rho = params
    for i in range(len(K)):
        obj += (asymp_form(T,K[i],F0,sigma0,alpha,beta,rho) - sigma[i]) **2
        
    return obj
    
    
#(f) 
def vol_price(F0,params,ann):
    change = np.array([-75,75]) / 10000
    K = np.array(2 * F0.tolist()).reshape(2,5).T + change
    sigma0 = params[:,0]
    alpha = params[:,1]
    rho = params[:,2]
    
    vol_n75 = asymp_form(5,K[:,0],F0,sigma0,alpha,0.5,rho)
    vol_p75 = asymp_form(5,K[:,1],F0,sigma0,alpha,0.5,rho)
    vol = np.array([vol_n75.tolist(),vol_p75.tolist()])
    vol = vol
    p_n75 = bachelir(ann,5,vol[0,:],F0,K[:,0])
    p_p75 = bachelir(ann,5,vol[1,:],F0,K[:,1])
    price = np.array([p_n75.tolist(),p_p75.tolist()])
    price = price

    
    return  vol,price
    
    
 #(g)
def BSmodel(ann, sigma, T, F0, K):
    d1 = (np.log(F0 / K) + 0.5 * sigma ** 2 * T) / (sigma * T ** 0.5)
    d2 = (np.log(F0 / K) - 0.5 * sigma ** 2 * T) / (sigma * T ** 0.5)
    return ann * (F0 * norm.cdf(d1) - K * norm.cdf(d2)), ann * norm.cdf(d1)
     
def bs_vol(ann, T, F0,pre):
    K = np.array(6 * F0.tolist()).reshape(6,5).T + change
    # # bsvol = np.array([5,6])
    # # for i in range(len(bsvol)):
    # #     bsvol[:,i] = root(lambda x : (BSmodel(ann,x,T,F0,K[:,i])[0]- pre[:,i]),0.1).x
    
    # a = root(lambda x : (BSmodel(ann,x,5,F0,K[:,0])[0]- pre[:,0]),0.1).x
    # return a
    
    K = np.array(6 * F0.tolist()).reshape(6,5).T + change
    bsvol = np.zeros([5,6])
    for i in range(len(bsvol)):
        for j in range(len(bsvol[0])):
            bsvol[i,j] = root(lambda x: (BSmodel(ann[i],x,5,F0[i],K[i,j])[0] - pre[i,j]),0.1).x
            
    return bsvol 
    
  
#(h)
def delta(bsvol,ann,T,F0):
    K = np.array(6 * F0.tolist()).reshape(6,5).T + change
    bsdelta = np.zeros([5,6])
    for i in range(len(bsvol)):
        for j in range(len(bsvol[0])):
            bsdelta[i,j] = BSmodel(ann[i],bsvol[i,j],5,F0[i],K[i,j])[1]
    return bsdelta
                            

            
#(i)
def adj_delta(params,sigma,ann):
    adj  = np.zeros([5,6])
    sigma0 = params[:,0]
    alpha = params[:,1]
    rho = params[:,2]
    for i in range(5):
        fr = F0[i] + 0.0001
        fd = F0[i] - 0.0001
        
        for j in range(6):
            volr = asymp_form(5,K[i,j],fr,sigma0[i],alpha[i],0.5,rho[i])
            vold = asymp_form(5,K[i,j],fd,sigma0[i],alpha[i],0.5,rho[i])
            pr =  bachelir(ann[i],5,volr,fr,K[i,j])
            pd =  bachelir(ann[i],5,vold,fr,K[i,j])
            
            por = bachelir(ann[i],5,sigma[i,j],fr,K[i,j])
            pod = bachelir(ann[i],5,sigma[i,j],fd,K[i,j])
            
            adj[i,j] = (por-pod+pr-pd)/(fr-fd) 
    return adj
            
  
    
  
    
  
    
    
if __name__ == '__main__':
    
    F0 = np.array([117.45,120.60,133.03,152.05,171.85])/10000
    Y = ['1Y','2Y','3Y','4Y','5Y']
    expiry = np.array([1,2,3,4,5])
    
#(a)
    instr = 2 * np.log(F0 /2 + 1)
    instr_ = pd.DataFrame({'Instantaneous Forward Rates':instr},index=Y)
    
#(b)
    ann = ann_swap(F0,Y,instr,expiry)
    ann_ = pd.DataFrame({'Annuity Value':ann},index=Y)

#(c)
    sigma = np.array([[58.31,51.72,46.29,45.72,44.92],
                      [51.51,46.87,44.48,41.80,40.61],
                      [49.28,43.09,43.61,38.92,37.69],
                      [48.74,42.63,39.36,38.19,36.94],
                      [41.46,38.23,35.95,34.41,33.36],
                      [37.33,34.55,32.55,31.15,30.21]])/10000
    sigma = sigma.T
    change = np.array([-50,-25,-5,5,25,50])/10000
    pre,K = premuim(sigma,F0,ann,change)
    pre_ = pd.DataFrame(pre,index=Y,columns=['ATM-50','ATM-25','ATM-5','ATM+5','ATM+25',
                                              'ATM+50'])
    
    
#(d)
    begin = [0.1,0.1,-0.1]
    params = np.zeros([5,3])
    for i in range(len(K)):
        opt = minimize(obj,begin,args = (5,K[i],F0[i],sigma[i],0.5), method = "SLSQP",bounds=((0.001,1.5),(0,1.5),(-1,1)))
        params[i] = opt["x"]
    params_ = pd.DataFrame(params,index=Y,columns=['sigma0','alpha','rho'])
    
#(f)
    vol,price = vol_price(F0,params,ann) 
    vol_ = pd.DataFrame(vol.T,index=Y,columns=['ATM-75','ATM+75'])
    price_ = pd.DataFrame(price.T,index=Y,columns=['ATM-75','ATM+75'])

#(g)
    bsvol = bs_vol(ann, 5, F0,pre)
    bsvol_ = pd.DataFrame(bsvol,index=Y,columns=['ATM-50','ATM-25','ATM-5','ATM+5','ATM+25',
                                              'ATM+50'])
#(h)
    delta = delta(bsvol,ann,5,F0)
    delta_ = pd.DataFrame(delta,index=Y,columns=['ATM-50','ATM-25','ATM-5','ATM+5','ATM+25',
                                              'ATM+50'])
#(i)
    adjdelta = adj_delta(params,sigma,ann)
    adj_ = pd.DataFrame(adjdelta,index=Y,columns=['ATM-50','ATM-25','ATM-5','ATM+5','ATM+25',
                                              'ATM+50'])

