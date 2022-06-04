# 
# HM2.py - MF728
# Name: Xuyang Liu
# Email address: xyangliu@bu.edu
#
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sympy
from scipy import optimize
#
#(a)
def gf1(sw):
    r1 = sympy.symbols('r1')
    swr1 = (0.5*1/0.5*(sympy.exp(0.5*r1)-1)*sympy.exp(-0.5*r1)+0.5*1/0.5 * (sympy.exp(0.5*r1)-1)*sympy.exp(-r1))
    swr2 = (0.5 * sympy.exp(-0.5*r1)+0.5 *sympy.exp(-r1))*sw
    r1 = sympy.solve(swr1-swr2,r1)[0]
    
    f1 = 1/0.5 * (math.exp(0.5*r1)-1)

    return math.log(math.exp(r1)),f1

#(b)
def gf2(sw,r1,f1):
    r2 = sympy.symbols('r2')
    swr1 = 0.5 * f1 * sympy.exp(-0.5 * r1)+0.5 * f1 * sympy.exp(-r1) + (sympy.exp(0.5*r2)-1)*sympy.exp(-r1-0.5*r2) + (sympy.exp(0.5*r2)-1)*sympy.exp(-r1-r2)
    swr2 = 0.5*sw * (sympy.exp(-0.5*r1)+sympy.exp(-r1)+sympy.exp(-r1-0.5*r2)+sympy.exp(-r1-r2))
    r2 = sympy.solve(swr1-swr2,r2)[0]
    
    f2 = 1/0.5 * (math.exp(0.5*r2)-1)
    
    return math.log(math.exp(r2)),f2

#(c)
def gf3(sw,r1,f1,r2,f2):
    r3 = sympy.symbols('r3')
    swr1 = sum([0.5*f1*sympy.exp(-i * r1 /2)for i in range(1,3)]) + sum([0.5*f2*sympy.exp(-r1-i*r2/2)for i in range(1,3)]) + sum([(sympy.exp(0.5*r3)-1) * sympy.exp(-r1-r2-i*r3/2)for i in range(1,3)])
    swr2 = sum([0.5*sympy.exp(-r1 * i/2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r2 * i/2 -r1)for i in range(1,3)]) + sum([0.5*sympy.exp(-r3 * i/2 -r1-r2)for i in range(1,3)])
    r3 = sympy.solve(swr1-swr2*sw,r3)[0]
    
    f3 = 1/0.5 * (math.exp(0.5*r3)-1)
    
    return math.log(math.exp(r3)),f3

def gf4(sw,r1,f1,r2,f2,r3,f3): 
    r4 = sympy.symbols('r4')
    swr1 = sum([0.5*f1*sympy.exp(-i * r1 /2)for i in range(1,3)]) + sum([0.5*f2*sympy.exp(-r1-i*r2/2)for i in range(1,3)]) + sum([(sympy.exp(0.5*r3)-1) * sympy.exp(-r1-r2-i*r3/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r4)-1) * sympy.exp(-r1-r2-r3-i*r4/2)for i in range(1,3)])
    swr2 = sum([0.5*sympy.exp(-r1 * i/2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r2 * i/2 -r1)for i in range(1,3)]) + sum([0.5*sympy.exp(-r3 * i/2 -r1-r2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r4 * i/2 -r1-r2-r3)for i in range(1,3)])      
        
    r4 = sympy.solve(swr1-swr2*sw,r4)[0]
    
    f4 = 1/0.5 * (math.exp(0.5*r4)-1)
    
    return math.log(math.exp(r4)),f4        

def gf5(sw,r1,f1,r2,f2,r3,f3,r4,f4): 
    r5 = sympy.symbols('r5')
    swr1 = sum([0.5*f1*sympy.exp(-i * r1 /2)for i in range(1,3)]) + sum([0.5*f2*sympy.exp(-r1-i*r2/2)for i in range(1,3)]) + sum([(sympy.exp(0.5*r3)-1) * sympy.exp(-r1-r2-i*r3/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r4)-1) * sympy.exp(-r1-r2-r3-i*r4/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r5)-1) * sympy.exp(-r1-r2-r3-r4-i*r5/2)for i in range(1,3)])
    swr2 = sum([0.5*sympy.exp(-r1 * i/2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r2 * i/2 -r1)for i in range(1,3)]) + sum([0.5*sympy.exp(-r3 * i/2 -r1-r2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r4 * i/2 -r1-r2-r3)for i in range(1,3)])+sum([0.5*sympy.exp(-r5 * i/2 -r1-r2-r3-r4)for i in range(1,3)])   
        
    r5 = sympy.solve(swr1-swr2*sw,r5)[0]
    
    f5 = 1/0.5 * (math.exp(0.5*r5)-1)
    
    return math.log(math.exp(r5)),f5        

def gf7(sw,r1,f1,r2,f2,r3,f3,r4,f4,r5,f5): 
    r7 = sympy.symbols('r7')
    swr1 = sum([0.5*f1*sympy.exp(-i * r1 /2)for i in range(1,3)]) + sum([0.5*f2*sympy.exp(-r1-i*r2/2)for i in range(1,3)]) + sum([(sympy.exp(0.5*r3)-1) * sympy.exp(-r1-r2-i*r3/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r4)-1) * sympy.exp(-r1-r2-r3-i*r4/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r5)-1) * sympy.exp(-r1-r2-r3-r4-i*r5/2)for i in range(1,3)])+ sum([(sympy.exp(0.5*r7)-1) * sympy.exp(-r1-r2-r3-r4-r5-i*r7/2)for i in range(1,5)])    
    swr2 = sum([0.5*sympy.exp(-r1 * i/2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r2 * i/2 -r1)for i in range(1,3)]) + sum([0.5*sympy.exp(-r3 * i/2 -r1-r2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r4 * i/2 -r1-r2-r3)for i in range(1,3)])+sum([0.5*sympy.exp(-r5 * i/2 -r1-r2-r3-r4)for i in range(1,3)])+sum([0.5*sympy.exp(-r7 * i/2 -r1-r2-r3-r4-r5)for i in range(1,5)])    
        
    r7 = sympy.solve(swr1-swr2*sw,r7)[0]
    
    f7 = 1/0.5 * (math.exp(0.5*r7)-1)
    
    return math.log(math.exp(r7)),f7   

def gf10(sw,r1,f1,r2,f2,r3,f3,r4,f4,r5,f5,r7,f7): 
    r10 = sympy.symbols('r10')
    swr1 = sum([0.5*f1*sympy.exp(-i * r1 /2)for i in range(1,3)]) + sum([0.5*f2*sympy.exp(-r1-i*r2/2)for i in range(1,3)]) + sum([(sympy.exp(0.5*r3)-1) * sympy.exp(-r1-r2-i*r3/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r4)-1) * sympy.exp(-r1-r2-r3-i*r4/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r5)-1) * sympy.exp(-r1-r2-r3-r4-i*r5/2)for i in range(1,3)])+ sum([(sympy.exp(0.5*r7)-1) * sympy.exp(-r1-r2-r3-r4-r5-i*r7/2)for i in range(1,5)]) + sum([(sympy.exp(0.5*r10)-1) * sympy.exp(-r1-r2-r3-r4-r5-2*r7-i*r10/2)for i in range(1,7)])     
    swr2 = sum([0.5*sympy.exp(-r1 * i/2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r2 * i/2 -r1)for i in range(1,3)]) + sum([0.5*sympy.exp(-r3 * i/2 -r1-r2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r4 * i/2 -r1-r2-r3)for i in range(1,3)])+sum([0.5*sympy.exp(-r5 * i/2 -r1-r2-r3-r4)for i in range(1,3)])+sum([0.5*sympy.exp(-r7 * i/2 -r1-r2-r3-r4-r5)for i in range(1,5)])  + sum([0.5*sympy.exp(-r10 * i/2 -r1-r2-r3-r4-r5-2*r7)for i in range(1,7)])   
        
    r10 = sympy.solve(swr1-swr2*sw,r10)[0]
    
    f10 = 1/0.5 * (math.exp(0.5*r10)-1)
    
    return math.log(math.exp(r10)),f10  

def gf30(sw,r1,f1,r2,f2,r3,f3,r4,f4,r5,f5,r7,f7,r10,f10):
# the slowest methods.
    r30 = sympy.symbols('r30')
    swr1 = sum([0.5*f1*sympy.exp(-i * r1 /2)for i in range(1,3)]) + sum([0.5*f2*sympy.exp(-r1-i*r2/2)for i in range(1,3)]) + sum([(sympy.exp(0.5*r3)-1) * sympy.exp(-r1-r2-i*r3/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r4)-1) * sympy.exp(-r1-r2-r3-i*r4/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r5)-1) * sympy.exp(-r1-r2-r3-r4-i*r5/2)for i in range(1,3)])+ sum([(sympy.exp(0.5*r7)-1) * sympy.exp(-r1-r2-r3-r4-r5-i*r7/2)for i in range(1,5)]) + sum([(sympy.exp(0.5*r10)-1) * sympy.exp(-r1-r2-r3-r4-r5-2*r7-i*r10/2)for i in range(1,7)]) + sum([(sympy.exp(0.5*r30)-1) * sympy.exp(-r1-r2-r3-r4-r5-2*r7-3*r10-i*r30/2)for i in range(1,41)])    
    swr2 = sum([0.5*sympy.exp(-r1 * i/2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r2 * i/2 -r1)for i in range(1,3)]) + sum([0.5*sympy.exp(-r3 * i/2 -r1-r2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r4 * i/2 -r1-r2-r3)for i in range(1,3)])+sum([0.5*sympy.exp(-r5 * i/2 -r1-r2-r3-r4)for i in range(1,3)])+sum([0.5*sympy.exp(-r7 * i/2 -r1-r2-r3-r4-r5)for i in range(1,5)])  + sum([0.5*sympy.exp(-r10 * i/2 -r1-r2-r3-r4-r5-2*r7)for i in range(1,7)])   + sum([0.5*sympy.exp(-r30 * i/2 -r1-r2-r3-r4-r5-2*r7-3*r10)for i in range(1,41)]) 
        
    r30 = sympy.solve(swr1-swr2*sw,r30)[0]
    
    f30 = 1/0.5 * (math.exp(0.5*r30)-1)
    
    return r30,f30 


def gf302(r30):
# a fast method:
    swr1 = sum([0.5*f1*math.exp(-i * r1 /2)for i in range(1,3)]) + sum([0.5*f2*math.exp(-r1-i*r2/2)for i in range(1,3)]) + sum([(math.exp(0.5*r3)-1) * math.exp(-r1-r2-i*r3/2)for i in range(1,3)])+sum([(math.exp(0.5*r4)-1) * math.exp(-r1-r2-r3-i*r4/2)for i in range(1,3)])+sum([(math.exp(0.5*r5)-1) * math.exp(-r1-r2-r3-r4-i*r5/2)for i in range(1,3)])+ sum([(math.exp(0.5*r7)-1) * math.exp(-r1-r2-r3-r4-r5-i*r7/2)for i in range(1,5)]) + sum([(math.exp(0.5*r10)-1) * math.exp(-r1-r2-r3-r4-r5-2*r7-i*r10/2)for i in range(1,7)]) + sum([(math.exp(0.5*r30)-1) * math.exp(-r1-r2-r3-r4-r5-2*r7-3*r10-i*r30/2)for i in range(1,41)])    
    swr2 = sum([0.5*math.exp(-r1 * i/2)for i in range(1,3)]) + sum([0.5*math.exp(-r2 * i/2 -r1)for i in range(1,3)]) + sum([0.5*math.exp(-r3 * i/2 -r1-r2)for i in range(1,3)]) + sum([0.5*math.exp(-r4 * i/2 -r1-r2-r3)for i in range(1,3)])+sum([0.5*math.exp(-r5 * i/2 -r1-r2-r3-r4)for i in range(1,3)])+sum([0.5*math.exp(-r7 * i/2 -r1-r2-r3-r4-r5)for i in range(1,5)])  + sum([0.5*math.exp(-r10 * i/2 -r1-r2-r3-r4-r5-2*r7)for i in range(1,7)])   + sum([0.5*math.exp(-r30 * i/2 -r1-r2-r3-r4-r5-2*r7-3*r10)for i in range(1,41)]) 
    return swr1-swr2*0.03237  



#(d) Compute the fair market, breakeven swap rate of a 15Y swap

def swr15y(r1,f1,r2,f2,r3,f3,r4,f4,r5,f5,r7,f7,r10,f10,r30,f30):
    
    swr1 = sum([0.5*f1*math.exp(-i * r1 /2)for i in range(1,3)]) + sum([0.5*f2*math.exp(-r1-i*r2/2)for i in range(1,3)]) + sum([(math.exp(0.5*r3)-1) * math.exp(-r1-r2-i*r3/2)for i in range(1,3)])+sum([(math.exp(0.5*r4)-1) * math.exp(-r1-r2-r3-i*r4/2)for i in range(1,3)])+sum([(math.exp(0.5*r5)-1) * math.exp(-r1-r2-r3-r4-i*r5/2)for i in range(1,3)])+ sum([(math.exp(0.5*r7)-1) * math.exp(-r1-r2-r3-r4-r5-i*r7/2)for i in range(1,5)]) + sum([(math.exp(0.5*r10)-1) * math.exp(-r1-r2-r3-r4-r5-2*r7-i*r10/2)for i in range(1,7)]) + sum([(math.exp(0.5*r30)-1) * math.exp(-r1-r2-r3-r4-r5-2*r7-3*r10-i*r30/2)for i in range(1,11)])    
    swr2 = sum([0.5*math.exp(-r1 * i/2)for i in range(1,3)]) + sum([0.5*math.exp(-r2 * i/2 -r1)for i in range(1,3)]) + sum([0.5*math.exp(-r3 * i/2 -r1-r2)for i in range(1,3)]) + sum([0.5*math.exp(-r4 * i/2 -r1-r2-r3)for i in range(1,3)])+sum([0.5*math.exp(-r5 * i/2 -r1-r2-r3-r4)for i in range(1,3)])+sum([0.5*math.exp(-r7 * i/2 -r1-r2-r3-r4-r5)for i in range(1,5)])  + sum([0.5*math.exp(-r10 * i/2 -r1-r2-r3-r4-r5-2*r7)for i in range(1,7)])   + sum([0.5*math.exp(-r30 * i/2 -r1-r2-r3-r4-r5-2*r7-3*r10)for i in range(1,11)]) 
    
    
    return swr1/swr2
    
#(e) Compute discount factors, see below.

#(f)  
def for_up_swr(df):
    fup = df['Forward Rate']+0.01
    rup = np.log(0.5 * fup+1) * 2
    f1,f2,f3,f4,f5,f7,f10,f30 = fup
    r1,r2,r3,r4,r5,r7,r10,r30 = rup
    swr1 = (0.5*1/0.5*(np.exp(0.5*rup[0])-1)*np.exp(-0.5*rup[0])+0.5*1/0.5 * (np.exp(0.5*rup[0])-1)*np.exp(-rup[0]))
    swr2 = (0.5 * np.exp(-0.5*rup[0])+0.5 *np.exp(-rup[0]))
    nr1 = swr1/swr2
    
    swr1 = 0.5 * fup[0] * sympy.exp(-0.5 * rup[0])+0.5 * fup[0] * sympy.exp(-rup[0]) + (sympy.exp(0.5*rup[1])-1)*sympy.exp(-rup[0]-0.5*rup[1]) + (sympy.exp(0.5*rup[1])-1)*sympy.exp(-rup[0]-rup[1])
    swr2 = 0.5 * (sympy.exp(-0.5*rup[0])+sympy.exp(-rup[0])+sympy.exp(-rup[0]-0.5*rup[1])+sympy.exp(-rup[0]-rup[1]))
    nr2 = swr1/swr2
    
    swr1 = sum([0.5*fup[0]*sympy.exp(-i * rup[0] /2)for i in range(1,3)]) + sum([0.5*fup[1]*sympy.exp(-rup[0]-i*rup[1]/2)for i in range(1,3)]) + sum([(sympy.exp(0.5*rup[2])-1) * sympy.exp(-rup[0]-rup[1]-i*rup[2]/2)for i in range(1,3)])
    swr2 = sum([0.5*sympy.exp(-rup[0] * i/2)for i in range(1,3)]) + sum([0.5*sympy.exp(-rup[1] * i/2 -rup[0])for i in range(1,3)]) + sum([0.5*sympy.exp(-rup[2] * i/2 -rup[0]-rup[1])for i in range(1,3)])
    nr3 = swr1/swr2
    
    swr1 = sum([0.5*f1*sympy.exp(-i * r1 /2)for i in range(1,3)]) + sum([0.5*f2*sympy.exp(-r1-i*r2/2)for i in range(1,3)]) + sum([(sympy.exp(0.5*r3)-1) * sympy.exp(-r1-r2-i*r3/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r4)-1) * sympy.exp(-r1-r2-r3-i*r4/2)for i in range(1,3)])
    swr2 = sum([0.5*sympy.exp(-r1 * i/2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r2 * i/2 -r1)for i in range(1,3)]) + sum([0.5*sympy.exp(-r3 * i/2 -r1-r2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r4 * i/2 -r1-r2-r3)for i in range(1,3)])      
    nr4 = swr1/swr2
    
    swr1 = sum([0.5*f1*sympy.exp(-i * r1 /2)for i in range(1,3)]) + sum([0.5*f2*sympy.exp(-r1-i*r2/2)for i in range(1,3)]) + sum([(sympy.exp(0.5*r3)-1) * sympy.exp(-r1-r2-i*r3/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r4)-1) * sympy.exp(-r1-r2-r3-i*r4/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r5)-1) * sympy.exp(-r1-r2-r3-r4-i*r5/2)for i in range(1,3)])
    swr2 = sum([0.5*sympy.exp(-r1 * i/2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r2 * i/2 -r1)for i in range(1,3)]) + sum([0.5*sympy.exp(-r3 * i/2 -r1-r2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r4 * i/2 -r1-r2-r3)for i in range(1,3)])+sum([0.5*sympy.exp(-r5 * i/2 -r1-r2-r3-r4)for i in range(1,3)])   
    nr5 = swr1/swr2
    
    swr1 = sum([0.5*f1*sympy.exp(-i * r1 /2)for i in range(1,3)]) + sum([0.5*f2*sympy.exp(-r1-i*r2/2)for i in range(1,3)]) + sum([(sympy.exp(0.5*r3)-1) * sympy.exp(-r1-r2-i*r3/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r4)-1) * sympy.exp(-r1-r2-r3-i*r4/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r5)-1) * sympy.exp(-r1-r2-r3-r4-i*r5/2)for i in range(1,3)])+ sum([(sympy.exp(0.5*r7)-1) * sympy.exp(-r1-r2-r3-r4-r5-i*r7/2)for i in range(1,5)])    
    swr2 = sum([0.5*sympy.exp(-r1 * i/2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r2 * i/2 -r1)for i in range(1,3)]) + sum([0.5*sympy.exp(-r3 * i/2 -r1-r2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r4 * i/2 -r1-r2-r3)for i in range(1,3)])+sum([0.5*sympy.exp(-r5 * i/2 -r1-r2-r3-r4)for i in range(1,3)])+sum([0.5*sympy.exp(-r7 * i/2 -r1-r2-r3-r4-r5)for i in range(1,5)])    
    nr7 = swr1/swr2    
    
    swr1 = sum([0.5*f1*sympy.exp(-i * r1 /2)for i in range(1,3)]) + sum([0.5*f2*sympy.exp(-r1-i*r2/2)for i in range(1,3)]) + sum([(sympy.exp(0.5*r3)-1) * sympy.exp(-r1-r2-i*r3/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r4)-1) * sympy.exp(-r1-r2-r3-i*r4/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r5)-1) * sympy.exp(-r1-r2-r3-r4-i*r5/2)for i in range(1,3)])+ sum([(sympy.exp(0.5*r7)-1) * sympy.exp(-r1-r2-r3-r4-r5-i*r7/2)for i in range(1,5)]) + sum([(sympy.exp(0.5*r10)-1) * sympy.exp(-r1-r2-r3-r4-r5-2*r7-i*r10/2)for i in range(1,7)])     
    swr2 = sum([0.5*sympy.exp(-r1 * i/2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r2 * i/2 -r1)for i in range(1,3)]) + sum([0.5*sympy.exp(-r3 * i/2 -r1-r2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r4 * i/2 -r1-r2-r3)for i in range(1,3)])+sum([0.5*sympy.exp(-r5 * i/2 -r1-r2-r3-r4)for i in range(1,3)])+sum([0.5*sympy.exp(-r7 * i/2 -r1-r2-r3-r4-r5)for i in range(1,5)])  + sum([0.5*sympy.exp(-r10 * i/2 -r1-r2-r3-r4-r5-2*r7)for i in range(1,7)])   
    nr10 = swr1/swr2   
    
    swr1 = sum([0.5*f1*sympy.exp(-i * r1 /2)for i in range(1,3)]) + sum([0.5*f2*sympy.exp(-r1-i*r2/2)for i in range(1,3)]) + sum([(sympy.exp(0.5*r3)-1) * sympy.exp(-r1-r2-i*r3/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r4)-1) * sympy.exp(-r1-r2-r3-i*r4/2)for i in range(1,3)])+sum([(sympy.exp(0.5*r5)-1) * sympy.exp(-r1-r2-r3-r4-i*r5/2)for i in range(1,3)])+ sum([(sympy.exp(0.5*r7)-1) * sympy.exp(-r1-r2-r3-r4-r5-i*r7/2)for i in range(1,5)]) + sum([(sympy.exp(0.5*r10)-1) * sympy.exp(-r1-r2-r3-r4-r5-2*r7-i*r10/2)for i in range(1,7)]) + sum([(sympy.exp(0.5*r30)-1) * sympy.exp(-r1-r2-r3-r4-r5-2*r7-3*r10-i*r30/2)for i in range(1,41)])    
    swr2 = sum([0.5*sympy.exp(-r1 * i/2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r2 * i/2 -r1)for i in range(1,3)]) + sum([0.5*sympy.exp(-r3 * i/2 -r1-r2)for i in range(1,3)]) + sum([0.5*sympy.exp(-r4 * i/2 -r1-r2-r3)for i in range(1,3)])+sum([0.5*sympy.exp(-r5 * i/2 -r1-r2-r3-r4)for i in range(1,3)])+sum([0.5*sympy.exp(-r7 * i/2 -r1-r2-r3-r4-r5)for i in range(1,5)])  + sum([0.5*sympy.exp(-r10 * i/2 -r1-r2-r3-r4-r5-2*r7)for i in range(1,7)])   + sum([0.5*sympy.exp(-r30 * i/2 -r1-r2-r3-r4-r5-2*r7-3*r10)for i in range(1,41)]) 
    nr30 = swr1/swr2  
    
    newr = [nr1,nr2,nr3,nr4,nr5,nr7,nr10,nr30]
    df['New Swap Rate']=newr
    return df
    
#(g)
def gf302_b(r30):
# a fast method:
    swr1 = sum([0.5*f1*math.exp(-i * r1 /2)for i in range(1,3)]) + sum([0.5*f2*math.exp(-r1-i*r2/2)for i in range(1,3)]) + sum([(math.exp(0.5*r3)-1) * math.exp(-r1-r2-i*r3/2)for i in range(1,3)])+sum([(math.exp(0.5*r4)-1) * math.exp(-r1-r2-r3-i*r4/2)for i in range(1,3)])+sum([(math.exp(0.5*r5)-1) * math.exp(-r1-r2-r3-r4-i*r5/2)for i in range(1,3)])+ sum([(math.exp(0.5*r7)-1) * math.exp(-r1-r2-r3-r4-r5-i*r7/2)for i in range(1,5)]) + sum([(math.exp(0.5*r10)-1) * math.exp(-r1-r2-r3-r4-r5-2*r7-i*r10/2)for i in range(1,7)]) + sum([(math.exp(0.5*r30)-1) * math.exp(-r1-r2-r3-r4-r5-2*r7-3*r10-i*r30/2)for i in range(1,41)])    
    swr2 = sum([0.5*math.exp(-r1 * i/2)for i in range(1,3)]) + sum([0.5*math.exp(-r2 * i/2 -r1)for i in range(1,3)]) + sum([0.5*math.exp(-r3 * i/2 -r1-r2)for i in range(1,3)]) + sum([0.5*math.exp(-r4 * i/2 -r1-r2-r3)for i in range(1,3)])+sum([0.5*math.exp(-r5 * i/2 -r1-r2-r3-r4)for i in range(1,3)])+sum([0.5*math.exp(-r7 * i/2 -r1-r2-r3-r4-r5)for i in range(1,5)])  + sum([0.5*math.exp(-r10 * i/2 -r1-r2-r3-r4-r5-2*r7)for i in range(1,7)])   + sum([0.5*math.exp(-r30 * i/2 -r1-r2-r3-r4-r5-2*r7-3*r10)for i in range(1,41)]) 
    return swr1-swr2*0.03737 





if __name__ == '__main__':

#(a),(b),(c)
    swr = [0.028438,0.03060,0.03126,0.03144,0.03150,0.03169,0.03210,0.03237]
    r1,f1= gf1(swr[0])
    r2,f2= gf2(swr[1],r1,f1)
    r3,f3 = gf3(swr[2],r1,f1,r2,f2)
    r4,f4 = gf4(swr[3],r1,f1,r2,f2,r3,f3)
    r5,f5 = gf5(swr[4],r1,f1,r2,f2,r3,f3,r4,f4)
    r7,f7 = gf7(swr[5],r1,f1,r2,f2,r3,f3,r4,f4,r5,f5)
    r10,f10 = gf10(swr[6],r1,f1,r2,f2,r3,f3,r4,f4,r5,f5,r7,f7)
    # r30,f30 = gf30(r1,f1,r2,f2,r3,f3,r4,f4,r5,f5,r7,f7,r10,f10)
   
    r30 = optimize.root(gf302,0).x[0]
    f30 = 1/0.5 * (math.exp(0.5*r30)-1)
    swr = [0.028438,0.03060,0.03126,0.03144,0.03150,0.03169,0.03210,0.03237]
    r = [r1,r2,r3,r4,r5,r7,r10,r30]
    f = [f1,f2,f3,f4,f5,f7,f10,f30]
    y = ['1Y','2Y','3Y','4Y','5Y','7Y','10Y','30Y']
    df = pd.DataFrame({'Swap Rate':swr,'Spot Rate':r, 'Forward Rate':f,},y)
    df.index.name='Years'
    df['Spot Rate']=df['Spot Rate'].apply(lambda x : round(x,10))
    df['Forward Rate']=df['Forward Rate'].apply(lambda x : round(x,10))
    print(df)
   
#(d)
    Swr15 = swr15y(r1,f1,r2,f2,r3,f3,r4,f4,r5,f5,r7,f7,r10,f10,r30,f30)
   
#(e)
    zero = np.exp(df['Spot Rate'])-1
    df['Zero Rate'] = zero
    print(zero)

#(f)
    df2 = for_up_swr(df)

# #(g)(h)
#    bswr=[0.028483,0.03060,0.03126,0.03194,0.03250,0.03319,0.03460,0.03737]
#    r1,f1= gf1(bswr[0])
#    r2,f2= gf2(bswr[1],r1,f1)
#    r3,f3 = gf3(bswr[2],r1,f1,r2,f2)
#    r4,f4 = gf4(bswr[3],r1,f1,r2,f2,r3,f3)
#    r5,f5 = gf5(bswr[4],r1,f1,r2,f2,r3,f3,r4,f4)
#    r7,f7 = gf7(bswr[5],r1,f1,r2,f2,r3,f3,r4,f4,r5,f5)
#    r10,f10 = gf10(bswr[6],r1,f1,r2,f2,r3,f3,r4,f4,r5,f5,r7,f7)
#    r30 = optimize.root(gf302_b,0).x[0]
#    f30 = 1/0.5 * (math.exp(0.5*r30)-1)  
   
#    br=[r1,r2,r3,r4,r5,r7,r10,r30]
#    bf=[f1,f2,f3,f4,f5,f7,f10,f30]

# #(i)(j)
#     bbswr=[0.023483,0.028100,0.029760,0.030440,0.031000,0.031690,0.032100,0.032370]
#     r1,f1= gf1(bbswr[0])
#     r2,f2= gf2(bbswr[1],r1,f1)
#     r3,f3 = gf3(bbswr[2],r1,f1,r2,f2)
#     r4,f4 = gf4(bbswr[3],r1,f1,r2,f2,r3,f3)
#     r5,f5 = gf5(bbswr[4],r1,f1,r2,f2,r3,f3,r4,f4)
#     r7,f7 = gf7(bbswr[5],r1,f1,r2,f2,r3,f3,r4,f4,r5,f5)
#     r10,f10 = gf10(bbswr[6],r1,f1,r2,f2,r3,f3,r4,f4,r5,f5,r7,f7)
#     r30 = optimize.root(gf302,0).x[0]
#     f30 = 1/0.5 * (math.exp(0.5*r30)-1)  
    
#     bbr=[r1,r2,r3,r4,r5,r7,r10,r30]
#     bbf=[f1,f2,f3,f4,f5,f7,f10,f30]

   # swr = [0.028438,0.03060,0.03126,0.03144,0.03150,0.03169,0.03210,0.03237]
   # gap = [1,1,1,1,1,2,3,20]
   # year = [1,2,3,4,5,7,10,30]
   
    
    