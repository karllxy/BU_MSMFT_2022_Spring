# 
# HM1.py - MF728
# Name: Xuyang Liu
# Email address: xyangliu@bu.edu
#
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sympy import * 



def inception(R,DF,s1,s2,s3,s5):
# obvoiusly, P0 = 1 , because nobody would default at beginning.    
    p1 = symbols('p1')
    premleg1 = s1/2 * DF * (p1+1)
    contleg1 = (1-R) * DF * (1-p1)
    p1 = solve(premleg1-contleg1,p1)[0]
    
    p2 = symbols('p2')
    premleg2 = s2/2 * (DF*(1+p1)+DF**2 * (p1+p2))
    contleg2 = (1-R) *(DF*(1-p1)+DF**2 * (p1-p2))
    p2 = solve(premleg2-contleg2,p2)[0]

    p3 = symbols('p3')
    premleg3 = s3/2 * (DF*(1+p1)+DF**2 * (p1+p2)+ DF**3 * (p2+p3))
    contleg3 = (1-R) *(DF*(1-p1)+DF**2 * (p1-p2)+ DF**3 * (p2-p3))
    p3 = solve(premleg3-contleg3,p3)[0]
    
    

    p5 = symbols('p5')
    p4 = p3 * (p5/p3)**0.5
    premleg5 = s5/2 * (DF*(1+p1)+DF**2 * (p1+p2)+ DF**3 * (p2+p3)+ DF**4 *(p3+p4)+ DF**5*(p4+p5))
    contleg5 = (1-R) *(DF*(1-p1)+DF**2 * (p1-p2)+ DF**3 * (p2-p3) + DF**4*(p3-p4)+DF**5 * (p4-p5))
    p5 = solve(premleg5-contleg5,p5)[0]
    
    return p1, p2, p3, p5


def p4(R,DF,s1,s2,s3,s5,p1,p2,p3,p5):
    hr3 = (math.log(p3) - math.log(p5))/2
    p4 = math.exp(math.log(p3) - hr3 * (4 - 3))
    
    s4 = symbols('s4')
    premleg4 = s4/2 * (DF*(1+p1)+DF**2 * (p1+p2)+ DF**3 * (p2+p3)+ DF**4 *(p3+p4))
    contleg4 = (1-R) *(DF*(1-p1)+DF**2 * (p1-p2)+ DF**3 * (p2-p3) + DF**4 * (p3-p4))
    s4 = solve(premleg4-contleg4,s4)[0]
    
    return p4, s4
    

def MM(R,DF,s1,s2,s3,s5,p1,p2,p3,p5):
    
    s0 = 80/10000
    ns1 = symbols('ns1')
    p4 = p3 * (p5/p3)**0.5
    premleg = ns1/2 * (DF**2 * (p1+p2)+ DF**3 * (p2+p3)+DF**4 *(p3+p4) +DF**5 *(p4+p5))
    contleg = (1-R) *(DF**2 * (p1-p2)+ DF**3 * (p2-p3) +DF**4*(p3-p4)+ DF**5 *(p4-p5))
    ns1 = solve(premleg-contleg,ns1)[0]
    
    MM = (ns1-s0) * (DF**2 * (p1+p2)+ DF**3 * (p2+p3)+ DF**5 *(p3+p5))/2
    
    return ns1,MM


def senCDS(R,DF,H0,H1,H2,H3,p0,p1,p2,p3,p4,s4):
    dv = 1/10000
# change in H0:
    H0l = H0+dv
    P1l = math.exp(math.log(p0) - H0l * (1- 0))
    P2l = math.exp(math.log(P1l) - H1 * (2- 1))
    P3l = math.exp(math.log(P2l) - H2 * (2- 1))
    P4l = math.exp(math.log(P3l) - H3 * (2- 1))
    
    s4_1 = symbols('s4_1')
    pre1 = s4_1 / 2 * (DF*(p0+P1l) + DF**2 * (P1l+P2l)+ DF**3 *(P2l+P3l) + DF**4 * (P3l+P4l))
    con1 = (1-R) * (DF*(p0 - P1l) + DF**2 * (P1l- P2l)+ DF**3 *(P2l - P3l) + DF**4 * (P3l - P4l))
    s4_1 = solve(pre1-con1,s4_1)[0]
    dv01_1 = (s4_1-s4) /dv
    
# change in H1,H2,H3:
    H1l = H1+dv
    P1l = p1
    P2l = math.exp(math.log(P1l) - H1l * (2- 1))
    P3l = math.exp(math.log(P2l) - H2 * (2- 1))
    P4l = math.exp(math.log(P3l) - H3 * (2- 1))
    
    s4_2 = symbols('s4_2')
    pre1 = s4_2 / 2 * (DF*(p0+P1l) + DF**2 * (P1l+P2l)+ DF**3 *(P2l+P3l) + DF**4 * (P3l+P4l))
    con1 = (1-R) * (DF*(p0 - P1l) + DF**2 * (P1l- P2l)+ DF**3 *(P2l - P3l) + DF**4 * (P3l - P4l))
    s4_2 = solve(pre1-con1,s4_2)[0]
    dv01_2 = (s4_2-s4) /dv

    H2l = H2+dv
    P1l = p1
    P2l = p2
    P3l = math.exp(math.log(P2l) - H2l * (2- 1))
    P4l = math.exp(math.log(P3l) - H3 * (2- 1))
    
    s4_3 = symbols('s4_3')
    pre1 = s4_3 / 2 * (DF*(p0+P1l) + DF**2 * (P1l+P2l)+ DF**3 *(P2l+P3l) + DF**4 * (P3l+P4l))
    con1 = (1-R) * (DF*(p0 - P1l) + DF**2 * (P1l- P2l)+ DF**3 *(P2l - P3l) + DF**4 * (P3l - P4l))
    s4_3 = solve(pre1-con1,s4_3)[0]
    dv01_3 = (s4_3-s4) /dv
    
    H3l = H3+dv
    P1l = p1
    P2l = p2
    P3l = p3
    P4l = math.exp(math.log(P3l) - H3l * (2- 1))
    
    s4_4 = symbols('s4_4')
    pre1 = s4_4 / 2 * (DF*(p0+P1l) + DF**2 * (P1l+P2l)+ DF**3 *(P2l+P3l) + DF**4 * (P3l+P4l))
    con1 = (1-R) * (DF*(p0 - P1l) + DF**2 * (P1l- P2l)+ DF**3 *(P2l - P3l) + DF**4 * (P3l - P4l))
    s4_4 = solve(pre1-con1,s4_4)[0]
    dv01_4 = (s4_4-s4) /dv
    
    return dv01_1,dv01_2,dv01_3,dv01_4
    
   

def senir(R,p0,p1,p2,p3,p4,s4):
    dv = 1/10000
    rl = 0.02 + dv
    DF = 1/(1+rl)
    s4r = symbols('s4r')
    pre = s4r/2 * (DF*(1+p1)+DF**2 * (p1+p2)+ DF**3 * (p2+p3)+ DF**4 *(p3+p4))
    con = (1-R) *(DF*(1-p1)+DF**2 * (p1-p2)+ DF**3 * (p2-p3) + DF**4 * (p3-p4))
    s4r = solve(pre-con,s4r)[0]
    
    return (s4r-s4) /dv
    
def senR(R,DF,p0,p1,p2,p3,p4,s4):
    dv = 1/10000
    R = R + dv
    s4R = symbols('s4R')
    pre = s4R/2 * (DF*(1+p1)+DF**2 * (p1+p2)+ DF**3 * (p2+p3)+ DF**4 *(p3+p4))
    con = (1-R) *(DF*(1-p1)+DF**2 * (p1-p2)+ DF**3 * (p2-p3) + DF**4 * (p3-p4))
    s4R = solve(pre-con,s4R)[0]
    
    return (s4R-s4) /dv








if __name__ == '__main__':
    R = 0.4
    DF = 1/(1.02)
    s1 = 0.01
    s2 = 0.011
    s3 = 0.012
    s5 = 0.014
    p0=1
    
    p1,p2,p3,p5 = inception(R,DF,s1,s2,s3,s5)
    H0 = (math.log(p0) - math.log(p1))/(1 - 0)
    H1 = (math.log(p1) - math.log(p2))/(2 - 1)
    H2 = (math.log(p2) - math.log(p3))/(3 - 2)
    H3 = (math.log(p3) - math.log(p5))/(5 - 3)
    p = [p1,p2,p3,p5]
    h = [H0,H1,H2,H3]
    print('The Survival Probability is about: ',p,'\nand the Harzard rate is: ', h)    
    p4,s4 = p4(R,DF,s1,s2,s3,s5,p1,p2,p3,p5)
    print('the fairly spread for 4CDS is : ', s4)
    ns1,MM = MM(R,DF,s1,s2,s3,s5,p1,p2,p3,p5)
    print('the Mark-to-Market value is about: ', MM)

    
    senCDS = senCDS(R,DF,H0,H1,H2,H3,p0,p1,p2,p3,p4,s4)
    print('DV01 wrt to CDS curve ', senCDS)
    senir = senir(R,p0,p1,p2,p3,p4,s4)
    print('DV01 wrt to interest rate  ',senir)
    senR = senR(R,DF,p0,p1,p2,p3,p4,s4)
    print('Sensitivity wrt R: ', senR)
    
    