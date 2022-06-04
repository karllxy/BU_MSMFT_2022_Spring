

import numpy as np
import pandas as pd
from scipy.stats import norm
from sympy import *
from sympy.stats import Normal, density 
from sympy.stats import Normal, cdf

'''This document uses sympy to calculate the derivative of BS formular.'''
f0 = Symbol('f0')
k = Symbol('k')
sigma = Symbol('sigma',positive=True)
t = Symbol('t',real=True)
factor =  Symbol('factor')  
d1 = (log(f0/k) + 0.5 * sigma **2 * t) / sigma/t**0.5
d2 = (log(f0/k) - 0.5 * sigma **2 * t) / sigma/t**0.5
N = Normal('N',0,1)  
p = 0.25 * factor* (cdf(N)(-d2) * k - f0 * cdf(N)(-d1))


delta = diff(p,f0)
gamma =  diff(delta,f0)
vega =  diff(p,sigma)
theta = diff(p,t)

