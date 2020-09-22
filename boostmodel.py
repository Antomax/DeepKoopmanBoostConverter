# -*- coding: utf-8 -*-
"""
State-Space Model of Boost DC-DC converter

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi


def boost_model(X,t,u_f):
    d = np.clip(u_f(t), 0, 1)
    
    L = 0.5e-3
    E = 15
    R = 10
    C = 1000e-6
    
        
    i     = X[0];
    u     = X[1];
   
    di = -(1-d)*u/L + E/L
    du = (1-d)*i/C - u/(R*C)
    
    dX = [di, du]
    return np.array(dX)  



if __name__ == "__main__":

    
    x0 = np.array([0, 0])
    t = np.linspace(0, 30, 1001)
    u = lambda t: t>1
    
    X = spi.odeintboost_model, x0, t, args=(u,))
    
