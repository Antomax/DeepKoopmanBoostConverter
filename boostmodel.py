# -*- coding: utf-8 -*-
"""
State-Space Model of Boost DC-DC converter

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi


def boost_model(X,t,u_f):
    # Physical Parameters of the system    
    L = 0.5e-3      # Inductance [H]
    E = 15          # Source Voltage [V]
    R = 10          # Load Resistance [Ohm]
    C = 1000e-6     # Capacitance [F]
        
    # State space description
    i     = X[0];   # Inductor current [A]
    u     = X[1];   # Capasitor Voltage [V]
    
    # Duty cycle as input, limited [0,1]
    d = np.clip(u_f(t), 0, 1)
    
    di = -(1-d)*u/L + E/L
    du = (1-d)*i/C - u/(R*C)
    
    dX = [di, du]
    return np.array(dX)  


if __name__ == "__main__":    
    x0 = np.array([0, 15])          # Initial conditions
    t = np.linspace(0, 0.2, 1001)   # Time span
    
    def u(t): return (t<0.1)*0.2 + 0.5  # Step input function
    
    X = spi.odeint(boost_model, x0, t, args=(u,))
    IL = X[:,0]
    Uc = X[:,1]
    
    # Visualisation
    plt.figure(figsize=(16,9))
    
    plt.subplot(3,1,1)
    plt.plot(t, u(t))
    plt.ylabel('Duty Cycle')
    plt.grid()
    
    plt.subplot(3,1,2)
    plt.plot(t, IL)
    plt.ylabel('Current I [A]')
    plt.grid()
    
    plt.subplot(3,1,3)
    plt.plot(t, Uc)
    plt.ylabel('Voltage Uc [V]')
    plt.xlabel('Time [s]')
    plt.grid()
    
    plt.savefig('StepResponce.png')
    
    
