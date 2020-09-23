# -*- coding: utf-8 -*-
"""
State-Space Model of Boost DC-DC converter

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.linalg as la



class Boost:
    def __init__(self, L=0.5e-3, E=15, R=10, C=1e-3):  
        # Physical Parameters of the system 
        self.L = L  # Inductance [H]
        self.Rl = R  # Load Resistance [Ohm]
        self.C = C  # Capacitance [F]
        self.E = E  # Source Voltage [V]
        
        # Referense/Linearisation point
        self.uref = 2*self.E
        self.u0 = self.uref
        self.d0 = 1-self.E/self.u0
        self.i0 = self.u0/self.Rl/(1-self.d0)
        
        # Initial state
        self.x0 = np.array([0, 15, 0])
        
        # Time parameters
        self.dt = 1e-4
        self.tend = 0.2
        self.t = np.arange(0, self.tend, self.dt)
        
        self.input=None      
    
    def ode(self, X, t):
        # States
        i = X[0]
        u = X[1]

        # Duty cycle as input, limited [0,1]
        d= np.clip(self.input(X, t), 0, 1)
        
        # Propagation
        di = -(1-d)*u/self.L + self.E/self.L
        du =  (1-d)*i/self.C - u/(self.Rl*self.C)
        de = self.uref - u
        
        return np.array([di, du, de])
    
    def calc_LQR(self, Q=None, R=None):
        if Q==None: Q = np.array([[0,0,0],[0,0,0],[0,0,100]])
        if R==None: R = np.array([[0.1]])
        #Linearised model
        A = np.array([[0,                      -(1-self.d0)/self.L,   0],
                      [(1-self.d0)/self.C,     -1/(self.Rl*self.C),   0],
                      [0,                      -1,                    0]])
        
        B = np.array([[ self.u0/self.L],
                      [-self.i0/self.C],
                      [ 0]])
        
        S = la.solve_continuous_are(A, B, Q, R)
        self.K = np.matmul(la.inv(R), np.matmul(B.T, S))
        return self.K
    
    def step_input(self, X, t):
        return (t<0.1)*0.2 + 0.5
    
    def state_feedback(self, X, t):
        return - self.K[0,0]*(X[0]-self.i0) - self.K[0,1]*(X[1]-self.u0) - self.K[0,2]*X[2]
    
    def sim(self):
        if self.input==None: self.input = self.step_input
        return spi.odeint(self.ode, self.x0, self.t)
    
    def plot(self):
        X = self.sim()
        
        # Visualisation
        plt.figure(figsize=(11,7))
        
        plt.subplot(3,1,1)
        plt.plot(self.t, boost.input(X.T, self.t))
        plt.ylabel('Duty Cycle')
        plt.grid()
        
        plt.subplot(3,1,2)
        plt.plot(self.t, X[:,0])
        plt.ylabel('Current I [A]')
        plt.grid()
        
        plt.subplot(3,1,3)
        plt.plot(self.t, X[:,1])
        plt.ylabel('Voltage Uc [V]')
        plt.xlabel('Time [s]')
        plt.grid()
        
        if self.input == self.step_input:
            plt.savefig('StepResponse.png')
            
        # plt.figure(figsize=(11,7))
        # plt.plot(self.t, X[:,2])
        # plt.ylabel('Error Integral')
        # plt.grid()
            

if __name__ == "__main__":    
    boost = Boost()
    
    K = boost.calc_LQR()
    # boost.input = boost.state_feedback
    X = boost.plot()
    
    
