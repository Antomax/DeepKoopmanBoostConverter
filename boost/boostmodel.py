# -*- coding: utf-8 -*-
"""
State-Space Model of Boost DC-DC converter

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.linalg as la
import scipy.io as io
import torch

class Boost:
    def __init__(self, L=0.5e-3, E=15, R=10, C=1e-3):  
        
        # Physical Parameters of the system 
        self.L = L      # Inductance [H]
        self.Rl = R     # Load Resistance [Ohm]
        self.C = C      # Capacitance [F]
        self.E = E      # Source Voltage [V]
        
        # Referense/Linearisation point
        self.uref = self.E
        self.u0 = 2*self.uref
        self.d0 = 1-self.E/self.u0
        self.i0 = self.u0/self.Rl/(1-self.d0)
        
        # Initial state
        self.x0 = np.array([self.i0, self.u0, 0])
        
        # Time parameters
        self.dt = 1e-4
        self.tend = 0.2
        self.t = np.arange(0, self.tend, self.dt)
        
        # Input parameters
        self.input=None  
        self.rand_dt = 0.05
        self.rand_values = np.random.rand(int(self.tend//self.rand_dt)+1)*0.7
        
        self.encoder = None
        
        # Normalisation vector for states: [Imax, Umax, Emax]
        # self.xmax = [20, 80, 1]
        self.xmax = [1, 1, 1]
        
    # Simulation Functions
    def ode(self, X, t):
        # Defines the ODEs for boost converter
        
        # States
        i = X[0]    # Inductor Current [A]
        u = X[1]    # Capacitor Voltage [V]

        # Duty cycle as input, limited [0,1]
        d = np.clip(self.input(X, t), 0, 1)
        
        # Propagation
        di = -(1-d)*u/self.L + self.E/self.L 
        du =  (1-d)*i/self.C - u/(self.Rl*self.C)
        de =  self.uref - u

        return np.array([di, du, de])
    
    def sim(self):
        # Solves the ODEs and normilizes the output
        if self.input==None: self.input = self.step_input
        return spi.odeint(self.ode, self.x0, self.t)
    
    def plot(self):
        X = self.sim()
        
        # Visualisation
        plt.figure(figsize=(11,7))
        
        plt.subplot(3,1,1)
        plt.plot(self.t, self.input(X, self.t))   
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
        plt.show()
        print(np.sum(X[1000:,2]))
        
        if self.input == self.step_input:
            plt.savefig('StepResponse.png')
    
    # Input functions
    def step_input(self, X, t):
        return (t<0.1)*0.2 + 0.5
    
    def rand_input(self, X, t):
        self.uref = self.E
        index = np.array(t//self.rand_dt, dtype=int)
        return self.rand_values[index]
    
    def state_feedback(self, X, t):
        self.uref = (t<0.1)*2*self.E + (t>=0.1)*3*self.E
        if X.ndim==1: X = np.expand_dims(X, 0)
        out = - self.K[0,0]*(X[:,0]-self.i0) - self.K[0,1]*(X[:,1]-self.u0) - self.K[0,2]*X[:,2]
        return out.squeeze()
    
    def koopman_feedback(self, X, t):
        self.uref = (t<0.1)*2*self.E + (t>=0.1)*3*self.E
        X = X/self.xmax
        if X.ndim==1:
            Z = self.encoder(torch.tensor(np.expand_dims(X[0:2],0)).float())
        else:
            Z = self.encoder(torch.tensor(X[:,0:2]).float())
        Z = Z.detach().numpy()
        out = - np.matmul(self.Knn[0,0:-1], Z.T) - self.Knn[0,-1]*X.T[2]*self.xmax[2]
        return out.squeeze()

    # Optimal Control Calculation
    def calc_LQR(self,A=None, B=None, Q=None, R=None):
        if Q is None: Q = np.array([[0,0,0],[0,0,0],[0,0,100]])
        if R is None: R = np.array([[1]])
        #Linearised model
        if A is None:
            A = np.array([[0,                      -(1-self.d0)/self.L,   0],
                          [(1-self.d0)/self.C,     -1/(self.Rl*self.C),   0],
                          [0,                      -1,                    0]])
        
        if B is None:
            B = np.array([[ self.u0/self.L],
                          [-self.i0/self.C],
                          [ 0]])
        
        S = la.solve_continuous_are(A, B, Q, R)
        self.K = np.matmul(la.inv(R), np.matmul(B.T, S))
        return self.K
    
    def calc_NN_LQR(self, Kw, Cw, Bw, Q=None, R=None):
        N = len(Kw)
        if Q==None: 
            Q = np.zeros((N+1, N+1))
            Q[-1,-1] = 100
        if R==None: R = np.array([[1]])
        #Linearised model
        A = np.concatenate([Kw, -Cw*self.dt*self.xmax[1]], axis=0)
        A = np.concatenate([A, np.zeros((N+1,1))], axis=1)
        A[-1,-1]=1
        
        B = np.concatenate([Bw, np.zeros((1,1))], axis=0)
        
        S = la.solve_discrete_are(A, B, Q, R)
        
        self.Knn = np.matmul(la.inv(R + np.matmul(np.matmul(B.T,S),B)), np.matmul(np.matmul(B.T, S), A))
        return self.Knn
    
    # Utills
    def rand_batch(self, x, u, batch=64, L=3, N=30):    
        index = np.random.randint(0, len(x)-L-N, size=(batch))
        X = torch.tensor([x[i:i+N+L,0:2] for i in index], dtype=torch.float32)
        U = torch.tensor([u[i:i+N] for i in index], dtype=torch.float32)
        return X, U
    
    def new_random_input(self):
        self.rand_values = np.random.rand(int(self.tend//self.rand_dt)+1)*0.7
        return self.rand_values
            

if __name__ == "__main__":    
    # Define model
    boost = Boost()
    
    # Test step input
    boost.input = boost.step_input
    boost.plot()
    
    # Test random input
    boost.input = boost.rand_input
    boost.new_random_input()
    boost.plot()
    
    # Test LQR control input
    boost.calc_LQR()
    boost.input = boost.state_feedback
    boost.plot()
    


    
    
    
