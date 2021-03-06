# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:46:14 2020

@author: maksakov
"""
import time
import matplotlib.pyplot as plt
import torch
from boost.boostmodel import Boost
from encoder import AutoEncoder
import timeit
import numpy as np
   
# device="cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

## Parameters
N = 30;
S = 2;
L = 1;
dt = 0.0001;

## Define Boost Converter Model
boost = Boost()
boost.input = boost.rand_input
# boost.rand_dt = 0.001

## Define Model
if 'model' not in locals():
    model = AutoEncoder(L=L, N=N, S=S, dt=dt).to(device)
    print('New model created!')
    
## Define Optimizer and Loss
from torch import optim, nn   
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-15)
# opt = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-15)
loss_fn = nn.SmoothL1Loss()
# loss_fn = nn.L1Loss()
# loss_fn = nn.MSELoss()

#%% Pretrain Autoencoder
lossV = []
model.train()

for epoch in range(100):
    # boost.rand_dt = boost.dt    
    boost.new_random_input()
    x = boost.sim()/boost.xmax
    u = boost.rand_input(0, boost.t)

    for it in range(100):
        X,U = boost.rand_batch(x,u, batch=64, L=L, N=N);
        
        # Autoencoder - Reconstruction Loss
        X_code    = model.encode(X[:,:L])          # Encoder transform to hidden space
        X_hat     = model.decode(X_code)           # Decoder transform to original space  
        X_hat_lin = model.lin_reconstruct(X_code)  # Linear reconstruction of output C
        
        loss_lin    = loss_fn(X_hat_lin, X[:,:L, 1])
        loss_nonlin = loss_fn(X_hat, X[:,:L])
        
        loss = loss_lin + loss_nonlin
        loss.backward()
        opt.step()
        opt.zero_grad()
      
        lossV.append(loss.item())
    print('{} Loss_lin: {}, Loss_nonlin: {}'.format(epoch, loss_lin.item(), loss_nonlin.item()))


plt.plot(lossV)
plt.show()





#%% Train model
model.train()
boost.rand_dt = 0.05
for epoch in range(100):
    
    #Generate Data
    boost.new_random_input()
    boost.input = boost.rand_input
    x = boost.sim()/boost.xmax
    u = boost.rand_input(0, boost.t)
    
    for it in range(50):
        X,U = boost.rand_batch(x,u, batch=64, L=L, N=N);
        
        # Transform every input window to hidden space
        X_codes   = []
        for i in range(N):
            X_codes.append( model.encode(X[:,i:L+i]) )
            
        X_hat     = model.decode(X_codes[0])
        X_hat_lin = model.lin_reconstruct(X_codes[0]) 
        
        X_codes_predict = model.propagate(X_codes[0], U) #Propagate from first to last    
        
        loss_pred_code, loss_pred_x = 0, 0;
        for i, Xk_code in enumerate(X_codes_predict):
            pred  = model.decode(Xk_code)
            loss_pred_x += loss_fn(pred, X[:,i:i+L])
            loss_pred_code  += loss_fn(Xk_code, X_codes[i+1])
            
        loss_nonlin  = loss_fn(X_hat, X[:,i:i+L])
        loss_lin     = loss_fn(X_hat_lin, X[:,i:i+L,1])
            
        loss = (loss_pred_x/(N-1) + loss_pred_code/(N-2) + (loss_nonlin + loss_lin)*0.1)    
              
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    print('{} Reconst:Lin: {}, Nonlin: {}, Pred: Hid:{}, Orig {}'.format(epoch, loss_lin.item(), loss_nonlin.item(), loss_pred_code.item(), loss_pred_x.item() ))
    if (epoch%10 == 0)  & (epoch!= 0):
        name = "buck" + str(epoch) +"_" + str(time.localtime().tm_mday) +"_"+ str(time.localtime().tm_hour) +"_"+ str(time.localtime().tm_min) + ".mat"
        model.matsave(name = name)
        
        with torch.no_grad():
            boost.encoder = model.encoder
            boost.calc_NN_LQR(model.K.weight.data.cpu().numpy(),
                              model.C.weight.data.cpu().numpy(),
                              model.B.weight.data.cpu().numpy())
            print(boost.Knn)
            boost.encoder = model.encoder
            boost.input = boost.koopman_feedback
            boost.plot()

name = "buck" + str(time.localtime().tm_mon) +"_" + str(time.localtime().tm_mday) +"_"+ str(time.localtime().tm_hour) +"_"+ str(time.localtime().tm_min) + ".mat"
model.matsave(name = name)
print("saved")






