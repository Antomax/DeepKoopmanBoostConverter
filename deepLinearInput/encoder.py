# -*- coding: utf-8 -*-
"""
Deep NN for Koopman subspace approximation

"""
from scipy import io
import torch
from torch import nn
import prepdata



class AutoEncoder(nn.Module):
    def __init__(self, L=8, N=3, S=1, dt=0.001):
        super(AutoEncoder, self).__init__()
        
        self.N      = N # Number of predictions
        self.L      = L # Length of window
        self.S      = S # Number of states
        self.dt     = dt
        
        self.in_dim  = self.L*self.S
        self.hid_dim = self.in_dim*3
        
        self.encoder = nn.Sequential(
                nn.Linear(self.in_dim, self.hid_dim*3//4),
                nn.ReLU(),
                nn.Linear(self.hid_dim*3//4, self.hid_dim*3//5),
                nn.ReLU(),
                nn.Linear(self.hid_dim*3//5, self.hid_dim))

        self.decoder = nn.Sequential(
                nn.Linear(self.hid_dim, self.hid_dim*3//5),
                nn.ReLU(),
                nn.Linear(self.hid_dim*3//5, self.hid_dim*3//4),
                nn.ReLU(),
                nn.Linear(self.hid_dim*3//4, self.in_dim))
        
        self.K = nn.Linear(self.hid_dim, self.hid_dim, bias=False)
        self.B = nn.Linear(1, self.hid_dim, bias=False)
        self.C = nn.Linear(self.hid_dim, 1, bias=False)
        
        self.input_transform = nn.Sequential(
                        nn.Linear(1, 4),
                        nn.ReLU(),
                        nn.Linear(4, 4),
                        nn.ReLU(),
                        nn.Linear(4, 1))
        self.inv_input_transform = nn.Sequential(
                        nn.Linear(1, 4),
                        nn.ReLU(),
                        nn.Linear(4, 4),
                        nn.ReLU(),
                        nn.Linear(4, 1))
        
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)
        
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)
        # m.weight.copy_(torch.triu(m.weight))
                
        


        
    def matsave(self, name="AutoEncoder.mat"):
        W ={}
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                W["Wenc" + str(i)]=layer.weight.data.cpu().numpy()
                W["Benc" + str(i)]=layer.bias.data.cpu().numpy()
                
        for i, layer in enumerate(self.input_transform):
            if isinstance(layer, nn.Linear):
                W["Win" + str(i)]=layer.weight.data.cpu().numpy()
                W["Bin" + str(i)]=layer.bias.data.cpu().numpy()
        
        for i, layer in enumerate(self.inv_input_transform):
            if isinstance(layer, nn.Linear):
                W["Win_inv" + str(i)]=layer.weight.data.cpu().numpy()
                W["Bin_inv" + str(i)]=layer.bias.data.cpu().numpy()
                
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.Linear):
                W["Wdec" + str(i)]=layer.weight.data.cpu().numpy()
                W["Bdec" + str(i)]=layer.bias.data.cpu().numpy()
        
        W["Kw"] = self.K.weight.data.cpu().numpy()
        W["Bw"] = self.B.weight.data.cpu().numpy()
        W["Cw"] = self.C.weight.data.cpu().numpy()
        
        W["L"]  = self.L
        W["dt"]  = self.dt
        
        io.savemat(name, W)
            
        
    def encode(self, x):
        # Encode input
        if self.S == 2:
            code = self.encoder(x.transpose(1,2).reshape(-1, self.L*self.S))
        else:
            code = self.encoder(x[:,:,0])
        return code
        
    
    def decode(self, code):
        x = self.decoder(code)
        if self.S == 2:
            return torch.stack((x[:,:self.L], x[:,self.L:]), dim=-1)
        else:
            return x.unsqueeze(-1)
            
    
    def forward(self, x):
        latent = self.encoder(x)
        # self.K.weight.copy_(torch.triu(self.K.weight))
        # Propagate 
        for _ in range(self.N):
            latent = self.K(latent)
        out = self.decoder(latent)  
        return out
    
    def reconstruct(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)
    
    def lin_reconstruct(self, x):
        return self.C(x)
    
    def propagate(self, code, U):
        # self.K.weight.copy_(torch.triu(self.K.weight))
        codes = [self.K(code) + self.B(self.input_transform(U[:,0:1]))] 
        for i in range(self.N-2):
            codes.append(self.K(codes[-1]) + self.B(self.input_transform(U[:,i+1:i+2])))
        return codes
        
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    model = AutoEncoder(L=8, N=1, S=2)
    x = torch.rand((64,8,2))
    code = model.encode(x)
    x_h = model.decode(code)
    
    print(x.shape)
    print(code.shape)
    print(x_h.shape)

    
    # plt.plot(x[0,:])
    # plt.plot(x_h[0,:])
    
