# 
#
#
import numpy as np
import scipy
import torch
import torch.nn as nn

# import alegnn.utils.graphML as gml
# import alegnn.utils.graphTools

# from alegnn.utils.dataTools import changeDataType
########################################################
############### SLOG-NET Modules #######################
########################################################
from SLoGexample import SLOGtools as SLOGtools
from SLoGexample import SLOGobjective as SLOGobj


class GraphSLoG_v3(nn.Module):
    def __init__(self, V, N, q, K, myobj_function, **kwargs):
        super().__init__()
        # Inputs:
        # V: Orthogonal base
        # N: number of nodes
        # q: number of constraint vectors. M.T v = m, M.size = (N,q), m.size = q
        # K: number of layers        
        # P: number of observations
        # gso: the graph shift operator
        
        # Z: computes Zv = y
        # self.Z = torch.tensor(Z)
        
        
        ### Functions
        ##  .reset_parameters(self)
        # No input. Reset parameters to default.
        
        ##  .forward(self, Y)
        # Input:
        # Y: NP * N
        # Output:
        # Objective function
        
        self.N = int(N)
#         self.C = int(C)
        self.K = int(K) 
        self.q = int(q)
#         self.gso = gso
        self.V = V
        # Parameters
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.eta_1 = nn.Parameter(torch.randn(self.K))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.M = nn.Parameter(torch.randn(self.N,self.q,self.K))  
        self.m = nn.Parameter(torch.randn(self.q,self.K))  
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.eta_1 = nn.Parameter(torch.randn(self.K))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.M = nn.Parameter(torch.randn(self.N,self.q,self.K))  
        self.m = nn.Parameter(torch.randn(self.q,self.K))          
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  

    def forward(self, Y):
#         return SLOGobj.myFunction_slog_1(self.rho_1,self.eta_1,self.lmbd,self.alpha_1,self.alpha_2, self.beta_1, self.beta_2, self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V, Y, self.C, self.K)
        return SLOGobj.myFunction_slog_3(self.rho_1,self.eta_1,self.lmbd,self.alpha_1,self.alpha_2, self.beta_1, self.beta_2, self.beta_3,self.gamma_1,self.M,self.m, self.V, Y, self.K)
    