#
#
#
import torch
import torch.nn as nn
import math
import numpy as np
from scipy import linalg
from numpy import linalg as LA
from torch.autograd import Variable

########################################################
############### SLOG-NET Modules #######################
########################################################
from SLoGexample import SLOGtools as SLOGtools


def myFunction_slog_1(rho_1,eta_1,lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # define variables
    K = rho_1.shape[0]
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()
    N = Y.shape[0]
    P = Y.shape[1]
    NP = int(N*P)
    # Initialization
    x = torch.randn(NP)
    u = np.zeros(NP)
    u = torch.tensor(u)
    eta = torch.tensor(np.zeros(1))
    II = torch.tensor(np.ones([N,N]))
    In = torch.tensor(np.ones(N)) # ones(N,1)
    # define function
    for k in range(K):
        # V update
        ZIk_inv = SLOGtools.fast_inverse(torch.matmul(torch.transpose(Z,0,1),Z),eta_1[k])
        v_temp = torch.matmul(torch.transpose(Z,0,1),x - rho_1[k]*u) + (eta_1[k]*C-rho_1[k]*eta)*In
        v = torch.matmul(ZIk_inv, v_temp)
        # X update
        x = SLOGtools.softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])  
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
    return x,v

def myFunction_slog_2(rho_1,lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3, V,Y,  K):
    # No constraint on g 
    # define variables
    K = rho_1.shape[0]
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()
    N = Y.shape[0]
    P = Y.shape[1]
    NP = int(N*P)
    # Initialization
    x = torch.randn(NP)
    u = np.zeros(NP)
    u = torch.tensor(u)
    # define function
    for k in range(K):
        # V update
        ZIk_inv = SLOGtools.fast_inverse_no_constrain(torch.matmul(torch.transpose(Z,0,1),Z))
        v_temp = torch.matmul(torch.transpose(Z,0,1),x - rho_1[k]*u)
        v = torch.matmul(ZIk_inv, v_temp)
        # X update
        x = SLOGtools.softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])  
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
    return x,v

def myFunction_slog_3(rho_1,eta_1,lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,M,m, V,Y, K):
    # Use new constrain on g: Mg = m
    # New learnable parameters: M(N,q,K), m(q,K)
    # define variables
    K = rho_1.shape[0]
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()
    N = Y.shape[0]
    P = Y.shape[1]
    q = M.shape[1]
    NP = int(N*P)
    # Initialization
    x = torch.randn(NP)
    u = np.zeros(NP)
    u = torch.tensor(u)
    eta = torch.tensor(np.zeros(q))
    II = torch.tensor(np.ones([N,N]))
    In = torch.tensor(np.ones(N)) # ones(N,1)
    # define function
    for k in range(K):
        # V update
#         ZIk_inv = SLOGtools.fast_inverse(torch.matmul(torch.transpose(Z,0,1),Z),eta_1[k])
        ZIk_inv = SLOGtools.fast_inverse_objf3(torch.matmul(torch.transpose(Z,0,1),Z),M[:,:,k],eta_1[k])
        v_temp = torch.matmul(torch.transpose(Z,0,1),x - rho_1[k]*u) + torch.matmul(M[:,:,k],eta_1[k]*m[:,k] - rho_1[k]*eta)
        v = torch.matmul(ZIk_inv, v_temp)
        # X update
        x = SLOGtools.softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])  
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
#         eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
        eta = gamma_1[k]*eta + torch.matmul(M[:,:,k].transpose(0,1),v) + m[:,k]
    return x,v