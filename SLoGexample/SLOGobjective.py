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


def myFunction_slog_1(rho_1, eta_1, lmbd, alpha_1, alpha_2, beta_1, beta_2, beta_3, gamma_1, gamma_2, gamma_3, V, Y, C, K):
    """
    This function implements the first version of an optimization procedure using SLOG (Structured Low-rank Optimization) techniques.

    Parameters:
    rho_1, eta_1, lmbd, alpha_1, alpha_2, beta_1, beta_2, beta_3, gamma_1, gamma_2, gamma_3: 1D tensors
        The parameters used in the iterative updates of the variables.
    V, Y: 2D tensors
        V is the matrix related to the features or observations, and Y represents the observations.
    C: 2D tensor
        A constant matrix used in the update equations.
    K: int
        The number of iterations to run the optimization procedure.

    Returns:
    x: Tensor
        The final updated solution vector.
    v: Tensor
        The final updated value for the vector v.
    """
    K = rho_1.shape[0]  # Extract the number of iterations from rho_1 shape
    # Compute the Khatri-Rao product of matrices
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y), V), V), requires_grad=False)
    ZT = Z.t()  # Transpose of Z
    N = Y.shape[0]  # Number of rows in Y (i.e., observations)
    P = Y.shape[1]  # Number of columns in Y (i.e., features)
    NP = int(N * P)  # Total number of elements in the unfolded vector

    # Initialization of variables
    x = torch.randn(NP)  # Initialize x with random values
    u = np.zeros(NP)  # Initialize u with zeros
    u = torch.tensor(u)  # Convert u to tensor
    eta = torch.tensor(np.zeros(1))  # Initialize eta with zero
    II = torch.tensor(np.ones([N, N]))  # Identity matrix (N x N) of ones
    In = torch.tensor(np.ones(N))  # Vector of ones (N, 1)

    # Start the iterative updates for K iterations
    for k in range(K):
        # Update the vector V
        ZIk_inv = SLOGtools.fast_inverse(torch.matmul(torch.transpose(Z, 0, 1), Z), eta_1[k])
        v_temp = torch.matmul(torch.transpose(Z, 0, 1), x - rho_1[k] * u) + (eta_1[k] * C - rho_1[k] * eta) * In
        v = torch.matmul(ZIk_inv, v_temp)  # Final value of v

        # Update the vector X
        x = SLOGtools.softshrink(alpha_1[k] * torch.matmul(Z, v) + alpha_2[k] * u, lmbd[k])

        # Update the auxiliary variable u
        u = beta_1[k] * u + beta_2[k] * torch.matmul(Z, v) + beta_3[k] * x

        # Update the parameter eta
        eta = gamma_1[k] * eta + gamma_2[k] * torch.matmul(torch.tensor(np.ones([1, N])), v) + gamma_3[k] * C

    return x, v  # Return the updated vectors x and v


def myFunction_slog_3(rho_1, eta_1, lmbd, alpha_1, alpha_2, beta_1, beta_2, beta_3, gamma_1, M, m, V, Y, K):
    """
    This function implements the second version of the SLOG optimization, introducing a new constraint on the variable g, where Mg = m.

    Parameters:
    rho_1, eta_1, lmbd, alpha_1, alpha_2, beta_1, beta_2, beta_3, gamma_1: 1D tensors
        These parameters control the iterative updates of the optimization process.
    M: 3D tensor
        A learnable matrix that introduces a new constraint on the variable g.
    m: 2D tensor
        A constant matrix that, together with M, defines the constraint Mg = m.
    V, Y: 2D tensors
        V represents features or observations, and Y represents the data/observations.
    K: int
        The number of iterations to run the optimization.

    Returns:
    x: Tensor
        The updated solution vector after optimization.
    v: Tensor
        The updated value of the vector v after optimization.
    """
    K = rho_1.shape[0]  # Extract the number of iterations from rho_1 shape
    # Compute the Khatri-Rao product of the matrices V and Y
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y), V), V), requires_grad=False)
    ZT = Z.t()  # Transpose of Z
    N = Y.shape[0]  # Number of observations (rows)
    P = Y.shape[1]  # Number of features (columns)
    q = M.shape[1]  # Number of columns in the M matrix (related to constraint)
    NP = int(N * P)  # Total number of elements in the unfolded vector

    # Initialization of variables
    x = torch.randn(NP)  # Initialize x with random values
    u = np.zeros(NP)  # Initialize u with zeros
    u = torch.tensor(u)  # Convert u to tensor
    eta = torch.tensor(np.zeros(q))  # Initialize eta with zeros
    II = torch.tensor(np.ones([N, N]))  # Identity matrix of ones (N x N)
    In = torch.tensor(np.ones(N))  # Vector of ones (N, 1)

    # Start the iterative updates for K iterations
    for k in range(K):
        # Update the vector V with the inverse of the updated matrix
        ZIk_inv = SLOGtools.fast_inverse_objf3(torch.matmul(torch.transpose(Z, 0, 1), Z), M[:, :, k], eta_1[k])
        v_temp = torch.matmul(torch.transpose(Z, 0, 1), x - rho_1[k] * u) + torch.matmul(M[:, :, k], eta_1[k] * m[:, k] - rho_1[k] * eta)
        v = torch.matmul(ZIk_inv, v_temp)  # Final value of v

        # Update the vector X
        x = SLOGtools.softshrink(alpha_1[k] * torch.matmul(Z, v) + alpha_2[k] * u, lmbd[k])

        # Update the auxiliary variable u
        u = beta_1[k] * u + beta_2[k] * torch.matmul(Z, v) + beta_3[k] * x

        # Update the parameter eta
        eta = gamma_1[k] * eta + torch.matmul(M[:, :, k].transpose(0, 1), v) + m[:, k]

    return x, v  # Return the updated vectors x and v
