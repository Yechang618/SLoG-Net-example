# 2024/08/15~
# Chang Ye, cye7@ur.rochester.edu

import numpy as np
from numpy import linalg as LA
import torch; torch.set_default_dtype(torch.float64)  # Set default data type for torch tensors to float64
import torch.nn as nn
import torch.optim as optim
import copy
from copy import deepcopy
import matplotlib.pyplot as plt  # Importing matplotlib for visualization
import matplotlib.cm as cm
from scipy import linalg  # For linear algebra operations in SciPy
from timeit import default_timer as timer  # For measuring execution time
import networkx as nx  # NetworkX for graph-related operations
import os  # For file path operations
import pickle  # For serializing and deserializing Python objects
import datetime  # For handling dates and times

#### Import SLOG packages
from SLOGmodules import SLOGtools as SLOGtools  # Importing SLOG utility functions
from SLOGmodules import SLOGobjective as SLOGobj  # Importing SLOG objectives
from SLOGmodules import SLOGarchitectures as SLOGarchi  # Importing SLOG network architectures
from SLOGmodules import SLOGtraining as SLOGtrainer  # Importing SLOG training tools
from SLOGmodules import SLOGmodel as SLOGmodel  # Importing SLOG model
from SLOGmodules import SLOGevaluation as SLOGevaluator  # Importing SLOG evaluation tools
from SLOGmodules import SLOGdata as SLOGdata  # Importing SLOG data utilities

#### Import GNN packages (Graph Neural Networks)
from SLOGmodules import graphTools as graphTools  # Tools for working with graphs
from SLOGmodules import dataTools as dataTools  # Data handling tools for graphs
from alegnn.utils import graphML as gml  # Import graphML for graph data handling

from alegnn.modules import architectures as archit  # Importing architectures from alegnn
from alegnn.modules import model as model  # Importing model components from alegnn
from alegnn.modules import training as training  # Importing training utilities from alegnn
from alegnn.modules import evaluation as evaluation  # Importing evaluation utilities from alegnn
from alegnn.modules import loss as loss  # Importing loss functions from alegnn
from alegnn.utils.miscTools import writeVarValues  # Utility for writing variable values
from alegnn.utils.miscTools import saveSeed  # Utility for saving random seeds

### Import trained models
from SLOGTrainedModels import trainedModels as trainedModels  # Importing pre-trained models

# Function to convert from Torch tensor to NumPy array
def to_numpy(x):
    dataType = type(x)  # Get the data type of the input variable
    if 'numpy' in repr(dataType):  # If it is already a NumPy array, return it as is
        return x
    elif 'torch' in repr(dataType):  # If it is a Torch tensor, convert it to NumPy
        x1 = x.clone().detach().requires_grad_(False)  # Detach from computation graph
        return x1.numpy()

# Function to convert from NumPy array to Torch tensor
def to_torch(x):
    dataType = type(x)  # Get the data type of the input variable
    if 'numpy' in repr(dataType):  # If it is a NumPy array, convert it to a Torch tensor
        return torch.tensor(x)
    elif 'torch' in repr(dataType):  # If it is already a Torch tensor, return it as is
        return x  

# Main function for noise testing with dropbox data
def noiseTest_dropbox(nNodes, P, S, modelDir, **kwargs):
    ## Assertion: No explicit assertion in the code, but it's implied that parameters must be provided
    
    ## Parameters loading (kwargs) - Handling model and simulation parameters
    if 'location' in kwargs.keys():
        location = kwargs['location']
    else:
        location = 'office'  # Default location is office

    # Model parameters and simulation parameters setup
    if 'modelParas' in kwargs.keys():
        modelParas = kwargs['modelParas']
    else:
        modelParas = {}

    # Number of communities or sources
    if 'q' in modelParas.keys():
        q = modelParas['q']
    else:
        q = 4  # Default number of sources

    # Simulation parameters setup
    if 'simuParas' in kwargs.keys():
        simuParas = kwargs['simuParas']
    else:
        simuParas = {}

    # Alpha parameter for graph signal generation
    if 'alpha' in simuParas.keys():
        alpha = simuParas['alpha']
    else:
        alpha = 1.0
        simuParas['alpha'] = alpha  # Default alpha value

    # Whether or not to normalize graph signal estimates
    if 'normalize_g_hat' in kwargs.keys():
        normalize_g_hat = kwargs['normalize_g_hat']
    else:
        normalize_g_hat = False  # Default is no normalization
    
    # Other simulation parameters like the number of classes, graph type, etc.
    # These parameters control how graphs and data are generated

    ## Model settings
    # Model-specific settings (loss function, evaluator, etc.)
    if 'modelSettings' in kwargs.keys():
        modelSettings = kwargs['modelSettings']
    else:
        modelSettings = {}

    # Default loss function is SLOGtools.myLoss
    if 'thisLoss' in modelSettings.keys():
        thisLoss = modelSettings['thisLoss']
    else:
        thisLoss = SLOGtools.myLoss

    # Default evaluator is SLOGevaluator.evaluate
    if 'thisEvaluator' in modelSettings.keys():
        thisEvaluator = modelSettings['thisEvaluator']
    else:
        thisEvaluator = SLOGevaluator.evaluate

    # Default objective function
    if 'thisObject' in modelSettings.keys():
        thisObject = modelSettings['thisObject']
    else:
        thisObject = SLOGobj.myFunction_slog_1

    model_name = 'SLOG-Net'  # Name of the model
    device = 'gpu'  # The device to run the model on
    optimAlg = 'ADAM'  # Optimizer algorithm
    learningRate = 0.001  # Learning rate for optimization
    beta1 = 0.9  # Beta1 parameter for Adam optimizer
    beta2 = 0.999  # Beta2 parameter for Adam optimizer

    ## Save directory setup
    # Defining directories to save results and trained models
    if location == 'home':
        print('Running test at ', location)
        saveDir_dropbox = r"C:\Users\Chang Ye\Dropbox\onlineResults\experiments"
    elif location == 'office':
        print('Running test at ', location)        
        saveDir_dropbox = '/Users/changye/Dropbox/onlineResults/experiments'
    else:
        saveDir_dropbox = r"C:\Users\Chang Ye\Dropbox\onlineResults\experiments"
        
    ## Generate modelSaveDir
    # Generate file paths for saving model-related data
    label = 'Best'  # Label to identify the best model
    saveDir = os.path.join(saveDir_dropbox, modelDir)
    gsoName = 'gso-' + graphType + '.npy'
    gsoDir = os.path.join(saveDir, gsoName)

    # Load graph data (Adjacency matrix)
    GA = np.load(gsoDir)
    d, An, eigenvalues, V = SLOGtools.get_eig_normalized_adj(GA)  # Normalize and get eigenvectors

    # Based on model number, choose the architecture version
    if model_number == 1:
        SLOG_net = SLOGarchi.GraphSLoG_v3(V, nNodes, q, K, thisObject)
    else:
        SLOG_net = SLOGarchi.GraphSLoG_v1(V, nNodes, C, K, thisObject)

    # Optimizer setup
    thisOptim = optim.Adam(SLOG_net.parameters(), lr=learningRate, betas=(beta1, beta2))

    # Trainer setup
    thisTrainer = SLOGtrainer.slog_Trainer

    # Create and load the trained model
    loadedModel = SLOGmodel.Model(SLOG_net, thisLoss, thisOptim, thisTrainer, thisEvaluator, device, model_name, None)
    loadedModel.load_from_dropBox(saveDir, label=label)

    # Start the test procedure
    result = {}  # Dictionary to store results
    re_x = np.zeros(N_realiz)  # To store reconstruction errors for x
    re_g = np.zeros(N_realiz)  # To store reconstruction errors for g

    for n_realiz in range(N_realiz):
        # Generate graph signals and noise
        X = SLOGtools.X_generate(nNodes, P, S)
        if filterType == 'g':
            g0 = SLOGtools.g_generate_gso(nNodes, alpha, eigenvalues, L)
        else:
            g0 = SLOGtools.h_generate_gso(nNodes, alpha, eigenvalues, L)

        # Convert to NumPy for compatibility with the model
        X = to_numpy(X)
        g0 = to_numpy(g0)
        V = to_numpy(V)

        # Normalize graph signal
        if normalize_g_hat:
            g0 = nNodes * g0 / np.sum(g0)
        else:
            g0 = C * g0 / np.sum(g0)

        h0 = 1. / g0  # Compute inverse of g0
        H = np.dot(V, np.dot(np.diag(h0), V.T))  # Compute H matrix

        # Add noise based on the selected noise type
        if noiseType == 'gaussion':
            noise = np.random.normal(0, 1, [nNodes, P])
            noise = noise / LA.norm(noise, 'fro') * LA.norm(X, 'fro')
        elif noiseType == 'uniform':
            noise = np.random.uniform(-1, 1, [nNodes, P])
            noise = noise / np.max(np.abs(noise)) * np.max(np.abs(X))
        else:
            noise = np.zeros([nNodes, P])  # No noise

        # Generate noisy observations Y
        Y = np.dot(H, X) + noiseLevel * noise
        Y_test = to_torch(Y)  # Convert to Torch tensor

        # Use the loaded model to make predictions
        x_hat, g_hat = loadedModel.archit(Y_test)
        g_hat = to_numpy(g_hat)

        # Normalize g_hat if needed
        if normalize_g_hat:
            g_hat = nNodes * g_hat / np.sum(g_hat)

        # Compute reconstruction errors
        Z = linalg.khatri_rao(np.dot(Y.T, V), V)
        x_recv = np.dot(Z, g_hat)
        X_recv = x_recv.reshape((P, nNodes)).T

        # Compute reconstruction error for x and g
        re_x_1 = LA.norm(X_recv - X, 'fro') / LA.norm(X, 'fro')
        re_g_1 = LA.norm(g0 - g_hat) / LA.norm(g0)
        re_x_2 = LA.norm(X_recv + X, 'fro') / LA.norm(X, 'fro')
        re_g_2 = LA.norm(g0 + g_hat) / LA.norm(g0)

        # Select the best reconstruction error (minimize)
        if re_g_1 > re_g_2:
            re_g[n_realiz] = re_g_2
            re_x[n_realiz] = re_x_2
        else:
            re_g[n_realiz] = re_g_1
            re_x[n_realiz] = re_x_1

    result['re_x'] = re_x  # Store results
    result['re_g'] = re_g

    return result  # Return the results dictionary


    
 