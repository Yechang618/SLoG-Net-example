# 2024/08/15~
# Chang Ye, cye7@ur.rochester.edu

import numpy as np
from numpy import linalg as LA
import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import copy
from copy import deepcopy
import matplotlib.pyplot as plt
# import mymodule as myModules
import matplotlib.cm as cm
from scipy import linalg
from timeit import default_timer as timer
import networkx as nx
import os
import pickle
import datetime
#### Import SLOG packakes
from SLOGmodules import SLOGtools as SLOGtools
from SLOGmodules import SLOGobjective as SLOGobj
from SLOGmodules import SLOGarchitectures as SLOGarchi 
from SLOGmodules import SLOGtraining as SLOGtrainer
from SLOGmodules import SLOGmodel as SLOGmodel
from SLOGmodules import SLOGevaluation as SLOGevaluator
from SLOGmodules import SLOGdata as SLOGdata

#### Import GNN packages
from SLOGmodules import graphTools as graphTools
from SLOGmodules import dataTools as dataTools
from alegnn.utils import graphML as gml

from alegnn.modules import architectures as archit
from alegnn.modules import model as model
from alegnn.modules import training as training
from alegnn.modules import evaluation as evaluation
from alegnn.modules import loss as loss
from alegnn.utils.miscTools import writeVarValues
from alegnn.utils.miscTools import saveSeed

### Import trained models
from SLOGTrainedModels import trainedModels as trainedModels


def to_numpy(x):
    dataType = type(x) # get data type so that we don't have to convert
    if 'numpy' in repr(dataType):
        return x
    elif 'torch' in repr(dataType):
        x1 = x.clone().detach().requires_grad_(False)
        return x1.numpy()
    
def to_torch(x):
    dataType = type(x) # get data type so that we don't have to convert
    if 'numpy' in repr(dataType):
        return torch.tensor(x)
    elif 'torch' in repr(dataType):
        return x  
    
def noiseTest_dropbox(nNodes,P,S, modelDir, **kwargs):
    ## Assertation
    
    ## Parameters loading (kwargs)
    if 'location' in kwargs.keys():
        location = kwargs['location']
    else:
        location = 'office'

    if 'modelParas' in kwargs.keys():
        modelParas = kwargs['modelParas']
    else:
        modelParas = {}

    if 'q' in modelParas.keys():
        q = modelParas['q']
    else:
        q = 4
        
    if 'simuParas' in kwargs.keys():
        simuParas = kwargs['simuParas']
    else:
        simuParas = {}        
    
    if 'alpha' in simuParas.keys():
        alpha = simuParas['alpha']
    else:
        alpha = 1.0
        simuParas['alpha'] = alpha
        
    if 'normalize_g_hat' in kwargs.keys():
        normalize_g_hat = kwargs['normalize_g_hat']
    else:
        normalize_g_hat = False       
        
    if 'N_C' in simuParas.keys(): 
        # Number of sources per signal in classification
        N_C = simuParas['N_C']
    else:
        N_C = 3
        simuParas['N_C'] = N_C    
        
    if 'nClasses' in simuParas.keys(): 
        # Number of classes (i.e. number of communities) in classification
        nClasses = simuParas['nClasses']
    else:
        nClasses = 3
        simuParas['nClasses'] = nClasses    
   
    if 'graphType' in simuParas.keys(): 
        # Number of classes (i.e. number of communities) in classification
        graphType = simuParas['graphType']
    else:
        graphType = 'ER'
        simuParas['graphType'] = graphType 

    if 'graphOptions' in simuParas.keys():
        graphOptions = simuParas['graphOptions']
    else:
        graphOptions = {} # Dictionary of options to pass to the graphTools.createGraph function
        graphOptions['probIntra'] = 0.3 # Probability of drawing edges
        simuParas['graphOptions'] = graphOptions

    if 'L' in simuParas.keys():
        L = simuParas['L']
    else:
        L = 5
        simuParas['L'] = L
        
    if 'filterType' in simuParas.keys():
        filterType = simuParas['filterType']
    else:
        filterType = 'h'
        simuParas['filterType'] = filterType    
        
    if 'noiseLevel' in simuParas.keys():
        noiseLevel = simuParas['noiseLevel']
    else:
        noiseLevel = 0
        simuParas['noiseLevel'] = noiseLevel        

    if 'noiseType' in simuParas.keys():
        noiseType = simuParas['noiseType']
    else:
        noiseType = 'gaussion'
        simuParas['noiseType'] = noiseType        

    if 'C' in simuParas.keys():
        C = simuParas['C']
    else:
        C = nNodes
        simuParas['C'] = C        

    if 'K' in simuParas.keys():
        K = simuParas['K']
    else:
        K = 5
        simuParas['K'] = K        

    if 'N_realiz' in simuParas.keys():
        N_realiz = simuParas['N_realiz']
    else:
        N_realiz = 10
        simuParas['N_realiz'] = N_realiz  

    ## Model settings
    #
    if 'modelSettings' in kwargs.keys():
        modelSettings = kwargs['modelSettings']
    else:
        modelSettings = {}
        
    if 'model_number' in simuParas.keys():
        model_number = simuParas['model_number']
    else:
        model_number = 0        
        
    if 'thisLoss' in modelSettings.keys():
        thisLoss = modelSettings['thisLoss']
    else:
        thisLoss = SLOGtools.myLoss
        
    if 'thisEvaluator' in modelSettings.keys():
        thisEvaluator = modelSettings['thisEvaluator']
    else:
        thisEvaluator = SLOGevaluator.evaluate   
        
    if 'thisObject' in modelSettings.keys():
        thisObject = modelSettings['thisObject']
    else:
        thisObject = SLOGobj.myFunction_slog_1
 
    model_name = 'SLOG-Net'
    device = 'gpu'
    optimAlg = 'ADAM'
    learningRate = 0.001
    beta1 = 0.9
    beta2 = 0.999

    ## Save dir
    
    if location == 'home':
        print('Running test at ', location)
        saveDir_dropbox = r"C:\Users\Chang Ye\Dropbox\onlineResults\experiments"
    elif location == 'office':
        print('Running test at ', location)        
        saveDir_dropbox = '/Users/changye/Dropbox/onlineResults/experiments'
    else:
        saveDir_dropbox = r"C:\Users\Chang Ye\Dropbox\onlineResults\experiments"
        
    ## Generate modelSaveDir
    # modelDirList
    label = 'Best'
    saveDir = os.path.join(saveDir_dropbox,modelDir)
    gsoName = 'gso-' + graphType + '.npy'
    gsoDir = os.path.join(saveDir, gsoName)
    GA = np.load(gsoDir)
    d,An, eigenvalues, V = SLOGtools.get_eig_normalized_adj(GA)
    gso = An
    if model_number == 1:
        SLOG_net = SLOGarchi.GraphSLoG_v3(V,nNodes,q,K, thisObject)
    else:
        SLOG_net = SLOGarchi.GraphSLoG_v1(V,nNodes,C,K, thisObject)
    thisOptim = optim.Adam(SLOG_net.parameters(), lr = learningRate, betas = (beta1,beta2))
    thisTrainer = SLOGtrainer.slog_Trainer    
    loadedModel = SLOGmodel.Model(SLOG_net,thisLoss,thisOptim, thisTrainer,thisEvaluator, device, model_name,  None)
    loadedModel.load_from_dropBox(saveDir, label = label)        
    
    # Test begins
    result = {}
    re_x = np.zeros(N_realiz)
    re_g = np.zeros(N_realiz)    
    for n_realiz in range(N_realiz):
        X = SLOGtools.X_generate(nNodes,P,S)
        if filterType == 'g':
            g0 = SLOGtools.g_generate_gso(nNodes,alpha, eigenvalues,L)
        else:
            g0 = SLOGtools.h_generate_gso(nNodes,alpha, eigenvalues,L)
        X = to_numpy(X)
        g0 = to_numpy(g0)
        V = to_numpy(V)
        if normalize_g_hat:
            g0 = nNodes*g0/np.sum(g0)
        else:
            g0 = C*g0/np.sum(g0)
        h0 = 1./g0
        H = np.dot(V,np.dot(np.diag(h0),V.T))
        if noiseType == 'gaussion':
            noise = np.random.normal(0,1,[nNodes, P])
            noise = noise/LA.norm(noise,'fro')*LA.norm(X,'fro')
        elif noiseType == 'uniform':
            noise = np.random.uniform(-1,1,[nNodes, P])
            noise = noise/np.max(np.abs(noise))*np.max(np.abs(X))
        else:
            noise = np.zeros([nNodes, P])
        Y = np.dot(H,X) + noiseLevel*noise
        Y_test = to_torch(Y)
        x_hat, g_hat = loadedModel.archit(Y_test)
        g_hat = to_numpy(g_hat)  
        
        if normalize_g_hat:
            g_hat = nNodes*g_hat/np.sum(g_hat)      
        
        Z = linalg.khatri_rao(np.dot(Y.T,V),V)
#         print('Sum of g', np.sum(g_hat), np.sum(g0))
        x_recv = np.dot(Z,g_hat)
        X_recv = x_recv.reshape((P,nNodes)).T
        re_x_1 = LA.norm(X_recv - X,'fro')/LA.norm(X,'fro')
        re_g_1 = LA.norm(g0 - g_hat)/LA.norm(g0)
        re_x_2 = LA.norm(X_recv + X,'fro')/LA.norm(X,'fro')
        re_g_2 = LA.norm(g0 + g_hat)/LA.norm(g0) 
        if re_g_1 > re_g_2:
            re_g[n_realiz] = re_g_2
            re_x[n_realiz] = re_x_2
        else:
            re_g[n_realiz] = re_g_1
            re_x[n_realiz] = re_x_1               
    result['re_x'] = re_x    
    result['re_g'] = re_g     
    
    return result

    
 