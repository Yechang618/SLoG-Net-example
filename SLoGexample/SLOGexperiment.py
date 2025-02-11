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
import matplotlib.cm as cm
from scipy import linalg
from timeit import default_timer as timer
import networkx as nx
import os
import pickle
import datetime
from sklearn.metrics import f1_score, accuracy_score
#### Import SLOG packakes
from SLoGexample import SLOGtools as SLOGtools
from SLoGexample import SLOGobjective as SLOGobj
from SLoGexample import SLOGarchitectures as SLOGarchi 
from SLoGexample import SLOGtraining as SLOGtrainer
from SLoGexample import SLOGmodel as SLOGmodel
from SLoGexample import SLOGevaluation as SLOGevaluator
from SLoGexample import SLOGdata as SLOGdata


class slog_experiments():
    def __init__(self, simuParas = None, 
                 graphOptions = None,  **kwargs):
        
        # Simulation parameters
        self.simuParas = simuParas
        self.nTrain = simuParas['nTrain']
        self.batchsize = simuParas['batchsize']
        self.nValid = simuParas['nValid']
        self.nTest = simuParas['nTest']
        self.L = simuParas['L']
        self.noiseLevel = simuParas['noiseLevel']
        self.noiseType = simuParas['noiseType']
        self.filterType = simuParas['filterType']
        self.signalMode = simuParas['signalMode']
        self.trainMode = simuParas['trainMode']
        self.filterMode = simuParas['filterMode']
        self.selectMode = simuParas['selectMode']
        self.nNodes = simuParas['nNodes']
        self.S = simuParas['S']
        self.graphType = simuParas['graphType']
        self.alpha = simuParas['alpha']
        self.nEpochs = simuParas['nEpochs']
                
        # Graph options
        self.graphOptions = graphOptions
        
        # Model parameters (optional)
        if 'modelParas' in kwargs.keys():
            self.modelParas = kwargs['modelParas']
            self.C = self.modelParas['C']
            self.K = self.modelParas['K']
            self.filterTrainType = self.modelParas['filterTrainType']
        else:
            self.C = self.nNodes
            self.K = 5
            self.filterTrainType = 'g'
            self.modelParas = {}
            self.modelParas['C']= self.C
            self.modelParas['K']= self.K
            self.modelParas['filterTrainType']= self.filterTrainType
        if 'q' in self.modelParas.keys():
            self.q = self.modelParas['q']
        else:
            self.q = 4
            
        # Experiment parameters (optional)
        if 'expParas' in kwargs.keys():
            self.expParas = kwargs['expParas']
            self.nRealiz = self.expParas['nRealiz']
        else:
            self.expParas = {}
            self.nRealiz = 1
            self.expParas['nRealiz'] = self.nRealiz
            

        self.thisFilename_SLOG = 'sourceLocSLOGNET'
        self.saveDirRoot = 'experiments' # Relative location where to save the file
        self.saveDir = os.path.join(self.saveDirRoot, self.thisFilename_SLOG) # Dir where to save all the results from each run

        self.saveSettings = {}
        self.saveSettings['thisFilename_SLOG'] = self.thisFilename_SLOG            
        self.saveSettings['saveDirRoot'] = self.saveDirRoot
        self.saveSettings['saveDir'] = self.saveDir

        self.experiment_results = []
        for i in range(self.nRealiz):
            result_i = self.run_single_experiment()
            self.experiment_results.append(result_i)
       
    def get_experiment_result(self):
        return self.experiment_results
            
    def run_single_experiment(self,**kwargs):    
        ## kwargs:

        #\\\ Create .txt to store the values of the setting parameters for easier
        # reference  when running multiple experiments
        today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # Append date and time of the run to the directory, to avoid several runs of
        # overwritting each other.
        saveDir = self.saveDir + '-' + self.graphType + '-' + today
        saveDirs = {}
        saveDirs['saveDir'] = saveDir

        # Create directory
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        # useGPU = True
        # if useGPU and torch.cuda.is_available():
        #     device = 'cuda:0'
        #     torch.cuda.empty_cache()
        # else:
        #     device = 'cpu'
        device = self.simuParas['device']
        # Notify:
        print("Device selected: %s" % device)   
  
        # Create the file where all the (hyper)parameters are results will be saved.
        varsFile = os.path.join(saveDir,'hyperparameters.txt')
        with open(varsFile, 'w+') as file:
            file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        #\\\ Save values:
        SLOGtools.writeVarValues(varsFile, {'nNodes': self.nNodes, 'graphType': self.graphType})
        SLOGtools.writeVarValues(varsFile, self.graphOptions)
        SLOGtools.writeVarValues(varsFile, self.simuParas)
        SLOGtools.writeVarValues(varsFile, self.modelParas)           
        SLOGtools.writeVarValues(varsFile, saveDirs)
        SLOGtools.writeVarValues(varsFile, {'nTrain_slog': self.nTrain,
                                  'nValid': self.nValid,
                                  'nTest': self.nTest,
                                  'useGPU': device})
        optimAlg = 'ADAM'   
        learningRate = 0.01 
        beta1 = 0.9 
        beta2 = 0.999
            
        ## Graph generation
        G = SLOGtools.Graph(self.graphType, self.nNodes, self.graphOptions, save_dir = saveDir)
        G.computeGFT()
        d,An, eigenvalues, V   = SLOGtools.get_eig_normalized_adj(G.A)
        
        ## Data generation
        data = SLOGdata.SLOG_GeneralData(G, self.nTrain, self.nValid, self.nTest, self.S, V, eigenvalues, L = self.L, alpha = self.alpha,filterType = self.filterType, noiseLevel = self.noiseLevel, noiseType = self.noiseType)
        data.expandDims()
        
        C = self.C
        K = self.K
        filterTrainType = self.filterTrainType #'g'
        thisLoss = SLOGtools.myLoss
        thisEvaluator = SLOGevaluator.evaluate
        
        thisObject = SLOGobj.myFunction_slog_3 # SLoG-Net with learnable constraint
        SLOG_net = SLOGarchi.GraphSLoG_v3(V,self.nNodes,self.q,self.K, thisObject)        

        model_name = 'SLOG-Net'

        thisOptim = optim.Adam(SLOG_net.parameters(), lr = learningRate, betas = (beta1,beta2))
        thisTrainer = SLOGtrainer.slog_Trainer

        myModel = SLOGmodel.Model(SLOG_net,thisLoss,thisOptim, thisTrainer,thisEvaluator,device, model_name,  saveDir)
        
        result_train = myModel.train(data,self.nEpochs, self.batchsize, validationInterval = 40,trainMode = self.trainMode, filterTrainType = self.filterTrainType) # model, data, nEpochs, batchSize
        
        best_model = result_train['bestModel']
        minLossValid = result_train['minLossValid']
        minLossTrain = result_train['minLossTrain']
          
        results = {}
        results['model'] = myModel
        results['training result'] = result_train
        results['Graph'] = G
        results['saveDir'] = saveDir
        
        return results
    
def test_local(nNodes, P, S, exp_result, **kwargs):
    """
    Function to test the SLOG-Net model locally using generated data.
    
    Parameters:
    nNodes : int
        Number of nodes in the graph.
    P : int
        Number of input signals.
    S : int
        Graph shift operator.
    exp_result : dict
        Dictionary containing experiment results, including the trained model and graph information.
    **kwargs : dict
        Optional parameters for model and simulation settings.
    """
    
    ## Extract model parameters from kwargs, or use default values
    if 'modelParas' in kwargs.keys():
        modelParas = kwargs['modelParas']
    else:
        modelParas = {}
    
    if 'q' in modelParas.keys():
        q = modelParas['q']
    else:
        q = 4
        
    ## Extract simulation parameters, or use default values
    if 'simuParas' in kwargs.keys():
        simuParas = kwargs['simuParas']
    else:
        simuParas = {}
    
    if 'alpha' in simuParas.keys():
        alpha = simuParas['alpha']
    else:
        alpha = 1.0
        simuParas['alpha'] = alpha
    
    graphType = 'ER'  # Default graph type is Erdos-Renyi
    
    ## Graph options setup
    if 'graphOptions' in simuParas.keys():
        graphOptions = simuParas['graphOptions']
    else:
        graphOptions = {}  # Dictionary of options for graph creation
        graphOptions['probIntra'] = 0.3  # Probability of drawing edges
        simuParas['graphOptions'] = graphOptions
    
    ## Define various simulation parameters if not provided
    if 'L' in simuParas.keys():
        L = simuParas['L']
    else:
        L = 5
        simuParas['L'] = L
    
    filterType = 'h'  # Default filter type
    simuParas['filterType'] = filterType    
    
    if 'noiseLevel' in simuParas.keys():
        noiseLevel = simuParas['noiseLevel']
    else:
        noiseLevel = 0
        simuParas['noiseLevel'] = noiseLevel        

    if 'noiseType' in simuParas.keys():
        noiseType = simuParas['noiseType']
    else:
        noiseType = 'gaussian'
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
    
    ## Extract or set model settings
    if 'modelSettings' in kwargs.keys():
        modelSettings = kwargs['modelSettings']
    else:
        modelSettings = {}
    
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
    
    if 'device' in simuParas.keys():
        device = simuParas['device']
    else:
        device = 'cpu'
        simuParas['device'] = device  

    ## Model and optimization settings
    model_name = 'SLOG-Net'
    optimAlg = 'ADAM'
    learningRate = 0.001
    beta1 = 0.9
    beta2 = 0.999

    ## Extract experiment results
    saveDir = exp_result['saveDir']
    G = exp_result['Graph']
    loadedModel = exp_result['model']
    GA = G.A
    d, An, eigenvalues, V = SLOGtools.get_eig_normalized_adj(GA)
    gso = An
    
    ## Initialize arrays to store results
    result = {}
    re_x = np.zeros(N_realiz)
    re_g = np.zeros(N_realiz)   
    acc_x = np.zeros(N_realiz)
    elapse = np.zeros(N_realiz) 
    
    ## Loop through realizations for testing
    for n_realiz in range(N_realiz):
        X = SLOGtools.X_generate(nNodes, P, S)  # Generate input data
        g0 = SLOGtools.h_generate_gso(nNodes, alpha, eigenvalues, L)  # Generate filter
        
        # Convert data to NumPy arrays
        X, g0, V = map(to_numpy, (X, g0, V))
        g0 = C * g0 / np.sum(g0)  # Normalize filter coefficients
        h0 = 1. / g0
        H = np.dot(V, np.dot(np.diag(h0), V.T))  # Compute filter matrix
        
        ## Generate noise based on noise type
        if noiseType == 'gaussian':
            noise = np.random.normal(0, 1, [nNodes, P])
            noise = noise / LA.norm(noise, 'fro') * LA.norm(X, 'fro')
        elif noiseType == 'uniform':
            noise = np.random.uniform(-1, 1, [nNodes, P])
            noise = noise / np.max(np.abs(noise)) * np.max(np.abs(X))
        else:
            noise = np.zeros([nNodes, P])
        
        Y = np.dot(H, X) + noiseLevel * noise  # Apply filter and add noise
        Y_test = to_torch(Y)  # Convert to torch tensor
        
        ## Perform model inference
        start_timer = timer()
        x_hat, g_hat = loadedModel.archit(Y_test)  # Get model predictions
        end_timer = timer()
        elapse[n_realiz] = end_timer - start_timer  # Measure execution time

        g_hat = to_numpy(g_hat)  # Convert to NumPy
        g_hat = C * g_hat / np.sum(g_hat)  # Normalize estimated filter
        
        ## Compute reconstruction error
        Z = linalg.khatri_rao(np.dot(Y.T, V), V)
        x_recv = np.dot(Z, g_hat)
        X_recv = x_recv.reshape((P, nNodes)).T
        
        re_x[n_realiz] = LA.norm(X_recv - X, 'fro') / LA.norm(X, 'fro')
        re_g[n_realiz] = LA.norm(g0 - g_hat) / LA.norm(g0)
        acc_x[n_realiz] = accuracy_score(X.reshape(nNodes * P) > 0.1, -X_recv.reshape(nNodes * P) > 0.1)
    
    ## Store and return results
    result = {'re_x': re_x, 're_g': re_g, 'acc_x': acc_x, 'elapse': elapse}
    return result

def test_admmCompare_local(nNodes, P, S, exp_result, **kwargs):
    ## Test: compare with ADMM with different P.

    if 'modelParas' in kwargs.keys():
        modelParas = kwargs['modelParas']
    else:
        modelParas = {}

    if 'visual' in kwargs.keys():
        visual = kwargs['visual']
    else:
        visual = False

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
           
    graphType = 'ER'

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
    supp_thres = 0.1

    ## Model settings
    if 'modelSettings' in kwargs.keys():
        modelSettings = kwargs['modelSettings']
    else:
        modelSettings = {}
              
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

    if 'device' in simuParas.keys():
        device = simuParas['device']
    else:
        device = 'cpu'
        simuParas['device'] = device  

    model_name = 'SLOG-Net'
    optimAlg = 'ADAM'
    learningRate, beta1, beta2 = 0.001,0.9, 0.999

    # modelDirList
    label = 'Best'
    saveDir = exp_result['saveDir']
    G = exp_result['Graph']
    loadedModel = exp_result['model']
    GA = G.A
    d,An, eigenvalues, V = SLOGtools.get_eig_normalized_adj(GA)
    gso = An
    # Test begins
    result = {}

    re_x_slog = np.zeros(N_realiz)
    re_g_slog = np.zeros(N_realiz)  
    acc_x_slog = np.zeros(N_realiz)
    re_x_admm   = np.zeros(N_realiz) 
    re_g_admm   = np.zeros(N_realiz) 
    acc_x_admm = np.zeros(N_realiz)       
    elapse_slog = np.zeros(N_realiz)
    elapse_admm = np.zeros(N_realiz)

    ## Visualization samples
    visual_X0 = np.zeros([nNodes, P, N_realiz])
    visual_Y = np.zeros([nNodes, P, N_realiz])
    visual_g0 = np.zeros([nNodes, N_realiz])
    visual_X_slog = np.zeros([nNodes, P, N_realiz])
    visual_g_slog = np.zeros([nNodes, N_realiz])
    visual_X_admm = np.zeros([nNodes, P, N_realiz])
    visual_g_admm = np.zeros([nNodes, N_realiz])
    
    for n_realiz in range(N_realiz):
        X = SLOGtools.X_generate(nNodes,P,S)
        g0 = SLOGtools.h_generate_gso(nNodes,alpha, eigenvalues,L)
        X = to_numpy(X)
        g0 = to_numpy(g0)
        V = to_numpy(V)
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

        start_timer = timer()
        x_hat, g_hat = loadedModel.archit(Y_test)     
        end_timer = timer()
        elapse_slog[n_realiz] = end_timer - start_timer

        g_hat = to_numpy(g_hat)  
        g_hat = C*g_hat/np.sum(g_hat)      
        Z = linalg.khatri_rao(np.dot(Y.T,V),V)

        re_g_1 = LA.norm(g0 - g_hat)/LA.norm(g0)
        re_g_2 = LA.norm(g0 + g_hat)/LA.norm(g0) 
        if re_g_1 > re_g_2:
            re_g_slog[n_realiz] = re_g_2
            g_hat = -g_hat
        else:
            re_g_slog[n_realiz]= re_g_1
        x_recv = np.dot(Z,g_hat)
        X_recv = x_recv.reshape((P,nNodes)).T
        re_x_slog[n_realiz] = LA.norm(X_recv - X,'fro')/LA.norm(X,'fro')
        acc_x_slog[n_realiz] = accuracy_score(X.reshape(nNodes*P)> supp_thres, -X_recv.reshape(nNodes*P)>supp_thres)  

        # ADMM solver
        rho_0 = torch.tensor(1).to(device,dtype = torch.float64)
        eta_0 = torch.tensor(1).to(device,dtype = torch.float64)    
        N_ite = 10000  
        max_re = 1e-6
        Ct = torch.tensor(C).to(device,dtype = torch.float64) 

        start_timer = timer()
        x_hat_admm,g_hat_admm,n_ite,max_re_matched = admm_solver(Y_test, V,rho_0,eta_0,Ct,N_ite,max_re = max_re,device = device)
        end_timer = timer()
        elapse_admm[n_realiz] = end_timer - start_timer

        x_hat_admm, g_hat_admm = x_hat_admm.cpu().numpy(), g_hat_admm.cpu().numpy()

        g_hat_admm = C*g_hat_admm/np.sum(g_hat_admm)             
        re_g_admm_1 = LA.norm(g0 - g_hat_admm)/LA.norm(g0)
        re_g_admm_2 = LA.norm(g0 + g_hat_admm)/LA.norm(g0) 
        if re_g_admm_1 > re_g_admm_2:
            g_hat_admm = -g_hat_admm
            re_g_admm[n_realiz] = re_g_admm_2          
        else:
            re_g_admm[n_realiz] = re_g_admm_1
        
        x_recv_admm = np.dot(Z,g_hat_admm)
        X_recv_admm = x_recv_admm.reshape((P,nNodes)).T
        re_x_admm[n_realiz] = LA.norm(X_recv_admm - X,'fro')/LA.norm(X,'fro')
        acc_x_admm[n_realiz] = accuracy_score(np.abs(X.reshape(nNodes*P))> supp_thres, np.abs(X_recv_admm.reshape(nNodes*P))>supp_thres)

        # Recording visualization
        if visual:
          visual_X0[:,:,n_realiz] = X
          visual_Y[:,:,n_realiz] = Y
          visual_g0[:,n_realiz] = g0
          visual_X_slog[:,:,n_realiz] = X_recv
          visual_g_slog[:,n_realiz] = g_hat
          visual_X_admm[:,:,n_realiz] = X_recv_admm
          visual_g_admm[:,n_realiz] = g_hat_admm
        print('SLOG-net: ',re_x_slog[n_realiz],re_g_slog[n_realiz],acc_x_slog[n_realiz],', run time:',elapse_slog[n_realiz])
        print('ADMM: ',re_x_admm[n_realiz],re_g_admm[n_realiz],acc_x_admm[n_realiz],', run time:',elapse_admm[n_realiz])
        if not np.isnan(re_x_admm[n_realiz]):
          n_realiz += 1                        
    result = {'re_x_slog': re_x_slog,  # Relative error of X for SLoG-Net
          're_g_slog': re_g_slog,  # Relative error of g for SLoG-Net
          'acc_x_slog': acc_x_slog, # Accuracy of support estimate for SLoG-Net
          'elapse_slog': elapse_slog, # Elapse time for SLoG-Net
          're_x_admm': re_x_admm,  # Relative error of X for ADMM slover
          're_g_admm': re_g_admm,  # Relative error of g for ADMM slover
          'acc_x_admm': acc_x_admm,   # Accuracy of support estimate for ADMM slover
          'elapse_admm': elapse_admm, # Elapse time for ADMM slover
          'visual_X0': visual_X0, 
          'visual_Y': visual_Y,
          'visual_g0': visual_g0,
          'visual_X_slog':visual_X_slog,
          'visual_g_slog':visual_g_slog,
          'visual_X_admm':visual_X_admm,
          'visual_g_admm':visual_g_admm                      
    }  
    return result

############### Functions ############# 
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
    
############### ADMM solver ###########
def admm_solver(Y,V,rho_0,eta_0,C,N_ite,max_re = 1e-6, device = 'cpu'):
    [N,P] = Y.shape
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V)).to(device,dtype = torch.float64)    
    V = torch.from_numpy(V).double()
    eta = torch.tensor(np.random.rand(1)).to(device,dtype = torch.float64)    
    g = torch.tensor(np.random.rand(N)).to(device,dtype = torch.float64)
    x = torch.tensor(np.random.rand(N*P)).to(device,dtype = torch.float64)
    u = torch.tensor(np.random.rand(N*P)).to(device,dtype = torch.float64)
    II = torch.ones(N,N).to(device,dtype = torch.float64)
    In = torch.ones(N,1).to(device,dtype = torch.float64) # ones(N,1)
    n_ite, max_re_matched = 0, 0
    C = N
    ZIk_inv = torch.inverse(rho_0 * Z.T @ Z + eta_0 * torch.ones(N,N,dtype=torch.double))
    while n_ite < N_ite and max_re_matched ==0 :
        g_old, x_old = g, x
        g = ZIk_inv @ (Z.T @ (rho_0*x-u) + (eta_0*C - eta)*torch.ones(N,dtype=torch.double))
        x = Z @ g + u/rho_0
        x = torch.sign(x)*torch.maximum(torch.abs(x)-1/rho_0, torch.zeros(N*P,dtype=torch.double))
        u = u + rho_0*(Z @ g - x)
        eta = eta + eta_0*(In.T @ g - C)
        re = torch.norm(g-g_old)/(1e-10+torch.norm(g_old))
        if re < max_re**2:
            max_re_matched = 1
            break
        n_ite += 1
    return x,g,n_ite,max_re_matched