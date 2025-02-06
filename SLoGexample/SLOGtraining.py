# 2024/08/15~
# Chang Ye, cye7@ur.rochester.edu

"""
training.py Training Module

Trainer classes

Trainer: general trainer that just computes a loss over a training set and
    runs an evaluation on a validation test
TrainerSingleNode: trainer class that computes a loss over the training set and
    runs an evaluation on a validation set, but assuming that the architectures
    involved have a single node forward structure and that the data involved
    has a method for identifying the target nodes
TrainerFlocking: trainer class that computes a loss over the training set,
    suited for the problem of flocking (i.e. it involves specific uses of
    the data, like computing trajectories or using DAGger)
"""

import torch
import numpy as np
import os
import pickle
import datetime
from scipy import linalg
from timeit import default_timer as timer
import copy
from numpy import linalg as LA

# Import necessary tools from the SLoGexample module
from SLoGexample import SLOGtools as SLOGtools
from SLoGexample import SLOGobjective as SLOGobj
from SLoGexample import SLOGarchitectures as SLOGarchi
from SLoGexample import SLOGmodel as SLOGmodel
from SLoGexample import SLOGevaluation as SLOGevaluator
from SLoGexample import SLOGdata as SLOGdata

# Main Trainer class definition
class slog_Trainer:
    def __init__(self, model, data, nEpochs, batchsize, **kwargs):
        # Initialize and store model, data, and other relevant parameters
        batchSize = batchsize
        self.model = model
        self.data = data
        self.V = model.archit.V  # Architecture-specific V matrix
        nTrain = data.nTrain  # Size of the training set
        
        # Set up default values or fetch from kwargs
        self.validationInterval = kwargs.get('validationInterval', 20)
        self.trainMode = kwargs.get('trainMode', 'default')
        alpha = kwargs.get('alpha', 1.0)
        self.tMax = kwargs.get('tMax', data.nNodes)
        filterTrainType = kwargs.get('filterTrainType', 'g')
        self.noiseType = kwargs.get('noiseType', 'gaussian')
        self.noiseLevel = kwargs.get('noiseLevel', 0.0)

        # Store sizes and dimensions
        self.nTrain = data.nTrain
        self.nTest = data.nTest
        self.nValid = data.nValid
        self.nNodes = data.nNodes
        self.filterType = data.filterType
        self.L = data.L
        self.tMax = self.tMax
        self.nEpochs = nEpochs

        # Prepare data samples
        XValid = (data.samples)['valid']['X0']
        XValid = SLOGdata.to_numpy(XValid)
        xValid = torch.tensor(np.reshape(XValid, [self.nNodes * self.nValid], order='F'))
        XTrain = (data.samples)['train']['X0']

        YTest = (data.samples)['test']['signals']
        YTest = SLOGdata.to_torch(YTest)
        YTest = torch.reshape(torch.transpose(YTest, 0, 2), (self.nNodes, self.nTest))
        yTest = torch.reshape(YTest, (self.nNodes * self.nTest,))

        XTest = (data.samples)['test']['X0']
        XTest = SLOGdata.to_numpy(XTest)
        xTest = torch.tensor(np.reshape(XTest, [self.nNodes * self.nTest], order='F'))

        self.ZTest = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(YTest.numpy()), self.V), self.V), requires_grad=False)
        g_test = (data.samples)['test']['g_test']
        self.Phi = (data.samples)['train']['Phi']
        self.XTest = XTest
        self.xTest = xTest
        self.YTest = YTest
        self.yTest = yTest

        self.XTrain = XTrain
        self.XValid = XValid
        self.xValid = xValid

        #### Training setting
        # Compute nBatches
        optimizer_slog = model.optim
        if nTrain < batchSize:
            nBatches = 1
            batchSize = [nTrain]
        elif nTrain % batchSize != 0:
            nBatches = np.ceil(nTrain / batchSize).astype(np.int64)
            batchSize = [batchSize] * nBatches
            while sum(batchSize) != nTrain:
                batchSize[-1] -= 1
        else:
            nBatches = int(nTrain / batchSize)
            batchSize = [batchSize] * nBatches
        batchIndex = np.cumsum(batchSize).tolist()
        batchIndex = [0] + batchIndex
        self.batchIndex = batchIndex
        self.nBatches = nBatches
        
        # Generate filter type based on filterTrainType
        if filterTrainType == 'g':
            filterType = 'g'
        else:
            filterType = self.filterType
        
        # Generate batch of g values based on data and parameters
        self.g_batch = SLOGtools.g_batch_generate(self.nNodes, nBatches, alpha, filterType=filterType, Phi=self.Phi, L=self.L, tMax=self.tMax)
        self.sampledTimes = np.random.choice(self.tMax, size=nBatches)

        # Initialize various tracking variables
        self.training_time = np.zeros([nEpochs, nBatches])  # Store training times for each epoch/batch
        self.lossTrain = []  # Losses on the training set
        self.lossValid = []  # Losses on the validation set
        self.errorTrain = []  # Errors on the training set
        self.loss_re_train = []  # Re-training losses
        self.loss_re_all = []  # All losses (train, valid, etc.)
        
        self.bestModel = dict()

        # Random permutation for the training set indices
        randomPermutation = np.random.permutation(nTrain)
        self.idxEpoch = [int(i) for i in randomPermutation]
        print([len(self.idxEpoch), nTrain])

        print('Number of Batches:', nBatches)
        
    def train(self, **kwargs):
        epoch = 0
        V = self.V
        N = self.nNodes
        type_loss = 4
        optimizer_slog = self.model.optim
        
        # Fetch noise parameters from kwargs or use defaults
        noiseType = kwargs.get('noiseType', self.noiseType)
        noiseLevel = kwargs.get('noiseLevel', self.noiseLevel)
                        
        # Loop over epochs
        while epoch < self.nEpochs:
            # Shuffle training indices for each epoch
            randomPermutation = np.random.permutation(self.nTrain)
            idxEpoch = [int(i) for i in randomPermutation]
            print("")
            print("Epoch %d" % (epoch + 1))

            batch = 0 
    
            # Loop over batches
            while batch < self.nBatches:
                # Determine indices for the current batch
                thisBatchIndices = idxEpoch[self.batchIndex[batch]: self.batchIndex[batch + 1]]
                
                # Generate random g or use precomputed batch
                if self.trainMode == 'default':
                    g = self.g_batch[:, batch]
                    h = 1. / g
                    H = np.dot(V, np.dot(np.diag(h), np.transpose(V)))
                else:
                    t = self.sampledTimes[batch]
                    lambda_t = self.Phi[:, t].reshape(N)
                    g = 1. / lambda_t
                    g = g / np.sum(g)
                    h = 1. / g
                    H = np.dot(V, np.dot(np.diag(h), np.transpose(V)))
                
                # Get the training data for this batch
                XTrainBatch = SLOGtools.to_numpy(self.XTrain[:, thisBatchIndices])
                P = XTrainBatch.shape[1]
                
                # Add noise to the training batch
                if noiseType == 'gaussian':
                    noise = np.random.normal(0, 1, [self.nNodes, P])
                    noise = noise / LA.norm(noise, 'fro') * LA.norm(XTrainBatch, 'fro')
                elif noiseType == 'uniform':
                    noise = np.random.uniform(-1, 1, [self.nNodes, P])
                    noise = noise / np.max(np.abs(noise)) * np.max(np.abs(XTrainBatch))
                else:
                    noise = np.zeros([self.nNodes, P])
                
                # Create the noisy target values for the batch
                YTrainBatch = np.dot(H, XTrainBatch) + noiseLevel * noise
                YTrainBatch = torch.tensor(YTrainBatch)  

                # Prepare the Z matrix for this batch
                ZTrainBatch = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(YTrainBatch.numpy()), V), V), requires_grad=False)
                xTrainBatch = torch.tensor(np.reshape(SLOGdata.to_numpy(XTrainBatch), [N * np.size(thisBatchIndices)], order='F'))
        
                # Record training time start
                start_timer = timer()

                # Zero the model gradients
                self.model.archit.zero_grad()

                # Forward pass: compute the predictions
                xHatTrainBatch, vHatTrainBatch = self.model.archit(YTrainBatch)

                # Compute the loss
                lossValueTrain = self.model.loss(torch.matmul(ZTrainBatch, vHatTrainBatch), xTrainBatch)     
    
                # Backward pass: compute the gradients
                lossValueTrain.backward()

                # Perform an optimization step
                optimizer_slog.step()
        
                # Record training time end
                end_timer = timer()
                self.training_time[epoch, batch] = end_timer - start_timer
        
                # Apply parameter constraints for model v5
                for p in self.model.archit.eta_1:
                    p.data.clamp_(0)
                for p in self.model.archit.rho_1:
                    p.data.clamp_(0)  
                for p in self.model.archit.lmbd:
                    p.data.clamp_(0)  
            
                # Track the loss
                self.lossTrain += [lossValueTrain.item()]
        
                # Compute relative error
                x_p_batch = xHatTrainBatch.clone().detach().requires_grad_(False)
                g_p_pre = vHatTrainBatch.clone().detach().requires_grad_(False)
                G_hat = np.dot(V, np.dot(np.diag(g_p_pre), np.transpose(V)))
                X_p_pre = np.dot(G_hat, YTrainBatch) 
                x_p_pre = torch.tensor(np.reshape(X_p_pre, [N * np.size(thisBatchIndices)], order='F'))
                x_batch = xTrainBatch.clone().detach().requires_grad_(False)
                x_diff_batch = x_p_pre - x_batch
                RE, sign = SLOGtools.min_RE(x_p_pre, x_batch)
                relative_error_batch = RE
        
                self.errorTrain += [relative_error_batch]
                self.loss_re_train += [[lossValueTrain.item(), relative_error_batch]]
        
                # Print batch and loss information at intervals
                if (epoch * self.nBatches + batch) % self.validationInterval == 0:
                    P_valid = self.XValid.shape[1]
                    # Add noise to validation set
                    if noiseType == 'gaussian':
                        noise = np.random.normal(0, 1, [self.nNodes, P_valid])
                        noise = noise / LA.norm(noise, 'fro') * LA.norm(self.XValid, 'fro')
                    elif noiseType == 'uniform':
                        noise = np.random.uniform(-1, 1, [self.nNodes, P_valid])
                        noise = noise / np.max(np.abs(noise)) * np.max(np.abs(self.XValid))
                    else:
                        noise = np.zeros([self.nNodes, P_valid])

                    # Create noisy validation data
                    YValid = np.dot(H, self.XValid) + noiseLevel * noise                                  
                    YValid = torch.tensor(YValid) 
                    ZValidBatch = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(YValid.numpy()), V), V), requires_grad=False) 

                    with torch.no_grad():
                        # Obtain the predictions for validation and test sets
                        xHatValid, vHatValid = self.model.archit(YValid)
                        xHatTest, vHatTest = self.model.archit(self.YTest)
                      
                    # Compute validation and test loss
                    lossValueValid = self.model.loss(torch.matmul(ZValidBatch, vHatValid), self.xValid)    
                    lossValueTest = self.model.loss(torch.matmul(self.ZTest, vHatTest), self.xTest)    
                    self.lossValid += [lossValueValid.item()]
            
                    # Compute relative error for validation
                    g_valid_pre = vHatValid.clone().detach().requires_grad_(False)
                    G_valid = np.dot(V, np.dot(np.diag(g_valid_pre), np.transpose(V)))
                    X_valid_pre = np.dot(G_valid, YValid) 
                    x_valid_pre = torch.tensor(np.reshape(X_valid_pre, [N * self.nValid], order='F'))
                    x_batch = self.xValid.clone().detach().requires_grad_(False)
                    x_diff_valid = x_valid_pre - x_batch
                    RE, sign = SLOGtools.min_RE(x_valid_pre, x_batch)
                    relative_error_valid = RE
        
                    self.loss_re_all += [[lossValueTrain.item(), lossValueValid.item(), relative_error_batch, relative_error_valid]]
                    print("\t Loss: %6.4f [T]" % (lossValueTrain) + " %6.4f [V]" % (lossValueValid) + 
                          " \t RE: %6.4f [T]" % (relative_error_batch) + " \t RE: %6.4f [V]" % (relative_error_valid) + 
                          " \t Best Loss: %6.4f [V]" % (min(self.lossValid)) + " \t Loss: %6.4f [Test]" % (lossValueTest))
                    
                    # Save the best model so far
                    if len(self.lossValid) > 1:
                        if lossValueValid <= min(self.lossValid):
                            self.bestModel = copy.deepcopy(self.model.archit)  
                            self.model.save(label="Best")
                    else:
                        self.bestModel = copy.deepcopy(self.model.archit)
                        self.model.save(label="Best")
                    
                batch += 1
            
            # Save the final model after the epoch
            self.model.save(label='Last')
            print("Mean training time = ", np.mean(self.training_time[epoch, :]))

            epoch += 1
    
        print("")
        return_terms = {}
        return_terms['bestModel'] = self.bestModel
        return_terms['minLossValid'] = min(self.lossValid)
        return_terms['minLossTrain'] = min(self.lossTrain)        
        return return_terms
