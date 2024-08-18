# 2024/08/15~
# Chang Ye, cye7@ur.rochester.edu

import numpy as np
import torch
from SLoGexample import SLOGtools as SLOGtools
from numpy import linalg as LA

def to_torch(x, **kwargs):
    """
    Change data type to dtype.
    """
    thisShape = x.shape # get the shape
    dataType = type(x) # get data type so that we don't have to convert
    if 'requires_grad' in kwargs.keys():
        requires_grad = kwargs['requires_grad']
    else:
        requires_grad = False
    if 'numpy' in repr(dataType):
        return torch.tensor(x)
    return x

def to_numpy(x, **kwargs):
    """
    Change data type to dtype.
    """
    thisShape = x.shape # get the shape
    dataType = type(x) # get data type so that we don't have to convert
    if 'requires_grad' in kwargs.keys():
        requires_grad = kwargs['requires_grad']
    else:
        requires_grad = False
        
    if 'torch' in repr(dataType):
        if requires_grad == False:
            x1 = x.clone().detach().requires_grad_(False)
            return x1.numpy()
        else:
            return x.numpy()
    return x
            
def assertDType(x, dtype, **kwargs):
    """
    Change data type to dtype.
    """
    thisShape = x.shape # get the shape
    dataType = type(x) # get data type so that we don't have to convert
    if 'requires_grad' in kwargs.keys():
        requires_grad = kwargs['requires_grad']
    else:
        requires_grad = False
    if 'numpy' in repr(dataType):
        if dtype == 'torch':
            return torch.tensor(x)

    elif 'torch' in repr(dataType):
        if dtype == 'numpy':
            if requires_grad == False:
                x1 = x.clone().detach().requires_grad_(False)
                return x1.numpy()
            else:
                return x.numpy()
    return x

def normalizeData(x, ax):
    """
    normalizeData(x, ax): normalize data x (subtract mean and divide by standard 
    deviation) along the specified axis ax
    """
    
    thisShape = x.shape # get the shape
    assert ax < len(thisShape) # check that the axis that we want to normalize
        # is there
    dataType = type(x) # get data type so that we don't have to convert

    if 'numpy' in repr(dataType):

        # Compute the statistics
        xMean = np.mean(x, axis = ax)
        xDev = np.std(x, axis = ax)
        # Add back the dimension we just took out
        xMean = np.expand_dims(xMean, ax)
        xDev = np.expand_dims(xDev, ax)

    elif 'torch' in repr(dataType):

        # Compute the statistics
        xMean = torch.mean(x, dim = ax)
        xDev = torch.std(x, dim = ax)
        # Add back the dimension we just took out
        xMean = xMean.unsqueeze(ax)
        xDev = xDev.unsqueeze(ax)

    # Subtract mean and divide by standard deviation
    x = (x - xMean) / xDev

    return x

def invertTensorEW(x):
    
    # Elementwise inversion of a tensor where the 0 elements are kept as zero.
    # Warning: Creates a copy of the tensor
    xInv = x.copy() # Copy the matrix to invert
    # Replace zeros for ones.
    xInv[x < zeroTolerance] = 1. # Replace zeros for ones
    xInv = 1./xInv # Now we can invert safely
    xInv[x < zeroTolerance] = 0. # Put back the zeros
    
    return xInv
def changeDataType(x, dataType):
    """
    changeDataType(x, dataType): change the dataType of variable x into dataType
    """
    
    # So this is the thing: To change data type it depends on both, what dtype
    # the variable already is, and what dtype we want to make it.
    # Torch changes type by .type(), but numpy by .astype()
    # If we have already a torch defined, and we apply a torch.tensor() to it,
    # then there will be warnings because of gradient accounting.
    
    # All of these facts make changing types considerably cumbersome. So we
    # create a function that just changes type and handles all this issues
    # inside.
    
    # If we can't recognize the type, we just make everything numpy.
    
    # Check if the variable has an argument called 'dtype' so that we can now
    # what type of data type the variable is
    if 'dtype' in dir(x):
        varType = x.dtype
    
    # So, let's start assuming we want to convert to numpy
    if 'numpy' in repr(dataType):
        # Then, the variable con be torch, in which case we move it to cpu, to
        # numpy, and convert it to the right type.
        if 'torch' in repr(varType):
            x = x.cpu().numpy().astype(dataType)
        # Or it could be numpy, in which case we just use .astype
        elif 'numpy' in repr(type(x)):
            x = x.astype(dataType)
    # Now, we want to convert to torch
    elif 'torch' in repr(dataType):
        # If the variable is torch in itself
        if 'torch' in repr(varType):
            x = x.type(dataType)
        # But, if it's numpy
        elif 'numpy' in repr(type(x)):
            x = torch.tensor(x, dtype = dataType)
            
    # This only converts between numpy and torch. Any other thing is ignored
    return x

def test_sample_generate(nNodes, S, P, nTest, gso, L = 3, noiseLevel = 0, alpha = 1.0, filterType = 'h'):
    d_slog,An_slog, eigenvalues_slog, V_slog = SLOGtools.get_eig_normalized_adj(gso)
    g_batch = SLOGtools.h_batch_generate_gso(nNodes,nTest,alpha, eigenvalues_slog,L)
    XTest = np.zeros([nNodes,P,nTest])
    YTest = np.zeros([nNodes,P,nTest])
    for n_t in range(nTest):
        X0 = X_generate(nNodes,P,S)
        XTest[:,:,n_t] = X0
        gt = g_batch[:,n_t]
        ht = 1./gt
        Ht = np.dot(V,np.dot(np.diag(ht),np.transpose(V)))
        YTest[:,:,n_t] = np.dot(Ht,X0)
    result = {}
    result['XTest'] = XTest
    result['YTest'] = YTest
    result['g_batch'] = g_batch   
    return result
    
class super_data:
    def __init__(self):
        # Minimal set of attributes that all data classes should have
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = None
        self.samples['train']['targets'] = None
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = None
        self.samples['valid']['targets'] = None
        self.samples['test'] = {}
        self.samples['test']['signals'] = None
        self.samples['test']['targets'] = None    
        



class _data:
    # Internal supraclass from which all data sets will inherit.
    # There are certain methods that all Data classes must have:
    #   getSamples(), expandDims(), to() and astype().
    # To avoid coding this methods over and over again, we create a class from
    # which the data can inherit this basic methods.
    
    # All the signals are always assumed to be graph signals that are written
    #   nDataPoints (x nFeatures) x nNodes
    # If we have one feature, we have the expandDims() that adds a x1 so that
    # it can be readily processed by architectures/functions that always assume
    # a 3-dimensional signal.
    
    def __init__(self):
        # Minimal set of attributes that all data classes should have
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = None
        self.samples['train']['targets'] = None
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = None
        self.samples['valid']['targets'] = None
        self.samples['test'] = {}
        self.samples['test']['signals'] = None
        self.samples['test']['targets'] = None
        
    def getSamples(self, samplesType, *args):
        # samplesType: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['targets']
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # Select the corresponding samples
                xSelected = x[selectedIndices]
                y = y[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                xSelected = x[args[0]]
                # And assign the labels
                y = y[args[0]]
                
            # If we only selected a single element, then the nDataPoints dim
            # has been left out. So if we have less dimensions, we have to
            # put it back
            if len(xSelected.shape) < len(x.shape):
                if 'torch' in self.dataType:
                    x = xSelected.unsqueeze(0)
                else:
                    x = np.expand_dims(xSelected, axis = 0)
            else:
                x = xSelected

        return x, y
    
    def expandDims(self):
        
        # For each data set partition
        for key in self.samples.keys():
            # If there's something in them
            if self.samples[key]['signals'] is not None:
                # And if it has only two dimensions
                #   (shape: nDataPoints x nNodes)
                if len(self.samples[key]['signals'].shape) == 2:
                    # Then add a third dimension in between so that it ends
                    # up with shape
                    #   nDataPoints x 1 x nNodes
                    # and it respects the 3-dimensional format that is taken
                    # by many of the processing functions
                    if 'torch' in repr(self.dataType):
                        self.samples[key]['signals'] = \
                                       self.samples[key]['signals'].unsqueeze(1)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(
                                                   self.samples[key]['signals'],
                                                   axis = 1)
                elif len(self.samples[key]['signals'].shape) == 3:
                    if 'torch' in repr(self.dataType):
                        self.samples[key]['signals'] = \
                                       self.samples[key]['signals'].unsqueeze(2)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(
                                                   self.samples[key]['signals'],
                                                   axis = 2)
        
    def astype(self, dataType):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        
        # The labels could be integers as created from the dataset, so if they
        # are, we need to be sure they are integers also after conversion. 
        # To do this we need to match the desired dataType to its int 
        # counterpart. Typical examples are:
        #   numpy.float64 -> numpy.int64
        #   numpy.float32 -> numpy.int32
        #   torch.float64 -> torch.int64
        #   torch.float32 -> torch.int32
        
        targetType = str(self.samples['train']['targets'].dtype)
        if 'int' in targetType:
            if 'numpy' in repr(dataType):
                if '64' in targetType:
                    targetType = np.int64
                elif '32' in targetType:
                    targetType = np.int32
            elif 'torch' in repr(dataType):
                if '64' in targetType:
                    targetType = torch.int64
                elif '32' in targetType:
                    targetType = torch.int32
        else: # If there is no int, just stick with the given dataType
            targetType = dataType
        
        # Now that we have selected the dataType, and the corresponding
        # labelType, we can proceed to convert the data into the corresponding
        # type
        for key in self.samples.keys():
            print('key:',key)
            self.samples[key]['signals'] = changeDataType(
                                                   self.samples[key]['signals'],
                                                   dataType)
            self.samples[key]['targets'] = changeDataType(
                                                   self.samples[key]['targets'],
                                                   targetType)

        # Update attribute
        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        # This can only be done if they are torch tensors
        if 'torch' in repr(self.dataType):
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                      = self.samples[key][secondKey].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device
                
class _dataForClassification(_data):
    # Internal supraclass from which data classes inherit when they are used
    # for classification. This renders the .evaluate() method the same in all
    # cases (how many examples are incorrectly labeled) so justifies the use of
    # another internal class.
    
    def __init__(self):
        
        super().__init__()
    

    def evaluate(self, yHat, y, tol = 1e-9):
        """
        Return the accuracy (ratio of yHat = y)
        """
        N = len(y)
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
            yHat = torch.argmax(yHat, dim = 1)
            #   And compute the error
            totalErrors = torch.sum(torch.abs(yHat - y) > tol)
            errorRate = totalErrors.type(self.dataType)/N
        else:
            yHat = np.array(yHat)
            y = np.array(y)
            #   We compute the target label (hardmax)
            yHat = np.argmax(yHat, axis = 1)
            #   And compute the error
            totalErrors = np.sum(np.abs(yHat - y) > tol)
            errorRate = totalErrors.astype(self.dataType)/N
        #   And from that, compute the accuracy
        return errorRate
    
## General data
class SLOG_GeneralData(_dataForClassification):

    def __init__(self, G, nTrain, nValid, nTest,S, V,eigenvalues,**kwargs):
        # Initialize parent
        super().__init__()
        # store attributes

        
        if 'L' in kwargs.keys():
            L = kwargs['L']
        else:
            L = 1
            
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 1.0
        if 'tMax' in kwargs.keys():
            tMax = kwargs['tMax']
        else:
            tMax = 1.0            
            
        if 'filterType' in kwargs.keys():
            filterType = kwargs['filterType']
        else:
            filterType = 'g'    

        if 'noiseLevel' in kwargs.keys():
            noiseLevel = kwargs['noiseLevel']
        else:
            noiseLevel = 0
            
        if 'noiseType' in kwargs.keys():
            noiseType = kwargs['noiseType']
        else:
            noiseType = 'gaussion'            
           
        if 'dataType' in kwargs.keys():
            dataType = kwargs['dataType']
        else:
            dataType = np.float64 
            
        if 'device' in kwargs.keys():
            device = kwargs['device']
        else:
            device = 'cpu'
            
        print(dataType)    
        self.dataType = dataType
        self.device = device
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        self.nNodes = G.N
        self.tMax = tMax
        self.V = V
        self.L = L
        self.S = S
        self.filterType = filterType
        self.Lambda = np.diag(eigenvalues)
        Phi = np.vander(eigenvalues,increasing=True) # Vandermonde Matrix 
        self.Phi = Phi
        #\\\ Generate the samples
        if 'filterType' == 'h':
            g_test = SLOGtools.h_generate_gso(self.nNodes,alpha, eigenvalues,L)
#         elif 'filterType' == 'wt':
#             g_test = SLOGtools.wt_generate_gso(self.nNodes,alpha, eigenvalues,tMax)            
        else:
            g_test = SLOGtools.g_generate_gso(self.nNodes,alpha, eigenvalues,L)
        h_test = 1./g_test      
        X_train = SLOGtools.X_generate(self.nNodes,nTrain,S)
        X_valid = SLOGtools.X_generate(self.nNodes,nValid,S)      
        X_test = SLOGtools.X_generate(self.nNodes,nTest,S)  
          
        V = to_numpy(V)
        h_test = to_numpy(h_test)
        
        H = np.dot(V,np.dot(np.diag(h_test),V.T))
        if noiseType == 'gaussion':
            noise = np.random.normal(0, 1, [self.nNodes,nTest])
            noise = noise/LA.norm(noise,'fro')*LA.norm(X_test,'fro')
        elif noiseType == 'uniform':
            noise = np.random.uniform(-1, 1, [self.nNodes,nTest])
            noise = noise/np.max(np.abs(noise))*np.max(np.abs(X_test))
        else:
            noise = np.zeros([nNodes,P])
        

        Y_test = np.dot(H,X_test) + noiseLevel*noise

        # Split and save them
        self.samples['train']['X0'] = X_train
        self.samples['valid']['X0'] = X_valid       
        self.samples['test']['signals'] = Y_test
        self.samples['test']['X0'] = X_test
        self.samples['test']['g_test'] = g_test
        self.samples['train']['Phi'] = Phi
        self.samples['train']['noiseLevel'] = noiseLevel      