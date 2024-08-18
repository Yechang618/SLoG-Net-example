#
#
#

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
#         nNodes = G.N
        self.V = V
        self.L = L
        self.S = S
        self.filterType = filterType
#         N_C = len(sourceNodes[0])
#         self.N_C = N_C        
#         C = len(sourceNodes)
#         self.C = C
        self.Lambda = np.diag(eigenvalues)
        Phi = np.vander(eigenvalues,increasing=True) # Vandermonde Matrix 
        self.Phi = Phi
        #\\\ Generate the samples
        if 'filterType' == 'h':
            g_test = SLOGtools.h_generate_gso(self.nNodes,alpha, eigenvalues,L)
        elif 'filterType' == 'wt':
            g_test = SLOGtools.wt_generate_gso(self.nNodes,alpha, eigenvalues,tMax)            
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
        # Change data to specified type and device
#         self.astype(self.dataType)
#         self.to(self.device)
        
    def get_test_Samples(self):
        return None
    
    def get_valid_Samples(self,g_test):
        return None   
    
    def generate_new_test_Samples(self, g_test):
        return None

    def identify_class_from_topN_signals(self, X_Hat, topN = 3):
        return None
            
        
    def evaluate_slog(self, x_Hat, targets, topN = 1, tol = 1e-9):
        return None


## Classification data    
class SLOG_ClassificationData(_dataForClassification):
    """
    SourceLocalization: Creates the dataset for a source localization problem

    Initialization:

    Input:
        G (class): Graph on which to diffuse the process, needs an attribute
            .N with the number of nodes (int) and attribute .W with the
            adjacency matrix (np.array)
        nTrain (int): number of training samples
        nValid (int): number of validation samples
        nTest (int): number of testing samples
        sourceNodes (list of int): list of indices of nodes to be used as
            sources of the diffusion process
        tMax (int): maximum diffusion time, if None, the maximum diffusion time
            is the size of the graph (default: None)
        dataType (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved.

    Methods:

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label
                
    .expandDims(): Adds the feature dimension to the graph signals (i.e. for
        graph signals of shape nSamples x nNodes, turns them into shape
        nSamples x 1 x nNodes, so that they can be handled by general graph
        signal processing techniques that take into account a feature dimension
        by default)

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    errorRate = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): unnormalized probability of each label (shape:
                nDataPoints x nClasses)
            y (dtype.array): correct labels (1-D binary vector, shape:
                nDataPoints)
            tol (float, default = 1e-9): numerical tolerance to consider two
                numbers to be equal
        Output:
            errorRate (float): proportion of incorrect labels

    """

    def __init__(self, G, nTrain, nValid, nTest, sourceNodes, communityList, communityLabels, V,eigenvalues,**kwargs):
        # Initialize parent
        super().__init__()
        # store attributes

        
        if 'L' in kwargs.keys():
            L = kwargs['L']
        else:
            L = 1

        if 'tMax' in kwargs.keys():
            tMax = kwargs['tMax']
        else:
            tMax = G.N
            
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 1.0
            
        if 'filterMode' in kwargs.keys():
            filterMode = kwargs['filterMode']
        else:
            filterMode = 'default'
            
        if 'selectMode' in kwargs.keys():
            selectMode = kwargs['selectMode']
        else:
            selectMode = 'default'
            
        if 'signalMode' in kwargs.keys():
            signalMode = kwargs['signalMode']
        else:
            signalMode = 'default'
            
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
        self.tMax = tMax
        self.nNodes = G.N
        self.V = V
        self.L = L
        self.filterType = filterType
        N_C = len(sourceNodes[0])
        self.N_C = N_C        
        C = len(sourceNodes)
        print('Number of input nodes per classes:',N_C,'; number of classes',C)
        self.C = C
        self.Lambda = np.diag(eigenvalues)
        Phi = np.vander(eigenvalues,increasing=True) # Vandermonde Matrix 
        self.Phi = Phi
        #\\\ Generate the samples
        
        # total number of samples
        nTotal = nTrain + nValid + nTest
        # sample source nodes
        
        # Let's generate slog input x
        # Random sample communities method 1
        sourceNode_set = np.array(sourceNodes)
        communityNode_set = np.array(communityList)
        sampledSources_index = np.random.choice(np.arange(len(sourceNodes)), size = nTotal)

        sampledSources = sourceNode_set[sampledSources_index]
        sampledCommunities  = communityNode_set[sampledSources_index]   
        self.sourceNodes = sourceNodes
        self.communityList = communityList
        self.communityLabels = communityLabels
        
        x_slog = torch.zeros(nTotal,G.N)
        X0 = torch.zeros(G.N,nTotal)
        N = G.N
#         gT = np.ones([N,nTotal])+alpha*np.random.normal(0, 1, [N,nTotal]) 
        gT = SLOGtools.g_batch_generate(N, nTotal, alpha, filterType = self.filterType, Phi = self.Phi, L = self.L, tMax = self.tMax)
        sampledTimes = np.random.choice(tMax, size = nTotal)
#         g_test = np.ones(N)+alpha*np.random.normal(0, 1, N) 
        g_test = gT[:, nTrain + nValid]
        sampledTimes_test = np.random.choice(tMax,1)
        for n_t in range(nTest):
            gT[:,nTrain + nValid + n_t] = g_test
            sampledTimes[nTrain + nValid + n_t] = sampledTimes_test
        
        if selectMode == 'default':
            # Default means fixxing the source nodes
            for n_t in range(nTotal):
                x0 = torch.zeros(G.N)
                for n_c in range(N_C):
                    if signalMode == 'default':
                        x0[sampledSources[n_t,n_c]] = 1
                    else:
                        x0[sampledSources[n_t,n_c]] = np.random.randn()
                X0[:,n_t] = x0
                if filterMode == 'default':
                    gt = gT[:,n_t]
                    ht = 1./gt
                    Ht = torch.tensor(np.dot(V,np.dot(np.diag(ht),np.transpose(V))))
                else:
                    t = sampledTimes[n_t]
                    lambda_t = self.Phi[:,t].reshape(N)
                    Ht = torch.tensor(np.dot(V,np.dot(np.diag(lambda_t),np.transpose(V))))
                x_slog[n_t,:] = torch.matmul(Ht,x0)
        else:
            for n_t in range(nTotal):
                x0 = torch.zeros(G.N)
                sampledCommunities_t = sampledCommunities[n_t]
                np.random.shuffle(sampledCommunities_t)
                for n_c in range(N_C):
                    if signalMode == 'default':
                        x0[sampledCommunities_t[n_c]] = 1
                    else:
                        x0[sampledCommunities_t[n_c]] = np.random.randn()
                X0[:,n_t] = x0
#                 gt = gT[:,n_t]
#                 ht = 1./gt
#                 Ht = torch.tensor(np.dot(V,np.dot(np.diag(ht),np.transpose(V))))
                if filterMode == 'default':
                    gt = gT[:,n_t]
                    ht = 1./gt
                    Ht = torch.tensor(np.dot(V,np.dot(np.diag(ht),np.transpose(V))))
                else:
                    t = sampledTimes[n_t]
                    lambda_t = self.Phi[:,t].reshape(N)
                    Ht = torch.tensor(np.dot(V,np.dot(np.diag(lambda_t),np.transpose(V))))
                x_slog[n_t,:] = torch.matmul(Ht,x0)      
        # Now, we have the signals and the labels
        if noiseType == 'gaussion':
            noise = np.random.normal(0, 1, [nTotal,G.N])        
            x_slog_np = to_numpy(x_slog)
            noise = noise/LA.norm(noise,'fro')*LA.norm(x_slog_np,'fro')            
        elif noiseType == 'uniform':
            noise = np.random.uniform(-1, 1, [nTotal,G.N])        
            x_slog_np = to_numpy(x_slog)
            noise = noise/np.max(np.abs(noise))*np.max(np.abs(x_slog_np))          
        else:
            noise = np.zeros([nTotal,G.N]) 
        signals = torch.tensor(to_numpy(x_slog) + noiseLevel*noise)# nTotal x N      
        print('Signal shape:',signals.shape)
        # Finally, we have to match the source nodes to the corresponding labels
        # which start at 0 and increase in integers.

        labels = sampledSources_index
        # Split and save them
        self.samples['train']['signals'] = signals[0:nTrain, :]
        self.samples['train']['targets'] = np.array(labels[0:nTrain])
        self.samples['train']['X0'] = X0[:,0:nTrain]
        self.samples['train']['gT'] = gT[:,0:nTrain]        
        self.samples['valid']['signals'] = signals[nTrain:nTrain+nValid, :]
        self.samples['valid']['targets'] =np.array(labels[nTrain:nTrain+nValid])
        self.samples['valid']['X0'] = X0[:,nTrain:nTrain+nValid]
        self.samples['valid']['gT'] = gT[:,nTrain:nTrain+nValid]        
        self.samples['test']['signals'] = signals[nTrain+nValid:nTotal, :]
        self.samples['test']['targets'] =np.array(labels[nTrain+nValid:nTotal])
        self.samples['test']['X0'] = X0[:,nTrain+nValid:nTotal]
        self.samples['test']['gT'] = gT[:,nTrain+nValid:nTotal]  
        self.samples['test']['g_test'] = g_test
        self.samples['test']['sampleTimes'] = sampledTimes_test
        self.samples['train']['Phi'] = Phi
        self.samples['train']['noiseLevel'] = noiseLevel      
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
        
    def get_test_Samples(self):
        Y = self.samples['test']['signals']
#         print(Y)
        Y = torch.tensor(Y)
        print(Y.shape)
        Y = torch.reshape(torch.transpose(Y, 0, 1),(self.nNodes,self.nTest))
#         Y = torch.reshape(torch.transpose(Y, 0, 2),(self.nNodes,self.nTest))        
        targets = self.samples['test']['targets']
        X = self.samples['test']['X0']
        print('Sample shape Y:', Y.shape, ', X:', X.shape, ', targets:', targets.shape)
        g_test = self.samples['test']['g_test']
        return Y,X,targets,g_test
    
    def get_valid_Samples(self,g_test):
#         Y = self.samples['valid']['signals']
#         Y = torch.reshape(torch.transpose(Y, 0, 2),(self.nNodes,self.nTest))
        V = self.V
        h_test = 1./g_test
        Ht = torch.tensor(np.dot(V,np.dot(np.diag(h_test),np.transpose(V))))
        targets = self.samples['valid']['targets']
        X = self.samples['valid']['X0']
        Y = torch.matmul(Ht,X)  
        print('Sample shape Y:', Y.shape, ', X:', X.shape, ', targets:', targets.shape)
#         g_test = self.samples['test']['g_test']
        return Y,X,targets    
    
    def generate_new_test_Samples(self, g_test):
#         gT_test = self.samples['test']['gT']
#         targets_test = self.samples['test']['targets']
#         X_test = self.samples['test']['X0']
        Y_test = torch.zeros(self.nTest,self.nNodes)
        V = self.V
        h_test = 1./g_test
        Ht = torch.tensor(np.dot(V,np.dot(np.diag(h_test),np.transpose(V))))
        for n_t in range(self.nTest):
            x0 = self.samples['test']['X0'][:,n_t]
#             X0 = 
            self.samples['test']['gT'][:,n_t] = g_test
#             print(Gt.shape, x0.shape, g_test.shape)
            Y_test[n_t,:] = torch.matmul(Ht,x0) 
        self.samples['test']['signal'] = Y_test
        print('Re-generate sample shape Y:', self.samples['test']['signal'].shape)
#         Y = self.samples['test']['signal']
        return Y_test

    def identify_class_from_topN_signals(self, X_Hat, topN = 3):
        # Input H_Hat should be tensor
        # self.communityLabels should be list
        nNodes, nSamples = X_Hat.shape
        node2class = self.communityLabels
        target_Hat = []
        for n_sample in range(nSamples):
            # Top-N absolute value of X_Hat[:,n_sample]
            x_ns = torch.abs(torch.reshape(X_Hat[:,n_sample],(nNodes,)))
            signal_indexSorted = torch.argsort(x_ns)
            # Convert node indices to class numbers
            class_indexSorted = node2class[signal_indexSorted]
            class_topN = torch.tensor(class_indexSorted[-topN:])
            class_identified = torch.mode(class_topN).values.numpy()
            target_Hat.append(class_identified.item())
        return target_Hat
            
        
    def evaluate_slog(self, x_Hat, targets, topN = 1, tol = 1e-9):
        if topN == None:
            topN = 1
        # self.communityLabels[]: maps Node number to class
        N = len(targets)
        X_Hat = torch.transpose(torch.reshape(x_Hat, (self.nTest,self.nNodes)), 0, 1)    
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
#             X_Hat = torch.argmax(torch.abs(X_Hat), dim = 0)
            targetHat = self.identify_class_from_topN_signals(X_Hat, topN)
#             print(targetHat)
            assert len(targetHat) == self.nTest
            targetHat = torch.tensor(targetHat)
#             target_pre = []
#             for n in range(self.nTest):
#                 target_pre.append(self.communityLabels[X_Hat[n]])
#             print(target_pre)
#             print(X_Hat)
#             print(targets)
            #   And compute the error
#             targetHat = torch.tensor(target_pre)
            totalErrors = torch.sum(torch.abs(targetHat - targets) > tol)
            errorRate = totalErrors.type(self.dataType)/N
        else:
            X_Hat = np.array(X_Hat)
            targets = np.array(targets)
            #   We compute the target label (hardmax)
            X_Hat = np.argmax(X_Hat, axis = 0)
            #   And compute the error
            
#             print(X_Hat.shape)
            assert X_Hat.shape[0] == self.nTest
            target_pre = self.communityList[X_Hat]
#             print(target_pre)
            
            totalErrors = np.sum(np.abs(X_Hat - targets) > tol)
#             errorRate = totalErrors.astype(self.dataType)/N
        #   And from that, compute the accuracy
#         errorRate = None
        return errorRate
        
class SLOG_ClassificationData_v2(_dataForClassification):
    """
    SourceLocalization: Creates the dataset for a source localization problem

    Initialization:

    Input:
        G (class): Graph on which to diffuse the process, needs an attribute
            .N with the number of nodes (int) and attribute .W with the
            adjacency matrix (np.array)
        nTrain (int): number of training samples
        nValid (int): number of validation samples
        nTest (int): number of testing samples
        sourceNodes (list of int): list of indices of nodes to be used as
            sources of the diffusion process
        tMax (int): maximum diffusion time, if None, the maximum diffusion time
            is the size of the graph (default: None)
        dataType (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved.

    Methods:

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label
                
    .expandDims(): Adds the feature dimension to the graph signals (i.e. for
        graph signals of shape nSamples x nNodes, turns them into shape
        nSamples x 1 x nNodes, so that they can be handled by general graph
        signal processing techniques that take into account a feature dimension
        by default)

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    errorRate = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): unnormalized probability of each label (shape:
                nDataPoints x nClasses)
            y (dtype.array): correct labels (1-D binary vector, shape:
                nDataPoints)
            tol (float, default = 1e-9): numerical tolerance to consider two
                numbers to be equal
        Output:
            errorRate (float): proportion of incorrect labels

    """

    def __init__(self, gso, nTrain, nValid, nTest, sourceNodes, communityList, communityLabels, V,eigenvalues,**kwargs):
        # Initialize parent
        super().__init__()
        # store attributes

        N = gso.shape[0]
        if 'L' in kwargs.keys():
            L = kwargs['L']
        else:
            L = 1

        if 'tMax' in kwargs.keys():
            tMax = kwargs['tMax']
        else:
            tMax = N
            
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 1.0
            
        if 'filterMode' in kwargs.keys():
            filterMode = kwargs['filterMode']
        else:
            filterMode = 'default'
            
        if 'selectMode' in kwargs.keys():
            selectMode = kwargs['selectMode']
        else:
            selectMode = 'default'
            
        if 'signalMode' in kwargs.keys():
            signalMode = kwargs['signalMode']
        else:
            signalMode = 'default'
            
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
        self.nNodes = N
        self.tMax = tMax
        self.V = V
        self.L = L
        self.filterType = filterType
        self.communityLabels = communityLabels
        N_C = len(sourceNodes[0])
        self.N_C = N_C       
        C = len(sourceNodes)
        print('Number of input nodes per classes:',N_C,'; number of classes',C)
        self.C = C
        self.Lambda = np.diag(eigenvalues)
        Phi = np.vander(eigenvalues,increasing=True) # Vandermonde Matrix 
        self.Phi = Phi
        #\\\ Generate the samples
        
        # total number of samples
        nTotal = nTrain + nValid + nTest
        # sample source nodes
        
        # Let's generate slog input x
        temp_result = SLOGtools.X_generate_fromSBM(N,nTotal,N_C,communityLabels,gso,**kwargs)
        X0 = temp_result['X0']
        sampledIndicesList = temp_result['sampledIndicesList']
        # Random sample communities method 1
        x_slog = torch.zeros(nTotal,N)
        print('Classification data v2, filterType = ', self.filterType)
        gT = SLOGtools.g_batch_generate(N, nTotal, alpha, filterType = self.filterType, Phi = self.Phi, L = self.L,tMax = self.tMax)
        sampledTimes = np.random.choice(tMax, size = nTotal)
        g_test = gT[:, nTrain + nValid]
        sampledTimes_test = np.random.choice(tMax,1)
        for n_t in range(nTest):
            gT[:,nTrain + nValid + n_t] = g_test
            sampledTimes[nTrain + nValid + n_t] = sampledTimes_test       
        for n_t in range(nTotal):
            x0 = X0[:,n_t]
            if filterMode == 'default':
                gt = gT[:,n_t]
                ht = 1./gt
                Ht = torch.tensor(np.dot(V,np.dot(np.diag(ht),np.transpose(V))))
            else:
                t = sampledTimes[n_t]
                lambda_t = self.Phi[:,t].reshape(N)
                Ht = torch.tensor(np.dot(V,np.dot(np.diag(lambda_t),np.transpose(V))))
                
            x_slog[n_t,:] = torch.matmul(Ht,to_torch(x0))   
        # Now, we have the signals and the labels
        if noiseType == 'gaussion':
            noise = np.random.normal(0, 1, [nTotal,N])        
            x_slog_np = to_numpy(x_slog)
            noise = noise/LA.norm(noise,'fro')*LA.norm(x_slog_np,'fro')            
        elif noiseType == 'uniform':
            noise = np.random.uniform(-1, 1, [nTotal,N])        
            x_slog_np = to_numpy(x_slog)
            noise = noise/np.max(np.abs(noise))*np.max(np.abs(x_slog_np))          
        else:
            noise = np.zeros([nTotal,G.N]) 
        signals = torch.tensor(to_numpy(x_slog) + noiseLevel*noise)# nTotal x N      
        print('Signal shape:',signals.shape)
        # Finally, we have to match the source nodes to the corresponding labels
        # which start at 0 and increase in integers.

        labels = sampledIndicesList
        # Split and save them
        self.samples['train']['signals'] = signals[0:nTrain, :]
        self.samples['train']['targets'] = np.array(labels[0:nTrain])
        self.samples['train']['X0'] = X0[:,0:nTrain]
        self.samples['train']['gT'] = gT[:,0:nTrain]        
        self.samples['valid']['signals'] = signals[nTrain:nTrain+nValid, :]
        self.samples['valid']['targets'] =np.array(labels[nTrain:nTrain+nValid])
        self.samples['valid']['X0'] = X0[:,nTrain:nTrain+nValid]
        self.samples['valid']['gT'] = gT[:,nTrain:nTrain+nValid]        
        self.samples['test']['signals'] = signals[nTrain+nValid:nTotal, :]
        self.samples['test']['targets'] =np.array(labels[nTrain+nValid:nTotal])
        self.samples['test']['X0'] = X0[:,nTrain+nValid:nTotal]
        self.samples['test']['gT'] = gT[:,nTrain+nValid:nTotal]  
        self.samples['test']['g_test'] = g_test
        self.samples['test']['sampleTimes'] = sampledTimes_test
        self.samples['train']['Phi'] = Phi
        self.samples['train']['noiseLevel'] = noiseLevel      
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
        
    def get_test_Samples(self):
        Y = self.samples['test']['signals']
        Y = torch.tensor(Y)
        print(Y.shape)
        Y = torch.reshape(torch.transpose(Y, 0, 1),(self.nNodes,self.nTest))        
        targets = self.samples['test']['targets']
        X = self.samples['test']['X0']
        print('Sample shape Y:', Y.shape, ', X:', X.shape, ', targets:', targets.shape)
        g_test = self.samples['test']['g_test']
        return Y,X,targets,g_test
    
    def get_valid_Samples(self,g_test):
#         Y = self.samples['valid']['signals']
#         Y = torch.reshape(torch.transpose(Y, 0, 2),(self.nNodes,self.nTest))
        V = self.V
        h_test = 1./g_test
        Ht = torch.tensor(np.dot(V,np.dot(np.diag(h_test),np.transpose(V))))
        targets = self.samples['valid']['targets']
        X = self.samples['valid']['X0']
        Y = torch.matmul(Ht,X)  
        print('Sample shape Y:', Y.shape, ', X:', X.shape, ', targets:', targets.shape)
        return Y,X,targets    
    
    def generate_new_test_Samples(self, g_test):
        Y_test = torch.zeros(self.nTest,self.nNodes)
        V = self.V
        h_test = 1./g_test
        Ht = torch.tensor(np.dot(V,np.dot(np.diag(h_test),np.transpose(V))))
        for n_t in range(self.nTest):
            x0 = self.samples['test']['X0'][:,n_t]
#             X0 = 
            self.samples['test']['gT'][:,n_t] = g_test
#             print(Gt.shape, x0.shape, g_test.shape)
            Y_test[n_t,:] = torch.matmul(Ht,x0) 
        self.samples['test']['signal'] = Y_test
        print('Re-generate sample shape Y:', self.samples['test']['signal'].shape)
#         Y = self.samples['test']['signal']
        return Y_test

    def identify_class_from_topN_signals(self, X_Hat, topN = 3):
        # Input H_Hat should be tensor
        # self.communityLabels should be list
        nNodes, nSamples = X_Hat.shape
        node2class = self.communityLabels
        target_Hat = []
        for n_sample in range(nSamples):
            # Top-N absolute value of X_Hat[:,n_sample]
            x_ns = torch.abs(torch.reshape(X_Hat[:,n_sample],(nNodes,)))
            signal_indexSorted = torch.argsort(x_ns)
            # Convert node indices to class numbers
            class_indexSorted = node2class[signal_indexSorted]
            class_topN = torch.tensor(class_indexSorted[-topN:])
            class_identified = torch.mode(class_topN).values.numpy()
            target_Hat.append(class_identified.item())
        return target_Hat
            
        
    def evaluate_slog(self, x_Hat, targets, topN = 1, tol = 1e-9):
        if topN == None:
            topN = 1
        # self.communityLabels[]: maps Node number to class
        N = len(targets)
        X_Hat = torch.transpose(torch.reshape(x_Hat, (self.nTest,self.nNodes)), 0, 1)    
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
#             X_Hat = torch.argmax(torch.abs(X_Hat), dim = 0)
            targetHat = self.identify_class_from_topN_signals(X_Hat, topN)
#             print(targetHat)
            assert len(targetHat) == self.nTest
            targetHat = torch.tensor(targetHat)
#             target_pre = []
#             for n in range(self.nTest):
#                 target_pre.append(self.communityLabels[X_Hat[n]])
#             print(target_pre)
#             print(X_Hat)
#             print(targets)
            #   And compute the error
#             targetHat = torch.tensor(target_pre)
            totalErrors = torch.sum(torch.abs(targetHat - targets) > tol)
            errorRate = totalErrors.type(self.dataType)/N
        else:
            X_Hat = np.array(X_Hat)
            targets = np.array(targets)
            #   We compute the target label (hardmax)
            X_Hat = np.argmax(X_Hat, axis = 0)
            #   And compute the error
            
#             print(X_Hat.shape)
            assert X_Hat.shape[0] == self.nTest
            target_pre = self.communityList[X_Hat]
#             print(target_pre)
            
            totalErrors = np.sum(np.abs(X_Hat - targets) > tol)
#             errorRate = totalErrors.astype(self.dataType)/N
        #   And from that, compute the accuracy
#         errorRate = None
        return errorRate

       
########################################################
############### Data Format ############################
########################################################                

    
class SourceLocalization_slog(_dataForClassification):


    def __init__(self, G, nTrain, nValid, nTest, sourceNodes, communityList, V, alpha = 1.0, selectMode = 'default',signalMode = 'default', tMax = None,
                 dataType = np.float64, device = 'cpu', **kwargs):
        # Initialize parent
        super().__init__()
        # store attributes
        self.dataType = dataType
        self.device = device
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        N_C = len(sourceNodes[0])
        self.N_C = N_C        
        C = len(sourceNodes)
        print('Number of input nodes per classes:',N_C,'; number of classes',C)
        self.C = C
        # If no tMax is specified, set it the maximum possible.
        if tMax == None:
            tMax = G.N
            
        #\\\ Generate the samples
        # Get the largest eigenvalue of the weighted adjacency matrix
        EW, VW = graph.computeGFT(G.W, order = 'totalVariation')
        eMax = np.max(EW)
        # Normalize the matrix so that it doesn't explode
        Wnorm = G.W / eMax
        # total number of samples
        nTotal = nTrain + nValid + nTest
        # sample source nodes
#         sampledSources = np.random.choice(sourceNodes, size = nTotal)
#         # sample diffusion times
#         sampledTimes = np.random.choice(tMax, size = nTotal)
#         # Since the signals are generated as W^t * delta, this reduces to the
#         # selection of a column of W^t (the column corresponding to the source
#         # node). Therefore, we generate an array of size tMax x N x N with all
#         # the powers of the matrix, and then we just simply select the
#         # corresponding column for the corresponding time
#         lastWt = np.eye(G.N, G.N)
#         Wt = lastWt.reshape([1, G.N, G.N])
#         for t in range(1,tMax):
#             lastWt = lastWt @ Wnorm
#             Wt = np.concatenate((Wt, lastWt.reshape([1, G.N, G.N])), axis = 0)
#         x = Wt[sampledTimes, :, sampledSources]
        
        
        # Let's generate slog input x
        sourceNode_set = np.array(sourceNodes)
        communityNode_set = np.array(communityList)
#         communityNode_set = communityList
        sampledSources_index = np.random.choice(np.arange(len(sourceNodes)), size = nTotal)
        sampledSources = sourceNode_set[sampledSources_index]
        sampledCommunities  = communityNode_set[sampledSources_index]      
        x_slog = torch.zeros(nTotal,G.N)
        X0 = torch.zeros(G.N,nTotal)
        N = G.N
        gT = np.ones([N,nTotal])+alpha*np.random.normal(0, 1, [N,nTotal]) 
        if selectMode == 'default':
            for n_t in range(nTotal):
                x0 = torch.zeros(G.N)
                for n_c in range(N_C):
                    if signalMode == 'default':
                        x0[sampledSources[n_t,n_c]] = 1
                    else:
                        x0[sampledSources[n_t,n_c]] = np.random.randn()
                X0[:,n_t] = x0
                gt = gT[:,n_t]
                Gt = torch.tensor(np.dot(V,np.dot(np.diag(gt),np.transpose(V))))
                x_slog[n_t,:] = torch.matmul(Gt,x0)
        else:
            for n_t in range(nTotal):
                x0 = torch.zeros(G.N)
                sampledCommunities_t = sampledCommunities[n_t]
                np.random.shuffle(sampledCommunities_t)
                for n_c in range(N_C):
                    if signalMode == 'default':
                        x0[sampledCommunities_t[n_c]] = 1
                    else:
                        x0[sampledCommunities_t[n_c]] = np.random.randn()
                X0[:,n_t] = x0
                gt = gT[:,n_t]
                Gt = torch.tensor(np.dot(V,np.dot(np.diag(gt),np.transpose(V))))
                x_slog[n_t,:] = torch.matmul(Gt,x0)      
        # Now, we have the signals and the labels
#         signals = x # nTotal x N
        signals = x_slog # nTotal x N        
        # Finally, we have to match the source nodes to the corresponding labels
        # which start at 0 and increase in integers.
#         nodesToLabels = {}
#         for it in range(len(sourceNodes)):
#             nodesToLabels[sampledSources_index[it]] = it
#         labels = [nodesToLabels[x] for x in sampledSources_index] # nTotal
        labels = sampledSources_index
        # Split and save them
        self.samples['train']['signals'] = signals[0:nTrain, :]
        self.samples['train']['targets'] = np.array(labels[0:nTrain])
        self.samples['train']['X0'] = X0[:,0:nTrain]
        self.samples['train']['gT'] = gT[:,0:nTrain]        
        self.samples['valid']['signals'] = signals[nTrain:nTrain+nValid, :]
        self.samples['valid']['targets'] =np.array(labels[nTrain:nTrain+nValid])
        self.samples['valid']['X0'] = X0[:,nTrain:nTrain+nValid]
        self.samples['valid']['gT'] = gT[:,nTrain:nTrain+nValid]        
        self.samples['test']['signals'] = signals[nTrain+nValid:nTotal, :]
        self.samples['test']['targets'] =np.array(labels[nTrain+nValid:nTotal])
        self.samples['test']['X0'] = X0[:,nTrain+nValid:nTotal]
        self.samples['test']['gT'] = gT[:,nTrain+nValid:nTotal]        
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)        
        
class noiseClassificationTest_generator(_dataForClassification):
    def __init__(self, gso, nTest, S, communityLabels, V,eigenvalues,**kwargs):
        # Initialize parent
        super().__init__()
        # store attributes
        N = gso.shape[0]
        nClass = np.max(communityLabels) + 1
        C = nClass
        print('nClass = ', nClass)
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 1.0
            
        if 'L' in kwargs.keys():
            L = kwargs['L']
        else:
            L = 1

        if 'tMax' in kwargs.keys():
            tMax = kwargs['tMax']
        else:
            tMax = N
            
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 1.0
            
        if 'filterMode' in kwargs.keys():
            filterMode = kwargs['filterMode']
        else:
            filterMode = 'default'
            
        if 'selectMode' in kwargs.keys():
            selectMode = kwargs['selectMode']
        else:
            selectMode = 'default' # means not random
            
        if 'signalMode' in kwargs.keys():
            signalMode = kwargs['signalMode']
        else:
            signalMode = 'default'
            
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
        nTrain = 0
        nValid = 0
        self.dataType = dataType
        self.device = device
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        self.nNodes = N
        self.N = N
        self.V = V
        self.L = L
        self.S = S
        self.alpha = alpha
        self.tMax = tMax
        self.filterType = filterType
        self.eigenvalues = eigenvalues
        self.signalMode = signalMode
        self.selectMode = selectMode
        self.filterMode = filterMode
        self.noiseType = noiseType
        self.noiseLevel = noiseLevel
        self.communityLabels = communityLabels
        self.gso = gso
        N_C = S
        self.N_C = N_C        
#         C = len(sourceNodes)
        print('Number of input nodes per classes:',N_C,'; number of classes',C)
        self.C = C
        self.Lambda = np.diag(eigenvalues)
        Phi = np.vander(eigenvalues,increasing=True) # Vandermonde Matrix 
        self.Phi = Phi
        #\\\ Generate the samples
        
        # total number of samples
        nTotal = nTrain + nValid + nTest
        # sample source nodes
        
        # Let's generate slog input x
        temp_result = SLOGtools.X_generate_fromSBM(N,nTest,N_C,communityLabels,gso,**kwargs)
        X0 = temp_result['X0']
        sampledIndicesList = temp_result['sampledIndicesList']
        # Random sample communities method 1
        x_slog = torch.zeros(nTest,N)
        
        sampledTime = np.random.choice(self.tMax, size = 1)
        if self.filterType == 'g':
            g_test = SLOGtools.g_generate_gso(self.nNodes,self.alpha, self.eigenvalues,self.L)
        elif self.filterType == 'wt':
            g_test = SLOGtools.wt_generate_gso(self.nNodes,self.alpha, self.eigenvalues,self.tMax)            
        else:
            g_test = SLOGtools.h_generate_gso(self.nNodes,self.alpha, self.eigenvalues,self.L)
        sampledTimes_test = np.random.choice(self.tMax, size = 1)
        if filterMode == 'default':
            gt = g_test
            ht = 1./gt
            Ht = torch.tensor(np.dot(V,np.dot(np.diag(ht),np.transpose(V))))
        else:
            t = sampledTime
            lambda_t = self.Phi[:,t].reshape(N)
            Ht = torch.tensor(np.dot(V,np.dot(np.diag(lambda_t),np.transpose(V))))     
        for n_t in range(nTotal):
            x0 = to_torch(X0[:,n_t])
            x_slog[n_t,:] = torch.matmul(Ht,x0)      
        # Now, we have the signals and the labels
        if noiseType == 'gaussion':
            noise = np.random.normal(0, 1, [self.nTest,self.N])        
            x_slog_np = to_numpy(x_slog)
            noise = noise/LA.norm(noise,'fro')*LA.norm(x_slog_np,'fro')            
        elif noiseType == 'uniform':
            noise = np.random.uniform(-1, 1, [nTest,self.N])        
            x_slog_np = to_numpy(x_slog)
            noise = noise/np.max(np.abs(noise))*np.max(np.abs(x_slog_np))          
        else:
            noise = np.zeros([self.nTest,self.N]) 
        signals = torch.tensor(to_numpy(x_slog) + noiseLevel*noise)# nTotal x N      
        print('Signal shape:',signals.shape)
        # Finally, we have to match the source nodes to the corresponding labels
        # which start at 0 and increase in integers.

        labels = sampledIndicesList
        # Split and save them
        self.samples['train']['signals'] = np.array([])
        self.samples['train']['targets'] = np.array([])
        self.samples['train']['X0'] = np.array([])
        self.samples['train']['gT'] = np.array([])        
        self.samples['valid']['signals'] = np.array([])
        self.samples['valid']['targets'] = np.array([])
        self.samples['valid']['X0'] = np.array([])
        self.samples['valid']['gT'] = np.array([])     
        self.samples['test']['signals'] = signals
        self.samples['test']['targets'] =np.array(labels)
        self.samples['test']['X0'] = X0
        self.samples['test']['gT'] = g_test
        self.samples['test']['g_test'] = g_test
        self.samples['test']['sampleTimes'] = sampledTimes_test
        self.samples['train']['Phi'] = Phi
        self.samples['train']['noiseLevel'] = noiseLevel      
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
        
    def get_test_Samples(self):
        Y = self.samples['test']['signals']
#         print(Y)
        Y = torch.tensor(Y)
        print(Y.shape)
        Y = torch.reshape(torch.transpose(Y, 0, 1),(self.nNodes,self.nTest))  
        targets = self.samples['test']['targets']
        X = self.samples['test']['X0']
        print('Sample shape Y:', Y.shape, ', X:', X.shape, ', targets:', targets.shape)
        g_test = self.samples['test']['g_test']
        return Y,X,targets,g_test
    
    def renew_testSet(self):
        
        temp_result = SLOGtools.X_generate_fromSBM(self.N,self.nTest,self.N_C,self.communityLabels,self.gso, selectMode = self.selectMode)
        X0 = temp_result['X0']
        sampledIndicesList = temp_result['sampledIndicesList']
        # Random sample communities method 1
        V = self.V
        N = self.N
        noiseLevel = self.noiseLevel
        noiseType = self.noiseType
        filterMode = self.filterMode 
        
        x_slog = torch.zeros(self.nTest,self.N)
        
        sampledTime = np.random.choice(self.tMax, size = 1)
        if self.filterType == 'g':
            g_test = SLOGtools.g_generate_gso(self.nNodes,self.alpha, self.eigenvalues,self.L)
        else:
            g_test = SLOGtools.h_generate_gso(self.nNodes,self.alpha, self.eigenvalues,self.L)
        sampledTimes_test = np.random.choice(self.tMax, size = 1)
        if filterMode == 'default':
            gt = g_test
            ht = 1./gt
            Ht = torch.tensor(np.dot(V,np.dot(np.diag(ht),np.transpose(V))))
        else:
            t = sampledTime
            lambda_t = self.Phi[:,t].reshape(N)
            Ht = torch.tensor(np.dot(V,np.dot(np.diag(lambda_t),np.transpose(V))))     
        for n_t in range(self.nTest):
            x0 = X0[:,n_t]
            x_slog[n_t,:] = torch.matmul(Ht,to_torch(x0))      
        # Now, we have the signals and the labels
        if noiseType == 'gaussion':
            noise = np.random.normal(0, 1, [self.nTest,self.N])        
            x_slog_np = to_numpy(x_slog)
            noise = noise/LA.norm(noise,'fro')*LA.norm(x_slog_np,'fro')            
        elif noiseType == 'uniform':
            noise = np.random.uniform(-1, 1, [self.nTest,self.N])        
            x_slog_np = to_numpy(x_slog)
            noise = noise/np.max(np.abs(noise))*np.max(np.abs(x_slog_np))          
        else:
            noise = np.zeros([self.nTest,self.N]) 
        signals = torch.tensor(to_numpy(x_slog) + noiseLevel*noise)# nTotal x N       
        print('Signal shape:',signals.shape)
        # Finally, we have to match the source nodes to the corresponding labels
        # which start at 0 and increase in integers.

        labels = sampledIndicesList
        # Split and save them

        self.samples['test']['signals'] = signals
        self.samples['test']['targets'] = np.array(labels)
        self.samples['test']['X0'] = X0
        self.samples['test']['gT'] = g_test
        self.samples['test']['g_test'] = g_test
        self.samples['test']['sampleTimes'] = sampledTime
 
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
    def generate_new_test_Samples(self, g_test):
#         gT_test = self.samples['test']['gT']
#         targets_test = self.samples['test']['targets']
#         X_test = self.samples['test']['X0']
        Y_test = torch.zeros(self.nTest,self.nNodes)
        V = self.V
        h_test = 1./g_test
        Ht = torch.tensor(np.dot(V,np.dot(np.diag(h_test),np.transpose(V))))
        for n_t in range(self.nTest):
            x0 = self.samples['test']['X0'][:,n_t]
#             X0 = 
            self.samples['test']['gT'][:,n_t] = g_test
#             print(Gt.shape, x0.shape, g_test.shape)
            Y_test[n_t,:] = torch.matmul(Ht,x0) 
        self.samples['test']['signal'] = Y_test
        print('Re-generate sample shape Y:', self.samples['test']['signal'].shape)
#         Y = self.samples['test']['signal']
        return Y_test

    def identify_class_from_topN_signals(self, X_Hat, topN = 3):
        # Input H_Hat should be tensor
        # self.communityLabels should be list
        nNodes, nSamples = X_Hat.shape
        node2class = self.communityLabels
        target_Hat = []
        for n_sample in range(nSamples):
            # Top-N absolute value of X_Hat[:,n_sample]
            x_ns = torch.abs(torch.reshape(X_Hat[:,n_sample],(nNodes,)))
            signal_indexSorted = torch.argsort(x_ns)
            # Convert node indices to class numbers
            class_indexSorted = node2class[signal_indexSorted]
            class_topN = torch.tensor(class_indexSorted[-topN:])
            class_identified = torch.mode(class_topN).values.numpy()
            target_Hat.append(class_identified.item())
        return target_Hat
            
        
    def evaluate_slog(self, x_Hat, targets, topN = 1, tol = 1e-9):
        if topN == None:
            topN = 1
        # self.communityLabels[]: maps Node number to class
        N = len(targets)
        X_Hat = torch.transpose(torch.reshape(x_Hat, (self.nTest,self.nNodes)), 0, 1)    
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
#             X_Hat = torch.argmax(torch.abs(X_Hat), dim = 0)
            targetHat = self.identify_class_from_topN_signals(X_Hat, topN)
#             print(targetHat)
            assert len(targetHat) == self.nTest
            targetHat = torch.tensor(targetHat)
#             target_pre = []
#             for n in range(self.nTest):
#                 target_pre.append(self.communityLabels[X_Hat[n]])
#             print(target_pre)
#             print(X_Hat)
#             print(targets)
            #   And compute the error
#             targetHat = torch.tensor(target_pre)
            totalErrors = torch.sum(torch.abs(targetHat - targets) > tol)
            errorRate = totalErrors.type(self.dataType)/N
        else:
            X_Hat = to_numpy(X_Hat)
            targets = np.array(targets)
            #   We compute the target label (hardmax)
            X_Hat = np.argmax(X_Hat, axis = 0)
            #   And compute the error
            
#             print(X_Hat.shape)
            assert X_Hat.shape[0] == self.nTest
            target_pre = self.communityLabels[X_Hat]
#             print(target_pre)
#             print(targets)
            totalErrors = np.sum(np.abs(target_pre - targets) > tol)
            errorRate = totalErrors.astype(self.dataType)/N
        #   And from that, compute the accuracy
#         errorRate = None
        return errorRate    