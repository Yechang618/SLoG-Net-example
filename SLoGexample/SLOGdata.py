# 2024/08/15~
# Chang Ye, cye7@ur.rochester.edu

import numpy as np
import torch
from SLoGexample import SLOGtools as SLOGtools
from numpy import linalg as LA

# Function to convert data to Torch tensor, while handling dtype and gradient tracking
def to_torch(x, **kwargs):
    """
    Convert input data to a Torch tensor. 
    If 'requires_grad' is passed, it can also handle whether the tensor 
    requires gradient tracking.
    """
    thisShape = x.shape  # Get the shape of the input data
    dataType = type(x)  # Get the type of input data
    
    # Check if 'requires_grad' is specified, else default to False
    requires_grad = kwargs.get('requires_grad', False)
    
    # If the input is numpy, convert to Torch tensor
    if 'numpy' in repr(dataType):
        return torch.tensor(x)
    return x

# Function to convert data to numpy array, while handling gradient tracking
def to_numpy(x, **kwargs):
    """
    Convert input data to a numpy array. 
    Handles tensors with gradient tracking.
    """
    thisShape = x.shape  # Get the shape of the input data
    dataType = type(x)  # Get the type of input data
    
    # Check if 'requires_grad' is specified, else default to False
    requires_grad = kwargs.get('requires_grad', False)
    
    # If the input is a Torch tensor, convert it to numpy
    if 'torch' in repr(dataType):
        if requires_grad == False:
            x1 = x.clone().detach().requires_grad_(False)  # Detach tensor if no gradient is needed
            return x1.numpy()  # Convert to numpy
        else:
            return x.numpy()  # Directly convert to numpy if gradients are tracked
    return x

# Function to assert and convert data to a specified type (torch or numpy)
def assertDType(x, dtype, **kwargs):
    """
    Assert and convert input data to a specified type (torch or numpy).
    """
    thisShape = x.shape  # Get the shape of the input data
    dataType = type(x)  # Get the type of input data
    
    # Check if 'requires_grad' is specified, else default to False
    requires_grad = kwargs.get('requires_grad', False)
    
    # If the input is numpy, convert it to Torch tensor if needed
    if 'numpy' in repr(dataType):
        if dtype == 'torch':
            return torch.tensor(x)
    # If the input is Torch tensor, convert it to numpy if needed
    elif 'torch' in repr(dataType):
        if dtype == 'numpy':
            if requires_grad == False:
                x1 = x.clone().detach().requires_grad_(False)
                return x1.numpy()
            else:
                return x.numpy()
    return x

# Function to normalize data along a specific axis by subtracting mean and dividing by standard deviation
def normalizeData(x, ax):
    """
    Normalize data by subtracting the mean and dividing by the standard deviation 
    along a specified axis.
    """
    thisShape = x.shape  # Get the shape of the input data
    assert ax < len(thisShape)  # Ensure the axis exists in the data

    dataType = type(x)  # Get the type of input data

    if 'numpy' in repr(dataType):
        # Compute the mean and standard deviation along the specified axis
        xMean = np.mean(x, axis=ax)
        xDev = np.std(x, axis=ax)
        # Expand dimensions to match original data
        xMean = np.expand_dims(xMean, ax)
        xDev = np.expand_dims(xDev, ax)
    elif 'torch' in repr(dataType):
        # Compute the mean and standard deviation along the specified axis
        xMean = torch.mean(x, dim=ax)
        xDev = torch.std(x, dim=ax)
        # Expand dimensions to match original data
        xMean = xMean.unsqueeze(ax)
        xDev = xDev.unsqueeze(ax)

    # Normalize by subtracting mean and dividing by standard deviation
    x = (x - xMean) / xDev
    return x

# Function for elementwise inversion of a tensor, where zero values are replaced with 1
def invertTensorEW(x):
    """
    Elementwise inversion of a tensor, while keeping zero elements as zero.
    """
    xInv = x.copy()  # Copy the input tensor
    xInv[x < zeroTolerance] = 1.  # Replace zeros with ones
    xInv = 1. / xInv  # Invert the tensor
    xInv[x < zeroTolerance] = 0.  # Restore the zeros
    return xInv

# Function to change the data type of the input variable to the specified type
def changeDataType(x, dataType):
    """
    Change the data type of variable x to the specified dataType.
    Handles conversion between numpy and torch data types.
    """
    # Check if the variable has a 'dtype' attribute to determine its current type
    if 'dtype' in dir(x):
        varType = x.dtype
    
    # Default behavior assumes conversion to numpy
    if 'numpy' in repr(dataType):
        if 'torch' in repr(varType):
            # Convert from torch to numpy, move tensor to CPU if necessary
            x = x.cpu().numpy().astype(dataType)
        elif 'numpy' in repr(type(x)):
            x = x.astype(dataType)  # Convert numpy array to desired type

    elif 'torch' in repr(dataType):
        if 'torch' in repr(varType):
            x = x.type(dataType)  # Convert within torch
        elif 'numpy' in repr(type(x)):
            x = torch.tensor(x, dtype=dataType)  # Convert from numpy to torch
            
    return x

# Function to generate test samples for graph signal processing tasks
def test_sample_generate(nNodes, S, P, nTest, gso, L=3, noiseLevel=0, alpha=1.0, filterType='h'):
    """
    Generate test samples for graph signal processing tasks.
    """
    # Get eigenvalues and eigenvectors of the graph Laplacian
    d_slog, An_slog, eigenvalues_slog, V_slog = SLOGtools.get_eig_normalized_adj(gso)
    
    # Generate the GSO (graph shift operator) for the test batch
    g_batch = SLOGtools.h_batch_generate_gso(nNodes, nTest, alpha, eigenvalues_slog, L)
    
    # Initialize arrays for test samples
    XTest = np.zeros([nNodes, P, nTest])
    YTest = np.zeros([nNodes, P, nTest])
    
    for n_t in range(nTest):
        # Generate random graph signals (X0)
        X0 = X_generate(nNodes, P, S)
        XTest[:, :, n_t] = X0
        
        # Get the graph shift operator for the current test case
        gt = g_batch[:, n_t]
        ht = 1. / gt
        Ht = np.dot(V, np.dot(np.diag(ht), np.transpose(V)))
        
        # Generate the output signals YTest
        YTest[:, :, n_t] = np.dot(Ht, X0)
    
    # Return the generated results
    result = {
        'XTest': XTest,
        'YTest': YTest,
        'g_batch': g_batch
    }
    return result

# Class for storing general data used in machine learning tasks
class super_data:
    def __init__(self):
        """
        Minimal set of attributes that all data classes should have, such as data type, device, and sample data.
        """
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {
            'train': {'signals': None, 'targets': None},
            'valid': {'signals': None, 'targets': None},
            'test': {'signals': None, 'targets': None}
        }

# Base class for all data sets with common methods for handling data
class _data:
    def __init__(self):
        """
        Minimal set of attributes for all data classes, including sample data and methods for handling them.
        """
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {
            'train': {'signals': None, 'targets': None},
            'valid': {'signals': None, 'targets': None},
            'test': {'signals': None, 'targets': None}
        }

    # Method to get samples (train/valid/test) based on type and additional arguments
    def getSamples(self, samplesType, *args):
        """
        Return samples (train/valid/test) based on the input type and additional arguments.
        """
        assert samplesType in ['train', 'valid', 'test']
        assert len(args) <= 1
        
        # Get the signals and targets for the specified sample type
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['targets']
        
        if len(args) == 1:
            # If argument is an integer, randomly select that number of samples
            if isinstance(args[0], int):
                nSamples = x.shape[0]
                assert args[0] <= nSamples
                selectedIndices = np.random.choice(nSamples, size=args[0], replace=False)
                xSelected = x[selectedIndices]
                y = y[selectedIndices]
            # If argument is a list or array, select specific samples
            else:
                xSelected = x[args[0]]
                y = y[args[0]]
                
            # Ensure correct shape if a single sample is selected
            if len(xSelected.shape) < len(x.shape):
                if 'torch' in self.dataType:
                    x = xSelected.unsqueeze(0)
                else:
                    x = np.expand_dims(xSelected, axis=0)
            else:
                x = xSelected

        return x, y

    # Method to expand dimensions of signal data to match the expected format
    def expandDims(self):
        """
        Expand the dimensions of signals in the dataset to match the expected format 
        (nDataPoints x 1 x nNodes or nDataPoints x 1 x 1 x nNodes).
        """
        for key in self.samples.keys():
            if self.samples[key]['signals'] is not None:
                if len(self.samples[key]['signals'].shape) == 2:
                    if 'torch' in repr(self.dataType):
                        self.samples[key]['signals'] = self.samples[key]['signals'].unsqueeze(1)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(self.samples[key]['signals'], axis=1)
                elif len(self.samples[key]['signals'].shape) == 3:
                    if 'torch' in repr(self.dataType):
                        self.samples[key]['signals'] = self.samples[key]['signals'].unsqueeze(2)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(self.samples[key]['signals'], axis=2)

    # Method to convert sample data to a specified data type (e.g., float32, float64)
    def astype(self, dataType):
        """
        Convert the sample data to the specified data type (e.g., float32, int64).
        """
        targetType = str(self.samples['train']['targets'].dtype)
        if 'int' in targetType:
            if 'numpy' in repr(dataType):
                targetType = np.int64 if '64' in targetType else np.int32
            elif 'torch' in repr(dataType):
                targetType = torch.int64 if '64' in targetType else torch.int32
        else:
            targetType = dataType

        for key in self.samples.keys():
            self.samples[key]['signals'] = changeDataType(self.samples[key]['signals'], dataType)
            self.samples

class _dataForClassification(_data):
    # Internal supraclass from which data classes inherit when they are used
    # for classification. This makes sure that the .evaluate() method works the
    # same way in all cases (by counting how many examples are incorrectly labeled),
    # justifying the use of another internal class for shared functionality.
    
    def __init__(self):
        # Initialize the parent class
        super().__init__()

    def evaluate(self, yHat, y, tol=1e-9):
        """
        Evaluates the accuracy of the classification by comparing the predicted 
        labels (yHat) to the actual labels (y).
        
        The method calculates how many predicted labels (yHat) do not match the 
        actual labels (y) and returns the error rate (percentage of incorrect labels).
        """
        N = len(y)  # Total number of examples
        
        if 'torch' in repr(self.dataType):
            # If the data type is based on PyTorch, use tensor operations
            
            # Compute the predicted class by applying the argmax to get the class with highest score
            yHat = torch.argmax(yHat, dim=1)
            
            # Calculate the number of errors (examples where yHat != y)
            totalErrors = torch.sum(torch.abs(yHat - y) > tol)
            
            # Calculate the error rate, converting the count of errors to the same type as self.dataType
            errorRate = totalErrors.type(self.dataType) / N
        else:
            # If the data type is numpy, handle the comparison with numpy operations
            yHat = np.array(yHat)
            y = np.array(y)
            
            # Compute the predicted class by applying the argmax to get the class with highest score
            yHat = np.argmax(yHat, axis=1)
            
            # Calculate the number of errors (examples where yHat != y)
            totalErrors = np.sum(np.abs(yHat - y) > tol)
            
            # Calculate the error rate, converting the count of errors to the same type as self.dataType
            errorRate = totalErrors.astype(self.dataType) / N
        
        # Return the calculated error rate (or accuracy can be 1 - errorRate)
        return errorRate


## General data class for SLOG model
class SLOG_GeneralData(_dataForClassification):
    # This class generates general data for the SLOG classification task, inheriting from _dataForClassification
    
    def __init__(self, G, nTrain, nValid, nTest, S, V, eigenvalues, **kwargs):
        # Initialize parent class (_dataForClassification)
        super().__init__()
        
        # Store and initialize various attributes (e.g., number of training/validation/test samples)
        if 'L' in kwargs.keys():
            L = kwargs['L']
        else:
            L = 1  # Default value if not provided
        
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 1.0  # Default value for alpha
        
        if 'tMax' in kwargs.keys():
            tMax = kwargs['tMax']
        else:
            tMax = 1.0  # Default value for tMax
        
        if 'filterType' in kwargs.keys():
            filterType = kwargs['filterType']
        else:
            filterType = 'g'  # Default filter type
        
        if 'noiseLevel' in kwargs.keys():
            noiseLevel = kwargs['noiseLevel']
        else:
            noiseLevel = 0  # Default noise level
        
        if 'noiseType' in kwargs.keys():
            noiseType = kwargs['noiseType']
        else:
            noiseType = 'gaussian'  # Default noise type
        
        if 'dataType' in kwargs.keys():
            dataType = kwargs['dataType']
        else:
            dataType = np.float64  # Default data type
        
        if 'device' in kwargs.keys():
            device = kwargs['device']
        else:
            device = 'cpu'  # Default device (CPU)
        
        print(dataType)  # Print the data type
        
        # Assigning the values to class variables
        self.dataType = dataType
        self.device = device
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        self.nNodes = G.N  # Number of nodes in the graph G
        self.tMax = tMax
        self.V = V  # Eigenvectors
        self.L = L  # Filter order (Laplacian size)
        self.S = S  # Signal
        self.filterType = filterType  # Type of filter ('g' for graph)
        self.Lambda = np.diag(eigenvalues)  # Diagonal matrix of eigenvalues
        Phi = np.vander(eigenvalues, increasing=True)  # Vandermonde matrix for eigenvalues
        self.Phi = Phi
        
        # Generate the samples using helper functions from SLOGtools
        if 'filterType' == 'h':
            g_test = SLOGtools.h_generate_gso(self.nNodes, alpha, eigenvalues, L)
        else:
            g_test = SLOGtools.g_generate_gso(self.nNodes, alpha, eigenvalues, L)
        
        h_test = 1. / g_test  # Inverse of the generated graph signal operator
        
        # Generate training, validation, and test samples using SLOGtools
        X_train = SLOGtools.X_generate(self.nNodes, nTrain, S)
        X_valid = SLOGtools.X_generate(self.nNodes, nValid, S)
        X_test = SLOGtools.X_generate(self.nNodes, nTest, S)
        
        # Convert to numpy arrays (if needed) for operations
        V = to_numpy(V)
        h_test = to_numpy(h_test)
        
        # Compute matrix H based on V and h_test
        H = np.dot(V, np.dot(np.diag(h_test), V.T))
        
        # Generate noise based on the specified type (Gaussian or Uniform)
        if noiseType == 'gaussian':
            noise = np.random.normal(0, 1, [self.nNodes, nTest])
            noise = noise / LA.norm(noise, 'fro') * LA.norm(X_test, 'fro')
        elif noiseType == 'uniform':
            noise = np.random.uniform(-1, 1, [self.nNodes, nTest])
            noise = noise / np.max(np.abs(noise)) * np.max(np.abs(X_test))
        else:
            noise = np.zeros([self.nNodes, nTest])  # No noise if not specified
        
        # Compute the noisy test samples Y_test
        Y_test = np.dot(H, X_test) + noiseLevel * noise
        
        # Split and store the generated samples for training, validation, and testing
        self.samples['train']['X0'] = X_train
        self.samples['valid']['X0'] = X_valid
        self.samples['test']['signals'] = Y_test
        self.samples['test']['X0'] = X_test
        self.samples['test']['g_test'] = g_test
        self.samples['train']['Phi'] = Phi
        self.samples['train']['noiseLevel'] = noiseLevel
    