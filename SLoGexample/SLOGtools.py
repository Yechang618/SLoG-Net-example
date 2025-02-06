# 2024/08/15~
# Chang Ye, cye7@ur.rochester.edu

import torch
import torch.nn as nn
import math
import numpy as np
from numpy import random
from scipy import linalg
from numpy import linalg as LA
from torch.autograd import Variable
from sklearn.cluster import SpectralClustering
import os
import pickle
zeroTolerance = 1e-9

########################################################
############### SLOG-NET Modules #######################
########################################################
# from SLoGexample import SLOGmodule as myModules

################################################################
############### Loss function ##################################
################################################################
def myLoss(x_pre, x_ture):
    """
    Custom loss function that returns the minimum between two possible losses:
    1. The squared error between predicted (x_pre) and true (x_ture) values.
    2. The squared error between predicted and the negative of the true values.

    Input:
        x_pre: Predicted values.
        x_ture: True values.
    
    Output:
        The minimum of the squared error between the predicted and true values or
        between the predicted and negative true values.
    """
    sizes = x_pre.size()
    # Return the minimum of the squared error or squared error between negative truth
    return torch.min(((x_pre-x_ture)**2).mean(),((x_pre+x_ture)**2).mean())
    

########################################################
############### Save and Load ##########################
########################################################

def writeVarValues(fileToWrite, varValues):
    """
    Write the value of several string variables specified by a dictionary into
    the designated .txt file.
    
    Input:
        fileToWrite (os.path): text file to save the specified variables.
        varValues (dictionary): values to save in the text file. They are
            saved in the format "key = value".
    """
    with open(fileToWrite, 'a+') as file:
        for key in varValues.keys():
            file.write('%s = %s\n' % (key, varValues[key]))
        file.write('\n')

def saveSeed(randomStates, saveDir):
    """
    Saves the random states for different modules (e.g., numpy, torch) to a file
    so that they can be loaded and reused later. This helps in making the results reproducible.
    
    Inputs:
        randomStates (list): List of dictionaries containing random generator states
            and seed values for each module (e.g., numpy, torch).
        saveDir (path): Directory to save the seed file.
    """
    pathToSeed = os.path.join(saveDir, 'randomSeedUsed.pkl')
    with open(pathToSeed, 'wb') as seedFile:
        pickle.dump({'randomStates': randomStates}, seedFile)
        
def loadSeed(loadDir):
    """
    Loads the saved random states and seeds for different modules (e.g., numpy, torch).
    This function ensures that the random number generators are restored to the same state as
    when they were saved, ensuring reproducibility.
    
    Inputs:
        loadDir (path): Directory where the seed file is stored.
    """
    pathToSeed = os.path.join(loadDir, 'randomSeedUsed.pkl')
    with open(pathToSeed, 'rb') as seedFile:
        randomStates = pickle.load(seedFile)
        randomStates = randomStates['randomStates']
    for module in randomStates:
        thisModule = module['module']
        if thisModule == 'numpy':
            np.random.RandomState().set_state(module['state'])  # Restore numpy RNG state
        elif thisModule == 'torch':
            torch.set_rng_state(module['state'])  # Restore torch RNG state
            torch.manual_seed(module['seed'])  # Restore torch seed

########################################################
############### Graph Tools ############################
########################################################

def get_eig_normalized_adj(gso):
    """
    Computes the normalized adjacency matrix for a given graph shift operator (GSO).
    This function computes both the normalized adjacency matrix and its eigenvalues/eigenvectors.

    Input:
        gso (np.array): Graph Shift Operator (Adjacency matrix).
    
    Output:
        d (np.array): Degree matrix.
        An (np.array): Normalized adjacency matrix.
        eigenvalues (np.array): Eigenvalues of the normalized adjacency matrix.
        V (np.array): Eigenvectors of the normalized adjacency matrix.
    """
    N,_ = np.shape(gso)
    # Compute degree matrix (diagonal matrix with degrees of each node)
    d = np.sum(gso, axis=1)
    # Compute Laplacian matrix
    Lp = np.diag(d) - gso
    Lpn = np.zeros((N,N))
    An = np.zeros((N,N))
    
    # Normalize Laplacian and Adjacency matrix
    for i in range(N):
        for j in range(N):
            Lpn[i,j] = Lp[i,j]/np.sqrt(d[i])/np.sqrt(d[j])
            An[i,j] = gso[i,j]/np.sqrt(d[i])/np.sqrt(d[j])

    # Eigenvalue decomposition of normalized adjacency matrix
    eigenvalues, V = np.linalg.eig(An)  
    V = np.real(V)
    eigenvalues = np.real(eigenvalues)
    
    return d, An, eigenvalues, V  

################################
####### Graph GENERATION #######
################################
class Graph():
    """
    A class that handles various graph properties and computations for graphs.
    This class can create different types of graphs (e.g., 'SBM', 'SmallWorld', 'fuseEdges', 'adjacency'),
    store graph properties like adjacency matrix, degree matrix, Laplacian, and compute Graph Fourier Transform (GFT).
    
    Initialization:
        graphType (string): Type of graph ('SBM', 'SmallWorld', 'fuseEdges', 'adjacency')
        N (int): Number of nodes in the graph.
        graphOptions (dict): Additional options specific to the graph type.
        [optionalArguments]: Additional options (e.g., load from file, save directory).

    Attributes:
        .N (int): Number of nodes.
        .M (int): Number of edges.
        .W (np.array): Weighted adjacency matrix.
        .D (np.array): Degree matrix.
        .A (np.array): Unweighted adjacency matrix.
        .L (np.array): Laplacian matrix (only for undirected graphs without self-loops).
        .S (np.array): Graph shift operator (default is weighted adjacency).
        .E (np.array): Eigenvalue matrix.
        .V (np.array): Eigenvector matrix.
        .undirected (bool): Whether the graph is undirected.
        .selfLoops (bool): Whether the graph has self-loops.

    Methods:
        .computeGFT(): Computes the Graph Fourier Transform (GFT).
        .setGSO(S, GFT = 'no'): Sets a new Graph Shift Operator (GSO) and optionally computes GFT.
    """
    def __init__(self, graphType, N, graphOptions, **kwargs):
        """
        Initializes the graph object by either loading a graph from a file
        or creating a new graph from scratch based on the specified graph type.

        Input:
            graphType (string): Type of graph to create ('SBM', 'SmallWorld', etc.).
            N (int): Number of nodes in the graph.
            graphOptions (dict): Options for graph creation (e.g., parameters for SBM).
            **kwargs: Optional keyword arguments (e.g., for saving/loading graph).
        """
        assert N > 0  # Ensure the number of nodes is positive
        
        # Handle save/load options
        if 'save_dir' in kwargs.keys():
            self.save_dir = kwargs['save_dir']
            self.save_to_local = True
        else:
            self.save_dir = None
            self.save_to_local = False       

        if 'load_dir' in kwargs.keys():
            self.load_dir = kwargs['load_dir']
            self.load_from_local = True
        else:
            self.load_dir = None
            self.load_from_local = False       

        # Create or load the graph
        if self.load_from_local == True:
            load_name = 'gso-' + graphType + '.npy'
            load_dir = os.path.join(self.load_dir, load_name) 
            self.W = np.load(load_dir)
        else:
            self.W = createGraph(graphType, N, graphOptions)
        
        # Initialize graph properties
        self.N = self.W.shape[0]  # Number of nodes
        self.undirected = np.allclose(self.W, self.W.T, atol = zeroTolerance)  # Check if graph is undirected
        self.selfLoops = np.sum(np.abs(np.diag(self.W)) > zeroTolerance) > 0  # Check if graph has self-loops
        
        # Degree matrix (diagonal matrix with node degrees)
        self.D = np.diag(np.sum(self.W, axis = 1))
        
        # Number of edges (sum of the upper triangle for undirected or all for directed)
        self.M = int(np.sum(np.triu(self.W)) if self.undirected else np.sum(self.W))
        
        # Unweighted adjacency matrix (0/1 matrix)
        self.A = (np.abs(self.W) > 0).astype(self.W.dtype)
        
        # Laplacian matrix (only for undirected graphs without self-loops)
        if self.undirected and not self.selfLoops:
            self.L = adjacencyToLaplacian(self.W)
        else:
            self.L = None
        
        # Graph shift operator (by default, the weighted adjacency matrix)
        self.S = self.W
        
        # Declare GFT variables (eigenvalues and eigenvectors), computed later
        self.E = None
        self.V = None
        
        # Save the graph if needed
        if self.save_to_local == True:
            graph_save_name = 'gso-' + graphType + '.npy'
            graph_save_dir = os.path.join(self.save_dir, graph_save_name) 
            np.save(graph_save_dir, self.A)

    def computeGFT(self):
        """
        Computes the Graph Fourier Transform (GFT) of the stored GSO.
        This function computes the eigenvalues (E) and eigenvectors (V) of the GSO.
        """
        if self.S is not None:
            self.E, self.V = computeGFT(self.S, order = 'totalVariation')

    def setGSO(self, S, GFT = 'no'):
        """
        Sets a new Graph Shift Operator (GSO) and optionally computes its GFT.
        
        Input:
            S (np.array): New GSO matrix.
            GFT (str): Whether to compute the GFT ('no', 'increasing', 'totalVariation').
        """
        assert S.shape[0] == S.shape[1] == self.N  # Ensure S has the correct shape
        assert GFT == 'no' or GFT == 'increasing' or GFT == 'totalVariation'  # Validate GFT option
        
        # Set the new GSO and compute GFT if requested
        self.S = S
        if GFT == 'no':
            self.E = None
            self.V = None
        else:
            self.E, self.V = computeGFT(self.S, order = GFT)
################################
####### Graph Functions ########
################################
def createGraph(graphType, N, graphOptions):
    """
    createGraph: creates a graph of a specified type
    
    Input:
        graphType (string): 'SBM', 'Random Geometric', 'ER','BA', and 'adjacency'
        N (int): Number of nodes
        graphOptions (dict): Depends on the type selected.
        Obs.: More types to come.
        
    Output:
        W (np.array): adjacency matrix of shape N x N
    
    Optional inputs (by keyword):
        graphType: 'SBM'
            'nCommunities': (int) number of communities
            'probIntra': (float) probability of drawing an edge between nodes
                inside the same community
            'probInter': (float) probability of drawing an edge between nodes
                of different communities
            Obs.: This always results in a connected graph.
        graphType: 'Random Geometric'
            'distance': (float) in [0,1)
        graphType: 'ER'
            'probIntra': (float) probability of drawing an edge between nodes
        graphType: 'BA'，Barabási–Albert
            
        graphType: 'adjacency'
            'adjacencyMatrix' (np.array): just return the given adjacency
                matrix (after checking it has N nodes)
    """
    # Check if the number of nodes N is valid (non-negative)
    assert N >= 0

    if graphType == 'SBM':
        # Assert that the correct number of keys (3) are in the graphOptions for SBM
        assert(len(graphOptions.keys())) == 3
        C = graphOptions['nCommunities'] # Number of communities
        assert int(C) == C # Ensure the number of communities is an integer
        pii = graphOptions['probIntra'] # Intracommunity probability
        pij = graphOptions['probInter'] # Intercommunity probability
        assert 0 <= pii <= 1 # Validate probabilities (should be between 0 and 1)
        assert 0 <= pij <= 1
        # Start creating the SBM graph using random probabilities for the edges
        nNodesC = [N//C] * C # Number of nodes per community: evenly distributed
        c = 0 # Community counter
        while sum(nNodesC) < N: # If there are still nodes to allocate, add them
            nNodesC[c] += 1
            c += 1
        # Now, nNodesC contains the number of nodes per community
        probMatrix = np.zeros([N,N]) # Initialize the probability matrix
        nNodesCIndex = [0] + np.cumsum(nNodesC).tolist() # Cumulative sum of node counts for each community
        # Create block diagonal structure for the probability matrix
        for c in range(C):
            probMatrix[ nNodesCIndex[c] : nNodesCIndex[c+1], \
                        nNodesCIndex[c] : nNodesCIndex[c+1] ] = \
                np.ones([nNodesC[c], nNodesC[c]])
        # Set intra-community and inter-community edge probabilities
        probMatrix = pii * probMatrix + pij * (1 - probMatrix)
        # Generate the adjacency matrix W using the probability matrix
        connectedGraph = False
        while not connectedGraph:
            W = np.random.rand(N,N) # Random adjacency matrix
            W = (W < probMatrix).astype(np.float64) # Set 1 where the random number is less than probability
            W = np.triu(W, 1) # Make it upper triangular (undirected graph, no self-loops)
            W = W + W.T # Make the graph undirected by adding the transpose
            connectedGraph = isConnected(W) # Check if the graph is connected
    
    elif graphType == 'Random Geometric':
        # Check that distance key exists in graphOptions for Random Geometric graph
        assert 'distance' in graphOptions.keys()
        d = graphOptions['distance']
        connectedGraph = False        
        while not connectedGraph:        
            xy = random.rand(N,2) # Generate random positions in a 2D plane
            W = np.zeros([N,N]) # Initialize adjacency matrix
            for n in range(N):
                x1 = xy[n]
                for m in range(N):
                    if n != m:
                        x2 = xy[m]
                        # If distance between nodes is less than threshold, add an edge
                        if LA.norm(x1 - x2) <= d:
                            W[n,m] = 1
            connectedGraph = isConnected(W) # Check if graph is connected
        assert W.shape[0] == W.shape[1] == N # Ensure the adjacency matrix is N x N

    elif graphType == 'ER':
        # If 'probIntra' is provided, use it; otherwise default to 0.3
        if 'probIntra' in graphOptions.keys():
            p = graphOptions['probIntra']
        else:
            p = 0.3
        connectedGraph = False   
        perm_amb = True
        # Keep generating the random graph until it's connected and no permutation ambiguity exists
        while perm_amb or not connectedGraph:        
            W = np.random.default_rng().uniform(0,1,[N,N]) # Generate random uniform matrix
            W = np.triu(W,k=1) # Upper triangular matrix
            W = W + W.T # Make the graph undirected
            W = (W < p) * 1 # Set edges based on probability p
            connectedGraph = isConnected(W) # Check if graph is connected
            perm_amb = perm_ambiguity_exam(W) # Check for permutation ambiguity
        assert W.shape[0] == W.shape[1] == N # Ensure the adjacency matrix is N x N

    elif graphType == 'BA':
        # If 'alpha' is provided, use it; otherwise default to 1.0
        if 'alpha' in graphOptions.keys():
            alpha = graphOptions['alpha']
        else:
            alpha = 1.0
        connectedGraph = False   
        perm_amb = True
        while perm_amb or not connectedGraph:        
            W = np.zeros([N,N]) # Initialize adjacency matrix
            degree = np.zeros(N) # Initialize degree vector
            W[0,1] = 1 # Initial edge between first two nodes
            W[1,0] = 1
            for n in range(2, N): # Add new nodes
                degree = np.sum(W, axis=0) # Compute current node degrees
                degree = degree ** alpha # Apply preferential attachment
                degree_sum = np.sum(degree) # Sum of degrees
                degree_n = 0
                while degree_n < 1:
                    for m in range(0, n):
                        prob = degree[m] / degree_sum
                        if random.rand() < prob: # Add edge with probability proportional to degree
                            degree[n] += 1
                            W[m,n] = 1
                            W[n,m] = 1
                    degree_n = degree[n]
            connectedGraph = isConnected(W)  # Check if graph is connected
            perm_amb = perm_ambiguity_exam(W) # Check for permutation ambiguity
        assert W.shape[0] == W.shape[1] == N # Ensure the adjacency matrix is N x N
    
    elif graphType == 'adjacency':
        # If 'adjacencyMatrix' is provided, simply return it after validating its size
        assert 'adjacencyMatrix' in graphOptions.keys()
        W = graphOptions['adjacencyMatrix']
        assert W.shape[0] == W.shape[1] == N # Ensure the adjacency matrix is N x N

    return W # Return the generated adjacency matrix

def adjacencyToLaplacian(W):
    """
    adjacencyToLaplacian: Computes the Laplacian from an Adjacency matrix

    Input:

        W (np.array): adjacency matrix

    Output:

        L (np.array): Laplacian matrix
    """
    # Check that the matrix is square
    assert W.shape[0] == W.shape[1]
    # Compute the degree vector (sum of each row in the adjacency matrix)
    d = np.sum(W, axis=1)
    # Build the degree matrix (diagonal matrix with degree values)
    D = np.diag(d)
    # Return the Laplacian matrix (degree matrix - adjacency matrix)
    return D - W

def perm_ambiguity_exam(W):
    """
    perm_ambiguity_exam: Examines if there is permutation ambiguity in the graph.

    Input:
        W (np.array): adjacency matrix

    Output:
        perm_amb (bool): True if permutation ambiguity exists, otherwise False
    """
    perm_amb = False
    d, Lpn, eigenvalues, V = get_eig_normalized_adj(W)
    N = V.shape[0]
    
    for n in range(N):
        v = V[:,n]
        v0 = v / np.sqrt(2) # Normalize the eigenvector
        if np.abs(np.sum(v0)) < zeroTolerance: # Check for permutation ambiguity
            perm_amb = True
    return perm_amb 

def normalizeAdjacency(W):
    """
    NormalizeAdjacency: Computes the degree-normalized adjacency matrix

    Input:

        W (np.array): adjacency matrix

    Output:

        A (np.array): degree-normalized adjacency matrix
    """
    # Check that the matrix is square
    assert W.shape[0] == W.shape[1]
    # Compute the degree vector (sum of each row in the adjacency matrix)
    d = np.sum(W, axis=1)
    # Invert the square root of the degree
    d = 1 / np.sqrt(d)
    # Build the degree matrix (diagonal matrix with the inverse square root of the degree)
    D = np.diag(d)
    # Return the normalized adjacency matrix (D @ W @ D)
    return D @ W @ D

def normalizeLaplacian(L):
    """
    NormalizeLaplacian: Computes the degree-normalized Laplacian matrix

    Input:

        L (np.array): Laplacian matrix

    Output:

        normL (np.array): degree-normalized Laplacian matrix
    """
    # Check that the matrix is square
    assert L.shape[0] == L.shape[1]
    # Compute the degree vector (diagonal elements of L)
    d = np.diag(L)
    # Invert the square root of the degree
    d = 1 / np.sqrt(d)
    # Build the degree matrix (diagonal matrix with the inverse square root of the degree)
    D = np.diag(d)
    # Return the normalized Laplacian matrix (D @ L @ D)
    return D @ L @ D

def computeGFT(S, order = 'no'):
    """
    computeGFT: Computes the frequency basis (eigenvectors) and frequency
        coefficients (eigenvalues) of a given GSO

    Input:

        S (np.array): graph shift operator matrix
        order (string): 'no', 'increasing', 'totalVariation' chosen order of
            frequency coefficients (default: 'no')

    Output:

        E (np.array): diagonal matrix with the frequency coefficients
            (eigenvalues) in the diagonal
        V (np.array): matrix with frequency basis (eigenvectors)
    """
    # Validate the order input
    assert order == 'totalVariation' or order == 'no' or order == 'increasing'
    # Check if the matrix is square
    assert S.shape[0] == S.shape[1]
    # Check if it is symmetric
    symmetric = np.allclose(S, S.T, atol = zeroTolerance)
    # Compute eigenvalues and eigenvectors
    if symmetric:
        e, V = np.linalg.eigh(S)
    else:
        e, V = np.linalg.eig(S)
    # Sort eigenvalues based on the desired ordering
    if order == 'totalVariation':
        eMax = np.max(e)
        sortIndex = np.argsort(np.abs(e - eMax))
    elif order == 'increasing':
        sortIndex = np.argsort(np.abs(e))
    else:
        sortIndex = np.arange(0, S.shape[0])
    e = e[sortIndex]
    V = V[:, sortIndex]
    E = np.diag(e)
    return E, V

def isConnected(W):
    """
    isConnected: Determine if a graph is connected.

    Input:
        W (np.array): Adjacency matrix of the graph.

    Output:
        connected (bool): True if the graph is connected, False otherwise.
    
    Obs.: If the graph is directed, we consider it connected when there is
    at least one edge that would make it connected (i.e., if we drop the 
    direction of all edges and just treat them as undirected, the resulting
    graph would be connected).
    """
    # Check if the graph is undirected by comparing the adjacency matrix to its transpose
    undirected = np.allclose(W, W.T, atol=zeroTolerance)
    
    # If the graph is directed, make it undirected by averaging the adjacency matrix and its transpose
    if not undirected:
        W = 0.5 * (W + W.T)
    
    # Convert adjacency matrix to Laplacian matrix
    L = adjacencyToLaplacian(W)
    
    # Compute graph Fourier transform (eigenvalues and eigenvectors)
    E, V = computeGFT(L)
    
    # Extract the eigenvalues
    e = np.diag(E)
    
    # Count the number of connected components based on eigenvalue analysis
    nComponents = np.sum(e < zeroTolerance)
    
    # If there is only one component, the graph is connected
    connected = (nComponents == 1)
    
    return connected

def adjacencyToLaplacian(W):
    """
    adjacencyToLaplacian: Converts an adjacency matrix to a Laplacian matrix.

    Input:
        W (np.array): Adjacency matrix.

    Output:
        L (np.array): Laplacian matrix.
    """
    # Ensure the adjacency matrix is square (i.e., has the same number of rows and columns)
    assert W.shape[0] == W.shape[1]
    
    # Compute the degree vector (sum of each row in the adjacency matrix)
    d = np.sum(W, axis=1)
    
    # Create a diagonal degree matrix
    D = np.diag(d)
    
    # Return the Laplacian matrix: D - W
    return D - W

def normalizeAdjacency(W):
    """
    NormalizeAdjacency: Computes the degree-normalized adjacency matrix.

    Input:
        W (np.array): Adjacency matrix.

    Output:
        A (np.array): Degree-normalized adjacency matrix.
    """
    # Ensure the adjacency matrix is square
    assert W.shape[0] == W.shape[1]
    
    # Compute the degree vector
    d = np.sum(W, axis=1)
    
    # Invert the square root of the degree values
    d = 1 / np.sqrt(d)
    
    # Create the diagonal matrix with the inverted square roots of degrees
    D = np.diag(d)
    
    # Return the degree-normalized adjacency matrix: D * W * D
    return D @ W @ D

# Data generation functions
def data_generate(N, P, V, theta, alpha):
    """
    Data generation function that produces the signal X, filter g, and noisy measurements Z and Y.

    Input:
        N (int): Number of nodes.
        P (int): Number of samples.
        V (np.array): Eigenvectors (used in generating the filter).
        theta (float): Proportion of nodes used for signal generation.
        alpha (float): Scaling factor for the filter.

    Output:
        X (np.array): Generated signal matrix (size N x P).
        g (np.array): Generated filter (size N x 1).
        Z (np.array): Noisy signal (size N x P).
        Y (np.array): Noisy measurements (size N x P).
    """
    # Generate filter g
    g = np.ones(N) + alpha * np.random.uniform(0, 1, N)
    g = N * g / sum(g)  # Normalize the filter
    h = 1. / g  # h is the inverse of g (tilde version)
    
    # Compute H (filter matrix) using V and h
    H = np.dot(V, np.dot(np.diag(h), np.transpose(V)))
    
    # Generate the signal X
    X = X_generate(N, P, int(N * theta))  # Signal generation with N * theta non-zero entries
    
    # Compute the noisy measurements Y
    Y = np.dot(H, X)
    
    # Generate noisy signal Z using the Kronecker product
    Z = linalg.khatri_rao(np.dot(np.transpose(Y), V), V)
    
    return X, g, Z, Y

################################
####### DATA GENERATION ########
################################

def data_generate(N,P,V,theta,alpha):
    # N: Number of nodes in the graph
    # P: Number of features
    # V: Eigenvectors for graph signal processing
    # theta: Fraction of nodes for signal generation
    # alpha: Scaling factor for random noise
    # This function generates the input data (X), filter g, and the output signal Z and Y.

    g = np.ones(N) + alpha * np.random.uniform(0, 1, N)  # Generate random filter g with added noise
    g = N * g / sum(g)  # Normalize g to sum to N
    h = 1. / g  # h is the inverse of g, used in filtering
    H = np.dot(V, np.dot(np.diag(h), np.transpose(V)))  # H is the diagonal filter matrix

    # Generate the signal X based on the size of N, P, and S (fraction of N as signal nodes)
    X = X_generate(N, P, S)

    # Output Y is the filtered signal X
    Y = np.dot(H, X)

    # Z is the result of a matrix product with the transpose of Y and V
    Z = linalg.khatri_rao(np.dot(np.transpose(Y), V), V)
    
    return X, g, Z, Y  # Return the generated data

def Xdata_generate(N,P,V,g,theta,alpha):
    # Similar to data_generate but focuses on generating X and Y based on g and theta

    S = int(N * theta)  # Number of signal nodes
    h = 1. / g  # Calculate the inverse of g
    H = np.dot(V, np.dot(np.diag(h), np.transpose(V)))  # Filter matrix H

    # Generate the signal X
    X = X_generate(N, P, S)

    # Output Y is the filtered signal X
    Y = np.dot(H, X)

    return X, Y  # Return the generated data

def Xsdata_generate(N,P,V,g,theta,alpha):
    # Similar to Xdata_generate, this function also generates X and Y
    
    S = int(N * theta)  # Number of signal nodes
    h = 1. / g  # Calculate the inverse of g
    H = np.dot(V, np.dot(np.diag(h), np.transpose(V)))  # Filter matrix H

    # Generate the signal X
    X = X_generate(N, P, S)

    # Output Y is the filtered signal X
    Y = np.dot(H, X)

    return X, Y  # Return the generated data

def Xdata_generate_v2(N,P,V,gs,theta,alpha):
    # Generate data based on multiple filters (gs) for each feature

    hs = 1. / gs  # Calculate the inverse of each filter
    X = X_generate(N, P, S)  # Generate the signal X
    Y = []

    # For each feature, apply a filter and generate the output Y
    for p in range(P):
        x = X[:, p]
        h_p = hs[:, p]
        H = np.dot(V, np.dot(np.diag(h_p), np.transpose(V)))  # Filter matrix H
        y = np.dot(H, x)
        y = y.reshape([N, 1])  # Reshape the result to match dimensions of Y
        if p == 0:
            Y = y
        else:
            Y = np.concatenate((Y, y), axis=1)  # Concatenate along columns for multiple features

    return X, Y  # Return the generated data

def g_generate_gso(N, alpha, eigenvalues, L):
    # Generate the filter g using Graph Signal Operator (GSO)

    Vd = np.vander(eigenvalues, L)  # Create the Vandermonde matrix from eigenvalues
    Vd = np.fliplr(Vd)  # Flip the matrix
    g = alpha * np.random.normal(0, 1, L)  # Generate random filter g with Gaussian noise
    g[0] = 1  # Set the first element of g to 1
    g_tilde = np.dot(Vd, g)  # Apply the Vandermonde transformation to g
    g_tilde = N * g_tilde / sum(g_tilde)  # Normalize the filter
    return g_tilde  # Return the generated filter

def h_generate_gso(N, alpha, eigenvalues, L):
    # Generate the filter h using Graph Signal Operator (GSO)

    Vd = np.vander(eigenvalues, L)  # Create the Vandermonde matrix from eigenvalues
    Vd = np.fliplr(Vd)  # Flip the matrix
    h = alpha * np.random.normal(0, 1, L)  # Generate random filter h with Gaussian noise
    h[0] = 1  # Set the first element of h to 1
    h_tilde = np.dot(Vd, h)  # Apply the Vandermonde transformation to h
    h_tilde = N * h_tilde / sum(h_tilde)  # Normalize the filter
    g_tilde = 1 / h_tilde  # Compute the inverse filter
    return g_tilde  # Return the inverse filter

def wt_generate_gso(N, alpha, eigenvalues, tMax):
    # Generate the filter wt using Graph Signal Operator (GSO)

    Vd = np.vander(eigenvalues, tMax)  # Create the Vandermonde matrix from eigenvalues
    Vd = np.fliplr(Vd)  # Flip the matrix
    t = np.random.randint(tMax, size=1)  # Randomly choose a time index
    h = np.zeros(tMax)  # Initialize the filter
    h[t] = 1  # Set the chosen time index to 1

    h_tilde = np.dot(Vd, h)  # Apply the Vandermonde transformation to h
    h_tilde = N * h_tilde / sum(h_tilde)  # Normalize the filter
    g_tilde = 1 / h_tilde  # Compute the inverse filter
    return g_tilde  # Return the inverse filter

def g_batch_generate(N, nBatches, alpha, **kwargs):
    # Generate a batch of filters g based on the provided parameters

    # Extract parameters from kwargs or set defaults
    Phi = kwargs.get('Phi', None)
    L = kwargs.get('L', None)
    tMax = kwargs.get('tMax', N)
    C = kwargs.get('C', N)
    filterType = kwargs.get('filterType', 'g')

    if filterType == 'h':
        # Generate h filters if specified
        print('(g_batch_generate) Generating h filter')
        h = alpha * np.random.normal(0, 1, [L, nBatches])  # Generate random h filters
        h[0, :] = 1  # Set the first row to 1
        Vd = Phi[:, 0:L]
        h_batch = np.dot(Vd, h)  # Apply Vandermonde transformation
        g_batch = 1. / h_batch  # Compute inverse filters
        for p in range(nBatches):
            g_batch[:, p] = C * g_batch[:, p] / sum(g_batch[:, p])  # Normalize the filters
        print(g_batch[:, 0])  # Print the first filter
    elif filterType == 'wt':
        # Generate wt filters if specified
        print('(g_batch_generate) Generating wt filter')
        Vd = Phi[:, 0:tMax]
        t = np.random.randint(tMax, size=(nBatches))
        h = np.zeros([tMax, nBatches])
        for n in range(nBatches):
            h[t[n], n] = 1
        h_batch = np.dot(Vd, h)  # Apply Vandermonde transformation
        g_batch = 1. / h_batch  # Compute inverse filters
        for p in range(nBatches):
            g_batch[:, p] = C * g_batch[:, p] / sum(g_batch[:, p])  # Normalize the filters
        print(g_batch[:, 0])  # Print the first filter
    else:
        # Generate standard g filters if no specific filter type is provided
        print('(g_batch_generate) Generating g filter')
        g_batch = np.ones([N, nBatches]) + alpha * np.random.normal(0, 1, [N, nBatches])
        for p in range(nBatches):
            g_batch[:, p] = N * g_batch[:, p] / sum(g_batch[:, p])  # Normalize the filters

    return g_batch  # Return the generated batch of filters

def g_batch_generate_gso(N, nBatches, alpha, eigenvalues, L):
    # Generate a batch of filters g using Graph Signal Operator (GSO)

    Vd = np.vander(eigenvalues, L)  # Create the Vandermonde matrix from eigenvalues
    Vd = np.fliplr(Vd)  # Flip the matrix
    g = alpha * np.random.normal(0, 1, [L, nBatches])  # Generate random g filters
    e1 = np.zeros([L, nBatches])
    e1[0, :] = 1  # Set the first row to 1
    g = g + e1  # Add a bias term to the filters
    g_batch = np.dot(Vd, g)  # Apply the Vandermonde transformation
    for p in range(nBatches):
        g_batch[:, p] = N * g_batch[:, p] / sum(g_batch[:, p])  # Normalize the filters
    return g_batch  # Return the batch of filters

def h_batch_generate_gso(N, nBatches, alpha, eigenvalues, L):
    # Generate a batch of filters h using Graph Signal Operator (GSO)

    Vd = np.vander(eigenvalues, L)  # Create the Vandermonde matrix from eigenvalues
    Vd = np.fliplr(Vd)  # Flip the matrix
    h = alpha * np.random.normal(0, 1, [L, nBatches])  # Generate random h filters
    e1 = np.zeros([L, nBatches])
    e1[0, :] = 1  # Set the first row to 1
    h = h + e1  # Add a bias term to the filters
    h_batch = np.dot(Vd, h)  # Apply the Vandermonde transformation
    g_batch = 1. / h_batch  # Compute the inverse filters
    for p in range(nBatches):
        g_batch[:, p] = N * g_batch[:, p] / sum(g_batch[:, p])  # Normalize the filters
    return g_batch  # Return the batch of filters

def wt_batch_generate_gso(N, nBatches, alpha, eigenvalues, tMax):
    # Generate a batch of wt filters using Graph Signal Operator (GSO)

    Vd = np.vander(eigenvalues, tMax)  # Create the Vandermonde matrix from eigenvalues
    Vd = np.fliplr(Vd)  # Flip the matrix
    t = np.random.randint(tMax, size=(nBatches))  # Randomly choose time indices
    h = np.zeros([tMax, nBatches])  # Initialize the filter
    for n in range(nBatches):
        h[t[n], n] = 1  # Set the chosen time index to 1
    h_batch = np.dot(Vd, h)  # Apply the Vandermonde transformation
    g_batch = 1. / h_batch  # Compute the inverse filters
    for p in range(nBatches):
        g_batch[:, p] = N * g_batch[:, p] / sum(g_batch[:, p])  # Normalize the filters
    return g_batch  # Return the batch of filters

def generate_normalized_gso_laplaciant(N, p):
    """
    This function generates a normalized Graph Shift Operator (GSO) Laplacian matrix with a given size N and connection probability p.

    Parameters:
    N : int
        The size of the graph (number of nodes).
    p : float
        The probability of an edge between nodes.

    Returns:
    gso : numpy.ndarray
        The normalized GSO Laplacian matrix.
    d : numpy.ndarray
        The degree vector (diagonal elements of the degree matrix).
    Lpn : numpy.ndarray
        The normalized Laplacian matrix.
    eigenvalues : numpy.ndarray
        The eigenvalues of the normalized adjacency matrix.
    V : numpy.ndarray
        The eigenvectors of the normalized adjacency matrix.
    """
    connected_nonperm = 0
    temp_count_perm_pairs = 10
    while temp_count_perm_pairs > 1e-10:
        temp_connect = 0
        while temp_connect < 1e-10:
            # Generate a random matrix to determine edge connections.
            random = np.random.random((N, N))
            tri = np.tri(N, k=-1)  # Generate a lower triangular matrix.
            # Initialize adjacency matrix with zeros
            gso = np.zeros((N, N))
            # Assign intra-community edges based on probability p
            gso[np.logical_and(tri, random < p)] = 1
            gso += gso.T  # Make the graph undirected by adding the transpose
            # Degree vector (sum of each row in the adjacency matrix)
            d = np.sum(gso, axis=1)
            temp_di_mutiply = 1
            for i in range(N):
                temp_di_mutiply = temp_di_mutiply * d[i]
            # Check if the graph is connected (non-zero degree product)
            if abs(temp_di_mutiply) > 1e-10:
                temp_connect = 1
        # Calculate Laplacian matrix (Degree matrix - Adjacency matrix)
        Lp = np.diag(d) - gso
        Lpn = np.zeros((N, N))  # Normalized Laplacian
        An = np.zeros((N, N))  # Normalized adjacency matrix
        for i in range(N):
            for j in range(N):
                # Normalize the Laplacian and adjacency matrix
                Lpn[i, j] = Lp[i, j] / np.sqrt(d[i]) / np.sqrt(d[j])
                An[i, j] = gso[i, j] / np.sqrt(d[i]) / np.sqrt(d[j])
        # Eigen decomposition of the normalized adjacency matrix
        eigenvalues, V = np.linalg.eig(An)        
        # Permutation check to ensure uniqueness
        temp_count_perm_pairs = 0
        for k in range(N):
            VV = np.outer(V[:, k], V[:, k])
            temp_non_zeros_entry_count = 0
            # Count non-zero entries in the outer product of eigenvectors
            for l1 in range(N):
                for l2 in range(N):
                    if abs(VV[l1, l2]) > 1e-5:
                        temp_non_zeros_entry_count += 1
            # If the number of non-zero entries is too low, increment permutation count
            if temp_non_zeros_entry_count < 5:
                temp_count_perm_pairs += 1
    return gso, d, Lpn, eigenvalues, V


########################################################
############### Functions ##############################
########################################################

def fast_inverse(A, rho):
    """
    Efficiently computes the inverse of a matrix A with regularization parameter rho.

    Parameters:
    A : torch.Tensor
        The square matrix to be inverted.
    rho : float
        The regularization parameter.

    Returns:
    torch.Tensor
        The inverse of matrix A.
    """
    N = A.shape[0]
    a = torch.diagonal(A, 0)  # Extract diagonal elements
    sum_a = sum(a)
    a_1 = (1 / a).reshape(N, 1)
    A_1 = torch.diag(a_1.reshape(N))  # Create diagonal matrix of 1/a
    return A_1 - (rho / (1 + rho * sum_a)) * torch.matmul(a_1, a_1.transpose(0, 1))

def fast_inverse_objf3(A, M, rho):
    """
    Fast inverse computation for objective function 3 with matrix M and regularization parameter rho.

    Parameters:
    A : torch.Tensor
        The square matrix to be inverted.
    M : torch.Tensor
        The matrix involved in the objective function.
    rho : float
        The regularization parameter.

    Returns:
    torch.Tensor
        The inverse of matrix A after considering the regularization and M.
    """
    N = A.shape[0]
    q = M.shape[1]
    a = torch.diagonal(A, 0)  # Extract diagonal elements
    sum_a = sum(a)
    a_1 = (1 / a).reshape(N, 1)
    A_1 = torch.diag(a_1.reshape(N))
    A1M = torch.matmul(A_1, M)
    # Compute core matrix for fast inverse
    core_matrix = torch.eye(q) + rho * torch.matmul(M.transpose(0, 1), A1M)
    return A_1 - rho * torch.matmul(torch.matmul(A1M, torch.inverse(core_matrix)), A1M.transpose(0, 1))

def fast_inverse_no_constrain(A):
    """
    Computes the inverse of matrix A without any constraints (just diagonal inverse).

    Parameters:
    A : torch.Tensor
        The square matrix to be inverted.

    Returns:
    torch.Tensor
        The inverse of matrix A.
    """
    N = A.shape[0]
    a = torch.diagonal(A, 0)
    A_1 = torch.diag(1. / a)
    return A_1


def min_RE(x_pre, x_true):
    """
    Computes the relative error between two vectors x_pre and x_true.

    Parameters:
    x_pre : numpy.ndarray
        The predicted vector.
    x_true : numpy.ndarray
        The true vector.

    Returns:
    RE : float
        The relative error between the two vectors.
    sign : bool
        A flag indicating if the relative error is based on x_pre - x_true or x_pre + x_true.
    """
    re_1 = LA.norm(x_pre - x_true) / LA.norm(x_true)
    re_2 = LA.norm(x_pre + x_true) / LA.norm(x_true)
    RE = min(re_1, re_2)
    sign = re_1 < re_2
    return RE, sign

def generate_V(N):
    """
    Generates an orthogonal matrix V of size NxN.

    Parameters:
    N : int
        The size of the matrix.

    Returns:
    Vout : torch.Tensor
        An NxN orthogonal matrix.
    """
    V = torch.randn([N, N])  # Generate a random matrix
    Vout = torch.zeros([N, N])
    Vout[0, :] = V[0, :] / torch.norm(V[0, :])
    for i in range(1, N):
        Vout[i, :] = V[i, :]
        for j in range(0, i):
            Vout[i, :] = Vout[i, :] - torch.dot(Vout[i, :], Vout[j, :]) * Vout[j, :]
        Vout[i, :] = Vout[i, :] / torch.norm(Vout[i, :])  # Normalize each row
    return Vout


# Function to generate signals for each community
def community_LabelsToNodeSets(communityLabels, gso, N_C, **kwargs):
    """
    Generates node sets for each community based on the community labels and graph structure.

    Input:
        communityLabels (np.array): Community labels for each node.
        gso (np.array): Adjacency matrix of the graph.
        N_C (int): Number of nodes to sample per community.

    Output:
        result (dict): Dictionary containing source nodes for each community, and community node list.
    """
    # Default mode is random sampling
    if 'mode' in kwargs.keys():
        mode = kwargs['mode']
    else:
        mode = 'random'
    
    N = len(communityLabels)
    A = abs(gso) > 1e-5  # Adjacency matrix
    nClass = np.max(communityLabels) + 1  # Number of communities
    sourceNodes = []
    degree = np.sum(A, axis=0)  # Degree of each node
    
    communityList = []
    for c in range(nClass):
        # Get the nodes belonging to the current community
        communityNodes = np.nonzero(communityLabels == c)[0]
        degreeSorted = np.argsort(degree[communityNodes])
        
        if mode == 'random':
            np.random.shuffle(degreeSorted)
        
        # Select top N_C nodes from the sorted degree list
        sourceNodes.append(communityNodes[degreeSorted[-N_C:]])
        communityList.append(communityNodes)
    
    # Create the result dictionary
    result = {
        'sourceNodes': sourceNodes,
        'communityList': communityList,
        'nClass': nClass
    }
    
    return result

def X_generate_fromSBM(N,P,S,communityLabels,gso,**kwargs):
    ## Generating full rank X matrix
    # communityLabels: size(N,), mapping node to class/community
    # gso: adjacency matrix, representing the graph structure
    # N: Number of nodes
    # P: Number of signals/features to generate
    # S: Number of source nodes per signal
    if 'selectMode' in kwargs.keys():
        selectMode = kwargs['selectMode']  # Mode for selecting source nodes (default or alternative)
    else:
        selectMode = 'default'
    
    if 'signalMode' in kwargs.keys():
        signalMode = kwargs['signalMode']  # Mode for generating signal values (default or random noise)
    else:
        signalMode = 'default'
    
    # Initialize X0 as a zero matrix of size (N, P)
    X0 = np.zeros([N,P])
    
    # Convert community labels and graph structure to a node set format
    temp_result = community_LabelsToNodeSets(communityLabels, gso, S)
    sourceNodes = temp_result['sourceNodes']  # List of source nodes
    communityNodeList = temp_result['communityList']  # Node list for each community
    nClass = temp_result['nClass']  # Number of communities/classes
    sourceNode_set = np.array(sourceNodes)  # Convert source nodes to a numpy array

    # Randomly select P indices from the list of communities
    sampledIndicesList = np.random.choice(np.arange(nClass), size=P)
    sampledSources = sourceNode_set[sampledIndicesList]  # Selected source nodes

    if selectMode == 'default':
        # In 'default' mode, fix the source nodes for each signal
        for p in range(P):
            x0 = torch.zeros(N)  # Initialize a zero vector for each signal
            for s in range(S):
                if signalMode == 'default':
                    # In 'default' signal mode, set the source node to 1
                    x0[sampledSources[p,s]] = 1
                else:
                    # In alternative signal mode, generate random values for the source nodes
                    x0[sampledSources[p,s]] = np.random.randn()
            X0[:,p] = x0  # Store the signal in the corresponding column of X0
    else:
        # In alternative mode, shuffle the source nodes for each signal
        for p in range(P):
            x0 = torch.zeros(N)  # Initialize a zero vector for each signal
            sampledSourceIndex = sampledIndicesList[p]  # Select a community index
            nodeSet_of_sampledSource = communityNodeList[sampledSourceIndex]  # Get the node list for the selected community
            np.random.shuffle(nodeSet_of_sampledSource)  # Shuffle the nodes within the community
            for s in range(S):
                if signalMode == 'default':
                    # In 'default' signal mode, set the selected node to 1
                    x0[nodeSet_of_sampledSource[s]] = 1
                else:
                    # In alternative signal mode, generate random values for the selected nodes
                    x0[nodeSet_of_sampledSource[s]] = np.random.randn()
            X0[:,p] = x0  # Store the signal in the corresponding column of X0

    result = {}
    result['X0'] = X0  # Return the generated signal matrix
    result['sampledIndicesList'] = sampledIndicesList  # Return the list of sampled community indices
    return result

def softshrink(x, lambd):
    """
    Soft thresholding function to enforce sparsity.
    It shrinks small values towards zero, effectively setting values within [-lambd, lambd] to zero.

    Args:
        x: The input tensor.
        lambd: The threshold value for soft thresholding.

    Returns:
        out: The tensor after applying the soft thresholding.
    """
    mask1 = x > lambd  # Mask for values greater than lambd
    mask2 = x < -lambd  # Mask for values less than -lambd
    out = torch.zeros_like(x)  # Initialize output tensor of the same size as x
    out += mask1.float() * -lambd + mask1.float() * x  # Apply soft thresholding for positive values
    out += mask2.float() * lambd + mask2.float() * x  # Apply soft thresholding for negative values
    return out

def to_torch(x, **kwargs):
    """
    Converts a numpy array to a PyTorch tensor.
    
    Args:
        x: The input numpy array.
        kwargs: Additional keyword arguments, such as 'requires_grad' (default: False).
    
    Returns:
        torch.tensor: The PyTorch tensor version of the input array.
    """
    thisShape = x.shape  # Get the shape of the input array
    dataType = type(x)  # Get the data type of the input
    
    if 'requires_grad' in kwargs.keys():
        requires_grad = kwargs['requires_grad']  # Check if gradients are required
    else:
        requires_grad = False  # Default to no gradient tracking
    
    if 'numpy' in repr(dataType):  # Check if the input is a numpy array
        return torch.tensor(x)  # Convert numpy array to torch tensor
    return x  # If it's already a tensor, return as-is

def to_numpy(x, **kwargs):
    """
    Converts a PyTorch tensor to a numpy array.
    
    Args:
        x: The input PyTorch tensor.
        kwargs: Additional keyword arguments, such as 'requires_grad' (default: False).
    
    Returns:
        numpy.ndarray: The numpy array version of the input tensor.
    """
    thisShape = x.shape  # Get the shape of the input tensor
    dataType = type(x)  # Get the data type of the input
    
    if 'requires_grad' in kwargs.keys():
        requires_grad = kwargs['requires_grad']  # Check if gradients are required
    else:
        requires_grad = False  # Default to no gradient tracking
        
    if 'torch' in repr(dataType):  # Check if the input is a torch tensor
        if requires_grad == False:
            x1 = x.clone().detach().requires_grad_(False)  # Detach from the computation graph if no gradient is required
            return x1.numpy()  # Convert tensor to numpy array
        else:
            return x.numpy()  # If gradients are required, return as numpy array
    return x  # If it's already a numpy array, return as-is

################################
############# ADMM #############
################################

def admm(Y,V,rho_0,eta_0,C,N_ite,max_re):
    """
    ADMM optimization algorithm for solving a specific optimization problem.

    Args:
        Y: Input matrix (e.g., measurements or observations).
        V: Another matrix involved in the factorization.
        rho_0: Regularization parameter.
        eta_0: Step size for updates.
        C: Constant used in the ADMM update step.
        N_ite: Maximum number of iterations.
        max_re: Convergence threshold.
    
    Returns:
        x: Optimized signal vector.
        v: Another solution vector.
        n_ite: Number of iterations performed.
        max_re_matched: Indicates if convergence is reached.
    """
    N = V.shape[0]  # Number of rows in matrix V
    P = Y.shape[1]  # Number of columns in matrix Y
    
    # Compute the Kronecker product of Y.T and V, and store in Z
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y), V), V), requires_grad=False)
    
    # Initialize variables
    v = torch.tensor(np.zeros(N))  # Initialize vector v (N elements)
    x = torch.tensor(np.zeros(N * P))  # Initialize vector x (N * P elements)
    u = torch.tensor(np.zeros(N * P))  # Initialize vector u (N * P elements)
    eta = torch.tensor(np.zeros(1))  # Initialize eta
    II = torch.tensor(np.ones([N, N]))  # Identity matrix of size N
    In = torch.tensor(np.ones(N))  # Vector of ones (size N)
    
    n_ite = 0  # Iteration counter
    max_re_matched = 0  # Flag for convergence
    
    while n_ite < N_ite and max_re_matched == 0:
        # Store old values of v and x for convergence check
        v_old = v
        x_old = x
        
        # Compute the inverse of (Z^T Z) and the update for v
        ZIk_inv = fast_inverse(torch.matmul(torch.transpose(Z, 0, 1), Z), eta_0 / rho_0) / rho_0
        v_temp = torch.matmul(torch.transpose(Z, 0, 1), rho_0 * x - u) + (eta_0 * C - eta) * In
        v = torch.matmul(ZIk_inv, v_temp)  # Update v
        
        # Apply soft thresholding to x
        X_update = torch.nn.Softshrink(lambd=1 / rho_0)  # Soft thresholding operator
        x = X_update(torch.matmul(Z, v) + 1 / rho_0 * u)  # Update x
        
        # Update u and eta for the next iteration
        u = u + rho_0 * (torch.matmul(Z, v) - x)
        eta = eta + eta_0 * (torch.matmul(torch.tensor(np.ones([1, N])), v) - C)
        
        # Compute the relative error for convergence check
        re = ((v - v_old) ** 2).mean() / (1e-10 + ((v_old) ** 2).mean())
        if re < max_re ** 2:
            max_re_matched = 1  # Convergence reached
        
        n_ite += 1  # Increment iteration counter
    
    # Return the results: optimized signal vector x, solution vector v, number of iterations, and convergence status
    return x, v, n_ite, max_re_matched

def X_generate(N,P,S):
## Generating full rank X
    miu = 0
    sigma = 1.0
    R = abs(np.random.normal(miu,sigma,[S,P]))
    sign = (np.random.binomial(1,0.5,[S,P])-0.5)*2
    X = np.multiply(R,sign)
    X0 = np.zeros([N-S, P])
    X = np.concatenate([X, X0]) # Pad zeros to generate the full signals 
    for p in range(0,P):
        temp_X = X[:,p]
#         idx = np.arange(N)
        Randindex =  np.random.permutation(N)
        X[:,p] = temp_X[Randindex]
    return X
