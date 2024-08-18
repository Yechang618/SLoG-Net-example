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
    sizes = x_pre.size()
#     return torch.min(torch.norm(x_pre-x_ture)**2,torch.norm(x_pre+x_true)**2)
    return torch.min(((x_pre-x_ture)**2).mean(),((x_pre+x_ture)**2).mean())
    


    
########################################################
############### Save and Load ##########################
########################################################

def writeVarValues(fileToWrite, varValues):
    """
    Write the value of several string variables specified by a dictionary into
    the designated .txt file.
    
    Input:
        fileToWrite (os.path): text file to save the specified variables
        varValues (dictionary): values to save in the text file. They are
            saved in the format "key = value".
    """
    with open(fileToWrite, 'a+') as file:
        for key in varValues.keys():
            file.write('%s = %s\n' % (key, varValues[key]))
        file.write('\n')

def saveSeed(randomStates, saveDir):
    """
    Takes a list of dictionaries of random generator states of different modules
    and saves them in a .pkl format.
    
    Inputs:
        randomStates (list): The length of this list is equal to the number of
            modules whose states want to be saved (torch, numpy, etc.). Each
            element in this list is a dictionary. The dictionary has three keys:
            'module' with the name of the module in string format ('numpy' or
            'torch', for example), 'state' with the saved generator state and,
            if corresponds, 'seed' with the specific seed for the generator
            (note that torch has both state and seed, but numpy only has state)
        saveDir (path): where to save the seed, it will be saved under the 
            filename 'randomSeedUsed.pkl'
    """
    pathToSeed = os.path.join(saveDir, 'randomSeedUsed.pkl')
    with open(pathToSeed, 'wb') as seedFile:
        pickle.dump({'randomStates': randomStates}, seedFile)
        
def loadSeed(loadDir):
    """
    Loads the states and seed saved in a specified path
    
    Inputs:
        loadDir (path): where to look for thee seed to load; it is expected that
            the appropriate file within loadDir is named 'randomSeedUsed.pkl'
    
    Obs.: The file 'randomSeedUsed.pkl' should contain a list structured as
        follows. The length of this list is equal to the number of modules whose
        states were saved (torch, numpy, etc.). Each element in this list is a
        dictionary. The dictionary has three keys: 'module' with the name of 
        the module in string format ('numpy' or 'torch', for example), 'state' 
        with the saved generator state and, if corresponds, 'seed' with the 
        specific seed for the generator (note that torch has both state and 
        seed, but numpy only has state)
    """
    pathToSeed = os.path.join(loadDir, 'randomSeedUsed.pkl')
    with open(pathToSeed, 'rb') as seedFile:
        randomStates = pickle.load(seedFile)
        randomStates = randomStates['randomStates']
    for module in randomStates:
        thisModule = module['module']
        if thisModule == 'numpy':
            np.random.RandomState().set_state(module['state'])
        elif thisModule == 'torch':
            torch.set_rng_state(module['state'])
            torch.manual_seed(module['seed'])        

########################################################
############### Graph Tools ############################
########################################################
        
# def computeSourceNodes_slog(A, C, N_C, mode = 'default',**kwargs):
#     """
#     computeSourceNodes: compute source nodes for the source localization problem
    
#     Input:
#         A (np.array): adjacency matrix of shape N x N
#         C (int): number of classes
        
#     Output:
#         sourceNodes (list): contains the indices of the C source nodes
        
#     Uses the adjacency matrix to compute C communities by means of spectral 
#     clustering, and then selects the node with largest degree within each 
#     community
#     """
#     if 'save_dir' in kwargs.keys():
#         save_dir = kwargs['save_dir']
#         save_to_local = True
#     else:
#         save_dir = None
#         save_to_local = False       
#     if 'save_dir_dropBox' in kwargs.keys():
#         save_dir_dropBox = kwargs['save_dir_dropBox']
#         save_to_dropBox = True
#     else:
#         save_dir_dropBox = None
#         save_to_dropBox = False    
#     if 'graphType' in kwargs.keys():
#         graphType = kwargs['graphType']
#     else:
#         graphType = 'SBM'
                        
#     sourceNodes = []
#     degree = np.sum(A, axis = 0) # degree of each vector
#     # Compute communities
#     communityClusters = SpectralClustering(n_clusters = C,
#                                            affinity = 'precomputed',
#                                            assign_labels = 'discretize')
#     communityClusters = communityClusters.fit(A)
#     communityLabels = communityClusters.labels_
#     communityList = []
#     # For each community
#     for c in range(C):
#         communityNodes = np.nonzero(communityLabels == c)[0]
#         degreeSorted = np.argsort(degree[communityNodes])
#         print('C = ',C,', Community nodes:', communityNodes)
#         print('Degress of community nodes:', degree[communityNodes])
#         print('Sorted degress:', degreeSorted)
#         if mode == 'random':
#             np.random.shuffle(degreeSorted)
#             print('Randomized sorted degress:', degreeSorted)
#         sourceNodes.append(communityNodes[degreeSorted[-N_C:]])
#         communityList.append(communityNodes)
#     ## Save the communityList to local or dropBox
#     cLabels_save_name = 'cLabels-' + graphType + '.npy'
#     if save_to_local == True:
#         cLabels_save_dir = os.path.join(save_dir, cLabels_save_name) 
#         np.save(cLabels_save_dir, communityLabels)
#         print('Saved to', cLabels_save_dir)
#     print('Source nodes:', sourceNodes)
#     print('Community Labels:', communityLabels)
#     print('Community List:', communityList)
    
#     return sourceNodes, communityLabels,communityList

def get_eig_normalized_adj(gso):
    # Input should be numpy tensor
    N,_ = np.shape(gso)
    d = np.sum(gso, axis=1)
    Lp = np.diag(d) - gso
    Lpn = np.zeros((N,N))
    An = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            Lpn[i,j] = Lp[i,j]/np.sqrt(d[i])/np.sqrt(d[j])
            An[i,j] = gso[i,j]/np.sqrt(d[i])/np.sqrt(d[j])
    eigenvalues, V = np.linalg.eig(An)  
    V = np.real(V)
    eigenvalues = np.real(eigenvalues)
    return d,An, eigenvalues, V  

################################
####### Graph GENERATION #######
################################
class Graph():
    """
    Graph: class to handle a graph with several of its properties

    Initialization:

        graphType (string): 'SBM', 'SmallWorld', 'fuseEdges', and 'adjacency'
        N (int): number of nodes
        [optionalArguments]: related to the specific type of graph; see
            createGraph() for details.

    Attributes:

        .N (int): number of nodes
        .M (int): number of edges
        .W (np.array): weighted adjacency matrix
        .D (np.array): degree matrix
        .A (np.array): unweighted adjacency matrix
        .L (np.array): Laplacian matrix (if graph is undirected and has no
           self-loops)
        .S (np.array): graph shift operator (weighted adjacency matrix by
           default)
        .E (np.array): eigenvalue (diag) matrix (graph frequency coefficients)
        .V (np.array): eigenvector matrix (graph frequency basis)
        .undirected (bool): True if the graph is undirected
        .selfLoops (bool): True if the graph has self-loops

    Methods:
    
        .computeGFT(): computes the GFT of the existing stored GSO and stores
            it internally in self.V and self.E (if this is never called, the
            corresponding attributes are set to None)

        .setGSO(S, GFT = 'no'): sets a new GSO
        Inputs:
            S (np.array): new GSO matrix (has to have the same number of nodes),
                updates attribute .S
            GFT ('no', 'increasing' or 'totalVariation'): order of
                eigendecomposition; if 'no', no eigendecomposition is made, and
                the attributes .V and .E are set to None
    """
    # in this class we provide, easily as attributes, the basic notions of
    # a graph. This serve as a building block for more complex notions as well.
    def __init__(self, graphType, N, graphOptions, **kwargs):
        assert N > 0
        
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

        #\\\ Create the graph (Outputs adjacency matrix):
        if self.load_from_local == True:
            load_name = 'gso-' + graphType + '.npy'
            load_dir = os.path.join(self.load_dir, load_name) 
            self.W = np.load(load_dir)
        else:
            self.W = createGraph(graphType, N, graphOptions)
        # TODO: Let's start easy: make it just an N x N matrix. We'll see later
        # the rest of the things just as handling multiple features and stuff.
        #\\\ Number of nodes:
        self.N = (self.W).shape[0]
        #\\\ Bool for graph being undirected:
        self.undirected = np.allclose(self.W, (self.W).T, atol = zeroTolerance)
        #   np.allclose() gives true if matrices W and W.T are the same up to
        #   atol.
        #\\\ Bool for graph having self-loops:
        self.selfLoops = True \
                        if np.sum(np.abs(np.diag(self.W)) > zeroTolerance) > 0 \
                        else False
        #\\\ Degree matrix:
        self.D = np.diag(np.sum(self.W, axis = 1))
        #\\\ Number of edges:
        self.M = int(np.sum(np.triu(self.W)) if self.undirected \
                                                    else np.sum(self.W))
        #\\\ Unweighted adjacency:
        self.A = (np.abs(self.W) > 0).astype(self.W.dtype)
        #\\\ Laplacian matrix:
        #   Only if the graph is undirected and has no self-loops
        if self.undirected and not self.selfLoops:
            self.L = adjacencyToLaplacian(self.W)
        else:
            self.L = None
        #\\\ GSO (Graph Shift Operator):
        #   The weighted adjacency matrix by default
        self.S = self.W
        #\\\ GFT: Declare variables but do not compute it unless specifically
        # requested
        self.E = None # Eigenvalues
        self.V = None # Eigenvectors
        
        ## Save the gso to local or dropBox
        self.graph_save_name = 'gso-' + graphType + '.npy'
        if self.save_to_local == True:
            graph_save_dir = os.path.join(self.save_dir, self.graph_save_name) 
            np.save(graph_save_dir, self.A)

    def computeGFT(self):
        # Compute the GFT of the stored GSO
        if self.S is not None:
            #\\ GFT:
            #   Compute the eigenvalues (E) and eigenvectors (V)
            self.E, self.V = computeGFT(self.S, order = 'totalVariation')

    def setGSO(self, S, GFT = 'no'):
        # This simply sets a matrix as a new GSO. It has to have the same number
        # of nodes (otherwise, it's a different graph!) and it can or cannot
        # compute the GFT, depending on the options for GFT
        assert S.shape[0] == S.shape[1] == self.N
        assert GFT == 'no' or GFT == 'increasing' or GFT == 'totalVariation'
        # Set the new GSO
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
    # Check
    assert N >= 0

    if graphType == 'SBM':
        assert(len(graphOptions.keys())) == 3
        C = graphOptions['nCommunities'] # Number of communities
        assert int(C) == C # Check that the number of communities is an integer
        pii = graphOptions['probIntra'] # Intracommunity probability
        pij = graphOptions['probInter'] # Intercommunity probability
        assert 0 <= pii <= 1 # Check that they are valid probabilities
        assert 0 <= pij <= 1
        # We create the SBM as follows: we generate random numbers between
        # 0 and 1 and then we compare them elementwise to a matrix of the
        # same size of pii and pij to set some of them to one and other to
        # zero.
        # Let's start by creating the matrix of pii and pij.
        # First, we need to know how many numbers on each community.
        nNodesC = [N//C] * C # Number of nodes per community: floor division
        c = 0 # counter for community
        while sum(nNodesC) < N: # If there are still nodes to put in communities
        # do it one for each (balanced communities)
            nNodesC[c] = nNodesC[c] + 1
            c += 1
        # So now, the list nNodesC has how many nodes are on each community.
        # We proceed to build the probability matrix.
        # We create a zero matrix
        probMatrix = np.zeros([N,N])
        # And fill ones on the block diagonals following the number of nodes.
        # For this, we need the cumulative sum of the number of nodes
        nNodesCIndex = [0] + np.cumsum(nNodesC).tolist()
        # The zero is added because it is the first index
        for c in range(C):
            probMatrix[ nNodesCIndex[c] : nNodesCIndex[c+1] , \
                        nNodesCIndex[c] : nNodesCIndex[c+1] ] = \
                np.ones([nNodesC[c], nNodesC[c]])
        # The matrix probMatrix has one in the block diagonal, which should
        # have probabilities p_ii and 0 in the offdiagonal that should have
        # probabilities p_ij. So that
        probMatrix = pii * probMatrix + pij * (1 - probMatrix)
        # has pii in the intracommunity blocks and pij in the intercommunity
        # blocks.
        # Now we're finally ready to generate a connected graph
        connectedGraph = False
        while not connectedGraph:
            # Generate random matrix
            W = np.random.rand(N,N)
            W = (W < probMatrix).astype(np.float64)
            # This matrix will have a 1 if the element ij is less or equal than
            # p_ij, so that if p_ij = 0.8, then it will be 1 80% of the times
            # (on average).
            # We need to make it undirected and without self-loops, so keep the
            # upper triangular part after the main diagonal
            W = np.triu(W, 1)
            # And add it to the lower triangular part
            W = W + W.T
            # Now let's check that it is connected
            connectedGraph = isConnected(W)
    elif graphType == 'Random Geometric':
        assert 'distance' in graphOptions.keys()
        d = graphOptions['distance']
        connectedGraph = False        
        while not connectedGraph:        
            xy = random.rand(N,2)
            W = np.zeros([N,N])
            for n in range(N):
                x1 = xy[n]
                for m in range(N):
                    if n != m:
                        x2 = xy[m]
                        if LA.norm(x1-x2) <= d:
                            W[n,m] = 1
            connectedGraph = isConnected(W)                            
        assert W.shape[0] == W.shape[1] == N
        
    elif graphType == 'ER':
        if 'probIntra' in graphOptions.keys():
            p = graphOptions['probIntra']
        else:
            p = 0.3
        connectedGraph = False   
        perm_amb = True
        while perm_amb or not connectedGraph:        
            W = np.random.default_rng().uniform(0,1,[N,N])
            W = np.triu(W,k=1)
            W = W + W.T            
            W = (W < p)*1
            connectedGraph = isConnected(W)  
            perm_amb = perm_ambiguity_exam(W)
        assert W.shape[0] == W.shape[1] == N

    elif graphType == 'BA':
        if 'alpha' in graphOptions.keys():
            alpha = graphOptions['alpha']
        else:
            alpha = 1.0
        connectedGraph = False   
        perm_amb = True
        while perm_amb or not connectedGraph:        
            W = np.zeros([N,N])
            degree = np.zeros(N)
#             degree[0] = 1
            W[0,1] = 1
            W[1,0] = 1
            for n in range(2,N):       # New node
                degree = np.sum(W,axis=0)
                degree = degree**alpha
                degree_sum = np.sum(degree)
                degree_n = 0
                while degree_n < 1:
                    for m in range(0,n):
                        prob = degree[m]/degree_sum 
                        if random.rand() < prob:
                            degree[n] += 1
                            W[m,n] = 1
                            W[n,m] = 1
                    degree_n = degree[n]
#             W = W + W.T
            connectedGraph = isConnected(W)  
            perm_amb = perm_ambiguity_exam(W)
        assert W.shape[0] == W.shape[1] == N
    
    elif graphType == 'adjacency':
        assert 'adjacencyMatrix' in graphOptions.keys()
        W = graphOptions['adjacencyMatrix']
        assert W.shape[0] == W.shape[1] == N
         
            
    return W

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
    # Compute the degree vector
    d = np.sum(W, axis = 1)
    # And build the degree matrix
    D = np.diag(d)
    # Return the Laplacian
    return D - W

def perm_ambiguity_exam(W):
    perm_amb = False
    d,Lpn, eigenvalues, V = get_eig_normalized_adj(W)
    N = V.shape[0]
    
    for n in range(N):
        v = V[:,n]
        v0 = v/np.sqrt(2)
#         v1 = abs(v0 - 0.5) < zeroTolerance
#         v2 = abs(v0 + 0.5) < zeroTolerance
#         if True in v1:
#             if True in v2:
#                 print(v)
#                 perm_amb = True
        if np.abs(np.sum(v0)) < zeroTolerance:
#             print(v0)
#             print(np.sum(v0))
            perm_amb = True
#             print('Permutation ambiguity exists:')
#             print(v)
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
    # Compute the degree vector
    d = np.sum(W, axis = 1)
    # Invert the square root of the degree
    d = 1/np.sqrt(d)
    # And build the square root inverse degree matrix
    D = np.diag(d)
    # Return the Normalized Adjacency
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
    d = 1/np.sqrt(d)
    # And build the square root inverse degree matrix
    D = np.diag(d)
    # Return the Normalized Laplacian
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
    # Check the correct order input
    assert order == 'totalVariation' or order == 'no' or order == 'increasing'
    # Check the matrix is square
    assert S.shape[0] == S.shape[1]
    # Check if it is symmetric
    symmetric = np.allclose(S, S.T, atol = zeroTolerance)
    # Then, compute eigenvalues and eigenvectors
    if symmetric:
        e, V = np.linalg.eigh(S)
    else:
        e, V = np.linalg.eig(S)
    # Sort the eigenvalues by the desired error:
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
    isConnected: determine if a graph is connected

    Input:
        W (np.array): adjacency matrix

    Output:
        connected (bool): True if the graph is connected, False otherwise
    
    Obs.: If the graph is directed, we consider it is connected when there is
    at least one edge that would make it connected (i.e. if we drop the 
    direction of all edges, and just keep them as undirected, then the resulting
    graph would be connected).
    """
    undirected = np.allclose(W, W.T, atol = zeroTolerance)
    if not undirected:
        W = 0.5 * (W + W.T)
    L = adjacencyToLaplacian(W)
    E, V = computeGFT(L)
    e = np.diag(E) # only eigenvavlues
    # Check how many values are greater than zero:
    nComponents = np.sum(e < zeroTolerance) # Number of connected components
    if nComponents == 1:
        connected = True
    else:
        connected = False
    return connected

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
    # Compute the degree vector
    d = np.sum(W, axis = 1)
    # And build the degree matrix
    D = np.diag(d)
    # Return the Laplacian
    return D - W

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
    # Compute the degree vector
    d = np.sum(W, axis = 1)
    # Invert the square root of the degree
    d = 1/np.sqrt(d)
    # And build the square root inverse degree matrix
    D = np.diag(d)
    # Return the Normalized Adjacency
    return D @ W @ D

################################
####### DATA GENERATION ########
################################
def data_generate(N,P,V, theta,alpha):
#     N = 10; theta = 0.3; N_realiz = 1;  # X parameters
#     S = int(N*theta); P = 2;
#     alpha = 0.1; # Graph & Filter parameters
    # SNR = 0.01;p = 0.2; L = 5;
    # V = myModules.generate_V(N)
    g = np.ones(N)+alpha*np.random.uniform(0,1,N)
    g = N*g/sum(g)
    h = 1./g;# h is h tilde here.
    H = np.dot(V,np.dot(np.diag(h),np.transpose(V)))
    X = X_generate(N,P,S)
    Y = np.dot(H,X)
    Z = linalg.khatri_rao(np.dot(np.transpose(Y),V),V)
    return X,g,Z,Y

def Xdata_generate(N,P,V,g, theta,alpha):
    S = int(N*theta); #P = 2;
    h = 1./g;# h is h tilde here.
    H = np.dot(V,np.dot(np.diag(h),np.transpose(V)))
    X = X_generate(N,P,S)
    Y = np.dot(H,X)
    return X,Y

def Xsdata_generate(N,P,V,g, theta,alpha):
    S = int(N*theta); #P = 2;
    h = 1./g;# h is h tilde here.
    H = np.dot(V,np.dot(np.diag(h),np.transpose(V)))
    X = X_generate(N,P,S)
    Y = np.dot(H,X)
    return X,Y

def Xdata_generate_v2(N,P,V,gs, theta,alpha):
    hs = 1./gs;# h is h tilde here.
    X = X_generate(N,P,S)
    Y = []
    for p in range(P):
        x = X[:,p]
        h_p = hs[:,p] 
        H = np.dot(V,np.dot(np.diag(h_p),np.transpose(V)))
        y = np.dot(H,x)
        y = y.reshape([N,1])
        if p == 0:
            Y = y
        else:
            Y = np.concatenate((Y, y), axis=1)
    return X,Y

# def g_generate(N,P,alpha):
#     gs = np.ones([N,P])+alpha*np.random.uniform(0,1,[N,P])
#     for p in range(P):
#         gs[:,p] = N*gs[:,p]/sum(gs[:,p])
#     return gs

def g_generate_gso(N,alpha, eigenvalues,L):
    Vd = np.vander(eigenvalues,L)
    Vd = np.fliplr(Vd)
    g = alpha*np.random.normal(0, 1, L)
    g[0] = 1
    g_tilde = np.dot(Vd,g)
    g_tilde = N*g_tilde/sum(g_tilde)
    return g_tilde

def h_generate_gso(N,alpha, eigenvalues,L):
    Vd = np.vander(eigenvalues,L)
    Vd = np.fliplr(Vd)
    h = alpha*np.random.normal(0, 1, L)
    h[0] = 1
    h_tilde = np.dot(Vd,h)
    h_tilde = N*h_tilde/sum(h_tilde)
    g_tilde = 1/h_tilde    
    return g_tilde

def wt_generate_gso(N,alpha, eigenvalues,tMax):
    
    Vd = np.vander(eigenvalues,tMax)
    Vd = np.fliplr(Vd)
    t = np.random.randint(tMax, size = 1)
    h = np.zeros(tMax)
    h[t] = 1      

    h_tilde = np.dot(Vd,h)
    h_tilde = N*h_tilde/sum(h_tilde)
    g_tilde = 1/h_tilde    
    return g_tilde

def g_batch_generate(N,nBatches,alpha,**kwargs):
    
    if 'Phi' in kwargs.keys():
        Phi = kwargs['Phi']
    else:
        Phi = None
        
    if 'L' in kwargs.keys():
        L = kwargs['L']
    else:
        L = None
        
    if 'tMax' in kwargs.keys():
        tMax = kwargs['tMax']
    else:
        tMax = N        

    if 'C' in kwargs.keys():
        C = kwargs['C']
    else:
        C = N        
        
    if 'filterType' in kwargs.keys():
        filterType = kwargs['filterType']
    else:
        filterType = 'g'
        
    if filterType == 'h':
        print('(g_batch_generate) Generating h filter')
        h = alpha*np.random.normal(0, 1, [L,nBatches])
        h[0,:] = 1
#         print(h)
        Vd = Phi[:,0:L]
        h_batch = np.dot(Vd,h)
        g_batch = 1./h_batch
        for p in range(nBatches):
            g_batch[:,p] = C*g_batch[:,p]/sum(g_batch[:,p])
        print(g_batch[:,0])  
    elif filterType == 'wt':
        Vd = Phi[:,0:tMax]
        t = np.random.randint(tMax, size = (nBatches))
        h = np.zeros([tMax,nBatches])
        for n in range(nBatches):
            h[t[n],n] = 1        
        print('(g_batch_generate) Generating wt filter')
#         print(h)
        Vd = Phi[:,0:tMax]
        h_batch = np.dot(Vd,h)
        g_batch = 1./h_batch
        for p in range(nBatches):
            g_batch[:,p] = C*g_batch[:,p]/sum(g_batch[:,p])
        print(g_batch[:,0])        
    else:
        print('(g_batch_generate) Generating g filter')        
        g_batch = np.ones([N,nBatches])+alpha*np.random.normal(0, 1, [N,nBatches])
        for p in range(nBatches):
            g_batch[:,p] = N*g_batch[:,p]/sum(g_batch[:,p])
    return g_batch


def g_batch_generate_gso(N,nBatches,alpha, eigenvalues,L):
    Vd = np.vander(eigenvalues,L)
    Vd = np.fliplr(Vd)
    g = alpha*np.random.normal(0, 1, [L,nBatches])
    e1 = np.zeros([L,nBatches])
    e1[0,:] = 1
    g = g + e1
    g_batch = np.dot(Vd,g)
    for p in range(nBatches):
#         g_batch[:,p] = N*g_batch[:,p]/LA.norm(g_batch[:,p],1)
        g_batch[:,p] = N*g_batch[:,p]/sum(g_batch[:,p])
    return g_batch

def h_batch_generate_gso(N,nBatches,alpha, eigenvalues,L):
    Vd = np.vander(eigenvalues,L)
    Vd = np.fliplr(Vd)
    h = alpha*np.random.normal(0, 1, [L,nBatches])
    e1 = np.zeros([L,nBatches])
    e1[0,:] = 1
    h = h + e1
    h_batch = np.dot(Vd,h)
    g_batch = 1/h_batch
    for p in range(nBatches):
        g_batch[:,p] = N*g_batch[:,p]/sum(g_batch[:,p])
    return g_batch

def wt_batch_generate_gso(N,nBatches,alpha, eigenvalues,tMax):
    Vd = np.vander(eigenvalues,tMax)
    Vd = np.fliplr(Vd)
    t = np.random.randint(tMax, size = (nBatches))
    h = np.zeros([tMax,nBatches])
    for n in range(nBatches):
        h[t[n],n] = 1            
    h_batch = np.dot(Vd,h)
    g_batch = 1/h_batch
    for p in range(nBatches):
        g_batch[:,p] = N*g_batch[:,p]/sum(g_batch[:,p])
    return g_batch

def generate_normalized_gso_laplaciant(N,p):
    connected_nonperm = 0
    temp_count_perm_pairs = 10
    while temp_count_perm_pairs > 1e-10:
        temp_connect = 0
        while temp_connect < 1e-10:
            random = np.random.random((N, N))
            tri = np.tri(N, k=-1)
            # initialize adjacency matrix
            gso = np.zeros((N, N))
            # assign intra-community edges
            gso[np.logical_and(tri, random < p)] = 1
            gso += gso.T
            # connected check
            d = np.sum(gso, axis=1)
            temp_di_mutiply = 1
            for i in range(N):
                temp_di_mutiply = temp_di_mutiply*d[i]
            if abs(temp_di_mutiply) > 1e-10:
                temp_connect = 1
        Lp = np.diag(d) - gso
        Lpn = np.zeros((N,N))
        An = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                Lpn[i,j] = Lp[i,j]/np.sqrt(d[i])/np.sqrt(d[j])
                An[i,j] = gso[i,j]/np.sqrt(d[i])/np.sqrt(d[j])
        eigenvalues, V = np.linalg.eig(An)        
        # Permutation check
        temp_count_perm_pairs = 0
        for k in range(N):
            VV = np.outer(V[:,k],V[:,k])
            temp_non_zeros_entry_count = 0
            for l1 in range(N):
                for l2 in range(N):
                    if abs(VV[l1,l2]) > 1e-5:
                        temp_non_zeros_entry_count += 1
            if temp_non_zeros_entry_count < 5:
                temp_count_perm_pairs += 1
    return gso, d,Lpn, eigenvalues, V  


########################################################
############### Functions ##############################
########################################################
def fast_inverse(A,rho):
    N = A.shape[0]
    a = torch.diagonal(A, 0)
    sum_a = sum(a)
    a_1 = (1/a).reshape(N,1)
    A_1 = torch.diag(a_1.reshape(N))
    return A_1 - (rho/(1+rho*sum_a))*torch.matmul(a_1,a_1.transpose(0,1))

def fast_inverse_objf3(A,M,rho):
    # Fast inverse for objective function 3
    # M (torch), size: N,q
    N = A.shape[0]
    q = M.shape[1]
    a = torch.diagonal(A, 0)
    sum_a = sum(a)
    a_1 = (1/a).reshape(N,1)
    A_1 = torch.diag(a_1.reshape(N))
    A1M = torch.matmul(A_1,M)
#     print(torch.eye(q).shape, M.transpose(0,1).shape, A1M.shape)
    core_matrix = torch.eye(q) + rho*torch.matmul(M.transpose(0,1),A1M)
    return A_1 - rho*torch.matmul(torch.matmul(A1M,torch.inverse(core_matrix)),A1M.transpose(0,1))
#     return A_1 - (rho/(1+rho*sum_a))*torch.matmul(a_1,a_1.transpose(0,1))

def fast_inverse_no_constrain(A):
    N = A.shape[0]
    a = torch.diagonal(A, 0)
    A_1 = torch.diag(1./a)
    return A_1


def min_RE(x_pre,x_ture):
    re_1 = LA.norm(x_pre - x_ture)/LA.norm(x_ture)
    re_2 = LA.norm(x_pre + x_ture)/LA.norm(x_ture)
    RE = min(re_1,re_2)
    sign = re_1 < re_2 
    return RE,sign
def generate_V(N):
    V = torch.randn([N,N])
    Vout = torch.zeros([N,N])
    Vout[0,:] = V[0,:]/torch.norm(V[0,:]);
    for i in range(1,N):
        Vout[i,:] = V[i,:]
        for j in range(0,i):
            Vout[i,:] = Vout[i,:] - torch.dot(Vout[i,:],Vout[j,:])*Vout[j,:];
        Vout[i,:] = Vout[i,:]/torch.norm(Vout[i,:]);
    return Vout

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

def community_LabelsToNodeSets(communityLabels,gso,N_C,**kwargs):
    # N_C: number of sources per sample/community
    if 'mode' in kwargs.keys():
        mode = kwargs['mode']
    else:
        mode = 'random'
        
    N = len(communityLabels)
    A = abs(gso) > 1e-5
    nClass = np.max(communityLabels) + 1
    sourceNodes = []
    degree = np.sum(A, axis = 0) # degree of each vector

    communityList = []
    # For each community
    for c in range(nClass):
        communityNodes = np.nonzero(communityLabels == c)[0]
        degreeSorted = np.argsort(degree[communityNodes])
        if mode == 'random':
            np.random.shuffle(degreeSorted)
#             print('Randomized sorted degress:', degreeSorted)
        sourceNodes.append(communityNodes[degreeSorted[-N_C:]])
        communityList.append(communityNodes)
    result = {}
    result['sourceNodes'] = sourceNodes
    result['communityList'] = communityList # Node list for each of the communities
    result['nClass'] = nClass
    return result

        
def X_generate_fromSBM(N,P,S,communityLabels,gso,**kwargs):
## Generating full rank X
    # communityLabels: size(N,), mapping node to class/community
    # gso is the adjacency matrix
    if 'selectMode' in kwargs.keys():
        selectMode = kwargs['selectMode']
    else:
        selectMode = 'default'
    if 'signalMode' in kwargs.keys():
        signalMode = kwargs['signalMode']
    else:
        signalMode = 'default'
    X0 = np.zeros([N,P])
    temp_result = community_LabelsToNodeSets(communityLabels,gso,S)
    sourceNodes = temp_result['sourceNodes']
    communityNodeList = temp_result['communityList'] # Node list for each of the communities
    nClass = temp_result['nClass']
    sourceNode_set = np.array(sourceNodes)

    sampledIndicesList = np.random.choice(np.arange(nClass), size = P)
    sampledSources = sourceNode_set[sampledIndicesList]

    if selectMode == 'default':
            # Default means fixxing the source nodes
        for p in range(P):
            x0 = torch.zeros(N)
            for s in range(S):
                if signalMode == 'default':
                    x0[sampledSources[p,s]] = 1
                else:
                    x0[sampledSources[p,s]] = np.random.randn()
            X0[:,p] = x0
    else:
        for p in range(P):
            x0 = torch.zeros(N)
            sampledSourceIndex = sampledIndicesList[p]
            nodeSet_of_sampledSource = communityNodeList[sampledSourceIndex]
            np.random.shuffle(nodeSet_of_sampledSource)
            for s in range(S):
                if signalMode == 'default':
                    x0[nodeSet_of_sampledSource[s]] = 1
                else:
                    x0[nodeSet_of_sampledSource[s]] = np.random.randn()
            X0[:,p] = x0
    result = {}
    result['X0'] = X0
    result['sampledIndicesList'] = sampledIndicesList
    return result

def softshrink(x, lambd):
    mask1 = x > lambd
    mask2 = x < -lambd
    out = torch.zeros_like(x)
    out += mask1.float() * -lambd + mask1.float() * x
    out += mask2.float() * lambd + mask2.float() * x
    return out

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

################################
############# ADMM #############
################################

def admm(Y,V,rho_0,eta_0,C,N_ite,max_re):
    N = V.shape[0]
    P = Y.shape[1]
#     Y = torch.tensor(Y)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    v = torch.tensor(np.zeros(N))
    x = torch.tensor(np.zeros(N*P))
    u = torch.tensor(np.zeros(N*P))
    eta = torch.tensor(np.zeros(1))
    II = torch.tensor(np.ones([N,N]))
    In = torch.tensor(np.ones(N)) # ones(N,1)
    n_ite = 0
    max_re_matched = 0
    while n_ite < N_ite and max_re_matched ==0 :
        v_old = v
        x_old = x
        ZIk_inv = fast_inverse(torch.matmul(torch.transpose(Z,0,1),Z),eta_0/rho_0)/rho_0
        v_temp = torch.matmul(torch.transpose(Z,0,1),rho_0*x - u) + (eta_0*C-eta)*In
        v = torch.matmul(ZIk_inv, v_temp)
        X_update = torch.nn.Softshrink(lambd = 1/rho_0)
        x = X_update(torch.matmul(Z,v) + 1/rho_0*u)
        u = u + rho_0*(torch.matmul(Z,v) - x)
        eta = eta + eta_0*(torch.matmul(torch.tensor(np.ones([1,N])),v) - C)
        re = ((v-v_old)**2).mean()/(1e-10 + ((v_old)**2).mean())
        if re < max_re**2:
#             print('re = ',re)
            max_re_matched = 1
        n_ite += 1
    return x,v,n_ite,max_re_matched