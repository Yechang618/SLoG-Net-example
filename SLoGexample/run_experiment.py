###
python run_experiment.py --output_file "output_results.json" --useGPU True
python run_experiment.py
###

import numpy as np
from numpy import linalg as LA
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pickle
import datetime
import matplotlib as mpl
import argparse
import json

# Update Matplotlib settings
mpl.rcParams.update(mpl.rcParamsDefault)

# Import SLOG packages
from SLoGexample import SLOGexperiment as SLOGexperiment
from SLoGexample import SLOGtools as SLOGtools
from SLoGexample import SLOGdata as SLOGdata

# Set up device (CPU/GPU)
def get_device(useGPU=True):
    if useGPU and torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.empty_cache()
    else:
        device = 'cpu'
    torch.manual_seed(0)
    return torch.device(device)

# Function to run the experiment
def run_experiment(output_file="result_1.json", useGPU=True):
    device = get_device(useGPU)
    print("Device selected: %s" % device)

    # Model parameters
    K = 5  # Number of layers

    # Simulation parameters
    P = 200
    nTrain = P * 500
    batchsize, nValid, nTest = P, P, P
    nEpochs, early_stop_criteria = 30, 1e-5

    # Data parameters
    L, alpha = 5, 1.0

    # Graph parameters
    nNodes = 20  # Number of nodes
    S = int(0.15 * nNodes)  # Number of sources per signal
    C = nNodes  # Constrain constant
    q = 2  # Number of constrain vectors for SLOG-Net-v3

    # Graph type and options
    graphType = 'ER'  # Type of graph
    graphOptions = {'probIntra': 0.3}  # Probability of drawing edges

    # Other parameters
    filterType = 'h'
    filterTrainType = 'h'
    signalMode = 'Gaussion'
    trainMode = 'default'
    filterMode = 'default'
    selectMode = 'random'
    noiseLevel = 0.0
    noiseType = 'uniform'

    # Set up parameter dictionaries
    simuParas = {
        'nNodes': nNodes,
        'S': S,
        'nTrain': nTrain,
        'batchsize': batchsize,
        'nValid': nValid,
        'nTest': nTest,
        'L': L,
        'noiseLevel': noiseLevel,
        'noiseType': noiseType,
        'filterType': filterType,
        'signalMode': signalMode,
        'trainMode': trainMode,
        'filterMode': filterMode,
        'selectMode': selectMode,
        'graphType': graphType,
        'alpha': alpha,
        'nEpochs': nEpochs,
        'device': str(device),
        'tMax': 5,
        'early_stop_criteria': early_stop_criteria
    }

    modelParas = {
        'filterTrainType': filterTrainType,
        'C': C,
        'q': q,
        'K': K
    }

    expParas = {
        'nRealiz': 5  # Number of realizations
    }

    # Run the SLOG experiment
    result_1 = SLOGexperiment.slog_experiments(
        simuParas=simuParas,
        graphOptions=graphOptions,
        modelParas=modelParas,
        expParas=expParas
    )

    # Save the result as a JSON file
    with open(output_file, "w") as f:
        json.dump(result_1, f, indent=4)

    print(f"Experiment completed. Results saved to {output_file}")

    return result_1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLOG experiment and save results.")
    parser.add_argument("--output_file", type=str, default="result_1.json", help="Output file to save results (JSON format)")
    parser.add_argument("--useGPU", type=bool, default=True, help="Use GPU if available (True/False)")
    
    args = parser.parse_args()
    run_experiment(output_file=args.output_file, useGPU=args.useGPU)
