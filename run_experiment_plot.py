import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import json
import random
import matplotlib as mpl

# Import SLOG packages
from SLoGexample import SLOGexperiment as SLOGexperiment

mpl.rcParams.update(mpl.rcParamsDefault)

def run_experiment(config_file):
    # Load configurations from JSON file
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Set device
    useGPU = config.get("useGPU", True)
    device = 'cuda:0' if useGPU and torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache() if device == 'cuda:0' else None
    device = torch.device(device)
    print("Device selected:", device)
    torch.manual_seed(0)
    random.seed(0)

    # Load parameters from config
    simuParas = config["simuParas"]
    modelParas = config["modelParas"]
    expParas = config["expParas"]
    simuParas["device"] = device

    # Training
    graphOptions = {'probIntra': 0.3}
    result_1 = SLOGexperiment.slog_experiments(simuParas=simuParas, graphOptions=graphOptions,
                                               modelParas=modelParas, expParas=expParas)
    experiment_results = result_1.experiment_results

    # Prepare test results
    N_realiz = expParas['nRealiz']
    N_model = len(experiment_results)
    noiseLvs = np.linspace(0, 0.1, 6)  # Adjust noise levels as needed

    result_exp_acc_slog = np.zeros((len(noiseLvs), N_model, N_realiz))
    result_exp_acc_admm = np.zeros((len(noiseLvs), N_model, N_realiz))

    for n_model, exp_result in enumerate(experiment_results):
        for n_nlvs, noise_level in enumerate(noiseLvs):
            simuParas['noiseLevel'] = noise_level
            print(f"Model {n_model}, Noise Level {noise_level}")

            result = SLOGexperiment.test_admmCompare_local(
                simuParas['nNodes'], simuParas['P'], simuParas['S'], exp_result,
                visual=False, simuParas=simuParas, modelParas=modelParas
            )

            result_exp_acc_slog[n_nlvs, n_model, :] = result['acc_x_slog']
            result_exp_acc_admm[n_nlvs, n_model, :] = result['acc_x_admm']

    # Compute Mean & Std for Accuracy
    acc_mean = np.mean(result_exp_acc_slog.reshape((len(noiseLvs), -1)), axis=1)
    acc_std = np.std(result_exp_acc_slog.reshape((len(noiseLvs), -1)), axis=1)
    acc_mean_admm = np.mean(result_exp_acc_admm.reshape((len(noiseLvs), -1)), axis=1)
    acc_std_admm = np.std(result_exp_acc_admm.reshape((len(noiseLvs), -1)), axis=1)

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noiseLvs, acc_mean_admm, color='#CC4F1B', linewidth=2, label='ADMM')
    ax.fill_between(noiseLvs, acc_mean_admm - acc_std_admm, acc_mean_admm + acc_std_admm, alpha=0.2, color='#FF9848')
    ax.plot(noiseLvs, acc_mean, color='#1B2ACC', linewidth=2, label='SLoG-Net')
    ax.fill_between(noiseLvs, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2, color='#089FFF')

    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid()

    # Save figure
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "accuracy_plot.png"))
    print(f"Figure saved at {output_dir}/accuracy_plot.png")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLoG-Net Experiment")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to the JSON config file")
    args = parser.parse_args()
    run_experiment(args.config)
