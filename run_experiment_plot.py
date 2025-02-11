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
    simuParas['N_realiz'] = expParas['nRealiz']

    # Training
    graphOptions = {'probIntra': 0.3}
    result_1 = SLOGexperiment.slog_experiments(simuParas=simuParas, graphOptions=graphOptions,
                                               modelParas=modelParas, expParas=expParas)
    experiment_results = result_1.experiment_results

    print("Training completed, with", len(experiment_results), "models")

    # Prepare test results
    N_realiz = expParas['nRealiz']
    N_model = len(experiment_results)
    noiseLvs = np.linspace(0, 0.1, 6)  # Adjust noise levels as needed

    result_exp_rex_slog = np.zeros((len(noiseLvs), N_model, N_realiz))
    result_exp_reg_slog = np.zeros((len(noiseLvs), N_model, N_realiz))
    result_elapse_slog = np.zeros((len(noiseLvs), N_model, N_realiz))
    result_exp_rex_admm = np.zeros((len(noiseLvs), N_model, N_realiz))
    result_exp_reg_admm = np.zeros((len(noiseLvs), N_model, N_realiz))
    result_elapse_admm = np.zeros((len(noiseLvs), N_model, N_realiz))

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

            result_exp_rex_slog[n_nlvs,n_model,:] = result['re_x_slog']
            result_exp_reg_slog[n_nlvs,n_model,:] = result['re_g_slog']
            result_elapse_slog[n_nlvs,n_model,:] = result['elapse_slog']
            result_exp_rex_admm[n_nlvs,n_model,:] = result['re_x_admm']
            result_exp_reg_admm[n_nlvs,n_model,:] = result['re_g_admm']
            result_elapse_admm[n_nlvs,n_model,:] = result['elapse_admm']

    # Compute Mean & Std for Accuracy
    acc_mean = np.mean(result_exp_acc_slog.reshape((len(noiseLvs), -1)), axis=1)
    acc_std = np.std(result_exp_acc_slog.reshape((len(noiseLvs), -1)), axis=1)
    acc_mean_admm = np.mean(result_exp_acc_admm.reshape((len(noiseLvs), -1)), axis=1)
    acc_std_admm = np.std(result_exp_acc_admm.reshape((len(noiseLvs), -1)), axis=1)

    # Compute Mean & Std for RE_x
    rex_mean = np.mean(result_exp_rex_slog.reshape((len(noiseLvs), -1)), axis=1)
    rex_std = np.std(result_exp_rex_slog.reshape((len(noiseLvs), -1)), axis=1)
    rex_mean_admm = np.mean(result_exp_rex_admm.reshape((len(noiseLvs), -1)), axis=1)
    rex_std_admm = np.std(result_exp_rex_admm.reshape((len(noiseLvs), -1)), axis=1)


    # Plot results
    fig, ax = plt.subplots(nrows=2, ncols=1, constrained_layout=True,figsize=(10,5))
    ax[0].plot(noiseLvs, acc_mean_admm, color='#CC4F1B', linewidth=2, label='ADMM')
    ax[0].fill_between(noiseLvs, acc_mean_admm - acc_std_admm, acc_mean_admm + acc_std_admm, alpha=0.2, color='#FF9848')
    ax[0].plot(noiseLvs, acc_mean, color='#1B2ACC', linewidth=2, label='SLoG-Net')
    ax[0].fill_between(noiseLvs, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2, color='#089FFF')

    ax[0].set_xlabel('Noise Level')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim([0, 1])
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(noiseLvs, rex_mean_admm, color='#CC4F1B', linewidth=2, label='ADMM')
    ax[1].fill_between(noiseLvs, rex_mean_admm - rex_std_admm, rex_mean_admm + rex_std_admm, alpha=0.2, color='#FF9848')
    ax[1].plot(noiseLvs, rex_mean, color='#1B2ACC', linewidth=2, label='SLoG-Net')
    ax[1].fill_between(noiseLvs, rex_mean - rex_std, rex_mean + rex_std, alpha=0.2, color='#089FFF')

    ax[1].set_xlabel('Noise Level')
    ax[1].set_ylabel('RE_x')
    ax[1].set_ylim([0, 1])
    ax[1].legend()
    ax[1].grid()    

    # Save figure
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "accuracy_plot.png")
    # save_path = "accuracy_plot.png"
    fig.savefig(save_path, format="pdf",transparent=True)
    print(f"Figure saved at {save_path}")

    plt.show()

    ################################################################################
    ######################### Saved figure  ####################################
    ################################################################################
    legend_fontsize = 15
    label_fontsize = 20
    line_wid = 4

    [N_noiseLvs,N_model,N_realiz]  = result_exp_acc_slog.shape
    acc_arg = result_exp_acc_slog.reshape((N_noiseLvs,N_model*N_realiz))
    acc_arg_admm = result_exp_acc_admm.reshape((N_noiseLvs,N_model*N_realiz))
    acc_mean = np.mean(acc_arg,axis = 1)
    acc_std = np.std(acc_arg,axis = 1)
    acc_mean_admm = np.mean(acc_arg_admm,axis = 1)
    acc_std_admm = np.std(acc_arg_admm,axis = 1)

    rex_arg = result_exp_rex_slog.reshape((N_noiseLvs,N_model*N_realiz))
    reg_arg = result_exp_reg_slog.reshape((N_noiseLvs,N_model*N_realiz))
    elps_arg = result_elapse_slog.reshape((N_noiseLvs,N_model*N_realiz))
    rex_arg_admm = result_exp_rex_admm.reshape((N_noiseLvs,N_model*N_realiz))
    reg_arg_admm = result_exp_reg_admm.reshape((N_noiseLvs,N_model*N_realiz))
    elps_arg_admm = result_elapse_admm.reshape((N_noiseLvs,N_model*N_realiz))

    rex_mean = np.mean(rex_arg,axis = 1)
    rex_std = np.std(rex_arg,axis = 1)
    reg_mean = np.mean(reg_arg,axis = 1)
    reg_std = np.std(reg_arg,axis = 1)
    elps_mean = np.mean(elps_arg,axis = 1)
    elps_std = np.std(elps_arg,axis = 1)

    rex_mean_admm = np.mean(rex_arg_admm,axis = 1)
    rex_std_admm = np.std(rex_arg_admm,axis = 1)
    reg_mean_admm = np.mean(reg_arg_admm,axis = 1)
    reg_std_admm = np.std(reg_arg_admm,axis = 1)
    elps_mean_admm = np.mean(elps_arg_admm,axis = 1)
    elps_std_admm = np.std(elps_arg_admm,axis = 1)

    fig_save_2, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True,figsize=(10,5))

    axes[0].plot(noiseLvs,rex_mean_admm, color='#CC4F1B',linewidth=line_wid, label='ADMM')
    axes[0].fill_between(noiseLvs,rex_mean_admm - rex_std_admm, rex_mean_admm + rex_std_admm,alpha= 0.2, edgecolor='#CC4F1B', facecolor='#FF9848',
        linewidth=0)
    axes[0].plot(noiseLvs,rex_mean, color='#1B2ACC',linewidth=line_wid, label='SLoG-Net')
    axes[0].fill_between(noiseLvs, rex_mean-rex_std, rex_mean+rex_std,alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
        linewidth=0)
    axes[0].set_ylabel('MRE of x', fontsize = label_fontsize)
    axes[0].set_xlabel('Noise Level', fontsize = label_fontsize)
    axes[0].set_ylim([0,1.0])
    axes[0].legend(loc = 'best',fontsize = 15)
    axes[0].grid()

    # ACC
    axes[1].plot(noiseLvs,acc_mean_admm, color='#CC4F1B',linewidth=line_wid, label='ADMM')
    axes[1].fill_between(noiseLvs,acc_mean_admm - acc_std_admm, acc_mean_admm + acc_std_admm,alpha= 0.2, edgecolor='#CC4F1B', facecolor='#FF9848',
        linewidth=0)
    axes[1].plot(noiseLvs,acc_mean, color='#1B2ACC',linewidth=line_wid, label='SLoG-Net')
    axes[1].fill_between(noiseLvs, acc_mean-acc_std, acc_mean+acc_std,alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
        linewidth=0)
    axes[1].set_ylabel('ACC of x', fontsize = label_fontsize)
    axes[1].set_xlabel('Noise Level', fontsize = label_fontsize)
    axes[1].set_ylim([0.0,1.0])
    axes[1].legend(loc = 'best',fontsize = 15)
    axes[1].grid()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLoG-Net Experiment")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to the JSON config file")
    args = parser.parse_args()
    run_experiment(args.config)
