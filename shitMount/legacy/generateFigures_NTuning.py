import preprocess
import evaluate

import joblib
import numpy as np
import tensorflow as tf
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as patches
from cycler import cycler
import time

def generate_basis(X):

    X_sq = X ** 2

    return np.concatenate([X, X_sq], axis=1)

# Load data
cwd = "../sim_VA/result/GMWB/nolapse/"
save_path = "./trainedModels/GMWB_PY/"


# Load models
save_path_LSTM = save_path + "NTuning/"

model_trained_LSTM_100 = tf.keras.models.load_model(save_path + "LSTM_32-4_tanh/trained_model")
model_trained_LSTM_001 = tf.keras.models.load_model(save_path_LSTM + "trained_model_001")
model_trained_LSTM_010 = tf.keras.models.load_model(save_path_LSTM + "trained_model_010")

save_path_figures = "./figures/NTuning_LSTM1/"
if not os.path.exists(save_path_figures):
    os.makedirs(save_path_figures)

# Load True Dataset
Price_T = np.load(cwd + "outerScenarios_GMWB_RS_nolapse.npy")
Loss_T = np.load(cwd + "hedgingLoss_GMWB_RS_10000_nolapse.npy")
Return_T = (Price_T[:, 1:] - Price_T[:, :-1]) / Price_T[:, :-1]

X_true_T, y_true_T = preprocess.transform_data(Return_T, Loss_T, training=False, y_mean=0, y_std=1, model="FFN")

Price = np.load(cwd + "outerScenarios_GMWB_RS_nolapse.npy")
Loss_001 = np.load(cwd + "hedgingLoss_GMWB_RS_001_nolapse.npy")
Return = (Price[:, 1:] - Price[:, :-1]) / Price[:, :-1]

# Load Train and Test Dataset
test_size = 3000
seed = 22
X_train_001, y_train_001, X_test, y_test, y_mean_001, y_std_001 = preprocess.transform_data(Return, Loss_001, True, test_size, seed, model="RNN")
X_train_001, y_train_001 = preprocess.transform_data(Return, Loss_001, training=False, y_mean=y_mean_001, y_std=y_std_001, model="RNN")

X_true_001, y_true_001 = preprocess.transform_data(Return_T, Loss_T, training=False, y_mean=y_mean_001, y_std=y_std_001, model="RNN")

Loss_010 = np.load(cwd + "hedgingLoss_GMWB_RS_010_nolapse.npy")
Return = (Price[:, 1:] - Price[:, :-1]) / Price[:, :-1]
X_train_010, y_train_010, X_test, y_test, y_mean_010, y_std_010 = preprocess.transform_data(Return, Loss_010, True, test_size, seed, model="RNN")
X_train_010, y_train_010 = preprocess.transform_data(Return, Loss_010, training=False, y_mean=y_mean_010, y_std=y_std_010, model="RNN")

X_true_010, y_true_010 = preprocess.transform_data(Return_T, Loss_T, training=False, y_mean=y_mean_010, y_std=y_std_010, model="RNN")

Loss_100 = np.load(cwd + "hedgingLoss_GMWB_RS_100_nolapse.npy")
Return = (Price[:, 1:] - Price[:, :-1]) / Price[:, :-1]
X_train_100, y_train_100, X_test, y_test, y_mean_100, y_std_100 = preprocess.transform_data(Return, Loss_100, True, test_size, seed, model="RNN")
X_train_100, y_train_100 = preprocess.transform_data(Return, Loss_100, training=False, y_mean=y_mean_100, y_std=y_std_100, model="RNN")

X_true_100, y_true_100 = preprocess.transform_data(Return_T, Loss_T, training=False, y_mean=y_mean_100, y_std=y_std_100, model="RNN")

Loss_1000 = np.load(cwd + "hedgingLoss_GMWB_RS_1000_nolapse_1683836051.npy")
X_train_1000, y_train_1000, X_test, y_test, y_mean_1000, y_std_1000 = preprocess.transform_data(Return, Loss_1000, True, test_size, seed, model="RNN")
_, y_train_1000 = preprocess.transform_data(Return, Loss_1000, training=False, y_mean=y_mean_1000, y_std=y_std_1000, model="FNN")

X_true_10000, y_true_10000 = preprocess.transform_data(Return_T, Loss_T, training=False, y_mean=y_mean_100, y_std=y_std_100, model="FFN")

def cal_CVaR(cwd, models, model_names, X_list, y_true_list, y_mean_list, y_std_list):

    alpha = 0.05

    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.tight_layout()
    plt.subplots_adjust(left=0.5, bottom=0.5)
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    

    fig_CVaR = plt.figure(figsize=(8, 4))
    ax_CVaR = fig_CVaR.add_subplot(1, 1, 1)

    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.15)

    fig_CVaR_2 = plt.figure(figsize=(8, 4))
    ax_CVaR_2 = fig_CVaR_2.add_subplot(1, 1, 1)

    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.15)


    default_cycler = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:purple',
                                    'tab:red', 'tab:brown', 'tab:pink', 'tab:gray']) +
                      cycler(linestyle=['--', '-.', ':', '-', '--', '-.', ':', '-']))

    ax_CVaR.set_prop_cycle(default_cycler)
    ax_CVaR_2.set_prop_cycle(default_cycler)

    for X, y_true, y_mean, y_std, model, model_name in zip(X_list, y_true_list, y_mean_list, y_std_list, models, model_names):
        
        if model_name == "Standard Procedure (N=1000)":

            start = int(alpha * y.shape[0])
            CVaR_sns = np.mean(np.sort(Loss_1000.flatten())[-start:])
            CVaR_sns = 226.97
            ax_CVaR.axhline(y=CVaR_sns, color="tab:gray", linewidth=2, linestyle=":")
            ax_CVaR_2.axhline(y=CVaR_sns, color="tab:gray", linewidth=2, linestyle=":")

        else:

            y = np.array(y_true).flatten()
            y_hat = model.predict(X).flatten()

            start = int(alpha * y.shape[0])
            end = int(start * 4)
            step = (end - start) // 20

            CVaR = {}
            CVaR_True = 226.7733

            ind_hat_sorted = np.argsort(y_hat)[::-1]

            for num in range(start, end, step):

                y_eval_sorted = np.sort(y[ind_hat_sorted][:num])[::-1][:start]

                CVaR[(num/start - 1)/20] = np.mean(y_eval_sorted) * y_std + y_mean

            df = pd.Series(CVaR)
            ax_CVaR.plot(df, linewidth=2)

            ax_CVaR_2.plot(df[0.075:], linewidth=2)

    ax_CVaR.axhline(y=CVaR_True, color="black", linewidth=2, linestyle="-")
    ax_CVaR.set_xticks(np.arange(0, 0.16, 0.05))
    rect = patches.Rectangle((0.075, 225.5), 0.074, 2.0, linewidth=1, edgecolor='tab:brown', facecolor='none')

    # ax_CVaR.add_patch(rect)

    ax_CVaR.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax_CVaR_2.set_ylim(226, 227)
    ax_CVaR.yaxis.get_ticklabels()[1].set_weight('bold')
    ax_CVaR.yaxis.get_ticklabels()[-2].set_weight('bold')
    ax_CVaR.legend([model_name for model_name in model_names] + ["True CVaR"], loc='lower right')
    ax_CVaR.set_xlabel("Safety Margin")
    ax_CVaR.set_ylabel("CVaR")
    fig_CVaR.savefig(cwd + "LSTM_LoCap_CVaR_lower.png")

    ax_CVaR_2.axhline(y=CVaR_True, color="black", linewidth=2, linestyle="-")
    ax_CVaR_2.set_xticks(np.arange(0.075, 0.16, 0.025))
    ax_CVaR_2.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax_CVaR_2.yaxis.get_ticklabels()[0].set_weight('bold')
    ax_CVaR_2.yaxis.get_ticklabels()[-1].set_weight('bold')
    ax_CVaR_2.legend([model_name for model_name in model_names] + ["True CVaR"], loc='lower left')
    ax_CVaR_2.set_xlabel("Safety Margin")
    ax_CVaR_2.set_ylabel("CVaR")
    fig_CVaR_2.savefig(cwd + "LSTM_LoCap_CVaR_higher.png")

    return None

cal_CVaR(save_path_figures,
        [model_trained_LSTM_001, model_trained_LSTM_010, model_trained_LSTM_100] + [1],
        ["Proxy - High Noise", "Proxy - Medium Noise", "Proxy - Low Noise", "Standard Procedure (N=1000)"],
        [X_true_001, X_true_010, X_true_100] + [1],
        [y_train_1000, y_train_1000, y_train_1000] + [1],
        [y_mean_1000, y_mean_1000, y_mean_1000] + [1],
        [y_std_1000, y_std_1000, y_std_1000] + [1])

# evaluate.generate_plots(save_path_figures,
#                         [model_trained_LSTM_001, model_trained_LSTM_010, model_trained_LSTM_100, y_train_1000],
#                         ["Proxy - High Noise", "Proxy - Medium Noise", "Proxy - Low Noise", "Standard Procedure (N=1000)"],
#                         [X_true_001, X_true_010, X_true_100, X_true_10000],
#                         [y_true_001, y_true_010, y_true_100, y_true_10000], lim_tail=(0.60, 1.0), lim_QQ=(-450, 800),
#                         scale_flag=True, 
#                         y_means=[y_mean_001, y_mean_010, y_mean_100, y_mean_1000], y_stds=[y_std_001, y_std_010, y_std_100, y_std_1000])

# evaluate.generate_plots(save_path_figures,
#                         [model_trained_LSTM_001, model_trained_LSTM_010, model_trained_LSTM_100],
#                         ["Proxy - High Noise", "Proxy - Medium Noise", "Proxy - Low Noise"],
#                         [X_true_001, X_true_010, X_true_100],
#                         [y_true_001, y_true_010, y_true_100], lim_tail=(0.60, 1.0), lim_QQ=(-450, 800),
#                         scale_flag=True, 
#                         y_means=[y_mean_001, y_mean_010, y_mean_100], y_stds=[y_std_001, y_std_010, y_std_100])