import preprocess
import evaluate

import joblib
import tensorflow as tf
import numpy as np
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as patches
from cycler import cycler
import time

physical_devices = tf.config.list_physical_devices('GPU')

def generate_basis(X):

    X_sq = X ** 2

    return np.concatenate([X, X_sq], axis=1)

# Load data
cwd = "../sim_VA/result/GMWB/nolapse/"
Price = np.load(cwd + "outerScenarios_GMWB_RS_nolapse.npy")
Loss = np.load(cwd + "hedgingLoss_GMWB_RS_100_nolapse.npy")
Return = (Price[:, 1:] - Price[:, :-1]) / Price[:, :-1]
# Train-test split

test_size = 3000
seed = 22
flag_training = True
X_train, y_train, X_test, y_test, y_mean, y_std = preprocess.transform_data(Return, Loss, flag_training, test_size, seed, model="FFN")
X_train_Q, X_test_Q = generate_basis(X_train), generate_basis(X_test)
X_train_RNN, y_train, X_test_RNN, y_test, y_mean, y_std = preprocess.transform_data(Return, Loss, flag_training, test_size, seed, model="RNN")

Loss = np.load(cwd + "hedgingLoss_GMWB_RS_10000_nolapse.npy")
Return = (Price[:, 1:] - Price[:, :-1]) / Price[:, :-1]
X_true, y_true = preprocess.transform_data(Return, Loss, training=False, y_mean=y_mean, y_std=y_std, model="FFN")
X_true_Q = generate_basis(X_true)
X_true_RNN, _ = preprocess.transform_data(Return, Loss, training=False, y_mean=y_mean, y_std=y_std, model="RNN")

Loss_1000 = np.load(cwd + "hedgingLoss_GMWB_RS_1000_nolapse_1683836051.npy")
X_train_1000, y_train_1000, X_test, y_test, y_mean_1000, y_std_1000 = preprocess.transform_data(Return, Loss_1000, True, test_size, seed, model="RNN")
_, y_train_1000 = preprocess.transform_data(Return, Loss_1000, training=False, y_mean=y_mean_1000, y_std=y_std_1000, model="FNN")

# Save path for figures
save_path_figures = "./figures/GMWB/"
if not os.path.isdir(save_path_figures):
    os.makedirs(save_path_figures)

# Load models
save_path_REG = "./trainedModels/GMWB_PY/REG/"
save_path_FNN = "./trainedModels/GMWB_PY/FFN_128-32_ReLU/"
save_path_RNN1 = "./trainedModels/GMWB_PY/RNN_32-4_tanh/"
save_path_LSTM1 = "./trainedModels/GMWB_PY/LSTM_32-4_tanh/"
save_path_RNN2 = "./trainedModels/GMWB_PY/RNN_64-8_tanh/"
save_path_LSTM2 = "./trainedModels/GMWB_PY/LSTM_64-8_tanh/"
save_path_LSTM3 = "./trainedModels/GMWB_PY/LSTM_64-16_tanh/"
save_path_LSTM4 = "./trainedModels/GMWB_PY/LSTM_128-16_tanh/"

model_trained_reg_L = joblib.load(save_path_REG + "reg_L.pkl")
model_trained_reg_Q = joblib.load(save_path_REG + "reg_Q.pkl")
model_trained_FNN = tf.keras.models.load_model(save_path_FNN + "trained_model")
model_trained_RNN1 = tf.keras.models.load_model(save_path_RNN1 + "trained_model")
model_trained_LSTM1 = tf.keras.models.load_model(save_path_LSTM1 + "trained_model")
model_trained_RNN2 = tf.keras.models.load_model(save_path_RNN2 + "trained_model")
model_trained_LSTM2 = tf.keras.models.load_model(save_path_LSTM2 + "trained_model")
model_trained_LSTM3 = tf.keras.models.load_model(save_path_LSTM3 + "trained_model")
model_trained_LSTM4 = tf.keras.models.load_model(save_path_LSTM4 + "trained_model")

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
        
        if (model_name == "Standard Procedure (N=1000)") or (model_name == "SNS (N=1000)"):

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
    rect = patches.Rectangle((0.075, 222.5), 0.074, 5.0, linewidth=1, edgecolor='tab:brown', facecolor='none')

    ax_CVaR.add_patch(rect)

    ax_CVaR.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax_CVaR.yaxis.get_ticklabels()[1].set_weight('bold')
    ax_CVaR.yaxis.get_ticklabels()[-2].set_weight('bold')
    ax_CVaR.legend([model_name for model_name in model_names] + ["True CVaR"], loc='lower right')
    ax_CVaR.set_xlabel("Safety Margin")
    ax_CVaR.set_ylabel("CVaR")
    fig_CVaR.savefig(cwd + "CVaR_lower.png")

    ax_CVaR_2.axhline(y=CVaR_True, color="black", linewidth=2, linestyle="-")
    ax_CVaR_2.set_xticks(np.arange(0.075, 0.16, 0.025))
    ax_CVaR_2.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax_CVaR_2.set_ylim(222.5, 227.5)
    ax_CVaR_2.yaxis.get_ticklabels()[1].set_weight('bold')
    ax_CVaR_2.yaxis.get_ticklabels()[-2].set_weight('bold')
    ax_CVaR_2.legend([model_name for model_name in model_names] + ["True CVaR"], loc='lower right')
    ax_CVaR_2.set_xlabel("Safety Margin")
    ax_CVaR_2.set_ylabel("CVaR")
    fig_CVaR_2.savefig(cwd + "CVaR_higher.png")

    return None

# cal_CVaR(save_path_figures,
#         [model_trained_reg_Q, model_trained_FNN, model_trained_RNN1, model_trained_LSTM1, y_train_1000],
#         ["QPR", "FNN", "RNN", "LSTM", "SNS (N=1000)"],
#         [X_true_Q, X_true, X_true_RNN, X_true_RNN] + [1],
#         [y_train_1000, y_train_1000, y_train_1000, y_train_1000] + [1],
#         [y_mean_1000, y_mean_1000, y_mean_1000, y_mean_1000] + [1],
#         [y_std_1000, y_std_1000, y_std_1000, y_std_1000] + [1])


# evaluate.generate_plots(save_path_figures,
#                         [model_trained_reg_Q, model_trained_FNN, model_trained_RNN1, model_trained_RNN2],
#                         ["QPR - True Dataset", "FNN - True Dataset", "RNN_LoCap - True Dataset", "RNN_HiCap - True Dataset"],
#                         [X_true_Q, X_true, X_true_RNN, X_true_RNN], [y_true, y_true, y_true, y_true], (0.30, 1.0), (-200, 500), scale_flag=True, y_mean=y_mean, y_std=y_std)

# evaluate.generate_plots(save_path_figures,
#                         [model_trained_LSTM1, model_trained_LSTM2, model_trained_LSTM3, model_trained_LSTM4],
#                         ["LSTM1 - True Dataset", "LSTM2 - True Dataset", "LSTM3 - True Dataset", "LSTM4 - True Dataset"],
#                         [X_true_RNN, X_true_RNN, X_true_RNN, X_true_RNN], [y_true, y_true, y_true, y_true], (0.75, 1.0), (-8, 8))

# evaluate.generate_plots(save_path_figures,
#                         [model_trained_LSTM1, model_trained_LSTM3, y_train_1000],
#                         ["LSTM_LoCap", "LSTM_HiCap", "SNS (N=1000)"],
#                         [X_true_RNN, X_true_RNN] + [1], [y_true, y_true, y_true], (0.75, 1.0), (-400, 800),
#                         scale_flag=True, y_means=[y_mean] * 3, y_stds=[y_std] * 3)

# evaluate.generate_plots(save_path_figures,
#                         [model_trained_LSTM1, model_trained_LSTM3, y_train_1000],
#                         ["LSTM_LoCap", "LSTM_HiCap", "Standard Procedure (N=1000)"],
#                         [X_true_RNN, X_true_RNN] + [1], [y_true, y_true, y_true], (0.75, 1.0), (-400, 800),
#                         scale_flag=True, y_means=[y_mean] * 3, y_stds=[y_std] * 3)

evaluate.generate_plots(save_path_figures,
                        [model_trained_reg_Q, model_trained_FNN, model_trained_RNN1, model_trained_LSTM1, y_train_1000],
                        ["QPR", "FNN", "RNN", "LSTM", "SNS (N=1000)"],
                        [X_true_Q, X_true, X_true_RNN, X_true_RNN] + [1], [y_true, y_true, y_true, y_true, y_true], (0.30, 1.0), (-400, 800), 
                        scale_flag=True, y_means=[y_mean] * 5, y_stds=[y_std] * 5)

# evaluate.generate_plots(save_path_figures,
#                         [model_trained_reg_Q, model_trained_FNN, model_trained_RNN2, model_trained_LSTM2],
#                         ["QPR - True Dataset", "FNN - True Dataset", "RNN_HiCap - True Dataset", "LSTM_HiCap - True Dataset"],
#                         [X_true_Q, X_true, X_true_RNN, X_true_RNN], [y_true, y_true, y_true, y_true], (0.30, 1.0), (-400, 800), 
#                         scale_flag=True, y_means=[y_mean] * 4, y_stds=[y_std] * 4)

# evaluate.generate_plots(save_path_figures,
#                         [model_trained_reg_L, model_trained_reg_L, model_trained_reg_L],
#                         ["MLR - Training Dataset", "MLR - Test Dataset", "MLR - True Dataset"],
#                         [X_train, X_test, X_true], [y_train, y_test, y_true], (0.30, 1.0), (-400, 800), 
#                         scale_flag=False, y_means=[y_mean] * 4, y_stds=[y_std] * 4)