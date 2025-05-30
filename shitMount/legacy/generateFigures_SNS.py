import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

import os
import joblib

import preprocess, fit_nn, evaluate

def generate_basis(X):

    X_sq = X ** 2

    return np.concatenate([X, X_sq], axis=1)
                          
VA_type = "GMWB"
lapse_type = "nolapse"

cwd = f"../sim_VA/result/{VA_type}/{lapse_type}/"
save_path = f"./trainedModels/{VA_type}_PY/{lapse_type}/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

price = np.load(cwd + f"outerScenarios_{VA_type}_RS_{lapse_type}.npy")
rtn = (price[:, 1:] - price[:, :-1]) / price[:, :-1]
loss_T = np.load(cwd + f"hedgingLoss_{VA_type}_RS_10000_{lapse_type}.npy")

substring_SNS = f"hedgingLoss_{VA_type}_RS_1000_{lapse_type}"

substring_lowNoise = f"hedgingLoss_{VA_type}_RS_100_{lapse_type}"
substring_mediumNoise = f"hedgingLoss_{VA_type}_RS_010_{lapse_type}"
substring_highNoise = f"hedgingLoss_{VA_type}_RS_001_{lapse_type}"

lossFiles_SNS = [file for file in os.listdir(cwd) if substring_SNS in file and os.path.isfile(os.path.join(cwd, file))]

lossFiles_lowNoise = [file for file in os.listdir(cwd) if substring_lowNoise in file and os.path.isfile(os.path.join(cwd, file))]
lossFiles_mediumNoise = [file for file in os.listdir(cwd) if substring_mediumNoise in file and os.path.isfile(os.path.join(cwd, file))]
lossFiles_highNoise = [file for file in os.listdir(cwd) if substring_highNoise in file and os.path.isfile(os.path.join(cwd, file))]

n_rep = 50
test_size = 3000
seed = 22

alpha = 0.05

model_name = "LSTM_HiCap_mediumNoise"

CVaR = np.zeros(n_rep)

for n in range(n_rep):
    
    print("Macro replication:", n + 1)

    loss_LN = np.load(cwd + lossFiles_SNS[n])
    
    CVaR[n] = np.mean(loss_LN[loss_LN > np.quantile(loss_LN, 1-alpha)])

print("CVaR:", np.mean(CVaR), np.std(CVaR))
print("Quantiles", np.quantile(CVaR, [0.05, 0.5, 0.95]))