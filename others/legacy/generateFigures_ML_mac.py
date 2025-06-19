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

model_name = "LSTM_LoCap_mediumNoise"

MSE_train = np.zeros(n_rep)
MSE_test = np.zeros(n_rep)
MSE_true = np.zeros(n_rep)
VaR = np.zeros(n_rep)
CVaR = np.zeros(n_rep)

for n in range(n_rep):
    
    print("Macro replication:", n + 1)

    loss_LN = np.load(cwd + lossFiles_mediumNoise[n])
    
    X_train, y_train, X_test, y_test, y_mean, y_std = preprocess.transform_data(rtn, loss_LN, True, test_size, seed, model="RNN")
    
    X_true, y_true = preprocess.transform_data(rtn, loss_T, training=False, y_mean=y_mean, y_std=y_std, model="RNN")
    
    y_true = np.array(y_true).flatten()

    if "REG" in model_name:
            
        X_train_Q, X_test_Q = generate_basis(X_train), generate_basis(X_test)
        X_true_Q = generate_basis(X_true)

        model = joblib.load(save_path + f"{model_name}_{n}/reg_Q.pkl")

        y_hat_train = model.predict(X_train_Q).flatten()
        y_hat_test = model.predict(X_test_Q).flatten()
        y_hat = model.predict(X_true_Q).flatten()

    else:
        model = tf.keras.models.load_model(save_path + f"{model_name}_{n}/trained_model")

        y_hat_train = model.predict(X_train).flatten()
        y_hat_test = model.predict(X_test).flatten()
        y_hat = model.predict(X_true).flatten()
    
    # Calculating MSE
    MSE_train[n] = np.mean((y_hat_train - y_train)**2)
    MSE_test[n] = np.mean((y_hat_test - y_test)**2)
    MSE_true[n] = np.mean((y_hat - y_true)**2)

    loss_hat = y_hat * y_std + y_mean
    
    q = np.sort(loss_hat)[-int(alpha * len(loss_hat))]
    
    VaR[n] = q
    CVaR[n] = np.mean(loss_hat[loss_hat > q])
    
    np.save(save_path + f"{model_name}_{n}", loss_hat)

model_name = "LSTM_HiCap_mediumNoise"

MSE_train = np.zeros(n_rep)
MSE_test = np.zeros(n_rep)
MSE_true = np.zeros(n_rep)
VaR = np.zeros(n_rep)
CVaR = np.zeros(n_rep)
VaR_sns = np.zeros(n_rep)
CVaR_sns = np.zeros(n_rep)

for n in range(n_rep):
    
    print("Macro replication:", n + 1)

    loss_sns = np.load(cwd + lossFiles_mediumNoise[n])
    
    X_train, y_train, X_test, y_test, y_mean, y_std = preprocess.transform_data(rtn, loss_LN, True, test_size, seed, model="RNN")
    
    X_true, y_true = preprocess.transform_data(rtn, loss_T, training=False, y_mean=y_mean, y_std=y_std, model="RNN")
    
    y_true = np.array(y_true).flatten()

    if "REG" in model_name:
            
        X_train_Q, X_test_Q = generate_basis(X_train), generate_basis(X_test)
        X_true_Q = generate_basis(X_true)

        model = joblib.load(save_path + f"{model_name}_{n}/reg_Q.pkl")

        y_hat_train = model.predict(X_train_Q).flatten()
        y_hat_test = model.predict(X_test_Q).flatten()
        y_hat = model.predict(X_true_Q).flatten()

    else:
        model = tf.keras.models.load_model(save_path + f"{model_name}_{n}/trained_model")

        y_hat_train = model.predict(X_train).flatten()
        y_hat_test = model.predict(X_test).flatten()
        y_hat = model.predict(X_true).flatten()
    
    # Calculating MSE
    MSE_train[n] = np.mean((y_hat_train - y_train)**2)
    MSE_test[n] = np.mean((y_hat_test - y_test)**2)
    MSE_true[n] = np.mean((y_hat - y_true)**2)

    loss_hat = y_hat * y_std + y_mean
    
    q = np.sort(loss_hat)[-int(alpha * len(loss_hat))]
    
    VaR[n] = q
    CVaR[n] = np.mean(loss_hat[loss_hat > q])
    
    q = np.sort(loss_sns)[-int(alpha * len(loss_hat))]
    
    VaR_sns[n] = q
    CVaR_sns[n] = np.mean(loss_sns[loss_sns > q])
    
    np.save(save_path + f"{model_name}_{n}", loss_hat)
    
np.save(save_path + f"{model_name}_CVaR", CVaR)
np.save(save_path + f"{model_name}_VaR", VaR)