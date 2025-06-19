import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os

import preprocess, fit_nn, evaluate

VA_type = "GMWB"
lapse_type = "nolapse"

cwd = f"../sim_VA/result/{VA_type}/{lapse_type}/"
save_path = f"./trainedModels/{VA_type}_PY/{lapse_type}/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

price = np.load(cwd + f"outerScenarios_{VA_type}_RS_{lapse_type}.npy")
rtn = (price[:, 1:] - price[:, :-1]) / price[:, :-1]

substring_SNS = f"hedgingLoss_{VA_type}_RS_1000_{lapse_type}"

substring_lowNoise = f"hedgingLoss_{VA_type}_RS_100_{lapse_type}"
substring_mediumNoise = f"hedgingLoss_{VA_type}_RS_10_{lapse_type}"
substring_highNoise = f"hedgingLoss_{VA_type}_RS_1_{lapse_type}"

lossFiles_lowNoise = [file for file in os.listdir(cwd) if substring_lowNoise in file and os.path.isfile(os.path.join(cwd, file))]
lossFiles_mediumNoise = [file for file in os.listdir(cwd) if substring_mediumNoise in file and os.path.isfile(os.path.join(cwd, file))]
lossFiles_highNoise = [file for file in os.listdir(cwd) if substring_highNoise in file and os.path.isfile(os.path.join(cwd, file))]

# LoCap LSTM, mediumNoise GMWB Datasets

model_name = "LSTM"
recurrent_layer_size = [32, 4]
dense_layer_size = 32
activation_function = "tanh"
lr = 0.001
dropout = 0.1
decay_rate = 0.9
patience = 10

test_size = 3000
seed = 22

n_epochs = 1000
batch_size = 4096

n_rep = 50

save_name = "LSTM_LoCap_mediumNoise"

for rep in range(n_rep):
    loss_file = substring_mediumNoise[rep]
    X_train, y_train, X_test, y_test, y_mean, y_std = preprocess.transform_data(rtn, np.load(cwd + loss_file), True, test_size, seed)

    # Build and train model
    model = fit_nn.build_model(X_train, 
                               model_name, recurrent_layer_size, dense_layer_size, 
                               activation_function, lr, dropout, decay_rate)

    path = save_path + f"{save_name}_{rep}/"
    model_trained, running_time = fit_nn.train_model(X_train, y_train, model, n_epochs, batch_size, patience, save_path)

    # Save model
    model_trained.save(path + f"trained_model")


# HiCap LSTM, mediumNoise GMWB Datasets
model_name = "LSTM"
recurrent_layer_size = [128, 16]
dense_layer_size = 64

save_name = "LSTM_HiCap_mediumNoise"

for rep in range(n_rep):
    loss_file = substring_mediumNoise[rep]
    X_train, y_train, X_test, y_test, y_mean, y_std = preprocess.transform_data(rtn, np.load(cwd + loss_file), True, test_size, seed)

    # Build and train model
    model = fit_nn.build_model(X_train, 
                               model_name, recurrent_layer_size, dense_layer_size, 
                               activation_function, lr, dropout, decay_rate)

    path = save_path + f"{save_name}_{rep}/"
    model_trained, running_time = fit_nn.train_model(X_train, y_train, model, n_epochs, batch_size, patience, save_path)

    # Save model
    model_trained.save(path + f"trained_model")
