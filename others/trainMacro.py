import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.linear_model import LinearRegression

import os

import preprocess, fit_nn, evaluate


def generate_basis(X):

    X_sq = X ** 2

    return np.concatenate([X, X_sq], axis=1)

def calculate_return(price):

    rtn = (price[:, 1:] - price[:, :-1]) / price[:, :-1]

    return rtn

def calculate_mse(y, y_hat):

    mse = np.mean((y.flatten() - y_hat.flatten())**2)

    return mse

def calculate_mean_std(mse_list):

    arr = np.array(mse_list)

    average = np.mean(arr)
    std = np.std(arr)

    return average, std


modelName = "LSTM"       # ["MLR", "QPR", "FNN", "RNN", "LSTM"]
noiseLevel = "low"      # ["low", "medium", "high"]
test_size = 3000

seed = 22
np.random.seed(seed)

macro = 100
start = int(input("Train from macro: "))
start -= 1

outerScenarios = np.load("./macroData/outerScenarios.npy")
trueLosses = np.load("./macroData/trueLosses.npy")

save_path = "./trainedModels/macro/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

outerScenarios = calculate_return(outerScenarios)

mse_train = []
mse_test = []
mse_true = []

for m in range(start, macro):

    print(f"Starting Macro Replication {m + 1}:")

    seed_m = np.random.randint(0, 32767)

    noisyLosses = np.load(f"./macroData/loss_{noiseLevel}Noise_{m}.npy")

    X_train, y_train, X_test, y_test, y_mean, y_std = preprocess.transform_data(outerScenarios, noisyLosses, True, test_size, seed_m, model="FNN")
    y = (noisyLosses - y_mean) / y_std

    if modelName == "MLR":
        model = LinearRegression().fit(X_train, y_train)
    if modelName == "QPR":
        if m == 0:
            X_train = generate_basis(X_train)
            X_test = generate_basis(X_test)
            outerScenarios = generate_basis(outerScenarios)

        model = LinearRegression().fit(X_train, y_train)

    if modelName == "FNN":
        
        architecture = "FFN"
        recurrent_layer_size = [128, 16]
        dense_layer_size = 64

        activation_function = "ReLU"

        lr = 0.001
        dropout = 0.1
        decay_rate = 0.9
        n_epochs = 1000
        batch_size = 4096
        patience = 25

        model = fit_nn.build_model(X_train, architecture, recurrent_layer_size, dense_layer_size, activation_function, lr, dropout, decay_rate)

        model, running_time = fit_nn.train_model(X_train, y_train, model, n_epochs, batch_size, patience, save_path)


    if modelName == "RNN":
        
        architecture = "RNN"
        recurrent_layer_size = [32, 4]
        dense_layer_size = 32

        activation_function = "ReLU"

        lr = 0.001
        dropout = 0.1
        decay_rate = 0.9
        n_epochs = 1000
        batch_size = 4096
        patience = 100

        model = fit_nn.build_model(X_train, architecture, recurrent_layer_size, dense_layer_size, activation_function, lr, dropout, decay_rate)

        model, running_time = fit_nn.train_model(X_train, y_train, model, n_epochs, batch_size, patience, save_path)

    if modelName == "LSTM":
        
        architecture = "LSTM"
        recurrent_layer_size = [32, 4]
        dense_layer_size = 32

        activation_function = "tanh"

        lr = 0.001
        dropout = 0.1
        decay_rate = 0.9
        n_epochs = 3000
        batch_size = 8192
        patience = 100

        model = fit_nn.build_model(X_train, architecture, recurrent_layer_size, dense_layer_size, activation_function, lr, dropout, decay_rate)

        model, running_time = fit_nn.train_model(X_train, y_train, model, n_epochs, batch_size, patience, save_path)

    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    y_hat = model.predict(outerScenarios)

    mse_train.append(calculate_mse(y_train, y_hat_train))
    mse_test.append(calculate_mse(y_test, y_hat_test))
    mse_true.append(calculate_mse(y, y_hat))

    print(f"MSE Train: {mse_train[-1]}, MSE Test: {mse_test[-1]}, MSE True: {mse_true[-1]}")

    np.save(f"./macroData/loss_{noiseLevel}Noise_{modelName}_{m}.npy", y_hat * y_std + y_mean)

mse_train, std_train = calculate_mean_std(mse_train)
mse_test, std_test = calculate_mean_std(mse_test)
mse_true, std_true = calculate_mean_std(mse_true)

mse_result = pd.DataFrame({"Training Error": mse_train,
                           "Test Error": mse_test,
                           "True Error": mse_true,
                           "Training SE": std_train,
                           "Test SE": std_test,
                           "True SE": std_true},
                           index=[f"{modelName}_{noiseLevel}Noise"])

mse_result.to_csv(f"./results/MSE_{noiseLevel}Noise_{modelName}.csv")

