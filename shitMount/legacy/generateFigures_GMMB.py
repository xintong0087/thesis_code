import preprocess
import evaluate

import joblib
import numpy as np
import tensorflow as tf
import os

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def generate_basis(X):

    X_sq = X ** 2

    return np.concatenate([X, X_sq], axis=1)

# Load data
cwd = "../sim_VA/result/GMMB/dlapse/"
save_path = "./trainedModels/GMMB_PY/"
Price = np.load(cwd + "outerScenarios_GMMB_RS_dlapse.npy")
Loss = np.load(cwd + "hedgingLoss_GMMB_RS_100_dlapse.npy")
Return = (Price[:, 1:] - Price[:, :-1]) / Price[:, :-1]

# Load Train and Test Dataset
test_size = 3000
seed = 22
flag_training = True
X_train, y_train, X_test, y_test, y_mean, y_std = preprocess.transform_data(Return, Loss, flag_training, test_size, seed, model="FFN")
X_train_Q, X_test_Q = generate_basis(X_train), generate_basis(X_test)
X_train_RNN, y_train, X_test_RNN, y_test, y_mean, y_std = preprocess.transform_data(Return, Loss, flag_training, test_size, seed, model="RNN")

# Load True Dataset
Price = np.load(cwd + "outerScenarios_GMMB_RS_dlapse.npy")
Loss = np.load(cwd + "hedgingLoss_GMMB_RS_10000_dlapse.npy")
Return = (Price[:, 1:] - Price[:, :-1]) / Price[:, :-1]
X_true, y_true = preprocess.transform_data(Return, Loss, training=False, y_mean=y_mean, y_std=y_std, model="FFN")
X_true_Q = generate_basis(X_true)
X_true_RNN, _ = preprocess.transform_data(Return, Loss, training=False, y_mean=y_mean, y_std=y_std, model="RNN")

# Load models
save_path_REG = save_path + "REG/"
save_path_FNN = save_path + "FFN_128-32_ReLU/"
save_path_RNN = save_path + "RNN_32-4_tanh/"
save_path_LSTM = save_path + "LSTM_32-4_tanh/"

model_trained_reg_L = joblib.load(save_path_REG + "reg_L.pkl")
model_trained_reg_Q = joblib.load(save_path_REG + "reg_Q.pkl")
model_trained_FNN = tf.keras.models.load_model(save_path_FNN + "trained_model")
model_trained_RNN = tf.keras.models.load_model(save_path_RNN + "trained_model")
model_trained_LSTM = tf.keras.models.load_model(save_path_LSTM + "trained_model")

# Save path for figures
save_path_figures = "./figures_ssc/GMMB/"

if not os.path.exists(save_path_figures):
    os.makedirs(save_path_figures)

# evaluate.generate_plots(save_path_figures,
#                         [model_trained_reg_Q, model_trained_FNN, model_trained_RNN, model_trained_LSTM],
#                         ["QPR", "FNN", "RNN", "LSTM"],
#                         [X_train_Q, X_train, X_train_RNN, X_train_RNN],
#                         [y_train, y_train, y_train, y_train],
#                         lim_tail=(0.40, 1.0), lim_QQ=(-200, 300), scale_flag=True, y_means=[y_mean] * 4, y_stds=[y_std] * 4)

evaluate.generate_plots(save_path_figures,
                        [model_trained_reg_Q, model_trained_FNN, model_trained_RNN, model_trained_LSTM],
                        ["QPR", "FNN", "RNN", "LSTM"],
                        [X_true_Q, X_true, X_true_RNN, X_true_RNN],
                        [y_true, y_true, y_true, y_true],
                        lim_tail=(0.40, 1.0), lim_QQ=(-200, 300), scale_flag=True, y_means=[y_mean] * 4, y_stds=[y_std] * 4)

