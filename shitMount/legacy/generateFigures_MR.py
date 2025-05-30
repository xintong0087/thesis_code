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

model_name = "RNN_LoCap_LowNoise"

print(model_name)
MSE_train = np.zeros(n_rep)
MSE_test = np.zeros(n_rep)
MSE_true = np.zeros(n_rep)
df_percent = pd.DataFrame()
df_CVaR = pd.DataFrame()

for n in range(n_rep):
    
    print("Macro replication:", df_percent.shape[1] + 1)

    loss_LN = np.load(cwd + lossFiles_lowNoise[n])
    
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

    # Calculating percent of tail matches
    start = int(y_hat.shape[0] * alpha)
    end = int(y_hat.shape[0] + 1)
    step = (end - start) // 50

    percent = evaluate.tail_scenarios(y_true, y_hat, start, end, step)
    s = pd.Series(percent)
    df_percent = pd.concat([df_percent, s], axis=1)

    # Calculating CVaR
    CVaR = {}
    ind_hat_sorted = np.argsort(y_hat)[::-1]

    loss_sns = np.load(cwd + lossFiles_SNS[n])
    _, _, _, _, y_mean_sns, y_std_sns = preprocess.transform_data(rtn, loss_sns, True, test_size, seed, model="RNN")
    _, y_sns = preprocess.transform_data(rtn, loss_sns, training=False, y_mean=y_mean_sns, y_std=y_std_sns, model="FNN")

    for num in range(start, end, step):

        y_eval_sorted = np.sort(y_sns[ind_hat_sorted][:num])[::-1][:start]

        CVaR[(num/start - 1)/20] = np.mean(y_eval_sorted) * y_std_sns + y_mean_sns

    s = pd.Series(CVaR)
    df_CVaR = pd.concat([df_CVaR, s], axis=1)
        
# np.save(save_path + f"{model_name}_MSE.npy", MSE)
# df_percent.to_csv(save_path + f"{model_name}_percent.csv", index=False)
# df_CVaR.to_csv(save_path + f"{model_name}_CVaR.csv", index=False)

# np.save(save_path + f"{model_name}_MSE.npy", MSE_true)
# np.save(save_path + f"{model_name}_MSE_train.npy", MSE_train)
# np.save(save_path + f"{model_name}_MSE_test.npy", MSE_test)
df_percent.to_csv(save_path + f"{model_name}_Q_percent_100.csv", index=False)
df_CVaR.to_csv(save_path + f"{model_name}_Q_CVaR_100.csv", index=False)


# model_name = "LSTM_HiCap_highNoise"

# MSE_train = np.zeros(n_rep)
# MSE_test = np.zeros(n_rep)
# MSE_true = np.zeros(n_rep)
# df_percent = pd.DataFrame()
# df_CVaR = pd.DataFrame()

# for n in range(n_rep):
    
#     print("Macro replication:", df_percent.shape[1] + 1)

#     loss_LN = np.load(cwd + lossFiles_highNoise[n])
    
#     X_train, y_train, X_test, y_test, y_mean, y_std = preprocess.transform_data(rtn, loss_LN, True, test_size, seed, model="RNN")
    
#     X_true, y_true = preprocess.transform_data(rtn, loss_T, training=False, y_mean=y_mean, y_std=y_std, model="RNN")
    
#     y_true = np.array(y_true).flatten()

#     if "REG" in model_name:
            
#         X_train_Q, X_test_Q = generate_basis(X_train), generate_basis(X_test)
#         X_true_Q = generate_basis(X_true)

#         model = joblib.load(save_path + f"{model_name}_{n}/reg_L.pkl")

#         y_hat_train = model.predict(X_train).flatten()
#         y_hat_test = model.predict(X_test).flatten()
#         y_hat = model.predict(X_true).flatten()

#     else:
#         model = tf.keras.models.load_model(save_path + f"{model_name}_{n}/trained_model")

#         y_hat_train = model.predict(X_train).flatten()
#         y_hat_test = model.predict(X_test).flatten()
#         y_hat = model.predict(X_true).flatten()
    
#     # Calculating MSE
#     MSE_train[n] = np.mean((y_hat_train - y_train)**2)
#     MSE_test[n] = np.mean((y_hat_test - y_test)**2)
#     MSE_true[n] = np.mean((y_hat - y_true)**2)

#     # Calculating percent of tail matches
#     start = int(y_hat.shape[0] * alpha)
#     end = int(start * 4)
#     step = (end - start) // 30

#     percent = evaluate.tail_scenarios(y_true, y_hat, start, end, step)
#     s = pd.Series(percent)
#     df_percent = pd.concat([df_percent, s], axis=1)

#     # Calculating CVaR
#     CVaR = {}
#     ind_hat_sorted = np.argsort(y_hat)[::-1]

#     loss_sns = np.load(cwd + lossFiles_SNS[n])
#     _, _, _, _, y_mean_sns, y_std_sns = preprocess.transform_data(rtn, loss_sns, True, test_size, seed, model="RNN")
#     _, y_sns = preprocess.transform_data(rtn, loss_sns, training=False, y_mean=y_mean_sns, y_std=y_std_sns, model="FNN")

#     for num in range(start, end, step):

#         y_eval_sorted = np.sort(y_sns[ind_hat_sorted][:num])[::-1][:start]

#         CVaR[(num/start - 1)/20] = np.mean(y_eval_sorted) * y_std_sns + y_mean_sns

#     s = pd.Series(CVaR)
#     df_CVaR = pd.concat([df_CVaR, s], axis=1)
        
# # np.save(save_path + f"{model_name}_MSE.npy", MSE)
# # df_percent.to_csv(save_path + f"{model_name}_percent.csv", index=False)
# # df_CVaR.to_csv(save_path + f"{model_name}_CVaR.csv", index=False)

# np.save(save_path + f"{model_name}_MSE.npy", MSE_true)
# np.save(save_path + f"{model_name}_MSE_train.npy", MSE_train)
# np.save(save_path + f"{model_name}_MSE_test.npy", MSE_test)
# df_percent.to_csv(save_path + f"{model_name}_percent.csv", index=False)
# df_CVaR.to_csv(save_path + f"{model_name}_CVaR.csv", index=False)