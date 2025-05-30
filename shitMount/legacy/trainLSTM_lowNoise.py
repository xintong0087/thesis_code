import numpy as np
import os
import preprocess, fit_nn

VA_type = "GMWB"
lapse_type = "nolapse"
asset_model = "RS"

part = True

if part:
    part_size = 2000
else:
    part_size = 50000

if VA_type == "GMAB":
    cwd = f"../sim_VA/result/GMAB/"
else:
    cwd = f"../sim_VA/result/{VA_type}/{lapse_type}/"

save_path = f"./trainedModels/{VA_type}_PY/{lapse_type}/"

if asset_model == "BS":
    cwd = f"../sim_VA/result_BS/"
    save_path = f"./trainedModels/{VA_type}_PY/{lapse_type}_BS/"

# make path if not exist, nested directory
if not os.path.exists(save_path):
    os.makedirs(save_path)

price = np.load(cwd + f"outerScenarios_{VA_type}_{asset_model}_{lapse_type}.npy")
rtn = (price[:, 1:] - price[:, :-1]) / price[:, :-1]

substring = f"hedgingLoss_{VA_type}_{asset_model}_100_{lapse_type}"
loss_file = [file for file in os.listdir(cwd) if substring in file and os.path.isfile(os.path.join(cwd, file))][0]

# LoCap LSTM, LowNoise GMWB Datasets

model_name = "LSTM"
recurrent_layer_size = [32, 4]
dense_layer_size = 32
activation_function = "tanh"
lr = 0.0005
dropout = 0.1
decay_rate = 0.9
patience = 30

test_size = 0
seed = 22

n_epochs = 1000
batch_size = 4096

loss = np.load(cwd + loss_file)

# loss = np.load("./GMMB.npy")
# save_path = f"./trainedModels/GMMB_PY/nolapse/"

save_name = "LSTM_LoCap_LowNoise"

X_train, y_train, X_test, y_test, y_mean, y_std = preprocess.transform_data(rtn, loss, True, test_size, seed,
                                                                            part=part, part_size=part_size)

# Build and train model
model = fit_nn.build_model(X_train, 
                            model_name, recurrent_layer_size, dense_layer_size, 
                            activation_function, lr, dropout, decay_rate)

path = save_path + f"{save_name}_{part_size}/"
if not os.path.exists(path):
    os.makedirs(path)
model_trained, running_time = fit_nn.train_model(X_train, y_train, model, n_epochs, batch_size, patience, path)

# Save model
model_trained.save(path + f"trained_model.keras")


# HiCap LSTM, LowNoise GMWB Datasets
# model_name = "LSTM"
# recurrent_layer_size = [128, 16]
# dense_layer_size = 64

# save_name = "LSTM_HiCap_LowNoise"

# X_train, y_train, X_test, y_test, y_mean, y_std = preprocess.transform_data(rtn, np.load(cwd + loss_file), True, test_size, seed,
#                                                                             part=False, part_size=part_size)

# # Build and train model
# model = fit_nn.build_model(X_train, 
#                             model_name, recurrent_layer_size, dense_layer_size, 
#                             activation_function, lr, dropout, decay_rate)

# path = save_path + f"{save_name}_{part_size}/"
# if not os.path.exists(path):
#     os.makedirs(path)
# model_trained, running_time = fit_nn.train_model(X_train, y_train, model, n_epochs, batch_size, patience, path)

# # Save model
# model_trained.save(path + f"trained_model.keras")

