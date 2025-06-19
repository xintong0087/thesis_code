import preprocess, fit_nn

import numpy as np
import tensorflow as tf

import os

model_name = "LSTM_LoCap"

architecture = "LSTM"
activation_function = "tanh"
lr = 0.001
dropout = 0.1
decay_rate = 0.9
patience = 50

if model_name == "LSTM_LoCap":
    recurrent_layer_size = [32, 4]
    dense_layer_size = 32
elif model_name == "LSTM_HiCap":
    recurrent_layer_size = [128, 16]
    dense_layer_size = 64
else:
    raise ValueError(f"Invalid model name: {model_name}")

n_epochs = 1000
batch_size = 4096

VA_types = ["GMWB", "GMMB"]
lapse_types = ["nolapse", "lapse", "dlapse"]
asset_models = ["RS", "BS"]

TL = input("Transfer Learning? (y/n): ")
if TL == "y":
    TL = True
else:
    TL = False

if TL:
    
    source_VA = input(f"Enter source VA type ({', '.join(VA_types)}): ")
    target_VA = input(f"Enter target VA type ({', '.join(VA_types)}): ")
    source_lapse = input(f"Enter source lapse type ({', '.join(lapse_types)}): ")
    target_lapse = input(f"Enter target lapse type ({', '.join(lapse_types)}): ")
    source_asset = input(f"Enter source asset model ({', '.join(asset_models)}): ")
    target_asset = input(f"Enter target asset model ({', '.join(asset_models)}): ")

    tl_method = input("TL method (finetune/freeze): ")

    if tl_method == "finetune":
        freeze = False
    else:
        freeze = True

        layers_to_freeze = input("Enter layers to freeze (LSTM/Dense): ")

    source_part = 50000
    target_part = 2000

    sourceScenario = np.load(f"./simData/{source_VA}/{source_lapse}/outerScenarios_{source_VA}_{source_asset}_{source_lapse}.npy")

    if (source_asset == "BS") and (source_VA == "GMWB") and (source_lapse == "nolapse"):
        sourceLoss = np.load(f"./simData/{source_VA}/{source_lapse}/hedgingLoss_{source_VA}_{source_asset}_10000_{source_lapse}.npy")
        pretrainModelPath = f"./trainedModels/{source_VA}_PY_/{source_lapse}_BS/"
    else:
        sourceLoss = np.load(f"./simData/{source_VA}/{source_lapse}/hedgingLoss_{source_VA}_{source_asset}_100_{source_lapse}.npy")
        pretrainModelPath = f"./trainedModels/{source_VA}_PY/{source_lapse}/"

    if os.path.exists(pretrainModelPath + f"sourcePretrainModel.keras"):
        model = tf.keras.models.load_model(pretrainModelPath + f"sourcePretrainModel.keras")
    else:
        X_train, y_train, X_test, y_test, y_mean, y_std = preprocess.transform_data(sourceScenario, sourceLoss, True, 0, 22, part=True, part_size=source_part)

        model = fit_nn.build_model(X_train, architecture, recurrent_layer_size, dense_layer_size, activation_function, lr, dropout, decay_rate)
        pretrain_model, running_time = fit_nn.train_model(X_train, y_train, model, n_epochs, batch_size, patience, pretrainModelPath)
        pretrain_model.save(pretrainModelPath + f"sourcePretrainModel.keras")

    targetScenario = np.load(f"./simData/{target_VA}/{target_lapse}/outerScenarios_{target_VA}_{target_asset}_{target_lapse}.npy")
    targetLoss = np.load(f"./simData/{target_VA}/{target_lapse}/hedgingLoss_{target_VA}_{target_asset}_100_{target_lapse}.npy")

    X_train, y_train, X_test, y_test, y_mean, y_std = preprocess.transform_data(targetScenario, targetLoss, True, 0, 22, part=True, part_size=target_part)

    if freeze:
        targetModelPath = f"./trainedModels/{target_VA}_PY/{target_lapse}_LF/"

        if not os.path.exists(targetModelPath):
            os.makedirs(targetModelPath)

        lr = 0.0002
        n_epochs = 1000
        batch_size = target_part

        if layers_to_freeze == "LSTM":
            for layer in model.layers[:-1]:
                layer.trainable = False
        else:
            for layer in model.layers[-1].layers:
                layer.trainable = False

    else:
        targetModelPath = f"./trainedModels/{target_VA}_PY/{target_lapse}_FT/"

        if not os.path.exists(targetModelPath):
            os.makedirs(targetModelPath)
    

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

    filename = targetModelPath + "training_history.csv"
    logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

    history = model.fit(X_train, y_train, epochs=n_epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[callback, logger])
    
    model.save(targetModelPath + f"trained_model.keras")


else:

    VA_type = input(f"Enter VA type ({', '.join(VA_types)}): ")
    lapse_type = input(f"Enter lapse type ({', '.join(lapse_types)}): ")
    asset_model = input(f"Enter asset model ({', '.join(asset_models)}): ")

    part = input("Part? (y/n): ")
    if part == "y":
        part = True
        part_size = int(input("Enter part size: "))
    else:
        part = False
        part_size = 50000

    outerScenarioPath = f"./simData/{VA_type}/{lapse_type}/outerScenarios_{VA_type}_{asset_model}_{lapse_type}.npy"
    trueLossPath = f"./simData/{VA_type}/{lapse_type}/hedgingLoss_{VA_type}_{asset_model}_10000_{lapse_type}.npy"

    if (asset_model == "BS") and (VA_type == "GMWB") and (lapse_type == "nolapse"):
        noisyLossPath = trueLossPath
        model_path = f"./trainedModels/{VA_type}_PY_/{lapse_type}_BS/"
    else:
        noisyLossPath = f"./simData/{VA_type}/{lapse_type}/hedgingLoss_{VA_type}_{asset_model}_100_{lapse_type}.npy"
        model_path = f"./trainedModels/{VA_type}_PY/{lapse_type}/"

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    save_path = f"{model_path}{model_name}_{part_size}/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    outerScenarios = np.load(outerScenarioPath)
    trueLoss = np.load(trueLossPath)
    noisyLoss = np.load(noisyLossPath)

    X_train, y_train, X_test, y_test, y_mean, y_std = preprocess.transform_data(outerScenarios, noisyLoss, True, 0, 22, part=part, part_size=part_size)

    model = fit_nn.build_model(X_train, architecture, recurrent_layer_size, dense_layer_size, activation_function, lr, dropout, decay_rate)
    model_trained, running_time = fit_nn.train_model(X_train, y_train, model, n_epochs, batch_size, patience, save_path)

    model_trained.save(save_path + f"trained_model.keras")

