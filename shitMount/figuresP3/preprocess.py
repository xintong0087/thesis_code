import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split


def load_data(cwd, total_size, save=True):

    if save:

        X_all = []
        y_all = []

        for k in range(total_size):
            name_X = cwd + "X_" + str(k) + ".csv"
            name_y = cwd + "y_" + str(k) + ".csv"

            X = pd.read_csv(name_X, index_col=0)
            y = pd.read_csv(name_y, index_col=0)

            X_all = X_all + [X]
            y_all = y_all + [y]

        X = pd.concat(X_all, axis=1)
        y = pd.concat(y_all, axis=1)

        Price = X.shift(1).dropna().T
        Return = (X - X.shift(1)) / X.shift(1)
        Return = Return.dropna().T

        Loss = y.T

        Price.to_csv(cwd + "Price.csv")
        Return.to_csv(cwd + "Return.csv")
        Loss.to_csv(cwd + "Loss.csv")

    else:

        Price = pd.read_csv(cwd + "Price.csv", index_col=0)
        Return = pd.read_csv(cwd + "Return.csv", index_col=0)
        Loss = np.array(pd.read_csv(cwd + "Loss.csv", index_col=0))

    return Price, Return, Loss


def transform_data(X, y,
                   training=True, test_size=3000, seed=22, y_mean=None, y_std=None, model="RNN",
                   part=False, part_size=10000):

    if part:
        X = X[:part_size, :]
        y = y[:part_size]

    if model == "RNN":
        X = np.expand_dims(np.array(X), axis=2)
    else:
        X = np.array(X)

    if training:
        y_mean = y.mean()
        y_std = y.std()
    else:
        y = np.array(y).flatten()

    y_norm = (y - y_mean) / y_std

    if training:

        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y_norm, test_size=test_size, random_state=seed)
        else:
            X_train, X_test = X, X
            y_train, y_test = y_norm, y_norm

        print("Data is split, training and test data has shape:")
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        return X_train, y_train, X_test, y_test, y_mean, y_std

    else:

        return X, y_norm


def load_data_npy(cwd, raw=True, data_size=241, data_type="train"):

    """
    :param cwd: directory to load data from
    :param raw: whether data is raw (in .npy format)
    :param data_size: number of time period in raw data
    :param data_type: "train" or "test"
    :return: Price: original outer path, without with initial asset price
             Return: asset return
             Loss: total hedging loss
    """

    if raw:

        if data_type == "test":
            filenames_X = list(set(glob.glob(cwd + "*10000_outerPath.npy"))
                               - set(glob.glob(cwd + "GMAB_RS_Sensitivity*10000_outerPath.npy")))
            filenames_y = list(set(glob.glob(cwd + "*10000_hedgingLoss.npy"))
                               - set(glob.glob(cwd + "GMAB_RS_Sensitivity*10000_hedgingLoss.npy")))
        elif data_type == "sensitivity_test":
            filenames_X = glob.glob(cwd + "GMAB_RS_Sensitivity*10000_outerPath.npy")
            filenames_y = glob.glob(cwd + "GMAB_RS_Sensitivity*10000_hedgingLoss.npy")
        elif data_type == "sensitivity_train":
            filenames_X = glob.glob(cwd + "GMAB_RS_Sensitivity*100_outerPath.npy")
            filenames_y = glob.glob(cwd + "GMAB_RS_Sensitivity*100_hedgingLoss.npy")
        else:
            filenames_X = list(set(glob.glob(cwd + "*100_outerPath.npy"))
                               - set(glob.glob(cwd + "GMAB_RS_Sensitivity*100_outerPath.npy")))
            filenames_y = list(set(glob.glob(cwd + "*100_hedgingLoss.npy"))
                               - set(glob.glob(cwd + "GMAB_RS_Sensitivity*100_hedgingLoss.npy")))

        X_all = np.empty((0, data_size))

        for filename in filenames_X:
            X = np.load(filename)
            X_all = np.concatenate([X_all, X], axis=0)

        y_all = np.empty(0)

        for filename in filenames_y:
            y = np.load(filename)
            y_all = np.concatenate([y_all, y])

        Price = X_all[:, 1:]
        Return = (X_all[:, 1:] - X_all[:, :-1]) / X_all[:, :-1]
        Loss = y_all

        if data_type == "test":
            np.save(cwd + "10000_Price", Price)
            np.save(cwd + "10000_Return", Return)
            np.save(cwd + "10000_Loss", Loss)
        elif data_type == "sensitivity":
            np.save(cwd + "Sensitivity_10000_Price", Price)
            np.save(cwd + "Sensitivity_10000_Return", Return)
            np.save(cwd + "Sensitivity_10000_Loss", Loss)
        else:
            np.save(cwd + "100_Price", Price)
            np.save(cwd + "100_Return", Return)
            np.save(cwd + "100_Loss", Loss)

    else:

        if data_type == "test":
            Price = np.load(cwd + "10000_Price")
            Return = np.load(cwd + "10000_Return")
            Loss = np.load(cwd + "10000_Loss")
        elif data_type == "sensitivity":
            Price = np.load(cwd + "Sensitivity_10000_Price")
            Return = np.load(cwd + "Sensitivity_10000_Return")
            Loss = np.load(cwd + "Sensitivity_10000_Loss")
        else:
            Price = np.load(cwd + "100_Price")
            Return = np.load(cwd + "100_Return")
            Loss = np.load(cwd + "100_Loss")

    return Price, Return, Loss
