import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from cycler import cycler


def generateQQ(loss, loss_pred, 
               lim_QQ=(-400, 800), 
               figSize=(8, 6), 
               figName="QQ", 
               xLabel="True Losses",
               yLabel="Predicted Losses"):

    lim_l, lim_u = lim_QQ
    fig_QQ = plt.figure(figsize=figSize)
    ax_QQ = fig_QQ.add_subplot(1, 1, 1)
    ax_QQ.scatter(loss, loss_pred)
    ax_QQ.set_xlim(lim_l, lim_u)
    x_lim = ax_QQ.get_xlim()
    ax_QQ.plot(x_lim, x_lim, color="grey", linestyle="--")
    ax_QQ.set_xlabel(xLabel)
    ax_QQ.set_ylabel(yLabel)
    ax_QQ.legend(["Predicted-Training Scatter", "45 Degree Line"], loc="upper left")
    plt.tight_layout()
    fig_QQ.savefig(f"./figuresP2/{figName}.png")

    return None

def generateTM(loss, loss_pred,
               figName="TM"):
    
    return None

def generateTabPlt():

    # Table 3.1 MSEs for all metamodels
    metamodels = ["MLR", "QPR", "FNN"]
    for metamodel in metamodels:
        mse = pd.read_csv(f"./results/MSE_lowNoise_{metamodel}.csv", index_col=0)
        # Change to scientific notation
        print(mse)
        mse = mse.map(lambda x: f"{x:.2e}")
        print(mse)

    # Figure 3.2 QQ plots for two RNNs

    loss = np.load("./macroData/loss_lowNoise_1.npy")
    loss_pred = np.load("./macroData/loss_lowNoise_RNN_1.npy")

    generateQQ(loss, loss_pred, figName="2a_QQ_good_training", xLabel="Training Losses")

    loss = np.load("./macroData/loss_lowNoise_56.npy")
    loss_pred = np.load("./macroData/loss_lowNoise_RNN_56.npy")

    generateQQ(loss, loss_pred, figName="2b_QQ_bad_training", xLabel="Training Losses")

    loss = np.load("./macroData/trueLosses.npy")
    loss_pred = np.load("./macroData/loss_lowNoise_RNN_56.npy")

    generateQQ(loss, loss_pred, figName="2c_QQ_bad_true", xLabel="True Losses")

    loss = np.load("./macroData/trueLosses.npy")
    loss_pred = np.load("./macroData/loss_lowNoise_RNN_1.npy")

    generateQQ(loss, loss_pred, figName="2d_QQ_good_true", xLabel="True Losses")

    return None


generateTabPlt()