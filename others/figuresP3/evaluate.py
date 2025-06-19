import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from cycler import cycler

import time

import preprocess


def tail_scenarios(vec_true, vec_hat, start=500, end=2000, step=100):

    percentage_vec = {}

    ind_true = np.argpartition(vec_true, -start)[-start:]

    for num in range(start, end, step):

        ind_hat = np.argpartition(vec_hat.flatten(), -num)[-num:]

        percentage_vec[(num/start - 1)/20] = np.intersect1d(ind_true, ind_hat).size / start

    return percentage_vec



def generate_plots(cwd, models, model_names, X_true, y_true, lim_tail=(0.6, 1.0), lim_QQ=(-20, 20), return_time=False, scale_flag=False, y_means=None, y_stds=None):

    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.locator_params(axis='y', nbins=5)
    plt.tight_layout()
    plt.subplots_adjust(left=0.5, bottom=0.5)
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    alpha = 0.05

    save_name_tail = ""

    fig_tail = plt.figure(figsize=(8, 4))
    ax_tail = fig_tail.add_subplot(1, 1, 1)


    default_cycler = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']) +
                      cycler(linestyle=['-', '--', '-.', ':']))

    ax_tail.set_prop_cycle(default_cycler)

    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.15)

    for X, y, model, model_name, y_mean, y_std in zip(X_true, y_true, models, model_names, y_means, y_stds):

        y = np.array(y).flatten()
        start = int(y.shape[0] * alpha)
        end = int(start * 4)
        step = (end - start) // 30

        # Make predictions
        start_time = time.time()
        if (model_name == "Standard Procedure (N=1000)") or (model_name == "SNS (N=1000)"):
            print("Standard Procedure (N=1000)")
            y_hat = model

            percent = tail_scenarios(y, y_hat, start=start, end=end, step=step)
            df = pd.Series(percent)

            ax_tail.axhline(y=df.iloc[0], color='tab:gray', linestyle='-', linewidth=2)
            print("Percentage of tail matches for model:", model_name, "is", df.iloc[0])
            
        else:
            print("Proxy Model:", model_name)
            y_hat = model.predict(X).flatten()

            execution_time = time.time() - start_time

            print("MSE for model:", model_name, "is", np.mean((y_hat - y)**2))

            # Tail Cutoff Plot
            percent = tail_scenarios(y, y_hat, start=start, end=end, step=step)
            df = pd.Series(percent)
            print("Percentage of tail matches for model:", model_name, "is", df)
            ax_tail.plot(df)
            ax_tail.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
            ax_tail.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

        save_name_tail = save_name_tail + str(model_name) + "_"

        # QQ-Plot
        lim_l, lim_u = lim_QQ
        fig_QQ = plt.figure(figsize=(8, 4))
        ax_QQ = fig_QQ.add_subplot(1, 1, 1)
        ax_QQ.scatter(y_hat * y_std + y_mean, y * y_std + y_mean)
        # if not scale_flag:
        ax_QQ.set_xlim(lim_l, lim_u)
        x_lim = ax_QQ.get_xlim()
        ax_QQ.plot(x_lim, x_lim, color="black")
        ax_QQ.set_xlabel('Proxy Loss')
        ax_QQ.set_ylabel('True Loss')
        ax_QQ.set_title("QQ Plot for: " + str(model_name) )
        ax_QQ.legend(["Predicted-True Scatter", "45 Degree Line"])
        plt.tight_layout()
        fig_QQ.savefig(cwd + str(model_name) + "_QQ.png")

    
    ax_tail.set_xticks(np.arange(0, 0.16, 0.05))
    lim_l, lim_u = lim_tail
    ax_tail.set_ylim(lim_l, lim_u)
    ax_tail.yaxis.get_ticklabels()[1].set_weight('bold')
    ax_tail.yaxis.get_ticklabels()[-1].set_weight('bold')
    ax_tail.legend(model_names)
    ax_tail.set_xlabel("Safety Margin")
    ax_tail.set_ylabel("Percentage of Matches")

    fig_tail.savefig(cwd + save_name_tail + "tailCutoff.png")

    print(save_name_tail)

    if return_time:
        return execution_time / y_hat.shape[0]


def plot_training_history(cwd, model_name):

    df = pd.read_csv(cwd + "training_history.csv")

    fig = plt.figure(figsize=(8, 4))
    ax_tail = fig.add_subplot(1, 1, 1)
    ax_tail.plot(df[["loss", "val_loss"]])

    ax_tail.legend(["Training Error", "Validation Error"])
    ax_tail.set_xlabel("# Epochs")
    ax_tail.set_ylabel("Mean Squared Error")
    fig.savefig(cwd + str(model_name) + "_trainingHistory.png")

    return None


def cal_CVaR(cwd, model, model_name, X, y_true, y_train, y_mean, y_std):
    plt.rcParams.update({'font.size': 12})

    alpha = 0.05

    L_true = y_true * y_std + y_mean

    fig_CVaR = plt.figure(figsize=(8, 4))
    ax_CVaR = fig_CVaR.add_subplot(1, 1, 1)

    fig_diff = plt.figure(figsize=(8, 4.5))
    ax_diff = fig_diff.add_subplot(1, 1, 1)
    default_cycler = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']) +
                      cycler(linestyle=['-', '--', '-.', ':']))

    ax_diff.set_prop_cycle(default_cycler)

    y = np.array(y_true).flatten()
    y_train = y_train.flatten()
    y_hat = model.predict(X).flatten()

    start = int(alpha * y.shape[0])
    end = int(start * 4)
    step = (end - start) // 20

    ind_true = np.argpartition(y, -start)[-start:]

    CVaR = {}
    CVaR_ = {}
    diff_p = {}
    CVaR_True = np.mean(np.sort(L_true)[-start:])

    for num in range(start, end, step):

        ind_hat = np.argpartition(y_hat.flatten(), -num)[-num:]

        ind_match, ind_1, ind_2 = np.intersect1d(ind_true, ind_hat, return_indices=True)

        y_remain = np.sort(np.delete(y_train, ind_match))
        num_remain = start - ind_match.size

        if num_remain > 0.1:

            CVaR[(num/start - 1)/20] = np.mean(np.concatenate([y[ind_match], y_remain[-num_remain:]])) * y_std + y_mean
        else:
            CVaR[(num/start - 1)/20] = CVaR_True
        # CVaR[num/start] = (np.mean(y[ind_match]) * y_std + y_mean) * ind_match.size / num \
        #                   + (np.mean(y_remain[-num_remain:]) * y_std + y_mean) * num_remain / num
        CVaR_[(num/start - 1)/20] = CVaR_True
        diff_p[(num/start - 1)/20] = np.abs(CVaR[(num / start - 1)/20] / CVaR_True - 1) * 100

    df = pd.Series(CVaR)
    ax_CVaR.plot(df)
    df_ = pd.Series(CVaR_)
    ax_CVaR.plot(df_)
    ax_CVaR.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    ax_CVaR.legend([model_name + "-Predicted CVaR", "True CVaR"])
    ax_CVaR.set_xlabel("Percentage Included for Tail Cutoff")
    ax_CVaR.set_ylabel("CVaR")
    fig_CVaR.savefig(cwd + model_name + "_CVaR.png")

    df = pd.Series(diff_p)
    ax_diff.plot(df)
    ax_diff.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    ax_diff.legend([model_name + " - Percentage Difference of the CVaR Estimate"])
    ax_diff.set_xlabel("Percentage Included for Tail Cutoff")
    ax_diff.set_ylabel("Percentage Difference")
    fig_diff.savefig(cwd + model_name + "_CVaR_diff.png")

    print(CVaR_True)
    return None
