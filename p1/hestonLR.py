import numpy as np
import pandas as pd
import os 
from joblib import Parallel, delayed

import methods


def lrEC(n_front, n, d, S_0, K, mu, sigma, r, tau, T, h):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma**2

    sample_outer_train = methods.GBM_front(n_front=n_front, d=d,
                                   S_0=S_0, drift_vec=mu, diffusion_mat=cov_mat, tau=tau)
    sample_inner_tau = methods.GBM_back(n_front=n_front, n_back=1, d=d,
                                S_tau=sample_outer_train, drift=r, diffusion=cov_mat, T=h)[:, :, 0]
    sample_inner_T = methods.GBM_back(n_front=n_front, n_back=1, d=d,
                              S_tau=sample_inner_tau, drift=r, diffusion=cov_mat, T=T-tau-h)
    sample_outer_test = methods.GBM_front(n_front=n_front, d=d,
                                          S_0=S_0, drift_vec=mu, diffusion_mat=cov_mat, tau=tau)

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(sample_inner_T - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    diff = portfolio_value_0 - payoff
    loss = np.zeros(n_front)

    Weight_D = methods.compute_Weight_D(S_0, mu, r, sigma, h, tau, sample_inner_tau)

    n_partition = int(n_front // n)

    for i in range(n_partition):

        LR_Loss_Matrix = np.zeros([n_front, n])

        for j in range(d):

            Weight_U = methods.compute_Weight_U(sample_inner_tau[j, :], sample_outer_test[j, i * n:(i + 1) * n],
                                                r, sigma, h)

            Weight = np.sqrt((tau + h) / h) * np.exp(Weight_D[j, :].reshape(-1, 1) - Weight_U)

            LR_Loss_Matrix = LR_Loss_Matrix + diff[j, :].reshape(-1, 1) * Weight

        loss[i * n:(i + 1) * n] = np.mean(LR_Loss_Matrix, axis=0)

    return loss


def calRM(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h=3/50, H=120, L0=100, 
          optionName="European", level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):

    if optionName == "European":
        loss = lrEC(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h)

    loss.sort()

    indicator = np.mean((loss > L0))
    hockey = np.mean(np.maximum(loss - L0, 0))
    quadratic = np.mean((loss - L0) ** 2)

    VaR = {}
    CVaR = {}
    for level in level_list:
        VaR[level] = loss[int(np.ceil(level * n_front)) - 1] 
        CVaR[level] = np.mean(loss[loss >= VaR[level]])
    
    RM = [indicator, hockey, quadratic]
    for level in level_list:
        RM.append(VaR[level])
        RM.append(CVaR[level])

    return RM





if __name__ == "__main__":
    optionName = input("Option Type? Please enter one of: European, Asian, BarrierUp, BarrierDown:")
    d = int(input("Please enter dimension:"))
    n_jobs = int(input("Please enter number of jobs:"))

    trueValueFolder = "./trueValues/"
    saveFolder = "./result/"

    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)

    S0 = 100
    K = [90, 100, 110]
    mu = 0.08
    r = 0.05
    tau = 3/50
    T = 1
    sigma = 0.1
    h = 1/50

    rho = 0.3
    kappa = 2
    theta = 0.04
    V0 = 0.04

    M_list = [int(1000 * (2**i)) for i in range(11)]
    N_list = [1] * len(M_list)

    n_rep = 1000

    trueValues = pd.read_csv(f"{trueValueFolder}trueValue_European.csv")

    L0 = trueValues["VaR_0.9"].values[0]
    trueValues = np.array(trueValues).flatten()[1:]

    result_table = {}

    for M, N in zip(M_list, N_list):

        temp = Parallel(n_jobs=n_jobs, verbose=1)(delayed(calRM)(M, N, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h, L0)
                                                for j in range(n_rep))
        temp = np.array(temp)

        Bias = np.mean(temp, axis=0) - trueValues
        Variance = np.var(temp, axis=0)

        MSE = np.array(Bias ** 2 + Variance).astype(float)

        RRMSE = np.sqrt(MSE) / trueValues

        result_table[M] = RRMSE

        print(f"LR Done for Heston, M={M}, N={N}")

        result_df = pd.DataFrame(result_table)
        print(result_df)
        result_df.to_csv(f"{saveFolder}LR_Heston.csv")


