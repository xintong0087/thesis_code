import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

import methods, methodsHeston


def kernelEC(M, N, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h=1/50, k_opt=50,
                 level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):
    
    portfolio_value_0 = 0
    for k in K:
        portfolio_value_0 += methodsHeston.priceHeston(S0, k, T-tau, r, kappa, theta, rho, sigma, V0, "C")

    S_front, V_front = methods.Heston_front(M, S0, V0, mu, rho, kappa, theta, sigma, tau, h)

    S_back, _ = methods.Heston_back(M, N, S_front, V_front, r, rho, kappa, theta, sigma, T-tau, h)

    S_T = S_back[:, :, -1]

    payoff = np.zeros(M)
    for j in range(len(K)):
        price = np.mean(np.maximum(S_T - K[j], 0), axis=1) * np.exp(-r * (T - tau))
        payoff += price

    X_train = S_front[:, -1].reshape(-1, 1)
    y_train = payoff.flatten()

    knn = KNeighborsRegressor(n_neighbors=k_opt).fit(X_train, y_train)
    
    S_front, _ = methods.Heston_front(M, S0, V0, mu, rho, kappa, theta, sigma, tau, h)

    X_test = S_front[:, -1].reshape(-1, 1)
    y_test = knn.predict(X_test)

    loss = portfolio_value_0 - y_test

    return loss


def cvEC(M, N, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h=1/50):
    
    portfolio_value_0 = 0
    for k in K:
        portfolio_value_0 += methodsHeston.priceHeston(S0, k, T-tau, r, kappa, theta, rho, sigma, V0, "C")

    S_front, V_front = methods.Heston_front(M, S0, V0, mu, rho, kappa, theta, sigma, tau, h)

    S_back, _ = methods.Heston_back(M, N, S_front, V_front, r, rho, kappa, theta, sigma, T-tau, h)

    S_T = S_back[:, :, -1]

    payoff = np.zeros(M)
    for j in range(len(K)):
        price = np.mean(np.maximum(S_T - K[j], 0), axis=1) * np.exp(-r * (T - tau))
        payoff += price

    X_train = S_front[:, -1].reshape(-1, 1)
    y_train = payoff.flatten()

    cv_kf = KFold(n_splits=5)
    k_range = np.linspace(5, min(np.floor(M/10), np.sqrt(M)), 10).astype(int)
    n_k = k_range.shape[0]
    cv_score = np.zeros(n_k)

    for k in range(n_k):

        for train_ind, val_ind in cv_kf.split(X_train, y_train):
            X = X_train[train_ind]
            X_val = X_train[val_ind]
            y = y_train[train_ind]
            y_val = y_train[val_ind]

            y_hat = KNeighborsRegressor(n_neighbors=k_range[k]).fit(X, y).predict(X_val)
            cv_score[k] = cv_score[k] + np.sum((y_hat - y_val) ** 2)

    k_opt = k_range[np.argmin(cv_score)]

    return k_opt


def calRM(M, N, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h, L0=100, k_opt=50,    
          level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):

    loss = kernelEC(M, N, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h, k_opt)

    loss.sort()

    indicator = np.mean(loss > L0)
    hockey = np.mean(np.maximum(loss - L0, 0))
    quadratic = np.mean((loss - L0) ** 2)

    VaR = {}
    CVaR = {}
    for level in level_list:
        VaR[level] = loss[int(np.ceil(level * M)) - 1]
        CVaR[level] = np.mean(loss[loss >= VaR[level]])

    RM = [indicator, hockey, quadratic]
    for level in level_list:
        RM.append(VaR[level])
        RM.append(CVaR[level])

    return RM


if __name__ == "__main__":
    n_jobs = int(input("Please enter the number of jobs: "))

    trueValueFolder = "./trueValueHeston/"
    saveFolder = "./resultHeston/"

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

        k_opt = cvEC(M, N, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h)

        print(f"End of CV, optimal # of neighbors is {k_opt}.")

        temp = Parallel(n_jobs=n_jobs, verbose=1)(delayed(calRM)(M, N, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h, L0, k_opt)
                                                for j in range(n_rep))
        temp = np.array(temp)

        Bias = np.mean(temp, axis=0) - trueValues
        Variance = np.var(temp, axis=0)

        MSE = np.array(Bias ** 2 + Variance).astype(float)

        RRMSE = np.sqrt(MSE) / trueValues

        result_table[M] = RRMSE

        print(f"Kernel Done for Heston, M={M}, N={N}")

        result_df = pd.DataFrame(result_table)
        print(result_df)
        result_df.to_csv(f"{saveFolder}Kernel_Heston.csv")


