import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

import methods, methodsHeston


def Laguerre_polynomial(X, degree=3):

    X_norm = ((X - X.mean()) / X.std()).reshape(-1, 1)
    # Normalization

    L_0 = np.exp(-X_norm / 2)
    L_1 = np.exp(-X_norm / 2) * (1 - X_norm)
    L_2 = np.exp(-X_norm / 2) * (1 - 2 * X_norm + (1 / 2) * (X_norm ** 2))
    L_3 = np.exp(-X_norm / 2) * (1 - 3 * X_norm + (3 / 2) * (X_norm ** 2) - (1 / 6) * (X_norm ** 3))

    L = [L_1, L_2, L_3]

    X_train = L_0
    for k in range(degree):
        X_train = np.concatenate([X_train, L[k]], axis=1)

    return X_train


def regressionEC(M, N, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h=1/50,
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

    X_train = Laguerre_polynomial(S_front[:, -1])
    y_train = payoff.flatten()

    reg = LinearRegression().fit(X_train, y_train)
    
    S_front, _ = methods.Heston_front(M, S0, V0, mu, rho, kappa, theta, sigma, tau, h)

    X_test = Laguerre_polynomial(S_front[:, -1])
    y_test = reg.predict(X_test)

    loss = portfolio_value_0 - y_test

    return loss


def calRM(M, N, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h, L0=100,
          level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):
    
    loss = regressionEC(M, N, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h)

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

        temp = Parallel(n_jobs=n_jobs, verbose=1)(delayed(calRM)(M, N, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h, L0)
                                                for j in range(n_rep))
        temp = np.array(temp)

        Bias = np.mean(temp, axis=0) - trueValues
        Variance = np.var(temp, axis=0)

        MSE = np.array(Bias ** 2 + Variance).astype(float)

        RRMSE = np.sqrt(MSE) / trueValues

        result_table[M] = RRMSE

        print(f"Regression Done for Heston, M={M}, N={N}")

        result_df = pd.DataFrame(result_table)
        print(result_df)
        result_df.to_csv(f"{saveFolder}Regression_Heston.csv")


