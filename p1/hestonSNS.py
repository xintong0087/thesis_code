import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

import methods, methodsHeston


def snsEC(M, N, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h=1/50,
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

    y_train = payoff.flatten()

    loss = portfolio_value_0 - y_train

    return loss


def bootstrapEC(Gamma, M_list, N_list, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h=1/50, L0=100, I=500,
                level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):

    n_estimate = len(level_list)*2 + 3

    portfolio_value_0 = 0
    for k in K:
        portfolio_value_0 += methodsHeston.priceHeston(S0, k, T-tau, r, kappa, theta, rho, sigma, V0, "C")

    M0 = M_list[-1]
    N0 = N_list[-1]

    S_front, V_front = methods.Heston_front(M0, S0, V0, mu, rho, kappa, theta, sigma, tau, h)
    S_back, _ = methods.Heston_back(M0, N0, S_front, V_front, r, rho, kappa, theta, sigma, T-tau, h)

    sample_inner_0 = S_back[:, :, -1]

    M_shape = len(M_list)
    alpha_mat = np.zeros([M_shape, n_estimate])

    for m, N in zip(range(M_shape), N_list):

        res = np.zeros([I, n_estimate])

        for i in range(I):

            index_outer = np.random.choice(M0, size=M0, replace=True)
            index_inner = np.random.choice(N0, size=N, replace=True)
            temp = sample_inner_0[index_outer, :]
            sample_inner_bs = temp[:, index_inner]

            payoff = np.zeros(M0)
            for j in range(len(K)):
                price = np.mean(np.maximum(sample_inner_bs - K[j], 0), axis=1) * np.exp(-r * (T - tau))
                payoff += price

            loss = portfolio_value_0 - payoff.flatten()

            res[i, 0] = np.nanmean((loss > L0))
            res[i, 1] = np.nanmean(np.maximum(loss - L0, 0))
            res[i, 2] = np.nanmean((loss - L0) ** 2)       
                
            loss.sort()

            for k in range(len(level_list)):
                res[i, 3 + 2*k] = loss[int(np.ceil(level_list[k] * M0)) - 1]
                res[i, 4 + 2*k] = np.nanmean(loss[loss >= res[i, 3 + 2*k]])

        alpha_mat[m, :] = np.nanmean(res, axis=0)

    N_shape = len(N_list)
    s_mat = np.zeros([N_shape, n_estimate])

    for n, M in zip(range(N_shape), M_list):

        res = np.zeros([I, n_estimate])

        for i in range(I):

            sample_inner_bs = sample_inner_0[np.random.choice(M0, size=M, replace=True), :]

            payoff = np.zeros(M)
            for j in range(len(K)):
                price = np.mean(np.maximum(sample_inner_bs - K[j], 0), axis=1) * np.exp(-r * (T - tau))
                payoff += price

            loss = portfolio_value_0 - payoff.flatten()

            res[i, 0] = np.nanmean((loss > L0))
            res[i, 1] = np.nanmean(np.maximum(loss - L0, 0))
            res[i, 2] = np.nanmean((loss - L0) ** 2)       
                
            loss.sort()

            for k in range(len(level_list)):
                res[i, 3 + 2*k] = loss[int(np.ceil(level_list[k] * M)) - 1]
                res[i, 4 + 2*k] = np.nanmean(loss[loss >= res[i, 3 + 2*k]])
        
        s_mat[n, :] = np.var(res, axis=0)
    
    M_opt = []
    N_opt = []
    M_array = np.array(M_list).reshape(-1, 1)
    N_array = np.array(N_list).reshape(-1, 1)
    
    for k in range(n_estimate):
        reg_A = LinearRegression().fit(1 / N_array, alpha_mat[:, k])
        reg_B = LinearRegression().fit(1 / M_array, s_mat[:, k])
        A = reg_A.coef_[0]
        B = reg_B.coef_[0]

        M_opt.append(int((B / (2 * A**2)) ** (1/3) * Gamma**(2/3)))
        N_opt.append(int(((2 * A**2) / B) ** (1/3) * Gamma**(1/3)))

    return M_opt, N_opt



def calRM(M_opt, N_opt, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h, L0=100,
          level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):

    VaR = {}
    CVaR = {}

    loss = snsEC(M_opt[0], N_opt[0], S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h)
    indicator = np.mean((loss > L0))

    loss = snsEC(M_opt[1], N_opt[1], S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h)
    hockey = np.mean(np.maximum(loss - L0, 0))

    loss = snsEC(M_opt[2], N_opt[2], S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h)
    quadratic = np.mean((loss - L0) ** 2)

    for k in range(len(level_list)):
        level = level_list[k]
        loss = snsEC(M_opt[3 + 2*k], N_opt[3 + 2*k], S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h)
        loss.sort()
        VaR[level] = loss[int(np.ceil(level * M_opt[3 + 2*k])) - 1] 
        loss = snsEC(M_opt[4 + 2*k], N_opt[4 + 2*k], S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h)
        loss.sort()
        Q = loss[int(np.ceil(level * M_opt[4 + 2*k])) - 1]
        CVaR[level] = np.mean(loss[loss >= Q])

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

    Gamma_list = [int(10000 * (2**i)) for i in range(11)]

    # Parameters for bootstrap
    M_array = np.arange(50, 101, 5)
    N_array = np.arange(50, 101, 5)
    I = 500

    n_rep = 1000

    trueValues = pd.read_csv(f"{trueValueFolder}trueValue_European.csv")

    L0 = trueValues["VaR_0.9"].values[0]
    trueValues = np.array(trueValues).flatten()[1:]

    result_table = {}

    for Gamma in Gamma_list:

        M_opt, N_opt = bootstrapEC(Gamma, M_array, N_array, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h, L0, I)

        temp = Parallel(n_jobs=n_jobs, verbose=1)(delayed(calRM)(M_opt, N_opt, S0, V0, K, mu, rho, kappa, theta, sigma, r, tau, T, h, L0)
                                                for j in range(n_rep))
        temp = np.array(temp)

        Bias = np.mean(temp, axis=0) - trueValues
        Variance = np.var(temp, axis=0)

        MSE = np.array(Bias ** 2 + Variance).astype(float)

        RRMSE = np.sqrt(MSE) / trueValues

        result_table[Gamma] = RRMSE

        print(f"SNS Done for Heston, Gamma={Gamma}")

        result_df = pd.DataFrame(result_table)
        print(result_df)
        result_df.to_csv(f"{saveFolder}SNS_Heston.csv")


