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


def lr_GA(n_front, n, d, S_0, K, mu, sigma, r, tau, T, h):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    sample_outer_train = methods.GBM_front(n_front=n_front, d=d,
                                   S_0=S_0, drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                   step_size=h, path=True)
    n_step_outer = sample_outer_train.shape[2] - 1

    sample_inner = methods.GBM_back(n_front=n_front, n_back=1, d=d, S_tau=sample_outer_train,
                            drift=r, diffusion=cov_mat, T=T - tau,
                            step_size=h, path=True)
    n_step_inner = sample_inner.shape[3] - 1
    sample_inner_tau = sample_inner[:, :, :, 1].reshape([d, n_front])

    geometric_sum_inner = np.prod(sample_inner[:, :, :, 1:], axis=3).reshape([d, n_front])

    sample_outer_test = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                  drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                  step_size=h, path=True)
    sample_outer_test_tau = sample_outer_test[:, :, -1]
    geometric_sum_outer_test = np.prod(sample_outer_test[:, :, 1:], axis=2)

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_Asian_G(S=S_0, T=T, sigma=sigma, r=r, K=K[j],
                                                              args=n_step_outer + n_step_inner)

    Weight_D = methods.compute_Weight_D(S_0, mu, r, sigma, h, tau, sample_inner_tau)

    loss = np.zeros(n_front)

    n_partition = int(n_front // n)

    for i in range(n_partition):

        LR_Loss_Matrix = np.zeros([n_front, n])

        for j in range(d):

            geometric_sum_partitioned = geometric_sum_inner[j, :].reshape(-1, 1) \
                                        * geometric_sum_outer_test[j, i * n:(i + 1) * n].reshape(1, -1)
            geometric_average_partitioned = geometric_sum_partitioned ** (1 / (n_step_outer + n_step_inner))

            payoff = np.zeros([n_front, n])
            for k in range(len(K)):
                price = np.maximum(geometric_average_partitioned - K[k], 0) * np.exp(-r * (T - tau))
                payoff = payoff + price

            diff = portfolio_value_0 - payoff

            Weight_U = methods.compute_Weight_U(sample_inner_tau[j, :], sample_outer_test_tau[j, i * n:(i + 1) * n],
                                        r, sigma, h)

            Weight = np.sqrt((tau + h) / h) * np.exp(Weight_D[j, :].reshape(-1, 1) - Weight_U)

            LR_Loss_Matrix = LR_Loss_Matrix + diff * Weight

        loss[i * n:(i + 1) * n] = np.mean(LR_Loss_Matrix, axis=0)

    return loss


def lr_BU(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, U, _n=1):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, U, 0, 0,
                                                                      option_type=np.array([["O", "U", "C"]]))

    sample_outer_train = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                   drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                   step_size=h, path=True)
    n_step_outer = sample_outer_train.shape[2] - 1

    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_max = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_max[:, :, n] = np.exp((np.log(sample_outer_train[:, :, n] * sample_outer_train[:, :, n + 1])
                                            + np.sqrt(
                    np.log(sample_outer_train[:, :, n + 1] / sample_outer_train[:, :, n]) ** 2
                    - 2 * sigma ** 2 * h * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    sample_inner = methods.GBM_back(n_front=n_front, n_back=1, d=d, S_tau=sample_outer_train,
                            drift=r, diffusion=cov_mat, T=T - tau,
                            step_size=h, path=True)

    n_step_inner = sample_inner.shape[3] - 1

    sample_inner_tau = sample_inner[:, :, :, 1].reshape([d, n_front])
    sample_inner_T = sample_inner[:, :, :, -1].reshape([d, n_front])

    inner_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_back, n_step_inner])
    sample_inner_max = np.zeros_like(inner_prob_knock_out)

    for i in range(n_back):
        for n in range(n_step_inner):
            sample_inner_max[:, :, i, n] = np.exp((np.log(sample_inner[:, :, i, n] * sample_inner[:, :, i, n + 1])
                                                   + np.sqrt(
                        np.log(sample_inner[:, :, i, n + 1] / sample_inner[:, :, i, n]) ** 2
                        - 2 * sigma ** 2 * h * np.log(inner_prob_knock_out[:, :, i, n]))) / 2)

    max_inner = np.max(sample_inner_max, axis=3)

    sample_outer_test = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                  drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                  step_size=h, path=True)
    sample_outer_test_tau = sample_outer_test[:, :, -1]

    n_step_outer = sample_outer_test.shape[2] - 1

    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_max = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_max[:, :, n] = np.exp((np.log(sample_outer_test[:, :, n] * sample_outer_test[:, :, n + 1])
                                            + np.sqrt(
                    np.log(sample_outer_test[:, :, n + 1] / sample_outer_test[:, :, n]) ** 2
                    - 2 * sigma ** 2 * h * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    max_outer_test = np.max(sample_outer_max, axis=2)

    max_inner = max_inner.reshape([d, n_front])

    Weight_D = methods.compute_Weight_D(S_0, mu, r, sigma, h, tau, sample_inner_tau)

    loss = np.zeros(n_front)
    n_partition = int(n_front // _n)
    for i in range(n_partition):

        LR_Loss_Matrix = np.zeros([n_front, _n])

        for j in range(d):

            payoff = np.zeros([n_front, _n])

            for k in range(len(K)):
                price = (max_outer_test[j, i * _n:(i + 1) * _n] <= U) \
                        * (np.maximum(sample_inner_T[j, :] - K[k], 0) * (max_inner[j, :] <= U)).reshape(-1, 1) \
                        * np.exp(-r * (T - tau))
                payoff = payoff + price

            diff = portfolio_value_0[0] - payoff

            Weight_U = methods.compute_Weight_U(sample_inner_tau[j, :], sample_outer_test_tau[j, i * _n:(i + 1) * _n],
                                        r, sigma, h)

            Weight = np.sqrt((tau + h) / h) * np.exp(Weight_D[j, :].reshape(-1, 1) - Weight_U)

            LR_Loss_Matrix = LR_Loss_Matrix + diff * Weight

        loss[i * _n:(i + 1) * _n] = np.mean(LR_Loss_Matrix, axis=0)

    return loss


def lr_BD(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, H, _n=1):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, H, 0, 0,
                                                                      option_type=np.array([["O", "D", "C"]]))

    sample_outer_train = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                   drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                   step_size=h, path=True)
    n_step_outer = sample_outer_train.shape[2] - 1

    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_min = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_min[:, :, n] = np.exp((np.log(sample_outer_train[:, :, n] * sample_outer_train[:, :, n + 1])
                                            - np.sqrt(
                    np.log(sample_outer_train[:, :, n + 1] / sample_outer_train[:, :, n]) ** 2
                    - 2 * sigma ** 2 * h * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    sample_inner = methods.GBM_back(n_front=n_front, n_back=1, d=d, S_tau=sample_outer_train,
                            drift=r, diffusion=cov_mat, T=T - tau,
                            step_size=h, path=True)

    n_step_inner = sample_inner.shape[3] - 1

    sample_inner_tau = sample_inner[:, :, :, 1].reshape([d, n_front])
    sample_inner_T = sample_inner[:, :, :, -1].reshape([d, n_front])

    inner_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_back, n_step_inner])
    sample_inner_min = np.zeros_like(inner_prob_knock_out)

    for i in range(n_back):
        for n in range(n_step_inner):

            sample_inner_min[:, :, i, n] = np.exp((np.log(sample_inner[:, :, i, n] * sample_inner[:, :, i, n + 1])
                                                   - np.sqrt(
                        np.log(sample_inner[:, :, i, n + 1] / sample_inner[:, :, i, n]) ** 2
                        - 2 * sigma ** 2 * h * np.log(inner_prob_knock_out[:, :, i, n]))) / 2)

    min_inner = np.min(sample_inner_min, axis=3)

    sample_outer_test = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                  drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                  step_size=h, path=True)
    sample_outer_test_tau = sample_outer_test[:, :, -1]

    n_step_outer = sample_outer_test.shape[2] - 1

    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_min = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_min[:, :, n] = np.exp((np.log(sample_outer_test[:, :, n] * sample_outer_test[:, :, n + 1])
                                            - np.sqrt(
                    np.log(sample_outer_test[:, :, n + 1] / sample_outer_test[:, :, n]) ** 2
                    - 2 * sigma ** 2 * h * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    min_outer_test = np.min(sample_outer_min, axis=2)
    min_inner = min_inner.reshape([d, n_front])

    Weight_D = methods.compute_Weight_D(S_0, mu, r, sigma, h, tau, sample_inner_tau)

    loss = np.zeros(n_front)
    n_partition = int(n_front // _n)
    for i in range(n_partition):

        LR_Loss_Matrix = np.zeros([n_front, _n])

        for j in range(d):

            payoff = np.zeros([n_front, _n])

            for k in range(len(K)):
                price = (min_outer_test[j, i * _n:(i + 1) * _n] >= H) \
                        * (np.maximum(sample_inner_T[j, :] - K[k], 0) * (min_inner[j, :] >= H)).reshape(-1, 1) \
                        * np.exp(-r * (T - tau))
                payoff = payoff + price

            diff = portfolio_value_0[0] - payoff

            Weight_U = methods.compute_Weight_U(sample_inner_tau[j, :], sample_outer_test_tau[j, i * _n:(i + 1) * _n],
                                        r, sigma, h)

            Weight = np.sqrt((tau + h) / h) * np.exp(Weight_D[j, :].reshape(-1, 1) - Weight_U)

            LR_Loss_Matrix = LR_Loss_Matrix + diff * Weight

        loss[i * _n:(i + 1) * _n] = np.mean(LR_Loss_Matrix, axis=0)

    return loss


def calRM(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h=3/50, H=120, L0=100, 
          optionName="European", level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):

    if optionName == "European":
        loss = lrEC(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h)
    elif optionName == "Asian":
        loss = lr_GA(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h)
    elif optionName == "BarrierUp":
        loss = lr_BU(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, H)
    elif optionName == "BarrierDown":
        loss = lr_BD(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, H)

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

    S_0 = 100
    K = [90, 100, 110]
    mu = 0.08
    r = 0.05
    tau = 3/50
    T = 1

    M_list = [int(1000 * (2**i)) for i in range(11)]
    N_list = [1] * len(M_list)

    n_rep = 1000

    if (d<1) or (d>100):
        print("Invalid Dimension!")
        exit()

    if optionName == "European":
        sigma = 0.1
        H = None
        h = 1/50
        
    elif optionName == "Asian":
        sigma = 0.3
        H = None
        h = T / 50
        tau = 3 * h

    elif optionName == "BarrierUp":
        sigma = 0.2
        H = 120
        h = T / 50
        tau = 3 * h

    elif optionName == "BarrierDown":
        sigma = 0.2
        H = 90
        h = T / 50
        tau = 3 * h

    else:
        print("Invalid Option Type!")
        exit()

    trueValues = pd.read_csv(f"{trueValueFolder}trueValue_{optionName}_{d}.csv")

    L0 = trueValues["VaR_0.9"].values[0]
    trueValues = np.array(trueValues).flatten()[1:]

    result_table = {}

    for M, N in zip(M_list, N_list):

        temp = Parallel(n_jobs=n_jobs, verbose=1)(delayed(calRM)(M, N, d, S_0, K, mu, sigma, r, tau, T, h, H, L0, optionName)
                                                for j in range(n_rep))
        temp = np.array(temp)

        Bias = np.mean(temp, axis=0) - trueValues
        Variance = np.var(temp, axis=0)

        MSE = np.array(Bias ** 2 + Variance).astype(float)

        RRMSE = np.sqrt(MSE) / trueValues

        result_table[M] = RRMSE

        print(f"LR Done for {optionName}, M={M}, N={N}, d={d}")

        result_df = pd.DataFrame(result_table)
        print(result_df)
        result_df.to_csv(f"{saveFolder}LR_{optionName}_{d}.csv")


