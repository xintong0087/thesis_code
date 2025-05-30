import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

import methods


def regressionEC(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma**2
    
    sample_outer_train = methods.GBM_front(n_front = n_front, d = d, S_0 = S_0,
                                           drift_vec = mu, diffusion_mat = cov_mat, tau = tau)
    sample_inner = methods.GBM_back(n_front = n_front, n_back = n_back, d = d, S_tau = sample_outer_train,
                                    drift = r, diffusion = cov_mat, T = T - tau)

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(sample_inner - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    X_train = methods.generate_basis(sample_outer_train)
    y_train = np.sum(payoff, axis=0)

    print(X_train.shape, y_train.shape)
    
    reg = LinearRegression().fit(X_train, y_train)

    sample_outer_test = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                          drift_vec=mu, diffusion_mat=cov_mat, tau=tau)

    X_test = methods.generate_basis(sample_outer_test)
    y_test = reg.predict(X_test)

    loss = d * portfolio_value_0 - y_test

    return loss


def regressionGA(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    sample_outer_train = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                           drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                           step_size=h, path=True)
    n_step_outer = sample_outer_train.shape[2] - 1
    geometric_sum_outer = np.prod(sample_outer_train[:, :, 1:], axis=2)

    sample_inner = methods.GBM_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer_train,
                                    drift=r, diffusion=cov_mat, T=T - tau,
                                    step_size=h, path=True)
    n_step_inner = sample_inner.shape[3] - 1
    geometric_sum_inner = np.prod(sample_inner[:, :, :, 1:], axis=3)

    geometric_average = np.zeros([d, n_front, n_back])
    for i in range(n_back):
        geometric_average[:, :, i] = (geometric_sum_outer * geometric_sum_inner[:, :, i]) \
                                     ** (1 / (n_step_outer + n_step_inner))

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(geometric_average - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    X_train = methods.generate_basis(sample_outer_train, option_type="Asian")
    y_train = np.sum(payoff, axis=0)

    reg = LinearRegression().fit(X_train, y_train)

    sample_outer_test = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                          drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                          step_size=h, path=True)

    X_test = methods.generate_basis(sample_outer_test, option_type="Asian")
    y_test = reg.predict(X_test)

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_Asian_G(S=S_0, T=T, sigma=sigma, r=r, K=K[j],
                                                              continuous=False, args=n_step_outer + n_step_inner)

    loss = d * portfolio_value_0 - y_test

    return loss


def regressionBU(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, H):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    sample_outer = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                     drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                     step_size=h, path=True)
    n_step_outer = sample_outer.shape[2] - 1

    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_max = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_max[:, :, n] = np.exp((np.log(sample_outer[:, :, n] * sample_outer[:, :, n+1])
                  + np.sqrt(np.log(sample_outer[:, :, n+1] / sample_outer[:, :, n])**2
                  - 2 * sigma**2 * h * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    max_outer = np.max(sample_outer_max, axis = 2)

    sample_inner = methods.GBM_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer,
                                    drift=r, diffusion=cov_mat, T=T - tau,
                                    step_size=h, path=True)
    n_step_inner = sample_inner.shape[3] - 1

    inner_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_back, n_step_inner])
    sample_inner_max = np.zeros_like(inner_prob_knock_out)

    for i in range(n_back):
        for n in range(n_step_inner):
            sample_inner_max[:, :, i, n] = np.exp((np.log(sample_inner[:, :, i, n] * sample_inner[:, :, i, n+1])
                  + np.sqrt(np.log(sample_inner[:, :, i, n+1] / sample_inner[:, :, i, n])**2
                  - 2 * sigma**2 * h * np.log(inner_prob_knock_out[:, :, i, n]))) / 2)

    max_outer = np.repeat(max_outer[:, :, np.newaxis], n_back, axis = 2)
    max_inner = np.max(sample_inner_max, axis = 3)

    payoff = np.zeros([d, n_front])
    S_T = sample_inner[:, :, :, -1]

    for j in range(len(K)):
        payoff = payoff + np.mean((max_outer < H) * (max_inner < H)
                                  * np.maximum(S_T - K[j], 0), axis = 2) * np.exp(-r * (T - tau))

    X_train = methods.generate_basis(sample_outer, option_type="Barrier_U", sample_max=max_outer[:, :, -1])

    y_train = np.sum(payoff, axis=0)
    reg = LinearRegression().fit(X_train, y_train)

    sample_outer_test = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                          drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                          step_size=h, path=True)
    n_step_outer = sample_outer_test.shape[2] - 1

    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_max = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_max[:, :, n] = np.exp((np.log(sample_outer_test[:, :, n] * sample_outer_test[:, :, n + 1])
                                            + np.sqrt(
                    np.log(sample_outer_test[:, :, n + 1] / sample_outer_test[:, :, n]) ** 2
                    - 2 * sigma ** 2 * h * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    max_outer = np.max(sample_outer_max, axis=2)

    X_test = methods.generate_basis(sample_outer_test, option_type="Barrier_U",
                                    sample_max=max_outer)
    y_test = reg.predict(X_test)

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, H, 0, 0,
                                                                      option_type=np.array([["O", "U", "C"]]))

    loss = d * portfolio_value_0 - y_test

    return loss


def regressionBD(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, H):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    sample_outer = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                     drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                     step_size=h, path=True)
    n_step_outer = sample_outer.shape[2] - 1

    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_min = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_min[:, :, n] = np.exp((np.log(sample_outer[:, :, n] * sample_outer[:, :, n + 1])
                                            - np.sqrt(np.log(sample_outer[:, :, n + 1] / sample_outer[:, :, n]) ** 2
                                            - 2 * sigma ** 2 * h * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    min_outer = np.min(sample_outer_min, axis=2)

    sample_inner = methods.GBM_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer,
                                    drift=r, diffusion=cov_mat, T=T - tau,
                                    step_size=h, path=True)
    n_step_inner = sample_inner.shape[3] - 1

    inner_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_back, n_step_inner])
    sample_inner_min = np.zeros_like(inner_prob_knock_out)

    for i in range(n_back):
        for n in range(n_step_inner):
            sample_inner_min[:, :, i, n] = np.exp((np.log(sample_inner[:, :, i, n] * sample_inner[:, :, i, n + 1])
                                                   - np.sqrt(
                        np.log(sample_inner[:, :, i, n + 1] / sample_inner[:, :, i, n]) ** 2
                        - 2 * sigma ** 2 * h * np.log(inner_prob_knock_out[:, :, i, n]))) / 2)

    min_outer = np.repeat(min_outer[:, :, np.newaxis], n_back, axis=2)
    min_inner = np.min(sample_inner_min, axis=3)

    payoff = np.zeros([d, n_front])
    S_T = sample_inner[:, :, :, -1]

    for j in range(len(K)):
        payoff = payoff + np.mean((min_outer > H) * (min_inner > H) * np.maximum(S_T - K[j], 0),
                                  axis=2) * np.exp(-r * (T - tau))

    X_train = methods.generate_basis(sample_outer, option_type="Barrier_D", sample_min=min_outer[:, :, -1])

    y_train = np.sum(payoff, axis=0)
    reg = LinearRegression().fit(X_train, y_train)

    sample_outer_test = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                          drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                          step_size=h, path=True)
    n_step_outer = sample_outer_test.shape[2] - 1

    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_min = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_min[:, :, n] = np.exp((np.log(sample_outer_test[:, :, n] * sample_outer_test[:, :, n + 1])
                                            - np.sqrt(
                    np.log(sample_outer_test[:, :, n + 1] / sample_outer_test[:, :, n]) ** 2
                    - 2 * sigma ** 2 * h * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    min_outer = np.min(sample_outer_min, axis=2)

    X_test = methods.generate_basis(sample_outer_test, option_type="Barrier_D",
                                    sample_min=min_outer)
    y_test = reg.predict(X_test)

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, H, 0, 0,
                                                                      option_type=np.array([["O", "D", "C"]]))

    loss = d * portfolio_value_0 - y_test

    return loss


def calRM(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h=3/50, H=120, L0=100, 
          optionName="European", level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):

    if optionName == "European":
        loss = regressionEC(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T)
    elif optionName == "Asian":
        loss = regressionGA(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h)
    elif optionName == "BarrierUp":
        loss = regressionBU(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, H)
    elif optionName == "BarrierDown":
        loss = regressionBD(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, H)

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
        h = None
        
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

        print(f"Regression Done for {optionName}, M={M}, N={N}, d={d}")

        result_df = pd.DataFrame(result_table)
        print(result_df)
        result_df.to_csv(f"{saveFolder}Regression_{optionName}_{d}.csv")


