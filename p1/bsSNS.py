import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

import methods


def snsEC(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma**2

    sample_outer = methods.GBM_front(n_front = n_front, d = d, S_0 = S_0,
                                     drift_vec = mu, diffusion_mat = cov_mat, tau = tau)
    sample_inner = methods.GBM_back(n_front = n_front, n_back = n_back, d = d, S_tau = sample_outer,
                                    drift = r, diffusion = cov_mat, T = T - tau)

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(sample_inner - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    loss = d * portfolio_value_0 - np.sum(payoff, axis=0)

    return loss


def snsGA(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h):

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma ** 2

    sample_outer = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                     drift_vec=mu, diffusion_mat=cov_mat, tau=tau,
                                     step_size=h, path=True)
    n_step_outer = sample_outer.shape[2] - 1
    geometric_sum_outer = np.prod(sample_outer[:, :, 1:], axis=2)

    sample_inner = methods.GBM_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer,
                                    drift=r, diffusion=cov_mat, T=T - tau,
                                    step_size=h, path=True)
    n_step_inner = sample_inner.shape[3] - 1
    geometric_sum_inner = np.prod(sample_inner[:, :, :, 1:], axis=3)

    geometric_average = np.zeros([d, n_front, n_back])
    for i in range(n_back):
        geometric_average[:, :, i] = (geometric_sum_outer * geometric_sum_inner[:, :, i]) ** (
                    1 / (n_step_outer + n_step_inner))

    payoff = np.zeros([d, n_front])
    for j in range(len(K)):
        price = np.mean(np.maximum(geometric_average - K[j], 0), axis=2) * np.exp(-r * (T - tau))
        payoff = payoff + price

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_Asian_G(S=S_0, T=T, sigma=sigma, r=r, K=K[j],
                                                                      continuous=False,
                                                                      args=n_step_outer + n_step_inner)

    loss = d * portfolio_value_0 - np.sum(payoff, axis=0)

    return loss


def snsBU(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, H):
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
        sample_outer_max[:, :, n] = np.exp((np.log(sample_outer[:, :, n] * sample_outer[:, :, n + 1])
                                            + np.sqrt(np.log(sample_outer[:, :, n + 1] / sample_outer[:, :, n]) ** 2
                                                      - 2 * sigma ** 2 * h * np.log(
                    outer_prob_knock_out[:, :, n]))) / 2)

    max_outer = np.max(sample_outer_max, axis=2)

    sample_inner = methods.GBM_back(n_front=n_front, n_back=n_back, d=d, S_tau=sample_outer,
                                    drift=r, diffusion=cov_mat, T=T - tau,
                                    step_size=h, path=True)
    n_step_inner = sample_inner.shape[3] - 1

    inner_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_back, n_step_inner])
    sample_inner_max = np.zeros_like(inner_prob_knock_out)

    for i in range(n_back):
        for n in range(n_step_inner):
            sample_inner_max[:, :, i, n] = np.exp((np.log(sample_inner[:, :, i, n] * sample_inner[:, :, i, n + 1])
                                                   + np.sqrt(
                        np.log(sample_inner[:, :, i, n + 1] / sample_inner[:, :, i, n]) ** 2
                        - 2 * sigma ** 2 * h * np.log(inner_prob_knock_out[:, :, i, n]))) / 2)

    max_outer = np.repeat(max_outer[:, :, np.newaxis], n_back, axis=2)
    max_inner = np.max(sample_inner_max, axis=3)

    payoff = np.zeros([d, n_front])
    S_T = sample_inner[:, :, :, -1]

    for j in range(len(K)):
        payoff = payoff + np.mean((max_outer < H) * (max_inner < H)
                                  * np.maximum(S_T - K[j], 0), axis=2) * np.exp(-r * (T - tau))

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, H, 0, 0,
                                                                      option_type=np.array([["O", "U", "C"]]))

    loss = d * portfolio_value_0 - np.sum(payoff, axis=0)

    return loss


def snsBD(n_front, n_back, d, S_0, K, mu, sigma, r, tau, T, h, H):

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

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, H, 0, 0,
                                                                      option_type=np.array([["O", "D", "C"]]))

    loss = d * portfolio_value_0 - np.sum(payoff, axis=0)

    return loss


def bootstrapEC(Gamma, M_list, N_list, d, S_0, K, mu, sigma, r, tau, T, L0, I, 
                level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):

    n_estimate = len(level_list)*2 + 3

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma**2

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    M0 = M_list[-1]
    N0 = N_list[-1]

    sample_outer_0 = methods.GBM_front(M0, d, S_0, mu, cov_mat, tau)
    sample_inner_0 = methods.GBM_back(M0, N0, d, sample_outer_0, r, cov_mat, T-tau)

    M_shape = len(M_list)
    alpha_mat = np.zeros([M_shape, n_estimate])

    for m, N in zip(range(M_shape), N_list):

        res = np.zeros([I, n_estimate])

        for i in range(I):

            index_outer = np.random.choice(M0, size=M0, replace=True)
            index_inner = np.random.choice(N0, size=N, replace=True)
            temp = sample_inner_0[:, index_outer, :]
            sample_inner_bs = temp[:, :, index_inner]

            payoff = np.zeros([d, M0])
            for j in range(len(K)):
                price = np.mean(np.maximum(sample_inner_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))
                payoff = payoff + price

            loss = d * portfolio_value_0 - np.sum(payoff, axis=0)

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

            sample_inner_bs = sample_inner_0[:, np.random.choice(M0, size=M, replace=True), :]

            payoff = np.zeros([d, M])
            for j in range(len(K)):
                price = np.mean(np.maximum(sample_inner_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))
                payoff = payoff + price

            loss = d * portfolio_value_0 - np.sum(payoff, axis=0)

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


def bootstrapGA(Gamma, M_list, N_list, d, S_0, K, mu, sigma, r, tau, T, h,
                L0, I, 
                level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):
    
    n_estimate = len(level_list)*2 + 3

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma**2

    M0 = M_list[-1]
    N0 = N_list[-1]

    sample_outer_0 = methods.GBM_front(M0, d, S_0, mu, cov_mat, tau, h, path=True)
    sample_inner_0 = methods.GBM_back(M0, N0, d, sample_outer_0, r, cov_mat, T-tau, h, path=True)

    n_outer = sample_outer_0.shape[2] - 1
    n_inner = sample_inner_0.shape[3] - 1

    geometricSum_outer = np.prod(sample_outer_0[:, :, 1:], axis=2)
    geometricSum_inner = np.prod(sample_inner_0[:, :, :, 1:], axis=3)

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_Asian_G(S_0, T, sigma, r, K[j], 
                                                                        continuous=False,
                                                                        args=n_outer + n_inner)
    geometricSum = np.zeros([d, M0, N0])
    for i in range(N0):
        geometricSum[:, :, i] = (geometricSum_outer * geometricSum_inner[:, :, i]) ** (1 / (n_outer + n_inner))

    M_shape = len(M_list)
    alpha_mat = np.zeros([M_shape, n_estimate])

    for m, N in zip(range(M_shape), N_list):

        res = np.zeros([I, n_estimate])

        for i in range(I):

            index_outer = np.random.choice(M0, size=M0, replace=True)
            index_inner = np.random.choice(N0, size=N, replace=True)
            temp = geometricSum[:, index_outer, :]
            geometricSum_bs = temp[:, :, index_inner]

            payoff = np.zeros([d, M0])
            for j in range(len(K)):
                price = np.mean(np.maximum(geometricSum_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))
                payoff = payoff + price

            loss = d * portfolio_value_0 - np.sum(payoff, axis=0)

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

            geometricSum_bs = geometricSum[:, np.random.choice(M0, size=M, replace=True), :]

            payoff = np.zeros([d, M])
            for j in range(len(K)):
                price = np.mean(np.maximum(geometricSum_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))
                payoff = payoff + price

            loss = d * portfolio_value_0 - np.sum(payoff, axis=0)

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


def bootstrapBU(Gamma, M_list, N_list, d, S_0, K, mu, sigma, r, tau, T, h, H,
                L0, I, 
                level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):
    
    n_estimate = len(level_list)*2 + 3

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma**2

    M0 = M_list[-1]
    N0 = N_list[-1]

    sample_outer_0 = methods.GBM_front(M0, d, S_0, mu, cov_mat, tau, h, path=True)
    sample_inner_0 = methods.GBM_back(M0, N0, d, sample_outer_0, r, cov_mat, T-tau, h, path=True)

    n_outer = sample_outer_0.shape[2] - 1
    n_inner = sample_inner_0.shape[3] - 1

    outer_prob_knockOut = np.random.uniform(low=0.0, high=1.0,
                                            size=[d, M0, n_outer])
    sample_outer_max = np.zeros_like(outer_prob_knockOut)

    for n in range(n_outer):
        sample_outer_max[:, :, n] = np.exp((np.log(sample_outer_0[:, :, n] * sample_outer_0[:, :, n + 1])
                                            + np.sqrt(np.log(sample_outer_0[:, :, n + 1] / sample_outer_0[:, :, n]) ** 2
                                            - 2 * sigma ** 2 * h * np.log(outer_prob_knockOut[:, :, n]))) / 2)
    max_outer = np.max(sample_outer_max, axis=2)
    max_outer = np.repeat(max_outer[:, :, np.newaxis], N0, axis=2)

    inner_prob_knockOut = np.random.uniform(low=0.0, high=1.0,
                                            size=[d, M0, N0, n_inner])
    sample_inner_max = np.zeros_like(inner_prob_knockOut)

    for i in range(N0):
        for n in range(n_inner):
            sample_inner_max[:, :, i, n] = np.exp((np.log(sample_inner_0[:, :, i, n] * sample_inner_0[:, :, i, n + 1])
                                                   + np.sqrt(
                        np.log(sample_inner_0[:, :, i, n + 1] / sample_inner_0[:, :, i, n]) ** 2
                        - 2 * sigma ** 2 * h * np.log(inner_prob_knockOut[:, :, i, n]))) / 2)
    max_inner = np.max(sample_inner_max, axis=3)

    S_T = sample_inner_0[:, :, :, -1]

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, H, 0, 0,
                                                                      option_type=np.array([["O", "U", "C"]]))

    M_shape = len(M_list)
    alpha_mat = np.zeros([M_shape, n_estimate])

    for m, N in zip(range(M_shape), N_list):

        res = np.zeros([I, n_estimate])

        for i in range(I):

            index_outer = np.random.choice(M0, size=M0, replace=True)
            index_inner = np.random.choice(N0, size=N, replace=True)
            temp = max_outer[:, index_outer, :]
            max_outer_bs = temp[:, :, index_inner]
            temp = max_inner[:, index_outer, :]
            max_inner_bs = temp[:, :, index_inner]
            temp = S_T[:, index_outer, :]
            S_T_bs = temp[:, :, index_inner]

            payoff = np.zeros([d, M0])
            for j in range(len(K)):
                price = np.mean((max_outer_bs < H) * (max_inner_bs < H)
                                * np.maximum(S_T_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))
                payoff += price

            loss = d * portfolio_value_0 - np.sum(payoff, axis=0)

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

            index_outer = np.random.choice(M0, size=M, replace=True)
            max_outer_bs = max_outer[:, index_outer, :]
            max_inner_bs = max_inner[:, index_outer, :]
            S_T_bs = S_T[:, index_outer, :]

            payoff = np.zeros([d, M])
            for j in range(len(K)):
                price = np.mean((max_outer_bs < H) * (max_inner_bs < H)
                                * np.maximum(S_T_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))
                payoff += price

            loss = d * portfolio_value_0 - np.sum(payoff, axis=0)

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


def bootstrapBD(Gamma, M_list, N_list, d, S_0, K, mu, sigma, r, tau, T, h, H,
                L0, I, 
                level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):
    
    n_estimate = len(level_list)*2 + 3

    cor_mat = np.ones((d, d)) * 0.3 + np.identity(d) * 0.7
    cov_mat = cor_mat * sigma**2

    M0 = M_list[-1]
    N0 = N_list[-1]

    sample_outer_0 = methods.GBM_front(M0, d, S_0, mu, cov_mat, tau, h, path=True)
    sample_inner_0 = methods.GBM_back(M0, N0, d, sample_outer_0, r, cov_mat, T-tau, h, path=True)

    n_outer = sample_outer_0.shape[2] - 1
    n_inner = sample_inner_0.shape[3] - 1

    outer_prob_knockOut = np.random.uniform(low=0.0, high=1.0,
                                            size=[d, M0, n_outer])
    sample_outer_min = np.zeros_like(outer_prob_knockOut)

    for n in range(n_outer):
        sample_outer_min[:, :, n] = np.exp((np.log(sample_outer_0[:, :, n] * sample_outer_0[:, :, n + 1])
                                            - np.sqrt(np.log(sample_outer_0[:, :, n + 1] / sample_outer_0[:, :, n]) ** 2
                                            - 2 * sigma ** 2 * h * np.log(outer_prob_knockOut[:, :, n]))) / 2)
        
    min_outer = np.min(sample_outer_min, axis=2)
    min_outer = np.repeat(min_outer[:, :, np.newaxis], N0, axis=2)

    inner_prob_knockOut = np.random.uniform(low=0.0, high=1.0,
                                            size=[d, M0, N0, n_inner])
    sample_inner_min = np.zeros_like(inner_prob_knockOut)

    for i in range(N0):
        for n in range(n_inner):
            sample_inner_min[:, :, i, n] = np.exp((np.log(sample_inner_0[:, :, i, n] * sample_inner_0[:, :, i, n + 1])
                                                    - np.sqrt(np.log(sample_inner_0[:, :, i, n + 1] / sample_inner_0[:, :, i, n]) ** 2
                                                              - 2 * sigma ** 2 * h * np.log(inner_prob_knockOut[:, :, i, n]))) / 2)
    min_inner = np.min(sample_inner_min, axis=3)

    S_T = sample_inner_0[:, :, :, -1]

    portfolio_value_0 = 0
    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, H, 0, 0,
                                                                      option_type=np.array([["O", "D", "C"]]))

    M_shape = len(M_list)
    alpha_mat = np.zeros([M_shape, n_estimate])

    for m, N in zip(range(M_shape), N_list):

        res = np.zeros([I, n_estimate])

        for i in range(I):

            index_outer = np.random.choice(M0, size=M0, replace=True)
            index_inner = np.random.choice(N0, size=N, replace=True)
            temp = min_outer[:, index_outer, :]
            min_outer_bs = temp[:, :, index_inner]
            temp = min_inner[:, index_outer, :]
            min_inner_bs = temp[:, :, index_inner]
            temp = S_T[:, index_outer, :]
            S_T_bs = temp[:, :, index_inner]

            payoff = np.zeros([d, M0])
            for j in range(len(K)):
                price = np.mean((min_outer_bs > H) * (min_inner_bs > H)
                                * np.maximum(S_T_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))
                payoff += price

            loss = d * portfolio_value_0 - np.sum(payoff, axis=0)

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

            index_outer = np.random.choice(M0, size=M, replace=True)
            min_outer_bs = min_outer[:, index_outer, :]
            min_inner_bs = min_inner[:, index_outer, :]
            S_T_bs = S_T[:, index_outer, :]

            payoff = np.zeros([d, M])
            for j in range(len(K)):
                price = np.mean((min_outer_bs > H) * (min_inner_bs > H)
                                * np.maximum(S_T_bs - K[j], 0), axis=2) * np.exp(-r * (T - tau))
                payoff += price

            loss = d * portfolio_value_0 - np.sum(payoff, axis=0)

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


def calRM(Gamma, M_opt, N_opt, d, S_0, K, mu, sigma, r, tau, T, h=3/50, H=120, L0=100, 
          optionName="European", I=500, level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):

    VaR = {}
    CVaR = {}

    if optionName == "European":

        loss = snsEC(M_opt[0], N_opt[0], d, S_0, K, mu, sigma, r, tau, T)
        indicator = np.mean((loss > L0))

        loss = snsEC(M_opt[1], N_opt[1], d, S_0, K, mu, sigma, r, tau, T)
        hockey = np.mean(np.maximum(loss - L0, 0))

        loss = snsEC(M_opt[2], N_opt[2], d, S_0, K, mu, sigma, r, tau, T)
        quadratic = np.mean((loss - L0) ** 2)

        for k in range(len(level_list)):
            level = level_list[k]
            loss = snsEC(M_opt[3 + 2*k], N_opt[3 + 2*k], d, S_0, K, mu, sigma, r, tau, T)
            loss.sort()
            VaR[level] = loss[int(np.ceil(level * M_opt[3 + 2*k])) - 1] 
            loss = snsEC(M_opt[4 + 2*k], N_opt[4 + 2*k], d, S_0, K, mu, sigma, r, tau, T)
            loss.sort()
            Q = loss[int(np.ceil(level * M_opt[4 + 2*k])) - 1]
            CVaR[level] = np.mean(loss[loss >= Q])

    elif optionName == "Asian":

        loss = snsGA(M_opt[0], N_opt[0], d, S_0, K, mu, sigma, r, tau, T, h)
        indicator = np.mean((loss > L0))

        loss = snsGA(M_opt[1], N_opt[1], d, S_0, K, mu, sigma, r, tau, T, h)
        hockey = np.mean(np.maximum(loss - L0, 0))

        loss = snsGA(M_opt[2], N_opt[2], d, S_0, K, mu, sigma, r, tau, T, h)
        quadratic = np.mean((loss - L0) ** 2)

        for k in range(len(level_list)):
            level = level_list[k]
            loss = snsGA(M_opt[3 + 2*k], N_opt[3 + 2*k], d, S_0, K, mu, sigma, r, tau, T, h)
            loss.sort()
            VaR[level] = loss[int(np.ceil(level * M_opt[3 + 2*k])) - 1]
            
            loss = snsGA(M_opt[4 + 2*k], N_opt[4 + 2*k], d, S_0, K, mu, sigma, r, tau, T, h)
            loss.sort()
            Q = loss[int(np.ceil(level * M_opt[4 + 2*k])) - 1]
            CVaR[level] = np.mean(loss[loss >= Q])

    elif optionName == "BarrierUp":

        loss = snsBU(M_opt[0], N_opt[0], d, S_0, K, mu, sigma, r, tau, T, h, H)
        indicator = np.mean((loss > L0))

        loss = snsBU(M_opt[1], N_opt[1], d, S_0, K, mu, sigma, r, tau, T, h, H)
        hockey = np.mean(np.maximum(loss - L0, 0))

        loss = snsBU(M_opt[2], N_opt[2], d, S_0, K, mu, sigma, r, tau, T, h, H)
        quadratic = np.mean((loss - L0) ** 2)

        for k in range(len(level_list)):
            level = level_list[k]
            loss = snsBU(M_opt[3 + 2*k], N_opt[3 + 2*k], d, S_0, K, mu, sigma, r, tau, T, h, H)
            loss.sort()
            VaR[level] = loss[int(np.ceil(level * M_opt[3 + 2*k])) - 1]

            loss = snsBU(M_opt[4 + 2*k], N_opt[4 + 2*k], d, S_0, K, mu, sigma, r, tau, T, h, H)
            loss.sort()
            Q = loss[int(np.ceil(level * M_opt[4 + 2*k])) - 1]
            CVaR[level] = np.mean(loss[loss >= Q])
    
    elif optionName == "BarrierDown":

        loss = snsBD(M_opt[0], N_opt[0], d, S_0, K, mu, sigma, r, tau, T, h, H)
        indicator = np.mean((loss > L0))

        loss = snsBD(M_opt[1], N_opt[1], d, S_0, K, mu, sigma, r, tau, T, h, H)
        hockey = np.mean(np.maximum(loss - L0, 0))

        loss = snsBU(M_opt[2], N_opt[2], d, S_0, K, mu, sigma, r, tau, T, h, H)
        quadratic = np.mean((loss - L0) ** 2)

        for k in range(len(level_list)):
            level = level_list[k]
            loss = snsBD(M_opt[3 + 2*k], N_opt[3 + 2*k], d, S_0, K, mu, sigma, r, tau, T, h, H)
            loss.sort()
            VaR[level] = loss[int(np.ceil(level * M_opt[3 + 2*k])) - 1]

            loss = snsBD(M_opt[4 + 2*k], N_opt[4 + 2*k], d, S_0, K, mu, sigma, r, tau, T, h, H)
            loss.sort()
            Q = loss[int(np.ceil(level * M_opt[4 + 2*k])) - 1]
            CVaR[level] = np.mean(loss[loss >= Q])

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

    Gamma_list = [int(10000 * (2**i)) for i in range(11)]

    # Parameters for bootstrap
    M_array = np.arange(50, 101, 5)
    N_array = np.arange(50, 101, 5)
    I = 500

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

    for Gamma in Gamma_list:

        if optionName == "European":
            M_opt, N_opt = bootstrapEC(Gamma, M_array, N_array, d, S_0, K, mu, sigma, r, tau, T, L0, I)

        elif optionName == "Asian":
            M_opt, N_opt = bootstrapGA(Gamma, M_array, N_array, d, S_0, K, mu, sigma, r, tau, T, h, L0, I)
        
        elif optionName == "BarrierUp":
            M_opt, N_opt = bootstrapBU(Gamma, M_array, N_array, d, S_0, K, mu, sigma, r, tau, T, h, H, L0, I)
        
        elif optionName == "BarrierDown":
            M_opt, N_opt = bootstrapBD(Gamma, M_array, N_array, d, S_0, K, mu, sigma, r, tau, T, h, H, L0, I)

        temp = Parallel(n_jobs=n_jobs, verbose=1)(delayed(calRM)(Gamma, M_opt, N_opt, d, S_0, K, mu, sigma, r, tau, T, h, H, L0, optionName, I)
                                                for j in range(n_rep))
        temp = np.array(temp)

        Bias = np.mean(temp, axis=0) - trueValues
        Variance = np.var(temp, axis=0)

        MSE = np.array(Bias ** 2 + Variance).astype(float)

        RRMSE = np.sqrt(MSE) / trueValues

        result_table[Gamma] = RRMSE

        print(f"SNS Done for {optionName}, Gamma={Gamma}, d={d}")

        result_df = pd.DataFrame(result_table)
        print(result_df)
        result_df.to_csv(f"{saveFolder}SNS_{optionName}_{d}.csv")


