import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os 

import methods


def ComputeTrueLossEC(n_front, d, S_0, K, mu, sigma, r, tau, T, level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):

    rho = 0.3
    cor_mat = methods.generate_cor_mat(d, rho)
    cov_mat_call = cor_mat * sigma ** 2

    print("European Call: Simulating Front Paths...")
    sample_outer_call = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                          drift_vec=mu, diffusion_mat=cov_mat_call, tau=tau)

    print("Calculating Value...")
    call_tau = np.zeros(n_front)
    for j in range(len(K)):
        call_tau_vec = Parallel(n_jobs=d, verbose=10)(
            delayed(methods.price_CP)(sample_outer_call[k, :], T - tau, sigma, r, K[j], 0, "C", "long")
            for k in range(d))
        call_tau = call_tau + np.sum(call_tau_vec, axis=0)
    print("End of European Call.")

    print(call_tau)
    portfolio_value_0 = 0

    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_CP(S_0, T, sigma, r, K[j], 0, "C", "long")

    print(d * portfolio_value_0)
    loss_true = d * portfolio_value_0 - call_tau

    loss_true.sort()

    L0 = loss_true[int(np.ceil(0.9 * n_front)) - 1]

    indicator_true = 0.1
    hockey_true = np.mean(np.maximum(loss_true - L0, 0))
    quadratic_true = np.mean((loss_true - L0) ** 2)

    VaR = {}
    CVaR = {}
    for level in level_list:
        VaR[level] = loss_true[int(np.ceil(level * n_front)) - 1] 
        CVaR[level] = np.mean(loss_true[loss_true >= VaR[level]])

    return indicator_true, hockey_true, quadratic_true, VaR, CVaR


def ComputeTrueLossGA(n_front, d, S_0, K, mu, sigma, r, tau, T, h_asian, level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):

    rho = 0.3
    cor_mat = methods.generate_cor_mat(d, rho)
    cov_mat_asian = cor_mat * sigma ** 2

    print("Geometric Asian: Simulating Front Paths...")
    sample_outer_asian = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                           drift_vec=mu, diffusion_mat=cov_mat_asian, tau=tau,
                                           step_size=h_asian, path=True)
    n_step_outer = sample_outer_asian.shape[2] - 1
    n_step_inner = int(T // h_asian) - n_step_outer
    S_tau = sample_outer_asian[:, :, 1:]

    print("Calculating Value...")
    asian_tau = np.zeros(n_front)
    for j in range(len(K)):
        asian_tau_vec = Parallel(n_jobs=d, verbose=10)(
            delayed(methods.price_Asian_G_tau)(S_tau[k, :, :], T - tau, sigma, r, K[j],
                                               False, args=(n_step_outer, n_step_inner))
            for k in range(d))
        asian_tau = asian_tau + np.sum(asian_tau_vec, axis=0)

    print("End of Geometric Asian.")

    portfolio_value_0 = 0

    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_Asian_G(S=S_0, T=T, sigma=sigma, r=r, K=K[j],
                                                                      continuous=False,
                                                                      args=n_step_outer + n_step_inner)

    loss_true = d * portfolio_value_0 - asian_tau

    loss_true.sort()

    L0 = loss_true[int(np.ceil(0.9 * n_front)) - 1]

    indicator_true = 0.1
    hockey_true = np.mean(np.maximum(loss_true - L0, 0))
    quadratic_true = np.mean((loss_true - L0) ** 2)

    VaR = {}
    CVaR = {}
    for level in level_list:
        VaR[level] = loss_true[int(np.ceil(level * n_front)) - 1] 
        CVaR[level] = np.mean(loss_true[loss_true >= VaR[level]])

    return indicator_true, hockey_true, quadratic_true, VaR, CVaR


def ComputeTrueLossBD(n_front, d, S_0, K, mu, sigma, r, tau, T, h_barrier, H, level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):

    rho = 0.3
    cor_mat = methods.generate_cor_mat(d, rho)
    cov_mat_barrier = cor_mat * sigma ** 2

    print("Barrier Options: Simulating Front Paths...")
    sample_outer_barrier = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                             drift_vec=mu, diffusion_mat=cov_mat_barrier, tau=tau,
                                             step_size=h_barrier, path=True)
    n_step_outer = sample_outer_barrier.shape[2] - 1

    print("Simulating Path Maximums and Minimums...")
    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_min = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_min[:, :, n] = np.exp((np.log(sample_outer_barrier[:, :, n] * sample_outer_barrier[:, :, n + 1])
                                            - np.sqrt(
                    np.log(sample_outer_barrier[:, :, n + 1] / sample_outer_barrier[:, :, n]) ** 2
                    - 2 * sigma ** 2 * h_barrier * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    min_outer = np.min(sample_outer_min, axis=2)

    print("Calculating Value...")
    barrier_tau = np.zeros(n_front)
    for j in range(len(K)):
        barrier_tau_vec = Parallel(n_jobs=d, verbose=10)(
            delayed(methods.price_barrier_down)(n_front, min_outer[k, :],
                                               sample_outer_barrier[k, :, -1], K[j],
                                               T - tau, sigma, r, H)
            for k in range(d))

        barrier_tau = barrier_tau + np.sum(barrier_tau_vec, axis=0)

    print("End of Barrier.")
    portfolio_value_0 = 0

    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, H, 0, 0,
                                                                      option_type=np.full([1, 3], ["O", "D", "C"]))

    loss_true = d * portfolio_value_0 - barrier_tau

    loss_true.sort()

    L0 = loss_true[int(np.ceil(0.9 * n_front)) - 1]

    indicator_true = 0.1
    hockey_true = np.mean(np.maximum(loss_true - L0, 0))
    quadratic_true = np.mean((loss_true - L0) ** 2)

    VaR = {}
    CVaR = {}
    for level in level_list:
        VaR[level] = loss_true[int(np.ceil(level * n_front)) - 1] 
        CVaR[level] = np.mean(loss_true[loss_true >= VaR[level]])

    return indicator_true, hockey_true, quadratic_true, VaR, CVaR


def ComputeTrueLossBU(n_front, d, S_0, K, mu, sigma, r, tau, T, h_barrier, H, level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):

    rho = 0.3
    cor_mat = methods.generate_cor_mat(d, rho)
    cov_mat_barrier = cor_mat * sigma ** 2

    print("Barrier Options: Simulating Front Paths...")
    sample_outer_barrier = methods.GBM_front(n_front=n_front, d=d, S_0=S_0,
                                             drift_vec=mu, diffusion_mat=cov_mat_barrier, tau=tau,
                                             step_size=h_barrier, path=True)
    n_step_outer = sample_outer_barrier.shape[2] - 1

    print("Simulating Path Maximums and Minimums...")
    outer_prob_knock_out = np.random.uniform(low=0.0, high=1.0,
                                             size=[d, n_front, n_step_outer])
    sample_outer_max = np.zeros_like(outer_prob_knock_out)

    for n in range(n_step_outer):
        sample_outer_max[:, :, n] = np.exp((np.log(sample_outer_barrier[:, :, n] * sample_outer_barrier[:, :, n + 1])
                                            + np.sqrt(
                    np.log(sample_outer_barrier[:, :, n + 1] / sample_outer_barrier[:, :, n]) ** 2
                    - 2 * sigma ** 2 * h_barrier * np.log(outer_prob_knock_out[:, :, n]))) / 2)

    max_outer = np.max(sample_outer_max, axis=2)

    print("Calculating Value...")
    barrier_tau = np.zeros(n_front)
    for j in range(len(K)):
        barrier_tau_vec = Parallel(n_jobs=d, verbose=10)(
            delayed(methods.price_barrier_up)(n_front, max_outer[k, :],
                                              sample_outer_barrier[k, :, -1], K[j],
                                              T - tau, sigma, r, H)
            for k in range(d))

        barrier_tau = barrier_tau + np.sum(barrier_tau_vec, axis=0)

    print("End of Barrier.")
    portfolio_value_0 = 0

    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methods.price_barrier([S_0], K[j], T, sigma, r, H, 0, 0,
                                                                      option_type=np.full([1, 3], ["O", "U", "C"]))

    loss_true = d * portfolio_value_0 - barrier_tau

    loss_true.sort()

    L0 = loss_true[int(np.ceil(0.9 * n_front)) - 1]

    indicator_true = 0.1
    hockey_true = np.mean(np.maximum(loss_true - L0, 0))
    quadratic_true = np.mean((loss_true - L0) ** 2)

    VaR = {}
    CVaR = {}
    for level in level_list:
        VaR[level] = loss_true[int(np.ceil(level * n_front)) - 1] 
        CVaR[level] = np.mean(loss_true[loss_true >= VaR[level]])

    return indicator_true, hockey_true, quadratic_true, VaR, CVaR


if __name__ == "__main__":
    optionName = input("Option Type? Please enter one of: European, Asian, BarrierUp, BarrierDown:")
    d = int(input("Please enter dimension:"))

    n = 10**7
    S_0 = 100
    K = [90, 100, 110]
    mu = 0.08
    r = 0.05
    tau = 3/50
    T = 1

    if (d<1) or (d>100):
        print("Invalid Dimension!")
        exit()

    if optionName == "European":
        sigma = 0.1

        indicator, hockeyStick, quadratic, VaR, CVaR = ComputeTrueLossEC(n, d, S_0, K, mu, sigma, r, tau, T)

    elif optionName == "Asian":
        sigma = 0.3
        h = T / 50
        tau = 3 * h

        indicator, hockeyStick, quadratic, VaR, CVaR = ComputeTrueLossGA(n, d, S_0, K, mu, sigma, r, tau, T, h)

    elif optionName == "BarrierUp":
        sigma = 0.2
        H = 120
        h = T / 50
        tau = 3 * h

        indicator, hockeyStick, quadratic, VaR, CVaR = ComputeTrueLossBU(n, d, S_0, K, mu, sigma, r, tau, T, h, H)

    elif optionName == "BarrierDown":
        sigma = 0.2
        H = 90
        h = T / 50
        tau = 3 * h

        indicator, hockeyStick, quadratic, VaR, CVaR = ComputeTrueLossBD(n, d, S_0, K, mu, sigma, r, tau, T, h, H)

    else:
        print("Invalid Option Type!")
        exit()


    result_true = [indicator, hockeyStick, quadratic]
    label_true = ["indicator", "hockeyStick", "quadratic"]
    for level in [0.8, 0.9, 0.95, 0.99, 0.996]:
        result_true.append(VaR[level])
        result_true.append(CVaR[level])
        label_true.append("VaR_" + str(level))
        label_true.append("CVaR_" + str(level))

    df = pd.DataFrame(result_true,
                    index=label_true,
                    columns=["True Value"]).T

    saveFolder = "./trueValues/"
    if os.path.exists(saveFolder) == False:
        os.makedirs(saveFolder)

    df.to_csv(f"{saveFolder}trueValue_{optionName}_{d}.csv")