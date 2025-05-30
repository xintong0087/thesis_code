import simGMWB

import numpy as np
from numpy.random import default_rng
from joblib import Parallel, delayed
import time
import os


def sim_inner_RS(n_cores=40):
    
    rng_0 = default_rng(seed_states_inner)
    rng_1 = default_rng(seed_diffusion_inner)
    rng_2 = default_rng(seed_dlapse_inner)

    res = Parallel(n_jobs=n_cores,
                   verbose=verbose_level)(delayed(simGMWB.RS_inner)(N, S_table[i, j], FV_table[i, j], GV_table[i, j], states[i, j], survival_rates[i, j],
                                                        r, drift_RS_inner, vol_RS, n_steps, n_steps - j,
                                                        withdrawal_rate, eta_g, eta_n,
                                                        base_lapse[0, :], lapse_flag, dlapse_flag,
                                                        transition_prob,
                                                        rng_0.integers(low=0, high=2147483647),
                                                        rng_1.integers(low=0, high=2147483647),
                                                        rng_2.integers(low=0, high=2147483647))
                              for i, j in np.ndindex(FV_table.shape))

    res = np.array(res).reshape([M, -1, 2])

    delta = res[:, :, 0]
    delta_se = res[:, :, 1]

    return delta, delta_se


def sim_inner_BS(n_cores=40):

    rng_1 = default_rng(seed_diffusion_inner)

    res = Parallel(n_jobs=n_cores,
                   verbose=10)(delayed(simGMWB.BS_inner)(N, S_table[i, j], FV_table[i, j], GV_table[i, j],
                                                         r, vol_BS, n_steps - j, withdrawal_rate, eta_g, eta_n,
                                                         base_lapse[0, :], lapse_flag, dlapse_flag,
                                                         rng_1.integers(low=0, high=2147483647))
                               for i, j in np.ndindex(FV_table.shape))

    res = np.array(res).reshape([M, -1, 2])

    delta = res[:, :, 0]
    delta_se = res[:, :, 1]

    return delta, delta_se


def cal_HE(delta, S, I, FV, eta_n, n_steps, r):

    discount_before = np.exp(-r * np.arange(1, n_steps + 1))
    discount_after = np.exp(-r * np.arange(n_steps))

    HE = np.sum(delta[:, :-1] * (discount_after * S[:, :-1] - discount_before * S[:, 1:]), axis=1)

    Payoff = np.sum(discount_before * (np.maximum(I[:, 1:] - FV[:, 1:], 0) - FV[:, 1:] * eta_n), axis=1)

    return HE + Payoff

# User-Defined Specifications
N = 10 ** 5

save_path = "./result_true/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

RS_flag = True
lapse_flag = False
dlapse_flag = False

assetModel = "BS"
if RS_flag:
    assetModel = "RS"

mortality = "nolapse"
if lapse_flag:
    mortality = "lapse"
    if dlapse_flag:
        mortality = "dlapse"

new_scenarios = True
if new_scenarios:
    if not os.path.exists("./result/"):
        os.makedirs("./result/")

seed_states_outer = 11
seed_diffusion_outer = 2 * (seed_states_outer) + 1
seed_dlapse_outer = 3 * (seed_states_outer) + 1

local = False

if local:
    n_cores = 8
    path_ref = f"./result/GMWB/{mortality}/outerScenarios_GMWB_{assetModel}_{mortality}.npy"
else:
    n_cores = 20
    path_ref = f"./result/outerScenarios_GMWB_{assetModel}_{mortality}_{str(seed_states_outer)}.npy"

verbose_level = 10

# Model Specifications
S_0 = 1000
fundValue_0 = S_0
guaranteeValue_0 = S_0
State_0 = 0
transition_prob = [0.04, 0.2]
r = 0.002
div = 0
d = 1
withdrawal_rate = 0.375 / 100
eta_g = 0.2 / 100
eta_n = 0.1 / 100

drift_BS = 0.00375
vol_BS = 0.0457627

drift_RS_outer = np.array([0.0085, -0.02])
vol_RS = np.array([0.035, 0.08])

drift_RS_inner = (r - div) - vol_RS ** 2 / 2

M = 10 ** 3

n_renewal = 120
n_steps = 240

SC_period = 7 * 12
lapse_rate = 0.1
base_lapse = np.concatenate([np.full([M, 1], 0),
                             np.full([M, SC_period], lapse_rate / 24),
                             np.full([M, n_steps - SC_period], lapse_rate / 12)],
                            axis=1)

start = time.time()

# Printing specifications
print(f"Starting simulating {mortality} GMWB contract losses on {assetModel} asset model with: M = {M}, N = {N}.")
print(f"Currently using {n_cores} cores.")
print(f"Random seeds for outer simulation is: {seed_states_outer}, {seed_diffusion_outer}, and {seed_dlapse_outer}. Starting outer simulations.")

# Simulation block
if RS_flag:
    states, S_table, FV_table, GV_table, I_table, survival_rates = simGMWB.RS_outer(M, S_0, drift_RS_outer, vol_RS,
                                                                                    guaranteeValue_0, State_0, transition_prob,
                                                                                    n_steps, withdrawal_rate, eta_g,
                                                                                    base_lapse, lapse_flag, dlapse_flag,
                                                                                    seed_states_outer, seed_diffusion_outer, seed_dlapse_outer)
    if new_scenarios:
        print(f"Simulated new outer scenarios with seeds: {seed_states_outer}, {seed_diffusion_outer}, and {seed_dlapse_outer}")
        np.save(path_ref, S_table)
    else:
        S_ref = np.load(path_ref)
        if np.sum(np.abs(S_ref - S_table)) > 0.01:
            print("Simulated outer scenarios differ from the reference file. Please check!")
        else:
            print("Simulated outer scenarios agree with the reference file. Starting inner simulation...")

    now = int(time.time())
    print(f"Saving Location: {save_path}hedgingLoss_GMWB_{assetModel}_{str(N).zfill(3)}_{mortality}_{str(now)}")
    np.random.seed(now)

    seeds_list = np.random.randint(0, 2147483647, size=3)

    seed_states_inner = seed_states_outer + seeds_list[0]
    seed_diffusion_inner = seed_diffusion_outer + seeds_list[1]
    seed_dlapse_inner = seed_dlapse_outer + seeds_list[2]

    delta, delta_se = sim_inner_RS(n_cores)

else:
    S_table, FV_table, GV_table, I_table, survival_rates = simGMWB.BS_outer(M, S_0, drift_BS, vol_BS, guaranteeValue_0,
                                                                            n_steps, withdrawal_rate, eta_g,
                                                                            base_lapse, lapse_flag, dlapse_flag,
                                                                            seed_diffusion_outer)
    delta, delta_se = sim_inner_BS(n_cores)

loss = cal_HE(delta, S_table, I_table, FV_table, eta_n, n_steps, r)

print(f"Inner simulation completed. Total time taken: {(time.time() - start)/3600} hours")

np.save(f"{save_path}hedgingLoss_GMWB_{assetModel}_{str(N).zfill(3)}_{mortality}_{str(now)}", loss)

print(f"Files saved at: {save_path}hedgingLoss_GMWB_{assetModel}_{str(N).zfill(3)}_{mortality}_{str(now)}")