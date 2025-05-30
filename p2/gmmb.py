import os
import time

import numpy as np
from joblib import Parallel, delayed
from numpy.random import default_rng

import simGMMB


def sim_inner_RS(n_cores=40):
    
    rng_0 = default_rng(seed_states_inner)
    rng_1 = default_rng(seed_diffusion_inner)

    res = Parallel(n_jobs=n_cores, verbose=1)(delayed(simGMMB.RS_inner)(N, FV_table[i, j], GV_table[i, j], states[i, j], survival_rates[i, j],
                                                             r, drift_RS_inner, vol_RS, n_steps - j, base_lapse[0, :],
                                                             lapse_flag, dlapse_flag, transition_prob,
                                                             rng_0.integers(low=0, high=2147483647),
                                                             rng_1.integers(low=0, high=2147483647))
                                   for i, j in np.ndindex(FV_table.shape))

    res = np.array(res).reshape([M, -1, 4])

    price = res[:, :, 0]
    delta = res[:, :, 1]
    price_se = res[:, :, 2]
    delta_se = res[:, :, 3]

    return price, delta, price_se, delta_se


def sim_inner_BS(n_cores=40):
    # res = np.zeros((dim_0, dim_1, 4))

    res = Parallel(n_jobs=n_cores, verbose=10)(delayed(simGMMB.BS_inner)(FV_table[i, j], GV_table[i, j], survival_rates[i, j],
                                                             r, vol_BS, n_steps - j, base_lapse[0, :], lapse_flag)
                                   for i, j in np.ndindex(FV_table.shape))

    res = np.array(res).reshape([M, -1, 2])

    price = res[:, :, 0]
    delta = res[:, :, 1]

    return price, delta


def cal_HE(price, delta, price_outer, n_steps, r, S_0):
    """
    :param price: mean hedging price vector
    :param delta: mean delta vector
    :param price_outer: matrix of asset value
    :param n_steps: total period of the contract
    :param r: risk-free rate
    :param S_0: initial asset value
    :return: HE: total hedging error =
                      PV of value of GMAB contract in absence of hedging
                    + loss between consequent hedging portfolios
                    + initial hedge cost
                    - value of the last hedging portfolio
                    + PV of payoff on renewal date
    """

    discount_period = np.arange(1, n_steps).reshape(1, -1)

    HE = np.exp(-r * n_steps) * price[:, n_steps] \
         + np.sum(np.exp(-r * discount_period) * price_outer[:, 1:-1] * (delta[:, 1:-1] - delta[:, :-2]), axis=1) \
         + delta[:, 0] * S_0 \
         - np.exp(-r * n_steps) * delta[:, -2] * price_outer[:, -1]

    return HE


lapse_flag = False
dlapse_flag = False
sensitivity_flag = False
RS_flag = False

n_cores = 20

save_path = "./result/"

print("Currently using", n_cores, "cores.")
if not os.path.exists(save_path):
    os.makedirs(save_path)

S_0 = 1000
fundValue_0 = S_0
guaranteeValue_0 = S_0
State_0 = 0
transition_prob = [0.04, 0.2]
r = 0.002
div = 0
d = 1

drift_BS = 0.00375
vol_BS = 0.0457627

if sensitivity_flag:
    # Sensitivity setting
    drift_RS_outer = np.array([0.0085, -0.05])
    vol_RS = np.array([0.035, 0.2])
else:
    # Regular setting
    drift_RS_outer = np.array([0.0085, -0.02])
    vol_RS = np.array([0.035, 0.08])

drift_RS_inner = (r - div) - vol_RS ** 2 / 2

M = 1 * 10 ** 5

n_renewal = 120
n_steps = 240

SC_period = 7 * 12
base_lapse = np.concatenate([np.full([M, 1], 0),
                             np.full([M, SC_period], 0.05 / 12),
                             np.full([M, n_steps - SC_period], 0.1 / 12)],
                            axis=1)
base_lapse = 0

seed_states_outer = 22
seed_diffusion_outer = 330

seed_states_inner = 133
seed_diffusion_inner = 181

# seed_states_inner = 222
# seed_diffusion_inner = 333

seed_states_outer = 11
seed_diffusion_outer = 2 * (seed_states_outer) + 1

print(f"Starting simulating GMMB contract losses with: M = {M}.")
print(f"Currently using {n_cores} cores.")
print(f"Random seeds for outer simulation is: {seed_states_outer} and {seed_diffusion_outer} Starting outer simulations.")


if RS_flag:
    states, S_table, FV_table, GV_table, survival_rates = simGMMB.RS_outer(M, S_0, drift_RS_outer, vol_RS,
                                                                        guaranteeValue_0, State_0, transition_prob,
                                                                        n_steps, base_lapse, lapse_flag, dlapse_flag,
                                                                        seed_states_outer, seed_diffusion_outer)
    price, delta, price_se, delta_se = sim_inner_RS(n_cores)
else:
    S_table, FV_table, GV_table, survival_rates = simGMMB.BS_outer(M, S_0, drift_BS, vol_BS, guaranteeValue_0,
                                                                n_steps, base_lapse, False, seed_diffusion_outer)
    price, delta = sim_inner_BS(n_cores)

a = "BS"
c = ""
if RS_flag:
    a = "RS"
if lapse_flag:
    c = "lapse"
if dlapse_flag:
    c = "dlapse"

start = time.time()
print(f"Saving Location: {save_path}hedgingLoss_GMMB_{a}_{str(start)}")


HE = cal_HE(price, delta, S_table, n_steps, r, S_0)

print(time.time() - start)



np.save(f"{save_path}outerScenarios_GMMB_{a}_{str(start)}", S_table)
np.save(f"{save_path}hedgingLoss_GMMB_{a}_{str(start)}", HE)
